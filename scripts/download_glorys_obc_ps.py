#!/usr/bin/env python3
"""
Download GLORYS12 snapshots along the open boundaries of the MITgcm/WRF
polar-stereographic grid, in short segments, one NetCDF file per segment.

Why segments: each straight model boundary curves through ~90 deg of longitude
(41 S - 25 S), so one lat/lon box per side would download most of the band.
Short segments keep every box close to the boundary.

Snapshot dates: ONE continuous sequence  anchor + n*period  (default 2015-01-01,
every 7 days) regardless of calendar year, so leap years never matter and any
date window you download (e.g. one month to test, or the whole run) lands on the
same sequence. "Snapshot" = the GLORYS daily mean of that day.

Each segment covers the boundary cells and the first interior cells (all points
OBCS needs: tracer points, normal and tangential velocity points), plus --pad
degrees. Segments with no wet boundary cell (per --bathy) are skipped.
A segment that crosses the 180 meridian is downloaded in two parts and joined;
its longitude is stored continuously (e.g. 175..185) so it is monotonic.

Output (in --outdir):
    obc_<side>_<seg>.nc   thetao, so, uo, vo, zos (time, depth, latitude, longitude)
    segments.csv          index range / box / wet cells per segment
    segments_map.png      the boxes over the model boundary (made with --dry too)

Requirements:  pip install copernicusmarine     (brings xarray)
               copernicusmarine login           (once)

Examples:
  # see the plan (segments, dates, size), download nothing
  python download_glorys_obc_ps.py geo_em_d01.nc --bathy bathy.bin \
      --start 2016-01-01 --end 2016-01-31 --dry

  # January 2016 test
  python download_glorys_obc_ps.py geo_em_d01.nc --bathy bathy.bin \
      --start 2016-01-01 --end 2016-01-31 --outdir obc_2016_01

  # whole run, incl. one record before the start and one after the end
  python download_glorys_obc_ps.py geo_em_d01.nc --bathy bathy.bin \
      --start 2014-12-25 --end 2020-01-07 --outdir obc_2015_2019

  # summarise downloaded files
  python download_glorys_obc_ps.py geo_em_d01.nc --bathy bathy.bin --check --outdir obc_2016_01
"""
import argparse
import csv
import glob
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mitgcm_grid import build  # noqa: E402

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["thetao", "so", "uo", "vo", "zos"]
NZ_GLORYS = 50
RES = 1.0 / 12.0


# ============================================================== snapshot dates
def snapshot_dates(start, end, anchor, period):
    n0 = -((anchor - start).days // period)            # ceil((start-anchor)/period)
    out = []
    n = n0
    while True:
        d = anchor + timedelta(days=n * period)
        if d > end:
            break
        if d >= start:
            out.append((n, d))
        n += 1
    return out


# ============================================================== segments
def make_segments(geo, bathy_path, sides, seg_len, pad):
    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(geo)
    XG, YG = g["XG"], g["YG"]                          # corners, (ny+1, nx+1)
    wet = None
    if bathy_path:
        wet = np.fromfile(bathy_path, ">f4").reshape(ny, nx) < 0
    segs = []
    for side in sides:
        n_along = ny if side in ("W", "E") else nx
        for s, k0 in enumerate(range(0, n_along, seg_len)):
            k1 = min(k0 + seg_len, n_along)            # cells k0..k1-1 along the side
            # block of cells: 2 cells deep (boundary + first interior), 1 extra along-side
            a0, a1 = max(k0 - 1, 0), min(k1 + 1, n_along)
            if side == "W":
                jj, ii = slice(a0, a1 + 1), slice(0, 3)
                wcells = None if wet is None else wet[k0:k1, 0]
            elif side == "E":
                jj, ii = slice(a0, a1 + 1), slice(nx - 2, nx + 1)
                wcells = None if wet is None else wet[k0:k1, nx - 1]
            elif side == "S":
                jj, ii = slice(0, 3), slice(a0, a1 + 1)
                wcells = None if wet is None else wet[0, k0:k1]
            else:  # N
                jj, ii = slice(ny - 2, ny + 1), slice(a0, a1 + 1)
                wcells = None if wet is None else wet[ny - 1, k0:k1]
            lat = YG[jj, ii].ravel()
            lon = XG[jj, ii].ravel()
            # continuous longitude around the segment centre (handles the 180 meridian)
            ref = lon[lon.size // 2]
            lonc = ref + (lon - ref + 180.0) % 360.0 - 180.0
            box = [lonc.min() - pad, lonc.max() + pad,
                   max(lat.min() - pad, -80.0), lat.max() + pad]
            nwet = None if wcells is None else int(wcells.sum())
            segs.append(dict(side=side, seg=s, k0=k0, k1=k1, box=box, nwet=nwet,
                             file=f"obc_{side}_{s:02d}.nc"))
    return segs, (P, g, nx, ny)


def lon_parts(box):
    """Split [lon0, lon1] (continuous, may extend past +-180) into ranges inside [-180, 180].
    Returns list of (lon_min, lon_max, shift) where shift is added back after download."""
    lo, hi = box[0], box[1]
    if lo < -180.0:
        return [(lo + 360.0, 180.0, -360.0), (-180.0, hi, 0.0)]
    if hi > 180.0:
        return [(lo, 180.0, 0.0), (-180.0, hi - 360.0, 360.0)]
    return [(lo, hi, 0.0)]


def est_mb(box, ntime):
    nlon = (box[1] - box[0]) / RES + 1
    nlat = (box[3] - box[2]) / RES + 1
    return nlon * nlat * (4 * NZ_GLORYS + 1) * 4 * ntime / 1e6


# ============================================================== download
def pick_days(ds, dates):
    """Select the records whose calendar day matches `dates` (robust to 00:00 / 12:00 stamps)."""
    days = ds["time"].values.astype("datetime64[D]")
    idx = []
    for d in dates:
        hit = np.nonzero(days == np.datetime64(d.isoformat(), "D"))[0]
        if hit.size == 0:
            raise RuntimeError(f"date {d} not found in the GLORYS response")
        idx.append(int(hit[0]))
    return ds.isel(time=idx)


def fetch_open(seg, dates, a):
    import copernicusmarine
    import xarray as xr
    parts = []
    for lo, hi, shift in lon_parts(seg["box"]):
        ds = copernicusmarine.open_dataset(
            dataset_id=a.dataset, variables=VARIABLES,
            minimum_longitude=lo, maximum_longitude=hi,
            minimum_latitude=seg["box"][2], maximum_latitude=seg["box"][3],
            minimum_depth=0.0, maximum_depth=a.max_depth,
            start_datetime=f"{dates[0].isoformat()}T00:00:00",
            end_datetime=f"{dates[-1].isoformat()}T23:59:59",
        )
        ds = pick_days(ds, dates).load()
        if shift:
            ds = ds.assign_coords(longitude=ds["longitude"] + shift)
        parts.append(ds)
    return join_parts(parts)


def join_parts(parts):
    """Join the two halves of a dateline segment: sorted, continuous, no duplicated column."""
    import xarray as xr
    if len(parts) == 1:
        return parts[0]
    ds = xr.concat(parts, dim="longitude").sortby("longitude")
    lon = ds["longitude"].values
    keep = np.r_[True, np.diff(lon) > 1e-4]
    return ds.isel(longitude=np.nonzero(keep)[0])


def fetch_subset(seg, dates, a, tmpdir):
    """Fallback: one `copernicusmarine subset` call per date (slower, very robust)."""
    import copernicusmarine
    import xarray as xr
    per_date = []
    for d in dates:
        parts = []
        for p, (lo, hi, shift) in enumerate(lon_parts(seg["box"])):
            fn = f"{seg['side']}_{seg['seg']:02d}_{d:%Y%m%d}_p{p}.nc"
            path = os.path.join(tmpdir, fn)
            if not os.path.exists(path):
                copernicusmarine.subset(
                    dataset_id=a.dataset, variables=VARIABLES,
                    minimum_longitude=lo, maximum_longitude=hi,
                    minimum_latitude=seg["box"][2], maximum_latitude=seg["box"][3],
                    minimum_depth=0.0, maximum_depth=a.max_depth,
                    start_datetime=f"{d.isoformat()}T00:00:00",
                    end_datetime=f"{d.isoformat()}T23:59:59",
                    output_directory=tmpdir, output_filename=fn,
                )
            ds = xr.open_dataset(path).load()
            if shift:
                ds = ds.assign_coords(longitude=ds["longitude"] + shift)
            parts.append(ds)
        one = join_parts(parts)
        per_date.append(pick_days(one, [d]))
    return xr.concat(per_date, dim="time")


def write_segment(ds, seg, dates, path, a):
    ds.attrs.update(
        title="GLORYS12 open-boundary snapshots for MITgcm (polar stereographic grid)",
        source_dataset=a.dataset,
        boundary_side=seg["side"], segment=seg["seg"],
        along_side_cells=f"{seg['k0']}..{seg['k1'] - 1} (0-based)",
        snapshot_anchor=a.anchor, snapshot_period_days=a.period,
        snapshot_index_first=int(dates[0][0]), snapshot_index_last=int(dates[-1][0]),
        longitude_note="continuous; may extend beyond +-180 for segments crossing the dateline",
        created=datetime.now().isoformat(timespec="seconds"),
    )
    # one chunk per record, so reading any single snapshot is cheap
    enc = {}
    for v in VARIABLES:
        if v in ds:
            shape = list(ds[v].shape)
            shape[list(ds[v].dims).index("time")] = 1
            enc[v] = {"zlib": True, "complevel": 4, "chunksizes": tuple(shape)}
    tmp = path + ".part"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


# ============================================================== check / map
def check(outdir):
    import xarray as xr
    files = sorted(glob.glob(os.path.join(outdir, "obc_*.nc")))
    if not files:
        raise SystemExit(f"no obc_*.nc files in {outdir}")
    print(f"{'file':16s} {'nt':>3s} {'nz':>3s} {'nlat':>4s} {'nlon':>4s}  {'lon range':>17s}  "
          f"{'ocean%':>6s} {'T min':>6s} {'T max':>6s} {'|u|max':>6s}  first..last")
    tot = 0
    for f in files:
        with xr.open_dataset(f) as ds:
            T = ds["thetao"].isel(depth=0).values
            spd = np.hypot(ds["uo"].isel(depth=0).values, ds["vo"].isel(depth=0).values)
            t = ds["time"].values.astype("datetime64[D]")
            print(f"{os.path.basename(f):16s} {ds.sizes['time']:3d} {ds.sizes['depth']:3d} "
                  f"{ds.sizes['latitude']:4d} {ds.sizes['longitude']:4d}  "
                  f"{float(ds.longitude.min()):8.2f}..{float(ds.longitude.max()):7.2f}  "
                  f"{100 * np.isfinite(T[0]).mean():6.1f} {np.nanmin(T):6.2f} {np.nanmax(T):6.2f} "
                  f"{np.nanmax(spd):6.2f}  {t[0]}..{t[-1]}")
        tot += os.path.getsize(f)
    print(f"{len(files)} files, {tot / 1e9:.2f} GB")


def plot_map(segs, P, g, nx, ny, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = P.fwd(g["YG"], g["XG"])
    for arr in (x, y):
        pass
    ax.plot(np.r_[x[0, :], x[:, -1], x[-1, ::-1], x[::-1, 0]] / 1e3,
            np.r_[y[0, :], y[:, -1], y[-1, ::-1], y[::-1, 0]] / 1e3, "k-", lw=1.5,
            label="model boundary")
    colors = {"W": "tab:blue", "E": "tab:orange", "S": "tab:green", "N": "tab:red"}
    for s in segs:
        lo0, lo1, la0, la1 = s["box"]
        t = np.linspace(0, 1, 50)
        bl = np.r_[lo0 + (lo1 - lo0) * t, np.full(50, lo1), lo1 - (lo1 - lo0) * t, np.full(50, lo0)]
        bt = np.r_[np.full(50, la0), la0 + (la1 - la0) * t, np.full(50, la1), la1 - (la1 - la0) * t]
        bx, by = P.fwd(bt, bl)
        skipped = s["nwet"] == 0
        ax.plot(bx / 1e3, by / 1e3, color="0.7" if skipped else colors[s["side"]],
                lw=0.8, ls=":" if skipped else "-")
    ax.set_aspect("equal")
    ax.set_title("GLORYS download boxes (dotted grey = all-land, skipped)")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geo", help="WPS geo_em file (defines the grid)")
    ap.add_argument("--bathy", help="MITgcm bathymetry (>f4); all-land segments are skipped")
    ap.add_argument("--start", help="first date to download, YYYY-MM-DD")
    ap.add_argument("--end", help="last date to download, YYYY-MM-DD")
    ap.add_argument("--anchor", default="2015-01-01", help="a date in the snapshot sequence")
    ap.add_argument("--period", type=int, default=7, help="days between snapshots")
    ap.add_argument("--sides", nargs="+", default=["W", "E", "S", "N"], choices=["W", "E", "S", "N"])
    ap.add_argument("--seg-len", type=int, default=10, help="model cells per segment")
    ap.add_argument("--pad", type=float, default=0.25, help="degrees added around each box")
    ap.add_argument("--max-depth", type=float, default=6000.0)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--method", choices=["open", "subset"], default="open",
                    help="open: lazy open_dataset, only the snapshot days are transferred (default); "
                         "subset: one subset call per date (fallback)")
    ap.add_argument("--outdir", default="obc_raw")
    ap.add_argument("--dry", action="store_true", help="print plan + map, download nothing")
    ap.add_argument("--check", action="store_true", help="summarise files already in --outdir")
    a = ap.parse_args()

    if a.check:
        check(a.outdir)
        return
    if not (a.start and a.end):
        ap.error("--start and --end are required")
    os.makedirs(a.outdir, exist_ok=True)

    anchor = date.fromisoformat(a.anchor)
    dates = snapshot_dates(date.fromisoformat(a.start), date.fromisoformat(a.end), anchor, a.period)
    if not dates:
        raise SystemExit("no snapshot dates fall in that window")
    segs, (P, g, nx, ny) = make_segments(a.geo, a.bathy, a.sides, a.seg_len, a.pad)

    print(f"snapshots (every {a.period} d, anchored at {anchor}): {len(dates)}")
    print("   " + ", ".join(f"{d}" for _, d in dates[:8]) + (" ..." if len(dates) > 8 else ""))
    todo = [s for s in segs if s["nwet"] != 0]
    tot = 0.0
    with open(os.path.join(a.outdir, "segments.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["side", "seg", "cell_first", "cell_last", "lon_min", "lon_max",
                    "lat_min", "lat_max", "wet_cells", "file", "status"])
        for s in segs:
            mb = est_mb(s["box"], len(dates))
            status = "skip (land)" if s["nwet"] == 0 else "download"
            if status == "download":
                tot += mb
            w.writerow([s["side"], s["seg"], s["k0"], s["k1"] - 1,
                        *[f"{v:.3f}" for v in s["box"]], s["nwet"], s["file"], status])
    print(f"segments: {len(segs)} ({len(todo)} to download, {len(segs) - len(todo)} all-land skipped), "
          f"{a.seg_len} cells each")
    print(f"estimated size (uncompressed upper bound): {tot / 1e3:.2f} GB "
          f"(~{tot / len(dates) / 1e3:.2f} GB per snapshot)")
    plot_map(segs, P, g, nx, ny, os.path.join(a.outdir, "segments_map.png"))
    print(f"wrote {a.outdir}/segments.csv and segments_map.png")
    if a.dry:
        print("dry run: nothing downloaded")
        return

    tmpdir = os.path.join(a.outdir, "_tmp")
    if a.method == "subset":
        os.makedirs(tmpdir, exist_ok=True)
    only_dates = [d for _, d in dates]
    failed = []
    for i, s in enumerate(todo, 1):
        path = os.path.join(a.outdir, s["file"])
        if os.path.exists(path):
            print(f"[{i}/{len(todo)}] {s['file']} exists, skipping")
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                ds = fetch_open(s, only_dates, a) if a.method == "open" else \
                     fetch_subset(s, only_dates, a, tmpdir)
                write_segment(ds, s, dates, path, a)
                print(f"[{i}/{len(todo)}] {s['file']}  {dict(ds.sizes)}  {time.time() - t0:.0f} s")
                break
            except Exception as e:  # network hiccups etc.
                print(f"[{i}/{len(todo)}] {s['file']} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            failed.append(s["file"])
    if failed:
        print(f"\n{len(failed)} segment(s) failed; rerun the same command to retry: {failed}")
    else:
        print("\nall segments done.  check with:  --check --outdir " + a.outdir)


if __name__ == "__main__":
    main()
