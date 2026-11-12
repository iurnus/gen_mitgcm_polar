#!/usr/bin/env python3
"""
MITgcm open-boundary (pkg/obcs) files for the WRF-co-located polar-stereographic
grid, from the per-segment GLORYS snapshot files written by
download_glorys_obc_ps.py (or the synthetic ones from make_synthetic_glorys_obc.py).

Boundary positions (0-based indices; OB_Iwest=1, OB_Ieast=Nx, OB_Jsouth=1, OB_Jnorth=Ny):
  side  tracers/Eta        normal velocity            tangential velocity
  W     cells i=0          U on face i=1 (OB_Iw+1)    V on south faces of column i=0
  E     cells i=Nx-1       U on face i=Nx-1 (OB_Ie)   V on south faces of column Nx-1
  S     cells j=0          V on face j=1 (OB_Js+1)    U on west faces of row j=0
  N     cells j=Ny-1       V on face j=Ny-1 (OB_Jn)   U on west faces of row Ny-1
Positions and rotation angles come from make_mitgcm_grid.build(); GLORYS east/north
velocities are rotated to the model axes at each velocity point:
    u_model = CS*uE + SN*vN,     v_model = -SN*uE + CS*vN

Per segment file and snapshot: fill land NaNs from the nearest wet GLORYS point
(per level), bilinear interpolation to the boundary points of that segment, then
vertical interpolation to the model levels (delZ from `data`). Velocities are set to
0 where the face is dry (per your bathymetry); T/S keep filled values there.
Points in all-land segments (not downloaded) get T/S copied from the nearest
along-boundary point and zero velocity.

Output (big-endian, float32 unless --prec 64), one file per side and variable,
all snapshots in time order:
    OB{W,E,S,N}_{T,S,U,V}.bin   record shape (Nr, Ny) for W/E, (Nr, Nx) for S/N
    OB{W,E,S,N}_Eta.bin         record shape (Ny,) or (Nx,)
    obcs_snippet.txt            lines for data.obcs, data.exf, data.pkg
    obc_report.txt              ranges + net volume transport through each side
    obc_check.png               sections and transport time series

usage:
  make_mitgcm_obc_ps.py SEGMENT_DIR geo_em_d01.nc --data data --bathy bathy.bin [--outdir obc]
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mitgcm_grid import build                                    # noqa: E402
from make_mitgcm_ic_ps import open_netcdf, read_delz, vinterp, angles, _scalar  # noqa: E402

VARS = ("thetao", "so", "uo", "vo")


# ============================================================== segment reader
def decode_time(tvar):
    units = tvar.attr("units")
    units = units.decode() if isinstance(units, bytes) else str(units)
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", units)
    if not m:
        raise SystemExit(f"cannot parse time units '{units}'")
    step = {"days": 86400.0, "day": 86400.0, "hours": 3600.0, "hour": 3600.0,
            "minutes": 60.0, "seconds": 1.0, "second": 1.0}[m.group(1).lower()]
    ref = m.group(2).strip().replace("T", " ").replace("Z", "")
    ref = re.sub(r"\s*(UTC|utc|[+-]00:?00)$", "", ref)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            r = datetime.strptime(ref.split(".")[0], fmt)
            break
        except ValueError:
            continue
    else:
        raise SystemExit(f"cannot parse time reference '{ref}'")
    return [r + timedelta(seconds=float(v) * step) for v in np.asarray(tvar[:]).ravel()]


class Segment:
    def __init__(self, path):
        self.path = path
        self.f, self.getvar = open_netcdf(path)
        lon = np.asarray(self.getvar("longitude")[:], dtype="f8")
        lat = np.asarray(self.getvar("latitude")[:], dtype="f8")
        self.depth = np.asarray(self.getvar("depth")[:], dtype="f8")
        self.days = [t.date() for t in decode_time(self.getvar("time"))]
        # Dateline segments are joined from two downloads: make longitude continuous,
        # sort it, and drop duplicated columns (e.g. 180 and -180 both present).
        lon = lon[0] + np.mod(lon - lon[0] + 180.0, 360.0) - 180.0
        self.ilon, ndup_lon = self._sorted_unique(lon)
        self.ilat, ndup_lat = self._sorted_unique(lat)
        self.lon, self.lat = lon[self.ilon], lat[self.ilat]
        self.ndup = ndup_lon + ndup_lat
        self.dup_diff = None
        if ndup_lon:                       # duplicated columns should hold identical data
            dropped = np.setdiff1d(np.arange(lon.size), self.ilon)
            v = self.getvar("thetao")
            sel = tuple(0 if d in ("time", "depth") else slice(None) for d in v.dims)
            sst = np.asarray(v[sel], dtype="f8")              # (lat, lon), raw values
            diffs = []
            for c in dropped:
                twin = self.ilon[np.argmin(np.abs(lon[self.ilon] - lon[c]))]
                a, b = sst[:, c], sst[:, twin]
                ok = np.isfinite(a) & np.isfinite(b)
                diffs.append(np.abs(a[ok] - b[ok]).max() if ok.any() else 0.0)
            self.dup_diff = float(max(diffs))
        dl, dp = np.diff(self.lon), np.diff(self.lat)
        if dl.size == 0 or dp.size == 0 or np.ptp(dl) > 1e-4 or np.ptp(dp) > 1e-4:
            gaps = np.round(np.unique(np.round(dl, 4)), 4)
            raise SystemExit(f"{path}: longitude/latitude not regularly spaced after sorting "
                             f"(lon steps found: {gaps[:6]}); lon = {self.lon[:3]} ... {self.lon[-3:]}")
        self.dlon, self.dlat = dl.mean(), dp.mean()
        self._fill_cache = {}

    def read(self, name, t):
        """Field for time index t (int -> (depth, lat, lon) or (lat, lon) for zos) or for a
        block of records (slice -> leading time axis). float32; NaN = no data."""
        v = self.getvar(name)
        idx = tuple(t if d == "time" else slice(None) for d in v.dims)
        raw = v[idx]
        a = raw.astype("f4")
        bad = ~np.isfinite(a)
        for key in ("_FillValue", "missing_value"):
            fv = _scalar(v.attr(key))
            if fv is not None and np.isfinite(fv):
                bad |= raw == np.asarray(fv).astype(raw.dtype)
        sf, ao = _scalar(v.attr("scale_factor")), _scalar(v.attr("add_offset"))
        if sf is not None:
            a *= np.float32(sf)
        if ao is not None:
            a += np.float32(ao)
        a[bad] = np.nan
        del raw, bad
        # apply the sort / de-duplication of latitude and longitude
        dims = [d for d in v.dims if d != "time" or isinstance(t, slice)]
        ax_lat = dims.index("latitude") if "latitude" in dims else a.ndim - 2
        ax_lon = dims.index("longitude") if "longitude" in dims else a.ndim - 1
        a = np.take(np.take(a, self.ilat, axis=ax_lat), self.ilon, axis=ax_lon)
        if ax_lat > ax_lon:                     # make it (..., lat, lon)
            a = np.swapaxes(a, ax_lat, ax_lon)
        return a

    @staticmethod
    def _sorted_unique(x, tol=1e-4):
        """Indices that sort x and keep the first of any values closer than tol."""
        order = np.argsort(x, kind="stable")
        xs = x[order]
        keep = np.r_[True, np.diff(xs) > tol]
        return order[keep], int((~keep).sum())

    def fill(self, f2, key):
        """Nearest-wet fill of a 2-D slice, caching the index map per (key, mask)."""
        bad = ~np.isfinite(f2)
        if not bad.any():
            return f2
        if bad.all():
            return None
        c = self._fill_cache.get(key)
        if c is None or not np.array_equal(c[0], bad):
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            c = (bad, tuple(idx))
            self._fill_cache[key] = c
        return f2[c[1]]

    def coords(self, lat, lon):
        lon = self.lon[0] + np.mod(lon - self.lon[0], 360.0)       # into the file's lon frame
        fi = (lon - self.lon[0]) / self.dlon
        fj = (lat - self.lat[0]) / self.dlat
        outside = (fi < -0.5) | (fi > self.lon.size - 0.5) | (fj < -0.5) | (fj > self.lat.size - 0.5)
        if outside.any():
            raise SystemExit(f"{self.path}: {outside.sum()} boundary points fall outside the file's box")
        return np.vstack([np.clip(fj, 0, self.lat.size - 1), np.clip(fi, 0, self.lon.size - 1)])


def interp_levels(seg, f, name, crd, nz):
    """Fill + bilinear interpolate every GLORYS level of field f (z,y,x) to points crd.
    Returns (nz_glorys, npts)."""
    out = np.empty((nz, crd.shape[1]))
    for k in range(nz):
        g = seg.fill(f[k], (name, k))
        if g is None:
            out[k] = out[k - 1]
        else:
            out[k] = map_coordinates(g, crd, order=1, mode="nearest")
    return out


# ============================================================== boundary geometry
def boundary_points(P, g, nx, ny, dx, dy):
    """Lat/lon of tracer, U and V points on each open boundary, plus angles,
    model indices used for masks, and face lengths for transports."""
    x0, y0 = P.fwd(g["YC"][0, 0], g["XC"][0, 0])
    x0 = np.round(x0 / (dx / 2)) * (dx / 2); y0 = np.round(y0 / (dy / 2)) * (dy / 2)
    xc = x0 + dx * np.arange(nx); yc = y0 + dy * np.arange(ny)
    xg = xc - dx / 2;             yg = yc - dy / 2
    B = {}
    for side in "WESN":
        if side in "WE":
            ic = 0 if side == "W" else nx - 1                 # tracer column
            iu = 1 if side == "W" else nx - 1                 # normal-U face index
            pts = {"T": (np.full(ny, xc[ic]), yc), "U": (np.full(ny, xg[iu]), yc),
                   "V": (np.full(ny, xc[ic]), yg)}
            B[side] = dict(n=ny, ic=ic, iu=iu, pts=pts, normal="U",
                           facelen=g["DYG"][:ny, iu], inflow_sign=1 if side == "W" else -1)
        else:
            jc = 0 if side == "S" else ny - 1
            jv = 1 if side == "S" else ny - 1
            pts = {"T": (xc, np.full(nx, yc[jc])), "U": (xg, np.full(nx, yc[jc])),
                   "V": (xc, np.full(nx, yg[jv]))}
            B[side] = dict(n=nx, jc=jc, jv=jv, pts=pts, normal="V",
                           facelen=g["DXG"][jv, :nx], inflow_sign=1 if side == "S" else -1)
        for key, (x, y) in pts.items():
            lat, lon = P.inv(x, y)
            cs, sn = angles(P, lon)
            B[side][key] = dict(lat=lat, lon=lon, cs=cs, sn=sn)
    return B


def side_depths(side, b, H, nx, ny):
    """Water depth at tracer points, U faces and V faces of one boundary (min of neighbours)."""
    if side in "WE":
        ic, iu = b["ic"], b["iu"]
        Ht = H[:, ic]
        Hu = np.minimum(H[:, iu], H[:, iu - 1])                # face between iu-1 and iu
        Hv = Ht.copy(); Hv[1:] = np.minimum(Ht[1:], Ht[:-1])    # south face of each cell
    else:
        jc, jv = b["jc"], b["jv"]
        Ht = H[jc, :]
        Hv = np.minimum(H[jv, :], H[jv - 1, :])
        Hu = Ht.copy(); Hu[1:] = np.minimum(Ht[1:], Ht[:-1])
    return Ht, Hu, Hv


def nearest_fill_along(a, have):
    """a: (..., n) values valid where `have` (n,) is True; fill the rest from the nearest valid index."""
    if have.all():
        return a
    idx = np.arange(have.size)
    src = idx[have]
    near = src[np.abs(idx[:, None] - src[None, :]).argmin(axis=1)]
    return a[..., near]


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segdir", help="folder with obc_<side>_<seg>.nc and segments.csv")
    ap.add_argument("geo")
    ap.add_argument("--data", required=True)
    ap.add_argument("--bathy", required=True)
    ap.add_argument("--sides", nargs="+", default=list("WESN"), choices=list("WESN"))
    ap.add_argument("--prec", type=int, choices=[32, 64], default=32)
    ap.add_argument("--record-hour", type=int, default=12,
                    help="hour of day each record represents (12 = middle of the GLORYS daily mean)")
    ap.add_argument("--time-block", type=int, default=0,
                    help="records read per block (0 = all records of a segment at once; "
                         "use e.g. 50 if memory is tight)")
    ap.add_argument("--outdir", default="obc")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(a.geo)
    delR = read_delz(a.data); Nr = delR.size
    rF = np.concatenate([[0.0], np.cumsum(delR)]); RC = 0.5 * (rF[:-1] + rF[1:])
    H = -np.fromfile(a.bathy, ">f4").reshape(ny, nx).astype("f8")
    B = boundary_points(P, g, nx, ny, dx, dy)

    rows = list(csv.DictReader(open(os.path.join(a.segdir, "segments.csv"))))
    days = None
    out = {}
    for side in a.sides:
        b = B[side]; n = b["n"]
        segrows = [r for r in rows if r["side"] == side]
        have = np.zeros(n, bool)
        T = S = U = V = E = None
        for r in segrows:
            k0, k1 = int(r["cell_first"]), int(r["cell_last"]) + 1
            path = os.path.join(a.segdir, r["file"])
            if not os.path.exists(path):
                if r["status"].startswith("skip"):
                    continue
                raise SystemExit(f"missing segment file {path} (status {r['status']})")
            tseg = time.time()
            seg = Segment(path)
            if seg.ndup:
                msg = f"  note: {r['file']}: dropped {seg.ndup} duplicated lon/lat column(s) (dateline join)"
                if seg.dup_diff is not None:
                    msg += f"; max surface-T difference between the copies = {seg.dup_diff:.2e}"
                    if seg.dup_diff > 1e-3:
                        msg += "  <-- WARNING: copies differ, please send me this message"
                print(msg)
            if days is None:
                days = seg.days
                nt = len(days)
            elif seg.days != days:
                raise SystemExit(f"{path}: snapshot dates differ from the other segments")
            if T is None:
                T = np.zeros((nt, Nr, n)); S = np.zeros_like(T)
                U = np.zeros_like(T); V = np.zeros_like(T); E = np.zeros((nt, n))
            sl = slice(k0, k1)
            crd = {key: seg.coords(b[key]["lat"][sl], b[key]["lon"][sl]) for key in ("T", "U", "V")}
            nzg = seg.depth.size
            # Read each variable in large blocks of records (not record by record):
            # compressed NetCDF-4 chunks usually span many records, and reading them one
            # record at a time decompresses the same chunk over and over.
            npt = {key: crd[key].shape[1] for key in crd}
            gi = {(nm, key): np.empty((nt, nzg, npt[key]), "f4")
                  for nm, keys in (("thetao", "T"), ("so", "T"), ("uo", "UV"), ("vo", "UV"))
                  for key in keys}
            zi = np.empty((nt, npt["T"]))
            blk = nt if a.time_block <= 0 else a.time_block
            for nm, keys in (("thetao", "T"), ("so", "T"), ("uo", "UV"), ("vo", "UV"), ("zos", "T")):
                for t0 in range(0, nt, blk):
                    t1 = min(t0 + blk, nt)
                    F = seg.read(nm, slice(t0, t1))
                    for it in range(t1 - t0):
                        if nm == "zos":
                            z = seg.fill(F[it], ("zos",))
                            zi[t0 + it] = map_coordinates(z, crd["T"], order=1, mode="nearest")
                            continue
                        for key in keys:
                            gi[(nm, key)][t0 + it] = interp_levels(seg, F[it], nm, crd[key], nzg)
                    del F
            for t in range(nt):
                Tg, Sg = gi[("thetao", "T")][t], gi[("so", "T")][t]
                Ug = b["U"]["cs"][sl] * gi[("uo", "U")][t] + b["U"]["sn"][sl] * gi[("vo", "U")][t]
                Vg = -b["V"]["sn"][sl] * gi[("uo", "V")][t] + b["V"]["cs"][sl] * gi[("vo", "V")][t]
                for dst, src in ((T, Tg), (S, Sg), (U, Ug), (V, Vg)):
                    dst[t, :, sl] = vinterp(src[:, :, None].astype("f8"), seg.depth, RC)[:, :, 0]
                E[t, sl] = zi[t]
            del gi
            have[sl] = True
            print(f"  {side} segment {r['seg']:>2s} (cells {k0}-{k1 - 1}) done  "
                  f"[{time.time() - tseg:.1f} s]", flush=True)
        if T is None:
            raise SystemExit(f"no segment files found for side {side}")
        # all-land segments: copy T/S/Eta from nearest along-boundary point
        T = nearest_fill_along(T, have); S = nearest_fill_along(S, have)
        E = nearest_fill_along(E, have)
        # masks from the model bathymetry
        Ht, Hu, Hv = side_depths(side, b, H, nx, ny)
        wu = rF[:-1, None] < Hu[None]; wv = rF[:-1, None] < Hv[None]
        U = np.where(wu[None], U, 0.0); V = np.where(wv[None], V, 0.0)
        E = np.where(Ht[None] > 0, E, 0.0)
        out[side] = dict(T=T, S=S, U=U, V=V, Eta=E, Ht=Ht, Hu=Hu, Hv=Hv)

    # ---------------- checks + write
    dt = ">f4" if a.prec == 32 else ">f8"
    lines = [f"snapshots: {len(days)}  ({days[0]} .. {days[-1]})"]
    steps = sorted({(days[i + 1] - days[i]).days for i in range(len(days) - 1)})
    period_days = steps[0] if steps else 7
    if len(steps) > 1:
        raise SystemExit(f"snapshot spacing is not constant: {steps} days")
    lines.append(f"record period: {period_days} days")
    lines.append("side field    min      max     (wet points)")
    transport = {}
    for side, o in out.items():
        b = B[side]
        for name in ("T", "S", "U", "V", "Eta"):
            arr = o[name]
            if not np.all(np.isfinite(arr)):
                raise SystemExit(f"non-finite values in OB{side}_{name}")
            if name == "Eta":
                w = arr[:, o["Ht"] > 0]
            else:
                Href = {"T": o["Ht"], "S": o["Ht"], "U": o["Hu"], "V": o["Hv"]}[name]
                w = arr[:, rF[:-1, None] < Href[None]] if arr.ndim == 3 else arr
            lines.append(f"  {side}  {name:4s} {w.min():8.3f} {w.max():8.3f}")
            np.ascontiguousarray(arr).astype(dt).tofile(os.path.join(a.outdir, f"OB{side}_{name}.bin"))
        # net inflow (Sv) through the side, partial-cell thickness
        Hn = o["Hu"] if b["normal"] == "U" else o["Hv"]
        thick = np.clip(Hn[None, :] - rF[:-1, None], 0.0, delR[:, None])     # (Nr, n)
        vel = o[b["normal"]]
        transport[side] = b["inflow_sign"] * np.einsum("tkn,kn,n->t", vel, thick, b["facelen"]) / 1e6
    net = sum(transport.values())
    lines.append("net inflow (Sv), first record: " + "  ".join(
        f"{s} {transport[s][0]:+.1f}" for s in transport) + f"   total {net[0]:+.2f}")
    lines.append(f"total net inflow over all records: mean {net.mean():+.2f} Sv, "
                 f"range {net.min():+.2f} .. {net.max():+.2f} Sv")
    lines.append("(a non-zero total changes mean sea level; useOBCSbalance=.TRUE. removes it)")
    rep = "\n".join(lines)
    open(os.path.join(a.outdir, "obc_report.txt"), "w").write(rep + "\n")
    print(rep)

    write_snippet(os.path.join(a.outdir, "obcs_snippet.txt"), list(out), nx, ny, days[0],
                  a.record_hour, period_days, a.prec)
    if not a.no_plot:
        quicklook(os.path.join(a.outdir, "obc_check.png"), out, RC, days, transport)
    print(f"wrote {a.outdir}/  (see obcs_snippet.txt)")


def write_snippet(path, sides, nx, ny, d0, hour, period_days, prec):
    full = {"W": "west", "E": "east", "S": "south", "N": "north"}
    idx = {"W": f"OB_Iwest  = {ny}*1,", "E": f"OB_Ieast  = {ny}*{nx},",
           "S": f"OB_Jsouth = {nx}*1,", "N": f"OB_Jnorth = {nx}*{ny},"}
    t = [" # ---------------- data.obcs", " &OBCS_PARM01"]
    t += [" " + idx[s] for s in sides]
    t += [" useOBCSprescribe = .TRUE.,", " useOBCSbalance   = .TRUE.,"]
    for s in sides:
        t += [f" OB{s}tFile = 'OB{s}_T.bin',", f" OB{s}sFile = 'OB{s}_S.bin',",
              f" OB{s}uFile = 'OB{s}_U.bin',", f" OB{s}vFile = 'OB{s}_V.bin',",
              f"# OB{s}etaFile = 'OB{s}_Eta.bin',   (only with a non-linear free surface)"]
    t += [" &", "", " # ---------------- data.exf  (time axis of the boundary records)", " &EXF_NML_OBCS"]
    for s in sides:
        t += [f" obcs{s}startdate1 = {d0:%Y%m%d},", f" obcs{s}startdate2 = {hour:02d}0000,",
              f" obcs{s}period     = {period_days * 86400:.1f},"]
    t += [" &", "",
          " # ---------------- data.pkg:  useOBCS=.TRUE., useEXF=.TRUE., useCAL=.TRUE.",
          " # ---------------- data:      readBinaryPrec=" + str(prec),
          " # ---------------- OBCS_OPTIONS.h: #define ALLOW_OBCS_PRESCRIBE (+ ALLOW_OBCS_BALANCE)",
          " # U/V are grid-relative (model i/j directions), already rotated from east/north.",
          f" # First record = {d0} {hour:02d}:00; the model start must lie at or after it."]
    open(path, "w").write("\n".join(t) + "\n")


def quicklook(path, out, RC, days, transport):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sides = list(out)
    fig, ax = plt.subplots(3, len(sides), figsize=(4.2 * len(sides), 10), constrained_layout=True,
                           squeeze=False)
    kmax = int(np.searchsorted(RC, 2000))
    for c, s in enumerate(sides):
        o = out[s]
        n = o["T"].shape[2]
        Hn = o["Ht"]
        land = RC[:kmax, None] >= Hn[None]
        pc = ax[0, c].pcolormesh(np.arange(n), RC[:kmax],
                                 np.ma.masked_where(land, o["T"][0, :kmax]), cmap="RdYlBu_r")
        fig.colorbar(pc, ax=ax[0, c], label="°C"); ax[0, c].set_ylim(RC[kmax - 1], 0)
        ax[0, c].set_title(f"OB{s} T, {days[0]}")
        nm = "U" if s in "WE" else "V"
        lim = max(np.abs(o[nm][0, :kmax]).max(), 1e-3)
        pc = ax[1, c].pcolormesh(np.arange(n), RC[:kmax],
                                 np.ma.masked_where(land, o[nm][0, :kmax]), cmap="RdBu_r",
                                 vmin=-lim, vmax=lim)
        fig.colorbar(pc, ax=ax[1, c], label="m/s"); ax[1, c].set_ylim(RC[kmax - 1], 0)
        ax[1, c].set_title(f"OB{s} normal velocity {nm} (grid-relative)")
        ax[2, c].plot(days, transport[s], "o-")
        ax[2, c].axhline(0, color="k", lw=0.5)
        ax[2, c].set_title(f"net inflow through {s} (Sv)")
        ax[2, c].tick_params(axis="x", rotation=45)
        for r in (0, 1):
            ax[r, c].set_xlabel("index along boundary"); ax[r, c].set_ylabel("depth (m)")
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
