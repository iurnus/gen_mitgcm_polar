#!/usr/bin/env python3
"""
Same check figure as plot_glorys_boundary.py, but from FULL-DOMAIN GLORYS files
(e.g. a Copernicus subset of all longitudes, 80S-23S) instead of the boundary segments.
Useful to verify the segment downloads and the OB*.bin files independently.

For each open boundary of the MITgcm polar-stereographic grid, GLORYS is sampled
densely along the boundary line (--step samples per model cell, default 12 = ~5 km),
on its own 50 levels, with velocities rotated to the model grid directions.

Figure (4 columns = W, E, S, N):
  row 1  GLORYS temperature along the boundary cells
  row 2  GLORYS normal velocity where OBCS applies it (grid-relative; + = +i or +j)
  row 3  transport into the domain through that side: inflow (red), outflow (blue),
         net (black); with --obc also the net transport of your OB*.bin files (dots)

Inputs: one or more NetCDF files (NetCDF-3 or -4), each with one or more time records;
records from all files are combined and sorted by date. Reads one level at a time, so a
full 1/12-deg file does not need to fit in memory.

usage:
  plot_glorys_boundary_full.py glorys_20160107.nc [more files ...] --geo geo_em_d01.nc \
      --data data --bathy bathy.bin [--obc obc] [--record 0] [-o glorys_full_boundary_check.png]

download example (one snapshot of the whole domain; all longitudes):
  copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
    --variable thetao --variable so --variable uo --variable vo --variable zos \
    --minimum-longitude -180 --maximum-longitude 180 \
    --minimum-latitude -80 --maximum-latitude -23 \
    --start-datetime 2016-01-07 --end-datetime 2016-01-07 -o glorys_20160107.nc
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mitgcm_grid import build                                   # noqa: E402
from make_mitgcm_ic_ps import Glorys, Interp, read_delz, angles      # noqa: E402
from make_mitgcm_obc_ps import decode_time                           # noqa: E402


def lines_for_side(side, x0, y0, dx, dy, nx, ny, step):
    """Dense sample positions (projection plane) along the tracer line and the normal-velocity
    line of one side, and the along-boundary coordinate in cells."""
    n = ny if side in "WE" else nx
    s = (np.arange(n * step) + 0.5) / step                  # fractional cell index along side
    if side in "WE":
        xt = x0 + dx * (0 if side == "W" else nx - 1)       # tracer column centre
        xu = x0 - dx / 2 + dx * (1 if side == "W" else nx - 1)   # normal-U face
        y = y0 - dy / 2 + s * dy
        return s, (np.full_like(s, xt), y), (np.full_like(s, xu), y)
    yt = y0 + dy * (0 if side == "S" else ny - 1)
    yv = y0 - dy / 2 + dy * (1 if side == "S" else ny - 1)
    x = x0 - dx / 2 + s * dx
    return s, (x, np.full_like(s, yt)), (x, np.full_like(s, yv))


def obc_transport(obcdir, side, g, H, delR, rF, nx, ny):
    nm = "U" if side in "WE" else "V"
    path = os.path.join(obcdir, f"OB{side}_{nm}.bin")
    if not os.path.exists(path):
        return None
    v = np.fromfile(path, ">f4").reshape(-1, delR.size, ny if side in "WE" else nx)
    if side in "WE":
        iu = 1 if side == "W" else nx - 1
        L = g["DYG"][:ny, iu]; Hf = np.minimum(H[:, iu], H[:, iu - 1])
    else:
        jv = 1 if side == "S" else ny - 1
        L = g["DXG"][jv, :nx]; Hf = np.minimum(H[jv, :], H[jv - 1, :])
    th = np.clip(Hf[None, :] - rF[:-1, None], 0.0, delR[:, None])
    sign = 1 if side in "WS" else -1
    return sign * np.einsum("tkn,kn,n->t", v, th, L) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="full-domain GLORYS NetCDF file(s)")
    ap.add_argument("--geo", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--bathy", required=True)
    ap.add_argument("--obc", help="folder with OB*.bin to overlay their net transports")
    ap.add_argument("--record", type=int, default=0, help="which (sorted) snapshot to show in the sections")
    ap.add_argument("--step", type=int, default=12, help="samples per model cell along the boundary")
    ap.add_argument("--zmax", type=float, default=1800.0)
    ap.add_argument("-o", "--output", default="glorys_full_boundary_check.png")
    a = ap.parse_args()

    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(a.geo)
    delR = read_delz(a.data); rF = np.r_[0, np.cumsum(delR)]
    H = -np.fromfile(a.bathy, ">f4").reshape(ny, nx).astype("f8")
    x0, y0 = P.fwd(g["YC"][0, 0], g["XC"][0, 0])
    x0 = np.round(x0 / (dx / 2)) * (dx / 2); y0 = np.round(y0 / (dy / 2)) * (dy / 2)

    # ---------------- sample positions for all four sides
    geom, ptsT, ptsN = {}, {}, {}
    for side in "WESN":
        s, (xt, yt), (xn, yn) = lines_for_side(side, x0, y0, dx, dy, nx, ny, a.step)
        latT, lonT = P.inv(xt, yt); latN, lonN = P.inv(xn, yn)
        cs, sn = angles(P, lonN)
        L = (g["DYG"][:ny, 1 if side == "W" else nx - 1] if side in "WE"
             else g["DXG"][1 if side == "S" else ny - 1, :nx])
        geom[side] = dict(s=s, cs=cs, sn=sn, ds=np.repeat(L, a.step) / a.step,
                          sign=1 if side in "WS" else -1)
        ptsT[side] = (latT, lonT); ptsN[side] = (latN, lonN)

    # ---------------- list all (file, time index, date) records
    recs = []
    for f in a.files:
        G = Glorys(f)
        days = [t.date() for t in decode_time(G.getvar("time"))]
        for i, d in enumerate(days):
            recs.append((d, f, i))
        G.f.close()
    recs.sort()
    days = [r[0] for r in recs]
    nt = len(recs)
    if not 0 <= a.record < nt:
        raise SystemExit(f"--record must be 0..{nt - 1}")
    print(f"{nt} snapshot(s): {days[0]} .. {days[-1]}")

    sec = {sd: {} for sd in "WESN"}
    tr = {sd: {k: np.zeros(nt) for k in ("in", "out")} for sd in "WESN"}
    I = None
    for t, (d, f, ti) in enumerate(recs):
        G = Glorys(f, time_index=ti)
        if I is None:
            if G.lat[0] > min(p[0].min() for p in ptsN.values()) or \
               G.lat[-1] < max(p[0].max() for p in ptsN.values()):
                raise SystemExit(f"{f}: latitude range {G.lat[0]:.1f}..{G.lat[-1]:.1f} does not cover "
                                 "the boundaries; download at least 45S..23S")
            if not G.periodic:
                print("WARNING: file is not global in longitude; boundary points outside it are clamped")
            depth = G.depth
            bnd = np.r_[0.0, 0.5 * (depth[1:] + depth[:-1]), depth[-1] + 0.5 * (depth[-1] - depth[-2])]
            dz = np.diff(bnd)
            nz = depth.size
            pts = {f"T{sd}": ptsT[sd] for sd in "WESN"}
            pts.update({f"N{sd}": ptsN[sd] for sd in "WESN"})
            I = Interp(G, pts)
            I_lon, I_lat = G.lon.copy(), G.lat.copy()
        elif G.lon.size != I_lon.size or G.lat.size != I_lat.size or \
                np.abs(G.lon - I_lon).max() > 1e-6 or np.abs(G.lat - I_lat).max() > 1e-6:
            raise SystemExit(f"{f}: different lon/lat grid from the first file")
        I.G = G
        show = t == a.record
        acc = {sd: dict(uE=np.zeros((nz, geom[sd]["s"].size)), vN=None, mu=None) for sd in "WESN"}
        for sd in "WESN":
            acc[sd]["vN"] = np.zeros_like(acc[sd]["uE"]); acc[sd]["mu"] = np.zeros_like(acc[sd]["uE"])
            if show:
                sec[sd]["T"] = np.full_like(acc[sd]["uE"], np.nan)
        for k in range(nz):
            if show:
                Tk = G.read("thetao", k); wT = np.isfinite(Tk)
                Tz, Tw = np.where(wT, Tk, 0.0), wT.astype("f4")
            uk = G.read("uo", k); vk = G.read("vo", k)
            wu = np.isfinite(uk)
            uz, vz, wf = np.where(wu, uk, 0.0), np.where(np.isfinite(vk), vk, 0.0), wu.astype("f4")
            for sd in "WESN":
                acc[sd]["uE"][k] = I(uz, f"N{sd}")
                acc[sd]["vN"][k] = I(vz, f"N{sd}")
                acc[sd]["mu"][k] = I(wf, f"N{sd}")
                if show:
                    v, m = I(Tz, f"T{sd}"), I(Tw, f"T{sd}")
                    sec[sd]["T"][k] = np.where(m > 0.5, v / np.maximum(m, 1e-6), np.nan)
        for sd in "WESN":
            gm, ac = geom[sd], acc[sd]
            if sd in "WE":
                vn = gm["cs"] * ac["uE"] + gm["sn"] * ac["vN"]
            else:
                vn = -gm["sn"] * ac["uE"] + gm["cs"] * ac["vN"]
            flux = gm["sign"] * vn * dz[:, None] * gm["ds"][None] / 1e6      # Sv per sample & level
            tr[sd]["in"][t] = flux[flux > 0].sum()
            tr[sd]["out"][t] = flux[flux < 0].sum()
            if show:
                sec[sd]["V"] = np.where(ac["mu"] > 0.5, vn / np.maximum(ac["mu"], 1e-6), np.nan)
        G.f.close()
        print(f"  {d} done", flush=True)

    # ---------------- report
    print(f"\n{'side':4s} {'inflow':>9s} {'outflow':>9s} {'net':>9s}   (mean over snapshots, Sv; + = into the domain)")
    net_tot = np.zeros(nt)
    for sd in "WESN":
        net = tr[sd]["in"] + tr[sd]["out"]; net_tot += net
        tr[sd]["net"] = net
        print(f"  {sd}  {tr[sd]['in'].mean():+9.2f} {tr[sd]['out'].mean():+9.2f} {net.mean():+9.2f}")
    print(f"  all {sum(tr[s]['in'].mean() for s in 'WESN'):+9.2f} "
          f"{sum(tr[s]['out'].mean() for s in 'WESN'):+9.2f} {net_tot.mean():+9.2f}"
          + (f"   (net range {net_tot.min():+.2f} .. {net_tot.max():+.2f})" if nt > 1 else ""))
    ob = {}
    if a.obc:
        for sd in "WESN":
            ob[sd] = obc_transport(a.obc, sd, g, H, delR, rF, nx, ny)
        print("OBC files net (same dates where available):")
        for sd in "WESN":
            if ob[sd] is not None:
                print(f"  {sd}: OBC records {ob[sd].size}, mean net {ob[sd].mean():+.2f} Sv")

    # ---------------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    kmax = int(np.searchsorted(depth, a.zmax)) + 1
    x_t = np.arange(nt)
    for c, sd in enumerate("WESN"):
        s = geom[sd]["s"]
        pc = ax[0, c].pcolormesh(s, depth[:kmax], sec[sd]["T"][:kmax], cmap="RdYlBu_r", shading="nearest")
        fig.colorbar(pc, ax=ax[0, c], label="°C")
        ax[0, c].set_title(f"GLORYS T along OB{sd}, {days[a.record]}")
        V = sec[sd]["V"][:kmax]
        lim = np.nanpercentile(np.abs(V), 99.5) if np.isfinite(V).any() else 1.0
        pc = ax[1, c].pcolormesh(s, depth[:kmax], V, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        fig.colorbar(pc, ax=ax[1, c], label="m/s")
        ax[1, c].set_title(f"GLORYS normal velocity, OB{sd} (grid-relative)")
        for r in (0, 1):
            ax[r, c].set_ylim(depth[kmax - 1], 0); ax[r, c].set_xlim(0, s[-1] + 0.5)
            ax[r, c].set_xlabel("index along boundary"); ax[r, c].set_ylabel("depth (m)")
        a3 = ax[2, c]
        xs = days if nt > 1 else x_t
        mk = "-" if nt > 1 else "o"
        a3.plot(xs, tr[sd]["in"], mk, color="tab:red", label="inflow")
        a3.plot(xs, tr[sd]["out"], mk, color="tab:blue", label="outflow")
        a3.plot(xs, tr[sd]["net"], mk, color="k", lw=1.5, label="net")
        if ob.get(sd) is not None and nt > 1 and ob[sd].size == nt:
            a3.plot(days, ob[sd], "o", ms=3, color="tab:green", label="OBC files net")
        a3.axhline(0, color="k", lw=0.5)
        a3.set_title(f"transport into domain through {sd} (Sv)")
        if nt > 1:
            a3.tick_params(axis="x", rotation=45)
        else:
            a3.set_xticks([0]); a3.set_xticklabels([str(days[0])])
        a3.legend(fontsize=8)
    fig.savefig(a.output, dpi=90)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
