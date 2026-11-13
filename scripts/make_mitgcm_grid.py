#!/usr/bin/env python3
"""
Build an MITgcm curvilinear grid exactly co-located with a WRF polar
stereographic (MAP_PROJ=2) domain read from a WPS geo_em file.

Co-location (both models are Arakawa C-grids):
    MITgcm XC,YC (tracer)       <-> WRF mass points  XLAT_M/XLONG_M
    MITgcm u-point (west face)  <-> WRF U points     XLAT_U/XLONG_U
    MITgcm v-point (south face) <-> WRF V points     XLAT_V/XLONG_V
    MITgcm XG,YG (SW corner)    <-> WRF corners      XLAT_C/XLONG_C

All positions are computed in double precision from the WRF projection
(sphere R = 6370 km, as WRF uses). Lengths and areas are exact integrals of
1/m and 1/m^2 over the projection plane (m = map factor), using Gauss-Legendre
quadrature, so they are consistent with WRF's conformal geometry.

Outputs (big-endian real*8, 18 records of (sNx+1)*(sNy+1)):
  * <name>.face001.bin  - whole domain as one facet (for use with pkg/exch2)
  * tiles/<name>.faceNNN.bin - one file per tile (for builds WITHOUT exch2,
    where MITgcm reads one file per tile, tile id = bi + (bj-1)*nSx*nPx)
Record order: XC YC DXF DYF RA XG YG DXV DYU RAZ DXC DYC RAW RAS DXG DYG
              AngleCS AngleSN

usage: make_mitgcm_grid.py geo_em.d01.nc OUTDIR [--name NAME] [--snx 30 --sny 30]
"""
import argparse, os
import numpy as np
from scipy.io import netcdf_file

R_EARTH = 6370000.0  # WRF sphere radius (m)


class WRFPolarStereo:
    def __init__(self, truelat1, stand_lon, dx, dy, nx, ny, hemi):
        self.h = hemi  # +1 north, -1 south
        self.lam0 = np.deg2rad(stand_lon)
        self.k = 1.0 + np.sin(np.deg2rad(abs(truelat1)))
        self.dx, self.dy, self.nx, self.ny = dx, dy, nx, ny

    def rho(self, phi):  # phi radians
        hp = self.h * phi
        return R_EARTH * self.k * np.cos(phi) / (1.0 + np.sin(hp))

    def fwd(self, lat, lon):
        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        r = self.rho(phi)
        x = r * np.sin(lam - self.lam0)
        y = -self.h * r * np.cos(lam - self.lam0)
        return x, y

    def inv(self, x, y):
        r = np.hypot(x, y)
        # r = R k cos(phi)/(1+sin(h phi))  =>  tan(pi/4 - h phi/2) = r/(R k)
        hphi = np.pi / 2 - 2.0 * np.arctan(r / (R_EARTH * self.k))
        lat = np.rad2deg(self.h * hphi)
        lam = self.lam0 + np.arctan2(x, -self.h * y)
        lon = (np.rad2deg(lam) + 180.0) % 360.0 - 180.0
        return lat, lon

    def mapfac_xy(self, x, y):
        lat, _ = self.inv(x, y)
        return self.k / (1.0 + np.sin(self.h * np.deg2rad(lat)))


GL_X, GL_W = np.polynomial.legendre.leggauss(8)


def seg_len(P, x0, y0, x1, y1):
    """True length of the straight projection-plane segment (x0,y0)->(x1,y1)."""
    L = np.hypot(x1 - x0, y1 - y0)
    s = 0.0
    for t, w in zip(GL_X, GL_W):
        f = 0.5 * (t + 1.0)
        s = s + 0.5 * w / P.mapfac_xy(x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return L * s


def rect_area(P, x0, x1, y0, y1):
    """True area of the projection-plane rectangle [x0,x1]x[y0,y1]."""
    A = (x1 - x0) * (y1 - y0)
    s = 0.0
    for tx, wx in zip(GL_X, GL_W):
        xx = x0 + 0.5 * (tx + 1.0) * (x1 - x0)
        for ty, wy in zip(GL_X, GL_W):
            yy = y0 + 0.5 * (ty + 1.0) * (y1 - y0)
            s = s + 0.25 * wx * wy / P.mapfac_xy(xx, yy) ** 2
    return A * s


def build(geo):
    f = netcdf_file(geo, "r", mmap=False)
    a = f._attributes
    assert int(a["MAP_PROJ"]) == 2, "not a polar stereographic WRF grid"
    nx = int(a["WEST-EAST_GRID_DIMENSION"]) - 1
    ny = int(a["SOUTH-NORTH_GRID_DIMENSION"]) - 1
    dx, dy = float(a["DX"]), float(a["DY"])
    hemi = -1 if float(a["TRUELAT1"]) < 0 else 1
    P = WRFPolarStereo(float(a["TRUELAT1"]), float(a["STAND_LON"]), dx, dy, nx, ny, hemi)
    wrf = {k: f.variables[k][0].astype(np.float64).copy() for k in
           ["XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
            "XLAT_C", "XLONG_C", "MAPFAC_M", "COSALPHA", "SINALPHA", "LANDMASK"]}
    f.close()

    # Locate the domain in projection space from WRF's own SW mass point and
    # verify the whole mass grid; corners are half a cell SW of mass points.
    x0, y0 = P.fwd(wrf["XLAT_M"][0, 0], wrf["XLONG_M"][0, 0])
    x0 = np.round(x0 / (dx / 2)) * (dx / 2)   # snap to half-cell (removes float32 noise)
    y0 = np.round(y0 / (dy / 2)) * (dy / 2)
    i = np.arange(0, nx + 2)                  # 0-based; index nx+1 = MITgcm's extra point
    j = np.arange(0, ny + 2)
    xc = x0 + i * dx; yc = y0 + j * dy        # cell centres (i=1..nx -> 0..nx-1)
    xg = xc - dx / 2; yg = yc - dy / 2        # SW corners / faces
    # values for index i-1 (needed for dxC, dxV, rAz, rAw ...)
    xcm = xc - dx; ycm = yc - dy

    X = lambda v: v[None, :]
    Y = lambda v: v[:, None]
    shp = (ny + 1, nx + 1)
    bc = lambda a: np.broadcast_to(a, (len(j), len(i)))[:ny + 1, :nx + 1].copy()

    g = {}
    g["YC"], g["XC"] = (bc(v) for v in P.inv(X(xc), Y(yc)))
    g["YG"], g["XG"] = (bc(v) for v in P.inv(X(xg), Y(yg)))
    g["DXF"] = bc(seg_len(P, X(xg), Y(yc), X(xg + dx), Y(yc)))
    g["DYF"] = bc(seg_len(P, X(xc), Y(yg), X(xc), Y(yg + dy)))
    g["RA"] = bc(rect_area(P, X(xg), X(xg + dx), Y(yg), Y(yg + dy)))
    g["DXV"] = bc(seg_len(P, X(xcm), Y(yg), X(xc), Y(yg)))
    g["DYU"] = bc(seg_len(P, X(xg), Y(ycm), X(xg), Y(yc)))
    g["RAZ"] = bc(rect_area(P, X(xcm), X(xc), Y(ycm), Y(yc)))
    g["DXC"] = bc(seg_len(P, X(xcm), Y(yc), X(xc), Y(yc)))
    g["DYC"] = bc(seg_len(P, X(xc), Y(ycm), X(xc), Y(yc)))
    g["RAW"] = bc(rect_area(P, X(xcm), X(xc), Y(yg), Y(yg + dy)))
    g["RAS"] = bc(rect_area(P, X(xg), X(xg + dx), Y(ycm), Y(yc)))
    g["DXG"] = bc(seg_len(P, X(xg), Y(yg), X(xg + dx), Y(yg)))
    g["DYG"] = bc(seg_len(P, X(xg), Y(yg), X(xg), Y(yg + dy)))
    # grid orientation: grid +x unit vector in (east, north) components
    lam = np.deg2rad(g["XC"]) - P.lam0
    if hemi < 0:
        g["AngleCS"], g["AngleSN"] = np.cos(lam), np.sin(lam)
    else:
        g["AngleCS"], g["AngleSN"] = np.cos(lam), -np.sin(lam)
    # pole (if it falls exactly on a corner): use WRF's longitude convention
    pole = np.abs(np.abs(g["YG"]) - 90.0) < 1e-9
    if pole.any():
        jj, ii = np.nonzero(pole)
        for a_, b_ in zip(jj, ii):
            if a_ <= ny and b_ <= nx:
                g["XG"][a_, b_] = wrf["XLONG_C"][a_, b_]

    # u / v point positions (for verification only)
    uv = {}
    uv["YU"], uv["XU"] = P.inv(X(xg[:nx + 1]), Y(yc[:ny]))
    uv["YV"], uv["XV"] = P.inv(X(xc[:nx]), Y(yg[:ny + 1]))
    mf = P.mapfac_xy(X(xc[:nx]), Y(yc[:ny]))
    return P, g, wrf, uv, mf, (nx, ny, dx, dy)


ORDER = ["XC", "YC", "DXF", "DYF", "RA", "XG", "YG", "DXV", "DYU", "RAZ",
         "DXC", "DYC", "RAW", "RAS", "DXG", "DYG", "AngleCS", "AngleSN"]


def write_face(path, g, i0, i1, j0, j1):
    """write sub-block [j0:j1+1, i0:i1+1] of every field (i fastest, >f8)."""
    with open(path, "wb") as fh:
        for k in ORDER:
            fh.write(np.ascontiguousarray(g[k][j0:j1 + 1, i0:i1 + 1]).astype(">f8").tobytes())


def dlon(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def verify(g, wrf, uv, mf, dims):
    nx, ny, dx, dy = dims
    lines = []
    def rep(name, v):
        lines.append(f"{name:42s} {v:.3e}")
    def latlon(n, lat, lon, wl, wn, polemask=None):
        dl = dlon(lon, wn)
        if polemask is not None:
            dl = np.where(polemask, 0.0, dl)
        rep(f"max |lat - WRF| {n} (deg)", np.max(np.abs(lat - wl)))
        rep(f"max |lon - WRF| {n} (deg, excl. pole)", np.max(dl))
    latlon("mass/XC,YC", g["YC"][:ny, :nx], g["XC"][:ny, :nx], wrf["XLAT_M"], wrf["XLONG_M"])
    latlon("U points", uv["YU"], uv["XU"], wrf["XLAT_U"], wrf["XLONG_U"],
           np.abs(uv["YU"]) > 89.999)
    latlon("V points", uv["YV"], uv["XV"], wrf["XLAT_V"], wrf["XLONG_V"],
           np.abs(uv["YV"]) > 89.999)
    latlon("corners/XG,YG", g["YG"], g["XG"], wrf["XLAT_C"], wrf["XLONG_C"],
           np.abs(g["YG"]) > 89.999)
    rep("max rel |mapfac - WRF MAPFAC_M|", np.max(np.abs(mf / wrf["MAPFAC_M"] - 1)))
    rep("max rel |DXF - DX/MAPFAC_M|", np.max(np.abs(g["DXF"][:ny, :nx] * wrf["MAPFAC_M"] / dx - 1)))
    rep("max rel |RA - DX*DY/MAPFAC_M^2|",
        np.max(np.abs(g["RA"][:ny, :nx] * wrf["MAPFAC_M"] ** 2 / (dx * dy) - 1)))
    rep("max |AngleCS - WRF COSALPHA|", np.max(np.abs(g["AngleCS"][:ny, :nx] - wrf["COSALPHA"])))
    rep("max |AngleSN - WRF SINALPHA|", np.max(np.abs(g["AngleSN"][:ny, :nx] - wrf["SINALPHA"])))
    rep("max |CS^2+SN^2-1|", np.max(np.abs(g["AngleCS"] ** 2 + g["AngleSN"] ** 2 - 1)))
    # area closure: sum of tracer-cell areas equals area of the whole square
    lines.append(f"{'DX range (km)':42s} {g['DXF'].min()/1e3:.3f} - {g['DXF'].max()/1e3:.3f}")
    lines.append(f"{'total domain area (1e6 km^2)':42s} {g['RA'][:ny,:nx].sum()/1e12:.4f}")
    return "\n".join(lines)


def plot_grid_check(path, P, g, wrf, dims, snx, sny, bathy=None, title="",
                    mask_bathy=None, mask_label="MITgcm"):
    """Four-panel check figure in the projection plane (km from the pole):
    1) bathymetry (or WRF LANDMASK if no bathymetry is given)
    2) lat/lon graticule and the tile layout (snx x sny)
    3) true grid spacing DXF (km)
    4) land/sea mask vs WRF LANDMASK (or the grid rotation angle if no bathymetry).
    bathy, mask_bathy: (ny, nx) arrays, negative = ocean. Panel 4 compares mask_bathy
    (default: bathy) with WRF; mask_label names it in the legend and title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    nx, ny, dx, dy = dims
    C_LAND, C_OCEAN, C_INK, C_GRID = "#d9d8d4", "#eef3f8", "#52514e", "#9b9a96"
    C_WET_ONLY, C_DRY_ONLY = "#eb6834", "#2a78d6"
    wrf_ocean = wrf["LANDMASK"] < 0.5
    if bathy is not None:
        H = -np.asarray(bathy, dtype=np.float64).reshape(ny, nx)
        ocean = H > 0
    else:
        H, ocean = None, wrf_ocean

    xg, yg = (v / 1e3 for v in P.fwd(g["YG"], g["XG"]))                 # corners (ny+1, nx+1)
    xc, yc = (v / 1e3 for v in P.fwd(g["YC"][:ny, :nx], g["XC"][:ny, :nx]))
    lat = g["YC"][:ny, :nx]
    hemi = "N" if P.h > 0 else "S"

    fig, ax = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)
    for a_ in ax.ravel():
        a_.set_aspect("equal")
        a_.set_xlabel("x (km)", color=C_INK); a_.set_ylabel("y (km)", color=C_INK)
        a_.tick_params(colors=C_INK, labelsize=8)
        for s in a_.spines.values():
            s.set_color(C_GRID)

    def coast(a_):
        a_.contour(xc, yc, ocean.astype(float), levels=[0.5], colors=C_INK, linewidths=0.6)

    # 1) bathymetry (or WRF land/sea mask)
    a1 = ax[0, 0]
    a1.pcolormesh(xg, yg, np.where(ocean, np.nan, 1.0), cmap=ListedColormap([C_LAND]), shading="flat")
    if H is not None:
        blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.18, 1.0, 256)))
        pc = a1.pcolormesh(xg, yg, np.ma.masked_where(~ocean, H), cmap=blues, vmin=0,
                           vmax=np.ceil(H.max() / 1000) * 1000, shading="flat")
        fig.colorbar(pc, ax=a1, shrink=0.8, label="depth (m)")
        a1.set_title("Bathymetry", fontsize=10)
    else:
        a1.pcolormesh(xg, yg, np.where(ocean, 1.0, np.nan), cmap=ListedColormap([C_OCEAN]), shading="flat")
        a1.set_title("WRF LANDMASK (give --bathy to show bathymetry)", fontsize=10)
    coast(a1)

    # 2) graticule + tiles
    a2 = ax[0, 1]
    a2.pcolormesh(xg, yg, np.where(ocean, 0.0, 1.0), cmap=ListedColormap([C_OCEAN, C_LAND]), shading="flat")
    coast(a2)
    lat_lv = np.arange(-80, 90, 10.0)
    cs = a2.contour(xc, yc, lat, levels=lat_lv, colors=C_GRID, linewidths=0.6)
    a2.clabel(cs, fmt=lambda v: f"{abs(v):g}°{'N' if v > 0 else 'S'}", fontsize=7)
    rmax = np.hypot(xg, yg).max()
    rl = 0.88 * min(np.abs(xg).max(), np.abs(yg).max())
    for L in np.arange(-180, 180, 30.0):               # meridians as rays from the pole
        px, py = P.fwd(np.array([89.999 * P.h]), np.array([L]))
        qx, qy = P.fwd(np.array([0.0]), np.array([L]))
        ux, uy = (qx - px)[0], (qy - py)[0]
        n = np.hypot(ux, uy)
        r = np.array([0.0, rmax])
        a2.plot(r * ux / n, r * uy / n, color=C_GRID, lw=0.6)
        lab = f"{abs(L):g}°" + ("" if abs(L) in (0, 180) else ("E" if L > 0 else "W"))
        a2.text(rl * ux / n, rl * uy / n, lab, fontsize=7, color=C_INK, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    a2.set_xlim(xg.min(), xg.max()); a2.set_ylim(yg.min(), yg.max())
    npx, npy = nx // snx, ny // sny
    for i in range(0, nx + 1, snx):
        a2.plot(xg[:, i], yg[:, i], color=C_DRY_ONLY, lw=0.9, ls="--")
    for j in range(0, ny + 1, sny):
        a2.plot(xg[j, :], yg[j, :], color=C_DRY_ONLY, lw=0.9, ls="--")
    for bj in range(npy):
        for bi in range(npx):
            jc, ic = bj * sny + sny // 2, bi * snx + snx // 2
            a2.text(xc[jc, ic], yc[jc, ic], f"{bi + 1 + bj * npx}", fontsize=7, color=C_DRY_ONLY,
                    ha="center", va="center")
    a2.set_title(f"Graticule and tiles: {npx} x {npy} tiles of {snx} x {sny}", fontsize=10)

    # 3) true grid spacing
    a3 = ax[1, 0]
    dxf = g["DXF"][:ny, :nx] / 1e3
    pc = a3.pcolormesh(xg, yg, dxf, cmap="Oranges", shading="flat")
    c3 = a3.contour(xc, yc, dxf, levels=np.arange(np.ceil(dxf.min() / 2.5) * 2.5, dxf.max(), 2.5),
                    colors=C_INK, linewidths=0.5)
    a3.clabel(c3, fmt="%g", fontsize=7)
    coast(a3)
    fig.colorbar(pc, ax=a3, shrink=0.8, label="DXF (km)")
    tl = abs(np.rad2deg(np.arcsin(P.k - 1)))
    a3.set_title(f"True grid spacing: {dxf.min():.1f} - {dxf.max():.1f} km "
                 f"(nominal {dx / 1e3:g} km at {tl:.0f}°{hemi})", fontsize=10)

    # 4) mask vs WRF (or rotation angle)
    a4 = ax[1, 1]
    if H is not None:
        mocean = ocean if mask_bathy is None else \
            (-np.asarray(mask_bathy, dtype=np.float64).reshape(ny, nx) > 0)
        cat = np.where(mocean & wrf_ocean, 0, np.where(~mocean & ~wrf_ocean, 1,
                       np.where(mocean & ~wrf_ocean, 2, 3)))
        a4.pcolormesh(xg, yg, cat, cmap=ListedColormap([C_OCEAN, C_LAND, C_WET_ONLY, C_DRY_ONLY]),
                      norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), shading="flat")
        n2, n3 = int((cat == 2).sum()), int((cat == 3).sum())
        a4.legend(handles=[Patch(fc=C_OCEAN, ec=C_GRID, label="ocean in both"),
                           Patch(fc=C_LAND, ec=C_GRID, label="land in both"),
                           Patch(fc=C_WET_ONLY, label=f"{mask_label} ocean, WRF land ({n2})"),
                           Patch(fc=C_DRY_ONLY, label=f"{mask_label} land, WRF ocean ({n3})")],
                  loc="lower left", fontsize=8, framealpha=0.9)
        a4.set_title(f"{mask_label} mask vs WRF LANDMASK: {100 * (cat < 2).mean():.2f}% agree",
                     fontsize=10)
    else:
        ang = np.rad2deg(np.arctan2(g["AngleSN"][:ny, :nx], g["AngleCS"][:ny, :nx]))
        pc = a4.pcolormesh(xg, yg, ang, cmap="twilight", vmin=-180, vmax=180, shading="flat")
        fig.colorbar(pc, ax=a4, shrink=0.8, label="angle of grid x-axis from east (deg)")
        a4.set_title("Grid rotation angle (atan2(AngleSN, AngleCS))", fontsize=10)
    coast(a4)

    fig.suptitle(title or f"MITgcm grid: {nx} x {ny} cells, polar stereographic "
                 f"({'north' if P.h > 0 else 'south'} pole)", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geo"); ap.add_argument("outdir")
    ap.add_argument("--name", default="grid_wrf_ps")
    ap.add_argument("--snx", type=int, default=30); ap.add_argument("--sny", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "tiles"), exist_ok=True)
    P, g, wrf, uv, mf, dims = build(a.geo)
    nx, ny, dx, dy = dims
    assert nx % a.snx == 0 and ny % a.sny == 0, "tile size must divide domain"

    write_face(os.path.join(a.outdir, f"{a.name}.face001.bin"), g, 0, nx, 0, ny)
    npx, npy = nx // a.snx, ny // a.sny
    for bj in range(npy):
        for bi in range(npx):
            tid = bi + 1 + bj * npx
            write_face(os.path.join(a.outdir, "tiles", f"{a.name}.face{tid:03d}.bin"), g,
                       bi * a.snx, (bi + 1) * a.snx, bj * a.sny, (bj + 1) * a.sny)
    # handy extras: centre-only fields (nx,ny) for forcing/regridding, >f4
    for k in ["XC", "YC", "AngleCS", "AngleSN"]:
        g[k][:ny, :nx].astype(">f4").tofile(os.path.join(a.outdir, f"{k}.bin"))
    np.savez(os.path.join(a.outdir, "grid_fields.npz"), **g, MAPFAC=mf,
             WRF_LANDMASK=wrf["LANDMASK"])
    rep = verify(g, wrf, uv, mf, dims)
    with open(os.path.join(a.outdir, "grid_verification.txt"), "w") as fh:
        fh.write(rep + "\n")
    print(rep)
    print(f"tiles: {npx} x {npy} of {a.snx} x {a.sny}")


if __name__ == "__main__":
    main()
