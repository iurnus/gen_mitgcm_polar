#!/usr/bin/env python3
"""
MITgcm initial conditions on the WRF-co-located polar-stereographic (curvilinear)
grid, from a GLORYS (or synthetic GLORYS-like) NetCDF file.

Grid and rotation come from make_mitgcm_grid.build(geo_em), so the points are
exactly the ones MITgcm uses:
    T, S, Eta : tracer points    (XC, YC)
    U         : west faces       (x_g, y_c in the projection plane)
    V         : south faces      (x_c, y_g)
GLORYS uo/vo are eastward/northward; they are rotated onto the model axes with
the angle evaluated at each U/V point:
    u_model =  CS*uE + SN*vN        v_model = -SN*uE + CS*vN
(inverse of MITgcm's uE = CS*u - SN*v, vN = SN*u + CS*v).

Processing is done one GLORYS depth level at a time, so a full 1/12-deg
Southern Ocean snapshot (~3 GB) does not have to fit in memory:
  for each GLORYS level:
      fill land NaNs from nearest wet point  ->  bilinear interpolation to model points
      (periodic in longitude; latitudes south of GLORYS' 80 S limit take the 80 S value)
  then vertical interpolation to model cell centres (delZ from your `data` file),
  rotation of velocities, and masking with your bathymetry (U/V = 0 where dry).

Outputs (big-endian, float32 unless --prec 64; shape (Nr, Ny, Nx) = Fortran (Nx,Ny,Nr)):
    T_init.bin S_init.bin U_init.bin V_init.bin Eta_init.bin
    data_snippet.txt   ic_report.txt   ic_check.png

Input formats: NetCDF-3 (scipy) or NetCDF-4/HDF5, read with `netCDF4` if installed,
otherwise `h5py` (pip install netCDF4  or  pip install h5py). Packed variables
(int16 + scale_factor/add_offset/_FillValue) are unpacked automatically.

usage:
  make_mitgcm_ic_ps.py GLORYS.nc geo_em.d01.nc --data data --bathy bathy.bin [--outdir ic]
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mitgcm_grid import build  # noqa: E402


# ============================================================== GLORYS reader
class _Var:
    """Uniform view of a variable: .dims (names), .attr(name), [index] -> raw ndarray."""

    def __init__(self, obj, dims, attr):
        self.obj, self.dims, self.attr = obj, dims, attr

    def __getitem__(self, idx):
        return np.asarray(self.obj[idx])


def _h5_dims(v):
    """Dimension names of an h5py dataset written by netCDF-4 (via attached scales)."""
    names = []
    try:
        for d in v.dims:
            keys = list(d.keys())
            names.append(keys[0].split("/")[-1] if keys else "")
    except Exception:
        names = [""] * v.ndim
    if len(names) == v.ndim and all(names):
        return names
    # fallback: standard Copernicus order
    std = {4: ["time", "depth", "latitude", "longitude"], 3: ["time", "latitude", "longitude"],
           1: [v.name.split("/")[-1]]}
    return std.get(v.ndim, names)


def open_netcdf(path):
    """Open NetCDF-3 (scipy) or NetCDF-4/HDF5 (netCDF4, else h5py).
    Returns (handle, getvar(name) -> _Var). No automatic masking/scaling;
    Glorys.read() applies _FillValue / scale_factor / add_offset itself."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:3] == b"CDF":
        from scipy.io import netcdf_file
        f = netcdf_file(path, "r", mmap=True)
        if magic[3] not in (1, 2):
            raise SystemExit(f"{path}: NetCDF-3 CDF-5 is not readable by scipy; "
                             "convert with  nccopy -k 64-bit-offset in.nc out.nc")

        def getvar(n):
            v = f.variables[n]
            return _Var(v, list(v.dimensions), lambda k, v=v: getattr(v, k, None))
        return f, getvar
    if magic == b"\x89HDF\r\n\x1a\n":
        try:
            import netCDF4
            f = netCDF4.Dataset(path, "r")
            f.set_auto_maskandscale(False)

            def getvar(n):
                v = f.variables[n]
                return _Var(v, list(v.dimensions),
                            lambda k, v=v: v.getncattr(k) if k in v.ncattrs() else None)
            return f, getvar
        except ImportError:
            pass
        try:
            import h5py
        except ImportError:
            raise SystemExit(
                f"{path} is NetCDF-4 (HDF5). Install a reader first:\n"
                "    pip install netCDF4        (or: pip install h5py)\n"
                "or convert it:  nccopy -k 64-bit-offset in.nc out.nc")
        f = h5py.File(path, "r")

        def getvar(n):
            v = f[n]
            return _Var(v, _h5_dims(v), lambda k, v=v: v.attrs.get(k))
        return f, getvar
    raise SystemExit(f"{path}: not a NetCDF-3 or NetCDF-4 file (starts with {magic!r})")


def _scalar(x):
    return None if x is None else np.asarray(x).ravel()[0]


class Glorys:
    """Level-by-level reader for GLORYS NetCDF-3 or NetCDF-4 files."""

    def __init__(self, path, time_index=0):
        self.t = time_index
        self.f, self.getvar = open_netcdf(path)
        get = lambda n: self.getvar(n)[:]
        self.lon = np.asarray(get("longitude"), dtype="f8")
        self.lat = np.asarray(get("latitude"), dtype="f8")
        self.depth = np.asarray(get("depth"), dtype="f8")
        self.lon = np.where(self.lon >= 180, self.lon - 360, self.lon)
        order = np.argsort(self.lon)
        self.lon_order = order if np.any(order != np.arange(order.size)) else None
        self.lon = self.lon[order]
        if self.lat[0] > self.lat[-1]:
            raise SystemExit("latitude must be increasing")
        dl = np.diff(self.lon); dp = np.diff(self.lat)
        if np.ptp(dl) > 1e-4 or np.ptp(dp) > 1e-4:
            raise SystemExit("GLORYS lon/lat must be regularly spaced")
        self.dlon, self.dlat = dl.mean(), dp.mean()
        self.periodic = abs(self.lon[-1] + self.dlon - self.lon[0] - 360.0) < 1e-3

    def read(self, name, k=None):
        """2-D (lat, lon) float64 field for one time and (if 3-D) one depth level; NaN = no data."""
        v = self.getvar(name)
        idx = []
        for d in v.dims:
            if d == "time":
                idx.append(self.t)
            elif d == "depth":
                idx.append(k)
            else:
                idx.append(slice(None))
        raw = v[tuple(idx)]
        if raw.ndim != 2:
            raise SystemExit(f"{name}: expected a 2-D slice after selecting time/depth, "
                             f"got shape {raw.shape} (dims {v.dims})")
        a = raw.astype("f8")
        bad = ~np.isfinite(a)
        for key in ("_FillValue", "missing_value"):
            fv = _scalar(v.attr(key))
            if fv is not None and np.isfinite(fv):
                bad |= raw == np.asarray(fv).astype(raw.dtype)
        sf, ao = _scalar(v.attr("scale_factor")), _scalar(v.attr("add_offset"))
        if sf is not None:
            a *= float(sf)
        if ao is not None:
            a += float(ao)
        a[bad] = np.nan
        if self.lon_order is not None:
            a = a[:, self.lon_order]
        return a


# ============================================================== helpers
def read_delz(path):
    txt = open(path).read()
    txt = "\n".join(l.split("#")[0] for l in txt.splitlines())
    m = re.search(r"\bdel[RZ]\s*=\s*(.*?)(?=\n\s*[A-Za-z_]\w*\s*=|&|/)", txt, re.S | re.I)
    if not m:
        raise SystemExit(f"no delR/delZ found in {path}")
    vals = []
    for tok in re.findall(r"(\d+\*)?([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", m.group(1)):
        rep = int(tok[0][:-1]) if tok[0] else 1
        vals += [float(tok[1].replace("d", "e").replace("D", "e"))] * rep
    return np.array(vals)


def fill_nearest(f):
    """Replace NaNs by the nearest finite value. Returns None if all NaN."""
    bad = ~np.isfinite(f)
    if not bad.any():
        return f
    if bad.all():
        return None
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return f[tuple(idx)]


class Interp:
    """Bilinear interpolation from the GLORYS grid to fixed sets of points."""

    def __init__(self, G, pts):
        self.G = G
        self.coords = {}
        for key, (lat, lon) in pts.items():
            fi = ((lon - G.lon[0]) / G.dlon)
            if G.periodic:
                fi = np.mod(fi, G.lon.size)              # may fall in [n-1, n): wrap column
            else:
                fi = np.clip(fi, 0, G.lon.size - 1)
            fj = np.clip((lat - G.lat[0]) / G.dlat, 0, G.lat.size - 1)
            self.coords[key] = np.vstack([fj.ravel(), fi.ravel()])
            self.shape = lat.shape

    def __call__(self, f, key):
        if self.G.periodic:
            f = np.concatenate([f, f[:, :1]], axis=1)
        return map_coordinates(f, self.coords[key], order=1, mode="nearest").reshape(self.shape)


def vinterp(f, z_src, z_dst):
    k = np.clip(np.searchsorted(z_src, z_dst) - 1, 0, z_src.size - 2)
    w = np.clip((z_dst - z_src[k]) / (z_src[k + 1] - z_src[k]), 0.0, 1.0)[:, None, None]
    return (1 - w) * f[k] + w * f[k + 1]


def angles(P, lon):
    lam = np.deg2rad(lon) - P.lam0
    return (np.cos(lam), np.sin(lam)) if P.h < 0 else (np.cos(lam), -np.sin(lam))


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glorys")
    ap.add_argument("geo")
    ap.add_argument("--data", required=True, help="MITgcm data file (delZ/delR is read from it)")
    ap.add_argument("--bathy", required=True, help="MITgcm bathymetry (>f4, negative = ocean)")
    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--prec", type=int, choices=[32, 64], default=32)
    ap.add_argument("--outdir", default="ic")
    ap.add_argument("--prefix", default="", help="prefix for output file names")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # ---------------- model grid
    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(a.geo)
    delR = read_delz(a.data)
    rF = np.concatenate([[0.0], np.cumsum(delR)])
    RC = 0.5 * (rF[:-1] + rF[1:])
    Nr = delR.size
    bathy = np.fromfile(a.bathy, ">f4")
    if bathy.size != nx * ny:
        raise SystemExit(f"bathy has {bathy.size} values, grid needs {nx}x{ny}")
    H = -bathy.reshape(ny, nx).astype("f8")                  # positive depth, 0 = land

    # point positions (projection plane -> lat/lon)
    x0, y0 = P.fwd(g["YC"][0, 0], g["XC"][0, 0])
    x0 = np.round(x0 / (dx / 2)) * (dx / 2); y0 = np.round(y0 / (dy / 2)) * (dy / 2)
    xc = x0 + dx * np.arange(nx); yc = y0 + dy * np.arange(ny)
    xg = xc - dx / 2;             yg = yc - dy / 2
    latC, lonC = g["YC"][:ny, :nx], g["XC"][:ny, :nx]
    latU, lonU = P.inv(xg[None, :], yc[:, None])
    latV, lonV = P.inv(xc[None, :], yg[:, None])
    csU, snU = angles(P, lonU)
    csV, snV = angles(P, lonV)

    # ---------------- GLORYS, level by level
    G = Glorys(a.glorys, a.time_index)
    print(f"GLORYS: {G.lon.size} x {G.lat.size} x {G.depth.size}, res {G.dlon:.4f} deg, "
          f"lat {G.lat[0]:.2f}..{G.lat[-1]:.2f}, periodic={G.periodic}")
    if not G.periodic and (lonC.min() < G.lon[0] or lonC.max() > G.lon[-1]):
        print("WARNING: GLORYS file is not global in longitude but the model grid circles the pole")
    if G.lat[-1] < latC.max():
        print(f"WARNING: model reaches {latC.max():.2f} N but GLORYS stops at {G.lat[-1]:.2f}; "
              "download a larger latitude range")
    I = Interp(G, {"C": (latC, lonC), "U": (latU, lonU), "V": (latV, lonV)})

    nzg = G.depth.size
    Tg = np.empty((nzg, ny, nx)); Sg = np.empty_like(Tg)
    Ug = np.empty_like(Tg); Vg = np.empty_like(Tg)
    for k in range(nzg):
        out = {}
        for name in ("thetao", "so", "uo", "vo"):
            f = fill_nearest(G.read(name, k))
            out[name] = f
        if any(v is None for v in out.values()):               # entirely dry level
            Tg[k], Sg[k], Ug[k], Vg[k] = Tg[k - 1], Sg[k - 1], Ug[k - 1], Vg[k - 1]
            continue
        Tg[k] = I(out["thetao"], "C")
        Sg[k] = I(out["so"], "C")
        uE_U, vN_U = I(out["uo"], "U"), I(out["vo"], "U")
        uE_V, vN_V = I(out["uo"], "V"), I(out["vo"], "V")
        Ug[k] = csU * uE_U + snU * vN_U                        # rotate to model axes
        Vg[k] = -snV * uE_V + csV * vN_V
        if (k + 1) % 10 == 0 or k == nzg - 1:
            print(f"  GLORYS level {k + 1}/{nzg} ({G.depth[k]:.0f} m) done")
    Eta = I(fill_nearest(G.read("zos")), "C")

    # ---------------- vertical interpolation to model levels
    T = vinterp(Tg, G.depth, RC)
    S = vinterp(Sg, G.depth, RC)
    U = vinterp(Ug, G.depth, RC)
    V = vinterp(Vg, G.depth, RC)

    # ---------------- masks from the model bathymetry
    wetC = rF[:-1, None, None] < H[None]                     # cell has some water
    wetW = wetC.copy(); wetW[:, :, 1:] &= wetC[:, :, :-1]    # west face: both neighbours wet
    wetS = wetC.copy(); wetS[:, 1:, :] &= wetC[:, :-1, :]
    U = np.where(wetW, U, 0.0)
    V = np.where(wetS, V, 0.0)
    Eta = np.where(H > 0, Eta, 0.0)
    # T/S keep the filled values on land (ignored by MITgcm, avoids sharp jumps)

    for name, arr in (("T", T), ("S", S), ("U", U), ("V", V), ("Eta", Eta)):
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"ERROR: non-finite values in {name}")

    # ---------------- write
    dt = ">f4" if a.prec == 32 else ">f8"
    files = {k: f"{a.prefix}{k}_init.bin" for k in ("T", "S", "U", "V", "Eta")}
    for k, arr in (("T", T), ("S", S), ("U", U), ("V", V), ("Eta", Eta)):
        np.ascontiguousarray(arr).astype(dt).tofile(os.path.join(a.outdir, files[k]))

    snippet = f""" &PARM01
 readBinaryPrec={a.prec},
 &
 &PARM05
 hydrogThetaFile='{files['T']}',
 hydrogSaltFile='{files['S']}',
 uVelInitFile='{files['U']}',
 vVelInitFile='{files['V']}',
 pSurfInitFile='{files['Eta']}',
 &
# U/V are grid-relative (model i/j directions), already rotated from east/north.
# Arrays: Nx={nx}, Ny={ny}, Nr={Nr} (delZ sum {delR.sum():.0f} m)
"""
    open(os.path.join(a.outdir, "data_snippet.txt"), "w").write(snippet)

    south = (latC < G.lat[0]) & (H > 0)
    lines = [f"model grid {nx} x {ny} x {Nr}; wet columns {int((H > 0).sum())}",
             f"wet model columns south of GLORYS limit ({G.lat[0]:.1f}): {int(south.sum())} "
             "(filled from nearest GLORYS row)",
             "field    min      max      mean   (wet points)"]
    for name, arr, m in (("T", T, wetC), ("S", S, wetC), ("U", U, wetW), ("V", V, wetS)):
        w = arr[m]
        lines.append(f"{name:4s} {w.min():8.3f} {w.max():8.3f} {w.mean():8.3f}")
    w = Eta[H > 0]
    lines.append(f"{'Eta':4s} {w.min():8.3f} {w.max():8.3f} {w.mean():8.3f}")
    spd = np.hypot(U[0], V[0])
    lines.append(f"max surface speed {spd.max():.3f} m/s")
    rep = "\n".join(lines)
    open(os.path.join(a.outdir, "ic_report.txt"), "w").write(rep + "\n")
    print(rep)

    if not a.no_plot:
        quicklook(os.path.join(a.outdir, "ic_check.png"), T, S, U, V, Eta, H, RC, wetC)
    print(f"wrote {a.outdir}/  (see data_snippet.txt)")


def quicklook(path, T, S, U, V, Eta, H, RC, wetC):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ny, nx = H.shape
    land = H <= 0
    k500 = int(np.argmin(np.abs(RC - 500)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

    def show(a_, f, title, cmap, label, vlim=None):
        m = np.ma.masked_where(land, f)
        pc = a_.pcolormesh(m, cmap=cmap, vmin=None if vlim is None else vlim[0],
                           vmax=None if vlim is None else vlim[1])
        a_.contour(land.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        a_.set_title(title); a_.set_aspect("equal")
        fig.colorbar(pc, ax=a_, shrink=0.8, label=label)

    show(ax[0, 0], T[0], "Surface T + velocity (grid-relative)", "RdYlBu_r", "°C")
    sk = 6
    jj, ii = np.mgrid[0:ny:sk, 0:nx:sk]
    uc = 0.5 * (U[0][:, :] + np.roll(U[0], -1, axis=1))     # to cell centres for plotting
    vc = 0.5 * (V[0][:, :] + np.roll(V[0], -1, axis=0))
    ax[0, 0].quiver(ii + 0.5, jj + 0.5, np.ma.masked_where(land[::sk, ::sk], uc[::sk, ::sk]),
                    np.ma.masked_where(land[::sk, ::sk], vc[::sk, ::sk]), scale=6, width=0.003)
    show(ax[0, 1], S[0], "Surface S", "viridis", "psu")
    show(ax[1, 0], Eta, "Eta (initial free surface)", "RdBu_r", "m")
    show(ax[1, 1], T[k500], f"T at {RC[k500]:.0f} m", "RdYlBu_r", "°C")
    for a_ in ax.ravel():
        a_.set_xlabel("i"); a_.set_ylabel("j")
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
