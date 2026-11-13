#!/usr/bin/env python3
"""
Cell-average a global topography (ETOPO2v2, ETOPO 2022, GEBCO; NetCDF-3 or -4) onto the MITgcm/WRF polar stereographic grid.

Every ETOPO2 node is projected into the WRF plane and binned into the
60 km tracer cell that contains it (exact cell membership, no interpolation).
Per cell:
    wet fraction  = fraction of nodes with z < 0
    depth         = mean of min(z,0) over all nodes in the cell
                    (land counts as 0 -> no deep bias along coasts)
    land if wet fraction < 0.5
Then: clip to the deepest model level, remove lakes / enclosed basins not
connected to the open ocean, and remove 1-cell-wide dead-end inlets that the
C-grid cannot ventilate.

usage: make_bathy.py ETOPO2v2g_f4.nc geo_em.d01.nc OUTDIR --maxdepth 6000
"""
import argparse, os, sys
import numpy as np
from scipy.io import netcdf_file
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from make_mitgcm_grid import build, plot_grid_check  # noqa: E402


LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
Z_NAMES = ("z", "elevation", "Band1", "topo", "bathy")


def _pick(names, avail, what, path):
    for nme in names:
        if nme in avail:
            return nme
    raise SystemExit(f"{path}: no {what} variable found (looked for {names}; file has {list(avail)})")


def open_topo(path):
    """Open a global lat/lon topography file in NetCDF-3 (e.g. ETOPO2v2) or
    NetCDF-4/HDF5 (e.g. ETOPO 2022, GEBCO) format.

    Returns (handle, lat, lon, read_rows(j0, j1) -> float64 array, nlon_used).
    Handles grid-registered files (duplicate column at +180, dropped) and
    cell-registered files (ETOPO 2022: centres at +-179.99..., nothing dropped).
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:3] == b"CDF":                       # NetCDF-3 classic / 64-bit offset
        f = netcdf_file(path, "r", mmap=True)
        V = f.variables
        get = lambda nme: V[nme]
        avail = V.keys()
    elif magic[:8] == b"\x89HDF\r\n\x1a\n":       # NetCDF-4 = HDF5
        try:
            import netCDF4
            f = netCDF4.Dataset(path, "r")
            f.set_auto_maskandscale(False)        # fill value + scaling handled below
            V = f.variables
            get = lambda nme: V[nme]
            avail = V.keys()
        except ImportError:
            try:
                import h5py
            except ImportError:
                raise SystemExit(
                    f"{path} is NetCDF-4 (HDF5). Install a reader first, e.g.\n"
                    "    pip install netCDF4      (or: pip install h5py)\n"
                    "or convert it to NetCDF-3:  nccopy -k classic in.nc out.nc")
            f = h5py.File(path, "r")
            get = lambda nme: f[nme]
            avail = [k for k in f.keys() if hasattr(f[k], "shape")]
    else:
        raise SystemExit(f"{path}: not a NetCDF-3 or NetCDF-4 file (starts with {magic!r})")

    latn = _pick(LAT_NAMES, avail, "latitude", path)
    lonn = _pick(LON_NAMES, avail, "longitude", path)
    zn = _pick(Z_NAMES, avail, "elevation", path)
    lat = np.asarray(get(latn)[:], dtype=np.float64)
    lon = np.asarray(get(lonn)[:], dtype=np.float64)
    zvar = get(zn)
    if tuple(zvar.shape) != (len(lat), len(lon)):
        raise SystemExit(f"{path}: {zn} has shape {zvar.shape}, expected (lat, lon)")
    # grid-registered global files repeat the 180 meridian as the last column
    nlon = len(lon) - 1 if (lon[-1] - lon[0]) >= 360.0 - 1e-6 else len(lon)
    lon = lon[:nlon]
    attrs = getattr(zvar, "attrs", None)            # h5py
    getattr_ = (lambda k: attrs.get(k)) if attrs is not None else (lambda k: getattr(zvar, k, None))
    fill = getattr_("_FillValue")
    scale = getattr_("scale_factor"); offset = getattr_("add_offset")
    fill = None if fill is None else np.asarray(fill).ravel()[0]
    scale = None if scale is None else float(np.asarray(scale).ravel()[0])
    offset = None if offset is None else float(np.asarray(offset).ravel()[0])

    def read_rows(j0, j1):
        z = np.array(zvar[j0:j1, :nlon], dtype=np.float64)
        if fill is not None and np.isfinite(fill):
            z[z == fill] = np.nan
        if scale is not None:
            z *= scale
        if offset is not None:
            z += offset
        return z

    return f, lat, lon, read_rows, nlon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etopo"); ap.add_argument("geo"); ap.add_argument("outdir")
    ap.add_argument("--maxdepth", type=float, default=6000.0)
    ap.add_argument("--name", default="bathy_antarctic60km")
    ap.add_argument("--snx", type=int, default=30, help="tile size in x for the figure (sNx)")
    ap.add_argument("--sny", type=int, default=30, help="tile size in y for the figure (sNy)")
    ap.add_argument("--no-plot", action="store_true", help="do not write grid_check.png")
    a = ap.parse_args()

    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(a.geo)
    x0 = P.fwd(g["YG"][0, 0], g["XG"][0, 0])  # SW corner of domain
    xs, ys = x0[0], x0[1]

    f, lat, lon, read_rows, nlon_use = open_topo(a.etopo)
    print(f"{a.etopo}: {len(lat)} x {nlon_use} points, "
          f"lon {lon[0]:.4f}..{lon[-1]:.4f}, lat {lat.min():.4f}..{lat.max():.4f}")
    latmax = g["YC"].max() + 1.0 if P.h < 0 else 90.0
    latmin = -90.0 if P.h < 0 else g["YC"].min() - 1.0
    jsel = np.nonzero((lat >= latmin) & (lat <= latmax))[0]
    s = np.zeros(nx * ny); n = np.zeros(nx * ny); w = np.zeros(nx * ny)
    zmin = np.full(nx * ny, np.inf)
    for j0 in range(jsel[0], jsel[-1] + 1, 200):
        j1 = min(j0 + 200, jsel[-1] + 1)
        z = read_rows(j0, j1)
        LON, LAT = np.meshgrid(lon, lat[j0:j1])
        x, y = P.fwd(LAT, LON)
        ii = np.floor((x - xs) / dx).astype(int)
        jj = np.floor((y - ys) / dy).astype(int)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & np.isfinite(z)
        k = (jj * nx + ii)[ok]; zz = z[ok]
        s += np.bincount(k, np.minimum(zz, 0.0), nx * ny)
        n += np.bincount(k, None, nx * ny)
        w += np.bincount(k, (zz < 0).astype(float), nx * ny)
        np.minimum.at(zmin, k, zz)
    f.close()
    assert n.min() > 0, "some cells received no ETOPO points"
    depth = (s / n).reshape(ny, nx)
    wetfrac = (w / n).reshape(ny, nx)
    npts = n.reshape(ny, nx)

    d = np.where(wetfrac >= 0.5, depth, 0.0)
    d = np.maximum(d, -a.maxdepth)

    # remove 1-cell dead-end inlets iteratively (wet cell with >=3 dry C-grid neighbours)
    for _ in range(50):
        wet = d < 0
        p = np.pad(wet, 1, constant_values=True)  # open boundary counts as wet
        nwet = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        kill = wet & (nwet <= 1)
        if not kill.any():
            break
        d[kill] = 0.0
    # keep only ocean connected to the open boundary (removes lakes/isolated basins)
    lab, nl = ndimage.label(d < 0)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    lakes = (d < 0) & ~np.isin(lab, edge)
    nlake = int(lakes.sum())
    d[lakes] = 0.0

    os.makedirs(a.outdir, exist_ok=True)
    d.astype(">f4").tofile(os.path.join(a.outdir, f"{a.name}.bin"))
    np.savez(os.path.join(a.outdir, "bathy_fields.npz"), depth=d, raw=depth,
             wetfrac=wetfrac, npts=npts)

    wrfocean = wrf["LANDMASK"] < 0.5
    ocean = d < 0
    msg = [
        f"source nodes per cell: {int(npts.min())} - {int(npts.max())}",
        f"wet cells: {int(ocean.sum())} of {nx*ny}",
        f"depth range: {-d[ocean].max():.1f} - {-d.min():.1f} m (clipped at {a.maxdepth:.0f})",
        f"cells removed as lakes/isolated basins: {nlake}",
        f"open-boundary wet cells: W {int(ocean[:,0].sum())}  E {int(ocean[:,-1].sum())}  "
        f"S {int(ocean[0].sum())}  N {int(ocean[-1].sum())}",
        f"agreement with WRF LANDMASK: {100*(ocean==wrfocean).mean():.2f}%  "
        f"(MITgcm wet/WRF land {int((ocean & ~wrfocean).sum())}, "
        f"MITgcm dry/WRF ocean {int((~ocean & wrfocean).sum())})",
    ]
    # variant with land/sea mask forced to WRF LANDMASK (for coupling);
    # newly-wet cells get at least the depth of the top two levels (9.2 m)
    dmin = -9.2
    d2 = np.where(wrfocean, np.minimum(np.where(d < 0, d, depth), dmin), 0.0)
    d2 = np.maximum(d2, -a.maxdepth)
    d2.astype(">f4").tofile(os.path.join(a.outdir, f"{a.name}_wrfmask.bin"))
    msg.append(f"WRF-mask variant written: {int((d2 < 0).sum())} wet cells")
    open(os.path.join(a.outdir, "bathy_report.txt"), "w").write("\n".join(msg) + "\n")
    print("\n".join(msg))

    # check figure: bathymetry from the WRF-mask file (what the model uses); the mask
    # panel compares the topography-derived mask with WRF LANDMASK
    if not a.no_plot:
        path = os.path.join(a.outdir, "grid_check.png")
        plot_grid_check(path, P, g, wrf, (nx, ny, dx, dy), a.snx, a.sny, bathy=d2,
                        mask_bathy=d, mask_label="Topography",
                        title=f"MITgcm grid from {os.path.basename(a.geo)}: {nx} x {ny} cells "
                              f"({a.name}_wrfmask.bin)")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
