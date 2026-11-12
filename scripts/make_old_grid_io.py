#!/usr/bin/env python3
"""
Write the WRF-co-located grid in MITgcm's OLD_GRID_IO format
(#define OLD_GRID_IO in CPP_OPTIONS.h).

OLD_GRID_IO reads 16 separate global files from the run directory with
READ_REC_3D_RS, one Nx x Ny record each, big-endian, precision = readBinaryPrec:

  LONC LATC DXF DYF RA LONG LATG DXV DYU RAZ DXC DYC RAW RAS DXG DYG  (.bin)

Because files are global Nx x Ny, any SIZE.h tiling works (no per-tile files).
There is no (Nx+1, Ny+1) row/column in this format and the grid angles are not
read (MITgcm computes them in CALC_GRID_ANGLES).

usage: make_old_grid_io.py geo_em.d01.nc OUTDIR [--prec 32|64|both]
       [--bathy bathy_antarctic60km_wrfmask.bin]   (real*4 BE; copied to real*8 for prec 64)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_mitgcm_grid import build  # noqa: E402

# MITgcm OLD_GRID_IO file name -> field computed by make_mitgcm_grid.build
FILES = [("LONC", "XC"), ("LATC", "YC"), ("DXF", "DXF"), ("DYF", "DYF"),
         ("RA", "RA"), ("LONG", "XG"), ("LATG", "YG"), ("DXV", "DXV"),
         ("DYU", "DYU"), ("RAZ", "RAZ"), ("DXC", "DXC"), ("DYC", "DYC"),
         ("RAW", "RAW"), ("RAS", "RAS"), ("DXG", "DXG"), ("DYG", "DYG")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geo"); ap.add_argument("outdir")
    ap.add_argument("--prec", default="both", choices=["32", "64", "both"])
    ap.add_argument("--bathy", default=None)
    a = ap.parse_args()
    P, g, wrf, uv, mf, (nx, ny, dx, dy) = build(a.geo)
    precs = ["32", "64"] if a.prec == "both" else [a.prec]
    for p in precs:
        d = os.path.join(a.outdir, f"prec{p}")
        os.makedirs(d, exist_ok=True)
        dt = ">f4" if p == "32" else ">f8"
        for fname, key in FILES:
            g[key][:ny, :nx].astype(dt).tofile(os.path.join(d, fname + ".bin"))
        if a.bathy:
            b = np.fromfile(a.bathy, ">f4").reshape(ny, nx)
            b.astype(dt).tofile(os.path.join(d, os.path.basename(a.bathy)))
        print(f"wrote {len(FILES)} grid files ({dt}, {nx}x{ny}) to {d}")


if __name__ == "__main__":
    main()
