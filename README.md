# MITgcm grid co-located with WRF `geo_em_d01` (Antarctic polar stereographic)

## Domain
| | |
|---|---|
| Projection | WRF polar stereographic (MAP_PROJ=2), true latitude 60°S, STAND_LON 0°, centred on the South Pole |
| Size | 180 × 180 tracer cells × 52 levels (6000 m) |
| Nominal spacing | 60 km at 60°S; true spacing 45.4 km (corners, ~25°S) to 64.3 km (pole) |
| Earth model | sphere, R = 6370 km (same as WRF) |
| Domain edge | ~41°S at edge midpoints, ~25°S at the corners |

## Co-location
Both models use an Arakawa C-grid, so every staggered point is the same point:

| MITgcm | WRF | max offset |
|---|---|---|
| tracer `XC, YC` (i, j) | mass point `XLAT_M, XLONG_M` (i, j) | 3 m |
| u point (west face of cell i) | `XLAT_U, XLONG_U` (i) | float32 noise |
| v point (south face of cell j) | `XLAT_V, XLONG_V` (j) | float32 noise |
| corner `XG, YG` (SW corner) | `XLAT_C, XLONG_C` (i, j) | 3 m |

Index (i, j) in MITgcm = (west_east, south_north) in WRF, with no flips and no offset. The ~3 m
offsets come from WRF storing lat/lon as float32. These files are computed in double
precision from the projection itself.

## Metrics
All 16 MITgcm metrics are **exact integrals over the projection plane**: lengths are ∫ ds/m and areas
are ∫∫ dA/m², with m the polar stereographic map factor, computed by 8-point Gauss–Legendre
quadrature. They agree with WRF's own `DX/MAPFAC_M` to 2e-6 and `DX·DY/MAPFAC_M²` to 9e-6.
Those residuals are the cell-to-cell variation of m, which a single-point map factor
can't represent. The extra (Nx+1, Ny+1) row and column are computed from the projection,
not extrapolated.

## Rotation angles (important for wind/stress forcing)
`AngleCS, AngleSN` follow the MITgcm convention:

    u_east  = AngleCS*u_grid - AngleSN*v_grid
    v_north = AngleSN*u_grid + AngleCS*v_grid

`AngleCS` equals WRF `COSALPHA`, but **`AngleSN` = −`SINALPHA`**. WRF stores `SINALPHA` with the
opposite sign convention. The sign used here was checked against WRF's own coordinates: at
the east edge, +x increases latitude, so the grid x-axis points north there and SN must be +1.
In practice, WRF grid-relative winds can be passed to MITgcm unrotated, because the grids
share axes: WRF u/v along its x/y are already MITgcm u/v. Rotate only if you have
earth-relative (east/north) winds, and then use these angles, not WRF's SINALPHA.
The grid file contains the angles as records 17–18, so MITgcm uses them directly.

## Files
| File | Content |
|---|---|
| `tiles/grid_antarctic60km.face001…036.bin` | grid, one file per 30×30 tile (no-exch2 build; matches `SIZE.h`) |
| `grid_antarctic60km.face001.bin` | same grid as one 180×180 facet (exch2 build, any tiling) |
| `bathy_antarctic60km_wrfmask.bin` | bathymetry with land/sea mask **identical to WRF LANDMASK** (recommended for coupling) |
| `bathy_antarctic60km.bin` | bathymetry with ETOPO2-derived mask |
| `SIZE.h` | 6×6 processes, 30×30 tiles, OLx=OLy=4, Nr=52 |
| `data_grid_snippet.txt` | PARM04/PARM05 entries, exch2 alternative |
| `XC.bin YC.bin AngleCS.bin AngleSN.bin` | 180×180 real*4 BE, handy for building forcing/OBCS files |
| `grid_check.png` | bathymetry, graticule + tiles, spacing, mask vs WRF |
| `old_grid_io/prec32/`, `old_grid_io/prec64/` | same grid as the 16 separate files read with `#define OLD_GRID_IO` (see `old_grid_io/README_OLD_GRID_IO.md`) |
| `scripts/` | `make_mitgcm_grid.py`, `make_bathy.py`, `make_old_grid_io.py`, to regenerate (numpy + scipy only) |

Grid files are big-endian real*8 with 18 records of (sNx+1)×(sNy+1), i fastest, in the order
XC YC DXF DYF RA XG YG DXV DYU RAZ DXC DYC RAW RAS DXG DYG AngleCS AngleSN. This is the
order `ini_curvilinear_grid.F` reads when `horizGridFile` is set.

To change the tiling (sNx and sNy must divide 180):

    python3 scripts/make_mitgcm_grid.py geo_em_d01.nc OUTDIR --name grid_antarctic60km --snx 45 --sny 36

## Bathymetry
ETOPO2v2g nodes (166–62,000 per cell) are projected into the WRF plane and binned into each
60 km cell, which gives an exact cell mean with no interpolation. Depth is the mean of
min(z,0); a cell is ocean if at least 50% of its nodes are below sea level. Depths are
clipped at 6000 m, the bottom of the delZ grid. Dead-end inlets one cell wide and
disconnected basins were removed; there were none to remove.

ETOPO2's mask agrees with WRF LANDMASK in 99.47% of cells, differing in 171 coastal cells.
The `_wrfmask` file forces exact agreement: 103 cells are set to land and 68 to ocean. Cells
newly made ocean get at least 9.2 m depth (the top two levels). It keeps 3 tiny enclosed
water bodies (1–2 cells) because WRF has them as water.

**Caveats**
* ETOPO2 gives the **ice-surface** elevation over Antarctica. The Ross, Ronne-Filchner,
  Amery and other ice shelves are therefore land, with no sub-shelf cavities. WRF's
  LANDMASK treats them the same way, so the two models are consistent. If you later want
  cavities (pkg/shelfice), use RTopo-2 or BedMachine Antarctica and regenerate.
* All four lateral boundaries are open ocean (W 128, E 143, S 135, N 178 wet cells).
  You need pkg/obcs (or a sponge or relaxation zone). Without exch2, MITgcm is periodic
  by default, which is wrong here.
* `XC.bin`, `YC.bin`, `AngleCS.bin` and `AngleSN.bin` are the arrays you need to interpolate
  initial conditions and OBCS fields (e.g. from ECCO, GLORYS or WOA) onto this grid.
