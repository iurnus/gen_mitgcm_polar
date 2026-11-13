## generate bathymetry
python3 scripts/make_bathy.py ETOPO_2022_v1_60s_N90W180_surface.nc geo_em.d01.nc grid --maxdepth 6000

## generate grid, with MITgcm "old" style
python3 scripts/make_old_grid_io.py geo_em.d01.nc grid --prec 32 --bathy grid/bathy_antarctic60km_wrfmask.bin

## generate MITgcm initial conditions
python3 scripts/make_mitgcm_ic_ps.py save_glorys/CMEMS_SO_2016-01-01.nc geo_em.d01.nc \
    --data data --bathy grid/bathy_antarctic60km_wrfmask.bin --outdir ic_obcs

## download the data
python3 scripts/download_glorys_obc_ps.py geo_em.d01.nc --bathy grid/bathy_antarctic60km_wrfmask.bin \
    --start 2016-01-01 --end 2019-01-07 --outdir save_glorys

## download the data
python3 save_glorys/download.py
mv CMEMS_SO_2016-01-01.nc save_glorys

## generate the boundary conditions
python3 scripts/make_mitgcm_obc_ps.py save_glorys geo_em.d01.nc --data data \
    --bathy grid/bathy_antarctic60km_wrfmask.bin --outdir ic_obcs
