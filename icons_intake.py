from getpass import getuser # Libaray to copy things
from pathlib import Path # Object oriented libary to deal with paths
from tempfile import NamedTemporaryFile, TemporaryDirectory # Creating temporary Files/Dirs
from dask.utils import format_bytes
from distributed import Client, progress, wait # Libaray to orchestrate distributed resources
from dask_jobqueue import SLURMCluster # Setting up distributed memories via slurm
import numpy as np # Pythons standard array library
import xarray as xr # Libary to work with labeled n-dimensional data
import dask # Distributed data libary
import dask.distributed
import multiprocessing
from subprocess import run, PIPE
import sys
import os
import warnings
warnings.filterwarnings(action='ignore')
dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
import pandas as pd
import intake

cat = intake.open_catalog("/home/m/m300827/nextgems/C2_hackathon_prep/getting_started/catalog.yaml")

def prepare_cpu(nworker=1,memory='64GB'):
    '''Provides information about the CPU and starts the dask client'''
    ncpu = multiprocessing.cpu_count()
    processes = False
    nworker = nworker
    threads = ncpu // nworker
    print(f"Number of CPUs: {ncpu}, number of threads: {threads}, number of workers: {nworker}, processes: {processes}")
    client = Client(processes=processes,
        threads_per_worker=threads,
        n_workers=nworker,
        memory_limit=memory
        )
    return client

def mask_coord(dset):
        filegr = '/work/mh0287/m300083/experiments/dpp0066/bc_land_frac.nc'
        gridset = xr.open_dataset(filegr,chunks='auto')
        land_sea_mask = gridset.notsea.persist()
        mask = land_sea_mask.values
        clat = gridset.clat.values*180/np.pi
        clon = gridset.clon.values*180/np.pi
        dset = dset.assign_coords(clon=("cell",clon),clat=("cell",clat))
        dset = dset.assign_coords(land_sea_mask=("cell",mask))
        return dset

def area_model(lat,lon,mask):
    file = '/work/mh0287/m300083/experiments/dpp0066/icon_grid_0015_R02B09_G.nc'
    dset = xr.open_dataset(file,chunks=({'cell': 1000000})).cell_area_p
    if mask is not None :
        #####mask
        maskl = data.notsea
        area = dset.where(((dset.clat>=lat[0]/180*np.pi) & (dset.clat<=lat[1]/180*np.pi)) & ((dset.clon>=lon[0]/180*np.pi) & (dset.clon<=lon[1]/180*np.pi)) & (maskl==mask),drop=True)
    else:
        area = dset.where(((dset.clat>=lat[0]/180*np.pi) & (dset.clat<=lat[1]/180*np.pi)) & ((dset.clon>=lon[0]/180*np.pi) & (dset.clon<=lon[1]/180*np.pi)),drop=True)
    global_area = area.sum()
    return area, global_area


def calculate_spatial(data,area,global_area,lat,lon,mask=None):
    if mask is not None:
        dsel = (data.where(((data.clat>=lat[0]/180*np.pi) & (data.clat<=lat[1]/180*np.pi)) & ((data.clon>=lon[0]/180*np.pi) & (data.clon<=lon[1]/180*np.pi)) & (data.notsea==mask),drop=True)).sum('cell')/global_area
    else:
        dsel = (data.where(((data.clat>=lat[0]/180*np.pi) & (data.clat<=lat[1]/180*np.pi)) & ((data.clon>=lon[0]/180*np.pi) & (data.clon<=lon[1]/180*np.pi)),drop=True)).sum('cell')/global_area
    return dsel

def calculate_temporal(data,lat,lon,mask=None):
    if mask is not None:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True).mean('time')
    else:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True).mean('time')
    return dsel

def calculate_zonal(data,lat,lon,interval,mask=None):
    if mask is not None:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True)
    else:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True)
    return dsel.groupby_bins('clat',np.arange(lat[0],lat[1]+interval,interval)).mean('ncells')

def calculate_meridional(data,lat,lon,interval,mask=None):
    if mask is not None:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True)
    else:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True)
    return dsel.groupby_bins('clon',np.arange(lon[0],lon[1]+interval,interval)).mean('ncells')

######final functions
######spatial mean
def spatial_mean(simualtion,var,time,lat,lon,t_step='1D',mask=None):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    run = cat.ICON[simualtion]
    data = run.atm_2d_ml_R02B09.to_dask()
    ###Reading grid cells
    variable = mask_coord(data)

#filegr1 = '/work/mh0287/m300083/experiments/dpp0066/icon_grid_0015_R02B09_G.nc'
#    gridset1 = xr.open_dataset(filegr1,chunks={'cell':5000000}).cell_area_p
#    ###Reading land_sea mask
#    filegr2 = '/work/mh0287/m300083/experiments/dpp0066/bc_land_frac.nc'
#    gridset2 = xr.open_dataset(filegr2,chunks={'cell':5000000}).notsea
    ###mixing grids
    
#    data_new = data.merge(gridset2.notsea).merge(gridset1.cell_area_p).chunk({'time':96,'cell':10000000}).sel(time=slice(time[0],time[1])).resample(time=t_step).mean('time')
    data_new = variable.chunk({'time':1,'cell':10000000}).sel(time=slice(time[0],time[1])).resample(time=t_step).mean('time')
 ####area
    area,global_area = area_model(lat,lon,mask)
    ###calculate
    cal_mean = calculate_spatial(data_new[var],area,global_area,lat,lon,mask)
    ###dask
    return cal_mean.compute()

#####temporal functions

def temporal_mean(files,var,time,lat,lon,mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    if better_time == True :
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var]
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).sel(time=slice(time[0],time[1]))
    else:
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var].sel(time=slice(time[0],time[1]))
        variable = dsetvar
    ####area
    variable = mask_coord(variable)
    ###calculate
    cal_mean = calculate_temporal(variable,lat,lon,mask)
    ###dask
    jobs = dask.persist(cal_mean)
    progress(jobs, notebook=False)
    var_final = dask.compute(cal_mean)
    return var_final[0]

######final functions
######zonal  mean
def zonal_mean(files,var,time,lat,lon,interval,t_step='1D',mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    if better_time == True :
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var]
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).sel(time=slice(time[0],time[1])).resample(time=t_step).mean('time')
    else:
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var].sel(time=slice(time[0],time[1]))
        variable = dsetvar.resample(time=t_step).mean('time')
    ####area
    variable = mask_coord(variable)
    ###calculate
    cal_mean = calculate_zonal(variable,lat,lon,interval,mask)
    ###dask
    jobs = dask.persist(cal_mean)
    progress(jobs, notebook=False)
    var_final = dask.compute(cal_mean)
    return var_final[0]

def meridional_mean(files,var,time,lat,lon,interval,t_step='1D',mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    if better_time == True :
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var]
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).sel(time=slice(time[0],time[1])).resample(time=t_step).mean('time')
    else:
        dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var].sel(time=slice(time[0],time[1]))
        variable = dsetvar.resample(time=t_step).mean('time')
    ####area
    variable = mask_coord(variable)
    ###calculate
    cal_mean = calculate_meridional(variable,lat,lon,interval,mask)
    ###dask
    jobs = dask.persist(cal_mean)
    progress(jobs, notebook=False)
    var_final = dask.compute(cal_mean)
    return var_final[0]
###to improve, to save or to load in notebook.
