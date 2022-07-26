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


def fix_time(ds):
    # we need be carefull witht he attributes in the time data array, because of a weird xarray bug: https://github.com/pydata/xarray/issues/3739
        t = ds.time
        time1 = pd.to_datetime(t.values, format='%Y%m%d')
        hours = np.round((t.values % 1) * 24,2)
        minutes = (hours % 1) * 60
        time2 = pd.to_datetime(hours, format='%H')
        time3 = pd.to_datetime(minutes, format='%M')
        return xr.DataArray(
                pd.to_datetime(pd.to_numeric(time3-time3[0]) + pd.to_numeric(time2-time2[0]) + pd.to_numeric(time1)),
                dims=t.dims,attrs={k:v for k,v in t.attrs.items() if k != "units" and k != "calendar"})


def mask_coord(dset):
        filegr = '/work/mh0287/m300083/experiments/dpp0066/bc_land_frac.nc'
        gridset = xr.open_dataset(filegr,chunks='auto').rename({'cell':'ncells'})
        land_sea_mask = gridset.notsea.persist()
        mask = land_sea_mask.values
        clat = gridset.clat.values*180/np.pi
        clon = gridset.clon.values*180/np.pi
        dset = dset.assign_coords(clon=("ncells",clon),clat=("ncells",clat))
        dset = dset.assign_coords(land_sea_mask=("ncells",mask))
        return dset


def area_model(lat,lon,mask):
    file = '/work/mh0287/m300083/experiments/dpp0066/icon_grid_0015_R02B09_G.nc'
    dset = xr.open_dataset(file,chunks=({'cell': 2097152})).cell_area_p.rename({'cell':'ncells'})
    if mask is not None :
        #####mask
        filegr = '/work/mh0287/m300083/experiments/dpp0066/bc_land_frac.nc'
        gridset = xr.open_dataset(filegr,chunks='auto').rename({'cell':'ncells'})
        land_sea_mask = gridset.notsea.persist()
        maskl = land_sea_mask.values
        area = dset.where(((dset.clat>=lat[0]/180*np.pi) & (dset.clat<=lat[1]/180*np.pi)) & ((dset.clon>=lon[0]/180*np.pi) & (dset.clon<=lon[1]/180*np.pi)) & (maskl==mask),drop=True)
    else:
        area = dset.where(((dset.clat>=lat[0]/180*np.pi) & (dset.clat<=lat[1]/180*np.pi)) & ((dset.clon>=lon[0]/180*np.pi) & (dset.clon<=lon[1]/180*np.pi)),drop=True)
    global_area = area.sum()
    return area, global_area


def calculate_spatial(data,area,global_area,lat,lon,mask=None):
    if mask is not None:
        dsel = (data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True).chunk({'time':15,'ncells':1000000}) * area).sum('ncells')/global_area
    else:
        dsel = (data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True).chunk({'time':15,'ncells':1000000}) * area).sum('ncells')/global_area
    return dsel

def calculate_temporal(data,lat,lon,mask=None):
    if mask is not None:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True).mean('time')
    else:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True).mean('time')
    return dsel

def calculate_subset(data,lat,lon,mask=None):
    if mask is not None:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])) & (data.land_sea_mask==mask),drop=True)
    else:
        dsel = data.where(((data.clat>=lat[0]) & (data.clat<=lat[1])) & ((data.clon>=lon[0]) & (data.clon<=lon[1])),drop=True)
    return dsel

######final functions
######spatial mean
def spatial_mean(files,var,lat,lon,outfile,path,mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var]
    if better_time == True :
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).resample(time='1D').mean('time')
    else:
        variable = dsetvar.resample(time='1D').mean('time')
    ####area
    variable = mask_coord(variable)
    area,global_area = area_model(lat,lon,mask)
    ###calculate
    cal_mean = calculate_spatial(variable,area,global_area,lat,lon,mask)
    ###dask
    jobs = dask.persist(cal_mean)
    progress(jobs, notebook=False)
    var_final = dask.compute(cal_mean)
    var_final[0].to_netcdf(path+var+outfile+'.nc', format = 'NETCDF4', mode ='w', group=None)
    return print('work done')

#####temporal functions

def temporal_mean(files,var,times,lat,lon,outfile,path,mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks='auto')[var]
    if better_time == True :
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).sel(time=slice(times[0],times[1]))
    else:
        variable = dsetvar.sel(time=slice(times[0],times[1]))
    ####area
    variable = mask_coord(variable)
    ###calculate
    cal_mean = calculate_temporal(variable,lat,lon,mask)
    ###dask
    jobs = dask.persist(cal_mean)
    progress(jobs, notebook=False)
    var_final = dask.compute(cal_mean)
    var_final[0].to_netcdf(path+var+outfile+'.nc', format = 'NETCDF4', mode ='w', group=None)
    return print('work done')

def ssubset(files,var,times,lat,lon,outfile,path,t_sample='1D',mask=None,better_time=True):
    ### mask = 0 to select ocean
    ### mask = 1 to select land
    dsetvar = xr.open_mfdataset(files, engine='netcdf4',combine='by_coords',chunks={'time':1,'ncells':-1})[var]
    if better_time == True :
        variable = dsetvar.assign_coords(time=fix_time(dsetvar)).sel(time=slice(times[0],times[1])).resample(time=t_sample).mean('time')
    else:
        variable = dsetvar.sel(time=slice(times[0],times[1])).resample(time=t_sample).mean('time')
    ####area
    variable = mask_coord(variable)
    ###calculate
    cal_mean = calculate_subset(variable,lat,lon,mask)
    write_dat = cal_mean.to_netcdf(path+var+outfile+'.nc', format = 'NETCDF4', mode ='w', group=None,compute=False)
    write_dat.compute()
    return print('work done')

###to improve, to save or to load in notebook.
