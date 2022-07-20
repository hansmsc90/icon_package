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


# To do:
# - make horizontal remapping optional
# - improve performance (reading the data)
# - poslish 2d processing
# - choose some vertical levels beforehand (e.g. 20-90)


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



def get_griddes(y_res, x_res, x_first, y_first, x_end, y_end):
        """Create a description for a regular global grid at given x, y resolution."""

        xsize = (x_end - x_first) / x_res
        ysize = (y_end - y_first) / y_res
        xfirst = x_first + x_res / 2
        yfirst = y_first + y_res / 2

        return f'''
#
# gridID 1
#
gridtype  = lonlat
gridsize  = {int(xsize * ysize)}
xsize     = {int(xsize)}
ysize     = {int(ysize)}
xname     = lon
xlongname = "longitude"
xunits    = "degrees_east"
yname     = lat
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = {xfirst}
xinc      = {x_res}
yfirst    = {yfirst}
yinc      = {y_res}


    '''


def gen_dis(dataset, xres, yres, xfirst, yfirst, xend, yend, gridfile):
    '''Create distance weights using cdo. You only need this if you do not have a corresponding weightfile.
    The weightfile will be stored in a scratch directory.

    dataset : the xarray dataset you want to interpolate
    xres, yres : longitude/latitude resolution in degrees
    gridfile : This is a file that has the source grid (right???) # Hans check this plz :)

    '''
    scratch_dir = Path('/scratch') / getuser()[0] / getuser() # Define the users scratch dir
    with TemporaryDirectory(dir=scratch_dir, prefix='Weights_') as td:
        in_file = Path(td) / 'in_file.nc'
        weightfile = scratch_dir / 'weight_file.nc'
        griddes = Path(td) / 'griddes.txt'
        with griddes.open('w') as f:
            f.write(get_griddes(xres, yres, xfirst, yfirst, xend, yend))
        #dataset = dataset.time.drop("calendar")
        dataset.to_netcdf(in_file, mode='w') # Write the file to a temorary netcdf file
        cmd = ('cdo', '-O', f'gencon,{griddes}', f'-setgrid,{gridfile}', str(in_file), str(weightfile))
        run_cmd(cmd)
        df = xr.open_dataset(weightfile).load()
        wait(df)
        return df

def run_cmd(cmd, path_extra=Path(sys.exec_prefix)/'bin'):
        '''Run a bash command.'''
        env_extra = os.environ.copy()
        env_extra['PATH'] = str(path_extra) + ':' + env_extra['PATH']
        status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
        if status.returncode != 0:
                error = f'''{' '.join(cmd)}: {status.stderr.decode('utf-8')}'''
                raise RuntimeError(f'{error}')
        return status.stdout.decode('utf-8')

@dask.delayed
def remap(dataset, x_res, y_res, x_first, y_first, x_end, y_end, gridfile, weights=None):
    """Perform a weighted remapping.
    Remaps dataset to a grid specified in gridfile.

    Parameters
    ==========

    dataset : xarray.dataset
        The dataset that will be regridded
    xres, yres : float
        resolution in degrees
    gridfile :
        Path to the file that has the correct grid
    weights : string
        Path to distance weights

    Returns
    =======
    xarray.dataset : Remapped dataset
    """

    if isinstance(dataset, xr.DataArray) :
        # If a dataArray is given create a dataset
        dataset = xr.Dataset(data_vars={dataset.name: dataset})
    scratch_dir = Path('/scratch') / getuser()[0] / getuser()  # Define the users scratch dir

    if weights==None:
        weights = gen_dis(dataset, x_res, y_res, x_first, y_first, x_end, y_end, gridfile)
        weightfile = scratch_dir / 'weight_file.nc'
    else:
        weightfile = weights
    #           weights.to_netcdf(scratch_dir / 'weight_file.nc')

    with TemporaryDirectory(dir=scratch_dir, prefix='Remap_') as td:
        infile = Path(td) / 'input_file.nc'
        griddes = Path(td) / 'griddes.txt'
        outfile = Path(td) / 'remaped_file.nc'
        with griddes.open('w') as f:
                f.write(get_griddes(x_res, y_res, x_first, y_first, x_end, y_end))
        dataset.to_netcdf(infile, mode='w') # Write the file to a temorary netcdf file
        cmd = ('cdo', '-O', f'remap,{griddes},{weightfile}', f'-setgrid,{gridfile}',
                str(infile), str(outfile))
        run_cmd(cmd)
        return xr.open_dataset(outfile).load()

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
                dims=t.dims,
        attrs={k:v for k,v in t.attrs.items() if k != "units" and k != "calendar"})


def interp1d_np(data, x, xi):
    ''' The core vertical interpolation function based on the numpy interpolate fucntion'''
    return np.interp(xi, x, data, left=float('NaN'),right=float('NaN'))

@dask.delayed
def xr_vertical_in(da,da_p,plevels) :
    '''applies the numpy function to xarray and makes sure that it is vectorized over
    the other dimensions. Follows the examples from here : https://github.com/pydata/xarray/issues/3931
    and here: http://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html

    Arguments:
    da -- the array on model levels you want interpolate
    da_p -- the corresponding array with pressure values, must have same dimensions as da
    plevels -- numpy array with the levels you want to interpolate to
    '''

    assert da.dims == da_p.dims,"Given arrays do not have the same dimensions"

    interped = xr.apply_ufunc(
        interp1d_np,  # first the function
        da,  # now arguments in the order expected by 'interp1_np'
        da_p,  # as above
        plevels,  # as above
        input_core_dims=[["height"], ["height"], ["plev"]],  # list with one entry per arg
        output_core_dims=[["plev"]],  # returned data has one dimension
        #exclude_dims=set(("height",)),  # dimensions allowed to change size. Must be a set!
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        output_dtypes=[da.dtype],  # one per output
        #allow_rechunk=True
    )
    interped["plev"] = plevels
    return interped

#@dask.delayed
def calc_temporal_mean(data,frequency) :
    return data.chunk({'ncells':80000, 'time' : -1 }).resample(time=frequency ).mean()

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


def get_data_3d(files,var,filesp,deltax,deltay,grid_file,outfile, plevs,
    xini=-180., yini=-90., xend=180., yend=90.,rangelev=None, weights=None,
    better_times=True, resample_time = True, temporal_mean='1D', pressure_name='pfull',heightchunks=30):
    '''
    This pieces everything together.
    Read the dataset
    Fix time (optional)
    Temporal mean: temporal_mean can be "D" (daily) or "H" (hourly), or even monthly ("M") if you have monthly files
    Horizontal remapping

    files: list
        list of files, has to be a list!
    var: string
        variable you want to process
    filesp: list
        list of pressure files
    deltax,deltay: float
        lon/lat resolution
    grid_file: string
        file that has the target grid already
    outfile: string
        first part of name of outfile

    xini,yini,xend,yend : float
        Borders of longitude and latitude if you want to subselect some data
    rangelev: list
        list range of levels to cur data(90 is lowest height)
    weights: string
        path of weightfile. If None then weightfile will be created in scartch directory
    better_times: Boolean
        If True time coordinate is fixed
    resample_time: Boolean
        If True then temporal mean weill be done according to temporal_mean
    pressure_name: string
        Usually "pfull" for ICON.

    '''
    for t in range(len(files)):
        ###variable
        print(files[t])
        variable = xr.open_mfdataset(files[t], engine='netcdf4',combine='by_coords', chunks={'height':heightchunks}, parallel=True)[var]
        ###pressure
        pressure = xr.open_mfdataset(filesp[t], engine='netcdf4',combine='by_coords', chunks={'height':heightchunks}, parallel=True)[pressure_name]

        if rangelev is not None :
            variable = variable.sel(height=rangelev)
            pressure = pressure.sel(height=rangelev)

        # Fix times:
        # Better Times should always be True
        if better_times == True :
            variable = variable.assign_coords(time=fix_time(variable))
            pressure = pressure.assign_coords(time=fix_time(pressure))
        if resample_time == True :
            print("temporal mean...")
            ####Hourly mean choose '1H' for daily mean '1D'
            variable = calc_temporal_mean(variable,temporal_mean)
            pressure = calc_temporal_mean(pressure,temporal_mean)

        #######interpolate variable
        print("\n horizontal remapping...")
        remap_var = remap(variable, deltax, deltay,xini,yini,xend,yend,grid_file,weights)
        #####
        remap_jobs = dask.persist(remap_var)
        progress(remap_jobs, notebook=False)
        var_rmap = dask.compute(remap_var)


        if (weights == None) & (t==0):
            scratch_dir = Path('/scratch') / getuser()[0] / getuser()
            weights = scratch_dir / 'weight_file.nc'
        #######interpolate variable
        remap_pr = remap(pressure, deltax, deltay,xini,yini,xend,yend,grid_file,weights)
        #####
        remap_jobs = dask.persist(remap_pr)
        progress(remap_jobs, notebook=False)
        pre_rmap = dask.compute(remap_pr)

        # verical interpolation
        print("\n vertical remapping...")
        var_int = xr_vertical_in(var_rmap[0][var].chunk({'lon':10,'lat':10}),pre_rmap[0]['pfull'].chunk({'lon':10,'lat':10}),plevs)
        press_jobs = dask.persist(var_int)
        progress(press_jobs, notebook=False)
        var_final = dask.compute(var_int)

        # write file
        if better_times == False :
            var_final = var_final.assign_coords(time=fix_time(variable))

        years = str(var_final[0]['time.year'].values[0])
        months = str(var_final[0]['time.month'].values[0]).zfill(2)
        days = str(var_final[0]['time.day'].values[0]).zfill(2)

        var_final[0].rename(var).to_netcdf(outfile+years+months+days+'_'+str(deltax)+'_'+str(deltay)+'.nc', format = 'NETCDF4', mode ='w', group=None)
    return print('work done')




def get_data_3d_native(files,var,filesp,outfile,plevs,lon=[-180,180],lat=[-90,90],rangelev=None,better_times=True, resample_time = True, temporal_mean='1D', pressure_name='pfull',heightchunks=30):
    '''
    This pieces everything together.
    Read the dataset
    Fix time (optional)
    Temporal mean: temporal_mean can be "D" (daily) or "H" (hourly), or even monthly ("M") if you have monthly files
    Horizontal remapping

    files: list
        list of files
    var: string
        variable you want to process
    filesp: list
        list of pressure files

        file that has the target grid already
    outfile: string
        first part of name of outfile

    lon: list
        eastern and western most point to cut the data (vmin=-180, vamax=180)
    lat: list
        southern and northern most point to cut the data (vmin=-90, vamax=90)
    rangelev: list
        list range of levels to cur data (90 is lowest height)
    better_times: Boolean
        If True time coordinate is fixed
    resample_time: Boolean
        If True then temporal mean weill be done according to temporal_mean
    pressure_name: string
        Usually "pfull" for ICON.

    '''
    for t in range(len(files)):
        print(files[t])
        print(filesp)
        ###variable
        variable = xr.open_mfdataset(files[t],preprocess=mask_coord, engine='netcdf4',combine='by_coords', chunks={'height':heightchunks},parallel=True)[var]
        ###pressure
        pressure = xr.open_mfdataset(filesp[t],preprocess=mask_coord, engine='netcdf4',combine='by_coords', chunks={'height':heightchunks},parallel=True)[pressure_name]

        ####Cutting the zone of interest
        variable = variable.where(((variable.clat>=lat[0]) & (variable.clat<=lat[1])) & ((variable.clon>=lon[0]) & (variable.clon<=lon[1])),drop=True)
        pressure = pressure.where(((pressure.clat>=lat[0]) & (pressure.clat<=lat[1])) & ((pressure.clon>=lon[0]) & (pressure.clon<=lon[1])),drop=True)

        if rangelev is not None :
            variable = variable.sel(height=rangelev)
            pressure = pressure.sel(height=rangelev)

        # Fix times:
        if better_times == True :
            variable = variable.assign_coords(time=fix_time(variable))
            pressure = pressure.assign_coords(time=fix_time(pressure))
            print(variable.time)
        if resample_time == True :
            print("\n temporal mean...")
            ####Hourly mean choose '1H' for daily mean '1D'
            variable = calc_temporal_mean(variable,temporal_mean)
            pressure = calc_temporal_mean(pressure,temporal_mean)


        # verical interpolation
        print("\n vertical remapping...")

        var_int = xr_vertical_in(variable.chunk({'height':-1}),pressure.chunk({'height':-1}),plevs)
        press_jobs = dask.persist(var_int)
        progress(press_jobs, notebook=False)
        var_final = dask.compute(var_int)

        # write file
        years = str(var_final[0]['time.year'].values[0])
        months = str(var_final[0]['time.month'].values[0]).zfill(2)
        days = str(var_final[0]['time.day'].values[0]).zfill(2)

        var_final[0].to_netcdf(outfile+years+months+days+'_'+str(lon[0])+'_'+str(lon[1])+'_'+str(lat[0])+'_'+str(lat[1])+'.nc', format = 'NETCDF4', mode ='w', group=None)
    return print('Work done')


def get_data_2d(files,var,deltax,deltay,grid_file,outfile,xini=-180., yini=-90., xend=180., yend=90., weights=None, better_times=True, resample_time = True, temporal_mean='1D'):
    '''
    This pieces everything together.
    Read the dataset
    Fix time (optional)
    Temporal mean: temporal_mean can be "D" (daily) or "H" (hourly), or even monthly ("M") if you have monthly files
    Horizontal remapping

    files: list
        list of files
    var: string
        variable you want to process
    deltax,deltay: float
        lon/lat resolution
    grid_file: string
        file that has the target grid already
    outfile: string
        first part of name of outfile

    xini,yini,xend,yend : float
        Borders of longitude and latitude if you want to subselect some data
    weights: string
        path of weightfile. If None then weightfile will be created in scartch directory
    better_times: Boolean
        If True time coordinate is fixed
    resample_time: Boolean
        If True then temporal mean weill be done according to temporal_mean

    '''
    for t in range(len(files)):
        print(files[t])
        variable = xr.open_mfdataset(files[t], engine='netcdf4',combine='by_coords', chunks={'height':30}, parallel=True)[var]

        # Fix times:
        # Better Times should always be True
        if better_times == True :
            variable = variable.assign_coords(time=fix_time(variable))
        if resample_time == True :
            print("temporal mean...")
            ####Hourly mean choose '1H' for daily mean '1D'
            variable = calc_temporal_mean(variable,temporal_mean)
        

        #######interpolate variable
        print("\n horizontal remapping...")
        remap_var = remap(variable, deltax, deltay,xini,yini,xend,yend,grid_file,weights)
        #####
        remap_jobs = dask.persist(remap_var)
        progress(remap_jobs, notebook=False)
        var_final = dask.compute(remap_var)
        
        if (weights == None) & (t==0):
            scratch_dir = Path('/scratch') / getuser()[0] / getuser()
            weights = scratch_dir / 'weight_file.nc'

#        if better_times == False :
#            var_final = var_final.assign_coords(time=fix_time(variable))
        # write file

        years = str(var_final[0]['time.year'].values[0])
        months = str(var_final[0]['time.month'].values[0]).zfill(2)
        days = str(var_final[0]['time.day'].values[0]).zfill(2)
        var_final[0].to_netcdf(outfile+years+months+days+'_'+str(deltax)+'_'+str(deltay)+'.nc', format = 'NETCDF4', mode ='w', group=None)
    return print('work done')

