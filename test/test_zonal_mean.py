####importing packages
import sys
sys.path.append('/work/mh0731/m300876/package')
import icons
from pathlib import Path
import importlib
import numpy as np
import xarray as xr
import pandas as pd


def mod_dataset(dataset,time_range,factor):
    clat = gridset.clat.values*180/np.pi
    clon = gridset.clon.values*180/np.pi
    data = gridset.assign_coords(clon=("ncells",clon),clat=("ncells",clat))
    data = data.expand_dims({'time':time_range})*0+factor
    time = pd.date_range("2000-01-01", freq="1D", periods=time_range)
    data['time'] = time
    return data


def test_mean(dataset,time_range,lat,lon,factor,interval):
    data = mod_dataset(dataset,time_range,factor)
    zonal = icons.calculate_zonal(data,lat,lon,interval,mask=None)
    assert zonal.mean().values == factor, 'Error: test failed with factor:'+ str(factor)

if __name__ == '__main__':
    icons.prepare_cpu(memory='50GB')
    gridfile = '/work/mh0287/m300083/experiments/dpp0066/bc_land_frac.nc'
    gridset = xr.open_dataset(gridfile,chunks='auto').rename({'cell':'ncells'})['notsea']

    test_mean(gridset,10,[-10,10],[-50,-20],1,0.5)
    test_mean(gridset,10,[-10,10],[-50,-20],2,0.5)
    test_mean(gridset,10,[-10,10],[-50,-20],3,0.5)
