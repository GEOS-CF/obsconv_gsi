#!/usr/local/other/python/GEOSpyD/2019.03_py3.7/2019-04-22/bin/python
'''
Check for completeness and invalid (nan) geocoordinates in the obs
files needed for GSI. If a file is missing for the specified time
period, a dummy file with all missing values will be created (from 
a template file. If an existing file contains nan coordinates, this 
file is moved (same filename + with_nan.nc) and a copy of the old 
file is made that replaces all nan coordinates with missing values.

EXAMPLES:
./sanitize.py -i "/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMSO2/nc/Y%Y/M%m/OMSO2v2.%Y%m%d.t%Hz.nc" -t "/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMSO2/nc/Y2022/M01/OMSO2v2.20220101.t00z.nc" -s "2022-12-01" -e "2023-01-01"
./sanitize.py -i "/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMNO2/nc/Y%Y/M%m/OMNO2v2.%Y%m%d.t%Hz.nc" -t "/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMNO2/nc/Y2022/M01/OMNO2v4.20220101.t00z.nc" -s "2022-12-01" -e "2023-01-01"
'''

import numpy as np
import os
import glob
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import argparse


def main(args):
    '''
    Walk through provided time range and check for completeness of file record and if there are any nan-coordinates in the existing files.
    '''
    idate = dt.datetime.strptime(args.start_date,"%Y-%m-%d") 
    end   = dt.datetime.strptime(args.end_date,"%Y-%m-%d") 
    while idate < end:
        print(idate.strftime("Working on %Y-%m-%d %Hz"))
        ifile = idate.strftime(args.ifile)
        if not os.path.isfile(ifile):
            print("File not found - create dummy for {}".format(ifile))
            make_dummy_file(args,idate,ifile)
        else:
            check_nan(args,ifile)
        idate = idate + dt.timedelta(hours=6)
    return


def check_nan(args,ifile):
    '''
    Check for nan's in longitude coordinate of the provided file. If there are any, replace them with missing values.
    '''
    ds = xr.open_dataset(ifile)
    # Get nan mask
    isnan = np.isnan(ds["Longitude"].values)
    nrec  = len(isnan) 
    if np.sum(isnan)>0:
        print("Found {} nan's, will create copy with those cells set to missing values".format(np.sum(isnan)))
        # write output file
        dso = ds.copy()
        for v in dso:
            dso[v] = ds[v].copy()
            if len(dso[v].shape)==1 and dso[v].shape[0]==nrec:
                dso[v].values[isnan] = args.missval
        # move original file and write new file to original file location
        tmp = ifile.split('.')
        fname = tmp[0:-1] + ["with_nan"] + [tmp[-1]]
        ofile = '.'.join(fname)
        os.rename(ifile,ofile)
        print("Original file moved to {}".format(ofile))
        dso.to_netcdf(ifile)
        dso.close()
    ds.close()
    return    


def make_dummy_file(args,idate,ofile):
    '''
    Create a dummy file for the given date using a template file. All values except for the date will be set to missing values.
    '''
    ds = xr.open_dataset(args.empty_template)
    dso = ds.copy()
    for v in dso:
        dso[v] = ds[v].copy()
        val = args.missval
        if v=='Year':
            val = idate.year
        if v=='Month':
            val = idate.month
        if v=='Day':
            val = idate.day
        if v=='Hour':
            val = idate.hour
        dso[v].values[:] = val
    # write to file
    ofile = idate.strftime(ofile)
    dso.to_netcdf(ofile)
    dso.close()
    print("Written dummy file: {}".format(ofile))
    ds.close()
    return


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-s', '--start_date',type=str,help='start date (%Y-%m-%d)',default="2022-01-01")
    p.add_argument('-e', '--end_date',type=str,help='end date (%Y-%m-%d)',default="2022-01-01")
    p.add_argument('-i', '--ifile',type=str,help='input file',default="/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMSO2/nc/Y%Y/M%m/OMSO2v2.%Y%m%d.t%Hz.nc")
    p.add_argument('-t', '--empty_template',type=str,help='template file for empty file',default="/discover/nobackup/projects/gmao/r21cchem/data/codas/omi/OMSO2/nc/Y2022/M01/OMSO2v2.20220101.t00z.nc")
    p.add_argument('-m', '--missval',type=float,help='missing value',default=-999.9)
    return p.parse_args()

 
if __name__ == '__main__':
    main(parse_args())
#
