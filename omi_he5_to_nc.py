#!/usr/local/other/python/GEOSpyD/2019.03_py3.7/2019-04-22/bin/python
'''
Routine to read data from OMNO2_003 level 2 he5 files and store them in netCDF files in 
such a form that they can be used by the GSI DAS code.
The variables to be read are defined in a YAML configuration file, which needs to list the
input file template as well as all of the data groups (and variables therein) to be read: 

Example YAML configuration file:
"""
file_template: '/discover/nobackup/projects/gmao/geos_cf_dev/obs/OMNO2_003/he5/%Y/m%m/OMI-Aura_L2-OMNO2_%Ym%m%d*.he5'
data:
  geo:
    group_name: 'HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields'
    vars:
      - 'Latitude'
      - 'Longitude'
      - 'SolarZenithAngle'
  dat:
    group_name: 'HDFEOS/SWATHS/ColumnAmountNO2/Data Fields'
    vars:
      - 'AmfStrat'
"""

EXAMPLE:
python omno2_he5_to_nc.py -y 2020 -m 1 -d 1 -o 'test.nc'

HISTORY: 
20210406 - christoph.a.keller@nasa.gov - initial version
'''

import logging
import argparse
import numpy as np
import sys
import os
import glob 
import datetime as dt
import pandas as pd
import xarray as xr
import yaml

REFTIME = dt.datetime(1993,1,1)
NLEV = 35
SUPPORTED_RETRIEVAL_TYPES = ['OMNO2_003','MINDS']

def main(args):
    '''
    Main routine to write OMNO2_003 he5 data to netcdf. Read the configuration
    file and invoke the driver routine n times for the selected day, each time
    aggregating all available OMI data for a selected time window. The length 
    of the time window is set in the input arguments. 
    '''
    log = logging.getLogger(__name__)
    config = _read_config(args.configfile)
    idate = dt.datetime(args.year,args.month,args.day)
    for h in np.arange(0,24,args.hour_window): 
        _he5_to_nc(args,config,idate,h,args.hour_window)
    return


def _he5_to_nc(args,config,idate,hour,hour_window):
    '''
    Read OMI NO2 he5 files for a given day and aggregate all data that falls 
    within the selected time window and store it in a newly created netCDF file.
    '''
    log = logging.getLogger(__name__)
    # get retrieval type
    rtype = config.get('retrieval_type','OMNO2_003')
    if rtype not in SUPPORTED_RETRIEVAL_TYPES:
        log.warning('unsupported retrieval type: {}'.format(rtype))
        assert(False)
    # set central time stamp and time window
    odate = dt.datetime(idate.year,idate.month,idate.day,hour)
    log.info(odate.strftime('Working on %Y-%m-%d %Hz:'))
    dh = hour_window / 2
    t1 = idate + dt.timedelta(hours=hour-dh)
    t2 = idate + dt.timedelta(hours=hour+dh)
    tai_t1 = (t1-REFTIME).total_seconds()
    tai_t2 = (t2-REFTIME).total_seconds()
    # get all input files
    file_template = config.get('file_template','unknown')
    itemplate = odate.strftime(file_template) 
    ifiles = glob.glob(itemplate)
    if len(ifiles)==0:
        log.warning(idate.strftime('No data found for %Y-%m-%d: {}'.format(itemplate)))
        return
    # also need files from previous day if day1 < day2
    if t1.day != t2.day:
        t1template = t1.strftime(file_template)
        prevfiles = glob.glob(t1template)
        if len(prevfiles)==0:
            log.warning(idate.strftime('No data found for previous day (%Y-%m-%d): {}'.format(t1template)))
        else:
            ifiles = ifiles + prevfiles
#---Read all files and write flattened data arrays to list. Will be merged into one single array at the end
    nrec = 0
    time1d = []
    row1d  = []
    data_dict = {}
    for f in sorted(ifiles):
        # read geo data first to extract time and row data, plus any additional variables as listed in the configuration file.
        # The time and row dimension (dims 1 and 2) are collapsed to one row, using Fortran order.
        config_geo = config.get('data').get('geo')
        group_name = config_geo.get('group_name','HDFEOS/SWATHS/ColumnAmountNO2/Geolocation Fields')
        geo = xr.open_dataset(f,group=group_name)
        # check time units
        time_long_name = 'unknown'
        if rtype=='OMNO2_003':
            time_long_name = geo.Time.Title
        if rtype=='MINDS':
            time_long_name = geo.Time.description
        assert('TAI93' in time_long_name)
        # get timestamps
        if rtype=='MINDS':
            time_values = pd.to_datetime(geo.Time.values)
            mint = t1            
            maxt = t2
        if rtype=='OMNO2_003':
            time_values = geo.Time.values
            mint = tai_t1
            maxt = tai_t2
        # go to next file if all observations outside of desired time window
        if (time_values.min()>maxt) or (time_values.max()<mint):
            #log.info('Skip because all data outside of time range')
            continue
        log.info(' -- Working on group "{}" in file {}'.format(group_name,f.split('/')[-1]))
        # get index of time stamps to be used 
        idx = (time_values>=mint)&(time_values<maxt)
        ntimes = np.sum(idx)
        rowName = 'unknown'
        if rtype=='OMNO2_003':
            rowName = 'phony_dim_5'
        if rtype=='MINDS':
            rowName = 'nXtrack'
        nrows = geo.dims[rowName]
        # create time and row data arrays and append to corresponding lists
        #time1d.append(np.tile(geo['Time'].values[idx],nrows))
        time1d.append(np.tile(time_values[idx],nrows))
        row1d.append((np.arange(nrows)+1).repeat(ntimes))
        # get all other geodata. 
        group_vars = config_geo.get('vars')
        for v in group_vars: 
            data_dict = _read_data(args,geo,v,data_dict,idx,ntimes,nrows)
        geo.close()
        # read all other data groups
        for igroup in config.get('data'):
            # can skip group 'geo' as this was already done above
            if igroup=='geo':
                continue
            config_group = config.get('data').get(igroup)
            group_name = config_group.get('group_name','unknown')
            log.info(' -- Working on group "{}" in file {}'.format(group_name,f.split('/')[-1]))
            ds = xr.open_dataset(f,group=group_name)
            # read all data as above 
            group_vars = config_group.get('vars')
            for v in group_vars: 
                data_dict = _read_data(args,ds,v,data_dict,idx,ntimes,nrows)
            ds.close()
        # count number of records
        nrec += ntimes*nrows
#---Write all data to netCDF file
    data_vars = {}
    # Create entries for row and time stamps
    timvec = np.concatenate(tuple(time1d),axis=0)
    if rtype=='OMNO2_003':
        times = [REFTIME+dt.timedelta(seconds=i) for i in timvec]
    if rtype=='MINDS':
        times = pd.to_datetime(timvec)
        #times = [i for i in timvec]
    data_vars['Year'] = (["nrec"], np.array([np.float(i.year) for i in times]), {"long_name":"Year at start of scan","unit":"1"})
    data_vars['Month'] = (["nrec"], np.array([np.float(i.month) for i in times]), {"long_name":"Month at start of scan","unit":"1"})
    data_vars['Day'] = (["nrec"], np.array([np.float(i.day) for i in times]), {"long_name":"Day at start of scan","unit":"1"})
    data_vars['Hour'] = (["nrec"], np.array([np.float(i.hour) for i in times]), {"long_name":"Hour at start of scan","unit":"1"})
    data_vars['Minute'] = (["nrec"], np.array([np.float(i.minute) for i in times]), {"long_name":"Minute at start of scan","unit":"1"})
    data_vars['Row'] = (["nrec"], np.concatenate(tuple(row1d),axis=0).astype(np.float), {"long_name":"Field_of_view_(row)","unit":"1"})
    # All other entries 
    for v in data_dict:
        data_vars[v] = _create_var(data_dict[v])
    # create general attributes 
    attrs = {"title":rtype,
             "observation_date":odate.strftime("%Y-%m-%d %Hz"),
             "time_window_in_hours":args.hour_window,
             "history":dt.datetime.today().strftime('Created on %Y-%m-%d %H:%M'),
             "author":"generated by script he5_to_nc.py",
             "contact":"christoph.a.keller@nasa.gov",
            }
    # create dataset and write to disk
    ds = xr.Dataset(data_vars=data_vars,attrs=attrs)
    ofile = odate.strftime(args.ofile)
    # check for subdir
    if '/' in ofile:
        path = '/'.join(ofile.split('/')[0:-1])
        os.makedirs(path,exist_ok=True)
    ds.to_netcdf(ofile)
    log.info(' -- {} entries written to file {}'.format(nrec,ofile))
    ds.close()
    return


def _read_config(configfile):
    '''Read configuration file'''
    log = logging.getLogger(__name__)
    if not os.path.isfile(configfile):
        log.error('Configuration file does not exist: {}'.format(configfile),exc_info=True)
        return None
    with open(configfile,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # make early error check: entry 'geo' is expected in configuration file
    if 'geo' not in config.get('data'):
        log.error("no data entry with name 'geo' found in configuration file {} - this is needed".format(configfile))
    return config


def _read_data(args,ds,varname,data_dict,idx,ntimes,nrows):
    '''
    Read data array from input dataset and store it in the local data dictionary.
    Data is reordered with the time and row axes collapsed into one, using Fortran ordering.
    The data arrays read from individual files are stored in a data list, and will be added
    together at the end when creating the data variable tuple.
    Returns the updated data dictionary.
    '''
    log = logging.getLogger(__name__)
    # make sure variable name actually exists in the data set
    if varname not in ds:
        log.warning('Variable not found: {}'.format(varname))
        assert(varname in ds)
    # get dimension of original array and create collapsed array based on dimension
    ndim = len(ds[varname].shape)
    # 3d: [times,rows,levels] -> [nrec,levels]
    if ndim==3:
        iarr = ds[varname].values[idx,:,:].reshape((ntimes*nrows,-1),order="F")
        dims = ["nrec","nlev"]
    # 2d: [times,rows] -> [nrec,]
    if ndim==2:
        iarr = ds[varname].values[idx,:].reshape(-1,order="F")
        dims = ["nrec"]
    # 1d data: if it is the time dimension, stack the same array ntimes times so that its 
    # dimension aligns with the 2d/3d arrays. If it's data on levels, simply use the array
    # as is as level data will be stored as single entry for the full dataset.
    if ndim==1:
        # 1d: [times,] -> nrow * [times,]
        if ds[varname].shape[0]==ntimes:
            iarr = np.tile(ds[varname].values[idx],nrows)
            dims = ["nrec"]
        # 1d: [levels,] -> [levels,]
        if ds[varname].shape[0]==NLEV:
            iarr = ds[varname].values[:]
            dims = ["nlev"]
    # if variable is not yet in data list dictionary, add item to it. Also store data units and long name
    if varname not in data_dict:
        ientry = {}
        ientry['data_list'] = [iarr]
        ientry['dims'] = dims 
        for key,value in ds[varname].attrs.items():
            ientry[key] = value
        #ientry['long_name'] = ds[varname].Title.replace(" ","_")
        #ientry['unit'] = ds[varname].Units
        data_dict[varname] = ientry.copy()
    # if entry already exists in dictionary, simply add array to data list of this entry
    else:
        data_dict[varname]['data_list'].append(iarr)
    return data_dict 


def _create_var(dict_entry):
    '''
    Convert an entry of the local data dictionary to a data variable tuple that can be handled by xr.Dataset.
    The local data dictionary contains the data read from multiple files, stored in a data list. This list is 
    here concatenated into a single array. In addition, the long name and array unit is also set (as stored in 
    the data dictionary.
    '''
    log = logging.getLogger(__name__)
    # if data is on levels only, restrict output array to first entry. This explicitly assumes that all of these  
    if dict_entry['dims']==["nlev"]:
        datalist = [dict_entry['data_list'][0]]
    else:
        if len(dict_entry['data_list'])>1:
            datalist = dict_entry['data_list'][:]
        else:
            datalist = [dict_entry['data_list'][:]]
    idict = {}
    for key,value in dict_entry.items():
        if key not in ['dims','data_list']:
            idict[key]=value
    #ivar = (dict_entry['dims'], np.concatenate(tuple(datalist),axis=0), {"long_name":dict_entry['long_name'], "unit":dict_entry['unit']})
    ivar = (dict_entry['dims'], np.concatenate(tuple(datalist),axis=0), idict)
    return ivar


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-c', '--configfile',type=str,help='configuration file',default='config/omno2_003_config.yaml')
    p.add_argument('-y', '--year',type=int,help='year',default=2020)
    p.add_argument('-m', '--month',type=int,help='month',default=1)
    p.add_argument('-d', '--day',type=int,help='day',default=1)
    p.add_argument('-o', '--ofile',type=str,help='output file template',default="nc/Y%Y/M%m/OMNO2v4.%Y%m%d.t%Hz.nc")
    p.add_argument('-w', '--hour_window',type=int,help='length of hour window',default=6)
    return p.parse_args()


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    main(parse_args())
