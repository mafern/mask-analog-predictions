"""
DATA PROCESSING CLASS
"""
import numpy as np
import glob
import os
from copy import deepcopy
import xarray as xr
xr.set_options(keep_attrs=True)
import xesmf as xe
# import cfgrib
import warnings
warnings.filterwarnings("ignore")
import regionmask
import json


def route_by_filetype(file_type, file):
    # call appropriate loading function: xarray for .nc, .hdf, .grib
    # only support for netcdf files for now
    if file_type == ".nc":
        dataset = load_nc(file)
    # elif file_type == ".hdf5":
        # load
        # convert to xarray format
    return dataset.squeeze()

def load_nc(file):
    return xr.load_dataset(file)


def combine_ncs(directory, nfiles, delete_old=False):
    """
    combine multiple subfiles into a single nc spanning the full time range.
    assumes that the filenames allow for sorting in the right order, and that
    they are split amongst the same number of subfiles.
    name will be based on the first and last of these subfiles.
    """
    all_files = np.sort(glob.glob(os.path.join(directory, '*')))
    indrange = np.linspace((0, nfiles), (allf.size-nfiles, all_files.size), int(all_files.size/nfiles), dtype=int)
    for inds in indrange:
        for ii in range(inds[0], inds[1]):
            # load each and concat
            ds = xr.load_dataset(all_files[ii])
            if ii == inds[0]:
                ds_complete = ds
                # file save name start
                fname_start = all_files[ii][:-9]
            else:
                ds_complete = xr.concat([ds_complete, ds], dim="time")

            # optionally delete old files
            if delete_old:
                os.remove(all_files[ii])
        # file save name ending
        fname_end = all_files[ii][-9:]
        # save as .nc
        ds_complete.to_netcdf(path=fname_start+fname_end)
        
        
def make_model_masks(file):
    """
    using a land-only variable, make a mask for the given model.
    will make both a land mask and an ocean mask and save these as .nc files.
    
    inputs:
        - file: the path and name of a file for a model and variable that is either
        land-only or ocean-only.
    """
    # load land-only variable
    mask_var = xr.load_dataarray(file).squeeze()
    # get a land mask and ocean mask
    landmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, np.nan, 1.0)
    oceanmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, 1.0, np.nan)
    
    # save in the appropriate directory, with the name landmask.nc and oceanmask.nc
    land_xr = xr.Dataset(
        data_vars=dict(mask=(["lat", "lon"], landmask.values)),
        coords=dict(
            lat=(["lat"], landmask.lat.values),
            lon=(["lon"], landmask.lon.values))
    )
    land_xr.to_netcdf(os.path.join(*file.split('/')[:-2], 'landmask.nc'))
    
    ocean_xr = xr.Dataset(
        data_vars=dict(mask=(["lat", "lon"], oceanmask.values)),
        coords=dict(
            lat=(["lat"], oceanmask.lat.values),
            lon=(["lon"], oceanmask.lon.values))
    )
    ocean_xr.to_netcdf(os.path.join(*file.split('/')[:-2], 'oceanmask.nc'))


class DataProcessing():
    """
    description
    
    inputs:
        - 
    outputs:
        - 
    """
    def __init__(self, model_list, variable_list, period, processing_list=None, savefile=None, n_ensembles=None, seed=123):
        ""
        # check if pre-saved processed data exists already (will need savefile to have metadata)
        
        # save some global variables (data mean, std, file names, etc.) for variables and models
        self.period = period[0].upper()
        self.period_long = period
        self.period_dictionary = {"D": 0, "M": 1, "Y": 2}
        self.month_dictionary = {1: "jan", 2: "feb", 3: "mar",
                                 4: "apr", 5: "may", 6: "jun",
                                 7: "jul", 8: "aug", 9: "sep",
                                 10: "oct", 11: "nov", 12: "dec"}
        self.global_map = True
        self.user_regions = json.load(open('regions.json', 'r'))
        self.regionmask_regions = ['regionmask.defined_regions.ar6.all[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.countries_10[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.us_states_10[box_or_regions]',
                              'regionmask.defined_regions.srex[box_or_regions]'
                             ]
        
        # if not, call load_data
        self.data_dictionary = self.load_data(model_list=model_list, variable_list=variable_list, period=period, n_ensembles=n_ensembles, seed=seed)

        # if not None, go through the steps in the processing list (order matters!)

        # save processed data in xx format, if savefile is given


    def load_data(self, model_list, variable_list, period, n_ensembles=None, seed=123):
        """
        load files for models in model_list for the variables in variable_list
        data needs to be stored with the directory structure: data/variable_name/model_name/files
        returns the data in a dictionary with xarray formatted entries
        
        inputs:
            - model_list: a list of model names, e.g. ['CESM2', 'MPI',]
            - variable_list: a list variable names, e.g. ['ts', 'pr',]
            - period: the sampling period of the model; 'monthly' or 'yearly' or 'daily'
            should agree with the folder structure within the data directory.
            - n_ensembles: number of ensemble members from each model to keep.
            - seed: random seed for selecting the ensemble members.
        outputs:
            - data_dictionary: a dictionary where the first layer is the variables, the second
            layer is the model, each of which is an xarray object containing all ensemble members.
        """
        data_dictionary = {}

        # save the minimum lat, lon range and common time / coarsest resolution for each variable
        self.attr_dictionary = {}

        # random seed for getting ensemble members
        rng_seed = np.random.default_rng(seed)

        print('loading variables:', *variable_list)
        print('from models:', *model_list)
        for variable in variable_list:
            # initialize dictionaries
            data_dictionary[variable] = {}
            self.attr_dictionary[variable] = {
                "time": ["0000-01-01", "3000-01-01"], 
                "res": [0, 0]
            }
            
            for model in model_list:
                # file path for this variable and model
                file_path = os.path.join('data', period, model, variable)
                # check that path exists, if not print warning and continue
                if not os.path.exists(file_path):
                    print('directory:', file_path, 'does not exist')
                    print('continuing without this dataset')
                    continue
                # get all files of this model and variable
                file_list = np.sort(glob.glob(os.path.join(file_path, '*')))

                # NEED TO HANDLE WHEN FILE IS SPLIT BETWEEN MULTIPLE SUB-FILES
                # could insist this is already done and write function for doing it (input path and time window, concat until time window is met)
                
                # randomly select n_ensembles members from the full list if enough available
                if n_ensembles is not None:
                    assert np.size(file_list) >= n_ensembles, ["more members requested than available. requested: "
                                                               + str(n_ensembles) + ", available: " + str(np.size(file_list))]
                    rnd_inds = rng_seed.choice(np.arange(len(file_list)), n_ensembles, replace=False, shuffle=False)
                    file_list = file_list[rnd_inds]

                for ii, file in enumerate(file_list):
                    # parse the file type
                    file_type = os.path.splitext(file)[1]
                    # load based on file type -- will be returned in xarray format
                    ens_member = route_by_filetype(file_type, file)
                    # create ensemble_list or add to ensemble_list
                    if ii == 0:
                        ensemble_list = ens_member
                    else:
                        ensemble_list = xr.concat([ensemble_list, ens_member], dim="member")

                    # update the attr_dictionary
                    # common time period between models: latest start time, earliest end time
                    self.attr_dictionary[variable]["time"] = [
                        np.max([ens_member.time.min().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][0]]), 
                        np.min([ens_member.time.max().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][1]])
                    ]
                    # latitude resolution, then longitude resolution
                    # will need to add altitude resolution when playing with 3D variables
                    self.attr_dictionary[variable]["res"] = [
                        np.max([np.diff(ens_member.lat[:2])[0], self.attr_dictionary[variable]["res"][0]]), 
                        np.max([np.diff(ens_member.lon[:2])[0], self.attr_dictionary[variable]["res"][1]])
                    ]

                # add model to the dictionary
                # resample step is to align data types and should not change anything meaningful
                ensemble_list = ensemble_list.resample(time=self.period).mean()
                data_dictionary[variable][model] = ensemble_list.convert_calendar("noleap")
                print('successfully loaded', ensemble_list['member'].size, 'ensemble members from', file_path)

        return data_dictionary
    
    
    # -------------------------------------------------------------------------------
    # --------------------------- temporal functions --------------------------------
    # -------------------------------------------------------------------------------
    # ---- select_season, smooth_time, rolling_time, sync_time, shift_inout_time ----
    # ENFORCE NO SMOOTHING AFTER SYNCING (CUTTING THE EDGES WILL LEAD TO SMOOTHING PROBLEMS)
    
    
    def select_season(self, data_dictionary, season):
        """
        select a yearly season from the data.
        data must be monthly.
        
        inputs:
            - season: a list with entries [season_length, season_start], with season_length the number
            of months to include (int > 0), and season_start the first month of the season (0 < int < 13)
        outputs:
            - data_dictionary with requested seasonal time steps
        """
        assert season[0] > 0, "season length must be a non-zero positive integer"
        assert season[1] > 0 and season[1] < 13, "season start must be an integer in the range [1, 12]"
        print('selecting season starting in', self.month_dictionary[season[1]], 'and running for', season[0], 'months')
        # for each variable in the dataset
        for var in data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in data_dictionary[var].keys():
                # get rolling seasons of given length
                data_dictionary[var][model] = data_dictionary[var][model].rolling(time=season[0]).mean().dropna("time", how='all')
                # then select only the seasons requested by season_start
                data_dictionary[var][model] = data_dictionary[var][model].sel(time=data_dictionary[var][model].time.dt.month == season[1])
            
        self.period = "Y"
        print('season selected, now', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    
    def smooth_time(self, data_dictionary, period=None):
        """
        smooth time by up-sampling, down-sampling, or resampling on same period, for all data.
        
        inputs:
            - period: a string with the desired time step size, e.g. "2M" for 2 months
            (valid options are "Y", "M", "D").
            if none given, retain current time step.
        outputs:
            - smoothed data_dictionary containing all data sets.
        """
        # determine if we are resampling, down-sampling, or up-sampling
        if period is None:
            sampling = "resampling"
        else:
            # upper-case and append a leading 1 if no leading integer
            if len(period) == 1:
                period = str(1)+period.upper()
            if len(self.period) == 1:
                self.period = str(1)+self.period
                
            # period type: D, M, or Y
            current_period_type = self.period_dictionary[self.period[-1]]
            requested_period_type = self.period_dictionary[period[-1]]
            
            # several cases -- make sure to get the correct one
            # if current period type is less (e.g. M -> Y), down-sampling
            if requested_period_type > current_period_type:
                sampling = "down-sampling"
            # if current period type is greater (e.g. from Y -> M), then up-sampling
            elif requested_period_type < current_period_type:
                sampling = "up-sampling"
            # finally, if same period type, use leading integer to determine up- or down-sampling
            else:
                # requested period is greater than current period
                if int(self.period[:-1]) < int(period[:-1]):
                    sampling = "down-sampling"
                # requested period is less than current period
                elif int(self.period[:-1]) > int(period[:-1]):
                    sampling = "up-sampling"
                # requested period is the same as current period
                else:
                    sampling = "resampling"
                  
        if sampling == "down-sampling":
            print(sampling, "data set from", self.period, "to", period)
            # for each variable in the dataset
            for var in data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=period).mean()
        elif sampling == "up-sampling":
            print(sampling, "data set from", self.period, "to", period)
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=period).interpolate(kind='linear')
                    # OTHER INTERPOLATION OPTIONS: "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
        else: # just resampling on the same period
            print(sampling, "data set")
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=self.period).mean()
            
        # update self.period to new period
        self.period = period
        print('time period is now', self.period, 'with', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    
    def rolling_time(self, data_dictionary, roll_period=1):
        """
        recast data as rolling averages over roll_period of the current time steps.
        
        inputs:
            - roll_period: number of current time steps to average over.
            can be positive or negative.
        outputs:
            - data_dictionary with rolling averaged time steps
        """
        assert roll_period != 0, "rolling average cannot be taken over 0 periods, use positive or negative integer"
        print('applying rolling average over', roll_period, self.period, 'periods')
        # for each variable in the dataset
        for var in data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in data_dictionary[var].keys():
                # rolling average over roll_period for the current time period of the data
                data_dictionary[var][model] = data_dictionary[var][model].rolling(time=abs(roll_period), center=False).mean().dropna("time", how="all")
                if roll_period < 0:
                    # shift time coordinates
                    new_times = xr.CFTimeIndex(data_dictionary[var][model].time.values).shift(n=roll_period+1, freq=self.period)
                    data_dictionary[var][model] = data_dictionary[var][model].assign_coords(time=new_times)

        print('rolling average applied, now', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    
    def sync_time(self, data_dictionary, time_range=None):
        """
        sync time period and dates between all data.
        
        inputs:
            - time_range: a list, [start date, end date] in year-month-day format as strings,
            e.g. ["1850-01-01", "2005-12-01"].
            if none given, defaults to the common period between all models.
        outputs:
            - synced data_dictionary containing all data sets.
        """
        print('syncing time period and dates between all variables and models')
        # for each variable in the dataset
        for var in data_dictionary.keys():
            # if no time range given, use attr_dictionary (common time period between models)
            if time_range is None:
                time_range = [self.attr_dictionary[var]['time'][0].item(), self.attr_dictionary[var]['time'][1].item()]
            # for each model (get down to the xarray level from the dictionary)
            for model in data_dictionary[var].keys():
                # align time periods (start, end)
                data_dictionary[var][model] = data_dictionary[var][model].sel(time=slice(time_range[0], time_range[1]))
            # align dates (only keep common dates)
            temp_data = xr.align(*[data_dictionary[var][model] for model in data_dictionary[var].keys()], exclude=["member", "lat", "lon"]) 
            # reassemble dictionary
            for ii, model in enumerate(data_dictionary[var].keys()):
                data_dictionary[var][model] = temp_data[ii]
                
        print('time period and dates synced, now using', data_dictionary[var][model].time.size, 'dates')
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    
    def shift_inout_time(self, data_dictionary, shift):
        """
        shift the data sets, i.e. create a leadtime by removing entries from the start or end.
        
        inputs:
            - shift: the number of entries to remove.
            negative values remove shift entries from the end (adding a lag),
            positive values remove shift entries from the start (adding a lead).
        outputs:
            - shifted data_dictionary.
        """
        assert shift != 0, "shift should be a positive or negative integer, not zero"
        print('shifting the data by', shift, 'time steps')
        # for each variable in the dataset
        for var in data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in data_dictionary[var].keys():
                    # drop the first or last shift entries in the data sets
                    if shift > 0:
                        data_dictionary[var][model] = data_dictionary[var][model].drop_isel(time=np.arange(shift))
                    else:
                        data_dictionary[var][model] = data_dictionary[var][model].drop_isel(time=-(np.arange(abs(shift))+1))
        
        print('data shifted, now using', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
        
    # -------------------------------------------------------------------------------
    # ---------------------------- spatial functions --------------------------------
    # -------------------------------------------------------------------------------
    # --------------------- smooth_space, mask_data, get_region ---------------------
    # GOOD: MASK, RESAMPLE, REGION OR RESAMPLE, MASK, REGION - NEVER REGION, RESAMPLE

                
    def smooth_space(self, data_dictionary, resolutions=None):
        """
        smooth resolution between all data; user selected or (default) coarsest data.
        should always be done on the global data before extracting a region.
        
        inputs:
            - resolutions: list of resolution in lat/lon order, e.g. [1.25, 1.75].
            if none given, defaults to the coarsest resolution amongst all models.
        outputs:
            - data_dictionary data sets mapped to the same lat/lon resolution, without altering the lat/lon range.
        """
        # for each variable in the dataset
        for var in data_dictionary.keys():
            # if no time range given, use attr_dictionary (common time period between models)
            if resolutions is None:
                resolutions = self.attr_dictionary[var]['res']
            print('regridding data sets to resolution', resolutions)
            # for each model (get down to the xarray level from the dictionary)
            for model in data_dictionary[var].keys():
                # assert that the map is global -- SHOULD CHANGE THIS TO GET RANGE FROM DATA?
                assert self.global_map, "up- or down-sampling should be done before extracting a region"
                
                # construct the regridder using xESMF: https://xesmf.readthedocs.io/en/latest/index.html
                latlon_dict = {
                    "lat": np.arange(-90+0.5*resolutions[0], 90, resolutions[0]),
                    "lon": np.arange(0, 360, resolutions[1])
                }
                regridder = xe.Regridder(data_dictionary[var][model], xr.Dataset(latlon_dict), "bilinear", periodic=True)
                # OTHER OPTIONS EXIST OTHER THAN BILINEAR: CONSERVATIVE, PATCH (SLOW), . . . 
                # do the regridding
                data_dictionary[var][model] = regridder(data_dictionary[var][model], keep_attrs=True)        
        
        print('data sets regridded')
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    
    def mask_data(self, data_dictionary, mask):
        """
        mask the oceans or the land.
        if available, uses model specific mask, else use natural earth 1:10m map
        
        inputs:
            - mask: a string for what to mask, either 'ocean' or 'land'
        outputs:
            - data_dictionary with either land or ocean masked from all data sets.
        """
        assert np.isin(mask, ['ocean', 'land']), "input mask should be either 'land' or 'ocean'"
        print('masking the', mask, 'from all data sets')
        
        # get the backup map from natural earth
        mask_map = regionmask.defined_regions.natural_earth_v5_0_0.land_10
        
        # for each variable in the dataset
        for var in data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in data_dictionary[var].keys():
                # try loading model specific mask first
                try:
                    # load map for this model
                    mask_map = xr.load_dataset(os.path.join('data', self.period_long, model, mask+'mask.nc'))
                    # adjust the resolution of the mask to be the same as the model and variable
                    regridder = xe.Regridder(mask_map, data_dictionary[var][model], "bilinear", periodic=True)
                    mask_map = regridder(mask_map, keep_attrs=True)
                    final_mask = mask_map.mask
                    print('masking', model, 'model using model specific mask')
                # no model specific mask, use natural earth
                except:
                    # divide into either land or ocean mask
                    final_mask = mask_map.mask(data_dictionary[var][model]) + 1
                    if mask == 'land':
                        final_mask = final_mask.fillna(0).where(final_mask.fillna(0) == 0) + 1
                    print('masking', model, 'using natural earth map')
                
                # mask the data sets
                data_dictionary[var][model] = data_dictionary[var][model] * final_mask
        
        print('data sets masked')
        return data_dictionary

    # -------------------------------------------------------------------------------

    def get_region(self, data_dictionary, box_or_regions):
        """
        return data for the specified region only.
        can be a list of numeric boundaries, or region names.
        
        inputs:
            - box_or_regions: list of boundary coordinates, i.e. [minlon, minlat, maxlon, maxlat],
            or a region name (or list of region names), e.g. ['NWN', 'E.North-America', 'NCA']
        outputs:
            - data_dictionary with data sets for only the specified regions.
        """
        # regionmask package is used when region names are passed and not in regions.json
        # see https://regionmask.readthedocs.io/en/stable/
        print('extracting region(s):', box_or_regions)
        assert type(box_or_regions[0]) is str or type(box_or_regions) is list, "input should be a list of bounds or name of region"

        # if the input is a string, search for that named region or regions
        if type(box_or_regions[0]) is str:

            # if a single region, make into a list (for loop)
            if not isinstance(box_or_regions, list):
                    box_or_regions = [box_or_regions,]

            # find which source to get from: regions.json, then going through the regionmask options
            # try the regions.json file first (user defined regions)
            try:
                for name in box_or_regions:
                    # get latitude and longitude range for named region
                    lat, lon = self.user_regions[name].values()
                    print(name, 'found in regions.json')
                # for each variable in the dataset
                for var in data_dictionary.keys():
                    # for each model (get down to the xarray level from the dictionary)
                    for model in data_dictionary[var].keys():
                        # make latitude and longitude masks
                        mask_lon = (data_dictionary[var][model].lon >= lon[0]) & (data_dictionary[var][model].lon <= lon[1])
                        mask_lat = (data_dictionary[var][model].lat >= lat[0]) & (data_dictionary[var][model].lat <= lat[1])
                        # apply masks and drop coordinates, etc. outside of region
                        data_dictionary[var][model] = data_dictionary[var][model].where(mask_lon & mask_lat, drop=True)
            # if not in regions.json, try regionmask defined regions
            except:
                for region_set in self.regionmask_regions:
                    try:
                        eval(region_set)
                        print(box_or_regions, 'found in', region_set[:-16])
                        region = eval(region_set)
                    except:
                        pass

                # get the associated numbers for the regions
                region_num = region.numbers

                for var in data_dictionary.keys():
                    for model in data_dictionary[var].keys():
                        # make geopandas object for masking (and get correct resolution of mask)
                        region_mask = regionmask.mask_geopandas(region.to_geodataframe(), data_dictionary[var][model]) 
                        # make the regions 0 or nan using region_num associated with requested regions
                        region_mask = region_mask.where(np.isin(region_mask, region_num))
                        region_mask = region_mask.where(region_mask.isnull(), 1)
                        # apply mask to data sets
                        data_dictionary[var][model] = data_dictionary[var][model] * region_mask
                        # drop coordinates, etc. outside of region
                        data_dictionary[var][model] = data_dictionary[var][model].where(abs(data_dictionary[var][model]) >= 0, drop=True)

        # if input is a list, use these as boundaries to define region
        else:
            assert len(box_or_regions) == 4, "box should have 4 entries, [minlon, minlat, maxlon, maxlat]"
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    # make latitude and longitude masks
                    mask_lon = (data_dictionary[var][model].lon >= box_or_regions[0]) & (data_dictionary[var][model].lon <= box_or_regions[2])
                    mask_lat = (data_dictionary[var][model].lat >= box_or_regions[1]) & (data_dictionary[var][model].lat <= box_or_regions[3])
                    # apply masks and drop coordinates, etc. outside of region
                    data_dictionary[var][model] = data_dictionary[var][model].where(mask_lon & mask_lat, drop=True)
        
        # set global_map switch to false (for up- or down-scaling resolution)
        self.global_map = False
        print('region extracted')
        return data_dictionary
    
    
    # -------------------------------------------------------------------------------
    # ---------------------------- data manipulation --------------------------------
    # -------------------------------------------------------------------------------
    # --------- remove_seasonality, remove_trend, norm_data, regional_mean ----------


    def remove_seasonality(self, data_dictionary, ):
        """
        remove the seasonal trend from the data, by ensemble, model, or entire dataset.
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # subtract monthly average across full time period for each grid cell
        
        
        print('')
        return data_dictionary

    # -------------------------------------------------------------------------------

    def remove_trend(self, data_dictionary, ):
        """
        remove the long term trend from the data, either by model or entire dataset.
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # subtract ensemble mean for each grid cell
        # may need to add fitting for cases with few ensemble members
        # could add option to remove via LIM least damped mode
        
        # could test auto-correlation coefficient to determine if a fit is needed (not enough ensemble members)
        
        
        print('')
        return data_dictionary

    # -------------------------------------------------------------------------------

    def norm_data(self, data_dictionary, ):
        """
        
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # normalize data (a few ways -- minus mean, divide by std, etc.)
        
        
        print('')
        return data_dictionary

    # -------------------------------------------------------------------------------

    def regional_mean(self, data_dictionary, ):
        """
        
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # calculate regional mean
        
        
        print('')
        return data_dictionary
    

    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    
    @property
    def data_operations_list(self):
        "print a list of the data processing operations available in the DataProcessing class"
        function_list = []
        for function in self.__dir__(): # for each entry in the directory of this class
            if not np.isin(function, ['data_operations_list', 'list_regions']): # ignore this function (else recursion issue)
                # check that it is a function (callable) and not built-in (__)
                if callable(getattr(self, function)) and not function.startswith("__"):
                    function_list.append(function)
        print(function_list)
    
    @property
    def list_regions(self):
        "list the available regions in the user defined json file and in the regionmask package"
        print('user defined regions:\n\n    ', list(self.user_regions.keys()), '\n\n')
        for regionmask_region in self.regionmask_regions:
            print(regionmask_region[:-16]+':\n\n    ', eval(regionmask_region[:-16]+'.names'), '\n\n')

    
    
    
    
    
    
    
    
    
    
    
    
#     def sync_space(self, data_dictionary, lat_range, lon_range):
#         """
#         sync coordinate range between all data; user selected or (default) narrowest range data.
#         inputs:
#             - lat_range: list of minimum and maximum latitudes to retain, e.g. [-80, 60].
#             - lon_range: same as lat_range, but for longitude, e.g. [25, 300].
#         outputs:
#             - data_dictionary data sets mapped to the same lat/lon range, without altering the resolution.
#         """
#         print('regridding data sets to: lat', lat_range, 'and lon', lon_range)
#         # for each variable in the dataset
#         for var in data_dictionary.keys():
#             # for each model (get down to the xarray level from the dictionary)
#             for model in data_dictionary[var].keys():
#                 # get the current resolutions as mean across cells
#                 lat_res = np.diff(data_dictionary[var][model].lat).mean()
#                 lon_res = np.diff(data_dictionary[var][model].lon).mean()
#                 # construct the regridder using xESMF: https://xesmf.readthedocs.io/en/latest/index.html
#                 latlon_dict = {
#                     "lat": np.arange(lat_range[0], lat_range[1], lat_res),
#                     "lon": np.arange(lon_range[0], lon_range[1], lon_res)
#                 }
#                 regridder = xe.Regridder(data_dictionary[var][model], xr.Dataset(latlon_dict), "bilinear", periodic=True)
#                 # OTHER OPTIONS EXIST OTHER THAN BILINEAR: CONSERVATIVE, PATCH (SLOW), . . . 
#                 # do the regridding
#                 data_dictionary[var][model] = regridder(data_dictionary[var][model], keep_attrs=True) 
        
#         print('data sets regridded')
#         return data_dictionary
    
    # -------------------------------------------------------------------------------
    
#     # version using a default map
#     def mask_data1(self, data_dictionary, mask):
#         """
#         mask the oceans or the land -- seperate from specifying region
#         inputs:
#             - mask: a string for what to mask, either 'ocean' or 'land'
#         outputs:
#             - 
#         """
#         assert np.isin(mask, ['ocean', 'land']), "input mask should be either 'land' or 'ocean'"
#         print('masking the', mask, 'from all data sets')
        
#         # get map and adjust coordinates to 0-360 for longitude
#         mask_map = xr.load_dataset('landseamask.nc').roll({"lon": 360})
#         mask_map.coords['lon'] = (mask_map.coords['lon'] + 180)
#         # divide into either land or ocean mask
#         if mask == 'land':
#             mask = deepcopy(mask_map).where(mask_map.mask == 0) + 1
#         else:
#             mask = deepcopy(mask_map).where(mask_map.mask != 0)
        
#         # for each variable in the dataset
#         for var in data_dictionary.keys():
#             # MAY WANT TO TRY LOADING MODEL SPECIFIC MASK, MOVE TO OTHER OPTION IF UNABLE
#             # this would mean keeping the masks for each model somewhere (and calculating them?)
            
#             # for each model (get down to the xarray level from the dictionary)
#             for model in data_dictionary[var].keys():
#                 # adjust the resolution of the mask to be the same as the variables
#                 regridder = xe.Regridder(mask, data_dictionary[var][model], "bilinear", periodic=True)
#                 mask = regridder(mask, keep_attrs=True)
#                 # mask the data sets
#                 data_dictionary[var][model] = data_dictionary[var][model] * mask.mask
        
#         print('data sets masked')
#         return data_dictionary

# def make_model_mask(filename, mask):
#     """
#     using a land-only or ocean-only variable, make a mask for the given model.
#     will make both a land mask and an ocean mask and save these as .nc files.
#     inputs:
#         - filename: the path and name of a file for a model and variable that is either
#         land-only or ocean-only.
#         - mask: specify if the file contains a 'land' or 'ocean' variable.
#     """
#     assert np.isin(mask, ['ocean', 'land']), "input for mask should be either 'land' or 'ocean'"
#     # load land-only or ocean-only variable
#     mask_var = xr.load_dataset(filename).squeeze()
    
#     # get a land mask from the variable, regardless of variable
#     if mask == 'land':
#         landmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, np.nan, 1.0)
#     else:
#         landmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, 1.0, np.nan)
    
#     # make land mask: replace ocean values with ones, land values with nan
#     # make ocean mask: replace land values with ones, ocean values with nan
    
#     # show preview of mask, to confirm it has worked correctly
    
#     # save in the appropriate directory, with the name landmask.nc and oceanmask.nc