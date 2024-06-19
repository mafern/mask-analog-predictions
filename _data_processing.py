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
# import warnings
# warnings.filterwarnings("ignore")
import regionmask
import json

# TODO: incorporate trends (tendencies) in the inputs for both short (1 year) and long (5 or 10 years), essentially want d/dt

def route_by_filetype(file_type, file):
    # TODO: add support for other file types
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


class DataProcessing():
    """
    description
    
    inputs:
        - 
    outputs:
        - 
    """
    def __init__(self, model_list, variable_list, period, n_members, seed, model_member_dict=None):
        ""
        
        # save some global variables that will be useful
        self.period_dictionary = {"D": 0, "M": 1, "Y": 2}
        self.month_dictionary = {1: "jan", 2: "feb", 3: "mar",
                                 4: "apr", 5: "may", 6: "jun",
                                 7: "jul", 8: "aug", 9: "sep",
                                 10: "oct", 11: "nov", 12: "dec"}
        self.user_regions = json.load(open(os.path.join(os.path.dirname(__file__), 'regions.json'), 'r'))
        self.regionmask_regions = ['regionmask.defined_regions.ar6.all[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.countries_10[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.us_states_10[box_or_regions]',
                              'regionmask.defined_regions.srex[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.land_10[box_or_regions]'
                             ]
        # save some global variables based on the experiment
        self.period = period[0].upper()
        self.period_long = period
        self.global_map = True

        # save model_member_dict
        self.model_member_dict = deepcopy(model_member_dict)

        # load the data from the requested models and variables
        self.data_dictionary = self.load_data(model_list, variable_list, period, n_members, seed)
 

    def load_data(self, model_list, variable_list, period, n_members, seed):
        """
        load files for models in model_list for the variables in variable_list
        data needs to be stored with the directory structure: data/variable_name/model_name/files
        returns the data in a dictionary with xarray formatted entries
        
        inputs:
            - model_list: a list of model names, e.g. ['CESM2', 'MPI',]
            - variable_list: a list variable names, e.g. ['ts', 'pr',]
            - period: the sampling period of the model; 'monthly' or 'yearly' or 'daily'
            should agree with the folder structure within the data directory.
            - n_members: number of ensemble members to use.
            - seed: random seed for selecting ensemble members
        outputs:
            - data_dictionary: a dictionary where the first layer is the variables, the second
            layer is the model, each of which is an xarray object containing all ensemble members.
        """
        data_dictionary = {}
        self.attr_dictionary = {}

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
                if type(n_members) == int:
                    # check there are still enough members for this model when training
                    assert np.size(self.model_member_dict[model]) >= n_members, ["more members requested than available. requested: " + str(n_members) + ", available: " + str(np.size(self.model_member_dict[model]))]
                
                # file path for this variable and model
                file_path = os.path.join(os.path.dirname(__file__), 'raw_data', period, model, variable)
                # check that path exists, if not print warning and continue
                if not os.path.exists(file_path):
                    print('directory:', file_path, 'does not exist')
                    print('continuing without this dataset')
                    continue
                # get all files of this model and variable
                file_list = np.sort(glob.glob(os.path.join(file_path, '*')))

                if type(n_members) is list:
                    # members have already been chosen (e.g. best_analogs process_new_data)
                    used_members = n_members
                else:
                    # generate randomly seeded choice object and randomly select members
                    rng_seed = np.random.default_rng(seed)
                    used_members = np.sort(rng_seed.choice(self.model_member_dict[model], n_members, replace=False, shuffle=False))
                    # update model_member_dictionary to remove these members for later splits
                    self.model_member_dict[model] = np.setdiff1d(self.model_member_dict[model], used_members)
                # select those files
                file_list = file_list[used_members]

                for ii, file in enumerate(file_list):
                    # parse the file type
                    file_type = os.path.splitext(file)[1]
                    # load based on file type -- will be returned in xarray format
                    ens_member = route_by_filetype(file_type, file)
                    # create ensemble_list or add to ensemble_list
                    if ii == 0:
                        # add member dimension
                        ensemble_list = ens_member.expand_dims(dim={"member": 1}, axis=0)
                    else:
                        ensemble_list = xr.concat([ensemble_list, ens_member], dim="member")

                    # update the attr_dictionary
                    # common time period between models: latest start time, earliest end time
                    self.attr_dictionary[variable]["time"] = [
                        np.max([ens_member.time.min().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][0]]), 
                        np.min([ens_member.time.max().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][1]])
                    ]
                    # latitude resolution, then longitude resolution
                    self.attr_dictionary[variable]["res"] = [
                        np.max([np.diff(ens_member.lat[:2])[0], self.attr_dictionary[variable]["res"][0]]), 
                        np.max([np.diff(ens_member.lon[:2])[0], self.attr_dictionary[variable]["res"][1]])
                    ]

                    ens_member.close()

                # add model to the dictionary
                # resample step is to align data types and should not change anything meaningful
                ensemble_list = ensemble_list.convert_calendar("noleap")
                data_dictionary[variable][model] = ensemble_list.resample(time=self.period).mean()
                print('successfully loaded', ii+1, 'ensemble members from', file_path)

        return data_dictionary
    
    
    # -------------------------------------------------------------------------------
    # --------------------------- temporal functions --------------------------------
    # -------------------------------------------------------------------------------
    # ---- select_season, smooth_time, rolling_time, sync_time, shift_inout_time ----
    
    
    def select_season(self, season):
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
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # get rolling seasons of given length
                self.data_dictionary[var][model] = self.data_dictionary[var][model].rolling(time=season[0]).mean().dropna("time", how='all')
                # then select only the seasons requested by season_start
                self.data_dictionary[var][model] = self.data_dictionary[var][model].sel(time=self.data_dictionary[var][model].time.dt.month == season[1])
            
        self.period = "Y"
        print('season selected, now', self.data_dictionary[var][model].time.size, 'time steps')
    
    # -------------------------------------------------------------------------------
    
    def smooth_time(self, period=None):
        """
        smooth time by up-sampling, down-sampling, or resampling on same period, for all data.
        
        inputs:
            - period: a string with the desired time step size, e.g. "2M" for 2 months
            (valid options are "Y", "M", "D").
            if none given, retain current time step.
        outputs:
            - smoothed data_dictionary containing all datasets.
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
            print(sampling, "dataset from", self.period, "to", period)
            # for each variable in the dataset
            for var in self.data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in self.data_dictionary[var].keys():
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].resample(time=period).mean()
        elif sampling == "up-sampling":
            print(sampling, "dataset from", self.period, "to", period)
            for var in self.data_dictionary.keys():
                for model in self.data_dictionary[var].keys():
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].resample(time=period).interpolate(kind='linear')
                    # OTHER INTERPOLATION OPTIONS: "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
        else: # just resampling on the same period
            print(sampling, "dataset")
            for var in self.data_dictionary.keys():
                for model in self.data_dictionary[var].keys():
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].resample(time=self.period).mean()
            
        # update self.period to new period
        self.period = period
        print('time period is now', self.period, 'with', self.data_dictionary[var][model].time.size, 'time steps')
    
    # -------------------------------------------------------------------------------
    
    def rolling_time(self, roll_period):
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
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # rolling average over roll_period for the current time period of the data
                self.data_dictionary[var][model] = self.data_dictionary[var][model].rolling(time=abs(roll_period), center=False).mean().dropna("time", how="all")
                if roll_period < 0:
                    # shift time coordinates
                    new_times = xr.CFTimeIndex(self.data_dictionary[var][model].time.values).shift(n=roll_period+1, freq=self.period)
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].assign_coords(time=new_times)
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=np.arange(abs(roll_period)-1))
                else:
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=-(np.arange(abs(roll_period-1))+1))

        print('rolling average applied, now', self.data_dictionary[var][model].time.size, 'time steps')

    # -------------------------------------------------------------------------------
    
    def window_time(self, window):
        """
        recast target data as rolling average over window of the current time steps.
        
        inputs:
            - window: number of current time steps to average over.
        outputs:
            - data_dictionary with rolling averaged time steps
        """
        assert window != 0, "window cannot be over 0 periods, use positive integer"
        print('applying window over', window, self.period, 'periods')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # rolling average over roll_period for the current time period of the data
                self.data_dictionary[var][model] = self.data_dictionary[var][model].rolling(time=window, center=True).mean().dropna("time", how="all")

        print('target windowed applied, now', self.data_dictionary[var][model].time.size, 'time steps')
    
    # -------------------------------------------------------------------------------
    
    def sync_time(self, time_range=None):
        """
        sync time period and dates between all models.
        
        inputs:
            - time_range: a list, [start date, end date] in year-month-day format as strings,
            e.g. ["1850-01-01", "2005-12-01"].
            if none given, defaults to the common period between all models.
        outputs:
            - synced data_dictionary containing all datasets.
        """
        print('syncing time period and dates between all variables and models')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # if no time range given, use attr_dictionary (common time period between models)
            if time_range is None:
                time_range = [self.attr_dictionary[var]['time'][0].item(), self.attr_dictionary[var]['time'][1].item()]
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # align time periods (start, end)
                self.data_dictionary[var][model] = self.data_dictionary[var][model].sel(time=slice(time_range[0], time_range[1]))
            # align dates (only keep common dates)
            temp_data = xr.align(*[self.data_dictionary[var][model] for model in self.data_dictionary[var].keys()], exclude=["member", "lat", "lon"]) 
            # reassemble dictionary
            for ii, model in enumerate(self.data_dictionary[var].keys()):
                self.data_dictionary[var][model] = temp_data[ii]
                
        print('time period and dates synced, now using', self.data_dictionary[var][model].time.size, 'dates')
    
    # -------------------------------------------------------------------------------
    
    def shift_inout_time(self, shift, window=1):
        """
        shift the datasets, i.e. create a leadtime by removing entries from the start or end.
        
        inputs:
            - shift: the number of entries to remove.
            negative values remove shift entries from the end (adding a lag, for input),
            positive values remove shift entries from the start (adding a lead, for target).
            - window: the window over which the target is being averaged.
        outputs:
            - shifted data_dictionary.
        """
        assert shift != 0, "shift should be a positive or negative integer, not zero"
        print('shifting the data by', shift, 'time steps')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in self.data_dictionary[var].keys():
                    # drop the first or last shift entries in the datasets
                    if shift > 0:
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=np.arange(shift - np.floor(window/2).astype(int)))
                    else:
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=-(np.arange(abs(shift) + np.floor(window/2).astype(int)) + 1))
        
        print('data shifted, now using', self.data_dictionary[var][model].time.size, 'time steps')
    
        
    # -------------------------------------------------------------------------------
    # ---------------------------- spatial functions --------------------------------
    # -------------------------------------------------------------------------------
    # --------------------- smooth_space, mask_data, get_region ---------------------

                
    def smooth_space(self, resolutions=None):
        """
        smooth resolution between all data; user selected or (default) coarsest data.
        should be done on the global data before extracting a region.
        
        inputs:
            - resolutions: list of resolution in lat/lon order, e.g. [1.25, 1.75].
            if none given, defaults to the coarsest resolution amongst all models.
        outputs:
            - data_dictionary datasets mapped to the same lat/lon resolution, without altering the lat/lon range.
        """
        # assert that the map is global
        assert self.global_map, "up- or down-sampling should be done before extracting a region"
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # if no time range given, use attr_dictionary (common time period between models)
            if resolutions is None:
                resolutions = self.attr_dictionary[var]['res']
            print('regridding datasets to resolution', resolutions)
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # construct the regridder using xESMF: https://xesmf.readthedocs.io/en/latest/index.html
                latlon_dict = {
                    "lat": np.arange(-90+0.5*resolutions[0], 90, resolutions[0]),
                    "lon": np.arange(0, 360, resolutions[1])
                }
                regridder = xe.Regridder(self.data_dictionary[var][model], xr.Dataset(latlon_dict), "bilinear", periodic=True)
                # do the regridding
                self.data_dictionary[var][model] = regridder(self.data_dictionary[var][model], keep_attrs=True)        
        
    # -------------------------------------------------------------------------------
    
    def mask_data(self, mask):
        """
        mask the oceans or the land.
        if available, uses model specific mask, else use natural earth 1:10m map
        
        inputs:
            - mask: a string for what to mask, either 'ocean' or 'land'
        outputs:
            - data_dictionary with either land or ocean masked from all datasets.
        """
        assert np.isin(mask, ['ocean', 'land']), "input mask should be either 'land' or 'ocean'"
        print('masking the', mask, 'from all datasets')
        
        # get the backup map from natural earth
        mask_map = regionmask.defined_regions.natural_earth_v5_0_0.land_10
        
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # try loading model specific mask first
                try:
                    # load map for this model
                    model_mask = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'raw_data', self.period_long, model, mask+'mask.nc'))
                    # if the current resolution is different from default (does not match saved mask)
                    if model_mask.dims != self.data_dictionary[var][model][["lat", "lon"]].dims:
                        # adjust the resolution of the mask to be the same as the model and variable
                        regridder = xe.Regridder(model_mask, self.data_dictionary[var][model], "bilinear", periodic=True)
                        # other methods: 'bilinear' 'nearest_s2d' 'nearest_d2s'
                        model_mask = regridder(model_mask, keep_attrs=True)
                    final_mask = model_mask.mask
                    print('masking', model, 'model using model specific mask')
                # no model specific mask, use natural earth
                except:
                    # divide into either land or ocean mask
                    final_mask = mask_map.mask(self.data_dictionary[var][model]) + 1
                    if mask == 'land':
                        final_mask = final_mask.fillna(0).where(final_mask.fillna(0) == 0) + 1
                    print('masking', model, 'using natural earth map')
                
                # mask the datasets
                self.data_dictionary[var][model] = self.data_dictionary[var][model] * final_mask
                # save the final_mask for later use
                self.final_mask = final_mask

    # -------------------------------------------------------------------------------

    def get_region(self, box_or_regions):
        """
        return data for the specified region only.
        can be a list of numeric boundaries, or region names.
        
        inputs:
            - box_or_regions: list of boundary coordinates, i.e. [minlon, minlat, maxlon, maxlat],
            or a region name (or list of region names), e.g. ['NWN', 'E.North-America', 'NCA']
        outputs:
            - data_dictionary with datasets for only the specified regions.
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
                for var in self.data_dictionary.keys():
                    # for each model (get down to the xarray level from the dictionary)
                    for model in self.data_dictionary[var].keys():
                        # make latitude and longitude masks
                        mask_lon = (self.data_dictionary[var][model].lon >= lon[0]) & (self.data_dictionary[var][model].lon <= lon[1])
                        mask_lat = (self.data_dictionary[var][model].lat >= lat[0]) & (self.data_dictionary[var][model].lat <= lat[1])
                        # apply masks and drop coordinates, etc. outside of region
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].where(mask_lon & mask_lat, drop=True)
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

                for var in self.data_dictionary.keys():
                    for model in self.data_dictionary[var].keys():
                        # make geopandas object for masking (and get correct resolution of mask)
                        region_mask = regionmask.mask_geopandas(region.to_geodataframe(), self.data_dictionary[var][model]) 
                        # make the regions 0 or nan using region_num associated with requested regions
                        region_mask = region_mask.where(np.isin(region_mask, region_num))
                        region_mask = region_mask.where(region_mask.isnull(), 1)
                        # apply mask to datasets
                        self.data_dictionary[var][model] = self.data_dictionary[var][model] * region_mask
                        # drop coordinates, etc. outside of region
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].where(abs(self.data_dictionary[var][model]) >= 0, drop=True)

        # if input is a list, use these as boundaries to define region
        else:
            assert len(box_or_regions) == 4, "box should have 4 entries, [minlon, minlat, maxlon, maxlat]"
            for var in self.data_dictionary.keys():
                for model in self.data_dictionary[var].keys():
                    # make latitude and longitude masks
                    mask_lon = (self.data_dictionary[var][model].lon >= box_or_regions[0]) & (self.data_dictionary[var][model].lon <= box_or_regions[2])
                    mask_lat = (self.data_dictionary[var][model].lat >= box_or_regions[1]) & (self.data_dictionary[var][model].lat <= box_or_regions[3])
                    # apply masks and drop coordinates, etc. outside of region
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].where(mask_lon & mask_lat, drop=True)
        
        # set global_map switch to false (for up- or down-scaling resolution)
        self.global_map = False
    
    
    # -------------------------------------------------------------------------------
    # ---------------------------- data manipulation --------------------------------
    # -------------------------------------------------------------------------------
    # -- remove_seasonality, remove_trend, norm_data, regional_mean, baseline_norm --


    def remove_seasonality(self, seasonality_dictionary=False):
        """
        remove seasonality (monthly) from the data.
        
        inputs:
            - seasonality_dictionary: boolean determining if the calculated seasonalities should be
            retained in a separate dictionary.
        outputs:
            - data_dictionary with the seasonality removed from each dataset.
        """
        print('removing seasonality from all datasets')
        if seasonality_dictionary:
            self.seasonality_dictionary = {}
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            if seasonality_dictionary: self.seasonality_dictionary[var] = {}
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # calculate the seasonality as the monthly average
                seasonality = self.data_dictionary[var][model].groupby('time.month').mean("time")
                
                if seasonality_dictionary:
                    # if requested, save the seasonal signal in a separate dictionary
                    self.seasonality_dictionary[var][model] = seasonality
                    
                # and remove it from the dataset
                self.data_dictionary[var][model] = (self.data_dictionary[var][model].groupby('time.month') - seasonality).drop("month")

    # -------------------------------------------------------------------------------

    def remove_trend(self, fit=None, trend_dictionary=False):
        """
        remove the long term trend from the data, either by ensemble mean, or a fit.
        
        inputs:
            - fit: whether to fit the trend using polyfit, a rolling mean, or ensemble mean.
            default is ensemble mean, if an integer is passed, that order polyfit is used,
            and if the string 'rolling' is passed, a five year rolling average is used.
            - trend_dictionary: boolean determining if the calculated trends should be
            retained in a separate dictionary.
        outputs:
            - data_dictionary with the trend removed from each dataset.
        """
        print('removing trend from all datasets')
        if trend_dictionary:
            self.trend_dictionary = {}
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            if trend_dictionary: self.trend_dictionary[var] = {}
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                if fit == 'rolling':
                    # use a rolling mean over 5 years
                    if self.period[-1] == 'Y':
                        assert self.data_dictionary[var][model]["time.year"].size > 5, "too short a time period to take rolling average"
                        roll = int(5/int(self.period[:-1]))
                        trend = self.data_dictionary[var][model].mean("member").rolling(time=roll, min_periods=1, center=True).mean()
                    elif self.period[-1] == 'M':
                        assert self.data_dictionary[var][model]["time"].size > 60, "too short a time period to take rolling average"
                        roll = int(60/int(self.period[:-1]))
                        trend = self.data_dictionary[var][model].mean("member").rolling(time=roll, min_periods=1, center=True).mean()
                    else:
                        assert False, "rolling average not implemented for daily data"
                elif type(fit) == int:
                    # use fit degree polyfit on the ensemble mean
                    trend = self.data_dictionary[var][model].mean("member").polyfit(dim="time", deg=fit)
                    trend = xr.polyval(self.data_dictionary[var][model].time, trend)
                else:
                    # print a note if using ensemble mean for trend with fewer than 10 members
                    if self.data_dictionary[var][model].member.size < 10:
                        print('!! removing trend using ensemble mean with only', self.data_dictionary[var][model].member.size, 'members !!')
                    if self.data_dictionary[var][model].member.size == 1:
                        print('single member dataset, skipping trend removal')
                        continue
                    trend = self.data_dictionary[var][model].mean("member")
                
                if trend_dictionary:
                    # if requested, save the trend in a separate dictionary
                    self.trend_dictionary[var][model] = trend
                    
                # remove the trend from the datasets
                self.data_dictionary[var][model] = self.data_dictionary[var][model] - trend

    # -------------------------------------------------------------------------------

    def norm_data(self):
        """
        remove the mean and divide by the standard deviation (over members and time) for each of the datasets.
        
        outputs:
            - standardized datasets
        """
        print('standardizing datasets by subtracting mean and dividing by std')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # get mean and std over all members and times
                model_mean = self.data_dictionary[var][model].mean(["member", "time"], skipna=True)
                model_std = self.data_dictionary[var][model].std(["member", "time"], skipna=True)
                # standardize the datasets
                self.data_dictionary[var][model] = (self.data_dictionary[var][model] - model_mean) / model_std

    # -------------------------------------------------------------------------------

    def regional_mean(self, var_list=None):
        """
        get the mean for a variable over the entire region for each dataset
        
        inputs:
            - var_list: a list of variables that should be converted to lat/lon averages.
            the entries in var_list need to match the names of the variables of the dataset.
        outputs:
            - data_dictionary with global (over remaining region) averages for the
            variables in var_list.
            shape will be [time, member].
        """
        # if no variables are specified, use all available variables in the data_dictionary
        if var_list is None:
            var_list = list(self.data_dictionary.keys())
        print('calculating mean for variables: ', *var_list)
        if type(var_list) is not list:
            var_list = [var_list,]
        
        # for each requested variable
        for var in var_list:
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # get the latitude weights
                weights = np.cos(np.deg2rad(self.data_dictionary[var][model].lat))
                # get the global mean
                self.data_dictionary[var][model] = self.data_dictionary[var][model].weighted(weights).mean(("lon", "lat"), skipna=True)
    
    # -------------------------------------------------------------------------------
    
    # TODO: explore dividing by the STD over the baseline period to normalize further
    def baseline_norm(self, baseline, var_list=None):
        """
        normalize the datasets by a baseline period averaged over time and members.
        
        inputs:
            - baseline: a list of a start and end year for the baseline, e.g. [1951, 1980]
        outputs:
            - data_dictionary where each model has been normalized by the mean of the baseline
            over time and members.
        """
        if var_list is None:
            var_list = list(self.data_dictionary.keys())
        print('baselining on', baseline, 'for variables:', *var_list)
        # for each requested variable
        for var in var_list:
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():                
                # get the time range indices
                baseline_inds = np.where(np.logical_and(self.data_dictionary[var][model]['time.year'] >= baseline[0], 
                                             self.data_dictionary[var][model]['time.year'] <= baseline[1]))[0]
                # baseline mean over time for each member
                baseline_mean = self.data_dictionary[var][model].isel(time=baseline_inds).mean(['time'], skipna=True)
                # remove the baseline from the dataset
                self.data_dictionary[var][model] = self.data_dictionary[var][model] - baseline_mean
    
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
