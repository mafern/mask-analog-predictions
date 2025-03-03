"""
DATA PROCESSING CLASS: only works with netcdf (.nc) files currently
"""
import numpy as np
import glob
import os
import xarray as xr
xr.set_options(keep_attrs=True)
import xesmf as xe
import regionmask
import json
from copy import deepcopy

class DataProcessing():
    """
    class to load models and perform various preprocessing steps to prepare the data for the mask training or analog library
    
    inputs:
        - model_list: list of models to use
        - variable_list: list of variable to use
        - n_members: either an integer number of members, or a list of the indices to use
        - seed: random seed for selection of members
        - resolution: resolution to use for members (also, where to look for presaved data)
        - model_member_dict: list of members for each model to help avoid using the same member twice
        - load_raw: whether to load from raw monthly data if presaved data is not available
    outputs:
        - 
    """
    def __init__(self, model_list, variable_list, n_members, seed, resolution, model_member_dict=None, load_raw=False):
        ""
        
        # save some global variables that will be useful
        self.user_regions = json.load(open(os.path.join(os.path.dirname(__file__), 'regions.json'), 'r'))
        self.regionmask_regions = ['regionmask.defined_regions.ar6.all[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.countries_10[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.us_states_10[box_or_regions]',
                              'regionmask.defined_regions.srex[box_or_regions]',
                              'regionmask.defined_regions.natural_earth_v5_0_0.land_10[box_or_regions]'
                             ]
        self.global_map = True

        # resolution, for checking previously saved preprocessed data
        self.resolution = resolution
        if model_member_dict is None:
            # save model_member_dict
            model_member_dict = {raw_data_dir.split('/')[-1][:-3]: xr.load_dataset(raw_data_dir).member.to_numpy().astype(int) for raw_data_dir in np.sort(glob.glob(os.path.join(os.path.dirname(__file__), f'processed_data/{variable_list[0]}_lat{np.round(self.resolution[0], 2)}_lon{np.round(self.resolution[1], 2)}', '*')))}
        self.model_member_dict = model_member_dict

        # whether to load raw data or cancel if preprocessed is not found
        self.load_raw = load_raw
        # load the data from the requested models and variables
        self.data_dictionary = self.load_data(model_list, variable_list, n_members, seed)
 

    def load_data(self, model_list, variable_list, n_members, seed):
        """
        load files for models in model_list for the variables in variable_list
        data needs to be stored with the directory structure: data/variable_name/model_name/files
        returns the data in a dictionary with xarray formatted entries
        
        inputs:
            - model_list: a list of model names, e.g. ['CESM2', 'MPI',]
            - variable_list: a list variable names, e.g. ['ts', 'pr',]
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
                "time": ["0000-01-01", "3000-01-01"]
            }
            # get a model_member_dictionary for this variable
            model_member_dict = deepcopy(self.model_member_dict)
            
            for model in model_list:
                nm = n_members
                if type(nm) == int:
                    # check there are still enough members for this model when training
                    if np.size(model_member_dict[model]) < nm:
                        print("more members requested than available. requested: " + str(nm) + ", available: " + str(np.size(model_member_dict[model])))
                        if np.size(model_member_dict[model]) == 0:
                            print('skipping', model)
                            continue
                        else:
                            print('using', np.size(model_member_dict[model]), 'members from', model)
                            nm = np.size(model_member_dict[model])
                if type(nm) is list:
                    # members have already been chosen (e.g. best_analogs process_new_data)
                    used_members = nm
                else:
                    # generate randomly seeded choice object and randomly select members
                    rng_seed = np.random.default_rng(seed)
                    used_members = np.sort(rng_seed.choice(model_member_dict[model], nm, replace=False, shuffle=False))

                if not self.load_raw:
                    # check if the preprocessed data exists for this model/variable/resolution
                    var_res_dir = variable+'_lat'+str(np.round(self.resolution[0], 2))+'_lon'+str(np.round(self.resolution[1], 2))
                    preprocessed_path = os.path.join(os.path.dirname(__file__), "processed_data", var_res_dir)
                    preprocessed_file = os.path.join(preprocessed_path, model+'.nc')
                    # if it does exist, load it, otherwise skip it
                    if os.path.exists(preprocessed_file):
                        print('found preprocessed data for', model)
                        # remove height coordinate if it exists here
                        ensemble_list = xr.load_dataset(preprocessed_file).sel(member=used_members)
                        if np.isin('height', list(ensemble_list._coord_names)):
                            ensemble_list = ensemble_list.drop('height')
                        data_dictionary[variable][model] = ensemble_list
                    else:
                        print('no preprocessed data found for', model)
                        print('skipping')
                        continue
                else:
                    print('processing from raw data')
                    # file path for this variable and model
                    file_path = os.path.join(os.path.dirname(__file__), 'raw_data/', model, variable)
                    # check that path exists, if not print warning and continue
                    if not os.path.exists(file_path):
                        print('directory:', file_path, 'does not exist')
                        print('continuing without this dataset')
                        continue
                    # get all files of this model and variable
                    file_list = np.sort(glob.glob(os.path.join(file_path, '*')))
                    # select files for correct members
                    file_list = file_list[used_members]
                    
                    for ii, file in enumerate(file_list):
                        # load file -- will be returned in xarray format
                        ens_member = xr.load_dataset(file).squeeze()
                        # resample for consistency with monthly data (day of month)
                        ens_member = ens_member.resample(time='M').mean()

                        # create ensemble_list or add to ensemble_list
                        if ii == 0:
                            # add member dimension
                            ensemble_list = ens_member.expand_dims(dim={"member": 1}, axis=0)
                        else:
                            ensemble_list = xr.concat([ensemble_list, ens_member], dim="member")
                        ens_member.close()

                    # add model to the dictionary
                    # resample step is to align data types and should not change anything meaningful
                    ensemble_list = ensemble_list.convert_calendar("noleap")
                    data_dictionary[variable][model] = ensemble_list.resample(time='1M').mean()
                    print('successfully loaded', ii+1, 'ensemble members from', file_path)

                # update the attr_dictionary
                # common time period between models: latest start time, earliest end time
                self.attr_dictionary[variable]["time"] = [
                    np.max([ensemble_list.dropna(dim='time').time.min().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][0]]), 
                    np.min([ensemble_list.dropna(dim='time').time.max().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["time"][1]])
                ]

                # update model_member_dict to remove these members for later splits
                model_member_dict[model] = np.setdiff1d(model_member_dict[model], used_members)

        # update self.model_member_dict (all variables should pick same members due to seed, so once at the end is fine here)
        self.model_member_dict = model_member_dict
        return data_dictionary
    
    # -------------------------------------------------------------------------------
    # --------------------------- temporal functions --------------------------------
    # -------------------------------------------------------------------------------
    # ------------------- smooth_time, sync_time, shift_inout_time ------------------
    
    def smooth_time(self, stat='mean'):
        """
        smooth time to yearly for data.
        """
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                print("downsampling dataset to yearly")
                self.data_dictionary[var][model] = self.data_dictionary[var][model].resample(time='1Y').mean()
        print('time period now contains', self.data_dictionary[var][model].time.size, 'time steps')

    # -------------------------------------------------------------------------------
    
    def window_time(self, window):
        """
        recast target data as rolling average over window of the current time steps.
        
        inputs:
            - window: number of current time steps to average over.
        outputs:
            - data_dictionary with rolling averaged time steps
        """
        assert window > 0, "window be a positive integer"
        print('applying window over', window, 'periods')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # rolling average over roll_period for the current time period of the data
                self.data_dictionary[var][model] = self.data_dictionary[var][model].rolling(time=window, center=False).mean().dropna("time", how="all")

        print('target windowed applied, now', self.data_dictionary[var][model].time.size, 'time steps')
    
    # -------------------------------------------------------------------------------
    
    def sync_time(self, time_range=None):
        """
        sync time period and dates between all models.
        
        inputs:
            - time_range: a list, [start date, end date] as strings,
            e.g. ["1850", "2005"].
            if none given, defaults to the common period between all models.
        outputs:
            - synced data_dictionary containing all datasets.
        """
        print('syncing time period and dates between all variables and models')
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():
                # if no time range given, use attr_dictionary (common time period between models)
                if time_range is None:
                    time_range = [self.attr_dictionary[var]['time'][0].item(), self.attr_dictionary[var]['time'][1].item()]
                    # align dates (only keep common dates)
                    temp_data = xr.align(*[self.data_dictionary[var][model] for model in self.data_dictionary[var].keys()], exclude=["member", "lat", "lon"]) 
                    # reassemble dictionary
                    for ii, model in enumerate(self.data_dictionary[var].keys()):
                        self.data_dictionary[var][model] = temp_data[ii]
                else:
                    # align time periods (start, end)
                    self.data_dictionary[var][model] = self.data_dictionary[var][model].sel(time=slice(time_range[0], time_range[1]))
                
        print('time period and dates synced, now using', self.data_dictionary[var][model].time.size, 'dates')
    
    # -------------------------------------------------------------------------------

    def shift_inout_time(self, shift):
        """
        shift the datasets, i.e. create a leadtime by removing entries from the start or end.

        inputs:
            - shift: the number of entries to remove.
            negative values remove shift entries from the end (adding a lag, for input),
            positive values remove shift entries from the start (adding a lead, for target).
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
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=np.arange(shift))
                    else:
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].drop_isel(time=-(np.arange(abs(shift)) + 1))

        print('data shifted, now using', self.data_dictionary[var][model].time.size, 'time steps')
    
        
    # -------------------------------------------------------------------------------
    # ---------------------------- spatial functions --------------------------------
    # -------------------------------------------------------------------------------
    # --------------------- smooth_space, mask_data, get_region ---------------------
                
    def smooth_space(self, resolutions):
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
                self.data_dictionary[var][model] = regridder(self.data_dictionary[var][model], keep_attrs=True, skipna=True)
        
    # -------------------------------------------------------------------------------
    
    def mask_data(self, mask):
        """
        mask the oceans or the land using natural earth 1:10m map
        
        inputs:
            - mask: a string for what to mask, either 'ocean' or 'land'
        outputs:
            - data_dictionary with either land or ocean masked from all datasets.
        """
        assert np.isin(mask, ['ocean', 'land']), "input mask should be either 'land' or 'ocean'"
        print('masking the', mask, 'from all datasets')
        
        # get the map from natural earth
        mask_map = regionmask.defined_regions.natural_earth_v5_0_0.land_10
        
        # for each variable in the dataset
        for var in self.data_dictionary.keys():
            # for each model (get down to the xarray level from the dictionary)
            for model in self.data_dictionary[var].keys():            
                # get the right grid for the mask
                final_mask = mask_map.mask(self.data_dictionary[var][model]) + 1
                # divide into either land or ocean mask
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
                        self.data_dictionary[var][model] = self.data_dictionary[var][model].where(~np.isnan(region_mask), drop=True)

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
    # ------------------- remove_trend, norm_data, baseline_norm --------------------

    def remove_trend(self, fit=None, trend_dictionary=True):
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
                    assert self.data_dictionary[var][model]["time.year"].size > 5, "too short a time period to take rolling average"
                    roll = 5
                    trend = self.data_dictionary[var][model].mean("member").rolling(time=roll, min_periods=1, center=True).mean()
                elif type(fit) == int:
                    # use fit degree polyfit on the ensemble mean
                    trend = self.data_dictionary[var][model].mean("member").polyfit(dim="time", deg=fit)
                    trend = xr.polyval(self.data_dictionary[var][model].time, trend)
                    trend = trend.rename_vars({var+'_polyfit_coefficients': var})
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
