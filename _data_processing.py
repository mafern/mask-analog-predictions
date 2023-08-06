"""
DATA PROCESSING CLASS
"""
import numpy as np
import glob
import os
import xarray as xr



def route_by_filetype(file_type, file):
    # determine file type

    # call appropriate loading function

    # convert to xarray format


def load_nc(file):
    # try except with xarray then scipy.io?


def load_hdf5(file):



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


        # if not, call load_data


        # save some global variables (data mean, std, file names, etc.) for variables and models
        self.period = period[0].upper()
        self.period_dictionary = {"Y": 0, "M": 1, "D": 2}


        # if not None, go through the steps in the processing list (order matters!)

        
        
        # -------------------- move to data_processing -----------------------
        def stack_samples(self, data_dictionary, ):
            "stack samples by given dimension (ensemble=0, model=1, . . .)"
            print('')


            print('')
            return data_dictionary


        # save processed data if savefile is given
        "save processed data in xx format"



    def load_data(self, model_list, variable_list, period, n_ensembles=None, seed=123):
        """
        load files for models in model_list for the variables in variable_list
        data needs to be stored with the directory structure: data/variable_name/model_name/files
        returns the data in a dictionary with xarray formatted entries
        
        inputs:
            - 
        outputs:
            - 
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
                "common_time": ["0000-01-01", "3000-01-01"], 
                "coarsest_res": [0, 0], 
                "lat_range": [-90, 90], 
                "lon_range": [0, 360]}
            
            for model in model_list:
                # file path for this variable and model
                file_path = os.path.join('data', variable, period, model)
                # check that path exists, if not print warning and continue
                if not os.path.exists(file_path):
                    print('directory:', file_path, 'does not exist')
                    print('continuing without this dataset')
                    continue
                # get all files of this model and variable
                file_list = np.sort(glob.glob(os.path.join(file_path, '*')))

                # NEED TO HANDLE WHEN FILE IS SPLIT BETWEEN MULTIPLE SUB-FILES
                # (in the random ensemble below [ensure getting all files for given ensemble] and the loading step)
                # for now, not including those files
                # could gather different variations on what meta data calls ensemble member (realization, etc.)
                # the find that for each, keep previous one saved and compare, if the same concat along time
                # otherwise concat along member -- would need to rearrange the random n_ensembles snippet

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
                    self.attr_dictionary[variable]["common_time"] = [
                        np.max([ens_member.time.min().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["common_time"][0]]), 
                        np.min([ens_member.time.max().dt.strftime("%Y-%m-%d").values, self.attr_dictionary[variable]["common_time"][1]])
                    ]
                    # latitude resolution, then longitude resolution
                    # will need to add altitude resolution when playing with 3D variables
                    self.attr_dictionary[variable]["coarsest_res"] = [
                        np.max([np.diff(ens_member.lat[:2]), self.attr_dictionary[variable]["coarsest_res"][0]]), 
                        np.max([np.diff(ens_member.lon[:2]), self.attr_dictionary[variable]["coarsest_res"][1]])
                    ]
                    # latitude range
                    self.attr_dictionary[variable]["lat_range"] = [
                        np.max([ens_member.lat.min(), self.attr_dictionary[variable]["lat_range"][0]]), 
                        np.min([ens_member.lat.max(), self.attr_dictionary[variable]["lat_range"][1]])
                    ]
                    # longitude range
                    self.attr_dictionary[variable]["lon_range"] = [
                        np.max([ens_member.lon.min(), self.attr_dictionary[variable]["lon_range"][0]]), 
                        np.min([ens_member.lon.max(), self.attr_dictionary[variable]["lon_range"][1]])
                    ]

                # add model to the dictionary
                data_dictionary[variable][model] = ensemble_list.convert_calendar("noleap")
                print('successfully loaded', ensemble_list['member'].size, 'ensemble members from', file_path)

        return data_dictionary
    
    
    # -------------------------------------------------------------------------------
    # --------------------------- temporal functions --------------------------------
    # -------------------------------------------------------------------------------
    # ---- select_season, smooth_time, rolling_time, sync_time, shift_inout_time ----
    
    # precautionary resample -- first step for all?
    # seasonal rolling -- select_season


    # ### If a certain season...
    # if 'season' in settings.keys():
    #     num_mos, cen_mo = settings["season"]
    #     da_all = da_all.rolling(min_periods=num_mos, center=True, time=num_mos).mean().dropna("time", how='all')
    #     da_all = da_all.sel(time=da_all.time.dt.month == cen_mo)


    def select_season(self, data_dictionary, ):
        """
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # select a certain season
        
        
        print('')
        return data_dictionary
    

    def shift_inout_time(self, data_dictionary, ):
        """
        
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # create a shift in the input/output
        
        
        print('')
        return data_dictionary
    
    
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
        current_period = self.period_dictionary[self.period]
        if period is None:
            requested_period = current_period
        else:
            period = period.upper()
            requested_period = self.period_dictionary[period[-1]]
        
        # first, resample to the correct period type: day, month, or year
        if requested_period < current_period:
            print("down-sampling data set from", self.period, "to", period)
            # for each variable in the dataset
            for var in data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=period).mean()
        elif requested_period > current_period:
            print("up-sampling data set from", self.period, "to", period)
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=period).interpolate(kind='linear')
                    # OTHER INTERPOLATION OPTIONS: "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
        else: # keep the same period
            print("resampling data set")
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=self.period).mean()
                    
        # update self.period to new period
        self.period = period
        print('time period is now', self.period, 'with', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
    
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
            for model in data_directory[var].keys():
                # align time periods (start, end)
                data_directory[var][model] = data_dictionary[var][model].sel(time=slice(time_range[0], time_range[1]))
            # align dates (only keep common dates)
            temp_data = xr.align(*[data_dictionary[var][model] for model in data_dictionary[var].keys()], exclude=["member", "lat", "lon"]) 
            # reassemble dictionary
            for ii, model in enumerate(data_dictionary[var].keys()):
                data_dictionary[var][model] = temp_data[ii]
                
        print('time period and dates synced, now using', data_dictionary[var][model].time.size, 'dates')
        return data_dictionary
    
    
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
        print('applying rolling average for', roll_period, self.period)
        try:
            # for each variable in the dataset
            for var in data_dictionary.keys():
                # for each model (get down to the xarray level from the dictionary)
                for model in data_dictionary[var].keys():
                    # rolling average over roll_period for the current time period of the data
                    data_dictionary[var][model] = data_dictionary[var][model].rolling(time=abs(roll_period), center=False).mean().dropna("time")
                    if roll_period < 0:
                        # shift time coordinates
                        new_times = xr.CFTimeIndex(data_dictionary[var][model].time.values).shift(n=roll_period+1, freq=self.period)
                        data_dictionary[var][model] = data_dictionary[var][model].assign_coords(time=new_times)
        except:
            for var in data_dictionary.keys():
                for model in data_dictionary[var].keys():
                    # may need to be resampled first
                    data_dictionary[var][model] = data_dictionary[var][model].resample(time=self.period).mean()
                    data_dictionary[var][model] = data_dictionary[var][model].rolling(time=abs(roll_period), center=False).mean().dropna("time")
                    if roll_period < 0:
                        # shift time coordinates
                        new_times = xr.CFTimeIndex(data_dictionary[var][model].time.values).shift(n=roll_period+1, freq=self.period)
                        data_dictionary[var][model] = data_dictionary[var][model].assign_coords(time=new_times)

        print('rolling average applied, now', data_dictionary[var][model].time.size, 'time steps')
        return data_dictionary
    
        
    # -------------------------------------------------------------------------------
    # ---------------------------- spatial functions --------------------------------
    # -------------------------------------------------------------------------------
    # --------------- sync_space, smooth_space, mask_data, get_region ---------------

        
    def sync_space(self, data_dictionary, ):
        """
        sync coordinate range between all data; user selected or (default) narrowest range data
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # make common lat and lon (and elev) range
        
        
        print('')
        return data_dictionary


    def smooth_space(self, data_dictionary, ):
        """
        smooth resolution between all data; user selected or (default) coarsest data
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # make same resolution -- up or down scaling required
        
        
        print('')
        return data_dictionary


    def mask_data(self, data_dictionary, ):
        """
        mask the oceans or the land -- seperate from specifying region
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        
        
        print('')
        return data_dictionary


    def get_region(self, data_dictionary, ):
        """
        return data for the specified region only
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # this may use a external package (would want to list options somehow)
        
        
        print('')
        return data_dictionary
    
    
    # -------------------------------------------------------------------------------
    # ---------------------------- data manipulation --------------------------------
    # -------------------------------------------------------------------------------
    # --------- remove_seasonality, remove_trend, norm_data, regional_mean ----------


    def remove_seasonality(self, data_dictionary, ):
        """
        remove the seasonal trend from the data, by ensemble, model, or entire dataset
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # subtract monthly average across full time period for each grid cell
        
        
        print('')
        return data_dictionary


    def remove_trend(self, data_dictionary, ):
        """
        remove the long term trend from the data, either by model or entire dataset
        inputs:
            - 
        outputs:
            - 
        """
        print('')
        # subtract ensemble mean for each grid cell
        # may need to add fitting for cases with few ensemble members
        # could add option to remove via LIM least damped mode
        
        
        print('')
        return data_dictionary


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


    @property
    def data_operations_list(self):
        "return a list of the data processing operations available in the DataProcessing class"
        function_list = []
        for function in self.__dir__(): # for each entry in the directory of this class
            if function != 'data_operations_list': # ignore this function (else recursion issue)
                # check that it is a function (callable) and not built-in (__)
                if callable(getattr(self, function)) and not function.startswith("__"):
                    function_list.append(function)
        return np.array(function_list)
