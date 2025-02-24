import numpy as np
import pickle
import xarray as xr
import glob
import _data_processing as _dp
import run_experiments as rexp
import os
import gzip
import copy
import bottleneck as bn

FLOAT_TYPE = np.float32
INT_TYPE = np.int32

# -------------------------------------------------------------------------------
# general use functions
# -------------------------------------------------------------------------------

def get_targets(experiment_name, soi_model, soi_member, tether, variable, n_analogs, prediction_dict=None,
                return_average=True, global_mean=False, max_year=None, rank_alpha=None, soi_signature=None,
                leadtime=None, library_name=None):
    """
    get the best analogs for the given soi model/member
    """
    # load the prediction_dict if none is passed
    if prediction_dict is None:
        prediction_dict = BestAnalogs(experiment_name, library_name=library_name).prediction_dict
    # make sure there are analogs to get targets from
    assert len(list(prediction_dict['analogs'].keys())) > 0, "no soi analogs calculated yet"

    if soi_signature is None:
        # get the soi savename corresponding to the requested soi_model, soi_member, and tether
        soi_signature = soi_model + '_mem' + str(soi_member) + '_teth' + str(tether)
        if library_name is not None:
            soi_signature = soi_signature + '_library_' + library_name

    # get the indices for the selected soi
    analog_path = os.path.join(os.path.dirname(__file__), "saved_analogs", experiment_name, soi_signature+'.pickle')
    with gzip.open(analog_path, 'rb') as file:
        analog_indices = pickle.load(file)

    # weights and leadtime
    weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
    if leadtime is None:
        leadtime = rexp.Experiments().get_experiment(experiment_name)['leadtime']

    # get the soi_ind corresponding to the requested soi_model and soi_member for the inputs and target
    if "global" in soi_model or "regional" in soi_model:
        # return the experiment name without the "global" or "regional" suffix
        suffix_ind = soi_model[::-1].find('_')
        soi_model = soi_model[:-(suffix_ind+1)]
    soi_index = np.where((prediction_dict['target'].model == soi_model) &
                         (prediction_dict['target'].member == soi_member))[0]
    
    # get the soi (aka the truth))
    truth = prediction_dict['target'].isel(sample=soi_index[tether-1:])[variable]
    truth = truth.drop('sample').rename({'sample': 'time'}).assign_coords({"member": ("time", truth.member.data), "time": ("time", truth.time.data)})
    if global_mean:
        truth = truth.weighted(weights).mean(("lon", "lat"), skipna=True)

    # get the target selection variables, either unconstrained or constrained to be earlier than max_year
    if max_year is not None:
        max_year_inds = analog_indices['best_input_times.year'].data < max_year
        # matching model, member, year (add leadtime to get prediction target)
        best_years = analog_indices['best_input_times.year'].data[max_year_inds].reshape(analog_indices.time.size, -1)[:, np.arange(n_analogs)] + leadtime
        best_years = xr.DataArray(best_years, dims=["time", "analogs"], coords={"time": ("time", analog_indices.soi_time.data)})
        best_years = (prediction_dict['target']['time.year'] == best_years.drop("time"))
        best_members = analog_indices['best_input_members'].data[max_year_inds].reshape(analog_indices.time.size, -1)[:, np.arange(n_analogs)]
        best_members = xr.DataArray(best_members, dims=["time", "analogs"], coords={"time": ("time", analog_indices.soi_time.data)})
        best_members = (prediction_dict['target'].member == best_members.drop("time"))
        best_models = analog_indices['best_input_models'].data[max_year_inds].reshape(analog_indices.time.size, -1)[:, np.arange(n_analogs)]
        best_models = xr.DataArray(best_models, dims=["time", "analogs"], coords={"time": ("time", analog_indices.soi_time.data)})
        best_models = (prediction_dict['target'].model == best_models.drop("time"))
        best_samples = best_models * best_members * best_years
    else:
        # matching model, member, year (add leadtime to get prediction target)
        best_models = analog_indices.isel(analogs=np.arange(n_analogs))['best_input_models']
        best_models = (prediction_dict['target'].model == best_models)
        best_members = analog_indices.isel(analogs=np.arange(n_analogs))['best_input_members']
        best_members = (prediction_dict['target'].member == best_members)
        best_years = analog_indices.isel(analogs=np.arange(n_analogs))['best_input_times.year'] + leadtime
        best_years = (prediction_dict['target']['time.year'] == best_years)
        best_samples = best_models * best_members * best_years

    # get the prediction -- first reorganize the coordinates for ease of use
    targets = prediction_dict['target'].drop('sample').assign_coords({'member': ("sample", prediction_dict['target'].member.data), 'analog_time': ("sample", prediction_dict['target']['time.year'].data)})
    for ii in range(analog_indices.time.size):
        best_inds, order = np.where(best_samples.isel(time=ii))
        best_inds = best_inds[np.argsort(order)]
        if ii == 0:
            soi_prediction = targets.isel(sample=best_inds).assign_coords({'time': ("time", [analog_indices['soi_time.year'][ii].data+leadtime]),'init_time': ("time", list(analog_indices['soi_time'][ii:ii+1].data))})
        else:
            soi_prediction = xr.concat([soi_prediction, targets.isel(sample=best_inds).assign_coords({'time': ("time", [analog_indices['soi_time.year'][ii].data+leadtime]),'init_time': ("time", list(analog_indices['soi_time'][ii:ii+1].data))})], dim='time')
    # now convert to global mean if requested (predictions)
    if global_mean:
        soi_prediction = soi_prediction.weighted(weights).mean(("lon", "lat"), skipna=True)
    
    # get the min and max at each time across the selected analogs    
    prediction_min = soi_prediction.min("sample", skipna=True)
    prediction_max = soi_prediction.max("sample", skipna=True)

    # return either the average of the analogs, or all of them still separated
    if return_average:
        # weight the analogs, if requested
        if type(rank_alpha) is float:
            analog_weights = 1/(np.arange(n_analogs) + 1).reshape(1, -1)
            soi_prediction = (soi_prediction * analog_weights**rank_alpha).mean('sample', skipna=True) / np.mean(analog_weights**rank_alpha, axis=1)
        else:
            soi_prediction = soi_prediction.mean('sample', skipna=True)

    return(truth.squeeze(), soi_prediction[variable].squeeze(), prediction_min[variable].squeeze(), prediction_max[variable].squeeze())

# -------------------------------------------------------------------------------

def get_full_analog_range(experiment_name, prediction_dict, soi_model, soi_member, variable,
                          global_mean=False, minmax=True, leadtime=None):
    if leadtime is None:
        leadtime = rexp.Experiments().get_experiment(experiment_name)['leadtime']
    # get the soi index and the full range of analog predictions
    if "global" in soi_model or "regional" in soi_model or "addregional" in soi_model:
        # return the experiment name without the "global", "regional", or "addregional" suffix
        suffix_ind = soi_model[::-1].find('_')
        soi_model = soi_model[:-(suffix_ind+1)]
    soi_index = np.where((prediction_dict['target'].model == soi_model) * 
                         (prediction_dict['target'].member == soi_member))[0]
    # get the indices for non soi targets
    all_inds = np.arange(prediction_dict['target'].sample.size)
    non_soi_inds = np.setdiff1d(all_inds, soi_index)
    # get the library minus the soi
    non_soi_analogs = prediction_dict['target'].isel(sample=non_soi_inds)
    # get the time range to return -- from the earliest soi, to the last prediction
    min_time, max_time = prediction_dict['target'].isel(sample=soi_index)['time.year'][[0, -1]] + [0, leadtime]
    time_inds = np.where((non_soi_analogs['time.year'] >= min_time) & (non_soi_analogs['time.year'] <= max_time))[0]
    non_soi_analogs = non_soi_analogs.isel(sample=time_inds).unstack()
    if global_mean:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        non_soi_analogs = non_soi_analogs.weighted(weights).mean(("lon", "lat"), skipna=True)
    if minmax:
        return(non_soi_analogs[variable].min(dim='member'), non_soi_analogs[variable].max(dim='member'))
    else:
        return(non_soi_analogs[variable].mean(dim='member'))

# -------------------------------------------------------------------------------

def get_target_region(experiment_name, prediction_dict):
    if prediction_dict is None:
        with gzip.open(os.path.join(os.path.dirname(__file__), 'saved_analogs', experiment_name, 'target.pickle'), 'rb') as file:
            prediction_dict = pickle.load(file)
    lat = prediction_dict['target'].lat
    lon = prediction_dict['target'].lon
    return(lat, lon)

# -------------------------------------------------------------------------------
# best analogs class
# -------------------------------------------------------------------------------

class BestAnalogs():
    """
    Class for processing analog library, selecting best analogs for a given SOI.
    """
    def __init__(self, experiment_name, library_name=None, model_list=None, n_members=None):
        # save the experiment_name, load the corresponding mask
        self.experiment_name = experiment_name
        analog_path = os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name)
        # make directory for saving analogs
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_analogs', experiment_name)):
            os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_analogs', experiment_name))
        # get experiment settings
        experiment_settings = rexp.Experiments().get_experiment(experiment_name)
        self.leadtime = experiment_settings['leadtime']
        self.mask_object = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc'))
        self.library_name = library_name
        
        # initialize the prediction_dict object (lat and lon from the training data)
        self.prediction_dict = {}
        self.prediction_dict['analogs'] = {}
        self.used_members = np.empty(0, dtype=int)
        self.used_models = np.empty(0, dtype=str)

        # construct or load a library of analogs
        if library_name is not None:
            # load inputs and targets from specified library
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", library_name, 'inputs.pickle'), "rb") as filename:
                self.prediction_dict['inputs'] = pickle.load(filename)
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", library_name, 'target.pickle'), "rb") as filename:
                self.prediction_dict['target'] = pickle.load(filename)
            print('loaded analog library from experiment called:', library_name)
        else:
            try:
                # check if processed prediction dataset file already exists by trying to load
                with gzip.open(analog_path+'/inputs.pickle', "rb") as filename:
                    self.prediction_dict['inputs'] = pickle.load(filename)
                with gzip.open(analog_path+'/target.pickle', "rb") as filename:
                    self.prediction_dict['target'] = pickle.load(filename)
            except:
                # if file does not exist, check that models and members have been passed
                assert (model_list is not None and n_members is not None), "no analog library, pass models and members to generate" 
                # start a new file by processing the models/members requested
                self.process_new_data(experiment_name, model_list, n_members)
                # and save the new prediction dataset
                with gzip.open(analog_path+'/inputs.pickle', "wb") as filename:
                    pickle.dump(self.prediction_dict['inputs'], filename)     
                with gzip.open(analog_path+'/target.pickle', "wb") as filename:
                    pickle.dump(self.prediction_dict['target'], filename)                
            else:
                # if try succeeded, we have a prediction dataset, but may want to add more data to it
                # add the used_members and use_models arrays with the prediction_dict (avoid saving the same data again)
                self.used_members = self.prediction_dict['inputs'].member.to_numpy()
                self.used_models = self.prediction_dict['inputs'].model.to_numpy()
                # call process_new_data with model_list and n_members which will add new data to the dataset
                if model_list is not None and n_members is not None:
                    self.process_new_data(experiment_name, model_list, n_members)
                    # and save the expanded prediction dataset
                    with gzip.open(analog_path+'/inputs.pickle', "wb") as filename:
                        pickle.dump(self.prediction_dict['inputs'], filename)
                    with gzip.open(analog_path+'/target.pickle', "wb") as filename:
                        pickle.dump(self.prediction_dict['target'], filename)

        # get lat and lon (global)
        self.prediction_dict['lat'] = self.prediction_dict['inputs']['lat']
        self.prediction_dict['lon'] = self.prediction_dict['inputs']['lon']
        
        # load any previously calculated analogs for specific sois
        try:
            analogs = glob.glob(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name, '*'))
            # drop the inputs and target from this list of files
            if np.any(['target.pickle' in files or 'inputs.pickle' in files for files in analogs]):
                analogs.remove(analog_path+'/inputs.pickle')
                analogs.remove(analog_path+'/target.pickle')
            print('previously saved analogs:', *[np.sort(analogs)[ii].split('/')[-1][:-7] for ii in range(len(analogs))])
            for analog in analogs:
                self.prediction_dict['analogs'][analog.split('/')[-1][:-7]] = {}
        except:
            print('no previouly saved analogs to load!')

# -------------------------------------------------------------------------------

    def process_new_data(self, experiment_name, model_list, new_members=-1):
        # get experiment settings
        experiment_settings = rexp.Experiments().get_experiment(experiment_name)
        variables = experiment_settings['variable_list']
        target_var = experiment_settings['target_var']
        # remove some of the processing steps that are not necessary for the prediction part
        # (sync_time not needed, shift_inout_time unnecessarily removes dates)
        remove_steps = ["sync_time(experiment_settings['time_range'])",
                        "shift_inout_time(-experiment_settings['leadtime'])",
                        "shift_inout_time(experiment_settings['leadtime'])",
                        "norm_data()"]
        for step in remove_steps:
            if np.isin(step, experiment_settings['inputs_proc_steps']):
                experiment_settings['inputs_proc_steps'].remove(step)
            if np.isin(step, experiment_settings['target_proc_steps']):
                experiment_settings['target_proc_steps'].remove(step)
        
        # option for doing all models
        if model_list == -1:
            models = glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data/*'))
            model_list = [os.path.basename(model) for model in models]
        
        # load the new raw data
        for ii, model in enumerate(model_list):
            # glob all the files and sort -- just used to get correct indices
            if 'max' in variables[0] or 'min' in variables[0]:
                var0 = variables[0][:-3]
            else:
                var0 = variables[0]
            files = np.sort(glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data', model, var0, '*')))
            # remove indices corresponding to model/members that are in this models used_members
            model_used_members = self.used_members[np.where(self.used_models == model)]
            model_new_members = np.setdiff1d(np.arange(files.size), model_used_members)
            # if -1 is passed, use all remaining members
            if new_members == -1:
                print('loading all available members, using', model_new_members.size, 'members from model:', model)
            # if more requested than available, use all remaining members
            elif model_new_members.size < new_members:
                print('*** Warning: more new members requested than available ***')
                print('loading all available, using', model_new_members.size, 'members from model:', model)
            # otherwise, return the number requested
            else:
                model_new_members = model_new_members[:new_members]

            # set up the data processing
            if model_new_members.size == 0:
                print("skipping "+model)
                continue
            input_raw = _dp.DataProcessing([model], variables, list(model_new_members), experiment_settings['seed'], experiment_settings['resolutions'])
            # get the targets
            if np.isin(target_var[0], variables):
                # deepcopy a version for the outputs
                output_raw = copy.deepcopy(input_raw)
                # variables that should be removed (non-target variables)
                remove_vars = np.setdiff1d(variables, target_var)
                # remove them
                for drop in remove_vars:
                    output_raw.data_dictionary.pop(drop)
            # target variable is not one of the input variables
            else:
                output_raw = _dp.DataProcessing([model], target_var, list(model_new_members), experiment_settings['seed'], experiment_settings['resolutions'])
                
            # run some processing steps for the inputs
            print('#################################################################')
            print('processing', model, 'input dataset')
            print('#################################################################')
            for step in experiment_settings['inputs_proc_steps']:
                eval('input_raw.'+step)
                print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            # run some processing steps for the targets
            print('#################################################################')
            print('processing', model, 'target dataset')
            print('#################################################################')
            for step in experiment_settings['target_proc_steps']:
                eval('output_raw.'+step)
                print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            # flatten variables into single dataset (if there is more than one variable)
            if len(variables) > 1:
                input_raw = xr.merge([input_raw.data_dictionary[var][model] for var in input_raw.data_dictionary.keys()], join='inner')
            else:
                input_raw = input_raw.data_dictionary[variables[0]][model]
            # now do the same for target
            output_raw = output_raw.data_dictionary[target_var[0]][model]

            # make a list of the model name repeating for the number of members
            repeat_model = [model] * model_new_members.size
            # add the member indices as a coordinate
            input_raw = input_raw.assign_coords(member=(model_new_members))
            output_raw = output_raw.assign_coords(member=(model_new_members))
            # and add the models as a coordinate
            input_raw = input_raw.assign_coords(model=('member', repeat_model))
            output_raw = output_raw.assign_coords(model=('member', repeat_model))
            
            # concatenate together the members across models
            if ii == 0:
                inputs_raw = input_raw
                outputs_raw = output_raw
            else:
                inputs_raw = xr.concat([inputs_raw, input_raw], dim="member")
                outputs_raw = xr.concat([outputs_raw, output_raw], dim="member")

        # re-type the data to 32 bit
        inputs_raw = inputs_raw.assign_coords(lat=inputs_raw['lat'].astype(FLOAT_TYPE),
                                              lon=inputs_raw['lon'].astype(FLOAT_TYPE),
                                              member=inputs_raw['member'].astype(INT_TYPE)).astype(FLOAT_TYPE)
        outputs_raw = outputs_raw.assign_coords(lat=outputs_raw['lat'].astype(FLOAT_TYPE),
                                                lon=outputs_raw['lon'].astype(FLOAT_TYPE),
                                                member=outputs_raw['member'].astype(INT_TYPE)).astype(FLOAT_TYPE)
        # compress dimensions time and member to sample
        inputs_raw = inputs_raw.stack(sample=("member", "time"))
        outputs_raw = outputs_raw.stack(sample=("member", "time"))
        # get rid of samples that are all nans
        good_inds = np.where(~np.isnan(inputs_raw[variables[0]].mean(['lat', 'lon'], skipna=True)))[0]
        inputs_raw = inputs_raw.isel(sample=good_inds)
        good_inds = np.where(~np.isnan(outputs_raw[target_var[0]].mean(['lat', 'lon'], skipna=True)))[0]
        outputs_raw = outputs_raw.isel(sample=good_inds)

        try:
            # concatenate this model/variable onto the existing self.prediction_dict["inputs"] and self.prediction_dict["target"]
            self.prediction_dict["inputs"] = xr.concat([self.prediction_dict["inputs"], inputs_raw], dim="sample")
            self.prediction_dict["target"] = xr.concat([self.prediction_dict["target"], outputs_raw], dim="sample")
        except:
            self.prediction_dict["inputs"] = inputs_raw
            self.prediction_dict["target"] = outputs_raw
    
# -------------------------------------------------------------------------------

    def _get_single_soi_analogs(self, data_iterable):
        """
        get indices ranking the analogs from best to worst for a single state of interest (soi),
        i.e. a single model, member, time or a single observation.
        
        WARNING: this does not automatically remove the SOI from the analog, better to use get_all_soi_analogs
        
        inputs:
            - data_iterable: a list with a single soi entry and all analogs (already weighted by mask),
            e.g. [weighted_soi, weighted_analogs].
        outputs:
            - soi_ds: a xarray DataSet containing the analog indices (member and time) sorted from best to worst,
            as well as the soi model name, soi member number, and soi time.
        """
        weighted_soi, weighted_analogs, weights = data_iterable

        mse = 0
        for var in list(weighted_soi.data_vars):
            # reshape analog and soi for faster computation (broadcasting)
            analogs = weighted_analogs[var].to_numpy()
            soi = weighted_soi.expand_dims('placeholder').transpose('trend', 'placeholder', 'lat', 'lon')[var].to_numpy()
            # calculate mse and sort by this
            if np.product(analogs.shape) < 2.5e9: # size of matrix is small enough to broadcast
                mse += np.mean(bn.nanmean(bn.nanmean((analogs - soi)**2, axis=3), axis=2) * weights, axis=0)
            else:
                # for each tether -- reduce matrix size
                for tt in range(np.shape(soi)[0]):
                    mse += bn.nanmean(bn.nanmean((analogs[tt] - soi[tt])**2, axis=2), axis=1) * weights[tt]
        best_samples = np.argsort(mse)

        # keep only the last time in the trend -- the actual map to predict from
        weighted_analogs = weighted_analogs.isel(trend=-1)

        # create a DataSet for this soi with: soi model, member, and time coordinates,
        # and analog time sorted indices, analog member & model sorted indices as data_variables
        soi_ds = xr.Dataset(
            data_vars=dict(
                best_input_times=(["time", "analogs"], weighted_analogs.time[best_samples].data.reshape(1, -1)),
                best_input_members=(["time", "analogs"], weighted_analogs.member[best_samples].data.reshape(1, -1).astype(INT_TYPE)),
                best_input_models=(["time", "analogs"], weighted_analogs.model[best_samples].data.reshape(1, -1)),
            ),
            coords=dict(
                soi_time=("time", list(weighted_soi.time[-1:].data))
                    )
        )
        return(soi_ds)
        
# -------------------------------------------------------------------------------
    
    def get_all_soi_analogs(self, soi_model, soi_member, tether=1, savename=None,
                            quant=None, mask_type='weights', leadtime=None):
        """
        multiprocessing for loop to get all soi analogs for an observational dataset or single model/member,
        i.e. the full time series of sois for a given model/member or observational dataset
        """
        assert tether >= 1, "must match on at least one time, minimum tether value is 1"
        if savename is None:
            savename = soi_model + '_mem' + str(soi_member) + '_teth' + str(tether)
            if self.library_name is not None:
                savename = savename + '_library_' + self.library_name
        # check that this analog is not already saved
        analog_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name, savename+'.pickle'))
        assert not analog_exists, 'analog with savename ' + savename + '.pickle exists -- to replace, delete the current file and re-run'
        # option to use a different leadtime (especially for trajectory learned masks)
        if leadtime is None:
            leadtime = self.leadtime

        mask_object = self.mask_object
        vars = list(self.prediction_dict['inputs'].data_vars)
        # remove mask values below some quantile threshhold
        if quant is not None:
            quant_mask = self.mask_object.quantile(quant)
            mask_object = mask_object.where(mask_object >= quant_mask, 0)
        # use a global 1s mask or a regional 1s mask
        if mask_type == 'global':
            mask_object = mask_object.where(mask_object < -9999, 1).astype(FLOAT_TYPE)
        elif mask_type == 'regional':
            mask_object = (
                (mask_object.lat <= self.prediction_dict['target'].lat.max()) *
                (mask_object.lat >= self.prediction_dict['target'].lat.min()) *
                (mask_object.lon <= self.prediction_dict['target'].lon.max()) *
                (mask_object.lon >= self.prediction_dict['target'].lon.min())
                ).astype(FLOAT_TYPE)
            mask_object = xr.merge([mask_object.to_dataset(name=var) for var in vars])
        
        # check if the requested SOI is in the prediction_dict yet -- for now, it has to be, otherwise
        # I will have to add a call to process_new_data that bypasses some of the concatenation, etc.
        try:
            soi_index = np.where((self.prediction_dict['inputs'].model == soi_model) *
                                 (self.prediction_dict['inputs'].member == soi_member))[0]
            soi_inputs = self.prediction_dict['inputs'].isel(sample=soi_index) * mask_object
        except:
            print('that SOI is not in the already processed prediction_dict')
            
        # now remove this particular soi member from the analog library
        analog_inds = np.setdiff1d(np.arange(self.prediction_dict['inputs'].member.size), soi_index)
        weighted_analogs = self.prediction_dict['inputs'].isel(sample=analog_inds) * mask_object
        
        # empty gets filled, so need to remove the first entry before using
        good_inds = np.empty([1, tether], dtype=int)
        # remove the last leadtime years of the inputs for each model/member (no corresponding target)
        for model in np.unique(weighted_analogs.model):
            for member in np.unique(weighted_analogs.isel(sample=np.where(weighted_analogs.model == model)[0]).member):
                # get each model/member to remove the last leadtime entries and rearrange to be trends for matching
                cinds = np.where((weighted_analogs.model == model) & (weighted_analogs.member == member))[0][:-leadtime]
                cinds = np.lib.stride_tricks.sliding_window_view(cinds, tether)
                good_inds = np.concatenate([good_inds, cinds])
        good_inds_xr = xr.DataArray(data=good_inds[1:], dims=['sample', 'trend'])
        weighted_analogs = weighted_analogs.isel(sample=good_inds_xr).transpose('trend', 'sample', 'lat', 'lon')

        # do the same for soi
        good_inds_xr = xr.DataArray(data=np.lib.stride_tricks.sliding_window_view(np.arange(soi_inputs.sample.size), tether), dims=['sample', 'trend'])
        soi_inputs = soi_inputs.isel(sample=good_inds_xr).transpose('trend', 'sample', 'lat', 'lon')
        
        # weights for tethering
        # tether_weights = np.array([1.]*tether).reshape(-1, 1).astype(np.float32)
        tether_weights = 1/(np.arange(tether, 0, -1)).reshape(-1, 1).astype(np.float32)

        for ii in range(soi_inputs.dims['sample']):
            if ii == 0:
                soi_ds = self._get_single_soi_analogs([soi_inputs.isel(sample=ii), weighted_analogs, tether_weights])
            else:
                soi_ds = xr.concat([soi_ds, self._get_single_soi_analogs([soi_inputs.isel(sample=ii), weighted_analogs, tether_weights])], dim='time')

        # save it now
        with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name, savename+'.pickle'), "wb", compresslevel=3) as filename:
            print('saving to:', os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name, savename+'.pickle'))
            pickle.dump(soi_ds, filename)

# -------------------------------------------------------------------------------

