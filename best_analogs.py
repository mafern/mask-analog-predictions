import numpy as np
import pickle
import xarray as xr
import multiprocessing as mp
import glob
import pandas as pd
import _data_processing as _dp
import run_experiments as rexp
import os
import gzip
import copy

# TODO: explore multiple masks -- different leadtimes to get a coherent time series (starting from a single soi time rather than rolling that forward and projecting 1 year over and over) -- tethering
#### REWRITE THIS SO THAT INDICES ARE SAVED SEPARATELY -- TAKES A LOT OF TIME TO RESAVE THE INPUTS AND TARGETS WHICH DON'T CHANGE (library file and analog file)

# -------------------------------------------------------------------------------
# general use functions
# -------------------------------------------------------------------------------

def get_targets(experiment_name, soi_model, soi_member, variable, n_analogs, prediction_dict=None, soi_index=None,
                return_average=True, global_mean=False, max_year=None, rank_alpha=None):
    """
    get the best analogs for the given soi model/member
    """
    # load the prediction_dict if none is passed
    if prediction_dict is None:
        with gzip.open(os.path.join(os.path.dirname(__file__), 'saved_analogs', experiment_name+'.pickle'), 'rb') as file:
            prediction_dict = pickle.load(file) 
    # make sure there are analogs to get targets from
    assert prediction_dict['indices'], "no soi analogs calculated yet"
    # get the soi_index corresponding to the requested soi_model and soi_member for the indices,
    # or directly if the soi_index argument is not None
    if soi_index is None:
        # get the soi based on model/member and take the most recent one (max)
        soi_index = np.max(np.where((prediction_dict['indices'].soi_model == soi_model) * 
                                    (prediction_dict['indices'].soi_member == soi_member))[0])

    # get the indices for the selected soi
    soi_indices = prediction_dict['indices'].isel(soi_index=soi_index)

    # weights and leadtime
    weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
    leadtime = rexp.Experiments().get_experiment(experiment_name)['leadtime']

    # get the soi_ind corresponding to the requested soi_model and soi_member for the inputs and target
    if "global" in soi_model or "regional" in soi_model or "mspecial" in soi_model:
        # return the experiment name without the "global", "regional", or "mspecial" suffix
        suffix_ind = soi_model[::-1].find('_')
        soi_model = soi_model[:-(suffix_ind+1)]
    soi_index = np.where((prediction_dict['target'].model == soi_model) &
                         (prediction_dict['target'].member == soi_member))[0]
    
    # get the soi (aka the truth))
    truth = prediction_dict['target'].isel(member=soi_index)[variable]
    if global_mean:
        truth = truth.weighted(weights).mean(("lon", "lat"), skipna=True)
    # get times when the truth is not nan
    times = np.unique(np.where(~np.isnan(truth))[0])
    truth = truth[times]
    # keep only the times where the truth is not nans (targets will extend leadtime further than this)
    soi_indices = soi_indices.isel(time=times)
    # get the target selection variables, either unconstrained or constrained to be earlier than max_year
    if max_year is not None:
        # separate out the data that will select the correct targets
        best_analog_times = soi_indices.best_analog_times
        best_analog_members = soi_indices.best_analog_members
        # make a boolean matrix of the analogs that are from earlier than max_year
        max_year_bool = soi_indices['best_input_times.year'] < max_year
        # need to go through each time, since these occur in different locations (then drop these)
        bat_list = [best_analog_times.isel(time=ii).where(max_year_bool.isel(time=ii)).dropna('analog_size') for ii in range(times.size)]
        bam_list = [best_analog_members.isel(time=ii).where(max_year_bool.isel(time=ii)).dropna('analog_size') for ii in range(times.size)]
        # keep only the number of analogs requested
        best_analog_times = xr.concat(bat_list, dim='time').isel(analog_size=np.arange(n_analogs)).astype(int)
        best_analog_members = xr.concat(bam_list, dim='time').isel(analog_size=np.arange(n_analogs)).astype(int)
    else:
        # separate out the data that will select the correct targets and keep only the number of analogs requested
        best_analog_times = soi_indices.best_analog_times.isel(analog_size=np.arange(n_analogs))
        best_analog_members = soi_indices.best_analog_members.isel(analog_size=np.arange(n_analogs))

    # get the prediction for the SOI
    soi_prediction = prediction_dict['target'].isel(member=best_analog_members, time=best_analog_times+leadtime)
    # add prediction time
    new_time_range = pd.date_range(start=str(soi_prediction['soi_time.year'].min().data + leadtime)[:4] + str(soi_prediction['soi_time'].min().data)[4:], 
                                   end=str(soi_prediction['soi_time.year'].max().data + leadtime)[:4] + str(soi_prediction['soi_time'].max().data)[4:], freq='1Y')
    soi_prediction['time'] = new_time_range
    soi_prediction = soi_prediction[variable].convert_calendar("noleap", dim='time')
    # add selected analog time
    soi_prediction['analog_time'] = soi_indices.best_input_times.isel(analog_size=np.arange(n_analogs))
    # now convert to global mean if requested (predictions)
    if global_mean:
        soi_prediction = soi_prediction.weighted(weights).mean(("lon", "lat"), skipna=True)
    
    # get the min and max at each time across the selected analogs    
    prediction_min = soi_prediction.min("analog_size", skipna=True)
    prediction_max = soi_prediction.max("analog_size", skipna=True)

    # return either the average of the analogs, or all of them still separated
    if return_average:
        # get the average of the analogs, if requested
        # TODO: weighted mean option as power law?
        if rank_alpha is None:
            soi_prediction = soi_prediction.mean('analog_size', skipna=True)
        else:
            soi_prediction = soi_prediction.mean('analog_size', skipna=True)

    return(truth.squeeze(), soi_prediction.squeeze(), prediction_min.squeeze(), prediction_max.squeeze())

# -------------------------------------------------------------------------------

def get_full_analog_range(prediction_dict, soi_model, soi_member, variable, global_mean=False, minmax=True):
    # get the soi index and the full range of analog predictions
    if "global" in soi_model or "regional" in soi_model or "mspecial" in soi_model:
        # return the experiment name without the "global", "regional", or "mspecial" suffix
        suffix_ind = soi_model[::-1].find('_')
        soi_model = soi_model[:-(suffix_ind+1)]
    soi_index = np.where((prediction_dict['target'].model == soi_model) * 
                         (prediction_dict['target'].member == soi_member))[0]
    # get the soi (aka the truth))
    truth = prediction_dict['target'].isel(member=soi_index)[variable]
    # get times when the truth is not nan
    times = np.unique(np.where(~np.isnan(truth))[0])
    all_inds = np.arange(prediction_dict['target'].member.size)
    non_soi_inds = np.setdiff1d(all_inds, soi_index)
    non_soi_analogs = prediction_dict['target'].isel(member=non_soi_inds, time=times)
    if global_mean:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        non_soi_analogs = non_soi_analogs.weighted(weights).mean(("lon", "lat"), skipna=True)
    if minmax:
        return(non_soi_analogs[variable].min(dim='member'), non_soi_analogs[variable].max(dim='member'))
    else:
        return(non_soi_analogs[variable].mean(dim='member'))

# -------------------------------------------------------------------------------

def get_target_region(experiment_name, prediction_dict=None):
    # load the prediction_dict if none is passed
    if prediction_dict is None:
        with gzip.open(os.path.join(os.path.dirname(__file__), 'saved_analogs', experiment_name+'.pickle'), 'rb') as file:
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
    def __init__(self, experiment_name, model_list=None, n_members=None, data_period=None):
        # save the experiment_name, load the corresponding mask
        self.experiment_name = experiment_name
        self.mask_object = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc'))
        
        # get the training and validation data so they can be avoided in the soi and analogs
        self.used_members = np.empty(0, dtype=int)
        self.used_models = np.empty(0, dtype=str)
        for split in ['training', 'validation']:
            with gzip.open(os.path.join(os.path.dirname(__file__), "processed_data", experiment_name+'_'+split+'.pickle'), "rb") as filename:
                # just getting the models and members, so either inputs or target is sufficient
                split_data = pickle.load(filename)
            # concatenate the training data used_models and used_members
            self.used_members = np.concatenate([self.used_members, split_data['inputs'].member.to_numpy()])
            self.used_models = np.concatenate([self.used_models, split_data['inputs'].model.to_numpy()])
        
        # initialize the prediction_dict object (lat and lon from the training data)
        self.prediction_dict = {}
        self.prediction_dict['lat'] = split_data['lat']
        self.prediction_dict['lon'] = split_data['lon']
            
        try:
            # check if processed prediction dataset file already exists by trying to load
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "rb") as filename:
                self.prediction_dict = pickle.load(filename)
        except:
            # if file does not exist, check that models and members have been passed
            assert (model_list is not None and n_members is not None), "no analog library, pass models and members to generate" 
            # start a new file by processing the models/members requested
            self.process_new_data(experiment_name, model_list, n_members, data_period)
            # and save the new prediction dataset
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "wb") as filename:
                pickle.dump(self.prediction_dict, filename)                
        else:
            # if try succeeded, we have a prediction dataset, but may want to add more data to it
            # add the used_members and use_models arrays with the prediction_dict (avoid saving the same data again)
            self.used_members = np.concatenate([self.used_members, self.prediction_dict['inputs'].member.to_numpy()])
            self.used_models = np.concatenate([self.used_models, self.prediction_dict['inputs'].model.to_numpy()])
            # call process_new_data with model_list and n_members which will add new data to the dataset
            if model_list is not None and n_members is not None:
                self.process_new_data(experiment_name, model_list, n_members, data_period)
                # and save the expanded prediction dataset
                with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "wb") as filename:
                    pickle.dump(self.prediction_dict, filename)

        # check if analogs have been selected for any sois
        try:
            print('previously saved analogs for:', *self.prediction_dict['indices'].soi_model.data)
        except:
            print('no previouly saved analogs to load!')

# -------------------------------------------------------------------------------

    def process_new_data(self, experiment_name, model_list, new_members=-1, data_period=None):
        # n_members=-1 means return all the rest of the members from each model
        
        # get experiment settings
        experiment_settings = rexp.Experiments().get_experiment(experiment_name)
        variables = experiment_settings['variable_list']
        # if no data_period given, default to previously used
        if data_period is None:
            data_period = experiment_settings['data_period']
        # remove some of the processing steps that are not necessary for the prediction part
        # (sync_time not needed, shift_inout_time unnecessarily removes dates, regional_mean can be done later)
        remove_steps = ["sync_time(experiment_settings['time_range'])",
                        "shift_inout_time(-experiment_settings['leadtime'], experiment_settings['window'])",
                        "shift_inout_time(experiment_settings['leadtime'], experiment_settings['window'])",
                        "regional_mean(experiment_settings['variable_list'])",
                        "norm_data()"]
        for step in remove_steps:
            if np.isin(step, experiment_settings['inputs_proc_steps']):
                experiment_settings['inputs_proc_steps'].remove(step)
            if np.isin(step, experiment_settings['target_proc_steps']):
                experiment_settings['target_proc_steps'].remove(step)
        
        # load the new raw data
        for ii, model in enumerate(model_list):
            # glob all the files and sort -- just used to get correct indices
            files = np.sort(glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data', data_period, model, variables[0], '*')))

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
            input_raw = _dp.DataProcessing([model], variables, data_period, list(model_new_members), experiment_settings['seed'])
            for var in variables:
                # make the resolution the same as the previously processed datasets
                input_raw.attr_dictionary[var]['res'] = [np.diff(self.prediction_dict['lat']).mean(), np.diff(self.prediction_dict['lon']).mean()]
            output_raw = copy.deepcopy(input_raw)

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
                output_raw = xr.merge([output_raw.data_dictionary[var][model] for var in output_raw.data_dictionary.keys()], join='inner')
            else:
                input_raw = input_raw.data_dictionary[variables[0]][model]
                output_raw = output_raw.data_dictionary[variables[0]][model]

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

        try:
            # concatenate this model/variable onto the existing self.prediction_dict["inputs"] and self.prediction_dict["target"]
            self.prediction_dict["inputs"] = xr.concat([self.prediction_dict["inputs"], inputs_raw], dim="member")
            self.prediction_dict["target"] = xr.concat([self.prediction_dict["target"], outputs_raw], dim="member")
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
        weighted_soi, weighted_analogs = data_iterable

        # get the mse between the single soi and the library of analogs (both weighted)
        mse = ((weighted_analogs - weighted_soi)**2).mean(dim=('lat', 'lon'), skipna=True).to_array().to_numpy().squeeze()
        
        if mse.ndim == 3:
            # if there are multiple variables, take mean over mse for all variables
            mse = mse.mean(axis=0)
        # argsort then get corresponding indices for best members and times
        best_times, best_members = np.dstack(np.unravel_index(np.argsort(mse.ravel()), mse.shape)).squeeze().T

        # create a DataSet for this soi with: soi model, member, and time coordinates,
        # and analog time sorted indices, analog member & model sorted indices as data_variables
        soi_ds = xr.Dataset(
            data_vars=dict(
                best_analog_times=(["soi_index", "time", "analog_size"], best_times.reshape(1, 1, -1)),
                best_analog_members=(["soi_index", "time", "analog_size"], best_members.reshape(1, 1, -1)),
                best_input_times=(["soi_index", "time", "analog_size"], weighted_analogs.time[best_times].data.reshape(1, 1, -1)),
                best_input_members=(["soi_index", "time", "analog_size"], weighted_analogs.member[best_members].data.reshape(1, 1, -1).astype(int)),
                best_input_models=(["soi_index", "time", "analog_size"], weighted_analogs.model[best_members].data.reshape(1, 1, -1))
            ),
            coords=dict(
                soi_model=("soi_index", [self.soi_model_name]),
                soi_member=("soi_index", [weighted_soi.member.data]),
                soi_time=("time", [pd.to_datetime(str(weighted_soi.time.dt.strftime("%Y-%m-%d").to_numpy()))])
                  )
        )
        
        return(soi_ds)
        
# -------------------------------------------------------------------------------
    
    def get_all_soi_analogs(self, soi_model, soi_member, n_proc=None, quant=None, mask_type='weights'):
        """
        multiprocessing for loop to get all soi analogs for an observational dataset or single model/member,
        i.e. the full time series of sois for a given model/member or observational dataset
        """
        # get experiment settings
        experiment_settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = experiment_settings['leadtime']
        # use a threshold for the mask to remove low relevance regions
        mask_object = self.mask_object
        # default soi name (can be updated if using global 1s map, or regional 1s map)
        self.soi_model_name = soi_model
        # remove mask values below some quantile threshhold
        if quant is not None:
            quant_mask = self.mask_object.pr.quantile(quant)
            mask_object = mask_object.where(mask_object >= quant_mask, 0)
        # use a global 1s mask or a regional 1s mask
        if mask_type == 'global':
            mask_object = mask_object.where(mask_object < -9999, 1)
            self.soi_model_name = soi_model+'_global'
        elif mask_type == 'regional':
            mask_object = ((mask_object.lat <= self.prediction_dict['target'].lat.max())
                           * (mask_object.lat >= self.prediction_dict['target'].lat.min())
                           * (mask_object.lon <= self.prediction_dict['target'].lon.max())
                           * (mask_object.lon >= self.prediction_dict['target'].lon.min())).astype(int)
            self.soi_model_name = soi_model+'_regional'
        elif mask_type == 'mspecial':
            # keep the weights outside the target the same, but replace values inside the target
            # region with the maximum value across the whole mask
            max_mask_weight = mask_object.max()
            bool_map = ((mask_object.lat <= self.prediction_dict['target'].lat.max())
                        * (mask_object.lat >= self.prediction_dict['target'].lat.min())
                        * (mask_object.lon <= self.prediction_dict['target'].lon.max())
                        * (mask_object.lon >= self.prediction_dict['target'].lon.min()))
            mask_object = mask_object.where(~bool_map, max_mask_weight)
            self.soi_model_name = soi_model+'_mspecial'


        # check if the requested SOI is in the prediction_dict yet -- for now, it has to be, otherwise
        # I will have to add a call to process_new_data that bypasses some of the concatenation, etc.
        try:
            soi_index = np.where((self.prediction_dict['inputs'].model == soi_model) *
                                 (self.prediction_dict['inputs'].member == soi_member))[0][0]
            soi_inputs = self.prediction_dict['inputs'].isel(member=soi_index) * mask_object
        except:
            print('that SOI is not in the already processed prediction_dict')
            
        # now remove this particular soi member from the analog library
        analog_inds = np.setdiff1d(np.arange(self.prediction_dict['inputs'].member.size), soi_index)
        weighted_analogs = self.prediction_dict['inputs'].isel(member=analog_inds) * mask_object

        # these steps ensure that inputs with corresponding targets are not picked (nan, so they move to the end of the best matches)
        # nan the last leadtime years of the inputs (no corresponding target)
        max_year = soi_inputs['time.year'][-1].data
        weighted_analogs = weighted_analogs.where(weighted_analogs['time.year'] < max_year - leadtime)
        # grab the first variable name (assumes all variables have the same nans)
        var = list(self.prediction_dict['inputs'].keys())[0]
        # find the times/members when the entire entry is nan for the target
        target_nan_grid = np.roll(self.prediction_dict['target'].isel(member=analog_inds)[var].mean(['lat', 'lon'], skipna=True).to_numpy(), -leadtime, axis=0)
        target_nan_grid = ~np.isnan(target_nan_grid.reshape(target_nan_grid.shape + (1, 1)))
        # make the inputs nan in the corresponding locations where the target is nan
        weighted_analogs = weighted_analogs.where(target_nan_grid)

        # set up multiprocessing to get the best analogs for this soi (all times)
        if n_proc is None:
            # use all available processors
            n_proc = mp.cpu_count()
        with mp.Pool(n_proc) as pool:
            # make an iterable function so it can run using mulitprocessing
            data_iterable = self.soi_iterable(soi_inputs, weighted_analogs)
            # chuncksize is number of states (which is number of times) divided by requested processes
            chunksize = int(soi_inputs.time.size / n_proc)
            # this will iterate through and make a list of results
            result = pool.map(self._get_single_soi_analogs, data_iterable, chunksize=chunksize)
            
        # now combine all the soi results into a single DataSet
        for ii in range(len(result)):
            if ii == 0:
                soi_ds = result[ii]
            else:
                soi_ds = xr.concat([soi_ds, result[ii]], dim='time')

        # combine the best analogs with the prediction_dict -- note that this has all the
        # targets, including the SOI, so this will have one entry more than the indices
        self.prediction_dict['indices'] = soi_ds
        
        try:
            # if file already exists, concatenate onto existing by the soi_index dimension
            failure_reason = "failed to load existing analogs\n" + "file path: " + os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle')
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "rb") as filename:
                prediction_dict = pickle.load(filename)
                self.prediction_dict['indices'] = prediction_dict['indices']
            # limit the number of analogs returned to the number from the first calculation
            # otherwise, it will try to concatenate different sized analog indices (since
            # more data could have been added) -- should avoid doing this either way, i.e.
            # should have the full analog set determined before running get_all_soi_analogs
            max_analogs = self.prediction_dict['indices'].dims['analog_size']
            failure_reason = "failed to concatenate new data with existing\n\n"+"new data:\n"+str(soi_ds)+"\n\nexisting data:\n"+str(self.prediction_dict['indices'])
            self.prediction_dict['indices'] = xr.concat([self.prediction_dict['indices'], soi_ds.isel(analog_size=np.arange(max_analogs))], dim='soi_index')
        except:
            print(failure_reason)
        # and save it
        with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "wb") as filename:
            print('saving to:', os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'))
            pickle.dump(self.prediction_dict, filename)
        
        return(self.prediction_dict)
      
# -------------------------------------------------------------------------------
    
    def soi_iterable(self, weighted_soi, weighted_analogs):
        # make data iterable for use in multiprocessing pool
        """
        Create an iterable for a parallel approach to metric assessment
        """
        for jj in range(weighted_soi.time.size):
            yield weighted_soi.isel(time=jj), weighted_analogs

# -------------------------------------------------------------------------------
