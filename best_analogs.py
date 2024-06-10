import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.colors as clr
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

# TODO: confirm that all times have been updated to prediction time
# TODO: compare the no mask results (global 1s) to our mask and to a regional mask, and to ensemble mean
# TODO: figure out a good way to reuse libraries (avoid reprocessing if possible) -- same for mask training
# TODO: explore multiple masks -- different leadtimes to get a coherent time series (starting from a single soi time rather than rolling that forward and projecting 1 year over and over)
# TODO: try earth mover's distance to determine quality of prediction
# TODO: add save option for predictions (instead of getting the targets every time)
# TODO: add function to use a thresholded mask, probably by percentile
# TODO: add some extremes functions (or let McKenna do this)


# define some colors and cmaps -- nuggets
c_flatirons = '#8B2131'
c_flatirons_l = '#c22e44'
c_flatirons_ll = '#d85b6e'
c_sunshine_o = '#FEC524'
c_sunshine = '#CA9500'
c_sunshine_l = '#ffbe09'
c_sunshine_ll = '#ffcf46'
c_skyline = '#1D428A'
c_skyline_l = '#295dc3'
c_skyline_ll = '#5583db'
c_midnight = '#0E2240'
red_skyline = ['#FFC72C', '#FF8200', '#FC4C02', '#DC4405', '#862633']
red_cmap = clr.LinearSegmentedColormap.from_list('red skyline', red_skyline, N=256)
# set up an alpha colorbar
alpha_colors = []
for ii in range(0, 11):
    alpha = np.sin(ii/10*np.pi/2)
    alpha_colors.append((1.0, 0.6445674740484428, 0.08593617839292579, alpha))
alpha_cmap = clr.LinearSegmentedColormap.from_list('alpha cmap', alpha_colors, N=11)

# csu colors
c_ovalgreen = '#006144'
c_loverslane = '#82C503'
c_energygreen = '#CFFC00'
c_flowertrial = '#E56A54'
c_powered = '#7E5475'
c_horsetooth = '#008FB3'
c_stalwart = '#105456'
c_csusun = '#FFC038'
c_csugreen = '#1E4D2B'
csu_greens = [c_csugreen, c_ovalgreen, c_loverslane, c_energygreen, '#FFFFFF'][::-1]
csu_cmap = clr.LinearSegmentedColormap.from_list('csu greens', csu_greens, N=256)

# set some plotting defaults
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.25
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.75
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.25
plt.rcParams['axes.linewidth'] = 2

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

class BestAnalogs():
    """
    Class for processing analog library, selecting best analogs for a given SOI.
    """
    def __init__(self, experiment_name, model_list=None, n_members=None, data_period=None, fig_path=None):
        if fig_path is None:
            fig_path = 'figures/'
        self.fig_path = fig_path

        # save the experiment_name, load the corresponding mask
        self.experiment_name = experiment_name
        self.mask_object = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc'))
        
        # get the training, validation, and testing data so they can be avoided in the soi and analogs
        self.used_members = np.empty(0, dtype=int)
        self.used_models = np.empty(0, dtype=str)
        for split in ['training', 'validation', 'testing']:
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
            input_raw = _dp.DataProcessing([model], variables, data_period, ensembles=model_new_members)
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
        mse = ((weighted_analogs - weighted_soi)**2).mean(dim=('lat', 'lon')).to_array().to_numpy().squeeze()
        if mse.ndim == 3:
            # if there are multiple variables, take mean over mse for all variables
            mse = mse.mean(axis=0)
        # argsort then get corresponding indices for best members and times
        best_times, best_members = np.dstack(np.unravel_index(np.argsort(mse.ravel()), mse.shape)).squeeze().T

        # create a DataSet for this soi with: soi model, member, and time coordinates,
        # and analog time sorted indices, analog member & model sorted indices as data_variables
        soi_ds = xr.Dataset(
            data_vars=dict(
                best_input_times=(["soi_index", "time", "analog_size"], weighted_analogs.time[best_times].data.reshape(1, 1, -1)),
                best_analog_times=(["soi_index", "time", "analog_size"], best_times.reshape(1, 1, -1)),
                best_analog_members=(["soi_index", "time", "analog_size"], weighted_analogs.member[best_members].data.reshape(1, 1, -1).astype(int)),
                best_analog_models=(["soi_index", "time", "analog_size"], weighted_analogs.model[best_members].data.reshape(1, 1, -1))
            ),
            coords=dict(
                soi_model=("soi_index", [weighted_soi.model.data]),
                soi_member=("soi_index", [weighted_soi.member.data]),
                soi_time=("time", [pd.to_datetime(str(weighted_soi.time.dt.strftime("%Y-%m-%d").to_numpy()))])
                  )
        )
        
        return(soi_ds)
        
# -------------------------------------------------------------------------------
    
    def get_all_soi_analogs(self, soi_model, soi_member, n_proc=None):
        """
        multiprocessing for loop to get all soi analogs for an observational dataset or single model/member,
        i.e. the full time series of sois for a given model/member or observational dataset
        """
        # check if the requested SOI is in the prediction_dict yet -- for now, it has to be, otherwise
        # I will have to add a call to process_new_data that bypasses some of the concatenation, etc.
        try:
            soi_index = np.where((self.prediction_dict['inputs'].model == soi_model) *
                                 (self.prediction_dict['inputs'].member == soi_member))[0][0]
            soi_inputs = self.prediction_dict['inputs'].isel(member=soi_index) * self.mask_object
        except:
            print('that SOI is not in the already processed prediction_dict')
            
        # now remove this particular soi member from the analog library
        analog_inds = np.setdiff1d(np.arange(self.prediction_dict['inputs'].member.size), soi_index)
        weighted_analogs = self.prediction_dict['inputs'].isel(member=analog_inds) * self.mask_object

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
            # self.prediction_dict['targets'] = self.prediction_dict['target']
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

    def get_full_analog_range(self, soi_model, soi_member, variable, global_mean=False, minmax=True):
        # get the soi index and the full range of analog predictions
        soi_index = np.where((self.prediction_dict['target'].model == soi_model) * 
                             (self.prediction_dict['target'].member == soi_member))[0]
        all_inds = np.arange(self.prediction_dict['target'].member.size)
        non_soi_inds = np.setdiff1d(all_inds, soi_index)
        non_soi_analogs = self.prediction_dict['target'].isel(member=non_soi_inds)
        if global_mean:
            weights = np.cos(np.deg2rad(self.prediction_dict['target'].lat))
            non_soi_analogs = non_soi_analogs.weighted(weights).mean(("lon", "lat"), skipna=True)
        if minmax:
            return(non_soi_analogs[variable].min(dim='member').data, non_soi_analogs[variable].max(dim='member').data)
        else:
            return(non_soi_analogs[variable].mean(dim='member').data)
        
# -------------------------------------------------------------------------------
    
    def get_targets(self, soi_model, soi_member, variable, n_analogs, return_average=True, leadstep=1, global_mean=False, max_year=None):
        """
        get the best analogs for the given soi model/member, skipping every leadstep (number of steps
        in 'internal' units, meaning consecutive time steps from the processed data)
        
        returns the targets for both the analogs, and the soi (will be different shapes if leadstep != 1)
        """
        # make sure there are analogs to get targets from
        assert self.prediction_dict['indices'], "no soi analogs calculated yet, run self.get_all_soi_analogs(soi_model, soi_member, n_proc)"
        # get the soi_index corresponding to the requested soi_model and soi_member for the indices
        soi_index = np.where((self.prediction_dict['indices'].soi_model == soi_model) * 
                             (self.prediction_dict['indices'].soi_member == soi_member))[0]
        
        ntimes = self.prediction_dict['indices'].time.size
        weights = np.cos(np.deg2rad(self.prediction_dict['target'].lat))
        leadtime = rexp.Experiments().get_experiment(self.experiment_name)['leadtime']

        if max_year is None:
            max_year = self.prediction_dict['indices']['soi_time.year'][-1]

        # for the specified soi, get the corresponding targets
        soi_prediction, prediction_min, prediction_max = [], [], []
        # for each time step
        for soi_time_index in range(0, ntimes, leadstep):
            current_time_ds = self.prediction_dict['indices'].isel(time=soi_time_index, soi_index=soi_index)

            ii = 0
            step_analogs = []
            while np.array(step_analogs).shape[0] < n_analogs * leadstep:
                # try the next best analog
                next_best = current_time_ds.isel(analog_size=ii)
                # if there is enough time, get the corresponding target and append it
                # if next_best.best_analog_times <= ntimes - leadstep - leadtime:
                if next_best.best_analog_times <= ntimes - leadstep - leadtime and next_best['best_input_times.year'] <= max_year:
                    # get indices for member and time for the target dataset
                    member_ind = np.where(np.logical_and(self.prediction_dict['target'].model == next_best.best_analog_models,
                                                         self.prediction_dict['target'].member == next_best.best_analog_members))[0].item()
                    time_inds = np.arange(next_best.best_analog_times+leadtime, next_best.best_analog_times+leadtime+leadstep)
                    # and check if the target is all nans
                    proposed_target = self.prediction_dict['target'].isel(time=time_inds, member=member_ind)[variable]
                    if ~np.all(np.isnan(proposed_target)):
                        # now convert to global mean if requested (predictions)
                        if global_mean:
                            proposed_target = proposed_target.weighted(weights).mean(("lon", "lat"), skipna=True)
                        step_analogs.append(proposed_target.data)
                ii += 1
        
            if return_average:
                # get the average of the analogs, if requested
                soi_prediction.append(np.nanmean(step_analogs, axis=0))
            else:
                # otherwise return all of them separately
                soi_prediction.append(step_analogs)
            prediction_min.append(np.nanmin(step_analogs, axis=0))
            prediction_max.append(np.nanmax(step_analogs, axis=0))
            # print(soi_time_index, ii)

        # get the soi_ind corresponding to the requested soi_model and soi_member for the inputs and target
        soi_ind = np.where((self.prediction_dict['target'].model == soi_model) &
                           (self.prediction_dict['target'].member == soi_member))[0]
        # get the soi (aka the truth)
        # truth = self.prediction_dict['target'].isel(member=soi_ind, time=np.arange(ntimes))[variable]
        truth = self.prediction_dict['target'].isel(member=soi_ind)[variable]
        if global_mean:
            truth = truth.weighted(weights).mean(("lon", "lat"), skipna=True)

        # return(truth.data.squeeze(), np.squeeze(soi_prediction)[:ntimes], np.squeeze(prediction_min)[:ntimes], np.squeeze(prediction_max)[:ntimes])
        return(truth.data.squeeze(), np.squeeze(soi_prediction), np.squeeze(prediction_min), np.squeeze(prediction_max))

# -------------------------------------------------------------------------------
        
    def plot_prediction_suite(self, soi_model, soi_member, variable, n_analogs, leadstep=1, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_suite.pdf'
        # get the targets for the requested soi and variable
        truth, prediction, prediction_min, prediction_max = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]

        # get the full range of analog predictions and the model mean
        all_min, all_max = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True)
        all_model_mean = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True, minmax=False)

        # now make the figure
        fig, ax = plt.subplots(4, 1, figsize=(14,5*4))
        # --------------- plot the analog prediction and truth --------------- #
        # plot the true values (i.e. the soi target)
        ax[0].plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the prediction
        ax[0].plot(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction[:ntimes], '-', label='analog prediction ('+str(n_analogs)+')',
                 lw=2.66, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # --------------- plot the ranges and truth (refining spread) --------------- #
        # plot the range of library
        ax[1].fill_between(self.prediction_dict['target']['time.year'][:ntimes], all_min[:ntimes], all_max[:ntimes], color=c_skyline_ll,
                         alpha=0.33, label='model range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline_l)])
        # plot the range of all the analogs
        ax[1].fill_between(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction_min[:ntimes], prediction_max[:ntimes], color=c_sunshine,
                         alpha=0.3, label='analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_sunshine)])
        # plot the true values (i.e. the soi target)
        ax[1].plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # --------------- plot the model mean and truth --------------- #
        # plot the true values (i.e. the soi target)
        ax[2].plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the model mean of library
        ax[2].plot(self.prediction_dict['target']['time.year'][:ntimes], all_model_mean[:ntimes], label='model mean',
                color=c_skyline_l, lw=3, path_effects=[patheffects.withStroke(linewidth=4.66, foreground="k")])

        # set up the legend and axes for each subplot
        for ii in range(4):
            # legend, labels, etc.
            handles, labels = ax[ii].get_legend_handles_labels()
            handles = [copy.copy(ha) for ha in handles]
            if ii == 1:
                [ha.set_alpha(0.7) for ha in handles[:2]]
            leg = ax[ii].legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'size':'15', 'style':'oblique'})
            ax[ii].spines['right'].set_visible(False)
            ax[ii].spines['top'].set_visible(False)
            ax[ii].yaxis.set_ticks_position('left')
            ax[ii].xaxis.set_ticks_position('bottom')    
            ax[ii].patch.set_facecolor('none')
            ax[ii].tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10, labelsize=16)
        # x and y labels, title
        ax[3].set_xlabel('prediction year', fontsize=18)
        ax[0].set_title(settings['region']+', '+variable+': '+str(settings['leadtime'])+' year lead, '+str(settings['window'])+' year window', fontsize=18, fontweight='bold')
        # # add y label just once in the middle
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        # plt.ylabel(variable, fontsize=22, labelpad=15)

        plt.subplots_adjust(hspace=0.17)
        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
        
    def plot_range(self, soi_model, soi_member, variable, n_analogs, leadstep=1, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_range.pdf'
        # get the targets for the requested soi and variable
        truth, _, prediction_min, prediction_max = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]

        # get the full range of analog predictions
        all_min, all_max = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True)

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the range of library
        ax.fill_between(self.prediction_dict['target']['time.year'][:ntimes], all_min[:ntimes], all_max[:ntimes], color=c_skyline_ll,
                         alpha=0.33, label='model range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline_l)])
        # plot the range of all the analogs
        ax.fill_between(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction_min[:ntimes], prediction_max[:ntimes], color=c_sunshine,
                         alpha=0.3, label='analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_sunshine)])
        # plot the true values (i.e. the soi target)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(settings['region']+'\n'+variable, fontsize=18)
        ax.set_title(str(settings['leadtime'])+' year lead, '+str(settings['window'])+' year window', fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
        
    def plot_prediction(self, soi_model, soi_member, variable, n_analogs, leadstep=1, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_prediction.pdf'
        # get the targets for the requested soi and variable
        truth, prediction, _, _ = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the true values (i.e. the soi target)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the prediction
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction[:ntimes], '-', label='analog prediction ('+str(n_analogs)+')',
                 lw=2.66, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:1]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(settings['region']+'\n'+variable, fontsize=18)
        ax.set_title(str(settings['leadtime'])+' year lead, '+str(settings['window'])+' year window', fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
        
    def plot_model_mean(self, soi_model, soi_member, variable, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_model_mean.pdf'
        # get the targets for the requested soi and variable
        truth, _, _, _ = self.get_targets(soi_model, soi_member, variable, 1, 1, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]
        all_model_mean = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True, minmax=False)

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the true values (i.e. the soi target)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=0.75, c='white', label='truth',
                 path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the model mean of library
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes], all_model_mean[:ntimes], label='model mean',
                color=c_skyline_l, lw=3, path_effects=[patheffects.withStroke(linewidth=4.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:1]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(settings['region']+'\n'+variable, fontsize=18)
        ax.set_title(str(settings['leadtime'])+' year lead, '+str(settings['window'])+' year window', fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
# -------------------------------------------------------------------------------

    def plot_prediction_difference(self, soi_model, soi_member, variable, n_analogs, leadstep=1, max_year=2100, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_difference.pdf'
        # get the targets for the requested soi and variable
        truth, prediction, prediction_min, prediction_max = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep, max_year=max_year, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]

        # get the full range of analog predictions
        all_min, all_max = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True)

        # now make the plot
        fig, ax = plt.subplots(2, 1, figsize=(17, 8.5), height_ratios=[2, 1], sharex=True)

        # top panel -- truth, prediction, analog ranges
        # plot the range of all analogs
        ax[0].fill_between(self.prediction_dict['target']['time.year'][:ntimes], all_min[:ntimes], all_max[:ntimes], color=c_skyline,
                         alpha=0.4, label='all analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline)])
        # plot the range of all the best analogs
        ax[0].fill_between(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction_min[:ntimes], prediction_max[:ntimes], color=c_flatirons,
                         alpha=0.3, label='best analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_flatirons)])
        # plot the true values (i.e. the soi target)
        ax[0].plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=3.33, c=c_midnight, label='truth',
                 path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
        # plot the prediction
        ax[0].plot(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction[:ntimes], label='best analogs ('+str(n_analogs)+')',
                 lw=3.33, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])
        # top panel settings
        ax[0].set_title(self.experiment_name, fontsize=18)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].yaxis.set_ticks_position('left')
        ax[0].xaxis.set_ticks_position('bottom')    
        ax[0].patch.set_facecolor('none')
        ax[0].tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        # legend, labels, etc.
        handles, labels = ax[0].get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = ax[0].legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        # bottom panel -- difference between truth and prediction
        ax[1].plot(self.prediction_dict['target']['time.year'][:ntimes][:-leadtime]+leadtime, truth[:ntimes][leadtime:]-prediction[:ntimes][:-leadtime], lw=0.5, c='white',
                   label='truth $-$ best analogs', path_effects=[patheffects.withStroke(linewidth=5, foreground=c_midnight)])
        # legend, labels, etc.
        handles, labels = ax[1].get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = ax[1].legend(handles=handles, labels=labels, loc='upper left', ncols=1, frameon=False, prop={'weight':'bold', 'size':'15'})
        # bottom panel settings
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].yaxis.set_ticks_position('right')
        ax[1].patch.set_facecolor('none')
        ax[1].tick_params(which='both', direction='inout', left=False, labelright=True, labelleft=False, bottom=False, labelbottom=False, length=10)
        ax[1].grid(axis='y')

        # figure labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.ylabel(settings['region']+'\n'+variable, fontsize=20, labelpad=5)
        plt.xlabel('prediction year', size=20, labelpad=-30)

        plt.subplots_adjust(hspace=0.2)
        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_model_distribution(self, soi_model, soi_member, variable, n_analogs, savename=None):
        # plot the number of times each model was selected as a top n_analogs analog
        if savename is None:
            savename = self.experiment_name+'_model_dist.pdf'
        
        # get the index of the soi corresponding to the given model and member
        soi_index = np.where((self.prediction_dict['indices'].soi_model == soi_model) * 
                            (self.prediction_dict['indices'].soi_member == soi_member))[0][0]
        
        # get the targets for the requested soi and variable to restrict the times
        truth, _, _, _ = self.get_targets(soi_model, soi_member, variable, n_analogs=1, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]
        # get the models that were selected for the analogs of this soi
        selected_model = self.prediction_dict['indices'].isel(soi_index=soi_index).best_analog_models[:ntimes, :n_analogs]
        # get the names of the models that were selected for this soi across all times
        models = np.unique(selected_model)

        # sort by total number of selections for each model (high to low) -- for keeping it to top ten
        model_counts = np.unique(selected_model, return_counts=True)[1]
        sorted = np.argsort(model_counts)[::-1]
        models = models[sorted]
        model_counts = model_counts[sorted]

        # convert selected models to integers
        selected_model_int = np.zeros(selected_model.shape, dtype=int)
        model_member_size = np.zeros(models.size, dtype=int)
        for ii, model in enumerate(models):
            selected_model_int[np.where(selected_model == model)] = int(ii)
            model_member_size[ii] = self.prediction_dict['inputs'].member[np.where(self.prediction_dict['inputs'].model == model)[0]].size

        # make the plot
        fig, ax = plt.subplots(1, 1, figsize=(np.min([models.size, 10]), 5))
        # histogram, stacked across ranking of analog (best, second best, etc.)
        n, bins, patches = ax.hist(selected_model_int, stacked=True, bins=np.arange(models.size+1), density=True,)
        # get the bin centers for placing the labels and coloring
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # the relevant bins are those with non-zero entries
        relevant_bins = np.where(np.any(n != 0, axis=0))[0]
        for c, p in zip(np.linspace(0, 1, n_analogs), patches):
            plt.setp(p, 'facecolor', red_cmap(c))
            plt.setp(p, 'edgecolor', 'k')
        # wrap the model name on the first dash appearing in the name (nicer printing)
        for ii, model in enumerate(models):
            model_parts = model.replace('-', '\n', 1)
            models[ii] = model_parts
        for ii in range(np.min([models.size, 10])):
            ax.text(bin_centers[relevant_bins[ii]], n[-1, ii]+0.01, str(model_member_size[ii]) + '/' + str(model_member_size.sum()) +
                    '\n(' + str(np.round(model_member_size[ii]/model_member_size.sum(), 2)) + ')', ha='center', fontsize=8)
        ax.set_xticks(bin_centers[relevant_bins][:10], models[:10], fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        # ax.set_title('SOI from: '+str(soi_model), fontsize=12, va='top')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        ax.tick_params(which='both', axis='x', bottom=False)
        ax.set_xlim([bins[0], bins[np.min([models.size, 10])]])
        
        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_model_times(self, soi_model, soi_member, variable, n_analogs, savename=None):
        # plot the number of times each model was selected as a top n_analogs analog
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_model_times.pdf'
        
        # get the index of the soi corresponding to the given model and member
        soi_index = np.where((self.prediction_dict['indices'].soi_model == soi_model) * 
                             (self.prediction_dict['indices'].soi_member == soi_member))[0][0]
        
        # get the targets for the requested soi and variable to restrict the times
        truth, _, _, _ = self.get_targets(soi_model, soi_member, variable, n_analogs=1, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]
        times = self.prediction_dict['target']['time.year'][:ntimes] + leadtime
        # get the models that were selected for the analogs of this soi
        selected_model = self.prediction_dict['indices'].isel(soi_index=soi_index).best_analog_models[:ntimes, :n_analogs]
        # get the names of the models that were selected for this soi across all times
        models = np.unique(selected_model)

        # sort by total number of selections for each model (high to low) -- for keeping it to top ten
        model_counts = np.unique(selected_model, return_counts=True)[1]
        sorted = np.argsort(model_counts)[::-1]
        models = models[sorted]
        model_counts = model_counts[sorted]

        # convert selected models to integers
        selected_model_int = np.zeros([selected_model.shape[0], models.size], dtype=int)
        for ii, model in enumerate(models):
            selected_model_int[:, ii] = np.sum(selected_model.to_numpy() == model, axis=1)
        # normalize to maximum count -- this just makes the relative color intensity look better (spans 0 to 1)
        selected_model_int = selected_model_int/selected_model_int.max()

        # make the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, np.min([models.size/2, 3.5])))
        ax.pcolormesh(times, models[:10], selected_model_int[:, :10].T, cmap=alpha_cmap)

        ax.set_yticks(np.arange(models[:10].size), models[:10], fontsize=11)
        ax.set_xlim([times.min()-2, times.max()+2])
        ax.set_ylim([0-0.5, np.min([models.size, 10])-0.5])  
        ax.set_xlabel('prediction year', fontsize=14)
        # ax.set_title('SOI from: '+str(soi_model), fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=0)
        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
# -------------------------------------------------------------------------------

    # TODO: this needs to be fixed
    def plot_selected_times(self, soi_model, soi_member, n_analogs, roll_years, savename=None):
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        if savename is None:
            savename = self.experiment_name+'_timecomp.pdf'
        leadtime = settings['leadtime']
        # roll overs some years (centered, so roll_years needs to be odd)
        spread = int((roll_years - 1)/2)

        # get the index of the soi corresponding to the given model and member
        soi_index = np.where((self.prediction_dict['indices'].soi_model == soi_model) * 
                            (self.prediction_dict['indices'].soi_member == soi_member))[0][0]

        # get the soi year and the spread of the selected analog years
        ######### fix this #########
        # np.sum(~np.isnan(self.prediction_dict['target'].isel(member=soi_index).mean(['lat', 'lon'], skipna=True)))
        # soi_times = self.prediction_dict['target']['time.year'][:172] + leadtime
        soi_times = self.prediction_dict['target']['time.year'] + leadtime
        cut = 171-17
        # ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]
        analog_times = self.prediction_dict['indices']['best_input_times.year'][soi_index, :, :n_analogs] + leadtime

        # getting the rolling mean, min, max
        best_analog = np.zeros(soi_times.size - roll_years + 1)
        min_analog = np.zeros(soi_times.size - roll_years + 1)
        max_analog = np.zeros(soi_times.size - roll_years + 1)
        for ii in range(spread, best_analog.size+spread):
            best_analog[ii-spread] = analog_times[ii-spread:ii+spread+1].mean()
            min_analog[ii-spread] = np.sort(analog_times[ii])[int(analog_times[ii].size * 0.2)]#analog_times[ii].min()
            max_analog[ii-spread] = np.sort(analog_times[ii])[int(analog_times[ii].size * 0.8)]#analog_times[ii].max()
        # roll the soi_times over the same period
        soi_times = soi_times.rolling(time=roll_years, center=False).mean().dropna("time", how="all")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # ax.errorbar(soi_times, best_analog, yerr=np.array([abs(best_analog-min_analog), abs(max_analog-best_analog)]), fmt='o', 
        #             elinewidth=1.1, markersize=4, c=alpha_colors[-1])
        ax.errorbar(soi_times[:cut], best_analog[:cut], yerr=np.array([abs(best_analog[:cut]-min_analog[:cut]), abs(max_analog[:cut]-best_analog[:cut])]), fmt='o', 
                    elinewidth=1.1, markersize=4, c=alpha_colors[-1])
        # ax.plot(soi_times, soi_times, c=c_midnight)
        ax.plot([np.min([soi_times[:cut], best_analog[:cut]]), np.max([soi_times[:cut], best_analog[:cut]])], 
                [np.min([soi_times[:cut], best_analog[:cut]]), np.max([soi_times[:cut], best_analog[:cut]])], c=c_midnight)

        ax.set_xlabel('prediction year', fontsize=14)
        ax.set_ylabel('analog year', fontsize=14)
        ax.set_xlim([np.min([soi_times[:cut], min_analog[:cut]])-5, np.max([soi_times[:cut], max_analog[:cut]])+5])
        ax.set_ylim([np.min([soi_times[:cut], min_analog[:cut]])-5, np.max([soi_times[:cut], max_analog[:cut]])+5])
        # ax.set_title('SOI from: '+str(soi_model), fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_constrained_prediction(self, soi_model, soi_member, variable, n_analogs, max_year, savename=None):
        # TODO: change this to only pass pre max_year analogs (will be much faster)
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if savename is None:
            savename = self.experiment_name+'_constrained_prediction.pdf'
        # get the targets for the requested soi and variable
        truth, prediction, prediction_min, prediction_max = self.get_targets(soi_model, soi_member, variable, n_analogs, return_average=True, global_mean=True)
        _, c_prediction, c_prediction_min, c_prediction_max = self.get_targets(soi_model, soi_member, variable, n_analogs, return_average=True, global_mean=True, max_year=max_year)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the range of all the best analogs (unconstrained)
        ax.fill_between(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction_min[:ntimes], prediction_max[:ntimes], color=c_flowertrial,
                         alpha=0.3, path_effects=[patheffects.withStroke(linewidth=1, foreground=c_flowertrial, alpha=0.3)])
        # plot the range of all the best analogs (constrained)
        ax.fill_between(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, c_prediction_min[:ntimes], c_prediction_max[:ntimes], color=c_horsetooth,
                         alpha=0.3, path_effects=[patheffects.withStroke(linewidth=1, foreground=c_horsetooth, alpha=0.3)])
        # plot the true values (i.e. the soi target)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes], truth[:ntimes], lw=3.33, c=c_midnight, label='truth',
                 path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
        # plot the prediction (unconstrained)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, prediction[:ntimes], label='analogs ('+str(n_analogs)+')',
                 lw=3.33, c=c_flowertrial, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])
        # plot the prediction (unconstrained)
        ax.plot(self.prediction_dict['target']['time.year'][:ntimes]+leadtime, c_prediction[:ntimes], label='constrained analogs ('+str(n_analogs)+')',
                 lw=3.33, c=c_horsetooth, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(settings['region']+'\n'+variable, fontsize=18)
        ax.set_title(self.experiment_name+r', $\leq$'+str(max_year), fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
        
    def prediction_mse(self, soi_model, soi_member, variable, n_analogs, leadstep=1):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        # get the targets for the requested soi and variable
        truth, prediction, _, _ = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep, global_mean=True)
        ntimes = np.shape(np.where(~np.isnan(truth))[0])[0]
        # get model mean
        all_model_mean = self.get_full_analog_range(soi_model, soi_member, variable, global_mean=True, minmax=False)

        model_mean_mse = np.mean((all_model_mean[:ntimes] - truth[:ntimes])**2)
        analog_mse = np.mean((prediction[:ntimes-leadtime] - truth[leadtime:ntimes])**2)
        return(analog_mse, model_mean_mse)

# -------------------------------------------------------------------------------

    ###################################################################################
    # add function to find the average number of leadsteps it takes to switch to
    # a new (set of) analogs -- basically how far out can we go on average
    ###################################################################################

# -------------------------------------------------------------------------------
