import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
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

# define some colors -- Nuggets here
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

# set some plotting defaults
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
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
    Class for . . . . 
    """
    def __init__(self, experiment_name, model_list=None, n_members=None):
        # save the experiment_name, load the corresponding mask
        self.experiment_name = experiment_name
        ###################################################################################
        # this is the mean mask over all seeds, but if we go with a single instance this needs to change
        self.mask = xr.load_dataset(os.path.join('saved_masks', experiment_name+'_mask.nc')).squeeze().mean("seed").mask
        ###################################################################################
        
        try:
            # check if processed projection dataset file already exists by trying to load
            with gzip.open(os.path.join("processed_data", experiment_name+'_projection.pickle'), "rb") as filename:
                self.projection_data = pickle.load(filename)
        except:
            # if file does not exist, load the analogs from training
            with gzip.open(os.path.join("processed_data", experiment_name+'_analog.pickle'), "rb") as filename:
                self.projection_data = pickle.load(filename)
            # save this as the starting point for the projection analog library
            with gzip.open(os.path.join("processed_data", experiment_name+'_projection.pickle'), "wb") as filename:
                pickle.dump(self.projection_data, filename)
                
        # initialize the used_members and use_models arrays with the projection_data
        self.used_members = self.projection_data['inputs'].member.to_numpy()
        self.used_models = self.projection_data['inputs'].model.to_numpy()
            
        # get the training, validation, and testing data so they can be avoided in the soi and analogs
        for split in ['training', 'validation', 'testing']:
            with gzip.open(os.path.join("processed_data", experiment_name+'_'+split+'.pickle'), "rb") as filename:
                # just getting the models and members, so either inputs or target is sufficient
                split_data = pickle.load(filename)['inputs']
            # concatenate the training data used_models and used_members
            self.used_members = np.concatenate([self.used_members, split_data.member.to_numpy()])
            self.used_models = np.concatenate([self.used_models, split_data.model.to_numpy()])
            ##### SHOULD I SAVE THESE? ONLY IF CALCULATING BECOMES SLOW
            
        # call process_new_data with model_list and n_members which will add new data to the dataset
        if model_list is not None and n_members is not None:
            self.process_new_data(experiment_name, model_list, n_members)
            # and save the new, expanded projection dataset
            with gzip.open(os.path.join("processed_data", experiment_name+'_projection.pickle'), "wb") as filename:
                pickle.dump(self.projection_data, filename)
            
        # weight all the input data by the mask
        self.projection_data["inputs"] = self.projection_data["inputs"] * self.mask
        
        # try loading saved analogs for the given experiment
        try:
            with gzip.open(os.path.join("saved_analogs", experiment_name+'.pickle'), "rb") as filename:
                self.projection_dict = pickle.load(filename)
                print('loaded previously saved analogs into self.projection_dict')
        except:
            print('no previouly saved analogs to load!')

# -------------------------------------------------------------------------------

    def process_new_data(self, experiment_name, model_list, new_members=-1):
        # n_members=-1 means return all the rest of the members from each model
        
        # get experiment settings
        experiment_settings = rexp.Experiments().get_experiment(experiment_name)
        variables = experiment_settings['variable_list']
        
        # load the new raw data
        for ii, model in enumerate(model_list):
            # glob all the files and sort -- just used to get correct indices
            files = np.sort(glob.glob(os.path.join('raw_data', experiment_settings['data_period'], model, variables[0], '*')))

            # remove indices corresponding to model/members that are in this models used_members
            model_used_members = self.used_members[np.where(self.used_models == model)]
            model_new_members = np.setdiff1d(np.arange(files.size), model_used_members)
            assert model_new_members.size != 0, "there are no more new members to load for model "+model
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
            input_raw = _dp.DataProcessing([model], variables, experiment_settings['data_period'], ensembles=model_new_members)
            for var in variables:
                # make the resolution the same as the previously processed datasets
                input_raw.attr_dictionary[var]['res'] = [np.diff(self.projection_data['lat']).mean(), np.diff(self.projection_data['lon']).mean()]
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

        # concatenate this model/variable onto the existing self.projection_data["inputs"] and self.projection_data["target"]
        self.projection_data["inputs"] = xr.concat([self.projection_data["inputs"], inputs_raw], dim="member")
        self.projection_data["target"] = xr.concat([self.projection_data["target"], outputs_raw], dim="member")
        
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
        # check if the requested SOI is in the projection_data yet -- for now, it has to be, otherwise
        # I will have to add a call to process_new_data that bypasses some of the concatenation, etc.
        try:
            soi_index = np.where((self.projection_data['inputs'].model == soi_model) *
                                 (self.projection_data['inputs'].member == soi_member))[0][0]
            soi_inputs = self.projection_data['inputs'].isel(member=soi_index)
        except:
            print('that SOI is not in the already processed projection_data')
            
        # now remove this particular soi from the analog library
        analog_inds = np.setdiff1d(np.arange(self.projection_data['inputs'].member.size), soi_index)
        weighted_analogs = self.projection_data['inputs'].isel(member=analog_inds)
        
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
            result =  pool.map(self._get_single_soi_analogs, data_iterable, chunksize=chunksize)
            
        # now combine all the soi results into a single DataSet
        for ii in range(len(result)):
            if ii == 0:
                soi_ds = result[ii]
            else:
                soi_ds = xr.concat([soi_ds, result[ii]], dim='time')

        # combine the best analogs with the targets in a dictionary -- note that this is all
        # the targets, including the SOI, so this will have one entry more than the indices
        self.projection_dict = {'indices': soi_ds, 'targets': self.projection_data['target']}
        
        try:
            # if file already exists, concatenate onto existing by the soi_index dimension
            failure_reason = "failed to load existing file\n" + "file path: " + os.path.join("saved_analogs", self.experiment_name+'.pickle')
            with gzip.open(os.path.join("saved_analogs", self.experiment_name+'.pickle'), "rb") as filename:
                self.projection_dict = pickle.load(filename)
            # limit the number of analogs returned to the number from the first calculation
            # otherwise, it will try to concatenate different sized analog indices (since
            # more data could have been added) -- should avoid doing this either way, i.e.
            # should have the full analog set determined before running get_all_soi_analogs
            max_analogs = self.projection_dict['indices'].dims['analog_size']
            failure_reason = "failed to concatenate new data with existing\n\n"+"new data:\n"+str(soi_ds)+"\n\nexisting data:\n"+str(self.projection_dict['indices'])
            self.projection_dict['indices'] = xr.concat([self.projection_dict['indices'], soi_ds.isel(analog_size=np.arange(max_analogs))], dim='soi_index')
            self.projection_dict['targets'] = self.projection_data['target']
        except:
            print(failure_reason)
        # and save it
        with gzip.open(os.path.join("saved_analogs", self.experiment_name+'.pickle'), "wb") as filename:
            print('saving to:', os.path.join("saved_analogs", self.experiment_name+'.pickle'))
            pickle.dump(self.projection_dict, filename)
        
        return(self.projection_dict)
      
# -------------------------------------------------------------------------------
    
    def soi_iterable(self, weighted_soi, weighted_analogs):
        # make data iterable for use in multiprocessing pool
        """
        Create an iterable for a parallel approach to metric assessment
        """
        for jj in range(weighted_soi.time.size):
            yield weighted_soi.isel(time=jj), weighted_analogs
            
# -------------------------------------------------------------------------------
        
    def plot_projection(self, soi_model, soi_member, variable, n_analogs, leadstep=1, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        if savename is None:
            savename = self.experiment_name+'_projection.png'
        # get the targets for the requested soi and variable
        truth, projection, projection_min, projection_max = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep)

        # get the soi index and the full range of analog projections
        soi_index = np.where((self.projection_dict['targets'].model == soi_model) * 
                             (self.projection_dict['targets'].member == soi_member))[0]
        all_analogs = np.arange(self.projection_dict['targets'].member.size)
        non_soi_analogs = np.setdiff1d(all_analogs, soi_index)
        all_min = self.projection_dict['targets'].isel(member=non_soi_analogs).min(dim='member')[variable].data
        all_max = self.projection_dict['targets'].isel(member=non_soi_analogs).max(dim='member')[variable].data

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the range of all analogs
        ax.fill_between(self.projection_dict['indices']['soi_time.year'][::leadstep], all_min, all_max, color=c_skyline,
                         alpha=0.4, label='all analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline)])
        # plot the range of all the best analogs
        ax.fill_between(self.projection_dict['indices']['soi_time.year'][::leadstep], projection_min, projection_max, color=c_flatirons,
                         alpha=0.3, label='best analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_flatirons)])
        # plot the true values (i.e. the soi target)
        ax.plot(self.projection_dict['indices']['soi_time.year'][::leadstep], truth, lw=3.33, c=c_midnight, label='truth',
                 path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
        # ax.plot(self.projection_dict['indices']['soi_time.year'][::leadstep], truth, lw=0.5, c='white', label='truth',
             # path_effects=[patheffects.withStroke(linewidth=5, foreground=c_midnight)])
        # plot the projection
        ax.plot(self.projection_dict['indices']['soi_time.year'][::leadstep], projection, label='best analogs ('+str(n_analogs)+')',
                 lw=3.33, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=4, frameon=False, mode='expand',
                        prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel(settings['region']+'\n'+variable, fontsize=18)
        ax.set_title(self.experiment_name, fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig('figures/'+savename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.show()
        
# -------------------------------------------------------------------------------

    def plot_projection_difference(self, soi_model, soi_member, variable, n_analogs, leadstep=1, savename=None):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        if savename is None:
            savename = self.experiment_name+'_difference.png'
        # get the targets for the requested soi and variable
        truth, projection, projection_min, projection_max = self.get_targets(soi_model, soi_member, variable, n_analogs, leadstep)

        # get the soi index and the full range of analog projections
        soi_index = np.where((self.projection_dict['targets'].model == soi_model) * 
                             (self.projection_dict['targets'].member == soi_member))[0]
        all_analogs = np.arange(self.projection_dict['targets'].member.size)
        non_soi_analogs = np.setdiff1d(all_analogs, soi_index)
        all_min = self.projection_dict['targets'].isel(member=non_soi_analogs).min(dim='member')[variable].data
        all_max = self.projection_dict['targets'].isel(member=non_soi_analogs).max(dim='member')[variable].data

        # now make the plot
        fig, ax = plt.subplots(2, 1, figsize=(15,7.5), height_ratios=[2, 1])

        # top panel -- truth, projection, analog ranges
        # plot the range of all analogs
        ax[0].fill_between(self.projection_dict['indices']['soi_time.year'][::leadstep], all_min, all_max, color=c_skyline,
                         alpha=0.4, label='all analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline)])
        # plot the range of all the best analogs
        ax[0].fill_between(self.projection_dict['indices']['soi_time.year'][::leadstep], projection_min, projection_max, color=c_flatirons,
                         alpha=0.3, label='best analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_flatirons)])
        # plot the true values (i.e. the soi target)
        ax[0].plot(self.projection_dict['indices']['soi_time.year'][::leadstep], truth-projection, lw=3.33, c=c_midnight, label='truth',
                 path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
        # plot the projection
        ax[0].plot(self.projection_dict['indices']['soi_time.year'][::leadstep], projection, label='best analogs ('+str(n_analogs)+')',
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
        # [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = ax[0].legend(handles=handles, labels=labels, loc='upper left', ncols=4, frameon=False, mode='expand',
                        prop={'weight':'bold', 'size':'15'})


        # bottom panel -- difference between truth and projection
        ax[1].plot(self.projection_dict['indices']['soi_time.year'][::leadstep], truth-projection, lw=0.5, c='white', label='truth $-$ best analogs',
             path_effects=[patheffects.withStroke(linewidth=5, foreground=c_midnight)])
        # legend, labels, etc.
        handles, labels = ax[1].get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = ax[1].legend(handles=handles, labels=labels, loc='lower left', ncols=4, frameon=False, mode='expand',
                        prop={'weight':'bold', 'size':'15'})
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
        plt.xlabel('year', size=20, labelpad=-30)

        plt.subplots_adjust(hspace=0.2)
        plt.savefig('figures/'+savename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.show()

# -------------------------------------------------------------------------------

    ###################################################################################
    # plot the worst lat/lon to give a sense of how well it is doing -- will have to
    # wait to do this until I run something with a target that isn't averaged over a
    # region, but it will be similar to the above figures
    ###################################################################################

# -------------------------------------------------------------------------------
    
    def get_targets(self, soi_model, soi_member, variable, n_analogs, leadstep=1):
        """
        get the best analogs for the given soi model/member, skipping every leadstep (number of steps
        in 'internal' units, meaning consecutive time steps from the processed data)
        
        returns the targets for both the analogs, and the soi (will be different shapes if leadstep != 1)
        """
        # make sure there are analogs to get targets from
        assert self.projection_dict, "no soi analogs calculated yet, run self.get_all_soi_analogs(soi_model, soi_member, n_proc=None)"
        # get the soi_index corresponding to the requested soi_model and soi_member
        soi_index = np.where((self.projection_dict['indices'].soi_model == soi_model) * 
                             (self.projection_dict['indices'].soi_member == soi_member))[0]
        
        top_analogs = self.projection_dict['indices'].isel(analog_size=np.arange(n_analogs), soi_index=soi_index)
        # at this point we are throwing xarray away and using numpy!
        best_model_list = top_analogs.best_analog_models.data[0]
        best_member_list = top_analogs.best_analog_members.data[0]
        best_time_list = top_analogs.best_analog_times.data[0]
        
        # for the specified soi, get the corresponding targets
        soi_projection, projection_min, projection_max = [], [], []
        # for each time step in steps of leadstep
        for ii in range(0, best_model_list.shape[0], leadstep):
            step_analogs = np.zeros(n_analogs)
            # for each analog
            for mm in range(step_analogs.size):
                # get indices for member and time for the target dataset
                member_ind = np.where(np.logical_and(self.projection_dict['targets'].model == best_model_list[ii, mm],
                                                     self.projection_dict['targets'].member == best_member_list[ii, mm]))[0]
                
                # now collect the number of analogs
                step_analogs[mm] = self.projection_dict['targets'].isel(time=best_time_list[ii, mm])[variable].isel(member=member_ind).data
            # now get the average of the analogs
            soi_projection.append(np.mean(step_analogs))
            projection_min.append(np.min(step_analogs))
            projection_max.append(np.max(step_analogs))
            
        # get the soi (aka the truth)
        soi_ind = np.where((self.projection_dict['targets'].model == top_analogs.soi_model.data) &
                           (self.projection_dict['targets'].member == top_analogs.soi_member.data))[0]
        truth = self.projection_dict['targets'].isel(member=soi_ind)[variable].data.squeeze()
        
        ###########################################################################################
        # save the resulting projection? pretty fast to compute individually, but for lots . . .
        ###########################################################################################
        return(truth, soi_projection, projection_min, projection_max)
        
# -------------------------------------------------------------------------------

    ###################################################################################
    # add function to threshold masks? edge detection even?
    ###################################################################################

# -------------------------------------------------------------------------------

    ###################################################################################
    # add function to find the average number of leadsteps it takes to switch to
    # a new (set of) analogs -- basically how far out can we go on average

    # add function to get n best analog targets with specific leadstep
    # # use these best indices to get the corresponding analog targets, (which are already offset by one year)
    #     best_analogs = np.zeros([leadstep, n_analogs])
    #     for jj in range(n_analogs):
    #         best_analogs[:, jj] = target_analogs.isel(sample=np.arange(ind_analogs[jj], ind_analogs[jj]+leadstep)).tas.to_numpy()
    ###################################################################################

# -------------------------------------------------------------------------------

