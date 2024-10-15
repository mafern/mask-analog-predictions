import json
import numpy as np
import os
import glob
import xarray as xr
import data_processing as dp
import pickle
import build_model
import train_model
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pprint import pprint

# define cmap for masks and some colors
c_energygreen = '#CFFC00'
c_flowertrial = '#E56A54'
c_powered = '#7E5475'
c_horsetooth = '#008FB3'
c_stalwart = '#105456'
blue_skyline = ['#0C2340', '#003057', '#1D4289', '#0057B7', '#418FDE', '#FFFFFF']
blue_cmap = clr.LinearSegmentedColormap.from_list('blue skyline', blue_skyline[::-1], N=256)

def get_settings(experiment_name):
    "helper to be consistent with existing code"
    experiments = Experiments().experiments
    return experiments[experiment_name]

# -------------------------------------------------------------------------------

def add_region(region_name, lats, lons):
    "add a region to regions.json with the given lat and lon bounds"
    # load current regions from regions.json
    current_regions = json.load(open('regions.json', 'r'))
    # add new region to the dictionary
    current_regions[region_name] = {"lats": lats, "lons": lons}
    # save the updated dictionary
    with open('regions.json', "w") as region_file:
        region_file.write(json.dumps(current_regions))
    print(region_name, 'added to regions.json')

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

class Experiments():
    """
    Class for running experiments: experiment setup, data processing, mask generation, and best analog selection.
    """
    def __init__(self, filename="experiments.json"):
        # load the dictionary and keys
        try:
            self.filename = filename
            self.experiments = json.load(open(os.path.join(os.path.dirname(__file__), self.filename), 'r'))
            self.fig_path = os.path.join(os.path.dirname(__file__), 'figures/')
            self.keys = list(self.experiments.keys())
        except ValueError:
            print("No experiment file found: ", self.filename)
            
    # -------------------------------------------------------------------------------

    def get_experiment(self, exp_name):
        "get the experiment dictionary entry with exp_name"
        return self.experiments[exp_name]
    
    # -------------------------------------------------------------------------------

    def add_experiment(self, new_exp):
        "add experiment(s) to the save file"
        # add to the dictionary, resave
        experiments = self.experiments.copy()
        for exp_name in new_exp:
            experiments[exp_name] = new_exp[exp_name]
        with open(self.filename, "w") as exp_file:
            exp_file.write(json.dumps(experiments))
        # reload the experiments
        self.experiments = experiments
        self.keys = list(self.experiments.keys())
        
    # -------------------------------------------------------------------------------

    def delete_experiment(self, exp_name):
        "delete an experiment from the main file, does not delete from any backups"
        self.experiments.pop(exp_name)
        with open(self.filename, "w") as exp_file:
            exp_file.write(json.dumps(self.experiments))
        # reload the exp name list
        self.keys = list(self.experiments.keys())
        
    # -------------------------------------------------------------------------------

    def save_backup(self, filename="experiments-backup.json"):
        "save a backup file!"
        with open(os.path.join(os.path.dirname(__file__), filename), "w") as exp_file:
            exp_file.write(json.dumps(self.experiments))
            
    # -------------------------------------------------------------------------------

    def restore_from_backup(self, backupfile, *, confirm=0):
        "Overwrite the main saved file with the dictionary in the backup file"
        if confirm == 1: # just to make sure
            backup = json.load(open(os.path.join(os.path.dirname(__file__), backupfile), 'r'))
            with open(os.path.join(os.path.dirname(__file__), self.filename), "w") as exp_file:
                exp_file.write(json.dumps(backup))
            self.experiments = json.load(open(os.path.join(os.path.dirname(__file__), self.filename), 'r'))
        else:
            print("must pass confirm=1 to overwrite main file with backup")
            
    # -------------------------------------------------------------------------------

    def make_exp_dictionary(self, experiment_name, *, variable_list, analog_models, soi_models, leadtime, split, region, val_models=None, target_var=None, prediction_nodes=[5, 5, 5], prediction_activation='elu', learning_rate=0.001, method='jacob', mask=False, output_period=False, roll_period=False, window=1, trajectory=1, resolutions=None, season=False, time_range=None, norm_data=True, baseline=False, remove_trend=False, remove_seasonality=False, loss='mse', patience=50, min_delta=0.0005, batch_size=64, val_batch_size=2500, max_epochs=5000, mask_activation='relu', mask_init="ones", mask_l1=0.0, mask_l2=0.0, norm_weights=True, transfer_exp=None, seed=20):
        """
        Make an experiment dictionary with the following settings:
            REQUIRED INPUTS
                - experiment_name: name of the experiment (how it will be saved and called)
                - variable_list: list of variable names to be included in the data
                - analog_models: which models to use for the analog split (run self.list_variables to see available options)
                - soi_models: which models to use for the training split (run self.list_variables to see available options)
                - leadtime: time shift between inputs and target (should be a positive integer)
                - split: list with integer number of members to include in each split of the data (order should be: analog, training, validation)
                - region: the name (run self.list_regions to get available names) or list of boundary coordinates, i.e. [minlon, minlat, maxlon, maxlat] for the target region. To skip this processing step, pass False.
                                
            OPTIONAL INPUTS
                - method: which method to use, 'jacob' (input mse predicts target mse) or 'martin' (input mse predicts target map, loss is mse)
                - val_models: which models to use for the validation split (if None, use the same models as soi_models)
                - target_var: variable to predict (if None, use the first entry of the variable_list)
                - mask: whether to mask land (pass 'land'), ocean (pass 'ocean'), or neither (pass False, default)
                - prediction_nodes: a list where the number of entries is the number of layers, and the values are the number of nodes for that layer
                - prediction_activation: activation function used for the network
                - learning_rate: step size for the training
                - output_period: either resample the data to a new period (pass a string such as "2M"), resample on the same period (pass None) or do nothing (pass False, default)
                - roll_period: calculate rolling mean for the data (pass a positive integer for number of time steps to roll), or do nothing (pass False, default)
                - window: calculate rolling mean over this number of steps (default is no windowing)
                - trajectory: number of time steps over which to match for training (default is 1, which is matching on input time only)
                - resolutions: resample spatial grid of data, either to coarsest resolution amongst models (pass None, default), to a specified resolution (pass a list [lat_res, lon_res]), or do nothing (pass False)
                - season: calculate a yearly season from the data (pass a list with entries [season_length, season_start]), or do nothing (pass False, default)
                - time_range: sync the time ranges between models, to the common time range (pass None, default), to a specified time range (pass a list [start_date, end_date]), or do nothing (pass False)
                - norm_data: whether to normalize the data by its mean and std (default is True)
                - baseline: normalize the datasets by a baseline period (pass a list of a start and end year for the baseline, e.g. [1951, 1980]) or do nothing (pass False, default)
                - remove_trend: whether to remove the trend via the ensemble mean (pass True) or do nothing (pass False, default)
                - remove_seasonality: whether to remove the seasonal cycle (pass True), or do nothing (pass False, default)
                - loss: loss function to use in training, default is 'mse'
                - patience: number of steps (epochs) to take without improving loss before early stopping, default is 50
                - min_delta: minimum change in the loss to qualify as an improvement, default is 0.0005
                - batch_size: training batch size, default is 64
                - val_batch_size: size of validation batch (this is static amongst training batches), default is 2500
                - max_epochs: maximum number of training epochs
                - mask_activation: activation function for the mask, default is 'relu'
                - mask_init: initial weights for the mask, default is 'ones'
                - mask_l1: l1 regularization for the mask, default is 0 (off)
                - mask_l2: l2 regularization for the mask, default is 0 (off)
                - norm_weights: whether to normalize the mask weights, default is True
                - transfer_exp: experiment name (string) from which to transfer learn (initializes mask, layers, or both, depending on training inputs)
                - seed: random seed to use in the member selection and the network training
                
        Returns a dictionary containing all of the requested experiment settings, including the processing steps (in a sensable order).
        """
        # check if the proposed experiment name already exists and inputs are sensible
        assert experiment_name not in self.keys, "experiment with that name already exists"
        assert leadtime > 0, "leadtime should be a positive integer"
        assert roll_period > 0 or roll_period is False, "roll_period should be positive or False"
        # if trajectory is greater than 1 (matching on input + previous times), leadtime must be 1
        assert trajectory >= 1, "trajectory must be a positive integer"
        if trajectory > 1:
            print("*** using a trajectory > 1, enforcing a leadtime of 1")
            leadtime = 1
        # get the target variable if one has not been passed
        if target_var is None:
            target_var = [variable_list[0],]
        # check validation models
        if val_models is None:
            val_models = soi_models
        # fill a dictionary with the data split values
        split = {"analog": split[0], "training": split[1], "validation": split[2]}

        # get locals (i.e. all of the inputs to make_exp_dictionary) before assigning anything here
        function_inputs = locals()
        
        inputs_process_dict = {"select_season(season=experiment_settings['season'])": season,
                               "smooth_time(period=experiment_settings['output_period'])": output_period,
                               "sync_time(experiment_settings['time_range'])": time_range,
                               "mask_data(experiment_settings['mask'])": mask,
                               "smooth_space(experiment_settings['resolutions'])": resolutions,
                               "baseline_norm(experiment_settings['baseline'])": baseline,
                               "remove_seasonality()": remove_seasonality,
                               "rolling_time(experiment_settings['roll_period'])": roll_period,
                               "window_time(experiment_settings['window'])": window,
                               "shift_inout_time(-experiment_settings['leadtime'], experiment_settings['window'])": leadtime,
                               "remove_trend()": remove_trend,
                               "norm_data()": norm_data}
        
        target_process_dict = {"select_season(season=experiment_settings['season'])": season,
                               "smooth_time(period=experiment_settings['output_period'])": output_period,
                               "sync_time(experiment_settings['time_range'])": time_range,
                               "mask_data(experiment_settings['mask'])": mask,
                               "smooth_space(experiment_settings['resolutions'])": resolutions,
                               "get_region(experiment_settings['region'])": region,
                               "baseline_norm(experiment_settings['baseline'])": baseline,
                               "remove_seasonality()": remove_seasonality,
                               "rolling_time(-experiment_settings['roll_period'])": roll_period,
                               "window_time(experiment_settings['window'])": window,
                               "shift_inout_time(experiment_settings['leadtime'], experiment_settings['window'])": leadtime,
                               "remove_trend()": remove_trend,
                               "norm_data()": norm_data}
        
        # make the processing list for the input data
        inputs_proc_steps = []
        for key, value in inputs_process_dict.items():            
            if value is not False:
                inputs_proc_steps.append(key)
        
        # make the processing list for the target data
        target_proc_steps = []
        for key, value in target_process_dict.items():            
            if value is not False:
                target_proc_steps.append(key)
            
        experiment_settings = {}
        # simple way to get all the inputs, does not include processing lists
        experiment_settings[experiment_name] = function_inputs
        # remove self (not needed) from dictionary
        experiment_settings[experiment_name].pop('self')
        # now add the processing lists
        experiment_settings[experiment_name]["inputs_proc_steps"] = inputs_proc_steps
        experiment_settings[experiment_name]["target_proc_steps"] = target_proc_steps
            
        print('experiment_settings are:')
        print(experiment_settings)    
        return experiment_settings

    # -------------------------------------------------------------------------------
    
    def get_experiment_data(self, experiment_name, overwrite_data=False):
        "process, or load previously processed, data. return stacked samples and place full data in class"
        # get settings
        experiment_settings = self.get_experiment(experiment_name)
        # make or load the data for this experiment
        self.processed_data = dp.process_data(experiment_settings, overwrite_data=overwrite_data)
        # remove bnds dimension, if it exists
        for key in self.processed_data.keys():
            for inputs_target in ['inputs', 'target']:
                try:
                    self.processed_data[key][inputs_target] = self.processed_data[key][inputs_target].drop_dims(["bnds", "nbnd"])
                except:
                    pass
        print('----------------------------------------------------------------')
        # stack/separate into inputs and target for the analog, training, and validation sets
        analog_inputs, analog_target = dp.neural_network_stack(self.processed_data['analog'], experiment_settings)
        training_inputs, training_target = dp.neural_network_stack(self.processed_data['training'], experiment_settings)
        validation_inputs, validation_target = dp.neural_network_stack(self.processed_data['validation'], experiment_settings)
        # get the lat and lon ------ THIS ASSUMES THAT THE LAT AND LON ARE THE SAME BETWEEN THE ANALOG AND TRAINING SETS
        lat, lon = self.processed_data['analog']['lat'], self.processed_data['analog']['lon']
        print('----------------------------------------------------------------')
        return [analog_inputs, analog_target, training_inputs, training_target,
                validation_inputs, validation_target, lat, lon]
    
    # -------------------------------------------------------------------------------
    
    def train_experiment_model(self, experiment_name, overwrite_data=False, overwrite_model=False, transfer=None):
        # get settings
        experiment_settings = self.get_experiment(experiment_name)
        # get data
        (analog_inputs, analog_target,
         training_inputs, training_target,
         validation_inputs, validation_target,
         lat, lon) = self.get_experiment_data(experiment_name, overwrite_data=overwrite_data)
        
        # reshape for multiple variables, since tensorflow flatten wants it this way (sample, var, lat, lon)
        analog_inputs = np.transpose(analog_inputs, (0, 3, 1, 2))
        training_inputs = np.transpose(training_inputs, (0, 3, 1, 2))
        validation_inputs = np.transpose(validation_inputs, (0, 3, 1, 2))
        
        # ensure model names are consistent, etc.
        tf.keras.backend.clear_session()
        np.random.seed(experiment_settings["seed"])
        random.seed(experiment_settings["seed"])
        tf.random.set_seed(experiment_settings["seed"])

        # build the full model for learning the mask: mask, dissimilarity, prediction models
        model, _, _, _ = build_model.build_interp_model(experiment_settings, np.shape(training_target)[1:], [training_inputs, analog_inputs], transfer=transfer)

        # train the model, which will check if the model is already saved first
        # (then load if it does, train and save if it does not)
        model, history = train_model.train_model(
            experiment_settings,
            model,
            analog_inputs,
            analog_target,
            training_inputs,
            training_target,
            validation_inputs,
            validation_target,
            overwrite_model=overwrite_model
        )
        
        return model

    # -------------------------------------------------------------------------------
    
    def make_mask(self, experiment_name, overwrite_data=False, overwrite_model=False, overwrite_mask=False, transfer=None):
        # get model (train or load) and the processed data
        models = self.train_experiment_model(experiment_name, overwrite_data=overwrite_data, overwrite_model=overwrite_model, transfer=transfer)
        # get the mask (both a raw version 'raw_mask' and a scaled/masked version 'mask')
        self.mask = self._make_mask(experiment_name, models, self.processed_data, overwrite_mask=overwrite_mask)
        return self.mask
        
    # -------------------------------------------------------------------------------

    def _make_mask(self, experiment_name, model, processed_data, overwrite_mask=False):
        """
        description
        """
        # check if mask already exists and no overwrite has been requested
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc')) or overwrite_mask is True:
            # either mask does not exist, or overwrite has been requested
            
            # get lat and lon
            lat, lon = processed_data['analog']['lat'], processed_data['analog']['lon']
            vars = np.array(processed_data['analog']['inputs'].data_vars)
            print('----------------------------------------------------------------')
            print('retrieving mask and saving to', os.path.join('masks', experiment_name+'_mask.nc'))
            
            # Get both the raw and scaled (masked, scale to mean=1) mask and save as a netcdf file
            raw_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy()
            raw_mask = raw_mask.reshape(vars.size, lat.size, lon.size)
            # get land or ocean mask (or neither)
            geomask = processed_data['analog']['land_ocean_mask'].to_numpy()
            mask = self.scale_mask(raw_mask, geomask)
            
            # save the mask as netcdf with lat and lon coordinates (raw and scaled)
            data_var_dict = {}
            for ii, var in enumerate(vars):
                data_var_dict[var] = (["lat", "lon"], np.array(mask)[ii])
            mask = xr.Dataset(
                data_vars=data_var_dict,
                coords=dict(lat=lat.data, lon=lon.data)
            )
            mask.to_netcdf(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc'))

            data_var_dict = {}
            for ii, var in enumerate(vars):
                data_var_dict[var] = (["lat", "lon"], np.array(raw_mask)[ii])
            mask = xr.Dataset(
                data_vars=data_var_dict,
                coords=dict(lat=lat.data, lon=lon.data)
            )
            mask.to_netcdf(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_raw_mask.nc'))
        
        # no need to get data, build/train model, or make/save mask -- just load the existing mask
        else:
            print('loading pre-saved mask')
            mask = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc')).squeeze()
            
        return mask

    # -------------------------------------------------------------------------------

    def scale_mask(self, weighted_mask, land_ocean_mask):
        # Set masked values to zero
        weighted_mask = weighted_mask * land_ocean_mask
        # Re-scale so the mean weight is 1
        weighted_mask = weighted_mask / np.nansum(weighted_mask) * np.nansum(land_ocean_mask)
        # Set masked values to the 0
        # weighted_mask = np.nan_to_num(weighted_mask)
        return weighted_mask
    
    # -------------------------------------------------------------------------------

    def plot_mask(self, experiment_name, vmin=None, vmax=None, savename=None, cmap=blue_cmap, colorbar=True):
        if savename is None:
            savename = experiment_name+'/'+experiment_name+'_mask'

        mask = self.make_mask(experiment_name)
        for var in np.array(mask.data_vars):
            
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
            data_crs = ccrs.PlateCarree()
            
            land_feature = cfeature.NaturalEarthFeature(
                category='physical',
                name='land',
                scale='110m',
                facecolor='none',
                edgecolor='k',
                linewidth=1.5,
            )
            ax.add_feature(land_feature)

            image = ax.pcolormesh(mask.lon, mask.lat, mask[var], transform=data_crs, cmap=cmap, vmin=vmin, vmax=vmax)
            if colorbar:
                cb = plt.colorbar(image, shrink=0.75, orientation="horizontal", pad=0.02)
            plt.savefig(self.fig_path+savename+'_'+var+'.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
            plt.show()

    # -------------------------------------------------------------------------------

    def make_mask_gif(self, experiment_name, cmap=blue_cmap):
        import imageio
        # get mask
        mask = self.make_mask(experiment_name)
        # get the files for this experiment
        mask_training_epochs_path = os.path.join(os.path.dirname(__file__), 'figures', experiment_name+'/mask_training')
        frames = np.loadtxt(mask_training_epochs_path).reshape(-1, mask.lat.size, mask.lon.size)
        images = []
        for ii, frame in enumerate(frames):
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
            data_crs = ccrs.PlateCarree()
            
            land_feature = cfeature.NaturalEarthFeature(
                category='physical',
                name='land',
                scale='110m',
                facecolor='none',
                edgecolor='k',
                linewidth=1.5,
            )
            ax.add_feature(land_feature)

            ax.pcolormesh(mask.lon, mask.lat, frame, transform=data_crs, cmap=cmap, vmin=frames.min(), vmax=frames.max())
            figure_name = os.path.join(os.path.dirname(__file__), 'figures', experiment_name, 'mask_figure'+str(ii).zfill(5)+'.png')
            plt.savefig(figure_name, dpi=400, bbox_inches='tight', pad_inches=0.1)
            images.append(imageio.imread(figure_name))
            plt.close()
            os.remove(figure_name)

        imageio.mimsave(os.path.join(os.path.dirname(__file__), 'figures', experiment_name, experiment_name+'_mask_gif.gif'), images, fps=12)

    # -------------------------------------------------------------------------------

    def plot_training_history(self, experiment_name, savename=None):
        # get the settings so the region name can be used in the figure
        settings = self.get_experiment(experiment_name)
        if savename is None:
            savename = experiment_name+'_training_history.png'
        # find the history file
        history_file = glob.glob(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, '*_history.pickle'))[0]
        # load the history
        with open(history_file, 'rb') as file:
            history = pickle.load(file)
        # plot the training and validation losses versus epoch
        fig, ax = plt.subplots(1, 1, figsize=(11,9))
        ax.plot(np.arange(len(history['loss'])), history['loss'], lw=2, color=c_powered, label='training loss')
        ax.plot(np.arange(len(history['loss'])), history['val_loss'], lw=2, color=c_energygreen, label='validation loss')
        ax.set_ylim([0.99*np.min(history['loss']), 1.05*np.sort(history['val_loss'])[-3]])
        ax.legend(loc='upper right', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('epoch', fontsize=18)
        ax.set_ylabel('loss', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)

        plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------    
            
    @property
    def get_exp_list(self):
        "return a list of the experiment names"
        print(self.keys)
    
    # -------------------------------------------------------------------------------
    
    @property
    def list_regions(self):
        "list the available regions in the user defined json file and in the regionmask package"
        # get all available regions
        user_regions = json.load(open(os.path.join(os.path.dirname(__file__), 'regions.json'), 'r'))
        import regionmask
        regionmask_regions = ['regionmask.defined_regions.ar6.all',
          'regionmask.defined_regions.natural_earth_v5_0_0.countries_10',
          'regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50',
          'regionmask.defined_regions.natural_earth_v5_0_0.us_states_10',
          'regionmask.defined_regions.srex',
          'regionmask.defined_regions.natural_earth_v5_0_0.land_10'
         ]
        # print these out
        print('user defined regions:\n\n', np.sort(list(user_regions.keys())), '\n\n')
        for regionmask_region in regionmask_regions:
            print(regionmask_region+':\n\n', np.sort(eval(regionmask_region+'.names')), '\n\n')
            
    # -------------------------------------------------------------------------------
    
    @property
    def data_operations_list(self):
        "print a list of the data processing operations available in the DataProcessing class"
        # import _data_processing and get the directories for the DataProcessing class
        import _data_processing as _dp
        _dp_dir = dir(_dp.DataProcessing)
        function_list = []
        for function in _dp_dir: # for each entry in the directory
            # ignore this function (else recursion issue) and others
            if not np.isin(function, ['data_operations_list', 'list_regions', 'load_data']):
                # check that it is a function (callable) and not built-in (__)
                if callable(getattr(_dp.DataProcessing, function)) and not function.startswith("__"):
                    function_list.append(function)
        print(function_list)
    
    # -------------------------------------------------------------------------------

    @property
    def list_variables(self):
        resolution_dictionary = {
            "daily": {},
            "monthly": {'GISS-E2-2-G': [2.0, 2.5], 'NorESM2-LM': [1.89, 2.5], 'ACCESS-ESM1-5': [1.25, 1.87], 'CanESM5-1': [2.79, 2.81], 'CMCC-CM2-HR4': [0.94, 1.25], 'CESM2': [0.94, 1.25], 'AWI-ESM-1-1': [1.86, 1.87], 'CAMS-CSM1': [1.12, 1.12], 'CMCC-ESM2': [0.94, 1.25], 'CESM2-WACCM': [0.94, 1.25], 'CAS-ESM2-0': [1.42, 1.41], 'MPI-ESM1-2-HR': [0.93, 0.93], 'AWI-CM-1-1': [0.93, 0.93], 'MIROC': [1.40, 1.41], 'MPI-ESM1-2-LR': [1.86, 1.87], 'GISS-E2-1-G': [2.0, 2.5], 'CNRM-CM6-1': [1.40, 1.41], 'CMCC-CM2-SR5': [0.94, 1.25], 'MRI-ESM2-0': [1.12, 1.12], 'MIROC-ES2H': [1.40, 1.41], 'CanESM5-CanOE': [2.79, 2.81], 'ACCESS-CM2': [1.25, 1.87], 'BCC-ESM1': [2.79, 2.81], 'UKESM1-1-LL': [1.25, 1.87], 'UKESM1-0-LL': [1.25, 1.87], 'CanESM5': [2.79, 2.81], 'BCC-CSM2': [1.12, 1.12], 'TaiESM': [0.94, 1.25], 'NorESM2-MM': [0.94, 1.25], 'MIROC-ES2L': [2.79, 2.81], 'ERA5': [0.25, 0.25], 'BEST': [1.0, 1.0]},
            "yearly": {'BEST-YEARLY': [1.0, 1.0], 'ERA5-YEARLY': [0.25, 0.25], 'GISTEMP': [2.5, 2.5], 'UDEL': [0.5, 0.5]}}

        for cadence in ['daily', 'monthly', 'yearly']:
            models = glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data/'+cadence+'/*'))
            models = np.sort([os.path.basename(models[i]) for i in range(len(models))])
            # initialize list
            allvars = []
            print('----------', cadence, '----------')
            for model in models:
                model_vars = glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data/'+cadence, model, '*/'))
                model_vars = [os.path.basename(model_vars[i][:-1]) for i in range(len(model_vars))]
                n_members = [len(glob.glob(os.path.join(os.path.dirname(__file__), 'raw_data/'+cadence, model, model_vars[i], '*'))) for i in range(len(model_vars))]
                print(model+":", *np.sort(model_vars), n_members, 'resolution:', resolution_dictionary[cadence][model])
            
        