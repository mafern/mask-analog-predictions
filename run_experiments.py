import json
import numpy as np
import os
import glob
import sys
import make_mask as mask
import data_processing as dp
import build_model
import train_model
import random
import tensorflow as tf
from pprint import pprint


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
            self.experiments = json.load(open(self.filename, 'r'))
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
        with open(filename, "w") as exp_file:
            exp_file.write(json.dumps(self.experiments))
            
    # -------------------------------------------------------------------------------

    def restore_from_backup(self, backupfile, *, confirm=0):
        "Overwrite the main saved file with the dictionary in the backup file"
        if confirm == 1: # just to make sure
            backup = json.load(open(backupfile, 'r'))
            with open(self.filename, "w") as exp_file:
                exp_file.write(json.dumps(backup))
            self.experiments = json.load(open(self.filename, 'r'))
        else:
            print("must pass confirm=1 to overwrite main file with backup")
            
    # -------------------------------------------------------------------------------

    def make_exp_dictionary(self, experiment_name, *, variable_list, analog_models, soi_models, leadtime, split, prediction_nodes, prediction_activation, learning_rate, mask=False, output_period=False, data_period='monthly', roll_period=False, region=False, resolutions=None, season=False, time_range=None, norm_data=True, baseline=False, remove_trend=False, remove_seasonality=False, avg_target=False, loss='mse', patience=50, min_delta=0.0005, batch_size=64, val_batch_size=2500, max_epochs=5000, mask_activation='relu', mask_init="ones", mask_l1=0.0, mask_l2=0.0, norm_weights=True, seed=0):
        ###################################################################################
        # COULD ADD PROCESSING_ORDER INPUT THAT OVERRIDES DEFAULT ORDER
        
        # things that could be added with additional functionality
        # "input_var": "ts",
        # "target_var": "ts",
        # "scenario": "historical",  # ("historical", "rcp45", "both)
        # "rng_seed_list": list(range(0,100, 10))
        ###################################################################################
        """
        Make an experiment dictionary with the following settings:
            REQUIRED INPUTS
                - experiment_name: name of the experiment (how it will be saved and called)
                - variable_list: list of variable names to be included in the data
                - analog_models: which models to use for the analog split (run self.list_variables to see available options)
                - soi_models: which models to use for the training, validation, and testing splits (run self.list_variables to see available options)
                - leadtime: time shift between inputs and target (should be a positive integer)
                - split: list with integer number of members to include in each split of the data (order should be: analog, training, validation, testing)
                - prediction_nodes: a list where the number of entries is the number of layers, and the values are the number of nodes for that layer #### GIVE THIS A DEFAULT ONCE A GOOD DEFAULT IS DETERMINED
                - prediction_activation: activation function used for the network #### GIVE THIS A DEFAULT ONCE A GOOD DEFAULT IS DETERMINED
                - learning_rate: step size for the training #### GIVE THIS A DEFAULT ONCE A GOOD DEFAULT IS DETERMINED
                                
            OPTIONAL INPUTS
                - mask: whether to mask land (pass 'land'), ocean (pass 'ocean'), or neither (pass False, default)
                - region: the name (run self.list_regions to get available names) or list of boundary coordinates, i.e. [minlon, minlat, maxlon, maxlat] for the target region. To skip this processing step, pass False (default).
                - output_period: either resample the data to a new period (pass a string such as "2M"), resample on the same period (pass None) or do nothing (pass False, default)
                - data_period: string containing the period for the raw data, should be 'daily' or 'monthly' (default) or 'yearly'
                - roll_period: calculate rolling mean for the data (pass a positive integer for number of time steps to roll), or do nothing (pass False, default)
                - resolutions: resample spatial grid of data, either to coarsest resolution amongst models (pass None, default), to a specified resolution (pass a list [lat_res, lon_res]), or do nothing (pass False)
                - season: calculate a yearly season from the data (pass a list with entries [season_length, season_start]), or do nothing (pass False, default)
                - time_range: sync the time ranges between models, to the common time range (pass None, default), to a specified time range (pass a list [start_date, end_date]), or do nothing (pass False)
                - norm_data: whether to normalize the data by its mean and std (default is True)
                - baseline: normalize the datasets by a baseline period (pass a list of a start and end year for the baseline, e.g. [1951, 1980]) or do nothing (pass False, default)
                - remove_trend: whether to remove the trend via the ensemble mean (pass True) or do nothing (pass False, default)
                - remove_seasonality: whether to remove the seasonal cycle (pass True), or do nothing (pass False, default)
                - avg_target: whether to average the target over lat/lon for all variables (pass True), or do nothing (pass False, default)
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
                - seed: random seed to use in the member selection (###### and possibly the network training)
                
        Returns a dictionary containing all of the requested experiment settings, including the processing steps (in a sensable order).
        """
        # check if the proposed experiment name already exists and inputs are sensible
        assert experiment_name not in self.keys, "experiment with that name already exists"
        assert leadtime > 0, "leadtime should be positive for projections"
        assert roll_period > 0 or roll_period is False, "roll_period should positive or False"
        assert np.isin(mask, ['land', 'ocean', False]), "mask should be 'land' or 'ocean' or False"
        assert np.isin(data_period, ['daily', 'monthly', 'yearly']), "data_period must be 'daily' or 'monthly' or 'yearly'"
        
        # fill a dictionary with the data split values
        split = {"analog": split[0], "training": split[1], "validation": split[2], "testing": split[3]}

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
                               "shift_inout_time(-experiment_settings['leadtime'])": leadtime,
                               "remove_trend()": remove_trend,
                               "norm_data()": norm_data}
        
        target_process_dict = {"select_season(season=experiment_settings['season'])": season,
                               "smooth_time(period=experiment_settings['output_period'])": output_period,
                               "sync_time(experiment_settings['time_range'])": time_range,
                               "mask_data(experiment_settings['mask'])": mask,
                               "smooth_space(experiment_settings['resolutions'])": resolutions,
                               "get_region(experiment_settings['region'])": region,
                               "baseline_norm(experiment_settings['baseline'])": baseline,
                               "regional_mean(experiment_settings['variable_list'])": avg_target,
                               "remove_seasonality()": remove_seasonality,
                               "rolling_time(-experiment_settings['roll_period'])": roll_period,
                               "shift_inout_time(experiment_settings['leadtime'])": leadtime,
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
        print('----------------------------------------------------------------')
        # stack/separate into inputs and target for the analog, training, validation, and testing sets
        analog_inputs, analog_target = dp.neural_network_stack(self.processed_data['analog'])
        training_inputs, training_target = dp.neural_network_stack(self.processed_data['training'])
        validation_inputs, validation_target = dp.neural_network_stack(self.processed_data['validation'])
        testing_inputs, testing_target = dp.neural_network_stack(self.processed_data['testing'])
        # get the lat and lon ------ THIS ASSUMES THAT THE LAT AND LON ARE THE SAME BETWEEN THE ANALOG AND TRAINING SETS
        lat, lon = self.processed_data['analog']['lat'], self.processed_data['analog']['lon']
        print('----------------------------------------------------------------')
        return [analog_inputs, analog_target, training_inputs, training_target,
                validation_inputs, validation_target, testing_inputs, testing_target, lat, lon]
    
    # -------------------------------------------------------------------------------
    
    def train_experiment_model(self, experiment_name, overwrite_data=False, overwrite_model=False):
        # get settings
        experiment_settings = self.get_experiment(experiment_name)
        # get data
        (analog_inputs, analog_target,
         training_inputs, training_target,
         validation_inputs, validation_target,
         testing_inputs, testing_target,
         lat, lon) = self.get_experiment_data(experiment_name, overwrite_data=overwrite_data)
        
        ###################################################################################
        # do i want to do multiple seeds? not for most of the results, as it doesn't matter
        # very much -- will want to show this though, so keep this code commented out, and
        # when showing it keep the seeds separate to show lack fo variance
        # will also need to change the make_mask.py code to remove multiple seeds
        models = []
        for seed in range(0, 100, 10):
            experiment_settings["seed"] = seed
        ###################################################################################
        
            # ensure model names are consistent, etc.
            tf.keras.backend.clear_session()
            np.random.seed(experiment_settings["seed"])
            random.seed(experiment_settings["seed"])
            tf.random.set_seed(experiment_settings["seed"])

            # build the full model for learning the mask: mask, dissimilarity, prediction models
            model, _, _, _ = build_model.build_interp_model(experiment_settings, [training_inputs, analog_inputs])

            # train the model, which will check if the model is already saved first
            # (then load if it does, train and save if it does not)
            model, fit_summary, history = train_model.train_model(
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
            pprint(fit_summary, width=80)
            
            ###################################################################################
            models.append(model)
            ###################################################################################
        return models

    # -------------------------------------------------------------------------------
    
    def make_mask(self, experiment_name, overwrite_data=False, overwrite_model=False, overwrite_mask=False):
        # function wrapping make_mask function from make_mask.py
        # get model (train or load) and the processed data
        models = self.train_experiment_model(experiment_name, overwrite_data=overwrite_data, overwrite_model=overwrite_model)
        
        # get the mask (both a raw version 'raw_mask' and a scaled/masked version 'mask')
        self.mask = mask.make_mask(experiment_name, models, self.processed_data, overwrite_mask=overwrite_mask)
        return self.mask
        
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
        user_regions = json.load(open('regions.json', 'r'))
        import regionmask
        regionmask_regions = ['regionmask.defined_regions.ar6.all',
          'regionmask.defined_regions.natural_earth_v5_0_0.countries_10',
          'regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50',
          'regionmask.defined_regions.natural_earth_v5_0_0.us_states_10',
          'regionmask.defined_regions.srex'
         ]
        # print these out
        print('user defined regions:\n\n    ', list(user_regions.keys()), '\n\n')
        for regionmask_region in regionmask_regions:
            print(regionmask_region+':\n\n    ', eval(regionmask_region+'.names'), '\n\n')
            
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
        ###################################################################################
        # get list of models and associated variables ----- only looking in monthly for now
        ###################################################################################
        models = glob.glob('raw_data/monthly/*')
        models = [os.path.basename(models[i]) for i in range(len(models))]
        # initialize list
        allvars = []
        for model in models:
            model_vars = glob.glob(os.path.join('raw_data/monthly', model, '*/'))
            model_vars = [os.path.basename(model_vars[i][:-1]) for i in range(len(model_vars))]
            print(model+":", *np.sort(model_vars))
        
        