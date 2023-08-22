import json
import numpy as np
import os
import glob
import sys
import make_mask
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

###################################################################################
def add_region(region_name, lats, lons):
    "add a region to regions.json with the given lat and lon bounds"
    new_region = {region_name: {"lats": lats, "lons": lons}}
    # open regions.json and add, save, close
    
    print(region_name, 'added to regions.json')
###################################################################################

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

    ###################################################################################
    # "rng_seed_list": list(range(0,100, 10))
    # things that could be added with additional functionality
    # "feature_var": "ts",
    # "target_var": "ts",
    # "scenario": "historical",  # ("historical", "rcp45", "both)
    ###################################################################################
    def make_exp_dictionary(self, experiment_name, *, variable_list, region, leadtime, prediction_nodes, prediction_activation, learning_rate, mask=False, output_period=False, data_period='monthly', analog_models=None, soi_models=None, split=None, roll_period=False, resolutions=None, season=False, time_range=None, norm_data=True, remove_trend=False, remove_seasonality=False, avg_target=False, loss='mse', patience=50, min_delta=0.0005, batch_size=64, val_batch_size=2500, max_epochs=5000, mask_activation='relu', mask_init="ones", mask_l1=0.0, mask_l2=0.0, norm_weights=True, seed=0):
        ###################################################################################
        # COULD ADD PROCESSING_ORDER INPUT THAT OVERRIDES DEFAULT ORDER
        ###################################################################################
        """
        Make an experiment dictionary with the following settings:
            REQUIRED INPUTS
                - experiment_name: name of the experiment (how it will be saved and called)
                - variable_list: 
                - region: 
                - leadtime: 
                - prediction_nodes: 
                - prediction_activation: 
                - learning_rate: 
                
                
                # "select_season(season=experiment_settings['season'])", #X only do if season is passed (not False) (both)
                # "smooth_time(period=experiment_settings['output_period'])", #X do nothing (False), resample (None), or use period if passed to output_period (both)
                # "sync_time(experiment_settings['time_range'])" #X default to common time range (None) unless False or time range is passed (both)
                # "mask_data(experiment_settings['mask'])", #X mask land ('land'), ocean ('ocean'), or neither (False) (both)
                # "smooth_space(experiment_settings['resolutions'])", #X default (None) coarsest resolution unless resolutions is passed or False (both)
                # "get_region(experiment_settings['region'])", #X get whatever region is passed to region (only target) or False
                # "regional_mean(experiment_settings['variable_list'])", #X include if avg_target is True (only target)   # this could be an issue if we want to average some variables but not others (either), assume all variables for now and only for the target (reproduce Jamin's results, but probably will use full maps from now on)
                # "rolling_time(experiment_settings['roll_period'])", #X default is do nothing (False), else input (negative for target, positive for input) (both)
                # "shift_inout_time(-experiment_settings['leadtime'])", #X whatever the input to leadtime is (negative for input, positive for target) (both)
                # "remove_seasonality()", #X remove_seasonality switch  # has an option, but not needed at this point (both?)
                # "remove_trend()", #X remove_trend switch  # has options, but not needed at this point (both?)
                # "norm_data()" #X norm_data switch (both?)
                
                
            OPTIONAL INPUTS
                - mask: 
                - output_period: 
                - data_period: 
                - analog_models: 
                - soi_models: 
                - split: 
                - roll_period: 
                - resolutions: 
                - season: 
                - time_range: 
                - norm_data: 
                - remove_trend: 
                - remove_seasonality: 
                - avg_target: 
                - loss: 
                - patience: 
                - min_delta: 
                - batch_size: 
                - val_batch_size: 
                - max_epochs: 
                - mask_activation: 
                - mask_init: 
                - mask_l1: 
                - mask_l2: 
                - norm_weights: 
                - seed: 
                
        Returns a dictionary containing all of the requested experiment settings, including the processing steps (in a sensable order).
        """
        ###################################################################################
        # check if the proposed experiment name already exists
        # assert experiment_name not in self.keys, "experiment with that name already exists"
        # assert leadtime > 0
        # assert roll_period > 0 or roll_period is None
        # assert np.isin(mask, ['land', 'ocean', None])
        # assert variable_list is in the raw_data (see property at end)
        # assert output_period is integer then DMY
        # assert data_period is in names in raw_data
        # assert analog_model are in available models (see property at end)
        # assert split adds up to available members
        # assert resolutions is 2 positive values
        # assert season is None or season is 2 integers (any integer, 1-12)
        # assert time_range is None or time_range is 2 strings
        ###################################################################################
        
        if split is None:
            ###################################################################################
            # use a fraction system?
            pass
            ###################################################################################
        else:
            # fill a dictionary
            split = {"analog": split[0], "training": split[1], "validation": split[2], "testing": split[3]}
            
        # get locals before assigning anything here
        function_inputs = locals()
        
        inputs_process_dict = {"select_season(season=experiment_settings['season'])": season, "smooth_time(period=experiment_settings['output_period'])": output_period,
                       "sync_time(experiment_settings['time_range'])": time_range, "mask_data(experiment_settings['mask'])": mask,
                       "smooth_space(experiment_settings['resolutions'])": resolutions, "rolling_time(experiment_settings['roll_period'])": roll_period,
                       "shift_inout_time(-experiment_settings['leadtime'])": leadtime, "remove_seasonality()": remove_seasonality,
                       "remove_trend()": remove_trend, "norm_data()": norm_data}
        
        target_process_dict = {"select_season(season=experiment_settings['season'])": season, "smooth_time(period=experiment_settings['output_period'])": output_period,
                       "sync_time(experiment_settings['time_range'])": time_range, "mask_data(experiment_settings['mask'])": mask,
                       "smooth_space(experiment_settings['resolutions'])": resolutions, "get_region(experiment_settings['region'])": region,
                       "regional_mean(experiment_settings['variable_list'])": avg_target, "rolling_time(-experiment_settings['roll_period'])": roll_period,
                       "shift_inout_time(experiment_settings['leadtime'])": leadtime, "remove_seasonality()": remove_seasonality,
                       "remove_trend()": remove_trend, "norm_data()": norm_data}
        
        # make the processing list for the input data
        inputs_proc_steps = []
        for key, value in inputs_process_dict.items():            
            if value is False:
                pass
            else:
                inputs_proc_steps.append(key)
        
        # make the processing list for the target data
        target_proc_steps = []
        for key, value in target_process_dict.items():            
            if value is False:
                pass
            else:
                target_proc_steps.append(key)
            
        experiment_settings = {}
        # simpler way, does not yet include processing lists
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
        # get the lat and lon ------ THIS ASSUMES THAT THE LAT AND LON ARE THE SAME BETWEEN THE ANAOLG AND TRAINING, ETC. SETS
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
        # do i want to do multiple seeds and take an average?
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
    
    # function wrapping make_mask function from make_mask.py
    def run_make_mask(self, experiment_name, overwrite_data=False, overwrite_model=False, overwrite_mask=False):
        # get model (train or load) and the processed data
        ###################################################################################
        models = self.train_experiment_model(experiment_name, overwrite_data=overwrite_data, overwrite_model=overwrite_model)
        ###################################################################################
        # get the mask (both a raw version 'raw_mask' and a scaled/masked version 'mask')
        self.mask = make_mask.make_mask(experiment_name, models, self.processed_data, overwrite_mask=overwrite_mask)
        return self.mask
        
    # -------------------------------------------------------------------------------
            
    ###################################################################################
    ### below is a function that calls the analog selection code which gets the analogs and puts them in an xarray and saves that?
    ### (even better, it just adds a new column of information rating the best to the worst analogs) -- what to save here though?
    ##### select the best analogs and save them
    ##### make a projection based on these analogs
    ###################################################################################
            
    # -------------------------------------------------------------------------------    
    
    ###################################################################################
    # plot the mask
    # model_diagnostics.visualize_interp_model(experiment_settings, weights_val, lat, lon)
    ###################################################################################
        
    # -------------------------------------------------------------------------------    
            
    @property
    def get_exp_list(self):
        "return a list of the experiment names"
        return self.keys
    
    # -------------------------------------------------------------------------------
    
    ###################################################################################
    # could make these a branching list of what is available
    @property
    def list_models(self):
        # get all models ----- only looking in monthyl for now, but easy to expand to others
        models = glob.glob('raw_data/monthly/*')
        return [os.path.basename(models[i]) for i in range(len(models))]
    
    # -------------------------------------------------------------------------------
        
    @property
    def list_variables(self):
        # get list of models
        models = self.list_models
        # initialize list
        allvars = []
        for model in models:
            model_vars = glob.glob(os.path.join('raw_data/monthly', model, '*/'))
            model_vars = [os.path.basename(model_vars[i][:-1]) for i in range(len(model_vars))]
            allvars.extend(model_vars)
            
        return list(np.unique(allvars))
    ###################################################################################
        
        