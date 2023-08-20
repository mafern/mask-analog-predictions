import numpy as np
import xarray as xr
import gzip
import os
import _data_processing as dp
import pickle
from copy import deepcopy



def process_data(experiment_settings, overwrite_data=False):
    """
    description

    inputs:
        - experiment_settings: a dictionary containing all of the information for
        the experiment.
        create or load via the class/functions in experiments.py
    outputs:
        - processed datasets for this experiment
    """
    # parse the experiment_settings
    experiment_name = experiment_settings['experiment_name']

    # analog, training, validation, testing split
    # this will be a dictionary with keys: 'analog', 'training', 'validation', 'testing'
    # and values: the number of members for each --- if 0 or None use number corresponding
    # to the smallest ensemble amongst models selected
    split = experiment_settings['split']
    
    # random seed for selecting ensemble members
    seed = experiment_settings['seed']
    rng_seed = np.random.default_rng(seed)
    # total number of members requested
    member_rng = np.arange(np.sum(list(split.values())))
    
    # initialize dictionary that will be returned at the end of processing
    processed_data = {}
    # for each of the split sets (analog, training, validation, testing)
    for split_set in split.keys():
        try:
            # pretty stupid switch to make try fail if overwrite_data is True
            1/(overwrite_data-1)
            
            # try loading pickle files which are DataProcessing objects
            # one each for the different split sets (analog, training, validation, testing)
            # each of these contains the inputs and outputs
            with gzip.open(os.path.join("processed_data", experiment_name+'_'+split_set+'.pickle'), "rb") as filename:
                processed_data[split_set] = pickle.load(filename)

        # if file does not exist, process the data and save
        except:
            # list of models to use for the analog set, and for the training/validation/testing sets
            if split_set == 'analog':
                model_list = experiment_settings['analog_models']
            else:
                model_list = experiment_settings['soi_models']

            # randomly select ensembles members (which may not be all available data)
            rnd_inds = rng_seed.choice(member_rng, split[split_set], replace=False, shuffle=False)
            # remove selected indices from member_rng
            member_rng = np.setdiff1d(member_rng, rnd_inds)
            
            # load the appropriate ensemble members for the requested models/variables
            input_raw = dp.DataProcessing(model_list, experiment_settings['variable_list'], experiment_settings['input_period'], rnd_inds)
            # deepcopy a version for the outputs
            output_raw = deepcopy(input_raw)

            # initialize dictionary for this split set
            current_data = {"inputs": {}, "target": {}}
            # run some processing steps for the inputs
            print('#################################################################')
            print('processing', split_set, 'input dataset')
            print('#################################################################')
            for step in experiment_settings['input_proc_steps']:
                eval('input_raw.'+step)
                print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            # run some processing steps for the targets
            print('#################################################################')
            print('processing', split_set, 'target dataset')
            print('#################################################################')
            for step in experiment_settings['target_proc_steps']:
                eval('output_raw.'+step)
                print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            # add these to the current dictionary, but first flatten models and variables into single DataSet
            for var in experiment_settings["variable_list"]:
                current_data["inputs"][var] = xr.concat([input_raw.data_dictionary[var][model] for model in input_raw.data_dictionary[var].keys()], dim="member")
                current_data["target"][var] = xr.concat([output_raw.data_dictionary[var][model] for model in output_raw.data_dictionary[var].keys()], dim="member")
                
            if len(experiment_settings["variable_list"]) > 1:
                current_data["inputs"] = xr.merge([current_data["inputs"][var] for var in current_data["inputs"].keys()], join='inner')
                current_data["target"] = xr.merge([current_data["target"][var] for var in current_data["target"].keys()], join='inner')
            else:
                current_data["inputs"] = current_data["inputs"][var]
                current_data["target"] = current_data["target"][var]
                
            # save processed data in pickle format
            with gzip.open(os.path.join("processed_data", experiment_name+'_'+split_set+'.pickle'), "wb") as filename:
                pickle.dump(current_data, filename)

            # update dictionary to return
            processed_data[split_set] = current_data

    # get latitude and longitude, which may be useful
    processed_data['lat'] = processed_data[split_set]['inputs'].lat.to_numpy()
    processed_data['lon'] = processed_data[split_set]['inputs'].lon.to_numpy()
    
    return processed_data

# -------------------------------------------------------------------------------

def neural_network_stack(dataset):
    "stack datasets for neural network, final shape is [samples (times x members), lat, lon, channels]"
    # remove bnds dimension, replace NaNs with zeros
    inputs = dataset['inputs'].drop_dims("bnds").fillna(0)
    target = dataset['target'].drop_dims("bnds").fillna(0)
    
    # stack time and members as samples, convert to numpy array with correct shape
    inputs = inputs.stack(sample=("member", "time")).to_array().transpose("sample", "lat", "lon", "variable").to_numpy()
    target = target.stack(sample=("member", "time")).to_array().transpose("sample", "variable").to_numpy()
    ####################################################################################
    # NOT SURE IT MAKES SENSE TO HAVE MULTIPLE DIFFERENT VARIABLES GO INTO THE NETWORK
    ####################################################################################
    print('input and target datasets reshaped to', np.shape(inputs), 'and', np.shape(target))
    return inputs, target

# -------------------------------------------------------------------------------

def combine_ncs(directory, nfiles, delete_old=False):
    """
    combine multiple subfiles into a single nc spanning the full time range.
    assumes that the filenames allow for sorting in the right order, and that
    they are split amongst the same number of subfiles.
    name will be based on the first and last of these subfiles.
    """
    all_files = np.sort(glob.glob(os.path.join(directory, '*')))
    indrange = np.linspace((0, nfiles), (allf.size-nfiles, all_files.size), int(all_files.size/nfiles), dtype=int)
    for inds in indrange:
        for ii in range(inds[0], inds[1]):
            # load each and concat
            ds = xr.load_dataset(all_files[ii])
            if ii == inds[0]:
                ds_complete = ds
                # file save name start
                fname_start = all_files[ii][:-9]
            else:
                ds_complete = xr.concat([ds_complete, ds], dim="time")

            # optionally delete old files
            if delete_old:
                os.remove(all_files[ii])
        # file save name ending
        fname_end = all_files[ii][-9:]
        # save as .nc
        ds_complete.to_netcdf(path=fname_start+fname_end)

# -------------------------------------------------------------------------------

def make_model_masks(file):
    """
    using a land-only variable, make a mask for the given model.
    will make both a land mask and an ocean mask and save these as .nc files.

    inputs:
        - file: the path and name of a file for a model and variable that is
        land-only.
    """
    # load land-only variable
    mask_var = xr.load_dataarray(file).squeeze()
    # get a land mask and ocean mask
    landmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, np.nan, 1.0)
    oceanmask = xr.where(abs(mask_var.isel(time=0)) >= 0.0, 1.0, np.nan)

    # save in the appropriate directory, with the name landmask.nc and oceanmask.nc
    land_xr = xr.Dataset(
        data_vars=dict(mask=(["lat", "lon"], landmask.values)),
        coords=dict(
            lat=(["lat"], landmask.lat.values),
            lon=(["lon"], landmask.lon.values))
    )
    land_xr.to_netcdf(os.path.join(*file.split('/')[:-2], 'landmask.nc'))

    ocean_xr = xr.Dataset(
        data_vars=dict(mask=(["lat", "lon"], oceanmask.values)),
        coords=dict(
            lat=(["lat"], oceanmask.lat.values),
            lon=(["lon"], oceanmask.lon.values))
    )
    ocean_xr.to_netcdf(os.path.join(*file.split('/')[:-2], 'oceanmask.nc'))
