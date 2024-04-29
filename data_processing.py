import numpy as np
import xarray as xr
import gzip
import os
import glob
import _data_processing as _dp
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
            ###################################################################################
            # pretty stupid switch to make try fail if overwrite_data is True
            1/(overwrite_data-1)
            ###################################################################################
            
            # try loading pickle files which are dictionaries with both inputs and target
            # for each of the different split sets (analog, training, validation, testing)
            with gzip.open(os.path.join(os.path.dirname(__file__), "processed_data", experiment_name+'_'+split_set+'.pickle'), "rb") as filename:
                processed_data[split_set] = pickle.load(filename)
            print('loading pre-saved', split_set, 'dataset')

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
            input_raw = _dp.DataProcessing(model_list, experiment_settings['variable_list'], experiment_settings['data_period'], rnd_inds)
            # deepcopy a version for the outputs
            output_raw = deepcopy(input_raw)

            # initialize dictionary for this split set
            current_data = {"inputs": {}, "target": {}, "lat": {}, "lon": {}, "land_ocean_mask": {}}
            # run some processing steps for the inputs
            print('#################################################################')
            print('processing', split_set, 'input dataset')
            print('#################################################################')
            for step in experiment_settings['inputs_proc_steps']:
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

            # add processed data to the current_data dictionary, but first flatten models and variables into single DataSet
            for var in experiment_settings["variable_list"]:
                # preserve the model of each member for downstream use
                for model in model_list:
                    # make a list of the model name repeating for the number of members
                    repeat_model = [model] * rnd_inds.size
                    # add the member indices as a coordinate
                    input_raw.data_dictionary[var][model] = input_raw.data_dictionary[var][model].assign_coords(member=(rnd_inds))
                    output_raw.data_dictionary[var][model] = output_raw.data_dictionary[var][model].assign_coords(member=(rnd_inds))
                    # and add the models as a coordinate
                    input_raw.data_dictionary[var][model] = input_raw.data_dictionary[var][model].assign_coords(model=('member', repeat_model))
                    output_raw.data_dictionary[var][model] = output_raw.data_dictionary[var][model].assign_coords(model=('member', repeat_model))
                    
                # now flatten over models
                current_data["inputs"][var] = xr.concat([input_raw.data_dictionary[var][model] for model in input_raw.data_dictionary[var].keys()],
                                                        dim="member", coords="minimal")
                current_data["target"][var] = xr.concat([output_raw.data_dictionary[var][model] for model in output_raw.data_dictionary[var].keys()],
                                                        dim="member", coords="minimal")
                
            # flatten variables into single dataset (if there is more than one variable)
            ###################################################################################
            ### this could be problematic if the resolution for the variables are different
            ###################################################################################
            if len(experiment_settings["variable_list"]) > 1:
                current_data["inputs"] = xr.merge([current_data["inputs"][var] for var in current_data["inputs"].keys()], join='inner')
                current_data["target"] = xr.merge([current_data["target"][var] for var in current_data["target"].keys()], join='inner')
            else:
                current_data["inputs"] = current_data["inputs"][var]
                current_data["target"] = current_data["target"][var]
                
            # get latitude and longitude, which will be useful
            current_data['lat'] = current_data['inputs'].lat.to_numpy()
            current_data['lon'] = current_data['inputs'].lon.to_numpy()
            # if a land or ocean mask was used, carry that over
            if experiment_settings['mask']:
                current_data['land_ocean_mask'] = input_raw.final_mask
            else:
                current_data['land_ocean_mask'] = xr.DataArray(data=1)
                
            # save processed data in pickle format
            with gzip.open(os.path.join(os.path.dirname(__file__), "processed_data", experiment_name+'_'+split_set+'.pickle'), "wb") as filename:
                pickle.dump(current_data, filename)

            # update dictionary to return
            processed_data[split_set] = current_data
    
    return processed_data

# -------------------------------------------------------------------------------

def neural_network_stack(dataset):
    "stack datasets for neural network, final shape is [samples (times x members), lat, lon, channels]"
    # replace NaNs with zeros
    inputs = dataset['inputs'].fillna(0)
    target = dataset['target'].fillna(0)

    # stack time and members as samples, convert to numpy array with correct shape
    inputs = inputs.stack(sample=("member", "time")).to_array().transpose("sample", "lat", "lon", "variable").to_numpy()
    target = target.stack(sample=("member", "time")).to_array().transpose("sample", "variable").to_numpy()
    
    print('input and target datasets reshaped to', np.shape(inputs), 'and', np.shape(target))
    return inputs, target

# -------------------------------------------------------------------------------

def combine_ncs(directory, delete_old=False):
    """
    combine multiple subfiles into a single nc spanning the full time range.
    assumes that the filenames allow for sorting in the right order, and that
    the file names end CMIP6 style (ending in time range).
    name will be based on the first and last of these subfiles.
    """
    # get the files from the requested directory
    files = np.sort(glob.glob(os.path.join(directory, '*')))
    # get a list of the files without the dates to identify duplicates (same simulation,
    # different times) -- could update this to find date-like name part and go from there
    cut_files = [file[:-16] for file in files]
    # now loop through the unique simulations and combine and save
    for file in np.unique(cut_files):
        # get all the subfile indices corresponding to this simulation
        simulation_inds = np.where(file == np.array(cut_files))[0]
        # if multiple subfiles, go through concatenating them together, else skip
        if np.size(simulation_inds) > 1:
            # concatenate all the subfiles together
            for index in simulation_inds:
                # open, concat, repeat
                ds = xr.load_dataset(files[index])
                if index == simulation_inds[0]:
                    ds_complete = ds
                    # file save name start
                    fname_start = files[index][:-9]
                else:
                    ds_complete = xr.concat([ds_complete, ds], dim="time")
                # optionally delete old files
                if delete_old:
                    os.remove(files[index])
            # file save name ending
            fname_end = files[index][-9:]
            # save as .nc
            ds_complete.to_netcdf(path=fname_start+fname_end)
            print('using indices:', *simulation_inds, 'saving to', fname_start+fname_end)

# -------------------------------------------------------------------------------

def combine_hist_ssp(directory, delete_old=False, grid='gn'):
    """
    combine multiple historical and ssp subfiles into a single nc spanning the full time range.
    assumes that the filenames allow for sorting in the right order, and that
    the file names end CMIP6 style (ending in time range).
    """
    # get the files from the requested directory
    files = np.sort(glob.glob(os.path.join(directory, '*')))
    # get a list of the member ids without the dates to identify matching historical and ssp runs
    mem_ids = np.array([file[file.find('_r')+1:file.find('_'+grid)] for file in files])
    # go through each file and find match, concatenate, save, optionally delete separate files
    for ii, file in enumerate(files):
        if 'historical' in file:
            non_ii_inds = np.setdiff1d(np.arange(mem_ids.size), ii)
            # add one since we remove the current index above (it's ok, we are always looking forward)
            ssp_ind = np.where(mem_ids[ii] == mem_ids[non_ii_inds])[0] + 1
            if np.size(ssp_ind) != 0:
                # now load the two datasets
                hist = xr.load_dataset(files[ii])
                ssp = xr.load_dataset(files[ssp_ind][0])
                combined = xr.concat([hist, ssp], dim='time')
                # get the file name
                fname_start = files[ssp_ind][0][:-16]
                fname_mid = files[ii][-16:-9]
                fname_end = files[ssp_ind][0][-9:]
                # save the combined file
                combined.to_netcdf(path=fname_start+fname_mid+fname_end)
                if delete_old:
                    os.remove(files[ii])
                    os.remove(files[ssp_ind][0])

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
