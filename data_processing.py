import numpy as np
import xarray as xr
import os
import glob
import _data_processing as _dp
from copy import deepcopy

FLOAT_TYPE = np.float32
INT_TYPE = np.int32

# RAW_DATA_PATH = os.path.dirname(__file__)
RAW_DATA_PATH = '/Volumes/Data/Martin_Fernandez/'

# -------------------------------------------------------------------------------

def preprocess_data(resolution, var_list=-1, model_list=-1, overwrite_data=False, scratch=False, res_check=True, stat=None):
    """
    spatially and temporally smooth the requested data and save;
    will ignore models that require upsampling based on the resolution

    Input:
        - resolution: [lat, lon] resolution
        - var_list: list of variables to process, -1 means all available
        - model_list: list of models to process, -1 means all available
        - overwrite_data: whether to overwrite any currently saved data that matches the given inputs
    """
    # get model list, if not specified
    if model_list == -1:
        model_list = glob.glob(os.path.join(RAW_DATA_PATH, 'raw_data/*'))
        model_list = np.sort([os.path.basename(model_list[i]) for i in range(len(model_list))])

    # loop through each model in model_list and preprocess
    for model in model_list:
        # get the available variables for this model
        if var_list == -1:
            var_list = glob.glob(os.path.join(RAW_DATA_PATH, 'raw_data/'+model+'/*'))
            var_list = np.sort([os.path.basename(var_list[i]) for i in range(len(var_list))])
        # check that the resolution of this model is not smaller than requested -- if so, skip
        # (do not want to downscale at this point, perhaps in the future)
        model_res_dict = get_model_res_dictionary(scratch=scratch)
        if res_check and np.any(np.array(model_res_dict[model]) > np.array(resolution)):
            print("requested resolution would require downscaling, skipping", model)
            continue

        # for each variable requested
        for var in var_list:
            # check if this combination of models, variables, and resolution has already been preprocessed
            preprocessed_folder = os.path.join(os.path.dirname(__file__), "processed_data", var+'_lat'+str(np.round(resolution[0], 2))+'_lon'+str(np.round(resolution[1], 2)))
            if stat is not None:
                preprocessed_folder = os.path.join(os.path.dirname(__file__), "processed_data", var+stat+'_lat'+str(np.round(resolution[0], 2))+'_lon'+str(np.round(resolution[1], 2)))
            # if the folder does not yet exist, create it
            if not os.path.exists(preprocessed_folder):
                os.mkdir(os.path.join(os.path.dirname(__file__), "processed_data", preprocessed_folder))
            preprocessed_filename = os.path.join(preprocessed_folder, model+'.nc')
            if not overwrite_data and os.path.exists(preprocessed_filename):
                print("already exists, skipping", model, var)
                continue
            # get available members for this model and variable
            files = np.sort(glob.glob(os.path.join(RAW_DATA_PATH, 'raw_data', model, var, '*')))
            # check that this model has this variable (if not, skip to next)
            if files.size == 0:
                print("variable not available, skipping", model, var)
                continue

            # now that we are actually doing some preprocessing, print model, var, number of members
            print('preprocessing', files.size, var, 'members from model', model)

            # define the preprocessing steps
            preprocess_steps = ["smooth_time(stat)",
                                "smooth_space(resolution)"]
            # get model dictionary from raw data
            model_member_dict = {model_path.split('/')[-1]: np.arange(len(glob.glob(model_path+f'/{var}/*'))) for model_path in np.sort(glob.glob(os.path.join(RAW_DATA_PATH+'raw_data/*'))) if len(glob.glob(model_path+f'/{var}/*')) > 0}
            # initialize the processing object, loads the data
            input_raw = _dp.DataProcessing([model], [var], list(np.arange(files.size)), 0, resolution, load_raw=True, model_member_dict=model_member_dict)

            # run some processing steps for the inputs
            print('#################################################################')
            print('processing', model, var, 'dataset')
            print('#################################################################')
            for step in preprocess_steps:
                eval('input_raw.'+step)
                print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            # get the xarray out of the dictionary
            input_raw = input_raw.data_dictionary[var][model]
            # make a list of the model name repeating for the number of members
            repeat_model = [model] * files.size
            # add the member indices as a coordinate
            input_raw = input_raw.assign_coords(member=(np.arange(files.size)))
            # and add the models as a coordinate
            input_raw = input_raw.assign_coords(model=('member', repeat_model))
            # re-type the data to 32 bit
            input_raw = input_raw.assign_coords(lat=input_raw['lat'].astype(FLOAT_TYPE),
                                                lon=input_raw['lon'].astype(FLOAT_TYPE),
                                                member=input_raw['member'].astype(INT_TYPE)).astype(FLOAT_TYPE)
            # save processed data
            input_raw.to_netcdf(preprocessed_filename)

# -------------------------------------------------------------------------------

def process_data(experiment_settings):
    """
    process previously saved data (which has been smoothed to yearly and smoothed to a common resolution)

    inputs:
        - experiment_settings: a dictionary containing all of the information for
        the experiment.
        create or load via the class/functions in experiments.py
    outputs:
        - processed datasets for this experiment
    """
    # analog, training, validation splits (process in reverse of this order)
    split = experiment_settings['split']
    resolution = experiment_settings['resolutions']
    variable_list = experiment_settings['variable_list']
    # set up local version of model/member dictionary, allows random selection of members without repeating
    model_member_dict = {raw_data_dir.split('/')[-1][:-3]: xr.load_dataset(raw_data_dir).member.to_numpy().astype(int) for raw_data_dir in np.sort(glob.glob(os.path.join(os.path.dirname(__file__), f'processed_data/{variable_list[0]}_lat{np.round(resolution[0], 2)}_lon{np.round(resolution[1], 2)}', '*')))}
    # random seed for selecting ensemble members
    seed = experiment_settings['seed']
    
    # initialize dictionary that will be returned at the end of processing
    processed_data = {}
    # for each of the split sets (analog, training, validation)
    for split_set in list(split.keys()):
        # list of models to use for the analog set, training, and validation sets
        if split_set == 'analog':
            model_list = experiment_settings['analog_models']
        elif split_set == 'training':
            model_list = experiment_settings['soi_models']
        else:
            model_list = experiment_settings['val_models']
        
        # load the appropriate ensemble members for the requested models/variables
        input_raw = _dp.DataProcessing(model_list, experiment_settings['variable_list'], split[split_set], seed, experiment_settings['resolutions'], model_member_dict)
        if np.isin(experiment_settings['target_var'][0], experiment_settings['variable_list']):
            # deepcopy a version for the outputs
            output_raw = deepcopy(input_raw)
            # variables that should be removed (non-target variables)
            remove_vars = np.setdiff1d(experiment_settings['variable_list'], experiment_settings['target_var'])
            # remove them
            for drop in remove_vars:
                output_raw.data_dictionary.pop(drop)
        # target variable is not one of the input variables
        else:
            output_raw = _dp.DataProcessing(model_list, experiment_settings['target_var'], split[split_set], seed, experiment_settings['resolutions'], model_member_dict)
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
            for model in input_raw.data_dictionary[var].keys():
                used_members = np.setdiff1d(model_member_dict[model], input_raw.model_member_dict[model])
                # make a list of the model name repeating for the number of members
                repeat_model = [model] * np.size(used_members)
                # add the member indices as a coordinate
                input_raw.data_dictionary[var][model] = input_raw.data_dictionary[var][model].assign_coords(member=(used_members))
                # and add the models as a coordinate
                input_raw.data_dictionary[var][model] = input_raw.data_dictionary[var][model].assign_coords(model=('member', repeat_model))
            # now flatten over models
            current_data["inputs"][var] = xr.concat([input_raw.data_dictionary[var][model] for model in input_raw.data_dictionary[var].keys()],
                                                    dim="member", coords="minimal")
        # flatten input variables into single dataset (if there is more than one variable)
        if len(experiment_settings["variable_list"]) > 1:
            # multiple variables, but same resolution for all
            current_data["inputs"] = xr.merge([current_data["inputs"][var] for var in current_data["inputs"].keys()], join='inner')
            # get latitude and longitude
            current_data['lat'] = current_data['inputs'].lat.to_numpy().astype(FLOAT_TYPE)
            current_data['lon'] = current_data['inputs'].lon.to_numpy().astype(FLOAT_TYPE)
        # if there is only a single variable, continue on
        else:
            current_data["inputs"] = current_data["inputs"][var]
            # get latitude and longitude
            current_data['lat'] = current_data['inputs'].lat.to_numpy().astype(FLOAT_TYPE)
            current_data['lon'] = current_data['inputs'].lon.to_numpy().astype(FLOAT_TYPE)

        # repeat for output
        target_var = experiment_settings['target_var'][0]
        for model in output_raw.data_dictionary[target_var].keys():
            used_members = np.setdiff1d(model_member_dict[model], output_raw.model_member_dict[model])
            # make a list of the model name repeating for the number of members
            repeat_model = [model] * np.size(used_members)
            output_raw.data_dictionary[target_var][model] = output_raw.data_dictionary[target_var][model].assign_coords(member=(used_members))
            output_raw.data_dictionary[target_var][model] = output_raw.data_dictionary[target_var][model].assign_coords(model=('member', repeat_model))
        current_data["target"][target_var] = xr.concat([output_raw.data_dictionary[target_var][model] for model in output_raw.data_dictionary[target_var].keys()],
                                                        dim="member", coords="minimal")
        # load in the target dataset
        current_data["target"] = current_data["target"][experiment_settings['target_var'][0]]

        # if a land or ocean mask was used, carry that over
        if experiment_settings['mask']:
            current_data['land_ocean_mask'] = input_raw.final_mask.astype(FLOAT_TYPE)
        else:
            current_data['land_ocean_mask'] = xr.DataArray(data=1).astype(FLOAT_TYPE)
            
        # re-type the data to 32 bit
        current_data['inputs'] = current_data['inputs'].assign_coords(lat=current_data['inputs']['lat'].astype(FLOAT_TYPE),
                                                                      lon=current_data['inputs']['lon'].astype(FLOAT_TYPE),
                                                                      member=current_data['inputs']['member'].astype(INT_TYPE)).astype(FLOAT_TYPE)
        current_data['target'] = current_data['target'].assign_coords(lat=current_data['target']['lat'].astype(FLOAT_TYPE),
                                                                      lon=current_data['target']['lon'].astype(FLOAT_TYPE),
                                                                      member=current_data['target']['member'].astype(INT_TYPE)).astype(FLOAT_TYPE)

        # update dictionary to return
        processed_data[split_set] = current_data
        # update model_member_dict for next split
        model_member_dict = input_raw.model_member_dict
    
    return processed_data

# -------------------------------------------------------------------------------

def neural_network_stack(dataset, settings):
    "stack datasets for neural network, final shape is [samples (times x members), lat, lon, channels]"
    inputs = dataset['inputs']
    target = dataset['target']

    # include trajectory for training (sliding window of time indices)
    traj_inds = np.lib.stride_tricks.sliding_window_view(np.arange(target.time.size), settings['trajectory'])
    traj_inds = xr.DataArray(data=traj_inds, dims=['time', 'trajectory'])
    target = target.isel(time=traj_inds).transpose('time', 'member', 'lat', 'lon', 'trajectory')
    target = target.drop('time').stack(sample=['time', 'member'])
    # need to be consistent for inputs as well
    traj_inds = np.lib.stride_tricks.sliding_window_view(np.arange(inputs.time.size), settings['trajectory'])
    traj_inds = xr.DataArray(data=traj_inds, dims=['time', 'trajectory'])
    inputs = inputs.isel(time=traj_inds).transpose('time', 'member', 'lat', 'lon', 'trajectory')
    inputs = inputs.drop('time').stack(sample=['time', 'member']).isel(trajectory=0)

    # drop NaNs across samples (when full map is nans)
    bad_inds_inputs = np.where(np.isnan(inputs.mean(['lat', 'lon'], skipna=True).to_array()))[1]
    bad_inds_target = np.where(np.isnan(target.mean(['lat', 'lon'], skipna=True).to_array()))[2]
    good_inds = np.setdiff1d(np.arange(inputs.sample.size), np.unique(list(bad_inds_inputs) + list(bad_inds_target)))
    inputs = inputs.isel(sample=good_inds)
    target = target.isel(sample=good_inds)

    # stack lat and lon as loc for target, drop NaNs for loc in target
    # (keep only the target region, not necessarily a box)
    target = target.stack(loc=["lat", "lon"]).dropna('loc')

    # drop NaNs across samples (when any cell is nan)
    bad_inds_inputs = np.where(np.isnan(inputs.mean(['lat', 'lon'], skipna=False).to_array().sum('variable', skipna=False)))[0]
    bad_inds_target = np.where(np.isnan(target.mean(['loc'], skipna=False).to_array().sum('variable', skipna=False)))[1]
    good_inds = np.setdiff1d(np.arange(inputs.sample.size), np.unique(list(bad_inds_inputs) + list(bad_inds_target)))
    inputs = inputs.isel(sample=good_inds)
    target = target.isel(sample=good_inds)

    # convert to numpy array with correct shape
    inputs = inputs.to_array().transpose("sample", "lat", "lon", "variable").to_numpy()
    target = target.to_array().transpose("sample", "loc", "variable", "trajectory").to_numpy()
    
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

def combine_hist_ssp(directory, delete_old=False, grid='gn', ssp_name='ssp370'):
    """
    combine multiple historical and ssp subfiles into a single nc spanning the full time range.
    assumes that the filenames allow for sorting in the right order, and that
    the file names end CMIP6 style (ending in time range).
    """
    # get the files from the requested directory
    files = np.sort(glob.glob(os.path.join(directory, '*')))
    # get a list of the member ids without the dates to identify matching historical and ssp runs
    mem_ids = np.array([file[:file.find('_'+grid)] for file in files])
    # go through each file and find match, concatenate, save, optionally delete separate files
    for ii, file in enumerate(files):
        if 'historical' in file:
            # find matching ssp, if it exists
            ssp_memid = mem_ids[ii].replace('historical', ssp_name)
            ssp_ind = np.where(mem_ids == ssp_memid)[0]
            # only if there is a single match (single historical to single ssp)
            # this should only be used after the historical and ssp times have been combined separately
            # (using the combine_ncs function above)
            if np.size(ssp_ind) == 1:
                # now load the two datasets
                hist = xr.load_dataset(file)
                ssp = xr.load_dataset(files[ssp_ind][0])
                combined = xr.concat([hist, ssp], dim='time')
                # get the file name
                fname_start = files[ssp_ind][0][:-16]
                fname_mid = file[-16:-9]
                fname_end = files[ssp_ind][0][-9:]
                # save the combined file
                combined.to_netcdf(path=fname_start+fname_mid+fname_end)
                if delete_old:
                    os.remove(file)
                    os.remove(files[ssp_ind][0])

# -------------------------------------------------------------------------------

def get_model_res_dictionary(scratch=False):
    if scratch:
        models = glob.glob(os.path.join(RAW_DATA_PATH, 'raw_data/*'))
        resolution_dictionary = {}
        for model in models:
            file1 = glob.glob(model+'/*/*')[0]
            data = xr.load_dataset(file1)
            resolution_dictionary[os.path.basename(model)] = [np.abs(np.unique(np.diff(data.lat))[0]), np.abs(np.unique(np.diff(data.lon))[0])]
    else:
        resolution_dictionary = {'GISS-E2-2-G': [2.0, 2.5], 'NorESM2-LM': [1.894, 2.5], 'ACCESS-ESM1-5': [1.25, 1.875], 'CanESM5-1': [2.767, 2.8125],  'CMCC-CM2-HR4': [0.9424, 1.25], 'CESM2': [0.9424, 1.25], 'AWI-ESM-1-1': [1.849, 1.875], 'CAMS-CSM1': [1.112, 1.125], 'CMCC-ESM2': [0.9424, 1.25], 'CESM2-WACCM': [0.9424, 1.25], 'MPI-ESM1-2-HR': [0.9272, 0.9375], 'AWI-CM-1-1': [0.9272, 0.9375], 'MIROC': [1.389, 1.40625], 'MPI-ESM1-2-LR': [1.849, 1.875], 'GISS-E2-1-G': [2.0, 2.5], 'CNRM-CM6-1': [1.389, 1.40625], 'CMCC-CM2-SR5': [0.9424, 1.25], 'MRI-ESM2-0': [1.112, 1.125], 'BEST': [1.0, 1.0], 'MIROC-ES2H': [1.389, 1.40625], 'CanESM5-CanOE': [2.767, 2.8125], 'ACCESS-CM2': [1.25, 1.875], 'BCC-ESM1': [2.767, 2.8125], 'UKESM1-1-LL': [1.25, 1.875], 'UKESM1-0-LL': [1.25, 1.875], 'CanESM5': [2.767, 2.8125], 'BCC-CSM2': [1.112, 1.125], 'ERA5': [0.25, 0.25], 'TaiESM': [0.9424, 1.25], 'NorESM2-MM': [0.9424, 1.25], 'MIROC-ES2L': [2.767, 2.8125]}
    return resolution_dictionary
