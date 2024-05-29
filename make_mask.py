import os
import numpy as np
import xarray as xr
import json
# import warnings
# warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# TODO: add function to threshold masks, probably by percentile

def make_mask(experiment_name, models, processed_data, overwrite_mask=False):
    """
    description
    """
    # check if mask already exists and no overwrite has been requested
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc')) or overwrite_mask is True:
        # either mask does not exist, or overwrite has been requested
        
        # get lat and lon and experiment settings
        exp_settings = json.load(open('experiments.json'))[experiment_name]
        lat, lon = processed_data['analog']['lat'], processed_data['analog']['lon']
        vars = np.array(processed_data['analog']['inputs'].data_vars)
        print('----------------------------------------------------------------')
        print('retrieving mask and saving to', os.path.join('masks', experiment_name+'_mask.nc'))
        
        # get the seed list and initialize the mask lists
        seed = list(exp_settings["seed"])
        raw_masks, masks = [], []
        if models is not list:
            models = [models,]
        for ii, model in enumerate(models):
            # Get both the raw and scaled (masked, scale to mean=1) mask and save as a netcdf file
            raw_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy()
            raw_mask = raw_mask.reshape(vars.size, lat.size, lon.size)
            # get land or ocean mask (or neither)
            geomask = processed_data['analog']['land_ocean_mask'].to_numpy()
            mask = scale_mask(raw_mask, geomask)
            # append each seed (model) mask to the lists            
            raw_masks.append(raw_mask)
            masks.append(mask)
        
        # save the mask as netcdf with lat and lon coordinates (raw and scaled)
        data_var_dict = {}
        for ii, var in enumerate(vars):
            data_var_dict[var] = (["seed", "lat", "lon"], np.array(masks)[:, ii])
        mask = xr.Dataset(
            data_vars=data_var_dict,
            coords=dict(seed=seed[ii], lat=lat, lon=lon)
        )
        mask.to_netcdf(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc'))
        data_var_dict = {}
        for ii, var in enumerate(vars):
            data_var_dict[var] = (["seed", "lat", "lon"], np.array(raw_masks)[:, ii])
        mask = xr.Dataset(
            data_vars=data_var_dict,
            coords=dict(seed=seed[ii], lat=lat, lon=lon)
        )
        mask.to_netcdf(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_raw_mask.nc'))
    
    # no need to get data, build/train model, or make/save mask -- just load the existing mask
    else:
        print('loading pre-saved mask')
        mask = xr.load_dataset(os.path.join(os.path.dirname(__file__), 'saved_masks', experiment_name+'_mask.nc')).squeeze()
        
    return mask

# -------------------------------------------------------------------------------

def scale_mask(weighted_mask, land_ocean_mask):
    # Set masked values to zero
    weighted_mask = weighted_mask * land_ocean_mask
    # Re-scale so the mean weight is 1
    weighted_mask = weighted_mask / np.nansum(weighted_mask) * np.nansum(land_ocean_mask)
    # Set masked values to the 0
    # weighted_mask = np.nan_to_num(weighted_mask)
    return weighted_mask

# -------------------------------------------------------------------------------

