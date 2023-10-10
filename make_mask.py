import os
import numpy as np
import xarray as xr
# import warnings
# warnings.filterwarnings(action='ignore', message='Mean of empty slice')

###################################################################################
def make_mask(experiment_name, models, processed_data, overwrite_mask=False):
###################################################################################
    """
    description
    """
    # check if mask already exists and no overwrite has been requested
    if not os.path.exists(os.path.join('saved_masks', experiment_name+'_mask.nc')) or overwrite_mask is True:
        # either mask does not exist, or overwrite has been requested
        
        # get lat and lon
        lat, lon = processed_data['analog']['lat'], processed_data['analog']['lon']
        
        print('----------------------------------------------------------------')
        print('retrieving mask and saving to', os.path.join('masks', experiment_name+'_mask.nc'))
        
        ###################################################################################
        # seed = np.arange(0, 100, 10)
        seed = [20,]
        raw_masks, masks = [], []
        for model in models:
        ###################################################################################
        
            # Get both the raw and scaled (masked, scale to mean=1) mask and save as a netcdf file
            raw_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy()
            raw_mask = raw_mask.reshape(lat.size, lon.size).squeeze()

            # get land or ocean mask (or neither)
            geomask = processed_data['analog']['land_ocean_mask'].to_numpy().squeeze()
            mask = scale_mask(raw_mask, geomask)
            
            ###################################################################################
            raw_masks.append(raw_mask)
            masks.append(mask)
            ###################################################################################
        
        # save the mask as netcdf with lat and lon coordinates (raw and scaled)
        mask = xr.Dataset(
            data_vars=dict(mask=(["seed", "lat", "lon"], masks),
                          raw_mask=(["seed", "lat", "lon"], raw_masks)),
            coords=dict(lat=lat, lon=lon, seed=seed)
        )
        mask.to_netcdf(os.path.join('saved_masks', experiment_name+'_mask.nc'))
    
    # no need to get data, build/train model, or make/save mask -- just load the existing mask
    else:
        print('loading pre-saved mask')
        mask = xr.load_dataset(os.path.join('saved_masks', experiment_name+'_mask.nc')).squeeze()
        
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

## add function to threshold masks, probably by percentile as Jamin did (supplemental in his paper, figure S4)

# -------------------------------------------------------------------------------

## add function to plot the mask
###################################################################################
# plot the mask
# model_diagnostics.visualize_interp_model(experiment_settings, weights_val, lat, lon)
###################################################################################

# -------------------------------------------------------------------------------

