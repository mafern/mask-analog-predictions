# list of metrics for network build/train

import numpy as np
import tensorflow as tf
import run_experiments as rexp
import best_analogs as ba
import xarray as xr
from geopy.distance import geodesic
import pyemd
import bottleneck as bn

# -------------------------------------------------------------------------------
# some general use metrics
# -------------------------------------------------------------------------------

def mse(analog, truth):
    difference = tf.keras.layers.subtract([analog, truth])
    difference_squared = tf.keras.backend.square(difference)
    return tf.math.reduce_mean(difference_squared, axis=-1)

# -------------------------------------------------------------------------------

def pred_range(y_true, y_pred):
    return tf.math.reduce_max(y_pred) - tf.math.reduce_min(y_pred)

# -------------------------------------------------------------------------------

def get_target_error(settings, soi, analog):
    target = get_analog_errors_sq(soi, analog)
    return target

# -------------------------------------------------------------------------------

def get_non_soi_analogs(prediction_dict, soi_model, soi_member, variable, global_average=True):
    # get all targets without the soi for the correct times
    soi_index = np.where((prediction_dict['target'].model == soi_model) *
                         (prediction_dict['target'].member == soi_member))[0]
    all_inds = np.arange(prediction_dict['target'].member.size)
    non_soi_inds = np.setdiff1d(all_inds, soi_index)
    non_soi_analogs = prediction_dict['target'].isel(sample=non_soi_inds)[variable]
    if global_average:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        non_soi_analogs = non_soi_analogs.weighted(weights).mean(("lon", "lat"), skipna=True)
    return non_soi_analogs

# -------------------------------------------------------------------------------

def get_soi(prediction_dict, soi_model, soi_member, variable, global_average=True):
    # get all targets without the soi for the correct times
    soi_index = np.where((prediction_dict['target'].model == soi_model) *
                         (prediction_dict['target'].member == soi_member))[0]
    truth = prediction_dict['target'].isel(sample=soi_index)[variable]
    if global_average:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        truth = truth.weighted(weights).mean(("lon", "lat"), skipna=True)
    return truth

# -------------------------------------------------------------------------------

def get_non_soi_inputs(prediction_dict, soi_model, soi_member, variable, global_average=True):
    # get all targets without the soi for the correct times
    soi_index = np.where((prediction_dict['inputs'].model == soi_model) *
                         (prediction_dict['inputs'].member == soi_member))[0]
    all_inds = np.arange(prediction_dict['inputs'].member.size)
    non_soi_inds = np.setdiff1d(all_inds, soi_index)
    non_soi_inputs = prediction_dict['inputs'].isel(sample=non_soi_inds)[variable]
    if global_average:
        weights = np.cos(np.deg2rad(prediction_dict['inputs'].lat))
        non_soi_inputs = non_soi_inputs.weighted(weights).mean(("lon", "lat"), skipna=True)
    return non_soi_inputs

# -------------------------------------------------------------------------------

def get_soi_inputs(prediction_dict, soi_model, soi_member, variable, global_average=True):
    # get all targets without the soi for the correct times
    soi_index = np.where((prediction_dict['inputs'].model == soi_model) *
                         (prediction_dict['inputs'].member == soi_member))[0]
    inputs = prediction_dict['inputs'].isel(sample=soi_index)[variable]
    if global_average:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        inputs = inputs.weighted(weights).mean(("lon", "lat"), skipna=True)
    return inputs

# -------------------------------------------------------------------------------
