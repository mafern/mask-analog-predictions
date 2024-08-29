# list of metrics for network build/train and for post prediction assessment

import numpy as np
import tensorflow as tf
import pickle
import run_experiments as rexp
import best_analogs as ba
import os
import gzip
from geopy.distance import geodesic
import pyemd

# TODO: add some extremes functions
# TODO: add more metrics: spread over ensemble, ACC, how often we are within some threshold compared to other methods
# TODO: crps statistic for evaluation (can compare to random 10 amongst others), and for picking n_analogs (https://github.com/stevenbrey/ensemble/blob/main)

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
    if settings['method'] == 'jacob':
        target = get_analog_errors_sq(soi, analog)
    else:
        target = get_analog_errors_abs(soi, analog)
    return target

# -------------------------------------------------------------------------------

def get_analog_errors_abs(soi, analog_prediction):
    return np.abs(soi - analog_prediction)

# -------------------------------------------------------------------------------

def get_analog_errors_sq(soi, analog_prediction):
    return np.square(soi - analog_prediction)

# -------------------------------------------------------------------------------

def compute_distance_matrix(lats, lons):
    """ Creates an N x N distance matrix from a dataframe of N locations 
    with a latitute column and a longitude column """

    def ellipsoidal_distance(point1, point2):
        """ Calculate distance (in kilometers) between two points, where 
        each point is represented as a tuple (lat, lon) """
        return geodesic(point1, point2).kilometers
    
    dist_matrix = np.zeros([lats.size*lons.size, lats.size*lons.size])
    latlon_pairs = np.array([np.meshgrid(lats, lons)[0].flatten(), np.meshgrid(lats, lons)[1].flatten()]).T

    for orig, orig_loc in enumerate(latlon_pairs):  # for each origin
        for dest, dest_loc in enumerate(latlon_pairs):  # for each destination
            dist_matrix[orig, dest] = ellipsoidal_distance(orig_loc, dest_loc)
    return dist_matrix

# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------
# metrics class for already calculated analogs
# -------------------------------------------------------------------------------

class AnalogMetrics():
    """
    Class for calculating performance of analog predictions
    """
    def __init__(self, experiment_name):
        # save the experiment_name
        self.experiment_name = experiment_name
        try:
            # check if processed prediction dataset exists by trying to load
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "rb") as filename:
                self.prediction_dict = pickle.load(filename)
            print('previously saved best analogs for:', *self.prediction_dict['indices'].soi_model.data)
        except:
            print('no previouly saved analog library or best analogs to load!')

    # -------------------------------------------------------------------------------
        
    def prediction_mse(self, soi_model, soi_member, variable, n_analogs, soi_index=None, model_mean=False):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        # get the targets for the requested soi and variable
        truth, prediction, _, _ = ba.get_targets(self.experiment_name, soi_model, soi_member, variable, n_analogs, soi_index=soi_index,
                                                 prediction_dict=self.prediction_dict, global_mean=True)
        if model_mean:
            # return the mse for the model mean
            all_model_mean = ba.get_full_analog_range(self.prediction_dict, soi_model, soi_member, variable, global_mean=True, minmax=False)
            model_mean_mse = np.nanmean((all_model_mean - truth)**2)
            return model_mean_mse
        else:
            # return the mse for the prediction
            prediction = np.roll(prediction, leadtime)
            analog_mse = np.nanmean((prediction - truth)**2)
            return analog_mse

    # -------------------------------------------------------------------------------

    def map_prediction_mse(self, soi_model, soi_member, variable, n_analogs, soi_index=None, model_mean=False):
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        leadtime = settings['leadtime']
        if model_mean:
            # get the targets for the requested soi and variable
            truth, _, _, _ = ba.get_targets(self.experiment_name, soi_model, soi_member, variable, 1, soi_index=soi_index,
                                            prediction_dict=self.prediction_dict, global_mean=False)
            # return the mse for the model mean
            all_model_mean = ba.get_full_analog_range(self.prediction_dict, soi_model, soi_member, variable, global_mean=False, minmax=False)
            model_mean_mse = np.nanmean((all_model_mean - truth)**2)
            return model_mean_mse
        else:
            truth, prediction, _, _ = ba.get_targets(self.experiment_name, soi_model, soi_member, variable, n_analogs, soi_index=soi_index,
                                                     prediction_dict=self.prediction_dict, global_mean=False)
            # return the mse for the prediction
            prediction = np.roll(prediction, leadtime, axis=0)
            analog_mse = np.nanmean((prediction - truth)**2)
            return analog_mse

    # -------------------------------------------------------------------------------

    def earth_movers_distance(self, soi_model, soi_member, variable, n_analogs, soi_index=None, model_mean=False, extra_mass_penalty=1.0):
        """calculate the EMD between the truth and a prediction (map)"""
        # get the settings so the region name can be used in the figure
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        # and the lead time for lining up the truth and prediction
        leadtime = settings['leadtime']
        # get the target region, for creating a distance matrix
        lats, lons = ba.get_target_region(self.experiment_name, self.prediction_dict)
        if lats.size * lons.size > 2000:
            print("Earth Mover's Distance relies on a distance matrix, which becomes large very quickly")
            print("The current region may take a very long time to run")
        # calculate the distance matrix from the targt lats and lons
        dist_matrix = compute_distance_matrix(lats, lons).astype('float64')
        if model_mean:
            # get the targets for the requested soi and variable
            truth, _, _, _ = ba.get_targets(self.experiment_name, soi_model, soi_member, variable, 1, soi_index=soi_index,
                                            prediction_dict=self.prediction_dict, global_mean=False)
            all_model_mean = ba.get_full_analog_range(self.prediction_dict, soi_model, soi_member, variable, global_mean=False, minmax=False)
            # emd needs non-negative values (I have checked, this does not affect the result)
            shift_up = 2 * np.nanmax([abs(truth), abs(all_model_mean)])
            truth = truth.astype('float64') + shift_up.astype('float64')
            all_model_mean = all_model_mean.astype('float64') + shift_up.astype('float64')
            # calculate EMD, one value for each prediction/truth pair (need to avoid nans, which breaks pyemd)
            emd = np.array([pyemd.emd(truth[ii].data.flatten(), all_model_mean[ii].flatten(), dist_matrix, extra_mass_penalty=extra_mass_penalty) for ii in range(np.shape(truth)[0])])
        else:
            # get the targets for the requested soi and variable
            truth, prediction, _, _ = ba.get_targets(self.experiment_name, soi_model, soi_member, variable, n_analogs, soi_index=soi_index,
                                                     prediction_dict=self.prediction_dict, global_mean=False)
            # emd needs non-negative values (I have checked, this does not affect the result)
            shift_up = 2 * np.nanmax([abs(truth), abs(prediction)])
            truth = truth.astype('float64') + shift_up.astype('float64')
            prediction = prediction.astype('float64') + shift_up.astype('float64')
            # roll the prediction so it lines up with the truth
            prediction = np.roll(prediction, leadtime, axis=0)
            # calculate EMD, one value for each prediction/truth pair (need to avoid nans, which breaks pyemd)
            emd = np.array([pyemd.emd(truth[ii].data.flatten(), prediction[ii].flatten(), dist_matrix, extra_mass_penalty=extra_mass_penalty) for ii in range(np.shape(truth)[0])])

        return emd

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

