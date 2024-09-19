# list of metrics for network build/train and for post prediction assessment

import numpy as np
import tensorflow as tf
import pickle
import run_experiments as rexp
import best_analogs as ba
import xarray as xr
import os
import gzip
from geopy.distance import geodesic
import pyemd

# TODO: add some extremes functions

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

def get_non_soi_analogs(prediction_dict, soi_model, soi_member, variable, global_average=True):
    # get all targets without the soi for the correct times
    soi_index = np.where((prediction_dict['target'].model == soi_model) *
                         (prediction_dict['target'].member == soi_member))[0]
    truth = prediction_dict['target'].isel(member=soi_index)[variable]
    times = np.unique(np.where(~np.isnan(truth))[0])
    all_inds = np.arange(prediction_dict['target'].member.size)
    non_soi_inds = np.setdiff1d(all_inds, soi_index)
    non_soi_analogs = prediction_dict['target'].isel(member=non_soi_inds, time=times)[variable]
    if global_average:
        weights = np.cos(np.deg2rad(prediction_dict['target'].lat))
        non_soi_analogs = non_soi_analogs.weighted(weights).mean(("lon", "lat"), skipna=True)
    return non_soi_analogs

# -------------------------------------------------------------------------------

def get_perfect_analogs(prediction_dict, experiment_name, soi_model, soi_member, variable, soi_index=None, global_average=True):
    truth, _, _, _ = ba.get_targets(experiment_name, soi_model, soi_member, variable, 1, soi_index=soi_index,
                                    prediction_dict=prediction_dict, global_mean=False)
    non_soi_analogs = get_non_soi_analogs(prediction_dict, soi_model, soi_member, variable, global_average=global_average)
    perfect_analogs = []
    # for each time in truth
    for ii in range(truth.shape[0]):
        # find the mean squared error between truth at this time and analog library
        mse = ((non_soi_analogs - truth[ii])**2).mean(["lat", "lon"], skipna=True).data
        # get the best times and members
        perfect_time, perfect_member = np.dstack(np.unravel_index(np.argsort(mse.ravel()), mse.shape)).squeeze().T
        # add only the top match
        top_match = non_soi_analogs.isel(time=perfect_time[0], member=perfect_member[0])
        top_match['soi_time'] = truth.isel(time=ii).time.data
        perfect_analogs.append(top_match)
    perfect_analogs = xr.concat(perfect_analogs, dim='soi_time')
    return perfect_analogs

# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------
# metrics class for calculated analogs
# -------------------------------------------------------------------------------
class AnalogMetrics():
    """
    Class for calculating performance of analog predictions
    """
    def __init__(self, experiment_name, soi_model, soi_member, variable, soi_index=None):
        # save the experiment_name and other useful variables
        self.experiment_name = experiment_name
        self.soi_model = soi_model
        self.soi_member = soi_member
        self.variable = variable
        self.soi_index = soi_index
        settings = rexp.Experiments().get_experiment(self.experiment_name)
        self.leadtime = settings['leadtime']
        try:
            # check if processed prediction dataset exists by trying to load
            with gzip.open(os.path.join(os.path.dirname(__file__), "saved_analogs", self.experiment_name+'.pickle'), "rb") as filename:
                self.prediction_dict = pickle.load(filename)
            print('previously saved analogs for:', *self.prediction_dict['indices'].soi_model.data)
        except:
            print('no previouly saved analog library or analogs to load!')
        # grab perfect analogs, analog library, and truth
        self.perfect_analogs = get_perfect_analogs(self.prediction_dict, self.experiment_name, soi_model, soi_member, variable,
                                                   soi_index=soi_index, global_average=False)
        self.analog_library = get_non_soi_analogs(self.prediction_dict, soi_model, soi_member, variable, global_average=False)
        self.truth, _, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, 1,
                                             soi_index=self.soi_index, prediction_dict=self.prediction_dict)
        # keep only the times where the analog prediction, library, and truth overlap
        self.perfect_analogs = self.perfect_analogs[self.leadtime:]
        self.analog_library = self.analog_library[self.leadtime:]
        self.truth = self.truth[self.leadtime:]

        # get the target region, for creating a distance matrix
        self.lats, self.lons = ba.get_target_region(self.experiment_name, self.prediction_dict)

    # -------------------------------------------------------------------------------
        
    def prediction_error(self, n_analogs, prediction="analog", time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the targets for the requested soi and variable, as well as the prediction (maps)
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                         soi_index=self.soi_index, prediction_dict=self.prediction_dict)
        # align analog prediction with analog library and truth
        analog = analog[:-self.leadtime]
        
        if prediction == "model":
            # mean map over members for each time
            all_model_mean = self.analog_library.mean("member", skipna=True)
            model_error = self.truth.sel(time=time_slice).data - all_model_mean.sel(time=time_slice).data
            # return a single error map that is the average over time
            return model_error
        elif prediction == "analog":
            analog_error = self.truth.sel(time=time_slice).data - analog.sel(time=time_slice).data
            return analog_error
        elif prediction == "perfect":
            # get the perfect analog prediction
            perfect_error = self.truth.sel(time=time_slice).data - self.perfect_analogs.sel(soi_time=time_slice).data
            return perfect_error
        else:
            print("option for prediction not recognized")
        
    # -------------------------------------------------------------------------------
        
    def prediction_spread(self, n_analogs, prediction="analog", time_range=None):
        """standard deviation for the prediction analogs or model library (i.e. the spread)"""
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        if prediction == "model":
            # return the std across members for the model library
            model_spread = self.analog_library.std("member", skipna=True).sel(time=time_slice)
            return model_spread
        elif prediction == "analog":
            # return the std across analogs for the prediction
            _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                             soi_index=self.soi_index, prediction_dict=self.prediction_dict, return_average=False)
            # align analog prediction with analog library and truth
            analog = analog[:-self.leadtime]
            analog_spread = analog.std("analog_size", skipna=True).sel(time=time_slice)
            return analog_spread
        else:
            print("option for prediction not recognized")
        
    # -------------------------------------------------------------------------------
        
    def prediction_mse(self, n_analogs, prediction="analog", map=False, time_range=None):
        if map:
            # return the mse for the prediction over time (returns a map)
            error = self.prediction_error(n_analogs, prediction=prediction, time_range=time_range)
            return np.nanmean(error**2, axis=0)
        else:
            # return the mse for the predictionover area (returns time series)
            error = self.prediction_error(n_analogs, prediction=prediction, time_range=time_range)
            return np.nanmean(error**2, axis=(1, 2))

    # -------------------------------------------------------------------------------

    def earth_movers_distance(self, n_analogs, prediction="analog", extra_mass_penalty=1.0, time_range=None):
        """calculate the EMD between the truth and a prediction (map)"""
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        if self.lats.size * self.lons.size > 2000:
            print("Earth Mover's Distance relies on a distance matrix, which becomes large very quickly")
            print("The current region may take a very long time to run")
        # calculate the distance matrix from the targt lats and lons
        dist_matrix = compute_distance_matrix(self.lats, self.lons).astype('float64')

        # get the targets for the requested soi and variable
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                         soi_index=self.soi_index, prediction_dict=self.prediction_dict)
        # align analog prediction with analog library and truth
        analog = analog[:-self.leadtime]

        # emd needs non-negative values (I have checked, this does not affect the result)
        shift_up = 2 * np.nanmax([abs(self.truth), abs(self.analog_library.mean("member", skipna=True))])
        # line the analog prediction up with the truth and analog library and select time range
        truth = self.truth.sel(time=time_slice).astype('float64') + shift_up.astype('float64')
        analog = analog.sel(time=time_slice).astype('float64') + shift_up.astype('float64')
        all_model_mean = self.analog_library.mean("member", skipna=True).sel(time=time_slice).astype('float64') + shift_up.astype('float64')
        perfect_analog = self.perfect_analogs.sel(soi_time=time_slice).astype('float64') + shift_up.astype('float64')

        # scale the distance matrix so values are not way bigger than the variable
        dist_mean, truth_mean = np.mean(dist_matrix), np.mean(truth.data)
        dist_matrix = dist_matrix / dist_mean * truth_mean # truth based units
        # dist_matrix = dist_matrix / 100 # units of 100 km
        
        # calculate EMD for multi-model mean prediction
        if prediction == "model":
            # calculate EMD, one value for each prediction/truth pair (need to avoid nans, which breaks pyemd)
            emd = np.array([pyemd.emd(truth[ii].data.flatten(), all_model_mean[ii].data.flatten(), dist_matrix, extra_mass_penalty=extra_mass_penalty) for ii in range(np.shape(truth)[0])])
            return emd / truth_mean * dist_mean
        # calculate EMD for our analog prediction
        elif prediction == "analog":
            # calculate EMD, one value for each prediction/truth pair (need to avoid nans, which breaks pyemd)
            emd = np.array([pyemd.emd(truth[ii].data.flatten(), analog[ii].data.flatten(), dist_matrix, extra_mass_penalty=extra_mass_penalty) for ii in range(np.shape(truth)[0])])
            return emd / truth_mean * dist_mean
        # calculate EMD for a perfect analog prediction
        elif prediction == "perfect":
            # calculate EMD, one value for each prediction/truth pair (need to avoid nans, which breaks pyemd)
            emd = np.array([pyemd.emd(truth[ii].data.flatten(), perfect_analog[ii].data.flatten(), dist_matrix, extra_mass_penalty=extra_mass_penalty) for ii in range(np.shape(truth)[0])])
            return emd / truth_mean * dist_mean
        else:
            print("option for prediction not recognized")

    # -------------------------------------------------------------------------------

    def variance_difference(self, n_analogs, prediction="analog", time_range=None):
        """calculate the variance of the truth over time"""
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                         soi_index=self.soi_index, prediction_dict=self.prediction_dict)
        # align analog prediction with analog library and truth
        analog = analog[:-self.leadtime]

        truth_var = np.nanvar(self.truth.sel(time=time_slice), axis=0)
        if prediction == "model":
            # calculate the variance of the models over time
            model_mean_var = self.analog_library.sel(time=time_slice).mean("member", skipna=True).var('time', skipna=True).to_numpy()
            # fractional difference
            var_diff = (truth_var - model_mean_var) / truth_var
            return var_diff
        elif prediction == "analog":
            analog_var = np.nanvar(analog.sel(time=time_slice), axis=0)
            # fractional difference
            var_diff = (truth_var - analog_var) / truth_var
            return var_diff
        elif prediction == "perfect":
            perfect_var = np.nanvar(self.perfect_analogs.sel(soi_time=time_slice), axis=0)
            var_diff = (truth_var - perfect_var) / truth_var
            return var_diff
        else:
            print("option for prediction not recognized")

    # -------------------------------------------------------------------------------

    def spread_ensemble(self, n_analogs, prediction="analog", map=False, time_range=None):
        # these are both time x lat x lon
        spread = self.prediction_spread(n_analogs, prediction=prediction, time_range=time_range)
        error = self.prediction_error(n_analogs, prediction=prediction, time_range=time_range)
        if prediction == "model":
            ensemble_size = self.analog_library.member.size
        elif prediction == "analog":
            ensemble_size = n_analogs
        else:
            print("option for prediction not recognized")
            return 0

        if map:
            # get map of spread over ensemble error
            rmse = np.sqrt(np.nanmean(error**2, axis=0))
            spread = np.nanmean(spread, axis=0)
            # normalization for number of analogs in spread
            spread_over_ensemble = (ensemble_size+1)/ensemble_size * (spread/rmse)
        else:
            # get time series of spread over ensemble error
            rmse = np.sqrt(np.nanmean(error**2, axis=(1, 2)))
            spread = np.nanmean(spread, axis=(1, 2))
            spread_over_ensemble = (ensemble_size+1)/ensemble_size * (spread/rmse)

        return spread_over_ensemble

    # -------------------------------------------------------------------------------

    def crps(self, n_analogs, prediction="analog", map=False, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get truth, prediction, and analog library
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                         soi_index=self.soi_index, prediction_dict=self.prediction_dict, return_average=False)
        # align truth, analog prediction, perfect analogs, and analog library time range
        analog = analog[:-self.leadtime].sel(time=time_slice)
        non_soi_analogs = self.analog_library.sel(time=time_slice)
        perfect_analogs = self.perfect_analogs.sel(soi_time=time_slice)
        truth = self.truth.sel(time=time_slice)
        # map by time number of entries
        time_x_map_size = np.prod(truth.shape)
        # set up interpolation axis, range, number of steps
        nsteps = 10000
        max_x = np.nanmax([np.nanmax(truth), np.nanmax(non_soi_analogs)])
        min_x = np.nanmin([np.nanmin(truth), np.nanmin(non_soi_analogs)])
        dx = (max_x - min_x) / nsteps
        interp_x = np.linspace(min_x, max_x, nsteps)
        # interpolated step functions for each truth
        interp_step = np.array([interp_x >= truth.data.reshape(-1)[ii] for ii in range(truth.size)]).reshape(truth.shape + (-1,)).astype(int)
        
        if prediction == "model":
            # models sorted along member axis, shape: member x time x lat x lon
            model_cdfx = np.transpose(np.sort(non_soi_analogs, axis=1), (1, 0, 2, 3))
            # flattened version for simplicity
            flat_model = model_cdfx.reshape(non_soi_analogs.member.size, -1)
            # cdf values
            model_cdfy = np.linspace(0, 1, non_soi_analogs.member.size)
            # get interpolated cdf, avoiding nans
            non_nan_model = np.isfinite(flat_model)
            model_interp = [np.interp(interp_x, flat_model[np.where(non_nan_model[:, ii])[0], ii], model_cdfy[np.where(non_nan_model[:, ii])[0]]) for ii in range(time_x_map_size)]
            model_interp = np.array(model_interp).reshape(truth.shape + (-1,))
            # crps values, shape: time x lat x lon
            model_crps = np.nansum((model_interp - interp_step)**2 * dx, axis=-1)
            if map:
                crps = np.nanmean(model_crps, axis=0)
            else:
                crps = np.nanmean(model_crps, axis=(1, 2))
            return crps
        elif prediction == "analog":
            # sorted along analog axis, shape: analog x time x lat x lon
            pred_cdfx = np.transpose(np.sort(analog, axis=1), (1, 0, 2, 3))
            flat_pred = pred_cdfx.reshape(n_analogs, -1)
            pred_cdfy = np.linspace(0, 1, n_analogs)
            # get interpolated cdf, avoiding nans
            non_nan_pred = np.isfinite(flat_pred)
            pred_interp = [np.interp(interp_x, flat_pred[np.where(non_nan_pred[:, ii])[0], ii], pred_cdfy[np.where(non_nan_pred[:, ii])[0]]) for ii in range(time_x_map_size)]
            pred_interp = np.array(pred_interp).reshape(truth.shape + (-1,))
            # crps values, shape: time x lat x lon
            pred_crps = np.nansum((pred_interp - interp_step)**2 * dx, axis=-1)
            if map:
                crps = np.nanmean(pred_crps, axis=0)
            else:
                crps = np.nanmean(pred_crps, axis=(1, 2))
            return crps
        elif prediction == "perfect":
            # for deterministic prediction, crps is just the mae
            if map:
                crps = np.nanmean(np.abs(truth.data - perfect_analogs.data), axis=0)
            else:
                crps = np.nanmean(np.abs(truth.data - perfect_analogs.data), axis=(1, 2))
            return crps
        else:
            print("option for prediction not recognized")

    # -------------------------------------------------------------------------------

    def class_accuracy(self, n_analogs, n_cells, prediction="analog", map=False, coarsen_time=1, time_range=None):
        """break up range into bins, count how often prediction and truth are in same (bin size flexible to change accuracy measure)"""
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get analog prediction, truth, library
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.variable, n_analogs,
                                         soi_index=self.soi_index, prediction_dict=self.prediction_dict)
        # align truth, analog prediction, perfect analogs, and library
        non_soi_analogs = self.analog_library.mean("member", skipna=True).sel(time=time_slice)
        analog = analog[:-self.leadtime].sel(time=time_slice)
        truth = self.truth.sel(time=time_slice)
        perfect_analogs = self.perfect_analogs.sel(soi_time=time_slice)

        # coarsen time, if desired (value other than 1)
        truth = np.array([np.mean(truth[ii:ii+coarsen_time].data, axis=0) for ii in range(0, truth.shape[0], coarsen_time)])
        analog = np.array([np.mean(analog[ii:ii+coarsen_time].data, axis=0) for ii in range(0, analog.shape[0], coarsen_time)])
        non_soi_analogs = np.array([np.mean(non_soi_analogs[ii:ii+coarsen_time].data, axis=0) for ii in range(0, non_soi_analogs.shape[0], coarsen_time)])
        perfect_analogs = np.array([np.mean(perfect_analogs[ii:ii+coarsen_time].data, axis=0) for ii in range(0, perfect_analogs.shape[0], coarsen_time)])

        # get range of variable
        min_val = np.nanmin([np.nanmin(truth), np.nanmin(non_soi_analogs)])
        max_val = np.nanmax([np.nanmax(truth), np.nanmax(non_soi_analogs)])
        val_range = np.linspace(min_val, max_val, n_cells)
        range_matrix = np.tile(val_range, np.product(truth.shape)).reshape(truth.shape + (-1,))

        # make coarsened truth (shape is [time x lat x lon x n_cells] with 1 for the cell the truth falls in, 0 everywhere else)
        truth_matrix = truth.reshape(truth.shape + (1,))
        coarse_truth = np.nanmin(abs(range_matrix - truth_matrix), axis=-1, keepdims=True) == abs(range_matrix - truth_matrix)
        # repeat for the analog prediction
        pred_matrix = analog.reshape(analog.shape + (1,))
        coarse_pred = np.nanmin(abs(range_matrix - pred_matrix), axis=-1, keepdims=True) == abs(range_matrix - pred_matrix)
        # and for the model mean
        model_matrix = non_soi_analogs.reshape(non_soi_analogs.shape + (1,))
        coarse_model = np.nanmin(abs(range_matrix - model_matrix), axis=-1, keepdims=True) == abs(range_matrix - model_matrix)
        # and for the perfect analogs
        perfect_matrix = perfect_analogs.reshape(perfect_analogs.shape + (1,))
        coarse_perfect = np.nanmin(abs(range_matrix - perfect_matrix), axis=-1, keepdims=True) == abs(range_matrix - perfect_matrix)
        # now, find where the predictions line up with the truth and count -- it either matched (1) or didn't (0)
        pred_classification = np.sum(coarse_truth * coarse_pred, axis=-1)
        model_classification = np.sum(coarse_truth * coarse_model, axis=-1)
        perfect_classification = np.sum(coarse_truth * coarse_perfect, axis=-1)
        if prediction == "model":
            if map:
                class_acc = np.mean(model_classification, axis=0)
            else:
                class_acc = np.mean(model_classification, axis=(1, 2))
            return class_acc
        elif prediction == "analog":
            if map:
                class_acc = np.mean(pred_classification, axis=0)
            else:
                class_acc = np.mean(pred_classification, axis=(1, 2))
            return class_acc
        elif prediction == "perfect":
            if map:
                class_acc = np.mean(perfect_classification, axis=0)
            else:
                class_acc = np.mean(perfect_classification, axis=(1, 2))
            return class_acc
        else:
            print("option for prediction not recognized")

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

