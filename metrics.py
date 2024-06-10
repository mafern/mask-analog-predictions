# list of metrics for network build/train and for post prediction assessment

import numpy as np
import tensorflow as tf


def mse(analog, truth):
    difference = tf.keras.layers.subtract([analog, truth])
    difference_squared = tf.keras.backend.square(difference)
    return tf.math.reduce_mean(difference_squared, axis=-1)

# -------------------------------------------------------------------------------

def pred_range(y_true, y_pred):
    return tf.math.reduce_max(y_pred) - tf.math.reduce_min(y_pred)

# -------------------------------------------------------------------------------

def get_targets(settings, soi, analog):
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
