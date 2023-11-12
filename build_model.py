import numpy as np
import tensorflow as tf
import metrics
import gc

__author__ = "Jamin K. Rader, Elizabeth A. Barnes"
__version__ = "30 March 2023"


# def parse_model(x_input, mask_model, dissimilarity_model, prediction_model):
    
#     weighted_soi, weighted_analog, weights = mask_model(x_input)
#     dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
#     prediction = prediction_model([dissimilarities])
#     __ = gc.collect()  # to fix memory leak

#     return weights.numpy(), dissimilarities.numpy(), prediction.numpy()


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# model takes in soi and analog maps, returns mask-weighted maps and mask weights
def make_weights_model(x_train, mask_model_act, mask_l1, mask_l2,
                       mask_initial_value, normalize_weights_bool):
    ""
    
    # get the dimensions of the maps
    map_dim = x_train[0].shape[1:]

    # state of interest and analog inputs, shape is lat x lon x channels
    soi_input_layer = tf.keras.layers.Input(shape=map_dim, name='soi_input')
    analog_input_layer = tf.keras.layers.Input(shape=map_dim, name='analogs_input')

    # Flatten Layer
    x = tf.keras.layers.Flatten()(soi_input_layer)
    
    # define a Bias-only layer (e.g. inputs don't affect mask)
    class WeightsLayer(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            super(WeightsLayer, self).__init__(*args, **kwargs)
        # build is invoked automatically before the first instance of call
        # i.e. between instantiation and call
        # defines self.bias which is then used in the call
        def build(self, input_shape):
            self.bias = self.add_weight('bias',
                                        shape=(input_shape[1:]),
                                        initializer=mask_initial_value,
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg(),)
        # redefine call -- normally this returns the inputs
        # in our case, it returns a x-shaped bias
        def call(self, x):
            return tf.zeros_like(x) + self.bias

    # create bias-only layer
    weights = WeightsLayer()
    weights_layer = weights(x)
    weights_layer = tf.keras.layers.Activation(mask_model_act)(weights_layer) 
    if normalize_weights_bool:
        # do not need float64 if the weights values are not restricted
        weights_mean = tf.cast(tf.math.reduce_mean(weights_layer, axis=-1, keepdims=True), dtype=tf.float32)
        weights_weighting = tf.cast(tf.math.reciprocal_no_nan(weights_mean), dtype=tf.float32)
        weights_layer = tf.math.multiply(weights_weighting, tf.cast(weights_layer, dtype=tf.float32))
    weights_layer = tf.keras.layers.ActivityRegularization(l1=mask_l1, l2=mask_l2)(weights_layer)
    weights_layer = tf.keras.layers.Reshape(map_dim)(weights_layer)
    weights_layer = tf.keras.layers.Layer()(weights_layer, name='weights_layer')

    # multiply weights layer by soi and analog inputs
    weighted_soi = tf.keras.layers.multiply([weights_layer, soi_input_layer], name='weighted_soi')
    weighted_analog = tf.keras.layers.multiply([weights_layer, analog_input_layer], name='weighted_analogs')
    
    # Define Model - in: soi and analog maps, out: weighted versions and weights
    mask_model = tf.keras.Model(
        inputs=[soi_input_layer, analog_input_layer],
        outputs=[weighted_soi, weighted_analog, weights_layer],
        name="mask_model",
        )

    return mask_model, soi_input_layer, analog_input_layer

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Second, creating the model that calculates the dissimilarity between the weighted maps
# model takes in weighted soi and analog maps, returns the mse
# (reminder, these are both still inputs that will be used in predicting future dissimilarity)
def make_dissimilarity_model(x_train):
    ""
    
    # get the dimensions of the maps
    map_dim = x_train[0].shape[1:]
    # input layers for the soi and analog weighted maps
    weighted_soi_input_layer = tf.keras.layers.Input(shape=map_dim)
    weighted_analog_input_layer = tf.keras.layers.Input(shape=map_dim)
    # flatten inputs
    weighted_soi_flat = tf.keras.layers.Flatten()(weighted_soi_input_layer)
    weighted_analog_flat = tf.keras.layers.Flatten()(weighted_analog_input_layer)

    # Calculate the MSE between the weighted SOI and the weighted analogs
    dissimilarity = metrics.mse(weighted_analog_flat, weighted_soi_flat)
    # single value MSE, so reshape accordiingly
    dissimilarity = tf.keras.layers.Reshape((1,))(dissimilarity)

    # define model - in: weighted maps, out: dissimilarity between these maps
    dissimilarity_model = tf.keras.Model(
        inputs=[weighted_soi_input_layer, weighted_analog_input_layer],
        outputs=[dissimilarity],
        name="dissimilarity_model"
        )

    return dissimilarity_model

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Third, creating the model that uses the dissimilarity score to predict the dissimilarity of the maps x years later
def make_prediction_model(prediction_model_nodes, prediction_model_act, rng_seed, output_activation="linear"):
    ""
    
    # input is mse, single-valued
    dissimilarity_input_layer = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Layer()(dissimilarity_input_layer)
    
    # Add all the Dense Layers
    for nodes in prediction_model_nodes:
        x = tf.keras.layers.Dense(
            nodes, activation=prediction_model_act,
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+4),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+1),
            )(x)

    # Prediction layer -- single value difference between targets
    prediction = tf.keras.layers.Dense(
        1,
        activation=output_activation,
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+2),
        )(x)

    # define model -- in: weighted input map dissimilarity, out: future dissimilarity
    prediction_model = tf.keras.Model(
        inputs=[dissimilarity_input_layer],
        outputs=[prediction],
        name='prediction_model'
        )

    return prediction_model

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Combining all three models
def build_interp_model(settings, x_train):
    ""
    
    # model takes in soi and analog and returns weighted versions as well as weight mask
    mask_model, soi_input, analog_input = make_weights_model(x_train,
                                                             mask_model_act=settings["mask_activation"],
                                                             mask_l1=settings["mask_l1"],
                                                             mask_l2=settings["mask_l2"],
                                                             mask_initial_value=settings["mask_init"],
                                                             normalize_weights_bool=settings["norm_weights"]
                                                             )
    
    # model takes in weighted maps and returns dissimiliarity (mse, assuming soi is truth)
    dissimilarity_model = make_dissimilarity_model(x_train)
    
    # model takes in the dissimilarity above (mse) and predicts dissimilarity in the future
    prediction_model = make_prediction_model(prediction_model_nodes=settings["prediction_nodes"],
                                             prediction_model_act=settings["prediction_activation"],
                                             rng_seed=settings["seed"],
                                             output_activation="linear",
                                             )
    
    # take in the soi and analog inputs, return weighted versions and weight mask
    weighted_soi, weighted_analog, weights = mask_model([soi_input, analog_input])
    
    # take in the weighted inputs from above, return their dissimilarity
    dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
    
    # take in the dissimilarity from above, return predicted dissimilarity in the future
    prediction = prediction_model([dissimilarities])

    # define the final model -- inputs to prediction for future dissimilarity
    full_model = tf.keras.Model(
        inputs=[soi_input, analog_input],
        outputs=[prediction],
        name='full_model'
        )

    return full_model, mask_model, dissimilarity_model, prediction_model