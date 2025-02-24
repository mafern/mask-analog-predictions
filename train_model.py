import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import metrics
import os
import logging
import pickle
import warnings
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

FLOAT_TYPE = np.float32
INT_TYPE = np.int32

def batch_generator(settings, soi_input, soi_output, analog_input, analog_output, batch_size, rng_seed=33):
    ""
    # random seed to select random subsets of the soi and analog sets
    rng = np.random.default_rng(rng_seed)
    # create a generator that returns random samples of the inputs (soi, analog) and the target (output difference)
    while True:
        # random subsamples of size batch_size
        i_soi = rng.choice(np.arange(0, soi_input.shape[0]), batch_size, replace=True)
        i_analog = rng.choice(np.arange(0, analog_input.shape[0]), batch_size, replace=True)
        # target mse is mean over trajectory (3), loc (1)
        targets = np.mean(np.mean(metrics.get_target_error(settings, soi_output[i_soi], analog_output[i_analog]), axis=3), axis=1)
        yield [soi_input[i_soi, :, :, :], analog_input[i_analog, :, :, :]], [targets]

# -------------------------------------------------------------------------------

def train_model(settings, model, analog_input, analog_output, soi_train_input, soi_train_output, soi_val_input, soi_val_output, overwrite_model=False):
    experiment_name = settings['experiment_name']
    # check if model already exists or overwrite is on
    if not (os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_weights.h5")) and os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_model"))) or overwrite_model is True:
        # model is not already saved or overwrite has been requested, so train and save model
    
        # EARLY STOPPING
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=settings["patience"], 
            min_delta=settings["min_delta"],
            verbose=1, mode='auto', 
            restore_best_weights=True)

        # LEARNING RATE
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=settings["learning_rate"])
        loss = settings["loss"]

        # COMPILE THE MODEL
        model.compile(loss=loss, optimizer=optimizer)#, metrics=[metrics.pred_range,],)

        # continuosly returns batch_size number of soi and analog inputs (features),
        # along with their target absolute differences (labels)
        # despite an individual batch of soi and analog being unlikely to be ideal for training,
        # this can be repeated with different random batches until the loss is acceptable 
        training_data = batch_generator(settings,
                                        soi_train_input, soi_train_output,
                                        analog_input, analog_output,
                                        batch_size=settings["batch_size"],
                                        rng_seed=settings["seed"]+2)

        # same as above, but for validation data
        gen = batch_generator(settings,
                              soi_val_input, soi_val_output,
                              analog_input, analog_output,
                              batch_size=settings["val_batch_size"],
                              rng_seed=settings["seed"]+1) 

        # gets the generator started (so it returns values)
        x_val, y_val = next(gen)
        # uses this for all training batches (static validation data)
        gen.close()

        class PrintCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.mask_training = []
                self.savepath = os.path.join(os.path.dirname(__file__), 'figures', experiment_name)
                if not os.path.exists(self.savepath):
                    os.makedirs(self.savepath)
            def on_epoch_begin(self, epoch, logs=None):
                current_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy().astype(FLOAT_TYPE)
                self.mask_training.append(current_mask)
            def on_train_end(self, logs=None):
                final_epoch = earlystopping.stopped_epoch - settings['patience'] + 1
                self.mask_training = self.mask_training[:final_epoch]
                np.savetxt(os.path.join(self.savepath, 'mask_training'), self.mask_training)

        # train the model
        history = model.fit(
            training_data,
            validation_data=(x_val, y_val),
            steps_per_epoch=soi_train_input.shape[0]//settings["batch_size"],
            epochs=settings["max_epochs"],
            callbacks=[earlystopping, PrintCallback()],
            verbose=1,
        )
        
        # save the model, then return the model along with fit_summary and history
        save_model_run(model, settings)
        return model, history
    
    else:
        print('loading pre-saved model')
        # load model
        model = load_model(settings)
        # load history
        with open(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_history.pickle"), 'rb') as file:
            history = pickle.load(file)
        return model, history

# -------------------------------------------------------------------------------

def save_model_run(model, settings):
    """Save the model, model weights, and training history;
            *_model,
            *_weights.h5,
            *_history.pickle

    Arguments
    ---------
    model : tensorflow.keras.models.Model

    settings : dict
        Dictionary of experiment settings for the run.
    """
    experiment_name = settings['experiment_name']
    # Save the model, weights, and history.
    try:
        with warnings.catch_warnings():  # catch very annoying tf warnings
            warnings.simplefilter("ignore")
            tf.keras.models.save_model(model, os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_model"), overwrite=True)
    except:
        print('unable to save the model, skipping saving the full model.')

    model.save_weights(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_weights.h5"))

    with open(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_history.pickle"), "wb") as handle:
        pickle.dump(model.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------

def load_model(settings):
    experiment_name = settings['experiment_name']
    model_savename = os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+"_model")
    model = tf.keras.models.load_model(model_savename, compile=False)
    return model
