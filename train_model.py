import time
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
# 0 = all messages are logged (default behavior) , 1 = INFO messages are not printed , 2 = INFO and WARNING messages
# are not printed , 3 = INFO, WARNING, and ERROR messages are not printed
tf.get_logger().setLevel('ERROR')

# TODO: incorporate trends (tendencies) in the inputs for both short (1 year) and long (5 or 10 years), essentially want d/dt
# TODO: try sample weighting on the target distance (inverse)

def batch_generator(settings, soi_input, soi_output, analog_input, analog_output, batch_size, rng_seed=33):
    ""
    # random seed to select random subsets of the soi and analog sets
    rng = np.random.default_rng(rng_seed)
    # create a generator that returns random samples of the inputs (soi, analog) and the target (output difference)
    while True:
        # random subsamples of size batch_size
        i_soi = rng.choice(np.arange(0, soi_input.shape[0]), batch_size, replace=True)
        i_analog = rng.choice(np.arange(0, analog_input.shape[0]), batch_size, replace=True)
        if settings['method'] == 'jacob':
            targets = np.mean(metrics.get_targets(settings, soi_output[i_soi], analog_output[i_analog]), axis=(1, 2))
        else:
            targets = metrics.get_targets(settings, soi_output[i_soi], analog_output[i_analog])
        yield [soi_input[i_soi, :, :, :], analog_input[i_analog, :, :, :]], [targets]

# -------------------------------------------------------------------------------

def train_model(settings, model, analog_input, analog_output, soi_train_input, soi_train_output, soi_val_input, soi_val_output, overwrite_model=False):
    """
    
    """
    experiment_name = settings['experiment_name']
    seed = settings['seed']
    # check if model already exists or overwrite is on
    if not (os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_weights.h5")) and os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_model"))) or overwrite_model is True:
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

        ###################################################################################
        # LOSS FUNCTION, INCLUDING SETTING HUBER LOSS DELTA PARAMETER
        # if settings["loss_f"] == 'huber':
        #     gen = batch_generator(settings,
        #                           soi_train_input, soi_train_output,
        #                           analog_input, analog_output, batch_size=2_500,
        #                           rng_seed=settings["seed"]+4)
        #     __, random_targets = next(gen)
        #     gen.close()
        #     huber_delta = np.percentile(random_targets[:], settings["percentile_huber_d"])
        #     settings["huber_delta"] = huber_delta
        #     loss = tf.keras.losses.Huber(delta=huber_delta,)
        # else:
        #     loss = settings["loss_f"]
        ###################################################################################
        
        loss = settings["loss"]

        # COMPILE THE MODEL
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.pred_range,],)

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
            def on_epoch_begin(self, epoch, logs=None):
                savepath = os.path.join(os.path.dirname(__file__), 'figures', experiment_name+'_train')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                current_mask = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy()
                print('')
                print(current_mask)
                if epoch % 2 == 0:
                    np.savetxt(os.path.join(savepath, 'mask_epoch'+str(epoch).zfill(5)), current_mask)

        # train the model
        start_time = time.time()
        history = model.fit(
            training_data,
            validation_data=(x_val, y_val),
            steps_per_epoch=soi_train_input.shape[0]//settings["batch_size"],
            epochs=settings["max_epochs"],
            callbacks=[earlystopping, PrintCallback()],
            verbose=1,
        )
        stop_time = time.time()

        # DISPLAY THE RESULTS
        best_epoch = np.argmin(history.history["val_loss"])
        fit_summary = {
            "elapsed_time": stop_time - start_time,
            "best_epoch": best_epoch + 1,
            "loss_train": history.history["loss"][best_epoch],
            "loss_valid": history.history["val_loss"][best_epoch],
        }
        
        # save the model, then return the model along with fit_summary and history
        save_model_run(model, settings)
        return model, fit_summary, history
    
    else:
        print('loading pre-saved model')
        # load model
        model = load_model(settings)
        # load history
        with open(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_history.pickle"), 'rb') as file:
            history = pickle.load(file)
        # for loaded model, just return the losses
        fit_summary = {
            "best_epoch": np.argmin(history["val_loss"]) + 1,
            "loss_train": history["loss"][np.argmin(history["val_loss"]) + 1],
            "loss_valid": history["val_loss"][np.argmin(history["val_loss"]) + 1],
        }
        return model, fit_summary, history

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
    seed = settings['seed']
    # Save the model, weights, and history.
    try:
        with warnings.catch_warnings():  # catch very annoying tf warnings
            warnings.simplefilter("ignore")
            tf.keras.models.save_model(model, os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_model"), overwrite=True)
    except:
        print('unable to save the model, skipping saving the full model.')

    model.save_weights(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_weights.h5"))

    with open(os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_history.pickle"), "wb") as handle:
        pickle.dump(model.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------

def load_model(settings):
    experiment_name = settings['experiment_name']
    seed = settings['seed']
    model_savename = os.path.join(os.path.dirname(__file__), 'saved_models', experiment_name, experiment_name+str(seed)+"_model")
    model = tf.keras.models.load_model(model_savename, compile=False)
    return model
