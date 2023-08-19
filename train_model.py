"""Train the models.

"""

import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import build_model
import metrics

__author__ = "Jamin K. Rader, Elizabeth A. Barnes"
__version__ = "30 March 2023"



def batch_generator(settings, soi_input, soi_output, analog_input, analog_output, batch_size, rng_seed=33):
    ""
    # random seed to select random subsets of the soi and analog sets
    rng = np.random.default_rng(rng_seed)
    # create a generator that returns random samples of the inputs (soi, analog) and the target (output difference)
    while True:
        # random subsamples of size batch_size
        i_soi = rng.choice(np.arange(0, soi_input.shape[0]), batch_size, replace=True)
        i_analog = rng.choice(np.arange(0, analog_input.shape[0]), batch_size, replace=True)
        targets = metrics.get_targets(settings, soi_output[i_soi], analog_output[i_analog])
        yield [soi_input[i_soi, :, :, :], analog_input[i_analog, :, :, :]], [targets]



def train_model(settings, model, analog_input, analog_output, soi_train_input, soi_train_output, soi_val_input, soi_val_output):
    ""
    
    # EARLY STOPPING
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=settings["patience"], 
        min_delta=settings["min_delta"],
        verbose=1, mode='auto', 
        restore_best_weights=True)

    # LEARNING RATE
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"])

    # LOSS FUNCTION, INCLUDING SETTING HUBER LOSS DELTA PARAMETER
    if settings["loss_f"] == 'huber':
        gen = batch_generator(settings,
                              soi_train_input, soi_train_output,
                              analog_input, analog_output, batch_size=2_500,
                              rng_seed=settings["rng_seed"]+4)
        __, random_targets = next(gen)
        gen.close()
        huber_delta = np.percentile(random_targets[:], settings["percentile_huber_d"])
        settings["huber_delta"] = huber_delta
        loss = tf.keras.losses.Huber(delta=huber_delta,)
    else:
        loss = settings["loss_f"]
    
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
                                    rng_seed=settings["rng_seed"]+2)
    
    # same as above, but for validation data
    gen = batch_generator(settings,
                          soi_val_input, soi_val_output,
                          analog_input, analog_output,
                          batch_size=settings["val_batch_size"],
                          rng_seed=settings["rng_seed"]+1) 
        
    # gets the generator started (so it returns values)
    x_val, y_val = next(gen)
    # uses this for all training batches (static validation data)
    gen.close()

    # train the model
    start_time = time.time()
    history = model.fit(
        training_data,
        validation_data=(x_val, y_val),
        steps_per_epoch=soi_train_input.shape[0]//settings["batch_size"],
        epochs=settings["max_epochs"],
        callbacks=[earlystopping, ],
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

    return model, fit_summary, history, settings, x_val, y_val
