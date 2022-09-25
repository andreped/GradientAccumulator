import numpy as np
import tensorflow as tf
import random as python_random
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator.GAModelWrapper import GAModelWrapper
import pandas as pd
import shutil
from tqdm import tqdm
from typing import Optional, Dict
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa


# constants
iter = 0
maxlen = 50
mask_value = 0
feature_names = ['accel_x', 'accel_y', 'accel_z']


def merge(x):
    out = []
    for key_ in feature_names:
        tmp = x[key_]
        out.append(tmp)
    out = tf.concat(out, axis=1)
    return out


def setup_dataset(dataset):
    dataset = dataset.map(lambda x, y: ({elem: tf.expand_dims(x[elem], axis=-1) for elem in x},
                                        tf.expand_dims(y, axis=0)))
    return dataset.map(lambda x, y: (merge(x), y))


def pad(dataset, value=mask_value):
    return dataset.map(lambda x, y: (tf.pad(x, [[0, maxlen - tf.shape(x)[0]], [0, 0]],
                                            mode='CONSTANT', constant_values=value), y))


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def reset():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(123)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(1234)

    # https://stackoverflow.com/a/71311207
    tf.config.experimental.enable_op_determinism()  # @TODO: Not working on windows with GPU (works fine with CPU)

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU faster for mnist experiments!


def run_experiment(bs, accum_steps, epochs, opt_name, norm, updates, dataset):
    # load dataset
    ds_train, ds_test = tfds.load(
        dataset,
        # @TODO: :80 to 80: not working for whatever reason...
        split=['train[:80%]', 'train[90%:]'] if dataset == "smartwatch_gestures" else ['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )

    # preprocess data if smartwatch_gestures dataset
    if dataset == "smartwatch_gestures":
        ds_train = setup_dataset(ds_train)
        ds_train = pad(ds_train)

        ds_test = setup_dataset(ds_test)
        ds_test = pad(ds_test)
    else:
        # normalize images
        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    # build train/test pipelines
    ds_train = ds_train.batch(bs).prefetch(tf.data.AUTOTUNE).repeat(-1)
    ds_test = ds_test.batch(bs).prefetch(tf.data.AUTOTUNE)

    # choose which normalization to use
    if norm == "batch":
        norm_layer = tf.keras.layers.BatchNormalization()
    elif norm == "group":
        norm_layer = tfa.layers.GroupNormalization()
    elif norm in ["none", "agc"]:
        norm_layer = tf.keras.layers.Layer()
    else:
        raise ValueError("No valid normalization method was used. Please, choose one of: {batch, group, none}.")

    if dataset == "smartwatch_gestures":
        nb_classes = 20
        # maxlen = 50
        inputs = tf.keras.layers.Input(shape=(maxlen, 3))

        x = tf.keras.layers.Masking(mask_value=mask_value, input_shape=(maxlen, 3))(inputs)
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(nb_classes)(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
    else:
        # choose input shape dependent on dataset
        if dataset == "mnist":
            input_shape = (28, 28, 1)
            nb_classes = 10
        elif dataset == "cifar10":
            input_shape = (32, 32, 3)
            nb_classes = 10
        elif dataset == "cifar100":
            input_shape = (32, 32, 3)
            nb_classes = 100
        else:
            raise ValueError("Unknown dataset chosen. Please, choose one of: {mnist, cifar10, cifar100}.")

        # create model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            norm_layer,
            tf.keras.layers.Dense(nb_classes)
        ])

    # wrap model to use gradient accumulation
    model = GAModelWrapper(accum_steps=accum_steps, inputs=model.input, outputs=model.output,
                           use_agc=True if norm == "agc" else False)

    # choose optimizer
    if opt_name == "SGD":
        opt = tf.keras.optimizers.SGD(1e-2)
    elif opt_name == "Adam":
        opt = tf.keras.optimizers.Adam(1e-3)
    else:
        raise ValueError("Unknown optimizer chosen. Please, choose between {SGD, Adam}.")

    # compile model
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    log_dir = "logs/fit/" + str(iter) + "_bs_" + str(bs) + "_acs_" + str(accum_steps) +\
              "_eps_" + str(epochs) + "_opt_" + str(opt_name) + "_norm_" + norm + "_updates_" + str(updates) +\
              "_dataset_" + dataset
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq="epoch")

    # train model
    model.fit(
        ds_train,
        epochs=epochs,
        batch_size=bs,
        steps_per_epoch=int(updates * accum_steps),  # k updates each epoch
        validation_data=ds_test,
        callbacks=[tensorboard_callback],
        validation_steps=None if dataset is "smartwatch_gestures" else 100,
        verbose=0,
    )


def test_expected_result():
    for bs, acs in tqdm(zip([1, 2, 4, 8, 16, 32, 64, 128] * 2 + [1] * 8,
                            [1] * 8 + [128, 64, 32, 16, 8, 4, 2, 1] * 2), total=24):
        for opt_ in ["Adam"]:  # , "SGD"]:
            for norm in ["none", "batch", "group", "agc"]:
                for dataset in ["smartwatch_gestures", "mnist", "cifar100"]:
                    try:
                        # skip specific normalizers for RNN experiments, as they are not relevant
                        if (dataset == "smartwatch_gestures") and (norm in ["batch", "group"]):
                            continue
                        # set seed
                        reset()

                        # run once
                        run_experiment(bs=bs, accum_steps=acs, epochs=50, opt_name=opt_, norm=norm, updates=100,
                                       dataset=dataset)

                        global iter
                        iter += 1
                    except Exception as e:
                        print("Something went wrong for setup:", bs, acs, opt_, norm, dataset)
                        print(e)


if __name__ == "__main__":
    # remove old logs
    if os.path.exists("./logs/"):
        shutil.rmtree("./logs/")

    # run experiments
    test_expected_result()
