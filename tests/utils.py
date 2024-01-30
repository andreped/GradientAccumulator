import os
import random as python_random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel, GradientAccumulateOptimizer

# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def reset(seed=123):
    # set tf log level
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # clear keras session
    tf.keras.backend.clear_session()

    os.environ["PYTHONHASHSEED"] = str(seed)

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(seed)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    # @TODO: Should this seed be different than for python and numpy?
    tf.random.set_seed(seed)

    # https://stackoverflow.com/a/71311207
    try:
        tf.config.experimental.enable_op_determinism()  # Exist only for TF > 2.7
    except AttributeError as e:
        print(e)

    # force cpu threading determinism
    # https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def get_opt(opt_name, tf_version=None):
    if tf_version is None:
        tf_version = int(tf.version.VERSION.split(".")[1])
    if opt_name == "adam":
        if tf_version > 10:
            curr_opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        else:
            curr_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif opt_name == "adadelta":
        if tf_version > 10:
            curr_opt = tf.keras.optimizers.legacy.Adadelta(learning_rate=1e-2)
        else:
            curr_opt = tf.keras.optimizers.Adadelta(learning_rate=1e-2)
    elif opt_name == "RMSprop":
        if tf_version > 10:
            curr_opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-3)
        else:
            curr_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    elif opt_name == "SGD":
        if tf_version > 10:
            curr_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
        else:
            curr_opt = tf.keras.optimizers.SGD(learning_rate=1e-2)
    else:
        raise ValueError("Unknown optimizer chosen.")

    return curr_opt


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def gray2rgb(image, label):
    """Converts images from gray to RGB."""
    return tf.concat([image, image, image], axis=-1), label


def resizeImage(image, label, output_shape=(32, 32)):
    """Resizes images."""
    return tf.image.resize(image, output_shape, method="nearest"), label


def run_experiment(bs=50, accum_steps=2, epochs=1, modeloropt="opt"):
    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    # create model
    input = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(10)(x)

    opt = get_opt(opt_name="SGD", tf_version=tf_version)

    if accum_steps == 1:
        model = tf.keras.Model(inputs=input, outputs=output)
    else:
        if modeloropt == "model":
            # wrap model to use gradient accumulation
            model = GradientAccumulateModel(
                accum_steps=accum_steps, inputs=input, outputs=output
            )
        else:
            # wrap optimizer to use gradient accumulation
            opt = GradientAccumulateOptimizer(opt, accum_steps=accum_steps)

            # compile model
            model = tf.keras.Model(inputs=input, outputs=output)

    # compile model
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)

    return result[1]
