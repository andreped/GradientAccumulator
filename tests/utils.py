import random as python_random
import tensorflow as tf
import numpy as np
import os


def reset(seed=123):
    # set tf log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    tf.random.set_seed(1234)  # @TODO: Should this seed be different than for python and numpy?

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
    return tf.cast(image, tf.float32) / 255., label
