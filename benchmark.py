import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import random as python_random

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
tf.config.experimental.enable_op_determinism()

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from GradientAccumulator.accumulator import GradientAccumulator, OldGradientAccumulator
from GradientAccumulator.adamAccumulate import AdamAccumulated
from GradientAccumulator.GAModelWrapper import GAModelWrapper
import argparse
import sys


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', metavar='--bs', type=int, nargs='?', default=64,
                        help="which batch size to use.")
    parser.add_argument('--accum_steps', metavar='--acs', type=int, nargs='?', default=1,
                        help="number of accumulation steps.")
    parser.add_argument('--epochs', metavar='--eps', type=int, nargs='?', default=3,
                        help="number of epochs.")
    parser.add_argument('--accum_opt', metavar='--opt', type=int, nargs='?', default=-1,
                        help="which gradient accumulator approach to use. Four available: {0, 1, 2, 3, -1}.")
    ret = parser.parse_args(sys.argv[1:]); print(ret)

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(ret.batchsize)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(ret.batchsize)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # compile model
    opt = tf.keras.optimizers.Adam(1e-3)
    # opt = tf.keras.optimizers.SGD(0.1)
    if ret.accum_opt == 0:
        pass  # no accumulated gradients - normal Adam
    elif ret.accum_opt == 1:
        opt = AdamAccumulated(accumulation_steps=ret.accum_steps)
    elif ret.accum_opt == 2:
        opt = GradientAccumulator(opt, accum_steps=ret.accum_steps)
    elif ret.accum_opt == 3:
        opt = OldGradientAccumulator(opt, accum_steps=ret.accum_steps)
    elif ret.accum_opt == -1:
        model = GAModelWrapper(n_gradients=ret.accum_steps, inputs=model.input, outputs=model.output)
    else:
        raise ValueError("Uknown accumulate gradient method was chosen. Available wrappers are: {0, 1, 2, 3}.")

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=ret.epochs,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True, custom_objects={
        "GradientAccumulator": GradientAccumulator,  # all these custom objects are added for convenience when testing - all are not needed otherwise
        "AdamAccumulated": AdamAccumulated,
        "OldGradientAccumulator": OldGradientAccumulator,
        "Adam": tf.keras.optimizers.Adam(1e-3)
        })

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
