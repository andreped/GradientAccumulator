import numpy as np
import tensorflow as tf
import random as python_random
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

#import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from GradientAccumulator.accumulator import GradientAccumulator
from GradientAccumulator.adamAccumulate import AdamAccumulated
import numpy as np


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":

    # params
    batchsize = 32  # 8 vs 32
    accum_steps = 1  # 4 vs 1
    nb_epochs = 3  # 3

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batchsize)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batchsize)
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
    model.compile(
        optimizer=opt if accum_steps == 1 else AdamAccumulated(accumulation_steps=accum_steps),  # GradientAccumulator(opt, accum_steps=accum_steps),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=nb_epochs * accum_steps,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True, custom_objects={"GradientAccumulator": GradientAccumulator})

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
