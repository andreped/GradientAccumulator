import numpy as np
import tensorflow as tf
import random as python_random
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator.GAModelWrapper import GAModelWrapper


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
    tf.config.experimental.enable_op_determinism()

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_experiment(bs=16, accum_steps=4, epochs=1):
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
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.BatchNormalization(),  # @TODO: BN not compatible with GA solution currently! Yields different results
        tf.keras.layers.Dense(10)
    ])

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        model = GAModelWrapper(n_gradients=accum_steps, inputs=model.input, outputs=model.output)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-3),
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


def test_expected_result():
    # set seed
    reset()

    # run once
    result1 = run_experiment(bs=32, accum_steps=1, epochs=2)

    # reset before second run to get identical results
    reset()

    # run again with different batch size and number of accumulations
    result2 = run_experiment(bs=16, accum_steps=2, epochs=2)

    # results should be identical (theoretically, even in practice on CPU)
    assert result1 == result2
