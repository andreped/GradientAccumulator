import numpy as np
import tensorflow as tf
import random as python_random
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model, load_model
from gradient_accumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, UpSampling2D,\
    MaxPooling2D, Activation


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def create_multi_input_output(image, label):
    return (image, image), (image, label)


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

    # get a smaller amount of samples for running the experiment
    ds_train = ds_train.take(1024)
    ds_test = ds_test.take(1024)

    # build train pipeline
    ds_train = ds_train.map(normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(create_multi_input_output, 
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(create_multi_input_output, 
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create multi-input multi-output model
    input1 = Input(shape=(28, 28, 1))
    input2 = Input(shape=(28, 28))

    x1 = Conv2D(8, (5, 5), activation="relu", padding="same")(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(8, (5, 5), activation="relu", padding="same")(x1)
    x1 = UpSampling2D((2, 2))(x1)
    x1 = Conv2D(8, (5, 5), padding="same", name="reconstructor")(x1)

    x2 = Flatten()(input2)
    x2 = Dense(32, activation="relu")(x2)
    x2 = Dense(10, name="classifier")(x2)

    # [2.0176000595092773, 0.09766767919063568, 1.9199140071868896, 0.46810001134872437]
    # [1.6557999849319458, 0.08378496021032333, 1.5719828605651855, 0.6689000129699707]

    model = Model(inputs=[input1, input2], outputs=[x1, x2])

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        model = GAModelWrapper(n_gradients=accum_steps, inputs=model.input, outputs=model.output)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-3),
        loss={"classifier": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              "reconstructor": "mse"},
        metrics={"classifier": tf.keras.metrics.SparseCategoricalAccuracy()},
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

    return np.array(result)


def test_multitask():
    # set seed
    reset()

    # run once
    result1 = run_experiment(bs=32, accum_steps=1, epochs=1)

    # reset before second run to get reproducible results
    reset()

    # run again with different batch size and number of accumulations
    result2 = run_experiment(bs=16, accum_steps=2, epochs=1)

    # results should be "identical" (on CPU, can be different on GPU)
    np.testing.assert_almost_equal(result1, result2, decimal=3)
