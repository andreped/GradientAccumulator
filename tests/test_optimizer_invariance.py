import os
import random as python_random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator import GradientAccumulateOptimizer

from .utils import get_opt
from .utils import normalize_img
from .utils import reset

# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def run_experiment(
    bs=100, accum_steps=1, epochs=1, opt_name="SGD", wrapper="model"
):
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
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    input = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(10)(x)
    model = tf.keras.models.Model(inputs=input, outputs=output)

    # define optimizer
    opt = get_opt(opt_name)

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        if wrapper == "model":
            model = GradientAccumulateModel(
                accum_steps=accum_steps, inputs=input, outputs=output
            )
        elif wrapper == "optimizer":
            opt = GradientAccumulateOptimizer(
                optimizer=opt, accum_steps=accum_steps
            )
        else:
            raise ValueError("Unknown wrapper was chosen:", wrapper)

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


def test_optimizer_invariance():
    # run experiment for different optimizers, to see if GA is consistent
    # within an optimizer. Note that it is expected for the results to
    # differ BETWEEN optimizers, as they behave differently.
    for wrapper in ["model", "optimizer"]:
        for opt_name in ["SGD", "adam", "RMSprop"]:
            print("Current experiment:", wrapper, opt_name)
            # set seed
            reset()

            # run once
            result1 = run_experiment(
                bs=100,
                accum_steps=1,
                epochs=2,
                opt_name=opt_name,
                wrapper=wrapper,
            )

            # reset before second run to get identical results
            reset()

            # run again with different batch size and number of accumulations
            result2 = run_experiment(
                bs=50,
                accum_steps=2,
                epochs=2,
                opt_name=opt_name,
                wrapper=wrapper,
            )

            # results should be "identical" (on CPU, can be different on GPU)
            np.testing.assert_almost_equal(
                result1, result2, decimal=2
            )  # decimals=3 OK for model wrapper but not optimizer
