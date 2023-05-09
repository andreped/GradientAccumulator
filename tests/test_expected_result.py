import numpy as np
import tensorflow as tf
import random as python_random
import os
from .utils import get_opt, normalize_img, reset
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel, GradientAccumulateOptimizer


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def run_experiment(bs=50, accum_steps=2, epochs=1, modeloropt="opt"):
    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
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
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(10)(x)

    opt = get_opt(opt_name="SGD", tf_version=tf_version)

    if accum_steps == 1:
        model = tf.keras.Model(inputs=input, outputs=output)
    else:
        if modeloropt == "model":
            # wrap model to use gradient accumulation
            model = GradientAccumulateModel(accum_steps=accum_steps, inputs=input, outputs=output)
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


def test_expected_result():
    # set seed
    reset()

    # run once
    result1 = run_experiment(bs=100, accum_steps=1, epochs=2, modeloropt="opt")

    # reset before second run to get identical results
    reset()

    # run again with different batch size and number of accumulations
    result2 = run_experiment(bs=50, accum_steps=2, epochs=2, modeloropt="opt")
    
    # reset again
    reset()

    # test with model wrapper instead
    result3 = run_experiment(bs=50, accum_steps=2, epochs=2, modeloropt="model")

    # results should be identical (theoretically, even in practice on CPU)
    if tf_version <= 10:
        assert result1 == result2
        assert result2 == result3
    else:
        # approximation worse for tf >= 2.11
        np.testing.assert_almost_equal(result1, result2, decimal=2)
        np.testing.assert_almost_equal(result2, result3, decimal=2)


if __name__ == "__main__":
    test_expected_result()
