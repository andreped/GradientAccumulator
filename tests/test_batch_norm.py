import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator.layers import AccumBatchNormalization
import random as python_random
import numpy as np
import os
from .utils import reset, normalize_img


def run_experiment(custom_bn:bool = True, bs:int = 100, accum_steps:int = 1, epochs:int = 3):
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
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    # define which normalization layer to use in network
    if custom_bn:
        normalization_layer = AccumBatchNormalization(accum_steps=accum_steps)
    elif not custom_bn:
        normalization_layer = tf.keras.layers.BatchNormalization()
    else:
        normalization_layer = tf.keras.layers.Activation("linear")

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32),
        normalization_layer,  # @TODO: BN before or after ReLU? Leads to different performance
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10)
    ])

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        model = GradientAccumulateModel(accum_steps=accum_steps, inputs=model.input, outputs=model.output)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-2),
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
    return result


def test_compare_bn_layers():
    # set seed
    reset()
    
    # custom BN without accum
    result1 = run_experiment(custom_bn=True, accum_steps=1, epochs=3)[1]
    
    # reset before second run to get "identical" results
    reset()

    # keras BN without accum
    result2 = run_experiment(custom_bn=False, accum_steps=1, epochs=3)[1]

    print(result1, result2)

    # results should be *identical* for accum_steps=1
    assert result1 == result2


def test_compare_accum_bn_expected_result():
    # set seed
    reset()
    
    # custom BN without accum
    result1 = run_experiment(custom_bn=True, accum_steps=4, bs=25)[1]
    
    # reset before second run to get "identical" results
    reset()

    # keras BN without accum
    result2 = run_experiment(custom_bn=True, accum_steps=1, bs=100)[1]

    print(result1, result2)

    np.testing.assert_almost_equal(result1, result2, decimal=2)
    # assert result1 == result2
