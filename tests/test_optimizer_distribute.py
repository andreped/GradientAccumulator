import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateOptimizer
import random as python_random
import os
import numpy as np


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def get_opt(opt_name):
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


def reset():
    # set tf log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    try:
        tf.config.experimental.enable_op_determinism()  # Exist only for TF > 2.7
    except AttributeError as e:
        print(e)


def run_experiment(opt_name="adam", bs=100, accum_steps=1, epochs=1, strategy_name="multi"):
    # setup single/multi-GPU strategy
    if strategy_name == "single":
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif strategy_name == "multi":
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise ValueError("Unknown distributed strategy chosen:", strategy_name)

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    with strategy.scope():
        # create model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # define optimizer - currently only SGD compatible with GAOptimizerWrapper
        opt = get_opt(opt_name=opt_name)

        # wrap optimizer to add gradient accumulation support
        opt = GradientAccumulateOptimizer(optimizer=opt, accum_steps=accum_steps)

        # add loss scaling relevant for mixed precision
        # opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)  # @TODO: Should this be after GAOptimizerWrapper?

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
        verbose=1
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
    return result[1]


def test_distributed_optimizer_invariance():
    # run experiment for different optimizers, to see if GA is consistent 
    # within an optimizer. Note that it is expected for the results to
    # differ BETWEEN optimizers, as they behave differently.
    for strategy_name in ["single", "multi"]:
        for opt_name in ["adam", "SGD"]:
            print("Current optimizer:" + opt_name)
            # set seed
            reset()

            # run once
            result1 = run_experiment(opt_name=opt_name, bs=100, accum_steps=1, epochs=1, strategy_name=strategy_name)

            # reset before second run to get identical results
            reset()

            # run again with different batch size and number of accumulations
            result2 = run_experiment(opt_name=opt_name, bs=50, accum_steps=2, epochs=1, strategy_name=strategy_name)

            # results should be "identical" (on CPU, can be different on GPU)
            np.testing.assert_almost_equal(result1, result2, decimal=3)
