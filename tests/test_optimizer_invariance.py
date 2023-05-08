import numpy as np
import tensorflow as tf
import random as python_random
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel, GradientAccumulateOptimizer


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
        raise ValueError("Unknown optimizer chosen:", opt_name)

    return curr_opt


def reset():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # clear keras session
    tf.keras.backend.clear_session()

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


def run_experiment(bs=100, accum_steps=1, epochs=1, opt_name="SGD", wrapper="model"):
    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # define optimizer
    opt = get_opt(opt_name)

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        if wrapper == "model":
            model = GradientAccumulateModel(accum_steps=accum_steps, inputs=model.input, outputs=model.output)
        elif wrapper == "optimizer":
            opt = GradientAccumulateOptimizer(optimizer=opt, accum_steps=accum_steps)
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
            result1 = run_experiment(bs=100, accum_steps=1, epochs=2, opt_name=opt_name, wrapper=wrapper)

            # reset before second run to get identical results
            reset()

            # run again with different batch size and number of accumulations
            result2 = run_experiment(bs=50, accum_steps=2, epochs=2, opt_name=opt_name, wrapper=wrapper)

            # results should be "identical" (on CPU, can be different on GPU)
            np.testing.assert_almost_equal(result1, result2, decimal=2)  # decimals=3 OK for model wrapper but not optimizer
