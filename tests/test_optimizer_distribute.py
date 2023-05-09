import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateOptimizer
import numpy as np
from .utils import reset, get_opt


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def run_experiment(opt_name="adam", bs=100, accum_steps=1, epochs=1, strategy_name="multi"):
    # setup single/multi-GPU strategy
    if strategy_name == "single":
        strategy = tf.distribute.get_strategy()  # get default strategy
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
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
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
        opt = get_opt(opt_name=opt_name, tf_version=tf_version)

        # wrap optimizer to add gradient accumulation support
        opt = GradientAccumulateOptimizer(optimizer=opt, accum_steps=accum_steps)

        # compile model
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # train model
    model.fit(
        ds_train,
        batch_size=bs,
        epochs=epochs,
        validation_data=ds_test,
        verbose=1
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    del strategy

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
    return result[1]


def test_distributed_optimizer_invariance():
    # run experiment for different optimizers, to see if GA is consistent 
    # within an optimizer. Note that it is expected for the results to
    # differ BETWEEN optimizers, as they behave differently.
    for strategy_name in ["single", "multi"]:
        for opt_name in ["SGD", "adam"]:
            print("Current optimizer:" + opt_name)
            # set seed
            reset()

            # run once
            result1 = run_experiment(opt_name=opt_name, bs=100, accum_steps=1, epochs=2, strategy_name=strategy_name)

            # reset before second run to get identical results
            reset()

            # run again with different batch size and number of accumulations
            result2 = run_experiment(opt_name=opt_name, bs=50, accum_steps=2, epochs=2, strategy_name=strategy_name)

            # results should be "identical" (on CPU, can be different on GPU)
            np.testing.assert_almost_equal(result1, result2, decimal=3)
