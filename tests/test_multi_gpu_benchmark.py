import numpy as np
import tensorflow as tf
import random as python_random
import os
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator.GAModelWrapper import GAModelWrapper
from gradient_accumulator.GAModelWrapperV2 import GAModelWrapperV2
from tensorflow.keras import mixed_precision


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def run_experiment(bs=16, accum_steps=4, epochs=1, strategy=None):

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

    # dataset sizes
    N_train = len(ds_train)
    N_test = len(ds_test)

    # distribute datasets
    #ds_train = strategy.experimental_distribute_dataset(ds_train)
    #ds_test = strategy.experimental_distribute_dataset(ds_test)

    # create model
    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),  # 32 multiplum of 8
            tf.keras.layers.Dense(10, dtype='float32')  # output not numerically stable with float16
        ])

        # wrap model to use gradient accumulation
        model = GAModelWrapperV2(accum_steps=accum_steps, mixed_precision=True, use_agc=False,
                                 inputs=model.input, outputs=model.output, model=model, strategy=strategy)

        # compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            run_eagerly=False,
        )

    # train model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        #steps_per_epoch=N_train,
        #validation_steps=N_test
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
    #reset()

    # run once
    result1 = run_experiment(bs=32, accum_steps=1, epochs=2)

    # reset before second run to get identical results
    #reset()

    # run again with different batch size and number of accumulations
    result2 = run_experiment(bs=16, accum_steps=2, epochs=2)

    # results should be identical (theoretically, even in practice on CPU)
    assert result1 == result2


if __name__ == "__main__":
    #def reset():
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

    # use multiple GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # set mixed global precision policy
    # Equivalent to the two lines above
    # https://www.tensorflow.org/guide/mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    # use distribution strategy to train with multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

    result1 = run_experiment(bs=32, accum_steps=2, epochs=2, strategy=strategy)

    #test_expected_result()
