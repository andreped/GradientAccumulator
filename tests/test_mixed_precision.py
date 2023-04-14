import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel
from tensorflow.keras import mixed_precision
import multiprocessing as mp
import os


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def run_experiment():
    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set mixed global precision policy
    # Equivalent to the two lines above
    # https://www.tensorflow.org/guide/mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # NOTE: On GPUs, ensure most tensor dimensions are a multiple
    # of 8 to maximize performance

    # build train pipeline
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(32)  # multiplum of 8
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(32)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32, activation='relu'),  # 32 multiplum of 8
        tf.keras.layers.Dense(10, dtype='float32')  # output not numerically stable with float16
    ])

    # wrap model to use gradient accumulation
    model = GradientAccumulateModel(accum_steps=4, mixed_precision=True, inputs=model.input, outputs=model.output)

    # need to scale optimizer for mixed precision
    opt = tf.keras.optimizers.Adam(1e-3)
    opt = mixed_precision.LossScaleOptimizer(opt)

    # compile model
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=1,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


def test_mixed_precision():
    # launch experiment in separate process, as we are enabling mixed precision
    # which will impact other unit tests, unless we do this
    p = mp.Process(target=run_experiment)
