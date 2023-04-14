import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateOptimizer


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def add_sample_weight(image, label):
    """Adds toy sample weight to data sample."""
    return (image, label, 1)  # sample_weight=1 is used to set equal weight to all inputs -> just used for unit testing


def test_train_mnist():
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
    ds_train = ds_train.map(add_sample_weight)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.map(add_sample_weight)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ])

    # wrap optimizer to add gradient accumulation support
    # need to dynamically handle which Optimizer class to use dependent on tf version
    if tf_version > 10:
        curr_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
    else:
        curr_opt = tf.keras.optimizers.SGD(learning_rate=1e-2)  # IDENTICAL RESULTS WITH SGD!!!

    opt = GradientAccumulateOptimizer(optimizer=curr_opt, accum_steps=4, reduction="MEAN")

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
    trained_model = load_model("./trained_model", compile=True, custom_objects={"SGD": curr_opt})

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


if __name__ == "__main__":
    test_train_mnist()
