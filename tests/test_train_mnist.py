import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


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
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(1)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # wrap model to use gradient accumulation
    model = GradientAccumulateModel(accum_steps=4, inputs=model.input, outputs=model.output)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=3,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


if __name__ == "__main__":
    test_train_mnist()
