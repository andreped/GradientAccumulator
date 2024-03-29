import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel

from .utils import get_opt


def test_model_distribute():
    strategy = tf.distribute.MirroredStrategy()

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(100)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.batch(100)
    ds_test = ds_test.prefetch(1)

    with strategy.scope():
        # create model
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )
        model = GradientAccumulateModel(
            accum_steps=4,
            inputs=model.input,
            outputs=model.output,
            experimental_distributed_support=True,
        )

        # define optimizer - currently only SGD compatible with GAOptimizerWrapper
        opt = get_opt("SGD")

        # compile model
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # train model
    model.fit(ds_train, epochs=3, validation_data=ds_test, verbose=1)

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


if __name__ == "__main__":
    test_model_distribute()
