import pytest
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator import GradientAccumulateOptimizer
from gradient_accumulator import unitwise_norm

from .utils import normalize_img, get_opt


tf_version = int(tf.version.VERSION.split(".")[1])

def test_unitwise_norm():
    for i in range(7):
        x = tf.zeros(
            [
                1,
            ]
            * i
        )
        try:
            unitwise_norm(x)
        except ValueError as e:
            # i=6 should yield ValueError. If it happens otherwise, raise error
            if i != 6:
                raise e


@pytest.fixture
def generate_experiment_prerequisites():
    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # NOTE: On GPUs, ensure most tensor dimensions are a multiple
    # of 8 to maximize performance

    # build train pipeline
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(100)  # multiplum of 8
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(100)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(1)

    # create model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(32, activation="relu"),  # 32 multiplum of 8
            tf.keras.layers.Dense(
                10, dtype="float32"
            ),  # output not numerically stable with float16
        ]
    )
    return model, ds_train, ds_test


def test_train_mnist_model(generate_experiment_prerequisites):

    model, ds_train, ds_test = generate_experiment_prerequisites

    # Test AGC with model
    # wrap model to use gradient accumulation
    model = GradientAccumulateModel(
        accum_steps=4,
        mixed_precision=False,
        use_agc=True,
        inputs=model.input,
        outputs=model.output,
    )

    # need to scale optimizer for mixed precision
    opt = get_opt(opt_name="SGD", tf_version=tf_version)
        
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


def test_train_mnist_optimizer(generate_experiment_prerequisites):

    model, ds_train, ds_test = generate_experiment_prerequisites


    # wrap model to use gradient accumulation
    model = tf.keras.Model(inputs=model.input, outputs=model.output)

    opt = get_opt(opt_name="SGD", tf_version=tf_version)

    # need to scale optimizer for mixed precision
    opt = GradientAccumulateOptimizer(opt, accum_steps=4, mixed_precision=False, use_agc=True)

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


# for running locally, outside pytest
if __name__ == "__main__":
    test_train_mnist_model()
    test_train_mnist_optimizer()
