import pytest
import tensorflow as tf
import multiprocessing as mp
from .utils import reset, get_opt


tf_version = int(tf.version.VERSION.split(".")[1])

@pytest.fixture
def generate_experiment_prerequisites():
    import os

    import tensorflow as tf
    import tensorflow_datasets as tfds

    from .utils import normalize_img

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(
        32
    )  # multiplum of 8 on GPU to maximize performance
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(32)
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


def run_experiment_model(generate_experiment_prerequisites):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    from gradient_accumulator import GradientAccumulateModel

    # set mixed global precision policy
    mixed_precision.set_global_policy("mixed_float16")

    model, ds_train, ds_test = generate_experiment_prerequisites

    # wrap model to use gradient accumulation
    model = GradientAccumulateModel(
        accum_steps=4,
        mixed_precision=True,
        inputs=model.input,
        outputs=model.output,
    )

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

    # save model on disk
    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = tf.keras.models.load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


def run_experiment_optimizer(generate_experiment_prerequisites):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    from gradient_accumulator import GradientAccumulateOptimizer

    # set mixed global precision policy
    mixed_precision.set_global_policy("mixed_float16")

    model, ds_train, ds_test = generate_experiment_prerequisites

    # wrap model to use gradient accumulation
    model = tf.keras.Model(inputs=model.input, outputs=model.output)

    opt = get_opt(opt_name="adam", tf_version=tf_version)

    # need to scale optimizer for mixed precision
    opt = GradientAccumulateOptimizer(opt, accum_steps=4, mixed_precision=True)

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

    # save model on disk
    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = tf.keras.models.load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)


def test_mixed_precision():
    # set seed
    reset()

    # Model with mixed precision

    # launch experiment in separate process, as we are enabling mixed precision
    # which will impact other unit tests, unless we do this
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()

    try:
        mp.set_start_method(
            "spawn", force=True
        )  # set start method to 'spawn' BEFORE instantiating the queue and the event
    except RuntimeError:
        pass

    p = mp.Process(target=run_experiment_model)
    try:
        p.start()
    finally:
        p.join()  # necessary so that the Process exists before the test suite exits (thus coverage is collected)

    reset()

    # Optimizer with mixed precision

    # launch experiment in separate process, as we are enabling mixed precision
    # which will impact other unit tests, unless we do this
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()

    try:
        mp.set_start_method(
            "spawn", force=True
        )  # set start method to 'spawn' BEFORE instantiating the queue and the event
    except RuntimeError:
        pass

    p = mp.Process(target=run_experiment_optimizer)
    try:
        p.start()
    finally:
        p.join()  # necessary so that the Process exists before the test suite exits (thus coverage is collected)