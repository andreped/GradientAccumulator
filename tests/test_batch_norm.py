import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator.layers import AccumBatchNormalization


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def reset():
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

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_experiment(custom_bn=True, accum=True):
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
        tf.keras.layers.Dense(128),
        AccumBatchNormalization() if custom_bn else tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10)
    ])

    # wrap model to use gradient accumulation
    if accum:
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
    return result


def test_compare_bn_layers():
    # custom BN without accum
    result1 = run_experiment(custom_bn=True, accum=False)

    # keras BN without accum
    result2 = run_experiment(custom_bn=False, accum=False)

    assert result1 == result2


def test_custom_bn_accum_compatibility():
    run_experiment(custom_bn=True, accum=True)


if __name__ == "__main__":
    test_compare_bn_layers()
    test_custom_bn_accum_compatibility()
