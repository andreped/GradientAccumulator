import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential, load_model

from gradient_accumulator import GradientAccumulateOptimizer

from tests.utils import reset

# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def preprocess_data(ds, vocab_size, max_length):
    def encode(x, y):
        x = tf.strings.substr(x, 0, max_length)
        x = tf.strings.reduce_join(
            tf.strings.unicode_split(x, input_encoding="UTF-8"), separator=" "
        )
        x = tf.strings.split(x)
        x_hashed = tf.strings.to_hash_bucket_fast(x, vocab_size)
        x_padded = tf.pad(
            x_hashed,
            paddings=[[0, max_length - tf.shape(x_hashed)[-1]]],
            constant_values=0,
        )
        return x_padded, y

    ds = ds.map(encode)
    return ds


def run_experiment(bs=100, accum_steps=1, epochs=2):
    # Load the IMDb dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "imdb_reviews",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Preprocess the dataset
    vocab_size = 10000
    max_length = 100
    ds_train = preprocess_data(ds_train, vocab_size, max_length)
    ds_test = preprocess_data(ds_test, vocab_size, max_length)

    # Batch the dataset
    ds_train = ds_train.batch(bs)
    ds_test = ds_test.batch(bs)

    # define model
    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length)
    )
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    # wrap optimizer to add gradient accumulation support
    # need to dynamically handle which Optimizer class to use dependent on tf version
    if tf_version > 10:
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
    else:
        opt = tf.keras.optimizers.SGD(
            learning_rate=1e-2
        )  # IDENTICAL RESULTS WITH SGD!!!

    if accum_steps > 1:
        opt = GradientAccumulateOptimizer(
            optimizer=opt, accum_steps=accum_steps, reduction="MEAN"
        )

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])

    model.fit(
        ds_train,
        batch_size=bs,
        epochs=epochs,
        verbose=1,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)

    return result[1]


def test_sparse_expected_results():
    # set seed
    # reset()

    # run once
    # result1 = run_experiment(bs=100, accum_steps=1, epochs=2)

    # reset before second run to get identical results
    reset()

    # run again with different batch size and number of accumulations
    result2 = run_experiment(bs=50, accum_steps=2, epochs=2)

    # results should be identical (theoretically, even in practice on CPU)
    # assert result1 == result2
