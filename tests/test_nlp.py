import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Embedding, LSTM, Dense
from gradient_accumulator import GradientAccumulateModel

def test_gradient_accumulator():
    # Load the IMDB reviews dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Prepare the data
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = ds_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Preprocess the data
    def preprocess(x, y):
        x = tf.strings.substr(x, 0, 300)
        x = tf.strings.lower(x)
        x = tf.strings.regex_replace(x, b"<br\\s*/?>", b" ")
        x = tf.strings.regex_replace(x, b"[^a-z]", b" ")
        return x, y

    train_dataset = train_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    # Construct a text vectorization layer
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    # Create the model
    def create_model():
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = encoder(inputs)
        x = Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True)(x)
        x = LSTM(64)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)
        return inputs, outputs

    # Compile and train the original model
    inputs, outputs = create_model()
    original_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    original_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    original_model.fit(train_dataset, epochs=1,
              validation_data=test_dataset,
              validation_steps=30)

    # Compile and train the model with gradient accumulation
    wrapped_model = GradientAccumulateModel(
        inputs=inputs, outputs=outputs, accum_steps=2)
    wrapped_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    wrapped_model.fit(train_dataset, epochs=1,
              validation_data=test_dataset,
              validation_steps=30)

    # Check the output against non-wrapped model
    for x, y in test_dataset.take(1):
        original_output = original_model.predict(x)
        wrapped_output = wrapped_model.predict(x)

        # assert near equality
        assert tf.reduce_all(tf.abs(original_output - wrapped_output) < 1e-6)
