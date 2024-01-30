import tensorflow as tf
from tensorflow.keras.layers import Dense

from gradient_accumulator import GradientAccumulateModel


def create_model():
    input = tf.keras.layers.Input(shape=(10,))
    x = Dense(32, input_shape=(10,), activation="relu")(input)
    x = Dense(16, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    return input, output


def count_params(model):
    return model.count_params()


def test_param_count_with_wrapper():
    # Create a model
    input, output = create_model()
    original_model = tf.keras.Model(inputs=input, outputs=output)

    # Count the parameters of the original model
    original_param_count = count_params(original_model)

    # Wrap the model with GradientAccumulateModel
    wrapped_model = GradientAccumulateModel(
        accum_steps=2, inputs=input, outputs=output
    )

    # Count the parameters of the wrapped model
    wrapped_param_count = count_params(wrapped_model)

    # Compile both models
    original_model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy"
    )
    wrapped_model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy"
    )

    # Check if the number of parameters in both models is the same
    assert original_param_count == wrapped_param_count, (
        f"Parameter count mismatch: Original model has {original_param_count} parameters, "
        f"wrapped model has {wrapped_param_count} parameters."
    )
