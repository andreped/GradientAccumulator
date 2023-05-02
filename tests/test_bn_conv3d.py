import tensorflow as tf
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator.layers import AccumBatchNormalization
import numpy as np


def test_bn_conv3d(custom_bn:bool = True, accum_steps:int = 1, epochs:int = 1):
    # make toy dataset
    data = np.random.randint(2, size=(16, 8, 8, 8, 1))
    gt = np.expand_dims(np.random.randint(2, size=16), axis=-1)

    # define which normalization layer to use in network
    if custom_bn:
        normalization_layer = AccumBatchNormalization(accum_steps=accum_steps)
    elif not custom_bn:
        normalization_layer = tf.keras.layers.BatchNormalization()
    else:
        normalization_layer = tf.keras.layers.Activation("linear")

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(4, 3, input_shape=(8, 8, 8, 1)),
        normalization_layer,
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4),
        normalization_layer,  # @TODO: BN before or after ReLU? Leads to different performance
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        model = GradientAccumulateModel(accum_steps=accum_steps, inputs=model.input, outputs=model.output)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-2),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["acc"],
    )

    # train model
    model.fit(
        data,
        gt,
        batch_size=1,
        epochs=epochs,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(data, gt, batch_size=1, verbose=1)
    print(result)
    return result
