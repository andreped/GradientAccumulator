import os
import random as python_random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator.layers import AccumBatchNormalization
from gradient_accumulator.utils import replace_batchnorm_layers

from .utils import gray2rgb
from .utils import normalize_img
from .utils import reset
from .utils import resizeImage


# def test_swap_layer(
#     custom_bn: bool = True, bs: int = 100, accum_steps: int = 1, epochs: int = 1
# ):
#     # load dataset
#     (ds_train, ds_test), ds_info = tfds.load(
#         "mnist",
#         split=["train", "test"],
#         shuffle_files=True,
#         as_supervised=True,
#         with_info=True,
#     )

#     # build train pipeline
#     ds_train = ds_train.map(normalize_img)
#     ds_train = ds_train.map(gray2rgb)
#     ds_train = ds_train.map(resizeImage)
#     ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
#     ds_train = ds_train.batch(bs)
#     ds_train = ds_train.prefetch(1)

#     # build test pipeline
#     ds_test = ds_test.map(normalize_img)
#     ds_test = ds_test.map(gray2rgb)
#     ds_test = ds_test.map(resizeImage)
#     ds_test = ds_test.batch(bs)
#     ds_test = ds_test.prefetch(1)

#     # create model
#     base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), weights="imagenet", include_top=False)
#     base_model = replace_batchnorm_layers(base_model, accum_steps=accum_steps)

#     input_ = tf.keras.layers.Input(shape=(32, 32, 3))
#     x = base_model(input_)
#     x = tf.keras.layers.Dense(10, activation="softmax")(x)
#     model = tf.keras.Model(inputs=input_, outputs=x)

#     # wrap model to use gradient accumulation
#     if accum_steps > 1:
#         model = GradientAccumulateModel(
#             accum_steps=accum_steps, inputs=model.input, outputs=model.output
#         )

#     # compile model
#     model.compile(
#         optimizer=tf.keras.optimizers.SGD(1e-2),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     )

#     # train model
#     model.fit(
#         ds_train,
#         epochs=epochs,
#         validation_data=ds_test,
#         steps_per_epoch=4,
#         validation_steps=4,
#     )

#     model.save("./trained_model")

#     # load trained model and test
#     del model
#     trained_model = load_model("./trained_model", compile=True)

#     result = trained_model.evaluate(ds_test, verbose=1)
#     print(result)
#     return result
