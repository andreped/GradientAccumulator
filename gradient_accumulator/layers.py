from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import tensorflow as tf


# https://stackoverflow.com/questions/65195956/keras-custom-batch-normalization-layer-with-an-extra-variable-that-can-be-change
# https://github.com/dksakkos/BatchNorm/blob/main/BatchNorm.py
@tf.keras.utils.register_keras_serializable()
class AccumBatchNormalization(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-3, **kwargs):
        self.momentum = momentum
        self.epsilon = epsilon

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1]),
            initializer="zeros",
            trainable=True,
            name="beta",
        )

        self.gamma = self.add_weight(
            shape=(input_shape[-1]),
            initializer="ones",
            trainable=True,
            name="gamma",
        )

        self.moving_mean = self.add_weight(
            shape=(input_shape[-1]),
            initializer="zeros",
            trainable=False,
            name="moving_mean",
        )

        self.moving_variance = self.add_weight(
            shape=(input_shape[-1]),
            initializer="ones",
            trainable=False,
            name="moving_variance",
        )

    def get_moving_average(self, statistic, new_value):
        #new_value = statistic * self.momentum + new_value * (1.0 - self.momentum)
        decay = tf.convert_to_tensor(1.0 - self.momentum, name="decay")
        if decay.dtype != statistic.dtype.base_dtype:
            decay = tf.cast(decay, statistic.dtype.base_dtype)
        new_value = statistic - (statistic - tf.cast(new_value, statistic.dtype)) * decay
        return statistic.assign(new_value)

    def call(self, inputs, training=None, mask=None):
        if training:
            assert len(inputs.shape) in (2, 4)
            if len(inputs.shape) > 2:
                axes = [0, 1, 2]
            else:
                axes = [0]
                
            mean, var = tf.nn.moments(inputs, axes=axes, keepdims=False)
            self.moving_mean.assign(self.get_moving_average(self.moving_mean, mean))
            self.moving_variance.assign(self.get_moving_average(self.moving_variance, var))
        else:
            mean, var = self.moving_mean, self.moving_variance
        
        scale = self.gamma
        offset = self.beta

        inv = tf.math.rsqrt(var + self.epsilon)
        if scale is not None:
            inv *= scale
        outputs =  inputs * tf.cast(inv, inputs.dtype) + \
            tf.cast(offset - mean * inv if offset is not None else -mean * inv, inputs.dtype)
        
        return outputs
    
    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
