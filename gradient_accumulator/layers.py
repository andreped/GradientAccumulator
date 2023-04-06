from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf


# https://stackoverflow.com/questions/65195956/keras-custom-batch-normalization-layer-with-an-extra-variable-that-can-be-change
# https://github.com/dksakkos/BatchNorm/blob/main/BatchNorm.py
class AccumBatchNormalization(BatchNormalization):
    def __init__(self, **kwargs):
        super(AccumBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )

        self.gamma = self.add_weight(
            shape=(input_shape[-1]),
            initializer="ones",
            trainable=True,
        )

        self.moving_mean = self.add_weight(
            shape=(input_shape[-1]),
            initializer=tf.initializers.zeros,
            trainable=False)

        self.moving_variance = self.add_weight(
            shape=(input_shape[-1]),
            initializer=tf.initializers.ones,
            trainable=False)

    def get_moving_average(self, statistic, new_value):
        momentum = 0.9
        new_value = statistic * momentum + new_value * (1 - momentum)
        return statistic.assign(new_value)

    def normalise(self, x, x_mean, x_var):
        return (x - x_mean) / tf.sqrt(x_var + 1e-6)

    def call(self, inputs, training):
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
        x = self.normalise(inputs, mean, var)
        return self.gamma * x + self.beta
