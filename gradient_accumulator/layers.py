from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf


# https://stackoverflow.com/questions/65195956/keras-custom-batch-normalization-layer-with-an-extra-variable-that-can-be-change
class AccumBatchNormalization(BatchNormalization):
    def __init__(self, **kwargs):
        super(AccumBatchNormalization, self).__init__(**kwargs)
        self.skip = False

    def call(self, inputs, training=None):
        if self.skip:
            tf.print("I'm skipping")
        else:
            tf.print("I'm not skipping")
        # raise NotImplementedError
        return super(AccumBatchNormalization, self).call(inputs, training)

    def build(self, input_shape):
        super(AccumBatchNormalization, self).build(input_shape)
