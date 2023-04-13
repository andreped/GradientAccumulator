from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import tensorflow as tf


# https://stackoverflow.com/questions/65195956/keras-custom-batch-normalization-layer-with-an-extra-variable-that-can-be-change
# https://github.com/dksakkos/BatchNorm/blob/main/BatchNorm.py
@tf.keras.utils.register_keras_serializable()
class AccumBatchNormalization(Layer):
    """Custom Batch Normaliztion layer with gradient accumulation support."""
    def __init__(self, accum_steps: int = 1, momentum: float = 0.9, epsilon:float = 1e-3, **kwargs):
        """Construct the AccumBatchNormalization layer.

        Args:
            accum_steps: int > 0. Update gradient in every accumulation steps.
            momentum: float [0, 1]. Momentum used in variable update.
            epsilon: float > 0: Small value to aid numerical stability.
            **kwargs: keyword arguments. Supports various arguments from the Keras' Layer class.
        """
        self.accum_steps = accum_steps
        self.accum_steps_tf = tf.constant(accum_steps, dtype=tf.int32, name="accum_steps")
        self.momentum = momentum
        self.epsilon = epsilon
        self.accum_step_counter = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Builds layer and variables."""
        self.num_features = input_shape[-1]

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

        self.accum_mean = tf.Variable(
            tf.zeros(input_shape[-1], dtype=tf.float32),
            trainable=False,
            name="accum_mean",
        )

        self.accum_variance = tf.Variable(
            tf.zeros(input_shape[-1], dtype=tf.float32),
            trainable=False,
            name="accum_variance",
        )

    def get_moving_average(self, statistic, new_value):
        """Returns the moving average given a statistic and current estimate."""
        decay = tf.convert_to_tensor(1.0 - self.momentum, name="decay")
        if decay.dtype != statistic.dtype.base_dtype:
            decay = tf.cast(decay, statistic.dtype.base_dtype)
        new_value = statistic - (statistic - tf.cast(new_value, statistic.dtype)) * decay
        return statistic.assign(new_value)
    
    def update_variables(self, mean, var):
        """Updates the batch normalization variables."""
        self.moving_mean.assign(self.get_moving_average(self.moving_mean, mean))
        self.moving_variance.assign(self.get_moving_average(self.moving_variance, var))

        self.reset_accum()
    
    def reset_accum(self):
        """Resets accumulator slots."""
        self.accum_mean.assign(tf.zeros(self.num_features))
        self.accum_variance.assign(tf.zeros(self.num_features))

        self.accum_step_counter.assign(0)

    def call(self, inputs, training=None, mask=None):
        """Performs the batch normalization step."""
        if training:
            assert len(inputs.shape) in (2, 4)
            if len(inputs.shape) > 2:
                axes = [0, 1, 2]
            else:
                axes = [0]
            
            # step accum count
            self.accum_step_counter.assign_add(1)
            
            # get batch norm statistics
            mean, var = tf.nn.moments(inputs, axes=axes, keepdims=False)

            # scale mean and variance to produce mean later
            mean /= tf.cast(self.accum_steps_tf, tf.float32)
            var /= tf.cast(self.accum_steps_tf, tf.float32)
            
            # accumulate statistics
            self.accum_mean.assign_add(mean)
            self.accum_variance.assign_add(var)

            # only update variables after n accumulation steps
            tf.cond(tf.equal(self.accum_step_counter, self.accum_steps_tf), true_fn=lambda: self.update_variables(mean, var), false_fn=lambda: None)
        else:
            mean, var = self.moving_mean, self.moving_variance
        
        # @ TODO: Why are we not doing this for training?
        # mean, var = self.moving_mean, self.moving_variance

        scale = self.gamma
        offset = self.beta
        
        inv = tf.math.rsqrt(var + self.epsilon)
        if scale is not None:
            inv *= scale
        
        # @TODO: Why are we using mean and var here and not self.moving_mean and self.moving_variance?
        outputs =  inputs * tf.cast(inv, inputs.dtype) + \
            tf.cast(offset - mean * inv if offset is not None else -mean * inv, inputs.dtype)

        return outputs
    
    def get_config(self):
        """Returns configurations as dict."""
        config = {
            'accum_steps': self.accum_steps,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
