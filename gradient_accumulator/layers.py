from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import tensorflow as tf


# https://stackoverflow.com/questions/65195956/keras-custom-batch-normalization-layer-with-an-extra-variable-that-can-be-change
# https://github.com/dksakkos/BatchNorm/blob/main/BatchNorm.py
@tf.keras.utils.register_keras_serializable()
class AccumBatchNormalization(Layer):
    """Custom Batch Normaliztion layer with gradient accumulation support."""
    def __init__(self, accum_steps: int = 1, momentum: float = 0.99, epsilon:float = 1e-3, trainable:bool = True, **kwargs):
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
        self.trainable = trainable
        self.accum_step_counter = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Builds layer and variables.
        
        Args:
            input_shape: input feature map size.
        """
        self.param_shape = input_shape[-1]

        self.beta = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="zeros",
            trainable=True,
            name="beta",
        )

        self.gamma = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="ones",
            trainable=True,
            name="gamma",
        )

        self.moving_mean = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="zeros",
            trainable=False,
            name="moving_mean",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
        )

        self.moving_variance = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="ones",
            trainable=False,
            name="moving_variance",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
        )

        self.accum_mean = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="zeros",
            trainable=False,
            name="accum_mean",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
        )

        self.accum_variance = self.add_weight(
            shape=(self.param_shape),
            dtype=self._param_dtype,
            initializer="zeros",  # NOTE: This should be "zeros" ass we use it for accumulation
            trainable=False,
            name="accum_variance",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
        )

    def get_moving_average(self, statistic, new_value):
        """Returns the moving average given a statistic and current estimate.
        
        Args:
            statistic: summary statistic e.g. average across for single feature over multiple samples
            new_value: statistic of single feature for single forward step.
        Returns:
            Updated statistic.
        """
        decay = tf.convert_to_tensor(1.0 - self.momentum, name="decay")
        if decay.dtype != statistic.dtype.base_dtype:
            decay = tf.cast(decay, statistic.dtype.base_dtype)
        delta = (statistic - tf.cast(new_value, statistic.dtype)) * decay
        return statistic.assign_sub(delta)  # statistic - delta
    
    def update_variables(self, mean, var):
        """Updates the batch normalization variables.
        
        Args:
            mean: average for single feature
            var: variance for single feature
        """
        self.moving_mean.assign(self.get_moving_average(self.moving_mean, mean))
        self.moving_variance.assign(self.get_moving_average(self.moving_variance, var))

        self.reset_accum()
    
    def reset_accum(self):
        """Resets accumulator slots."""
        self.accum_mean.assign(tf.zeros_like(self.accum_mean))
        self.accum_variance.assign(tf.zeros_like(self.accum_variance))

        self.accum_step_counter.assign(0)

    def call(self, inputs, training=None, mask=None):
        """Performs the batch normalization step.
        
        Args:
            inputs: input feature map to apply batch normalization across.
            training: whether layer should be in training mode or not.
            mask: whether to calculate statistics within masked region of feature map.
        Returns:
            Normalized feature map.
        """
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
            mean_scaled = mean / tf.cast(self.accum_steps_tf, self._param_dtype)
            var_scaled = var / tf.cast(self.accum_steps_tf, self._param_dtype)
            
            # accumulate statistics
            self.accum_mean.assign_add(mean_scaled)
            self.accum_variance.assign_add(var_scaled)

            # only update variables after n accumulation steps
            tf.cond(
                tf.equal(self.accum_step_counter, self.accum_steps_tf),
                true_fn=lambda: self.update_variables(self.accum_mean, self.accum_variance),
                false_fn=lambda: None
            )
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
    
    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
    
    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32
    
    def get_config(self):
        """Returns configurations as dict."""
        config = {
            'accum_steps': self.accum_steps,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
