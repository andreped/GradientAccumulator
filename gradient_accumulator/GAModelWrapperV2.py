import tensorflow as tf
from . import agc
from .utils import GradientAccumulator


# https://stackoverflow.com/a/66524901
# https://keras.io/guides/customizing_what_happens_in_fit/
@tf.keras.utils.register_keras_serializable()  # adding this avoids needing to use custom_objects when loading model
class GAModelWrapperV2(tf.keras.Model):
    def __init__(self, accum_steps=1, mixed_precision=False, use_agc=False, clip_factor=0.01, eps=1e-3, strategy=None, model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="accum_counter",
                                              synchronization=tf.VariableSynchronization.ON_READ,
                                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                              )
        self.accum_steps = accum_steps
        self.first_call = True
        self.gradient_accumulation = None
        self.mixed_precision = mixed_precision
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.eps = eps
        self._strategy = strategy
        self._model = model

        self._generator_gradient_accumulator = GradientAccumulator()
        self._generator_gradient_accumulator.reset()
    
    #@tf.function
    def distributed_step(self, data):
        #assert tf.distribute.get_replica_context() is None
        
        # RuntimeError: Method requires being in cross-replica context, use get_replica_context().merge_call()
        #per_replica_gradients = self._strategy.run(self.single_step_per_replica, args=(data,))

        #gradients = self._strategy.reduce(
        #    tf.distribute.ReduceOp.SUM, per_replica_gradients, axis=None
        #)

        # this works, but I believe we want to do self._strategy.run(), or?
        per_replica_gradients = self.single_step_per_replica(data)

        # update accumulator
        self._generator_gradient_accumulator(per_replica_gradients)
    
    #@tf.function
    def single_step_per_replica(self, data):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass
            #y_pred = self._model(x, training=True)
            #y_pred = self._strategy.run()

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
            loss = loss / tf.cast(self.accum_steps, tf.float32)  # MEAN reduction here IMPORTANT! Don't use SUM!

            # scale loss if mixed precision is enabled
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
        
        # Calculate batch gradients -> these are scaled gradients if mixed precision is enabled
        gradients = tape.gradient(loss, self.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # scale gradients if mixed precision is enabled
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # apply adaptive gradient clipping -> should be AFTER unscaling gradients
        if self.use_agc:
            gradients = agc.adaptive_clip_grad(
                self.trainable_variables, gradients, clip_factor=self.clip_factor, eps=self.eps)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # return gradient
        return gradients

    def train_step(self, data):
        # need to reinit accumulator for models subclassed from tf.keras.Model
        if self.first_call:
            self._generator_gradient_accumulator.reset()

        # calculate gradients across multiple GPUs
        gradients = self.distributed_step(data)

        # update weights based on accumulated gradients and reset
        self.optimizer.apply_gradients(
            zip(self._generator_gradient_accumulator.gradients, self.trainable_variables)
        )
        self._generator_gradient_accumulator.reset()

        # return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def reinit_grad_accum(self):
        # reinitialize gradient accumulator
        self._generator_gradient_accumulator.reset()
