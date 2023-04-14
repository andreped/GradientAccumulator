import tensorflow as tf
from . import agc


# need to dynamically handle which Optimizer class to use dependent on tf version
opt = tf.keras.optimizers.Optimizer
if int(tf.version.VERSION.split(".")[1]) > 10:
    opt = tf.keras.optimizers.legacy.Optimizer


# https://stackoverflow.com/a/66524901
# https://keras.io/guides/customizing_what_happens_in_fit/
@tf.keras.utils.register_keras_serializable()  # adding this avoids needing to use custom_objects when loading model
class GradientAccumulateModel(tf.keras.Model):
    """Model wrapper for gradient accumulation."""
    def __init__(self, accum_steps:int = 1, mixed_precision:bool = False, use_agc:bool = False,
                 clip_factor:float = 0.01, eps:float = 1e-3, *args, **kwargs):
        """Adds gradient accumulation support to existing Keras' Model.

        Args:
            accum_steps: int > 0. Update gradient in every accumulation steps.
            mixed_precision: bool. Whether to enable mixed precision.
            use_agc: bool. Whether to enable adaptive gradient clipping.
            clip_factor: float > 0. Upper limit to gradient clipping.
            eps: float > 0. Small value to aid numerical stability.
            **kwargs: keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32, name="accum_steps")
        self.accum_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="accum_counter",
                                              synchronization=tf.VariableSynchronization.ON_READ,
                                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                              )
        self.first_call = True
        self.gradient_accumulation = None
        self.reinit_grad_accum()
        self.mixed_precision = mixed_precision
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.eps = eps

    def train_step(self, data):
        """Performs single train step."""
        # need to reinit accumulator for models subclassed from tf.keras.Model
        if self.first_call:
            self.reinit_grad_accum()
            self.first_call = False

        self.accum_step_counter.assign_add(1)

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # NOTE that x and y are lists of inputs and outputs,
        # hence this wrapper supports multi-input-output models
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass

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

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i], read_value=False)

        # If accum_step_counter reach the accum_steps then we apply accumulated gradients to update the variables
        # otherwise do nothing
        tf.cond(tf.equal(self.accum_step_counter, self.accum_steps), true_fn=self.apply_accu_gradients,
                false_fn=lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        """Performs gradient update and resets slots afterwards."""
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i], dtype=tf.float32), read_value=False)
    
    def reinit_grad_accum(self):
        """Reinitialized gradient accumulator slots."""
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False,
            name="accum_" + str(i),
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            ) for i, v in enumerate(self.trainable_variables)
        ]


# Implementation was derived from:
# https://github.com/fsx950223/addons/blob/67c1e8ea19e82c3f2a5706674dd81f15ab5002a2/tensorflow_addons/optimizers/gradient_accumulator.py
@tf.keras.utils.register_keras_serializable()
class GradientAccumulateOptimizer(opt):
    """Optimizer wrapper for gradient accumulation."""
    def __init__(self, optimizer="SGD", accum_steps=1, reduction: str = "MEAN", name: str = "GradientAccumulateOptimizer", **kwargs):
        """Construct a new GradientAccumulateOptimizer optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            reduction: str. Which gradient reduction method to use. Defaults to 'SUM'.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulateOptimizer".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        self.optimizer = tf.keras.optimizers.get(optimizer)
        # self._optimizer = self.optimizer
        self.accum_steps = accum_steps
        self.reduction = reduction
        super().__init__(name, **kwargs)

    def _create_slots(self, var_list):
        """Creates slots for optimizer gradients."""
        self.optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Updated gradients in optimizer."""
        self.optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Performs gradient update on dense tensor."""
        accum_gradient = self.get_slot(var, "ga")

        if accum_gradient is not None and grad is not None:
            if self.reduction == "MEAN":
                grad /= tf.cast(self.accum_steps, grad.dtype)

            accum_gradient.assign_add(
                grad, use_locking=self._use_locking, read_value=False
            )

        def _apply():
            if "apply_state" in self.optimizer._dense_apply_args:
                train_op = self.optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var, apply_state=apply_state
                )
            else:
                train_op = self.optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var
                )

            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self.accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    # Example implementation about this method can be seen here:
    # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/average_wrapper.py#L93
    #@tf.function
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """Performs gradient update on sparse tensor."""
        
        accum_gradient = self.get_slot(var, "ga")
        
        if accum_gradient is not None and grad is not None:
            if self.reduction == "MEAN":
                grad /= tf.cast(self.accum_steps, grad.dtype)

            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self.optimizer._sparse_apply_args:
                # @TODO: Results in KeyError for Embedding layer
                train_op = self.optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self.optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices), var, indices
                )

            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self.accum_steps == 0, _apply, lambda: tf.no_op()  # tf.no_op: Does nothing - placeholder
        )
        return apply_op

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, apply_state=None):
        """Performs gradient update on sparse tensor."""
        
        accum_gradient = self.get_slot(var, "ga")
        
        if accum_gradient is not None and grad is not None:
            if self.reduction == "MEAN":
                grad /= tf.cast(self.accum_steps, grad.dtype)

            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self.optimizer._sparse_apply_args:
                train_op = self.optimizer._resource_apply_sparse_duplicate_indices(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self.optimizer._resource_apply_sparse_duplicate_indices(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                )

            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations + 1) % self.accum_steps == 0, _apply, lambda: tf.no_op()  # tf.no_op: Does nothing - placeholder
        )
        return apply_op

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def learning_rate(self):
        """Returns the learning rate of the optimizer."""
        return self.optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """Sets the learning rate of the optimizer."""
        self.optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        """Returns the configuration as dict."""
        config = {
            "optimizer": tf.keras.optimizers.get(self.optimizer),
            "accum_steps": self.accum_steps,
            "reduction": self.reduction,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
