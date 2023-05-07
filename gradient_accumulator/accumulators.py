import tensorflow as tf

from . import agc

# dynamically handle which Optimizer class to use dep on tf version
opt = tf.keras.optimizers.Optimizer
if int(tf.version.VERSION.split(".")[1]) > 10:
    opt = tf.keras.optimizers.legacy.Optimizer


# https://stackoverflow.com/a/66524901
# https://keras.io/guides/customizing_what_happens_in_fit/
@tf.keras.utils.register_keras_serializable()
class GradientAccumulateModel(tf.keras.Model):
    """Model wrapper for gradient accumulation."""

    def __init__(
        self,
        accum_steps: int = 1,
        mixed_precision: bool = False,
        use_agc: bool = False,
        clip_factor: float = 0.01,
        eps: float = 1e-3,
        experimental_distributed_support: bool = False,
        *args,
        **kwargs
    ):
        """Adds gradient accumulation support to existing Keras Model.

        Args:
            accum_steps: int > 0. Update gradient in every accumulation steps.
            mixed_precision: bool. Whether to enable mixed precision.
            use_agc: bool. Whether to enable adaptive gradient clipping.
            clip_factor: float > 0. Upper limit to gradient clipping.
            eps: float > 0. Small value to aid numerical stability.
            experimental_distributed_support: bool. Whether to enable
                experimental multi-gpu support. Only compatible with SGD. Can
                be used with other optimizers but we do not have complete
                control of the optimizer's state between accum_steps.
            **kwargs: keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.accum_steps = tf.constant(
            accum_steps, dtype=tf.int32, name="accum_steps"
        )
        self.accum_step_counter = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.first_call = True
        self.mixed_precision = mixed_precision
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.eps = eps
        self.experimental_distributed_support = experimental_distributed_support
        self.dtype_value = self.dtype
        self.gradient_accumulation = None
        self.reinit_grad_accum()

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
            loss = loss / tf.cast(
                self.accum_steps, loss.dtype
            )  # MEAN reduction here IMPORTANT! Don't use SUM!

            # scale loss if mixed precision is enabled
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Calculate batch gradients -> these are scaled gradients if mixed
        # precision is enabled
        gradients = tape.gradient(
            loss,
            self.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # scale gradients if mixed precision is enabled
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # apply adaptive gradient clipping -> should be AFTER unscaling
        if self.use_agc:
            gradients = agc.adaptive_clip_grad(
                self.trainable_variables,
                gradients,
                clip_factor=self.clip_factor,
                eps=self.eps,
            )

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(
                gradients[i], read_value=False
            )

        # accumulate gradients only after certain number of steps
        # self.accum_steps.assign(self.accum_steps * tf.cast(tf.logical_not(\
        #   tf.equal(self.accum_step_counter,self.accum_steps)), tf.int32))
        if not self.experimental_distributed_support:
            tf.cond(
                tf.equal(self.accum_step_counter, self.accum_steps),
                true_fn=self.apply_accu_gradients,
                false_fn=lambda: None,
            )

        else:
            # NOTE: This enabled multi-gpu support, but only for SGD (!)
            should_apply = tf.equal(self.accum_step_counter, self.accum_steps)
            logical_grads = [
                tf.cast(should_apply, grad_component.dtype) * grad_component
                for grad_component in self.gradient_accumulation
            ]
            self.optimizer.apply_gradients(
                zip(logical_grads, self.trainable_variables)
            )
            self.accum_step_counter.assign(
                self.accum_step_counter
                * tf.cast(tf.logical_not(should_apply), tf.int32)
            )
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(-1 * logical_grads[i])

        # update metrics
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight
        )
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        """Performs gradient update and resets slots afterwards."""
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(
                    self.trainable_variables[i], dtype=self.dtype_value
                ),
                read_value=False,
            )

    def reinit_grad_accum(self):
        """Reinitialized gradient accumulator slots."""
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(
                tf.zeros_like(v, dtype=self.dtype_value),
                trainable=False,
                name="accum_" + str(i),
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            for i, v in enumerate(self.trainable_variables)
        ]


# Implementation was derived from:
# https://github.com/fsx950223/addons/blob/67c1e8ea19e82c3f2a5706674dd81f15ab5002a2/tensorflow_addons/optimizers/gradient_accumulator.py  # noqa
# https://github.com/FreddeFrallan/Multilingual-CLIP/blob/5c82118452b3b59b41bb53714d61cd4990b1588d/multilingual_clip/TeacherLearning/Utils.py#L84  # noqa
@tf.keras.utils.register_keras_serializable()
class GradientAccumulateOptimizer(opt):
    """Optimizer wrapper for gradient accumulation."""

    def __init__(
        self,
        optimizer="SGD",
        accum_steps=1,
        reduction: str = "MEAN",
        name: str = "GradientAccumulateOptimizer",
        **kwargs
    ):
        """Construct a new GradientAccumulateOptimizer optimizer.

        Adding support for sparse tensors was tricky, but this resource was
        helpful. Note that you need to implement both _resource_apply_sparse()
        and _resource_apply_sparse_duplicate_indices() for it to work as
        intended.

        See here for more information regarding implementation:
        * https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/average_wrapper.py#L93  # noqa

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            reduction: str. Which gradient reduction method to use. Defaults
                to 'SUM'.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulateOptimizer".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        self._optimizer = tf.keras.optimizers.get(optimizer)
        self._accum_steps = accum_steps
        self._reduction = reduction
        self._step = None
        super().__init__(name, **kwargs)

    def _create_slots(self, var_list):
        """Creates slots for optimizer gradients.

        Args:
            List of trainable variables.
        """
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def step(self):  # pragma: no cover
        """The number of training steps this Optimizer has run.
        Initializes step variable if None.

        Returns:
            Current number of optimizer steps.
        """
        if self._step is None:
            with self._distribution_strategy_scope():
                self._step = self.add_weight(
                    "iter",
                    shape=[],
                    initializer="ones",
                    dtype=tf.int64,
                    trainable=False,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            self._weights.append(self._step)
        return self._step

    @step.setter
    def step(self, variable):
        """Sets the step value."""
        if self._step is not None:
            raise RuntimeError(
                "Cannot set `step` to a new Variable after "
                "the Optimizer weights have been created"
            )
        self._step = variable
        self._weights.append(self._step)

    @property
    def gradients(self):  # pragma: no cover
        """The accumulated gradients on the current replica.

        Returns:
            Current gradients in optimizer.
        """
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the"
                "gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Updates weights using gradients.

        Args:
            grads_and_vars: dict containing variables and corresponding
                gradients.
            name: name to set when applying gradients.
            **kwargs: keyword arguments.
        Return:
            Updated weights.
        """
        train_op = super().apply_gradients(grads_and_vars, name, **kwargs)
        with tf.control_dependencies([train_op]):
            with tf.control_dependencies(
                [
                    self._optimizer.iterations.assign_add(
                        tf.cast(
                            tf.where(self.step % self._accum_steps == 0, 1, 0),
                            tf.int64,
                        ),
                        read_value=False,
                    )
                ]
            ):
                return self.step.assign_add(1, read_value=False)

    def _resource_apply_dense(
        self, grad, var, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on dense tensor.

        Args:
            grad: current gradient.
            var: current variable.
            apply_state: whether to apply X.
        Returns:
            apply_op.
        """
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad / self._accum_steps,
                use_locking=self._use_locking,
                read_value=False,
            )

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )

            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    grad, var, apply_state=apply_state
                )
            else:
                train_op = self.optimizer._resource_apply_dense(grad, var)

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def _resource_apply_sparse(
        self, grad, var, indices, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on sparse tensor.

        Args:
            grad: current gradient.
            var: current variable.
            indices: relevant indices to be used for masking the sparse tensor
                during update.
        Returns:
            apply_op.
        """

        accum_gradient = self.get_slot(var, "ga")

        if accum_gradient is not None and grad is not None:
            grad /= tf.cast(self._accum_steps, dtype=grad.dtype)
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )
            if "apply_state" in self.optimizer._sparse_apply_args:
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

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    # TODO: needs to be updated and tested
    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on sparse tensor.

        Args:
            grad: current gradient.
            var: current variable.
            indices: relevant indices to be used for masking the sparse tensor
                during update.
        Returns:
            apply_op.
        """

        accum_gradient = self.get_slot(var, "ga")

        if accum_gradient is not None and grad is not None:
            grad /= tf.cast(self._accum_steps, dtype=grad.dtype)
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )
            if "apply_state" in self.optimizer._sparse_apply_args:
                train_op = (
                    self.optimizer._resource_apply_sparse_duplicate_indices(
                        accum_gradient.sparse_read(indices),
                        var,
                        indices,
                        apply_state=apply_state,
                    )
                )
            else:
                train_op = (
                    self.optimizer._resource_apply_sparse_duplicate_indices(
                        accum_gradient.sparse_read(indices), var, indices
                    )
                )

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def reset(self):  # pragma: no cover
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
    def optimizer(self):
        """The optimizer that this AccumOptimizer is wrapping."""
        return self._optimizer

    @property
    def iterations(self):
        """Returns current iteration value of optimizer.

        Returns:
            iterations of optimizer."""
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        """Sets the iterations value of optimizer."""
        self._optimizer.iterations = variable

    @property
    def learning_rate(self):  # pragma: no cover
        """Returns the learning rate of the optimizer.

        Returns:
            learning rate of optimizer.
        """
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):  # pragma: no cover
        """Sets the learning rate of the optimizer.

        Args:
            learning_rate: which learning rate to set in the optimizer.
        """
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        """Returns the configuration as dict."""
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "accum_steps": self._accum_steps,
            "reduction": self._reduction,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Gets config of original optimizer and deserializes it."""
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)
