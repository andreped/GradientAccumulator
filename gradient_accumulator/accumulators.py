from typing import Optional

import tensorflow as tf

from . import agc
from .utils import get_gradients

# dynamically handle which Optimizer class to use dep on tf version
opt = tf.keras.optimizers.Optimizer
if int(tf.version.VERSION.split(".")[1]) > 10:
    opt = tf.keras.optimizers.legacy.Optimizer


# https://stackoverflow.com/a/66524901
# https://keras.io/guides/customizing_what_happens_in_fit/ # noqa
@tf.keras.utils.register_keras_serializable("gradient-accumulator")
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
        **kwargs,
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
@tf.keras.utils.register_keras_serializable("gradient-accumulator")
class GradientAccumulateOptimizer(opt):
    """Optimizer wrapper for gradient accumulation."""

    def __init__(
        self,
        optimizer: str = "SGD",
        accum_steps: int = 1,
        reduction: str = "MEAN",
        use_agc: bool = False,
        mixed_precision: bool = False,
        name: str = "GradientAccumulateOptimizer",
        dtype: tf.dtypes.DType = tf.float32,
        **kwargs,
    ):
        """
        Construct a new GradientAccumulateOptimizer optimizer.

        Parameters
        ----------
        optimizer : str or tf.keras.optimizers.Optimizer
            Optimizer that will be used to compute and apply gradients.
        accum_steps : int, optional
            Update gradient in every accumulation steps, must be > 0.
        reduction : str, optional
            Gradient reduction method to use. Can be 'MEAN' or 'SUM'.
        use_agc : bool, optional
            Whether to use adaptive gradient clipping. Defaults to False.
        mixed_precision : bool, optional
            Whether to use mixed precision. Defaults to False.
        name : str, optional
            Name for the operations created when applying gradients. Defaults to
            "GradientAccumulateOptimizer".
        **kwargs : dict
            Additional keyword arguments. Allowed keys are:
            - `clip_factor`: Sets upper limit for gradient clipping. Defaults to 0.01.
            - `lr`: Learning rate, included for backward compatibility. Use
            `learning_rate` instead.

        Notes
        -----
        Adding support for sparse tensors was tricky. For correct implementation, both
        `_resource_apply_sparse()`
        and `_resource_apply_sparse_duplicate_indices()` methods need to be implemented.

        References
        ----------
        .. [1] https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/average_wrapper.py#L93 # noqa

        """
        clip_factor = kwargs.pop("clip_factor", 0.01)
        super().__init__(name, **kwargs)
        optimizer = tf.keras.optimizers.get(optimizer)
        self._optimizer = (
            tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            if mixed_precision
            and not isinstance(
                optimizer, tf.keras.mixed_precision.LossScaleOptimizer
            )
            else optimizer
        )
        self.base_optimizer = (
            self._optimizer.inner_optimizer
            if mixed_precision
            else self._optimizer
        )
        self.mixed_precision = mixed_precision
        self._mixed_precision = tf.constant(mixed_precision, dtype=tf.bool)
        self.accum_steps = accum_steps
        self._accum_steps = tf.constant(accum_steps, dtype=tf.int64)
        self.reduction = reduction
        self._reduction = tf.constant(reduction, dtype=tf.string)
        self._step = tf.Variable(
            initial_value=1,
            trainable=False,
            dtype=tf.int64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        if not hasattr(self, "_weights"):
            self._weights = []  # noqa
        if not hasattr(self, "_gradients"):
            self._gradients = []
        self._weights.append(self._step)
        self._zero = tf.constant(0, dtype=tf.int64)
        self.dtype = dtype
        self.use_agc = use_agc
        self._use_agc = tf.constant(use_agc)
        if use_agc:
            self.clip_factor = tf.constant(clip_factor, dtype=tf.float32)
        else:
            self.clip_factor = tf.constant(0.0, dtype=tf.float32)

    def get_slot(self, *args, **kwargs):
        """Returns a slot created by the optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def add_slot(self, var, slot_name, initializer):
        """Adds a new slot to the optimizer."""
        slot = self._optimizer.add_slot(var, slot_name, initializer=initializer)
        self._gradients.append(slot)
        return slot

    def _create_slots(self, var_list: list):
        """Creates slots for the optimizer."""
        # create slots using the base optimizer
        self.base_optimizer._create_slots(var_list=var_list)

        base_optimizer_slots = self.base_optimizer.get_slot_names()

        for var in var_list:
            for slot_name in base_optimizer_slots:
                self.add_slot(
                    var,
                    slot_name,
                    initializer=tf.zeros_like(var),
                )

        # create slots for accumulated gradients
        for var in var_list:
            self.add_slot(var, "ga", initializer=tf.zeros_like(var))

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def step(self) -> tf.Variable:
        """Returns the number of training steps this Optimizer has run."""
        return self._step

    @step.setter
    def step(self, variable: tf.Variable):
        """Sets the step value."""
        self._step = variable
        self._weights.append(self._step)

    @property
    def gradients(self) -> list:
        """Returns the current accumulated gradients on the replica."""
        tf.debugging.assert_greater(
            tf.size(self._gradients),
            tf.cast(self._zero, tf.int32),
            message="Gradients have not been computed yet. "
            "If you're using GradientAccumulateOptimizer with "
            "a custom training loop, please make sure to call "
            "optimizer.apply_gradients() before accessing "
            "optimizer.gradients.",
        )

        return get_gradients(self._gradients)

    def apply_gradients(
        self, grads_and_vars: dict, name: Optional[str] = None, **kwargs
    ) -> tf.Operation:
        """Applies gradients to variables and updates the optimizer's state.

        Parameters
        ----------
        grads_and_vars : dict
            A dictionary of {gradient: variable} pairs.
        name : Optional[str], optional
            The name for the operation. Defaults to None.

        Returns
        -------
        tf.Operation
            The operation after applying the gradients.

        """
        train_op = super().apply_gradients(grads_and_vars, name, **kwargs)
        with tf.control_dependencies([train_op]):
            with tf.control_dependencies(
                [
                    self._optimizer.iterations.assign_add(
                        tf.cast(
                            tf.equal(
                                tf.math.mod(
                                    self.step,
                                    self._accum_steps,
                                ),
                                self._zero,
                            ),
                            tf.int64,
                        ),
                        read_value=False,
                    )
                ]
            ):
                return self.step.assign_add(1, read_value=False)

    @tf.function
    def _apply_agc(self, grad: tf.Tensor, var: tf.Variable) -> tf.Tensor:
        """Applies adaptive gradient clipping to the gradient."""
        return agc.adaptive_clip_grad(
            [var], [grad], clip_factor=self.clip_factor
        )[0]

    @tf.function
    def _parse_grad(
        self, accum_gradient: tf.Tensor, var: tf.Variable
    ) -> tf.Tensor:
        """Parses the accumulated gradient and returns the gradient to be
        applied."""

        apply_condition = tf.fill(
            tf.shape(accum_gradient),
            tf.equal(tf.math.mod(self.step, self._accum_steps), self._zero),
        )

        def apply_agc():
            return self._apply_agc(accum_gradient, var)

        def return_grad():
            return accum_gradient

        return tf.where(
            apply_condition,
            tf.cond(self._use_agc, apply_agc, return_grad),
            tf.zeros_like(var, dtype=accum_gradient.dtype),
        )

    @tf.function
    def reset_accum_gradient(self, accum_gradient: tf.Tensor, grad: tf.Tensor):
        """Resets the accumulated gradient to zero after applying."""
        return tf.where(
            tf.math.equal(grad, accum_gradient),
            tf.zeros_like(accum_gradient),
            accum_gradient,
        )

    def _resource_apply_dense(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        apply_state: Optional[str] = None,
    ) -> tf.Operation:
        """
        Performs gradient update on sparse tensor.

        Parameters
        ----------
        grad : tensor
            Current gradient.
        var : tensor
            Current variable.
        apply_state : str, optional
            State of the optimizer. Defaults to None.

        Returns
        -------
        tensor
            apply_op.

        """
        accum_gradient = self.get_slot(var, "ga")

        # undo loss scaling and revert to higher precision
        grad_to_use = (
            self._optimizer.get_unscaled_gradients([grad])[0]
            if self.mixed_precision
            else grad
        )

        # scale down the gradient and add it to the accumulated gradient
        scaled_grad = tf.math.divide_no_nan(
            grad_to_use, tf.cast(self._accum_steps, dtype=grad_to_use.dtype)
        )

        accum_gradient.assign_add(
            scaled_grad, use_locking=self._use_locking, read_value=False
        )

        def _apply(accum_gradient, var, apply_state):
            grad = self._parse_grad(accum_gradient, var)

            train_op = self.base_optimizer._resource_apply_dense(
                grad,
                var,
                apply_state=apply_state if apply_state else None,
            )

            reset_op = accum_gradient.assign(
                self.reset_accum_gradient(accum_gradient, grad),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def _resource_apply_sparse(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Optional[str] = None,
    ) -> tf.Operation:
        """Performs gradient update on sparse tensor.

        Parameters
        ----------
        grad : tensor
            The current gradient.
        var : tensor
            The current variable.
        indices : tensor
            Relevant indices to be used for masking the sparse tensor during
            update.
        apply_state : str, optional
            State of the optimizer. Defaults to None.

        Returns
        -------
        tensor
            The operation after applying the gradient update.

        """

        accum_gradient = self.get_slot(var, "ga")

        # undo loss scaling and revert to higher precision
        grad_to_use = (
            self._optimizer.get_unscaled_gradients([grad])[0]
            if self.mixed_precision
            else grad
        )

        # scale down the gradient and add it to the accumulated gradient
        scaled_grad = tf.math.divide_no_nan(
            grad_to_use, tf.cast(self._accum_steps, dtype=grad_to_use.dtype)
        )

        self._resource_scatter_add(
            accum_gradient,
            indices,
            scaled_grad,
        )

        def _apply(accum_gradient, var, apply_state):
            grad = self._parse_grad(accum_gradient, var)

            train_op = self.base_optimizer._resource_apply_sparse(
                accum_gradient.sparse_read(indices),
                var,
                indices,
                apply_state=apply_state if apply_state else None,
            )

            reset_op = accum_gradient.assign(
                self.reset_accum_gradient(accum_gradient, grad),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def _resource_apply_sparse_duplicate_indices(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Optional[str] = None,
    ) -> tf.Operation:
        """
        Performs gradient update on sparse tensor with duplicate indices.

        Parameters
        ----------
        grad : tf.Tensor
            Current gradient.
        var : tf.Variable
            Current variable.
        indices : tf.Tensor
            Relevant indices to be used for masking the sparse tensor during
            update.
        apply_state : str, optional
            State of the optimizer. Defaults to None.

        """
        accum_gradient = self.get_slot(var, "ga")

        # undo loss scaling and revert to higher precision
        grad_to_use = (
            self._optimizer.get_unscaled_gradients([grad])[0]
            if self.mixed_precision
            else grad
        )

        # scale down the gradient and add it to the accumulated gradient
        scaled_grad = tf.math.divide_no_nan(
            grad_to_use, tf.cast(self._accum_steps, dtype=grad_to_use.dtype)
        )

        self._resource_scatter_add(
            accum_gradient,
            indices,
            scaled_grad,
        )

        def _apply(accum_gradient, var, apply_state):
            grad = self._parse_grad(accum_gradient, var)

            train_op = (
                self.base_optimizer._resource_apply_sparse_duplicate_indices(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state if apply_state else None,
                )
            )

            reset_op = accum_gradient.assign(
                self.reset_accum_gradient(accum_gradient, grad),
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def _reset_single_gradient(self, gradient: tf.Tensor):
        """Resets the accumulated gradient on the current replica."""
        return gradient.assign(
            tf.zeros_like(gradient),
            use_locking=self._use_locking,
            read_value=False,
        )

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        reset_ops = [
            self._reset_single_gradient(gradient)
            for gradient in self._gradients
            if tf.reduce_all(tf.not_equal(tf.size(gradient), 0))
        ]
        return tf.group(*reset_ops)

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        """The optimizer that this AccumOptimizer is wrapping. In the case of mixed
        precision, this is the LossScaleOptimizer."""
        return self._optimizer

    @property
    def iterations(self) -> tf.Variable:
        """Returns current iteration value of optimizer."""
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable: tf.Variable):
        """Sets the iterations value of optimizer."""
        self._optimizer.iterations = variable

    @property
    def lr(self) -> float:
        """Returns the learning rate of the optimizer."""
        return self.base_optimizer.learning_rate

    @lr.setter
    def lr(self, lr):
        """Sets the learning rate of the optimizer."""
        self.base_optimizer.learning_rate = lr
        self._learning_rate = lr

    @property
    def learning_rate(self):
        return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self.base_optimizer.learning_rate = lr

    @property
    def _learning_rate(self) -> float:
        """Returns the learning rate of the optimizer."""
        return self.lr

    def get_config(self) -> dict:
        """Returns the configuration as a dictionary."""
        config = super().get_config()
        custom_config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "accum_steps": self.accum_steps,
            "reduction": self.reduction,
            "use_agc": self.use_agc,
            "mixed_precision": self.mixed_precision,
            "dtype": self.dtype.name,
        }
        config.update(custom_config)
        return config

    @classmethod
    def from_config(
        cls, config: dict, custom_objects: Optional[str] = None
    ) -> object:
        """Creates an instance of the optimizer from its config."""
        optimizer_config = config.pop("optimizer")
        optimizer = tf.keras.optimizers.deserialize(
            optimizer_config, custom_objects=custom_objects
        )
        return cls(optimizer=optimizer, **config)
