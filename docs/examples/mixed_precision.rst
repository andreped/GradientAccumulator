Mixed Precision
---------------

There has also been added experimental support for mixed precision:
``
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam

mixed_precision.set_global_policy('mixed_float16')
model = GradientAccumulateModel(accum_steps=4, mixed_precision=True, inputs=model.input, outputs=model.output)

opt = Adam(1e-3, epsilon=1e-4)
opt = mixed_precision.LossScaleOptimizer(opt)
``

If using TPUs, use `bfloat16` instead of `float16`, like so:
``
mixed_precision.set_global_policy('mixed_bfloat16')
``

There is also an example of how to use gradient accumulation with
mixed precision `here <https://github.com/andreped/GradientAccumulator/blob/main/tests/test_mixed_precision.py#L58>`_.
