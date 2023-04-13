Distributed training
--------------------

In order to train with multiple GPUs, you will have to use the Optimizer wrapper:
``
opt = GradientAccumulateOptimizer(accum_steps=4, optimizer=tf.keras.optimizers.SGD(1e-2))
``

Just remember to wrap the optimizer within the `tf.distribute.MirroredStrategy`. For an 
example, see `here <https://github.com/andreped/GradientAccumulator/blob/main/tests/test_optimizer_distribute.py>`_.

DISCLAIMER: The GradientAccumulateOptimizer is a VERY experimental feature. It is not
reaching the same results as GradientAccumulateModel with a single GPU, and does not work
(yet) with multiple GPUs. Hence, I would recommend using GradientAccumulateModel with a
single GPU in its current state.**
