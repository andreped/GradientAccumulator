TensorFlow >= 2.11 legacy
-------------------------

Note that for `tensorflow>=2.11`, there has been some major changes
to the `Optimizer` class. Our current implementation is not compatible
with the new one. Based on which TensorFlow version you have, our
`GradientAccumulateOptimizer` dynamically chooses which Optimizer to use.

However, you will need to choose a legacy optimizer to use with the
Optimizer wrapper, like so:


.. code-block:: python

    import tensorflow as tf
    from gradient_accumulator import GradientAccumulateOptimizer

    opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
    opt = GradientAccumulateOptimizer(optimizer=opt, accum_steps=4)

