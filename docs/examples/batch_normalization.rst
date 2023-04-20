Batch Normalization
===================

As Keras' Batch Normalization layer is not directly compatible
with gradient accumulation, we have implemented a custom BN
layer called `AccumBatchNormalization`, which supports it.

The layer can be used as a drop-in replacement of Keras'
BatchNormalization layer. Just note that it has reduced
functionality, not including techniques such as batch
renormalization or ghost batches. However, in general
the *vanilla* batch normalization layer is the most used.


.. code-block:: python

    from gradient_accumulator import GradientAccumulateModel, AccumBatchNormalization

    # sets it here as we will set it for both the layer and model wrapper
    accum_steps = 4

    # simple mnist classifier model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32),
        AccumBatchNormalization(accum_steps=accum_steps),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10)
    ])

    # needs this as well to update the remaining variables
    model = GradientAccumulateModel(accum_steps=accum_steps, inputs=model.input, outputs=model.output)


Note that Batch Normalization is a unique layer in Keras.
It has two sets of variables. The first two `mean` and 
`variance` are updated during the *forward pass*, whereas
the remaining two `gamma` and `beta` are updated during the
backward pass.

Hence, it is crucial to include the model wrapper when
also using the `AccumBatchNormalization` with 
`accum_steps > 1`.