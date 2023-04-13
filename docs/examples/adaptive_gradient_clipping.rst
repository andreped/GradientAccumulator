Gradient Clipping
==========================

There has also been added support for adaptive gradient clipping, based on `this <https://github.com/sayakpaul/Adaptive-Gradient-Clipping>`_ implementation:

.. code-block:: python

    model = GradientAccumulateModel(
        accum_steps=4, use_agc=True, clip_factor=0.01, eps=1e-3, inputs=model.input, outputs=model.output
    )


The hyperparameters values for `clip_factor` and `eps` presented here are the default values.
