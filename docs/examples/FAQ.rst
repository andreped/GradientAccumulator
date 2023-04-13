FAQ
===

Below are listed some details from frequently asked questions (FAQ).

Model format
------------

It is recommended to use the SavedModel format when using this implementation.
That is because the HDF5 format is only compatible with `TF <= 2.6` when using
the model wrapper. However, if you are using older TF versions, both formats
work out-of-the-box. The SavedModel format works fine for all versions of TF 2.x.


macOS compatibility
-------------------

Note that GradientAccumulator is perfectly compatible with macOS, both with
and without GPUs. In order to have GPU support on macOS, you will need to
install the tensorflow-compiled version that is compatible with metal:

.. code-block:: bash

    pip install tensorflow-metal


GradientAccumulator can be used as usually. However, note that there only
exists one tf-metal version, which should be equivalent to TF==2.5.


TensorFlow 1.x compatibility
--------------------------

For TF 1, I suggest using the AccumOptimizer implementation in the
`H2G-Net repository <https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139>`_.

This wrapper works similarly to our `GradientAccumulateOptimizer`
wrapper.

An equivalent `GradientAccumulateModel` wrapper does not exist in
TF 1.x as overloading of the `train_step` was a new feature
introduced in `tensorflow==2.2`.

Hence, also note that for `tensorflow<2.2>=2.0` only the
`GradientAccumulateOptimizer` is compatible.


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


PyTorch
-------

For PyTorch, I would recommend using
`accelerate <https://pypi.org/project/accelerate/>`_.
HuggingFace :hugs: has a great tutorial on how to use it
`here <https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation>`_.

However, if you wish to use native PyTorch and you are implementing
your own training loop, you could do something like this:

.. code-block:: python

    # batch accumulation parameter
    accum_iter = 4

    # loop through enumaretad batches
    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # passes and weights update
        with torch.set_grad_enabled(True):
            
            # forward pass 
            preds = model(inputs)
            loss  = criterion(preds, labels)

            # scale loss prior to accumulation
            loss = loss / accum_iter

            # backward pass
            loss.backward()

            # weights update and reset gradients
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
