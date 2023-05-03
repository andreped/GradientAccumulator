Gradient Accumulation
=====================

Gradient accumulation (GA) enables reduced GPU memory consumption through
dividing a batch into smaller reduced batches, and performing gradient
computation either in a distributing setting across multiple GPUs or
sequentially on the same GPU. When the full batch is processed, the
gradients are the *accumulated* to produce the full batch gradient.

.. image:: ../../assets/grad_accum.png
  :width: 70%
  :align: center
  :alt: Gradient accumulation update

The size and number of mini-batches are set by accum_steps and batch_size, where accum_steps and batch_size corresponds to the number and size of the mini-batches respectively. Together they approximate a batch size of accum_steps * batch_size.   
