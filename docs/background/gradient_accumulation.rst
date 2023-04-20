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
