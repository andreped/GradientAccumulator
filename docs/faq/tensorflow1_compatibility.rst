TensorFlow 1.x compatibility
----------------------------

For TF 1, I suggest using the AccumOptimizer implementation in the
`H2G-Net repository <https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139>`_.

This wrapper works similarly to our `GradientAccumulateOptimizer`
wrapper.

An equivalent `GradientAccumulateModel` wrapper does not exist in
TF 1.x as overloading of the `train_step` was a new feature
introduced in `tensorflow==2.2`.

Hence, also note that for `tensorflow<2.2>=2.0` only the
`GradientAccumulateOptimizer` is compatible.
