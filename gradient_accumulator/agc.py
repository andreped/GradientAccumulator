import tensorflow as tf


# implementation from: https://github.com/sayakpaul/Adaptive-Gradient-Clipping/blob/main/agc.py
def compute_norm(x, axis, keepdims):
    """
    Computes the euclidean norm of a tensor :math:`x`.

    Args:
        x: input tensor.
        axis: which axis to compute norm across.
        keepdims: whether to keep dimension after applying along axis.
    
    Returns:
        Euclidean norm.
    """
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    """
    Wrapper class which dynamically sets `axis` and `keepdims` given an
    input `x` for calculating euclidean norm.

    Args:
        x: input tensor.

    Returns:
        Euclidean norm.
    """
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2]
        keepdims = True
    elif len(x.get_shape()) == 5:  # Conv kernels of shape HWDIO
        axis = [0, 1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4, 5]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor: float = 0.01, eps: float = 1e-3):
    """
    Performs adaptive gradient clipping on a given set of parameters and gradients.

    * Official JAX implementation (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
    * Ross Wightman's implementation https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py

    Args:
        parameters: Which parameters to apply method on.
        gradients: Which gradients to apply clipping on.
        clip_factor: Sets upper limit for gradient clipping.
        eps: Epsilon - small number in :math:`max()` to avoid zero norm and preserve numerical stability.
    
    Returns:
        Updated gradients after gradient clipping.
    """
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads
