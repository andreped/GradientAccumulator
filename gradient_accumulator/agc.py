import tensorflow as tf


# implementation from: https://github.com/sayakpaul/Adaptive-Gradient-Clipping/blob/main/agc.py  # noqa
SCALAR = tf.constant([], dtype=tf.int32)
LINEAR = tf.constant([0], dtype=tf.int32)
TENSOR2D = tf.constant([0, 1], dtype=tf.int32)
TENSOR3D = tf.constant([0, 1, 2], dtype=tf.int32)
TENSOR4D = tf.constant([0, 1, 2, 3], dtype=tf.int32)


@tf.function
def compute_norm(x, axis, keepdims):
    """
    Computes the euclidean norm of a tensor :math:`x`.
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims))


@tf.function
def unitwise_norm(x):
    """
    Computes the unitwise norm of a tensor.
    """

    def compute_reduction_axes(r):
        axes = tf.case(
            [
                (
                    tf.equal(r, 1),
                    lambda: SCALAR,
                ),
                (
                    tf.equal(r, 2),
                    lambda: LINEAR,
                ),
                (
                    tf.equal(r, 3),
                    lambda: TENSOR2D,
                ),
                (
                    tf.equal(r, 4),
                    lambda: TENSOR3D,
                ),
                (
                    tf.equal(r, 5),
                    lambda: TENSOR4D,
                ),
            ],
            default=lambda: SCALAR,
        )
        return axes

    return compute_norm(x, axis=compute_reduction_axes(tf.rank(x)), keepdims=True)


@tf.function
def adaptive_clip_grad(
    parameters, gradients, clip_factor: float = 0.01, eps: float = 1e-3
):
    """
    Performs adaptive gradient clipping on a given set of parameters and gradients.
    """

    def clip_grad(param, grad):
        max_norm = tf.math.multiply(
            tf.math.maximum(unitwise_norm(param), eps), clip_factor
        )
        grad_norm = unitwise_norm(grad)
        adjusted_norm = tf.math.divide(max_norm, tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(
            tf.math.less(grad_norm, max_norm),
            grad,
            tf.math.multiply(grad, adjusted_norm),
        )
        return new_grad

    new_grads = tf.map_fn(
        lambda x: clip_grad(x[0], x[1]), (parameters, gradients), dtype=tf.float32
    )

    return new_grads
