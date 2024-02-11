import pytest
import tensorflow as tf
from gradient_accumulator import GradientAccumulateOptimizer
from .utils import get_opt

tf_version = int(tf.version.VERSION.split(".")[1])

tf.config.run_functions_eagerly(True)

@pytest.fixture
def optimizer():
    opt = get_opt(opt_name="SGD", tf_version=tf_version)
    return GradientAccumulateOptimizer(optimizer=opt, accum_steps=2)

def test_learning_rate_getter(optimizer):
    assert optimizer.learning_rate == 0.01

def test_learning_rate_setter(optimizer):
    optimizer.learning_rate = 0.02
    assert optimizer.learning_rate == 0.02

def test_lr_getter(optimizer):
    assert optimizer.lr == 0.01

def test_lr_setter(optimizer):
    optimizer.lr = 0.02
    assert optimizer.lr == 0.02

def test__learning_rate(optimizer):
    assert optimizer._learning_rate == 0.01
    optimizer.learning_rate = 0.02
    assert optimizer._learning_rate == 0.02

def test_step_getter(optimizer):
    assert optimizer.step == 1

def test_step_setter(optimizer):
    optimizer.step = 1
    assert optimizer.step == 1

def test_iterations_setter(optimizer):
    optimizer.iterations = 1
    assert optimizer.iterations == 1

def test_optimizer_prop(optimizer):
    assert optimizer.optimizer.__class__ == get_opt(opt_name="SGD", tf_version=tf_version).__class__

def test_reset_single_gradient(optimizer):
    var = tf.Variable([1.0, 2.0], dtype=tf.float32)
    optimizer.add_slot(var, "ga", initializer=tf.constant([3.0, 4.0]))
    gradient = optimizer.get_slot(var, "ga")
    optimizer._reset_single_gradient(gradient)
    assert tf.reduce_all(tf.equal(gradient, tf.zeros_like(gradient)))

def test_reset(optimizer):
    var1 = tf.Variable([1.0, 2.0], dtype=tf.float32)
    var2 = tf.Variable([3.0, 4.0], dtype=tf.float32)
    optimizer.add_slot(var1, "ga", initializer=tf.constant([5.0, 6.0]))
    optimizer.add_slot(var2, "ga", initializer=tf.constant([7.0, 8.0]))
    for var in [var1, var2]:
        gradient = optimizer.get_slot(var, "ga")
        assert tf.reduce_all(tf.equal(gradient, tf.zeros_like(gradient))).numpy() == False

    optimizer.reset()
    for var in [var1, var2]:
        gradient = optimizer.get_slot(var, "ga")
        assert tf.reduce_all(tf.equal(gradient, tf.zeros_like(gradient))).numpy() == True


@pytest.mark.parametrize("accum_steps", [1, 2])
@pytest.mark.parametrize("use_agc", [True, False])
def test_parse_grad(optimizer, use_agc, accum_steps):
    var = tf.Variable([1.0, 2.0], dtype=tf.float32)
    if accum_steps == 1:
        expected_grad = tf.zeros_like(var)  # gradients should not be applied yet
    else:
        expected_grad = tf.constant([3.0, 4.0])
    optimizer.add_slot(var, "ga", initializer=expected_grad)
    accum_gradient = optimizer.get_slot(var, "ga")

    optimizer.use_agc = use_agc
    optimizer.step.assign(accum_steps)

    parsed_grad = optimizer._parse_grad(accum_gradient, var)
    assert tf.reduce_all(tf.equal(parsed_grad, expected_grad)).numpy() == True


@pytest.fixture
def optimizer_with_grads(optimizer):
    opt = optimizer
    var = tf.Variable([1.0, 2.0], dtype=tf.float32)

    opt.add_slot(var, "ga", initializer=tf.constant([3.0, 4.0]))

    return opt, var

def test_reset_accum_gradient_condition(optimizer_with_grads):
    optimizer, var = optimizer_with_grads

    accum_grad = optimizer.get_slot(var, "ga")
    accum_grad.assign(tf.constant([3.0, 4.0], dtype=tf.float32))

    current_grad = tf.constant([3.0, 4.0], dtype=tf.float32)

    result_grad = optimizer.reset_accum_gradient(accum_grad, current_grad)

    expected_grad = tf.zeros_like(accum_grad)

    tf.debugging.assert_equal(result_grad, expected_grad, message="Gradients should be reset to zeros")

@pytest.fixture
def optimizer_adam():
    opt = get_opt(opt_name="adam", tf_version=tf_version)
    return GradientAccumulateOptimizer(optimizer=opt, accum_steps=2)

@pytest.fixture
def optimizer_with_sparse_grads(optimizer_adam):
    opt = optimizer_adam
    var = tf.Variable(tf.zeros([10, 10]), dtype=tf.float32)

    opt.add_slot(var, "ga", initializer=tf.zeros_like(var))
    opt.add_slot(var, "m", initializer=tf.zeros_like(var))
    opt.add_slot(var, "v", initializer=tf.zeros_like(var))

    return opt, var

def test_resource_apply_sparse(optimizer_with_sparse_grads):
    optimizer, var = optimizer_with_sparse_grads

    indices = tf.constant([0, 1], dtype=tf.int64)
    updates = tf.constant([[0.1] * 10, [0.2] * 10], dtype=tf.float32)

    optimizer._reset_single_gradient(optimizer.get_slot(var, "ga"))

    grad = tf.IndexedSlices(values=updates, indices=indices, dense_shape=var.shape)

    optimizer._resource_apply_sparse(grad.values, var, grad.indices)
    optimizer._resource_apply_sparse(grad.values, var, grad.indices)

    accumulated_grads = optimizer.get_slot(var, "ga")
    expected_accumulated_grads = tf.scatter_nd(tf.expand_dims(indices, 1), updates * 2, var.shape) / optimizer.accum_steps
    tf.debugging.assert_near(accumulated_grads, expected_accumulated_grads, atol=1e-5)


if __name__ == "__main__":
    test__learning_rate(optimizer())
    test_learning_rate_getter(optimizer())
    test_learning_rate_setter(optimizer())
    test_lr_getter(optimizer())
    test_lr_setter(optimizer())
    test_step_getter(optimizer())
    test_step_setter(optimizer())
    test_optimizer_prop(optimizer())
    test_reset_single_gradient(optimizer())
    test_reset(optimizer())
    test_parse_grad(optimizer())
    test_reset_accum_gradient_condition(optimizer_with_grads())
    test_resource_apply_sparse(optimizer_with_sparse_grads())