import tensorflow as tf
#from keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


# https://github.com/sokrypton/AccAdam_TF2
class AdamAccumulated(tf.keras.optimizers.Optimizer):  # optimizer_v2.OptimizerV2):
    def __init__(self,
                 accumulation_steps=1,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name='AdamAccumulated',
                 **kwargs):
        super(AdamAccumulated, self).__init__(name, **kwargs)
        self._set_hyper('accumulation_steps', accumulation_steps)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('epsilon', epsilon)

    def _create_slots(self, var_list):
        for var in var_list: self.add_slot(var, 'm')
        for var in var_list: self.add_slot(var, 'v')
        for var in var_list: self.add_slot(var, 'g')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamAccumulated, self)._prepare_local(var_device, var_dtype, apply_state)

        accumulation_steps = self._get_hyper('accumulation_steps', self.iterations.dtype)

        update_cond = tf.equal((self.iterations + 1) % accumulation_steps, 0)
        sub_step = self.iterations % accumulation_steps + 1
        local_step = math_ops.cast(self.iterations // accumulation_steps + 1, var_dtype)

        learning_rate = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        epsilon = array_ops.identity(self._get_hyper('epsilon', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))

        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = learning_rate * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                update_cond=update_cond,
                sub_step=sub_step,
                epsilon=epsilon,
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        accumulation_steps = self._get_hyper('accumulation_steps', self.iterations.dtype)
        update_cond = coefficients['update_cond']
        sub_step = coefficients['sub_step']
        eps = coefficients['epsilon']

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        g = self.get_slot(var, 'g')

        grad = grad / math_ops.cast(accumulation_steps, var_dtype)
        g_t = tf.where(tf.equal(sub_step, 1), grad, g + (grad - g) / math_ops.cast(sub_step, var_dtype))
        m_t = tf.where(update_cond, m * coefficients['beta_1_t'] + g_t * coefficients['one_minus_beta_1_t'], m)
        v_t = tf.where(update_cond, v * coefficients['beta_2_t'] + (g_t * g_t) * coefficients['one_minus_beta_2_t'], v)
        var_t = tf.where(update_cond, coefficients['lr'] * m_t / (math_ops.sqrt(v_t) + eps), tf.zeros_like(grad))

        var_update = state_ops.assign_sub(var, var_t, use_locking=self._use_locking)
        m_update = state_ops.assign(m, m_t, use_locking=self._use_locking)
        v_update = state_ops.assign(v, v_t, use_locking=self._use_locking)
        g_update = state_ops.assign(g, g_t, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_update, v_update, g_update])

    def get_config(self):
        config = super(AdamAccumulated, self).get_config()
        config.update({
            'accumulation_steps': self._serialize_hyperparameter('accumulation_steps'),
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self._serialize_hyperparameter('epsilon'),
        })
        return config
