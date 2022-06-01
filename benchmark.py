import numpy as np
import tensorflow as tf
import random as python_random
import os

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# https://stackoverflow.com/a/71311207
tf.config.experimental.enable_op_determinism()

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from GradientAccumulator.accumulator import GradientAccumulator, OldGradientAccumulator
from GradientAccumulator.adamAccumulate import AdamAccumulated


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


# https://stackoverflow.com/a/66524901
class CustomModel(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data

        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


if __name__ == "__main__":

    # params
    batchsize = 64  # 8 vs 32
    accum_steps = 1  # 4 vs 1
    nb_epochs = 3  # 3
    accum_opt = 2  # which implementation of accumulated gradients to use. Default 2 (should be the best one)

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batchsize)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # build test pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batchsize)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # compile model
    opt = tf.keras.optimizers.Adam(1e-3)
    # opt = tf.keras.optimizers.SGD(0.1)
    if accum_opt == 0:
        pass  # no accumulated gradients - normal Adam
    elif accum_opt == 1:
        opt = AdamAccumulated(accumulation_steps=accum_steps)
    elif accum_opt == 2:
        opt = GradientAccumulator(opt, accum_steps=accum_steps)
    elif accum_opt == 3:
        opt = OldGradientAccumulator(opt, accum_steps=accum_steps)
    elif accum_opt == -1:
        model = CustomModel(n_gradients=accum_steps, inputs=model.input, outputs=model.output)
    else:
        raise ValueError("Uknown accumulate gradient method was chosen. Available wrappers are: {0, 1, 2, 3}.")

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=nb_epochs * accum_steps,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True, custom_objects={
        "GradientAccumulator": GradientAccumulator,  # all these custom objects are added for convenience when testing - all are not needed otherwise
        "AdamAccumulated": AdamAccumulated,
        "OldGradientAccumulator": OldGradientAccumulator,
        "Adam": tf.keras.optimizers.Adam(1e-3)
        })

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
