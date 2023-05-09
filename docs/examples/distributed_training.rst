Distributed training
====================

Optimizer wrapper
-----------------

In order to train with multiple GPUs, you can use the Optimizer wrapper:


.. code-block:: python

    import tensorflow as tf
    from gradient_accumulator import GradientAccumulateOptimizer

    opt = GradientAccumulateOptimizer(accum_steps=4, optimizer=tf.keras.optimizers.SGD(1e-2))


Just remember to wrap the optimizer within the `tf.distribute.MirroredStrategy`.

A more comprehensive example can be seen below:


.. code-block:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds
    from gradient_accumulator import GradientAccumulateOptimizer


    # tf.keras.mixed_precision.set_global_policy("mixed_float16")  # Don't have GPU on the cloud when running CIs
    strategy = tf.distribute.MirroredStrategy()

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(1)

    with strategy.scope():
        # create model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # define optimizer - currently only SGD compatible with GAOptimizerWrapper
        if int(tf.version.VERSION.split(".")[1]) > 10:
            curr_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
        else:
            curr_opt = tf.keras.optimizers.SGD(learning_rate=1e-2)

        # wrap optimizer to add gradient accumulation support
        opt = GradientAccumulateOptimizer(optimizer=curr_opt, accum_steps=10)

        # compile model
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # train model
    model.fit(
        ds_train,
        epochs=3,
        validation_data=ds_test,
        verbose=1
    )


Model wrapper
-------------

If model wrapping is more of interest, experimental multi-GPU support can be
made available through the *experimental_distributed_support* flag:

.. code-block: python
    from gradient_accumulator import GradientAccumulateModel

    model = GradientAccumulateModel(
        accum_steps=8, experimental_distributed_support=True,
        inputs=model.input, outputs=model.output
    )

To test usage, replace the optimizer wrapper in the example above with this
model wrapper.
