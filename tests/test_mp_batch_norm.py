import multiprocessing as mp


def run_experiment(custom_bn:bool = True, bs:int = 100, accum_steps:int = 1, epochs:int = 3, queue=None, mixed_precision=True):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.models import load_model
    from gradient_accumulator import GradientAccumulateModel
    from gradient_accumulator.layers import AccumBatchNormalization
    import random as python_random
    import numpy as np
    import os


    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label


    ## reset session and seed stuff before running experiment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    # set mixed global precision policy
    if mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    # define which normalization layer to use in network
    if custom_bn:
        normalization_layer = AccumBatchNormalization(accum_steps=accum_steps)
    elif not custom_bn:
        normalization_layer = tf.keras.layers.BatchNormalization()
    else:
        normalization_layer = tf.keras.layers.Activation("linear")

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(32),
        normalization_layer,  # @TODO: BN before or after ReLU? Leads to different performance
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(10, dtype=tf.float32)
    ])

    # wrap model to use gradient accumulation
    if accum_steps > 1:
        model = GradientAccumulateModel(
            accum_steps=accum_steps, mixed_precision=mixed_precision,
            inputs=model.input, outputs=model.output
        )

    # need to scale optimizer for mixed precision
    opt = tf.keras.optimizers.SGD(1e-2)
    opt = mixed_precision.LossScaleOptimizer(opt)

    # compile model
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True)

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)

    # add result to queue to fetch it later on
    queue.put(result)


def run_experiment_wrapper(custom_bn=True, bs=100, accum_steps=1, epochs=3, mixed_precision=True):
    # launch experiment in separate process, as we are enabling mixed precision
    # which will impact other unit tests, unless we do this
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()
    
    try:
        mp.set_start_method('spawn', force=True)  # set start method to 'spawn' BEFORE instantiating the queue and the event
    except RuntimeError:
        pass
    
    queue = mp.Queue()
    p = mp.Process(target=run_experiment(custom_bn=custom_bn, bs=bs, accum_steps=accum_steps, epochs=epochs, queue=queue))
    try:
        p.start()
    finally:
        p.join()  # necessary so that the Process exists before the test suite exits (thus coverage is collected)
    
    return queue.get()


def test_mixed_precision():
    import numpy as np

    # custom BN without accum
    result1 = run_experiment_wrapper(custom_bn=True, accum_steps=4, bs=25, mixed_precision=False)[1]

    # keras BN without accum
    result2 = run_experiment_wrapper(custom_bn=True, accum_steps=1, bs=100, mixed_precision=False)[1]

    # assert result1 == result2
    np.testing.assert_almost_equal(result1, result2, decimal=2)


    # custom BN with accum with mixed precision
    result3 = run_experiment_wrapper(custom_bn=True, accum_steps=4, bs=25, mixed_precision=True)[1]

    # keras BN without accum
    result4 = run_experiment_wrapper(custom_bn=True, accum_steps=1, bs=100, mixed_precision=True)[1]

    np.testing.assert_almost_equal(result3, result4, decimal=2)
