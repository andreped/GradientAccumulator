import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Embedding, Dense
from tensorflow.keras.models import load_model
from gradient_accumulator import GradientAccumulateOptimizer


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def test_sparse_optimizer():

    # Define 10 restaurant reviews
    reviews =[
            'Never coming back!',
            'horrible service',
            'rude waitress',
            'cold food',
            'horrible food!',
            'awesome',
            'awesome services!',
            'rocks',
            'poor work',
            'couldn\'t have done better'
    ]

    #Define labels
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # setup toy dataset
    vocab_size = 50
    encoded_reviews = [one_hot(d, vocab_size) for d in reviews]
    max_length = 4
    padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

    # define model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # wrap optimizer to add gradient accumulation support
    # need to dynamically handle which Optimizer class to use dependent on tf version
    if tf_version > 10:
        curr_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-2)
    else:
        curr_opt = tf.keras.optimizers.SGD(learning_rate=1e-2)  # IDENTICAL RESULTS WITH SGD!!!
    
    opt = GradientAccumulateOptimizer(optimizer=curr_opt, accum_steps=4, reduction="MEAN")

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['acc']
    )

    model.fit(
        padded_reviews,
        labels,
        epochs=2,
        verbose=1,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    trained_model = load_model("./trained_model", compile=True, custom_objects={"SGD": curr_opt})

    result = trained_model.evaluate(padded_reviews, labels, verbose=1)
    print(result)
