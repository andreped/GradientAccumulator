import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from gradient_accumulator import GradientAccumulateModel
from gradient_accumulator import GradientAccumulateOptimizer

from .utils import get_opt
from .utils import normalize_img
from .utils import reset
from .utils import run_experiment

# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])


def test_model_expected_result():
    # set seed
    reset()

    # run once
    result1 = run_experiment(
        bs=100, accum_steps=1, epochs=2, modeloropt="model"
    )

    # reset before second run to get identical results
    reset()

    # test with model wrapper instead
    result2 = run_experiment(bs=50, accum_steps=2, epochs=2, modeloropt="model")

    # results should be identical (theoretically, even in practice on CPU)
    if tf_version <= 10:
        assert result1 == result2
    else:
        # approximation worse for tf >= 2.11
        np.testing.assert_almost_equal(result1, result2, decimal=2)
