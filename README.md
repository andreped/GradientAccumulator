<div align="center">
<h1 align="center">GradientAccumulator</h1>
<h3 align="center">Seemless gradient accumulation for TensorFlow 2</h3>

[![Pip Downloads](https://img.shields.io/pypi/dm/gradient-accumulator?label=pip%20downloads&logo=python)](https://pypi.org/project/gradient-accumulator/)
[![PyPI version](https://badge.fury.io/py/gradient-accumulator.svg)](https://badge.fury.io/py/gradient-accumulator)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7306095.svg)](https://doi.org/10.5281/zenodo.7306095)
[![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)](https://github.com/andreped/GradientAccumulator/actions)
 
**GradientAccumulator** enables gradient accumulation (GA) by overloading the train_step of any given tf.keras.Model, to update correctly according to a user-specified number of accumulation steps. GA enables theoretically infinitely large batch size, with the same memory consumption as for a regular mini batch, at the cost of increased runtime. To improve runtime, mixed precision is supported. As batch normalization is not natively compatible with GA, support for adaptive gradient clipping has been added as an alternative.

Package is compatible with and have been tested against TF >= 2.2 and Python >= 3.6 (tested with 3.6-3.10), and works cross-platform (Ubuntu, Windows, macOS).
</div>


## Install

Stable release from PyPI:
```
pip install gradient-accumulator
```

Or from source:
```
pip install git+https://github.com/andreped/GradientAccumulator
```

## Usage
```
from gradient_accumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras.models import Model

model = Model(...)
model = GAModelWrapper(accum_steps=4, inputs=model.input, outputs=model.output)
```

Then simply use the `model` as you normally would!

#### Mixed precision
There has also been added experimental support for mixed precision:
```
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam

mixed_precision.set_global_policy('mixed_float16')
model = GAModelWrapper(accum_steps=4, mixed_precision=True, inputs=model.input, outputs=model.output)

opt = Adam(1e-3, epsilon=1e-4)
opt = mixed_precision.LossScaleOptimizer(opt)
```

If using TPUs, use `bfloat16` instead of `float16`, like so:
```
mixed_precision.set_global_policy('mixed_bfloat16')
```

There is also an example of how to use gradient accumulation with mixed precision [here](https://github.com/andreped/GradientAccumulator/blob/main/tests/test_mixed_precision.py#L58).

#### Adaptive gradient clipping
There has also been added support for adaptive gradient clipping, based on [this](https://github.com/sayakpaul/Adaptive-Gradient-Clipping) implementation:
```
model = GAModelWrapper(accum_steps=4, use_agc=True, clip_factor=0.01, eps=1e-3, inputs=model.input, outputs=model.output)
```

The hyperparameters values for `clip_factor` and `eps` presented here are the default values.

#### Model format
It is recommended to use the SavedModel format when using this implementation. That is because the HDF5 format is only compatible with `TF <= 2.6` when using the model wrapper. However, if you are using older TF versions, both formats work out-of-the-box. The SavedModel format works fine for all versions of TF 2.x

#### macOS compatibility
Note that GradientAccumulator is perfectly compatible with macOS, both with and without GPUs. In order to have GPU support on macOS, you will need to install the tensorflow-compiled version that is compatible with metal:
```
pip install tensorflow-metal
```

GradientAccumulator can be used as usually. However, note that there only exists one tf-metal version, which should be equivalent to TF==2.5.


## Disclaimer
In theory, one should be able to get identical results for batch training and using gradient accumulation. However, in practice, one may observe a slight difference. One of the cause may be when operations are used (or layers/optimizer/etc) that update for each step, such as Batch Normalization. It is **not** recommended to use BN with GA, as BN would update too frequently. However, you could try to adjust the `momentum` of BN (see [here](https://keras.io/api/layers/normalization_layers/batch_normalization/)).

It was also observed a small difference when using adaptive optimizers, which I believe might be due to how frequently they are updated. Nonetheless, for the optimizers, the difference was quite small, and one may approximate batch training quite well using our GA implementation, as rigorously tested [here](https://github.com/andreped/GradientAccumulator/tree/main/tests)).

## TODOs:
- [ ] Add multi-GPU support

## Acknowledgements
The gradient accumulator model wrapper is based on the implementation presented in [this thread](https://stackoverflow.com/a/66524901) on stack overflow.

The adaptive gradient clipping method is based on [the implementation by @sayakpaul](https://github.com/sayakpaul/Adaptive-Gradient-Clipping).

This repository serves as an open solution for everyone to use, until TF/Keras integrates a proper solution into their framework(s).

## Troubleshooting
Overloading of `train_step` method of tf.keras.Model was introduced in TF 2.2, hence, this code is compatible with TF >= 2.2.

Also, note that TF depends on different python versions. If you are having problems getting TF working, try a different TF version or python version.

For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead, which wraps the optimizer instead of overloading the train_step of the Model itself (new feature in TF2).

## How to cite
If you use this package in your research, please, cite this reference:
```
@software{andre_pedersen_2022_7023582,
  author       = {André Pedersen and
                  David Bouget},
  title        = {andreped/GradientAccumulator: v0.2.1},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.2.1},
  doi          = {10.5281/zenodo.7023582},
  url          = {https://doi.org/10.5281/zenodo.7023582}}
```
