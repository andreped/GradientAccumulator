<div align="center">
<h1 align="center">GradientAccumulator</h1>
<h3 align="center">Seemless gradient accumulation for TensorFlow 2</h3>

[![Pip Downloads](https://img.shields.io/pypi/dm/gradient-accumulator?label=pip%20downloads&logo=python)](https://pypi.org/project/gradient-accumulator/)
[![PyPI version](https://badge.fury.io/py/gradient-accumulator.svg)](https://badge.fury.io/py/gradient-accumulator)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7581815.svg)](https://doi.org/10.5281/zenodo.7581815)
[![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)](https://github.com/andreped/GradientAccumulator/actions)

**GradientAccumulator** was developed by SINTEF Health due to the lack of an easy-to-use method for gradient accumulation in TensorFlow 2.

The package is available on PyPI and is compatible with and have been tested against TF >= 2.2 and Python >= 3.6 (tested with 3.6-3.10), and works cross-platform (Ubuntu, Windows, macOS).
</div>

## What?
Gradient accumulation (GA) enables reduced GPU memory consumption through dividing a batch into smaller reduced batches, and performing gradient computation either in a distributing setting across multiple GPUs or sequentially on the same GPU. When the full batch is processed, the gradients are the _accumulated_ to produce the full batch gradient.

<p align="center">
<img src="assets/grad_accum.png" width="50%">
</p>


## Why?
In TensorFlow 2, there did not exist a plug-and-play method to use gradient accumulation with any custom pipeline. Hence, we have implemented two generic TF2-compatible approaches:

| Method | Usage |
| - | - |
| `GAModelWrapper` | `model = GAModelWrapper(accum_steps=4, inputs=model.input, outputs=model.output)` |
| `GAOptimizerWrapper` | `opt = GAOptimizerWrapper(accum_steps=4, optimizer=tf.keras.optimizers.Adam(1e-3))` |

1) A generic approach which overloads the `train_step` of any given `tf.keras.Model` and 2) simple optimizer wrapper which changed how frequently the gradients should update.

For our single-GPU approach, our implementation enables theoretically **infinitely large batch size**, with **identical memory consumption** as for a regular mini batch. This comes at the cost of increased training runtime. Multiple GPUs could be used to increase runtime performance. However, our `train_step` approach is not currently compatible with TensorFlow's `tf.distribute` (thoroughly discussed [here](https://github.com/keras-team/keras/issues/17429#issuecomment-1405612981)).

As batch normalization is not natively compatible with GA, support for adaptive gradient clipping has been added as an alternative. We have also added support for mixed precision and both GPU and TPU support.


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

<details open>
<summary>

#### Mixed precision</summary>

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
</details>


<details>
<summary>

#### Adaptive gradient clipping</summary>

There has also been added support for adaptive gradient clipping, based on [this](https://github.com/sayakpaul/Adaptive-Gradient-Clipping) implementation:
```
model = GAModelWrapper(accum_steps=4, use_agc=True, clip_factor=0.01, eps=1e-3, inputs=model.input, outputs=model.output)
```

The hyperparameters values for `clip_factor` and `eps` presented here are the default values.
</details>


<details>
<summary>

#### Model format</summary>

It is recommended to use the SavedModel format when using this implementation. That is because the HDF5 format is only compatible with `TF <= 2.6` when using the model wrapper. However, if you are using older TF versions, both formats work out-of-the-box. The SavedModel format works fine for all versions of TF 2.x
</details>


<details>
<summary>

#### macOS compatibility</summary>
Note that GradientAccumulator is perfectly compatible with macOS, both with and without GPUs. In order to have GPU support on macOS, you will need to install the tensorflow-compiled version that is compatible with metal:
```
pip install tensorflow-metal
```

GradientAccumulator can be used as usually. However, note that there only exists one tf-metal version, which should be equivalent to TF==2.5.
</details>


<details>
<summary>

#### TF 1.x</summary>
For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead, which wraps the optimizer instead of overloading the train_step of the Model itself (new feature in TF2).
</details>


<details>
<summary>

#### PyTorch</summary>
For PyTorch, I would recommend using [accelerate](https://pypi.org/project/accelerate/). HuggingFace :hugs: has a great tutorial on how to use it [here](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation).
</details>


<details>
<summary>

#### Troubleshooting</summary>

Overloading of `train_step` method of tf.keras.Model was introduced in TF 2.2, hence, this code is compatible with TF >= 2.2.

Also, note that TF depends on different python versions. If you are having problems getting TF working, try a different TF version or python version.
</details>


<details>
<summary>

#### Disclaimer</summary>
In theory, one should be able to get identical results for batch training and using gradient accumulation. However, in practice, one may observe a slight difference. One of the cause may be when operations are used (or layers/optimizer/etc) that update for each step, such as Batch Normalization. It is **not** recommended to use BN with GA, as BN would update too frequently. However, you could try to adjust the `momentum` of BN (see [here](https://keras.io/api/layers/normalization_layers/batch_normalization/)).

It was also observed a small difference when using adaptive optimizers, which I believe might be due to how frequently they are updated. Nonetheless, for the optimizers, the difference was quite small, and one may approximate batch training quite well using our GA implementation, as rigorously tested [here](https://github.com/andreped/GradientAccumulator/tree/main/tests)).
</details>
  

## Acknowledgements
The gradient accumulator model wrapper is based on the implementation presented in [this thread](https://stackoverflow.com/a/66524901) on stack overflow. The adaptive gradient clipping method is based on [the implementation by @sayakpaul](https://github.com/sayakpaul/Adaptive-Gradient-Clipping).
The optimizer wrapper is derived from [the implementation by @fsx950223 and @stefan-falk](https://github.com/tensorflow/addons/pull/2525).

  
## How to cite?
If you used this package or found the project relevant in your research, please, considering including the following citation:

```
@software{andre_pedersen_2023_7581815,
  author       = {André Pedersen and David Bouget},
  title        = {andreped/GradientAccumulator: v0.3.0},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.7581815},
  url          = {https://doi.org/10.5281/zenodo.7581815}
}
```
