<div align="center">
<img src="assets/accum_grad_v5_reduced.png" width="35%" alt='gradient-accumulator'>
<h1 align="center">GradientAccumulator</h1>
<h3 align="center">Seemless gradient accumulation for TensorFlow 2</h3>

[![Pip Downloads](https://img.shields.io/pypi/dm/gradient-accumulator?label=pip%20downloads&logo=python)](https://pypi.org/project/gradient-accumulator/)
[![PyPI version](https://badge.fury.io/py/gradient-accumulator.svg)](https://badge.fury.io/py/gradient-accumulator)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7890319.svg)](https://doi.org/10.5281/zenodo.5597521)

**GradientAccumulator** was developed by SINTEF Health due to the lack of an easy-to-use method for gradient accumulation in TensorFlow 2.

The package is available on PyPI and is compatible with and have been tested against `TensorFlow 2.2-2.12` and `Python 3.6-3.12`, and works cross-platform (Ubuntu, Windows, macOS).
</div>

## [Continuous integration](https://github.com/andreped/GradientAccumulator#continuous-integration)

| Build Type | Status |
| - | - |
| **Code coverage** | [![codecov](https://codecov.io/gh/andreped/GradientAccumulator/branch/main/graph/badge.svg?token=MWLK71V750)](https://codecov.io/gh/andreped/GradientAccumulator) |
| **Documentations** | [![Documentation Status](https://readthedocs.org/projects/gradientaccumulator/badge/?version=latest)](https://gradientaccumulator.readthedocs.io/en/latest/?badge=latest) |
| **Unit tests** | [![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)](https://github.com/andreped/GradientAccumulator/actions) |


## [Install](https://github.com/andreped/GradientAccumulator#install)

Stable release from PyPI:
```
pip install gradient-accumulator
```

Or from source:
```
pip install git+https://github.com/andreped/GradientAccumulator
```

## [Getting started](https://github.com/andreped/GradientAccumulator#getting-started)

A simple example to add gradient accumulation to an existing model is by:
```
from gradient_accumulator import GradientAccumulateModel
from tensorflow.keras.models import Model

model = Model(...)
model = GradientAccumulateModel(accum_steps=4, inputs=model.input, outputs=model.output)
```

Then simply use the `model` as you normally would!

In practice, using gradient accumulation with a custom pipeline might require some extra overhead and tricks to get working.

For more information, see documentations which are hosted at [gradientaccumulator.readthedocs.io](https://gradientaccumulator.readthedocs.io/en/latest/)


## [What?](https://github.com/andreped/GradientAccumulator#what)
Gradient accumulation (GA) enables reduced GPU memory consumption through dividing a batch into smaller reduced batches, and performing gradient computation either in a distributing setting across multiple GPUs or sequentially on the same GPU. When the full batch is processed, the gradients are the _accumulated_ to produce the full batch gradient.

<p align="center">
<img src="assets/grad_accum.png" width="70%">
</p>


## [Why?](https://github.com/andreped/GradientAccumulator#why)
In TensorFlow 2, there did not exist a plug-and-play method to use gradient accumulation with any custom pipeline. Hence, we have implemented two generic TF2-compatible approaches:

| Method | Usage |
| - | - |
| `GradientAccumulateModel` | `model = GradientAccumulateModel(accum_steps=4, inputs=model.input, outputs=model.output)` |
| `GradientAccumulateOptimizer` | `opt = GradientAccumulateOptimizer(accum_steps=4, optimizer=tf.keras.optimizers.SGD(1e-2))` |

Both approaches control how frequently the weigths are updated, but in their own way. Approach (1) is for single-GPU only, whereas (2) supports both single-GPU and distributed training (multi-GPU). However, note that (2) is not yet working as intended. Hence, use (1) for most applications.

Our implementations enable theoretically **infinitely large batch size**, with **identical memory consumption** as for a regular mini batch. If a single GPU is used, this comes at the cost of increased training runtime. Multiple GPUs could be used to improve runtime performance.

| Technique | Usage |
| - | - |
| `Batch Normalization` | `layer = AccumBatchNormalization(accum_steps=4)` |
| `Adaptive Gradient Clipping` | `model = GradientAccumulateModel(accum_steps=4, agc=True, inputs=model.input, outputs=model.output)` |
| `Mixed precision` | `model = GradientAccumulateModel(accum_steps=4, mixed_precision=True, inputs=model.input, outputs=model.output)` |

* As batch normalization (BN) is not natively compatible with GA, we have implemented a custom BN layer which can be used as a drop-in replacement.
* Support for adaptive gradient clipping has been added as an alternative to BN.
* Mixed precision can also be utilized on both GPUs and TPUs.

For more information on usage, supported techniques, and examples, refer to [the documentations](https://gradientaccumulator.readthedocs.io/en/latest/).


## [Acknowledgements](https://github.com/andreped/GradientAccumulator#acknowledgements)
The gradient accumulator model wrapper is based on the implementation presented in [this thread](https://stackoverflow.com/a/66524901) on stack overflow. The adaptive gradient clipping method is based on [the implementation by @sayakpaul](https://github.com/sayakpaul/Adaptive-Gradient-Clipping).
The optimizer wrapper is derived from [the implementation by @fsx950223 and @stefan-falk](https://github.com/tensorflow/addons/pull/2525).

The documentations hosted [here](https://gradientaccumulator.readthedocs.io/en/latest/index.html) was made possible by the incredible [ReadTheDocs team](https://readthedocs.org/) which offer free documentation hosting!

  
## [How to cite?](https://github.com/andreped/GradientAccumulator#how-to-cite)
If you used this package or found the project relevant in your research, please, include the following citation:

```
@software{andre_pedersen_2023_7890319,
  author       = {André Pedersen and Javier Pérez de Frutos and David Bouget},
  title        = {andreped/GradientAccumulator: v0.4.2},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.2},
  doi          = {10.5281/zenodo.5597521},
  url          = {https://doi.org/10.5281/zenodo.5597521}
}
```
