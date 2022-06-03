# GradientAccumulator

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)

This repo contains a TensorFlow 2 compatible implementation of accumulated gradients.

The proposed implementation simply overloads the train_step of a given tf.keras.Model, to update correctly according to a user-specified number of accumulation steps. This enables gradient accumulation, which reduces memory consumption and enables usage of theoretically infinitely large batch size (among other things), at the cost of increased training runtime.

## Install

#### From latest release:
```
pip install https://github.com/andreped/GradientAccumulator/releases/download/v0.1.2/GradientAccumulator-0.1.2-py3-none-any.whl
```

#### Or from source:
```
pip install git+https://github.com/andreped/GradientAccumulator
```

## Usage
```
from GradientAccumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras.models import Model

model = Model(...)
model = GAModelWrapper(n_gradients=4, inputs=model.input, outputs=model.output)
```

Then simply use the `model` as you normally would!

## TODOs:
- [x] Add generic wrapper class for adding accumulated gradients to any optimizer
- [x] Add CI to build wheel and test that it works across different python versions, TF versions, and operating systems.
- [x] Add benchmarks to verfiy that accumulated gradients actually work as intended
- [x] Add class_weight support
- [x] Add multi-input/-output architecture support
- [x] GAModelWrapper gets expected identical results to batch training!
- [x] Test method for memory leaks
- [ ] Add wrapper class for BatchNormalization layer, similar as done for optimizers
- [ ] Add proper multi-GPU support

## Acknowledgements
This implementation is derived from the work of @fsx950223, @stefan-falk, and others, which is a closed PR https://github.com/tensorflow/addons/pull/2525 to TF-addons. The model wrapper solution was also inspired by [this thread](https://stackoverflow.com/a/66524901) on stack overflow. Hence, all credit to them and the people who contributed to the work! This could not have been possible without the people asking the right questions and the people contributing with solutions!

This repository serves as an open solution for everyone to use, until TF/Keras integrates a proper solution into their framework(s).

## Troubleshooting
Overloading of `train_step` method of tf.keras.Model was introduced in TF 2.2, hence, this code is compatible with TF >= 2.2.

For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead, which wraps the optimizer instead of overloading the train_step of the Model itself (new feature in TF2).
