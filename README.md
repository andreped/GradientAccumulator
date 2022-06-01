# GradientAccumulator

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)

### **DISCLAIMER: This is an experimental project - current solution is not stable. Use with caution.**

This repo contains a TensorFlow 2 compatible implementation of accumulated gradients.

The proposed implementation simply overloads the train_step of a given tf.keras.Model, to update correctly according to a user-specified number of accumulation steps.

This enables gradient accumulation, which reduces memory consumption and enables usage of theoretically infinitely large batch size (among other things), at the cost of increased training runtime.

Precompiled wheel compatible with Python 3.7-3.9 and TensorFlow 2.7-2.9 exist in [Release](https://github.com/andreped/GradientAccumulator/releases/tag/v0.1.0),
but you can build from source if you want to test if it works in your setup (see [here](https://github.com/andreped/GradientAccumulator#or-from-source-code)).

For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead, which wraps the optimizer instead of overloading the train_step of the Model itself (new feature in TF2).

## Install

#### From latest release:
```
pip install https://github.com/andreped/GradientAccumulator/releases/download/v0.1.1/GradientAccumulator-0.1.1-py3-none-any.whl
```

#### Or from source code:
```
pip install git+https://github.com/andreped/GradientAccumulator
```

## Usage
```
from GradientAccumulator.GAModelWrapper import GAModelWrapper
from tensorflow.keras.models import Model

model = Model(...)
model = GAModelWrapper(nb_gradients=4, inputs=model.input, outputs=model.output)
```

Then simply use the `model` as you normally would!

## Experiments
To verify that this implementation works, a benchmark was performed, comparing multiple alternative solutions proposed by others.

To perform the benchmark, create a virtual environment and install dependencies:
```
virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

Then just run this command:
```
python benchmark.py
```

An in-detail discussion and experiment results were presented in Issue https://github.com/andreped/GradientAccumulator/issues/2.

## TODOs:
- [x] Add generic wrapper class for adding accumulated gradients to any optimizer
- [x] Add CI to build wheel and test that it works across different python versions, TF versions, and operating systems.
- [x] Add benchmarks to verfiy that accumulated gradients actually work as intended
- [ ] Add wrapper class for BatchNormalization layer, similar as done for optimizers
- [ ] Test method for memory leaks
- [ ] Verify that implementation works in multi-GPU setups
- [ ] Add proper multi-GPU support

## Disclaimer
Note that this implementation is only compatible with newer versions of TensorFlow. This is because the way Optimizers behave in TF
has changed in TF 2. Slight modifications can likely be made to make this work for older versions, but I would recommend using
newer versions of TF 2 instead, as it has become more stable and feature rich than recent versions.

Also note that this implementation **does not work with TF 1**. For the same reason as it does not work with older TF 2 versions.
However, a TF 1 implementation can be found in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139).

## Acknowledgements
This implementation is derived from the work of @fsx950223, @stefan-falk, and others, which is a closed PR https://github.com/tensorflow/addons/pull/2525 to TF-addons. Hence, all credit to them and the people who contributed to the work! Sadly, the proposed implementation was not merged,
as there were some unresolved issues with it, especially regarding multi-GPU training. However, I believe the current implementation is working well
for single-GPU scenarios, which should already be of interest to the community.
