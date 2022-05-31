# GradientAccumulator

![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)

This repo contains a TensorFlow 2 compatible implementation of accumulated gradients.

Simply wrap the accumulator over any optimizer, and specify `accum_steps` to control number of accumulations.

Precompiled wheel compatible with Python 3.7-3.9 and TensorFlow 2.7-2.9 exist in [Release](https://github.com/andreped/GradientAccumulator/releases/tag/v0.1.0),
but you can build from source if you want to test if it works in your setup (see [here](https://github.com/andreped/GradientAccumulator#or-from-source-code)).

For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead.

## Install

#### From latest release:
```
pip install https://github.com/andreped/GradientAccumulator/releases/download/v0.1.0/GradientAccumulator-0.1.0-py3-none-any.whl
```

#### Or from source code:
```
pip install git+https://github.com/andreped/GradientAccumulator
```

## Usage

```
from GradientAccumulator.accumulator import GradientAccumulator
from tensorflow.keras.optimizers import Adam

opt = Adam(1e-3)
wrapped_opt = GradientAccumulator(opt, accum_steps=4)
```

Then pass wrapped_opt to `model.compile()` as optimizer, like so:
```
model.compile(optimizer=wrapped_opt, ...)
```

The implementation is derived and adjusted from the discussion at [this](https://github.com/tensorflow/addons/issues/2260#issuecomment-1136967629) TensorFlow Issue.

## TODOs:
- [x] Add generic wrapper class for adding accumulated gradients to any optimizer
- [x] Add CI to build wheel and test that it works across different python versions, TF versions, and operating systems.
- [ ] Add wrapper class for BatchNormalization layer, similar as done for optimizers
- [ ] Test method for memory leaks
- [ ] Verify that implementation works in multi-GPU setups
- [ ] Add benchmarks to verfiy that accumulated gradients actually work as intended

## Disclaimer
Note that this implementation is only compatible with newer versions of TensorFlow. This is because the way Optimizers behave in TF
has changed in TF 2. Slight modifications can likely be made to make this work for older versions, but I would recommend using
newer versions of TF 2 instead, as it has become more stable and feature rich than recent versions.

Also note that this implementation **does not work with TF 1**. For the same reason as it does not work with older TF 2 versions.
However, a TF 1 implementation can be found in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139).

## Tips
Remember to pass the wrapper to the `custom_objects` in `load_model` if you wish to load a trained model. This is only
necessary if you are setting `compile=True` in `load_model`, which is relevant for finetuning or to use `model.evaluate()`.
