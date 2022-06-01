# GradientAccumulator

![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)

### **DISCLAIMER: This is an experimental project - current solution is not stable. Use with caution.**
- Current benchmark produces different results with and without accumulated gradients - weights are likely updated too often using GA.

This repo contains a TensorFlow 2 compatible implementation of accumulated gradients.

Simply wrap the accumulator over any optimizer, and specify `accum_steps` to control number of accumulations.

Precompiled wheel compatible with Python 3.7-3.9 and TensorFlow 2.7-2.9 exist in [Release](https://github.com/andreped/GradientAccumulator/releases/tag/v0.1.0),
but you can build from source if you want to test if it works in your setup (see [here](https://github.com/andreped/GradientAccumulator#or-from-source-code)).

For TF 1, I suggest using the AccumOptimizer implementation in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139) instead.

## Experiments
To perform the benchmark, just run:
```
python benchmark.py
```

You should get the same model performance from using batch_size=64 & accum_steps=1 vs batch_size=8 & accum_steps=4, but that is **not** the case currently! Need to debug the issue further...

To reproduce issue, just run:
```
python benchmark.py --batchsize 64 --accum_steps 1
python benchmark.py --batchsize 8 --accum_steps 4
```

Note that using accumulated gradients, the training runs `accum_steps` more epochs to reach the same number of updates.

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
- [ ] Add proper multi-GPU support

## Disclaimer
Note that this implementation is only compatible with newer versions of TensorFlow. This is because the way Optimizers behave in TF
has changed in TF 2. Slight modifications can likely be made to make this work for older versions, but I would recommend using
newer versions of TF 2 instead, as it has become more stable and feature rich than recent versions.

Also note that this implementation **does not work with TF 1**. For the same reason as it does not work with older TF 2 versions.
However, a TF 1 implementation can be found in the [H2G-Net repository](https://github.com/andreped/H2G-Net/blob/main/src/utils/accum_optimizers.py#L139).

## Tips
Remember to pass the wrapper to the `custom_objects` in `load_model` if you wish to load a trained model. This is only
necessary if you are setting `compile=True` in `load_model`, which is relevant for finetuning or to use `model.evaluate()`.
```
from tensorflow.keras.models import load_model

model = load_model("/path/to/model", compile=True, custom_objects={"GradientAccumulator": GradientAccumulator})
```

## Acknowledgements
This implementation is derived from the work of @fsx950223, @stefan-falk, and others, which is a closed PR https://github.com/tensorflow/addons/pull/2525 to TF-addons. Hence, all credit to them and the people who contributed to the work! Sadly, the proposed implementation was not merged,
as there were some unresolved issues with it, especially regarding multi-GPU training. However, I believe the current implementation is working well
for single-GPU scenarios, which should already be of interest to the community.
