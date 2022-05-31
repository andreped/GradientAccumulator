# GradientAccumulator

![CI](https://github.com/andreped/GradientAccumulator/workflows/CI/badge.svg)

This repo contains a TensorFlow 2.x compatible implementation of accumulated gradients.

Simply wrap the accumulator over any optimizer, and specify `accum_steps` to control number of accumulations.

Precompiled wheel compatible with Python 3.7-3.9 and TensorFlow 2.5-2.9 exist in [Release](https://github.com/andreped/GradientAccumulator/releases/tag/v0.1.0),
but you can build from source if you want to test if it works in your setup (see [here](https://github.com/andreped/GradientAccumulator#or-from-source-code)).

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