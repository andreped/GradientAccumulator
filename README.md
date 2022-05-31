# GradientAccumulator

This repo contains a TensorFlow 2.x compatible implementation of accumulated gradients.

Simply wrap the accumulator over any optimizer.

Example:

```
from accumulator import GradientAccumulator
from tensorflow.keras.optimizers import Adam

opt = Adam(1e-3)
wrapped_opt = GradientAccumulator(opt, accum_steps=4)
```

Then pass wrapped_opt to `model.compile()` as optimizer, like so:
```
model.compile(optimizer=wrapped_opt, ...)
```