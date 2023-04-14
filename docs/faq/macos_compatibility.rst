macOS compatibility
-------------------

Note that GradientAccumulator is perfectly compatible with macOS, both with
and without GPUs. In order to have GPU support on macOS, you will need to
install the tensorflow-compiled version that is compatible with metal:

.. code-block:: bash

    pip install tensorflow-metal


GradientAccumulator can be used as usually. However, note that there only
exists one tf-metal version, which should be equivalent to TF==2.5.
