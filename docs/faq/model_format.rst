Model format
------------

It is recommended to use the SavedModel format when using this implementation.
That is because the HDF5 format is only compatible with `TF <= 2.6` when using
the model wrapper. However, if you are using older TF versions, both formats
work out-of-the-box. The SavedModel format works fine for all versions of TF 2.x.
