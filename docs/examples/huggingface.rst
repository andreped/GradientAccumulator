HuggingFace
===========

Note that HuggingFace provides a variety of different pretrained models. However, it was observed that when loading these models into TensorFlow, the computational graph may not be set up correctly, such that the `model.input` and `model.output` exist.

To fix this, we basically wrap the model into a new `tf.keras.Model`, but define the inputs and outputs ourselves:


.. code-block:: python

    from gradient_accumulator import GradientAccumulateModel
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from transformers import TFx

    #load your model checkpoint
    HF_model = TFx.from_pretrained(checkpoint)

    # define model inputs and outputs -> for different models, different inputs/outputs need to be defined
    input_ids = tf.keras.Input(shape=(None,), dtype='int32', name="input_ids")
    attention_mask = tf.keras.Input(shape=(None,), dtype='int32', name="attention_mask")
    model_input={'input_ids': input_ids, 'attention_mask': attention_mask}

    #create a new Model which has model.input and model.output properties
    new_model = Model(inputs=model_input, outputs=HF_model(model_input))

    #create the GA model
    model = GradientAccumulateModel(accum_steps=4, inputs=new_model.input, outputs=new_model.output)


For more details, see `this <https://github.com/andreped/GradientAccumulator/blob/main/notebooks/GA_for_HuggingFace_TF_models.ipynb>`_ jupyter notebook.
