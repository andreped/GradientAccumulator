import tensorflow as tf

from gradient_accumulator.layers import AccumBatchNormalization


def replace_batchnorm_layers(model, accum_steps, position="replace"):
    # Auxiliary dictionary to describe the network graph
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict["input_layers_of"]:
                network_dict["input_layers_of"].update(
                    {layer_name: [layer.name]}
                )
            else:
                network_dict["input_layers_of"][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict["new_output_tensor_of"].update(
        {model.layers[0].name: model.input}
    )

    # Iterate over all layers after the input
    model_outputs = []
    iter_ = 0
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            if position == "replace":
                x = layer_input
            else:
                raise ValueError("position must be: replace")

            # build new layer
            new_layer = AccumBatchNormalization(
                accum_steps=accum_steps,
                name="AccumBatchNormalization_" + str(iter_),
            )
            new_layer.build(input_shape=layer.input_shape)

            iter_ += 1

            # set weights in new layer to match old layer
            new_layer.accum_mean = layer.moving_mean
            new_layer.moving_mean = layer.moving_mean

            new_layer.accum_variance = layer.moving_variance
            new_layer.moving_variance = layer.moving_variance

            # forward step
            x = new_layer(x)

        else:
            x = layer(layer_input)

        # Set new output tensor (original one/the one of the inserted layer)
        network_dict["new_output_tensor_of"].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return tf.keras.Model(inputs=model.inputs, outputs=x)
