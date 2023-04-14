PyTorch
-------

For PyTorch, I would recommend using
`accelerate <https://pypi.org/project/accelerate/>`_.
HuggingFace :hugs: has a great tutorial on how to use it
`here <https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation>`_.

However, if you wish to use native PyTorch and you are implementing
your own training loop, you could do something like this:

.. code-block:: python

    # batch accumulation parameter
    accum_iter = 4

    # loop through enumaretad batches
    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # passes and weights update
        with torch.set_grad_enabled(True):
            
            # forward pass 
            preds = model(inputs)
            loss  = criterion(preds, labels)

            # scale loss prior to accumulation
            loss = loss / accum_iter

            # backward pass
            loss.backward()

            # weights update and reset gradients
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
