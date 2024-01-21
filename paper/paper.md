---
title: 'GradientAccumulator: Efficient and seamless gradient accumulation for TensorFlow'
tags:
  - Python
  - TensorFlow
  - Deep Learning
  - Gradient Descent
authors:
  - name: André Pedersen
    orcid: 0000-0002-3637-953X
    affiliation: 1
    corresponding: true
  - name: Tor-Arne S. Nordmo
    orcid: 0000-0002-2843-9643
    affiliation: 4
  - name: David Bouget
    orcid: 0000-0002-5669-9514
    affiliation: 2
affiliations:
 - name: Department of Clinical and Molecular Medicine, Norwegian University of Science and Technology (NTNU), Trondheim, Norway
   index: 1
 - name: Department of Health Research, SINTEF, Trondheim, Norway
   index: 2
 - name: Application Solutions, Sopra Steria, Trondheim, Norway
   index: 3
 - name: Department of Computer Science, UiT: The Arctic University of Norway, Tromsø, Norway
   index: 4
date: 22 January 2024
bibliography: paper.bib
---

# Summary 

Deep neural networks (DNNs) are the current state-of-the-art for various tasks, such as image recognition and natural language processing. For image analysis tasks, convolutional neural networks (CNNs) of vision transformers (ViTs) are generally preferred. However, training these DNNs are generally a complex task, and may require vast amounts of GPU memory and resources due to the complex nature of these networks. DNNs are commonly trained using gradient descent-like optimization techniques, where backpropagation is used to update the network's weight iteratively over time. Instead of learning by one example at a time, it is common to update the weights by propagating a set of samples through the network, where the set is commonly referred to as a batch. Increasing the batch size, can is many cases lead to better generalization, as the network's weights are updated based on the ensemble of more examples. However, increasing the batch size results in higher GPU memory usage.


A popular approach to counter this problem, is to utilize technique called gradient accumulation (GA). GA enables to theoretically increase the GPU memory use


# Statement of need 

Despite existing open-source implementation of gradient accumulation in various frameworks, TensorFlow [@tensorflow2015abadi] has yet to offer a seemless and easy to use solution for gradient accumulation. To the best of our knowledge, `GradientAccumulator` is the first open-source Python package to enable gradient accumulation to be seamlessly added to TensorFlow model training pipelines. 


GradientAccumulator has already been used in several research studies [@pedersen2023h2gnet; @helland2023postopglioblastoma].

  
# Implementation 

`GradientAccumulator` implements two main approaches to add gradient accumulation support to an existing TensorFlow model. GA support can either be added through model or optimizer wrapping. By wrapping the model, the `train_step` of a given Keras [@chollet2015keras] model is updated such that the gradients are updated only after a user-defined number of backward steps. Wrapping the optimizer works somewhat similar, but this update control is handled directly in the optimizer itself. This is done in such a way that _any_ optimizer can be used with this approach.


More details and tutorials on getting started with the `GradientAccumulator` package, can be found in the `GradientAccumulator `\href{(https://gradientaccumulator.readthedocs.io/}{documentation}.


# Acknowledgements

This work was supported by The Liaison Committee for Education, Research and Innovation in Central Norway [Grant Number 2018/42794]; The Joint Research Committee between St. Olavs Hospital and the Faculty of Medicine and Health Sciences, NTNU (FFU) [Grant Number 2019/38882]; and The Cancer Foundation, St. Olavs Hospital, Trondheim University Hospital [Grant Number 13/2021].


# References
