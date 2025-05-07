# Sentinel-3 Remote Sensing Reflectance Inpainting Model

This repository contains the code written during an internship at Kyoto University of Advanced Science

To anyone finding this repository and wanting to build upon this work, please be aware that balancing losses is tricky and should not be done randomly by tweaking the weights by hand.
A curious reader can explore what happens when you increase the Kullback-Leibler Divergence in a Gaussian VAE (another type of generative models) for instance: image will come out blurry and reconstruction will be very poor.
However, balancing the loss perfectly (notably by deriving the loss by hand, which is an easy task in a Gaussian VAE) leads to decent reconstruction and generation capacity !

## Technical Specificity
This code was built on Tensorflow version 2.16.1 and the Keras API 3.3.3

## Architecture

The architecture is based on the works of  [Context-based multiscale unified network for missing data reconstruction in remote sensing images from Shao et. al](https://ieeexplore.ieee.org/abstract/document/9198931).

It consists of a General Adversial Network in two stages :
 - The first stage is an auto-encoder architecture using Gated Convolutions.
 - The second stage is a similar Encoder Decoder architecture using a Self Attention Layer in its deepest layer.

## Specificities
GANs are known to be rather difficult to make converge, to alleviate this problem, several techniques have been implemented, namely :
 - Gradient Penalty
 - Spectral Regularization instead of Normalization
 - Implementing the Pytorch VGG16-BN model under Tensorflow
 - Channel wise normalization

Some techniques have been tried with little to no results. Notably, we tried modifying the values to which the masked pixels were mapped. No significant impact have been found.

## Trials

Several tweaks have been tried and done on this model.



Using a __Self Attention Discriminator__ does not yield decent results, probably due to a faulty implementation

---

Using a __Wasserstein loss__ : in this setup, the model did not seem to learn anything during its trial

---

Using tensorflow's implementation of the __random_crop method__ :
This leads to issues during the differentiation process, we found that implementing a custom cropping functions gives better results.
Tensorflow's implementation makes the Nested Gradient Tapes of the discriminator penalization fail.

---

Using a base filter number of 64 for both generator and discriminator. This leads to no significant increase in performance and is not necessary. Using 16 for the discriminator seems to work fine.

--- 

Using mixed 16 bits precision, this led to instability at the time of trial. Due to renormalization of data since then, it could be re-tried but might not work properly.

## Metrics

The only quality metrics used that is not a loss is the __SSIM__.
The SSIM outputs a value between 0 and 1 indicating the similarity of one image to the other.
Please note that this metric is inherently flawed by the size of the cloud masks used : The smaller the mask, the more pixels are copied from the original image and as a result the higher the SSIM.
