#Best method to tackle this is to do a GAN network
#This allows us to fill in content and reproduce the "style of the picture" so it is as realistic as possible.

#Same Idea as https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/models/sa_gan.py
#But we will be using a different input type.

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from utils import GatedConv2D, GatedDeConv2D
import keras


class GenModel(keras.Model):
    def __init__(self, band_num=11):
        super.__init__()
        cn_num = 32
        self.coarseNet = keras.Sequential([
            GatedConv2D(cn_num, 5, 1),
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(2*cn_num, 4, 1),
            GatedConv2D(4*cn_num, 3, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1, 2),
            GatedConv2D(4*cn_num, 3, 1, 4),
            GatedConv2D(4*cn_num, 3, 1, 8),
            GatedConv2D(4*cn_num, 3, 1, 16),
            GatedConv2D(4*cn_num, 3, 1),
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 4, 1),
            GatedConv2D(cn_num, 3, 1),
            GatedConv2D(cn_num, 3, 1),
            GatedConv2D(band_num, 3, 1,activation='none'),
        ])