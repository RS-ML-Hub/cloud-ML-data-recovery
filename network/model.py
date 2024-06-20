#Best method to tackle this is to do a GAN network
#This allows us to fill in content and reproduce the "style of the picture" so it is as realistic as possible.

#Same Idea as https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/models/sa_gan.py
#But we will be using a different input type.

from network.gated_conv import GatedConv2D, GatedDeConv2D
import keras


class GenModel(keras.Model):
    def __init__(self, band_num=11):
        super().__init__()
        cn_num = 32
        self.coarseNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num)),
            GatedConv2D(cn_num, 5, 1),
            
        ])