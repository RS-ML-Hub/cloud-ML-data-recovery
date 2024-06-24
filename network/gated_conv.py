import tensorflow as tf
import keras

# Implementation of the GatedConv2D and GatedDeConv2D classes as per both 
# https://github.com/Oliiveralien/Inpainting-on-RSI/ and https://github.com/JiahuiYu/generative_inpainting/

class GatedConv2D(keras.Model):
    def __init__(self, cn_out, ker_size=5, stride=1, dilation=1, activation='relu', trainable=True, dtype=tf.float32):
        super(GatedConv2D, self).__init__()
        # TODO determine if bn is needed 
        self.ac = activation
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.ReLU()
        self.conv = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.gate = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation="sigmoid", dilation_rate=dilation, kernel_initializer='he_normal')
    def call(self, input):
        x = self.conv(input)
        gated = self.gate(input)
        if self.ac is None:
            return self.bn(x *gated)
        else:
            return self.bn(self.act(x) * gated)
    


class GatedDeConv2D(keras.Model):
    def __init__(self, scale, cn_out, ker_size, stride, dilation=1, trainable=True, dtype=tf.float32):
        super(GatedDeConv2D, self).__init__()
        self.upsample = keras.layers.UpSampling2D(size=(scale, scale), interpolation='bilinear')
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.LeakyReLU(negative_slope=0.2)
        self.conv = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.gate = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.sigmoid = keras.layers.Activation('sigmoid')   
    
    def call(self, input):
        x = self.upsample(input)
        mask = self.gate(x)
        x = self.conv(x)
        return self.bn(self.act(x) * self.sigmoid(mask))
