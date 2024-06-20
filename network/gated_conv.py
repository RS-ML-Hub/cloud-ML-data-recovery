
import tensorflow as tf
import keras

#Implementation of the GatedConv2D and GatedDeConv2D classes as per both 
#https://github.com/Oliiveralien/Inpainting-on-RSI/ and https://github.com/JiahuiYu/generative_inpainting/

class GatedConv2D(keras.Model):
    def __init__(self, cn_out, ker_size=5, stride=1, dilation=1, activation='relu'):
        super(GatedConv2D,self).__init__()
        #TODO determine if bn is needed 
        self.ac=activation
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation(activation)
        self.conv = keras.layers.Conv2D(filters=cn_out,strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation)
        self.gate = keras.layers.Conv2D(filters=cn_out,strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation)
        self.sigmoid = keras.layers.Activation('sigmoid')
    def call(self, input):
        x = self.conv(input)
        gated = self.gate(input)
        if self.ac is None:
            return self.bn(x * self.sigmoid(gated))
        else:
            return self.bn(self.act(x) * self.sigmoid(gated))
        


class GatedDeConv2D(keras.Model):
    def __init__(self, scale, cn_out, ker_size, stride, dilation=1):
        super(GatedDeConv2D,self).__init__()
        self.upsample =  keras.layers.UpSampling2D(size=(scale,scale), interpolation='nearest')
        self.conv = GatedConv2D(cn_out, ker_size, stride, dilation, activation=keras.layers.LeakyReLU(0.2))

    def call(self, input):
        x = self.upsample(input)
        return self.conv(x)