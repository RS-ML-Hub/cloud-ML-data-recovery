import tensorflow as tf
import keras
from keras.src.layers.input_spec import InputSpec
from keras.src.layers import Wrapper

# Implementation of the GatedConv2D and GatedDeConv2D classes as per both 
# https://github.com/Oliiveralien/Inpainting-on-RSI/ and https://github.com/JiahuiYu/generative_inpainting/

class GatedConv2D(keras.Model):
    def __init__(self, cn_out, ker_size=5, stride=1, dilation=1, activation='relu', trainable=True, dtype=tf.float32):
        super(GatedConv2D, self).__init__()
        # TODO determine if bn is needed 
        self.ac = activation
        self.bn = keras.layers.BatchNormalization()

        self.act = keras.layers.LeakyReLU(alpha=0.2)
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
        self.upsample = keras.layers.UpSampling2D(size=(scale, scale), interpolation='nearest')
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.LeakyReLU(alpha=0.2)
        self.conv = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.gate = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.sigmoid = keras.activations.sigmoid
    
    def call(self, input):
        x = self.upsample(input)
        mask = self.gate(x)
        x = self.conv(x)
        return self.bn(self.act(x) * self.sigmoid(mask))

class SNConv(keras.Model):
    def __init__(self, cn_out, ker_size, stride, dilation=1, activation='relu'):
        super(SNConv, self).__init__()
        self.ac = activation
        self.conv = keras.layers.Conv2D(filters=cn_out, strides=stride, kernel_size=ker_size, padding='same', activation=None, dilation_rate=dilation, kernel_initializer='he_normal')
        self.SN = Spectral_Regularization(self.conv)
        self.activation = keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input):
        x = self.SN(input)
        if self.ac is None:
            return x
        else:
            return self.activation(x)

class Attention_Layer(keras.Model):
    def __init__(self, cn_num, activation='relu'):
        super(Attention_Layer,self).__init__()
        self.c = cn_num
        self.query_conv = keras.layers.Conv2D(int(cn_num//8), 1, padding="same", kernel_initializer="he_normal")
        self.key_conv = keras.layers.Conv2D(int(cn_num//8), 1, padding="same", kernel_initializer="he_normal")
        self.value_conv = keras.layers.Conv2D(cn_num, 1, padding="same", kernel_initializer="he_normal")
        self.softmax = keras.layers.Softmax()

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=[1], initializer="zeros", trainable=True)
        return super(Attention_Layer,self).build(input_shape)

    def call(self, x):
        
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        proj_query = tf.reshape(self.query_conv(x), [batch_size, h*w, int(self.c/8)])
        proj_key = tf.transpose(tf.reshape(self.key_conv(x), [batch_size, h*w, int(self.c/8)]), [0,2,1])
        proj_value = tf.transpose(tf.reshape(self.value_conv(x), [batch_size, h*w, int(self.c)]),[0,2,1])

        energy = tf.matmul(proj_query, proj_key)
        attention = tf.transpose(self.softmax(energy), [0,1,2])
        out = tf.matmul(proj_value, attention)
        out = tf.transpose(out, [2,0,1])
        out = tf.reshape(out,(batch_size, h, w, int(self.c)))
        return self.gamma*out + x

class Spectral_Regularization(Wrapper):
    def __init__(self, layer, **kwargs):
        super(Spectral_Regularization,self).__init__(layer, **kwargs)
        self.layer = layer

    def build(self, input_shape):
        super().build(input_shape)
        self.input_spec = InputSpec(shape=[None] + list(input_shape[1:]))

        self.kernel = self.layer.kernel
        self.kernel_shape = self.kernel.shape
        self.vector_U = self.add_weight(name="vector_u", shape=(1, self.kernel_shape[-1]), initializer=keras.initializers.TruncatedNormal(stddev=0.02), trainable=False, dtype=self.kernel.dtype)
    
    def call(self, inputs, training=False):
        if training:
            new_u, new_w = self.regularized_weights()
            self.vector_U.assign(new_u)
            self.kernel.assign(new_w)

        return self.layer(inputs)
        

    
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    def regularized_weights(self):
        w = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        w_shape = self.kernel_shape
        u = self.vector_U.value
        v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True))
        u = tf.nn.l2_normalize(tf.matmul(v, w))
        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

        #Delta D is sigma minus the identity matrix times the first singular value 
        delta_d = sigma - tf.eye(w_shape[-1])*sigma[0,:]
        delta_w = tf.matmul(tf.matmul(u, delta_d), v, transpose_b=True)
        new_w = (w + delta_w)/sigma
        kernel = tf.reshape(new_w, w_shape)
        return u, kernel