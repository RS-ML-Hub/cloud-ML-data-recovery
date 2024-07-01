import tensorflow as tf
import keras
from network.layers import SNConv, Attention_Layer

class SelfAttentionDiscriminator(keras.Model):
    def __init__(self, band_num=1):
        super().__init__()
        self.cnum = 32
        
        self.SAD = keras.Sequential([
            keras.layers.InputLayer(input_shape=(256,256,band_num+1)),
            SNConv(2*self.cnum, 4, 2),
            SNConv(4*self.cnum, 4, 2),
            SNConv(8*self.cnum, 4, 2),
            SNConv(8*self.cnum, 4, 2),
            SNConv(8*self.cnum, 4, 2),
            SNConv(8*self.cnum, 4, 2),
            Attention_Layer(8*self.cnum,'relu'),
            SNConv(8*self.cnum, 4, 2),
            keras.layers.Flatten()
        ]) 
    def call(self,x):
        return self.SAD(x)

class LocalDiscriminator(keras.Model):
    def __init__(self):
        super().__init__()
        cnum = 32
        self.localD= keras.Sequential([
            SNConv(2*cnum, 4, 2),
            SNConv(4*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
        ])

    def call(self,x):
        out = self.localD(x)
        return tf.reshape(out, (out.shape[0], -1))
        
class GlobalDiscriminator(keras.Model):
    def __init__(self):
        super().__init__()
        cnum = 32
        self.globalD = keras.Sequential([
            SNConv(2*cnum, 4, 2),
            SNConv(4*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
        ])

    def call(self,x):
        out = self.globalD(x)
        #reshape to 1D
        return tf.reshape(out, (out.shape[0], -1))

class MultiDiscriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.input_shape = (2048)
        self.localD = LocalDiscriminator()
        self.globalD = GlobalDiscriminator()
        self.act = keras.activations.sigmoid
        self.lin2 = keras.layers.Dense(1024)
        self.lin3 = keras.layers.Dense(1024)


    def call(self, x):

        local_out = self.localD(x)
        global_out = self.globalD(x)
        out = self.act(self.lin2(local_out)+ self.lin3(global_out))
        return self.act(out)
        
#Not sure if MultiDiscriminator is the right way to tackle actual clouds as they are likely "free-form-like" masks
