import tensorflow as tf
import keras
from network.layers import SNConv, Attention_Layer
import numpy as np

def random_crop(image, crop_height, crop_width):
    """Randomly crops an image to the specified height and width."""
    batch_size, img_height, img_width, channels = image.shape[0],image.shape[1],image.shape[2],image.shape[3]
    
    crop_y = np.random.randint(0, img_height - crop_height -1)
    crop_x = np.random.randint(0, img_width - crop_width -1)

    return image[:, crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :]
    
class SelfAttentionDiscriminator(keras.Model):
    def __init__(self, band_num=11, cn_num=32):
        super().__init__()
        self.cnum = cn_num
        
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
    def __init__(self, crop_size=64, band_num=11):
        super().__init__()
        cnum = 16
        self.localD= keras.Sequential([
            keras.layers.InputLayer(input_shape=(crop_size,crop_size,band_num+1)),
            SNConv(2*cnum, 4, 2),
            SNConv(4*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
            SNConv(8*cnum, 4, 2),
        ])

    def call(self,x):
        x_view = self.localD(x)
        return tf.reshape(x_view,[tf.shape(x)[0],-1])
        
class GlobalDiscriminator(keras.Model):
    def __init__(self):
        super().__init__()
        cnum = 16
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
        x_view = self.globalD(x)
        return tf.reshape(x_view,[tf.shape(x)[0],-1])


class MultiDiscriminator(keras.Model):
    def __init__(self, crop_size=128):
        super().__init__()

        self.crop_size = crop_size
        
        self.localD = LocalDiscriminator(crop_size=crop_size)
        self.globalD = GlobalDiscriminator()
        self.act = keras.activations.tanh
        self.lin1 = keras.layers.Dense(1, activation=None, kernel_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/2048),maxval=np.sqrt(1/2048)),bias_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/2048),maxval=np.sqrt(1/2048)))
        #self.lin2 = keras.layers.Dense(1, activation=None)
        #self.lin3 = keras.layers.Dense(1, activation=None)


    def call(self, x):
        cropped_x = random_crop(x, self.crop_size, self.crop_size)
        #cropped_x = tf.image.random_crop(x, size=[tf.shape(x)[0],self.crop_size, self.crop_size, tf.shape(x)[3]], seed=4862)
        local_out = self.localD(cropped_x)
        global_out = self.globalD(x)
        out = self.lin1(tf.concat([local_out,global_out], axis=-1))
        #out = self.act(self.lin2(local_out)) + self.act(self.lin3(global_out))
        return self.act(out)
        
#Not sure if MultiDiscriminator is the right way to tackle actual clouds as they are likely "free-form-like" masks