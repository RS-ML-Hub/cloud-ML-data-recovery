#Best method to tackle this is to do a GAN network
#This allows us to fill in content and reproduce the "style of the picture" so it is as realistic as possible.

#Same Idea as https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/models/sa_gan.py
#But we will be using a different input type.

from network.gated_conv import GatedConv2D, GatedDeConv2D
import keras
import tensorflow as tf
from network.loss import lossL1
from network.metrics import SSIM_metric

class Attention_Layer(keras.Layer):
    def __init__(self, cn_num):
        super(Attention_Layer,self).__init__()
        self.query_conv = keras.layers.Conv2D(int(4*cn_num//8), 1, padding="same", kernel_initializer="he_normal")
        self.key_conv = keras.layers.Conv2D(int(4*cn_num//8), 1, padding="same", kernel_initializer="he_normal")
        self.value_conv = keras.layers.Conv2D(4*cn_num, 1, padding="same", kernel_initializer="he_normal")
        self.softmax = keras.layers.Softmax()

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=[1], initializer="ones", trainable=True)
        return super(Attention_Layer,self).build(input_shape)

    def call(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = tf.matmul(proj_query, proj_key, transpose_b=True)
        attention = self.softmax(energy)
        out = tf.matmul(attention, proj_value)
        out = self.gamma*out + x
        return out
    

class GenModel(keras.Model):
    def __init__(self, cn_num=64, band_num=1, dropout=False):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.ssim_metric = SSIM_metric()
        self.coarseNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 4, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1, 2),
            GatedConv2D(4*cn_num, 3, 1, 4),
            GatedConv2D(4*cn_num, 3, 1, 8),
            GatedConv2D(4*cn_num, 3, 1, 16),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1,activation=None),
            ])

        self.refineNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            GatedConv2D(cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1,2),
            GatedConv2D(4*cn_num, 3, 1,4),
            GatedConv2D(4*cn_num, 3, 1,8),
            GatedConv2D(4*cn_num, 3, 1,16),
        ])

        self.refineNet_Attention = keras.Sequential([
            keras.Input(shape=(64,64,4*cn_num)),
            Attention_Layer(cn_num),
        ])

        self.refineNetUpSample = keras.Sequential([
            keras.Input(shape=(64,64,4*cn_num)),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1, activation=None)
        ])

    def call(self, x):
        coarse_out = self.coarseNet(x)
        refine_downsample = self.refineNet(coarse_out)
        refine_attention = self.refineNet_Attention(refine_downsample)
        refine_upsample = self.refineNetUpSample(tf.concat([refine_downsample, refine_attention], axis=-1))
        return coarse_out, refine_upsample


    def train_step(self, data):
        x, y = data
        mask = x[:,:,:,-1]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_pred = tf.cast(y_pred, tf.float64)
            # Compute our own loss
            loss = lossL1(y, y_pred, mask)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def test_step(self, data):
        x,y = data
        y_pred = self(x, training=False)
        mask = x[:,:,:,-1]
        y_pred = tf.cast(y_pred, tf.float64)
        loss = lossL1(y, y_pred, mask)
        self.loss_tracker.update_state(loss)
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric, self.ssim_metric]


def build_model(hp):
    model = GenModel(hp.Int("cn_num", 32, 256, 32))
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model