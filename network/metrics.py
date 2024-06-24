import keras
import tensorflow as tf

class SSIM_metric(keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM_metric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(y_true.shape)
        print(y_pred.shape)
        y_true = tf.expand_dims(y_true, axis=0)
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=0)
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.ssim.assign(tf.reduce_mean(ssim))

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)