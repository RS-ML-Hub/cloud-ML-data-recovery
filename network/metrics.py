import keras
import tensorflow as tf
import tensorflow_gan as tfgan

class SSIM_metric(keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM_metric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.ssim.assign(tf.reduce_mean(ssim))

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)

class FID_metric(keras.metrics.Metric):
    def __init__(self, name='fid', **kwargs):
        super(FID_metric, self).__init__(name=name, **kwargs)
        self.fid = self.add_weight(name='fid', initializer='zeros')

    @tf.function
    def get_fid_score(real_image, gen_image):
        size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        resized_real_images = tf.image.resize(real_image, [size, size], method=tf.image.ResizeMethod.BILINEAR)
        resized_generated_images = tf.image.resize(gen_image, [size, size], method=tf.image.ResizeMethod.BILINEAR)
        num_inception_images = 1
        num_batches = 20 // num_inception_images
        fid = tfgan.eval.frechet_inception_distance(resized_real_images, resized_generated_images, num_batches=num_batches)
        return fid

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.fid = get_fid_score(y_true, y_pred)

    def result(self):
        return self.fid
    
    def reset_states(self):
        self.fid.assign(0.0)

class Weighted_loss(keras.metrics.Metric):
    def __init__(self, name='weighted_loss', **kwargs):
        super(Weighted_loss, self).__init__(name=name, **kwargs)
        self.weighted_loss = self.add_weight(name='weighted_loss', initializer='zeros')

    def update_state(self, val_loss, ssim, fid, sample_weight=None):
        self.weighted_loss.assign(val_loss*(1/ssim)*fid)

    def result(self):
        return self.weighted_loss
    
    def reset_states(self):
        self.weighted_loss.assign(0.0)