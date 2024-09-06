import keras
import tensorflow as tf


def normalize_with_mask(tensor, mask):
    # Masked min and max calculation
    masked_tensor = tf.where(mask == 0, tensor, tf.fill(tf.shape(tensor), tf.float64.max))
    min_val = tf.reduce_min(masked_tensor, axis=(1,2), keepdims=True)
    
    masked_tensor = tf.where(mask == 0, tensor, tf.fill(tf.shape(tensor), tf.float64.min))
    max_val = tf.reduce_max(masked_tensor, axis=(1,2), keepdims=True)
    
    # Normalize the entire tensor
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized


class SSIM_metric(keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM_metric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')

    def update_state(self, y_true, y_pred, masks, sample_weight=None):
        batch = tf.shape(y_true)[0]
        h = tf.shape(y_true)[1]
        ytn = tf.zeros((batch,h,h,0),dtype=tf.float64)
        ypn = tf.zeros((batch,h,h,0),dtype=tf.float64)
        for i in range(11):
          #ytn_temp = (y_true[:,:,:,i:i+1] - tf.reduce_min(y_true[:,:,:,i:i+1], axis=0, keepdims=True))/(tf.reduce_max(y_true[:,:,:,i:i+1], axis=0, keepdims=True) - tf.reduce_min(y_true[:,:,:,i:i+1], axis=0, keepdims=True))
          #ypn_temp = (y_pred[:,:,:,i:i+1] - tf.reduce_min(y_pred[:,:,:,i:i+1], axis=0, keepdims=True))/(tf.reduce_max(y_pred[:,:,:,i:i+1], axis=0, keepdims=True) - tf.reduce_min(y_pred[:,:,:,i:i+1], axis=0, keepdims=True))
          ytn_temp = normalize_with_mask(y_true[:,:,:,i:i+1], masks) 
          ypn_temp = normalize_with_mask(y_pred[:,:,:,i:i+1], masks)
          
          ytn = tf.concat([ytn,ytn_temp],axis=-1)
          ypn = tf.concat([ypn,ypn_temp],axis=-1)
        
        ssim = tf.image.ssim(ytn, ypn, max_val=1.0)
        self.ssim.assign(tf.reduce_mean(ssim))

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)