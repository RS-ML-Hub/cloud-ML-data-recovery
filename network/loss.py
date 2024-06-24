import tensorflow as tf

def lossL1(y_true, y_pred, mask):
    y_pred = y_pred[:,:,:,0]
    chan1 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(mask[:,:,:])/tf.reduce_mean(mask[:,:,:], keepdims=True))
    chan2 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(1- mask[:,:,:])/(1-tf.reduce_mean(mask[:,:,:], keepdims=True)))
    chan_loss = chan1 + chan2

    return chan_loss
