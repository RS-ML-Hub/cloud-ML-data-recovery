import tensorflow as tf

def lossL1(y_true, y_pred):
    for chan in range(11):
        chan_loss = tf.reduce_mean(tf.abs(y_true[:,:,:,chan] - y_pred[:,:,:,chan]))
        if chan == 0:
            loss = chan_loss
        else:
            loss = loss + chan_loss
    return loss
