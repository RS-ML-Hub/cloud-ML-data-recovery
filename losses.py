import tensorflow as tf

def lossL1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))