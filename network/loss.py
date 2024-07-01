import tensorflow as tf
import keras

def DiscriminatorLoss(pos,neg):
    return tf.reduce_mean(tf.nn.relu(1-pos)) + tf.reduce_mean(tf.nn.relu(1+neg))

def GeneratorLoss(neg):
    return -tf.reduce_mean(neg)



def coarseLoss(y_true, y_pred, mask):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    mask = mask[:,:,:,0]
    chan1 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(mask[:,:,:])/tf.reduce_mean(tf.reshape(mask[:,:,:], (-1,1,1))))
    chan2 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(1- mask[:,:,:])/(1-tf.reduce_mean(tf.reshape(mask[:,:,:], (-1,1,1)))))
    chan_loss = chan1 + chan2
    return chan_loss

def reconstructLoss(y_true, y_pred,mask):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    mask = mask[:,:,:,0]
    chan1 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(mask[:,:,:])/tf.reduce_mean(tf.reshape(mask[:,:,:], (-1,1,1))))
    chan2 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(1- mask[:,:,:])/(1-tf.reduce_mean(tf.reshape(mask[:,:,:],(-1,1,1)))))
    chan_loss = chan1 + chan2
    return chan_loss


def gram_matrix(input_tensor):
    return tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) / tf.cast(input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3], tf.float32)

def StyleLoss(feats_img, feats_refined_img):
    #Calculate style loss
    style_loss = 0
    for i in range(len(feats_img)):
        gram_img = gram_matrix(feats_img[i])
        gram_refined_img = gram_matrix(feats_refined_img[i])
        style_loss += tf.reduce_mean(tf.abs(gram_img - gram_refined_img))
    return style_loss

def PerceptualLoss(feats_img, feats_refined_img):

    percloss=0
    #Calculate perceptual loss
    for i in range(len(feats_img)):
        percloss += tf.reduce_mean(tf.abs(feats_img[i] - feats_refined_img[i]))
    return percloss