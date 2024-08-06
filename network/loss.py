import tensorflow as tf
import keras

def DiscriminatorLoss(pos,neg):
    return tf.reduce_mean(tf.nn.relu(1-pos)) + tf.reduce_mean(tf.nn.relu(1+neg))

def GeneratorLoss(neg):
    return -0.005*tf.reduce_mean(neg)

def reconstructLoss(y_true, coarse, refined, mask):
    mask = mask[:,:,:,0]
    chan_loss = 0
    mask_vectorized= tf.reshape(mask, [tf.shape(mask)[0],-1])
    mvm = tf.reduce_mean(mask_vectorized, axis=1)
    mvmr = tf.reshape(mvm, [-1,1,1,1])
    coarse1 = tf.reduce_mean(tf.abs(y_true[:,:,:,:] - coarse[:,:,:,:])*(tf.expand_dims(mask[:,:,:],axis=-1))/(mvmr))
    coarse2 = tf.reduce_mean(tf.abs(y_true[:,:,:,:] - coarse[:,:,:,:])*(1- tf.expand_dims(mask[:,:,:],axis=-1))/(1-mvmr))
    refine1 = tf.reduce_mean(tf.abs(y_true[:,:,:,:] - refined[:,:,:,:])*(tf.expand_dims(mask[:,:,:],axis=-1))/(mvmr))
    refine2 = tf.reduce_mean(tf.abs(y_true[:,:,:,:] - refined[:,:,:,:])*(1- tf.expand_dims(mask[:,:,:],axis=-1))/(1-mvmr))
    chan_loss = 3*coarse1+coarse2+3*refine1+refine2
    return chan_loss


def gram_matrix(x):
    gram = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]*tf.shape(x)[3]))
    return tf.matmul(gram, tf.transpose(gram, perm=[0,2,1]))

def StyleLoss(feats_img, feats_refined_img):
    #Calculate style loss
    style_loss = 0
    for i in range(len(feats_img)):
        gram_img = gram_matrix(feats_img[i])
        gram_refined_img = gram_matrix(feats_refined_img[i])
        style_loss += (tf.reduce_mean(tf.abs(gram_img - gram_refined_img)))/tf.cast(tf.shape(feats_img[i])[2]*tf.shape(feats_img[i])[3], tf.float32)
    return 120*style_loss/11

def PerceptualLoss(feats_img, feats_refined_img):

    percloss=0
    #Calculate perceptual loss
    for i in range(len(feats_img)):
        percloss += tf.reduce_mean(tf.abs(feats_img[i] - feats_refined_img[i]))
    return 5*percloss/11