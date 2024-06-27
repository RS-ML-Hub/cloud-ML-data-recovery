import tensorflow as tf
import keras

def DiscriminatorLoss(pos,neg):
    return tf.reduce_mean(tf.nn.relu(1-pos)) + tf.reduce_mean(tf.nn.relu(1+neg))

def GeneratorLoss(pos,neg):
    return -tf.reduce_mean(neg)



def coarseLoss(y_true, y_pred, mask):
    y_pred = y_pred[:,:,:,0]
    chan1 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(mask[:,:,:])/tf.reduce_mean(mask[:,:,:], keepdims=True))
    chan2 = tf.reduce_mean(tf.abs(y_true[:,:,:] - y_pred[:,:,:])*(1- mask[:,:,:])/(1-tf.reduce_mean(mask[:,:,:], keepdims=True)))
    chan_loss = chan1 + chan2

    return chan_loss

def gram_matrix(input_tensor):
    return tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) / tf.cast(input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3], tf.float32)

def StyleLoss(img, refined_img):
    img = tf.image.resize(img, (224,224))
    img = keras.applications.vgg16.preprocess_input(img)
    refined_img = tf.image.resize(refined_img, (224,224))
    refined_img = keras.applications.vgg16.preprocess_input(refined_img)
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    style_loss = 0

    #Start feature extraction from VGG16 on both images
    feats_img = []
    feats_refined_img = []
    for i, layer in enumerate(vgg.layers):
        feat = layer(img)
        feat_refined = layer(refined_img)
        if i in [0,9,13,17]:
            feats_img.append(feat)
            feats_refined_img.append(feat_refined)
    
    #Calculate style loss
    for i in range(len(feats_img)):
        gram_img = gram_matrix(feats_img[i])
        gram_refined_img = gram_matrix(feats_refined_img[i])
        style_loss += tf.reduce_mean(tf.abs(gram_img - gram_refined_img))

    return style_loss

def PerceptualLoss(img, refined_img):
    img = tf.image.resize(img, (224,224))
    img = keras.applications.vgg16.preprocess_input(img)
    refined_img = tf.image.resize(refined_img, (224,224))
    refined_img = keras.applications.vgg16.preprocess_input(refined_img)
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')

    #Start feature extraction from VGG16 on both images
    feats_img = []
    feats_refined_img = []
    for i, layer in enumerate(vgg.layers):
        feat = layer(img)
        feat_refined = layer(refined_img)
        if i in [0,9,13,17]:
            feats_img.append(feat)
            feats_refined_img.append(feat_refined)
    percloss=0

    #Calculate perceptual loss
    for i in range(len(feats_img)):
        percloss += tf.reduce_mean(tf.abs(feats_img[i] - feats_refined_img[i]))
    
    return percloss