#Best method to tackle this is to do a GAN network
#This allows us to fill in content and reproduce the "style of the picture" so it is as realistic as possible.

#Same Idea as https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/models/sa_gan.py
#But we will be using a different input type.

from network.layers import GatedConv2D, GatedDeConv2D, Attention_Layer
import keras
from numpy import float32
import tensorflow as tf
from network.loss import DiscriminatorLoss, GeneratorLoss, StyleLoss, PerceptualLoss, reconstructLoss
from network.metrics import SSIM_metric
from network.discriminator import SelfAttentionDiscriminator, MultiDiscriminator
from network.vgg_tf import VGG_fromTorch

"""
class VGG_Feature_Extractor(keras.Model):
    def __init__(self, band_num=11):
        super().__init__()
        self.vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        self.extractor = keras.Model(inputs=self.vgg.input, outputs=[self.vgg.layers[i].output for i in [0, 9, 13, 17]])

    def call(self, x):
        img = tf.image.resize(x, (224, 224))
        band = keras.applications.vgg16.preprocess_input(tf.image.grayscale_to_rgb(tf.expand_dims(img, axis=-1)))
        return self.extractor(band)
"""



class GenModel(keras.Model):
    def __init__(self, cn_num=32, band_num=11, dropout=False):
        super().__init__()
        self.coarseNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            #DownSample to 128*128*64
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            #Downsample to 64*64*128
            GatedConv2D(4*cn_num, 4, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            #atrous
            GatedConv2D(4*cn_num, 3, 1, 2),
            GatedConv2D(4*cn_num, 3, 1, 4),
            GatedConv2D(4*cn_num, 3, 1, 8),
            GatedConv2D(4*cn_num, 3, 1, 16),

            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            #upsample
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1,activation=None),
            ])

        self.refineNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            #Downsample to 128*128*64
            GatedConv2D(cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            #Downsample to 64*64*128
            GatedConv2D(2*cn_num, 4, 2),

            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            #atrous
            GatedConv2D(4*cn_num, 3, 1,2),
            GatedConv2D(4*cn_num, 3, 1,4),
            GatedConv2D(4*cn_num, 3, 1,8),
            GatedConv2D(4*cn_num, 3, 1,16),
        ])

        self.refineNet_Attention = keras.Sequential([
            keras.Input(shape=(64,64,4*cn_num)),
            Attention_Layer(4*cn_num),
        ])

        self.refineNetUpSample = keras.Sequential([
            keras.Input(shape=(64,64,4*cn_num)),
            
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            #upsample to 128*128*64
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            #upsample to 256*256*32
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1, activation=None)
        ])

    def call(self, x):
        coarse = self.coarseNet(x)
        coarse = tf.clip_by_value(coarse, -1, 1)
        mask = tf.expand_dims(x[:,:,:,-1], axis=-1)
        complete_coarse = coarse*mask + x[:,:,:,:-1]*(1-mask)
        coarse_out = tf.concat([complete_coarse, tf.expand_dims(x[:,:,:,-1], axis=-1)], axis=-1)
        refine_downsample = self.refineNet(coarse_out)
        refine_attention = self.refineNet_Attention(refine_downsample)
        refine_upsample = self.refineNetUpSample(refine_attention)
        batch = tf.shape(coarse)[0]
        h = tf.shape(coarse)[1]
        w = tf.shape(coarse)[2]
        refine_upsample_out = tf.zeros((batch,h,w,0))
        for i in range(11):
          refine_upsample_temp = (refine_upsample[:,:,:,i:i+1] - tf.reduce_mean(refine_upsample[:,:,:,i:i+1],axis=0, keepdims=True))/(2*tf.math.reduce_std(refine_upsample[:,:,:,i:i+1],axis=0, keepdims=True))
          refine_upsample_out = tf.concat([refine_upsample_out, refine_upsample_temp],axis=-1)
        refine_upsample = tf.clip_by_value(refine_upsample_out, -1, 1)
        return coarse, refine_upsample

    """ 
    def train_step(self, data):
        x, y = data
        mask = x[:,:,:,-1]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_pred = tf.cast(y_pred, tf.float64)
            # Compute our own loss
            coarse_loss = coarseLoss(y, y_pred, mask)
            style_loss= StyleLoss(y, y_pred)
            perceptual_loss = PerceptualLoss(y, y_pred)
            gen_loss = GeneratorLoss(neg)
            dis_loss = DiscriminatorLoss(pos,neg)
            total_loss = coarse_loss + style_loss + perceptual_loss + gen_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
    
    def test_step(self, data):
        x,y = data
        y_pred = self(x, training=False)
        mask = x[:,:,:,-1]
        y_pred = tf.cast(y_pred, tf.float64)
        loss = coarseLoss(y, y_pred, mask)
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
    """

class SAGAN(keras.Model):
    def __init__(self,cn_num=32):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruct_metric = keras.metrics.Mean(name="reconstruct_loss")
        self.style_metric = keras.metrics.Mean(name="style_loss")
        self.perceptual_metric = keras.metrics.Mean(name="perceptual_loss")
        self.gen_metric = keras.metrics.Mean(name="gen_loss")
        self.dis_metric = keras.metrics.Mean(name="dis_loss")
        self.ssim_metric = SSIM_metric()
        self.generator = GenModel(cn_num)
        self.discriminator = MultiDiscriminator(128)
        self.vgg = VGG_fromTorch()
        self.vgg.net.load_weights("./cloud_inpainting/network/vgg16_bn.h5")
        self.extractor = self.vgg.extractor
        self.gp_center=1
        self.gam_reg=2

    def compile(self,strategy, gen_optimizer, dis_optimizer):
        super().compile()
        if strategy is not None:
            with strategy.scope():
                self.generator.compile(optimizer=gen_optimizer, loss="mse")
                self.discriminator.compile(optimizer=dis_optimizer, loss="mse")
        else:
            self.generator.compile(optimizer=gen_optimizer, loss="mse")
            self.discriminator.compile(optimizer=dis_optimizer, loss="mse")
    
    def train_step(self, data):
        masks, y = data
        y = tf.cast(y, tf.float64)
        masks= tf.cast(masks, tf.float64)
        batch = tf.shape(y)[0]
        h = tf.shape(y)[1]
        w = tf.shape(y)[2]

        x = tf.zeros(shape=(batch,h,w,0), dtype=tf.float64)
        for i in range(11):
          x = tf.concat([x,tf.expand_dims(y[:,:,:,i] * (1 - masks[:,:,:,0]) + masks[:,:,:,0], axis=-1)], axis=-1)
        x = tf.concat([x, masks], axis=-1)

        with tf.GradientTape() as dis_tape:
          coarse, refined = self.generator(x, training=False)
          coarse = tf.cast(coarse, tf.float64)
          refined = tf.cast(refined, tf.float64)            
          epsilon = tf.random.uniform([batch, 1, 1, 1], 0.0, 1.0, dtype=tf.float64)
          complete = masks * refined + (1 - masks) * y
          pos = tf.concat([y, masks], axis=-1)
          neg = tf.concat([complete, masks], axis=-1)
          x_hat = epsilon * pos + (1 - epsilon) * neg
          with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(x_hat)  
            d_hat = self.discriminator(x_hat, training = False)
          gradients = t.gradient(d_hat, x_hat)
          ddx = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
          d_regularizer = tf.reduce_mean(tf.square(ddx-self.gp_center))
          
          pos_neg = tf.concat([pos, neg], axis=0)
          pos_neg_pred = self.discriminator(pos_neg, training=True)
          pred_pos, pred_neg = tf.split(pos_neg_pred, 2, axis=0)
          dis_loss = tf.cast(DiscriminatorLoss(pred_pos, pred_neg),tf.float64)
          dis_loss = dis_loss +self.gam_reg*d_regularizer
        # Compute gradients for the discriminator
        dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))
        #del gradients, dis_gradients
        with tf.GradientTape() as gen_tape:
            coarse, refined = self.generator(x, training=True)
            coarse = tf.cast(coarse, tf.float64)
            refined = tf.cast(refined, tf.float64)
            complete = masks * refined + (1 - masks) * y
            
            neg = tf.concat([complete, masks], axis=-1)
            pred_neg = self.discriminator(neg, training=False)
            gen_loss = tf.cast(GeneratorLoss(pred_neg),tf.float64)

            reconstruct_loss = tf.cast(reconstructLoss(y, coarse, refined, masks),tf.float64)
            
            style_loss = tf.constant(0.0, dtype=tf.float64)
            perceptual_loss = tf.constant(0.0, dtype=tf.float64)
            for i in range(11):
                y_norm = (y[:,:,:,i] - tf.reduce_min(y[:,:,:,i],axis=0))/((tf.reduce_max(y[:,:,:,i],axis=0)-tf.reduce_min(y[:,:,:,i],axis=0))+1e-6)
                feats_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(y_norm,axis=-1),(224,224))))
                refined_norm = (refined[:,:,:,i] - tf.reduce_min(refined[:,:,:,i],axis=0))/((tf.reduce_max(refined[:,:,:,i],axis=0)-tf.reduce_min(refined[:,:,:,i],axis=0))+1e-6)
                feats_refined_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(refined_norm,axis=-1),(224,224))))
                complete_norm = (complete[:,:,:,i] - tf.reduce_min(complete[:,:,:,i], axis=0))/((tf.reduce_max(complete[:,:,:,i],axis=0)-tf.reduce_min(complete[:,:,:,i],axis=0))+1e-6)
                feats_complete_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(complete_norm,axis=-1),(224,224))))
                style_loss = tf.math.add(style_loss,tf.math.add(tf.cast(StyleLoss(feats_img, feats_refined_img), tf.float64), tf.cast(StyleLoss(feats_img, feats_complete_img), tf.float64)))
                perceptual_loss = tf.math.add(perceptual_loss, tf.math.add(tf.cast(PerceptualLoss(feats_img, feats_refined_img), tf.float64), tf.cast(PerceptualLoss(feats_img, feats_complete_img), tf.float64)))            
            total_loss = tf.math.add(reconstruct_loss, tf.math.add(tf.math.scalar_mul(0.001, tf.math.add(style_loss, perceptual_loss)), gen_loss))
        # Compute gradients for the generator
        gen_gradients = gen_tape.gradient(total_loss, self.generator.trainable_variables)
        del refined
        del pred_neg
        del pred_pos
        del coarse
        # Apply gradients to the generator
        self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        # Compute our own metrics

        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    
    def test_step(self, data):
        masks, y = data
        masks= tf.cast(masks, tf.float64)
        y = tf.cast(y, tf.float64)
        batch = tf.shape(y)[0]
        h = tf.shape(y)[1]
        w = tf.shape(y)[2]

        x = tf.zeros(shape=(batch,h,w,0), dtype=tf.float64)
        for i in range(11):
          x = tf.concat([x,tf.expand_dims(y[:,:,:,i] * (1 - masks[:,:,:,0]) + masks[:,:,:,0], axis=-1)], axis=-1)

        x = tf.concat([x, masks], axis=-1)
        coarse, refined = self.generator(x, training=False)
        coarse = tf.cast(coarse, tf.float64)
        refined = tf.cast(refined, tf.float64)
        #refined = (refined - tf.reduce_min(refined, axis=[0,3], keepdims=True))/(tf.reduce_max(refined, axis=[0,3], keepdims=True)-tf.reduce_min(refined, axis=[0,3], keepdims=True))
        #coarse = (coarse - tf.reduce_min(coarse, axis=[0,3], keepdims=True))/(tf.reduce_max(coarse, axis=[0,3], keepdims=True)-tf.reduce_min(coarse, axis=[0,3], keepdims=True))
        complete = masks*refined + (1-masks)*y
        reconstruct_loss = tf.cast(reconstructLoss(y, coarse, refined, masks),tf.float64)
        del coarse
        
        pos = tf.concat([y,masks], axis=-1)
        neg = tf.concat([complete,masks], axis=-1)
        pos_neg = tf.concat([pos,neg], axis=0)

        pos_neg_pred = self.discriminator(pos_neg, training=False)

        pred_pos, pred_neg = tf.split(pos_neg_pred, 2, axis=0)
        dis_loss = tf.cast(DiscriminatorLoss(pred_pos, pred_neg), tf.float64)
        del pred_pos
        gen_loss = tf.cast(GeneratorLoss(pred_neg),tf.float64)

        #print("dis_loss", dis_loss)
        
        #img = tf.image.resize(y[:,:,:,3:6], (224,224))
        #img = keras.applications.vgg16.preprocess_input(img)
        #refined_img = tf.image.resize(refined[:,:,:,3:6], (224,224))
        #refined_img = keras.applications.vgg16.preprocess_input(refined_img)
        #complete_img = tf.image.resize(complete[:,:,:,3:6], (224,224))
        #complete_img = keras.applications.vgg16.preprocess_input(complete_img)
        #print("img", img.shape)
        #print("refined_img", refined_img.shape)


        style_loss =tf.constant(0.0, dtype=tf.float64)
        perceptual_loss = tf.constant(0.0, dtype=tf.float64)
        for i in range(11):                
          y_norm = (y[:,:,:,i] - tf.reduce_min(y[:,:,:,i],axis=0))/((tf.reduce_max(y[:,:,:,i],axis=0)-tf.reduce_min(y[:,:,:,i],axis=0))+1e-6)
          feats_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(y_norm,axis=-1),(224,224))))
          refined_norm = (refined[:,:,:,i] - tf.reduce_min(refined[:,:,:,i],axis=0))/((tf.reduce_max(refined[:,:,:,i],axis=0)-tf.reduce_min(refined[:,:,:,i],axis=0))+1e-6)
          feats_refined_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(refined_norm,axis=-1),(224,224))))
          complete_norm = (complete[:,:,:,i] - tf.reduce_min(complete[:,:,:,i], axis=0))/((tf.reduce_max(complete[:,:,:,i],axis=0)-tf.reduce_min(complete[:,:,:,i],axis=0))+1e-6)
          feats_complete_img = self.extractor(tf.image.grayscale_to_rgb(tf.image.resize(tf.expand_dims(complete_norm,axis=-1),(224,224))))
          style_loss = tf.math.add(style_loss,tf.math.add(tf.cast(StyleLoss(feats_img, feats_refined_img), tf.float64), tf.cast(StyleLoss(feats_img, feats_complete_img), tf.float64)))
          perceptual_loss = tf.math.add(perceptual_loss, tf.math.add(tf.cast(PerceptualLoss(feats_img, feats_refined_img), tf.float64), tf.cast(PerceptualLoss(feats_img, feats_complete_img), tf.float64)))            
        total_loss = tf.math.add(reconstruct_loss, tf.math.add(tf.math.scalar_mul(0.001, tf.math.add(style_loss, perceptual_loss)), gen_loss))
        del refined
        del pred_neg


        # Store our own metrics
        self.loss_tracker.update_state(total_loss)
        self.reconstruct_metric.update_state(reconstruct_loss)
        self.style_metric.update_state(style_loss)
        self.perceptual_metric.update_state(perceptual_loss)
        self.gen_metric.update_state(gen_loss)
        self.dis_metric.update_state(dis_loss)
        self.ssim_metric.update_state((y+1)/2, (complete+1)/2)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.reconstruct_metric, self.style_metric, self.perceptual_metric, self.gen_metric, self.dis_metric , self.ssim_metric]

    def call(self, x):
        return self.generator(x, training=False)