#Best method to tackle this is to do a GAN network
#This allows us to fill in content and reproduce the "style of the picture" so it is as realistic as possible.

#Same Idea as https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/models/sa_gan.py
#But we will be using a different input type.

from network.layers import GatedConv2D, GatedDeConv2D, Attention_Layer
import keras
import tensorflow as tf
from network.loss import coarseLoss, DiscriminatorLoss, GeneratorLoss, StyleLoss, PerceptualLoss, reconstructLoss
from network.metrics import SSIM_metric
from network.discriminator import SelfAttentionDiscriminator

class GenModel(keras.Model):
    def __init__(self, cn_num=32, band_num=1, dropout=False):
        super().__init__()
        self.coarseNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 4, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1, 2),
            GatedConv2D(4*cn_num, 3, 1, 4),
            GatedConv2D(4*cn_num, 3, 1, 8),
            GatedConv2D(4*cn_num, 3, 1, 16),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1,activation=None),
            ])

        self.refineNet = keras.Sequential([
            keras.Input(shape=(256, 256, band_num+1)),
            GatedConv2D(cn_num, 5, 1),
            GatedConv2D(cn_num, 4, 2),
            GatedConv2D(2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 4, 2),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
            GatedConv2D(4*cn_num, 3, 1),
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
            GatedDeConv2D(2, 2*cn_num, 3, 1),
            GatedConv2D(2*cn_num, 3, 1),
            GatedDeConv2D(2, cn_num, 3, 1),
            GatedConv2D(cn_num, 3, 1),
            GatedConv2D(int(cn_num/2), 3, 1),
            GatedConv2D(band_num, 3, 1, activation=None)
        ])

    def call(self, x):
        coarse_out = self.coarseNet(x)
        coarse_out = tf.concat([coarse_out, tf.expand_dims(x[:,:,:,-1], axis=-1)], axis=-1)
        refine_downsample = self.refineNet(coarse_out)
        refine_attention = self.refineNet_Attention(refine_downsample)
        refine_upsample = self.refineNetUpSample(refine_attention)
        return coarse_out, refine_upsample

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
    def __init__(self):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.coarse_metric = keras.metrics.Mean(name="coarse_loss")
        self.refine_metric = keras.metrics.Mean(name="refine_loss")
        self.style_metric = keras.metrics.Mean(name="style_loss")
        self.perceptual_metric = keras.metrics.Mean(name="perceptual_loss")
        self.gen_metric = keras.metrics.Mean(name="gen_loss")
        self.dis_metric = keras.metrics.Mean(name="dis_loss")
        self.ssim_metric = SSIM_metric()
        self.generator = GenModel()
        self.discriminator = SelfAttentionDiscriminator()
        self.vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
        self.extractor = keras.Model(inputs=self.vgg.inputs, outputs=[self.vgg.layers[i].output for i in [0,9,13,17]], trainable=False)
        
    def compile(self,strategy, gen_optimizer, dis_optimizer):
        super().compile()
        self.generator.compile(optimizer=gen_optimizer, loss="mse")
        self.discriminator.compile(optimizer=dis_optimizer, loss="mse")

    def train_step(self, data):
        x, y = data
        y = tf.expand_dims(y, axis=-1)
        mask = tf.expand_dims(x[:, :, :, -1], axis=-1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            coarse, refined = self.generator(x, training=True)
            coarse = tf.cast(coarse, tf.float64)
            refined = tf.cast(refined, tf.float64)

            complete = mask * refined + (1 - mask) * y

            pos = tf.concat([y, mask], axis=-1)
            neg = tf.concat([complete, mask], axis=-1)
            pos_neg = tf.concat([pos, neg], axis=0)

            pos_neg_pred = self.discriminator(pos_neg, training=True)

            pred_pos, pred_neg = tf.split(pos_neg_pred, 2, axis=0)

            dis_loss = tf.cast(DiscriminatorLoss(pred_pos, pred_neg), tf.float64)
            
            img = tf.image.resize(y, (224,224))
            img = keras.applications.vgg16.preprocess_input(tf.image.grayscale_to_rgb(img))
            refined_img = tf.image.resize(refined, (224,224))
            refined_img = keras.applications.vgg16.preprocess_input(tf.image.grayscale_to_rgb(refined_img))
            feats_img = self.extractor(img)
            feats_refined_img = self.extractor(refined_img)

            coarse_loss = coarseLoss(y, coarse, mask)
            refined_loss = reconstructLoss(y, refined, mask)
            style_loss = tf.cast(StyleLoss(feats_img, feats_refined_img), tf.float64)
            perceptual_loss = tf.cast(PerceptualLoss(feats_img, feats_refined_img), tf.float64)
            gen_loss = GeneratorLoss(neg)
            total_loss = tf.math.add(tf.math.scalar_mul(1.2, tf.math.add(coarse_loss, refined_loss)), tf.math.add(tf.math.scalar_mul(0.001, tf.math.add(tf.math.scalar_mul(120, style_loss), tf.math.scalar_mul(5, perceptual_loss))), tf.math.scalar_mul(0.005, gen_loss)))

        # Compute gradients
        dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        gen_gradients = gen_tape.gradient(total_loss, self.generator.trainable_variables)

        # Apply gradients
        self.discriminator.optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))
        self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x,y = data
        y_pred = self.generator(x, training=False)
        mask = x[:,:,:,-1]
        y_pred = tf.cast(y_pred, tf.float64)
        coarse, refined = self.generator(x, training=True)
        coarse = tf.cast(coarse, tf.float64)
        refined = tf.cast(refined, tf.float64)

        complete = mask*refined + (1-mask)*y


        pos = tf.concat([y,mask], axis=-1)
        neg = tf.concat([complete,mask], axis=-1)
        pos_neg = tf.concat([pos,neg], axis=0)

        pos_neg_pred = self.discriminator(pos_neg)

        pred_pos, pred_neg = tf.split(pos_neg_pred, 2, axis=0)

        dis_loss = DiscriminatorLoss(pred_pos, pred_neg)
        coarse_loss = coarseLoss(y, coarse, mask)
        refined_loss = reconstructLoss(y, refined, mask)
        style_loss= StyleLoss(y, refined)
        perceptual_loss = PerceptualLoss(y, refined)
        gen_loss = GeneratorLoss(neg)


        total_loss = 1.2*(coarse_loss + refined_loss) + 0.001*(120*style_loss + 5*perceptual_loss) + 0.005*gen_loss

        


        self.loss_tracker.update_state(total_loss)
        self.coarse_metric.update_state(coarse_loss)
        self.refine_metric.update_state(refined_loss)
        self.style_metric.update_state(style_loss)
        self.perceptual_metric.update_state(perceptual_loss)
        self.gen_metric.update_state(gen_loss)
        self.dis_metric.update_state(dis_loss)
        self.ssim_metric.update_state(y, refined)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.coarse_metric, self.refine_metric, self.style_metric, self.perceptual_metric, self.gen_metric, self.dis_metric , self.ssim_metric]



def build_model(hp):
    model = GenModel(hp.Int("cn_num", 32, 256, 32))
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model