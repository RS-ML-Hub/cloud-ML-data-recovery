import numpy as np
import keras
import tensorflow as tf

class LearningRateSchedulerCallback(keras.callbacks.Callback):
    def __init__(self, sagan, patience, factor=0.1, min_delta=0.001):
        super().__init__()
        self.sagan = sagan
        self.factor = factor
        self.best_loss = np.Inf
        self.patience = patience
        self.last_improvement=0
        self.min_delta = min_delta

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss') if 'val_loss' in logs else logs.get('loss')
        lr = self.sagan.optimizer.get_config()['learning_rate']
        if self.best_loss - current_loss > self.min_delta:
            self.last_improvement = epoch
            self.best_loss = current_loss
        else:
            if epoch - self.last_improvement > self.patience:
                self.sagan.optimizer.learning_rate =  lr*self.factor
                self.sagan.generator.optimizer.learning_rate =lr*self.factor
                self.sagan.discriminator.optimizer.learning_rate = 4*lr*self.factor
                self.last_improvement = epoch
                print("Learning rate reduced to ", lr*self.factor)

class OutputImageToTensorBoard(keras.callbacks.Callback):
    def __init__(self, sagan, epochs, test_ds ,log_dir, n_images=5):
        super().__init__()
        self.frequency = epochs
        self.n_images=n_images
        for masks,images in test_ds.take(1):
            indices = np.random.choice(images.shape[0], n_images, replace=False)
            self.masks = masks.numpy()[indices]
            self.images = images.numpy()[indices]
        self.sagan = sagan
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            bands = [3,5,9]
            x = self.images*(1-self.masks)+self.masks
            x = np.concatenate((x,self.masks), axis=-1)
            with self.file_writer.as_default():
                coarse, fine = self.sagan(x, training=False)
                coarse = tf.cast(coarse, tf.float64)
                fine = tf.cast(fine, tf.float64)
                fine = fine*self.masks + (1-self.masks)*self.images
                #shape of coarse
                coarse = np.stack([coarse[:,:,:,i] for i in bands], axis=-1)
                fine = np.stack([fine[:,:,:,i] for i in bands], axis=-1)
                for j in range(self.n_images):
                    tf.summary.image("Generated coarse_"+str(j), tf.expand_dims((coarse[j,:,:,:]+1)/2,axis=0), step=epoch)
                    tf.summary.image("Generated refined_"+str(j), tf.expand_dims((fine[j,:,:,:]+1)/2,axis=0), step=epoch)    
        return