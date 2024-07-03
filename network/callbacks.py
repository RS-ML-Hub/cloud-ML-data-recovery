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

