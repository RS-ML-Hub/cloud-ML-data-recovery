import utils.prepare_dataset as predat
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network.model import GenModel
from network.loss import lossL1
import keras


def main():
    dsnp_c = np.moveaxis(np.load("Cloudy_DS2.npy"),3,0)
    dsnp = np.moveaxis(np.load("Ground_truth2.npy"),3,0)

    #Data normalisation

    for i in range(dsnp.shape[3]):
        for j in range(11):
            dsnp[:,:,i,j] = (dsnp[:,:,i,j] - np.min(dsnp[:,:,i,j]))/(np.max(dsnp[:,:,i,j]) - np.min(dsnp[:,:,i,j]))
            dsnp_c[:,:,i,j] = (dsnp_c[:,:,i,j] - np.min(dsnp_c[:,:,i,j]))/(np.max(dsnp_c[:,:,i,j]) - np.min(dsnp_c[:,:,i,j]))

            #Split into training and test set


    train_ds = tf.data.Dataset.from_tensor_slices((dsnp_c[:800,:,:,:], dsnp[:800,:,:,:]))
    test_ds = tf.data.Dataset.from_tensor_slices((dsnp_c[800:,:,:,:], dsnp[800:,:,:,:]))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 800

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
    test_ds = test_ds.batch(BATCH_SIZE)

    model = GenModel()
    coarseNet = model.coarseNet
    coarseNet.compile(optimizer='adam', loss=lossL1, metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error' , 'cosine_similarity', 'logcosh'])
    print(coarseNet.summary())
    coarseNet.fit(train_ds, epochs=1000, validation_data=test_ds, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True), keras.callbacks.ModelCheckpoint(filepath='coarseNet.keras', save_best_only=True), keras.callbacks.TensorBoard(log_dir='./logs'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0), keras.callbacks.CSVLogger('coarseNet.csv'), keras.callbacks.TerminateOnNaN()])
    print("Training complete")

if __name__ == "__main__":
    main()
    