import utils.prepare_dataset as predat
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network.model import coarseNet
from network.loss import lossL1
import keras


def main():
    dsnp_c = np.moveaxis(np.load("Cloudy_DS2.npy"),3,0)
    dsnp = np.moveaxis(np.load("Ground_truth2.npy"),3,0)

    train_ds = tf.data.Dataset.from_tensor_slices((np.stack((dsnp_c[:800,:,:,3],dsnp_c[:800,:,:,-1]), axis=-1), dsnp[:800,:,:,3]))
    test_ds = tf.data.Dataset.from_tensor_slices((np.stack((dsnp_c[800:,:,:,3], dsnp_c[800:,:,:,-1]),axis=-1), dsnp[800:,:,:,3]))
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 800

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
    test_ds = test_ds.batch(BATCH_SIZE)

    model = coarseNet()
    opti = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opti, loss="mse")
    print(model.model.summary())
    model.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), keras.callbacks.ModelCheckpoint(filepath='coarseNet.keras', save_best_only=True, save_freq="epoch"), keras.callbacks.TensorBoard(log_dir='./logs'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0), keras.callbacks.CSVLogger('coarseNet.csv'), keras.callbacks.TerminateOnNaN()])
    print("Training complete")

    extractor = keras.Model(inputs=model.model.inputs, outputs=[layer.output for layer in model.model.layers])
    features = extractor.predict(dsnp_c[1:2,:,:,1:3])  


    fig = plt.figure()
    for i, feature in enumerate(features):
        ax = fig.add_subplot(5,4,i+1)
        ax.imshow(feature[0,:,:,0])
    fig.savefig('features_no_DO.png')

    model_DO = coarseNet(dropout=True)
    model_DO.compile(optimizer=opti, loss="mse")
    print(model_DO.model.summary())
    model_DO.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), keras.callbacks.ModelCheckpoint(filepath='coarseNet_DO.keras', save_best_only=True, save_freq="epoch"), keras.callbacks.TensorBoard(log_dir='./logs/DO'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0), keras.callbacks.CSVLogger('coarseNet_DO.csv'), keras.callbacks.TerminateOnNaN()])
    print("Training complete with dropout")

    extractor_DO = keras.Model(inputs=model_DO.model.inputs, outputs=[layer.output for layer in model_DO.model.layers])
    features_DO = extractor_DO.predict(dsnp_c[1:2,:,:,1:3])

    fig_DO = plt.figure()
    for i, feature in enumerate(features_DO):
        ax = fig_DO.add_subplot(5,4,i+1)
        ax.imshow(feature[0,:,:,0])
    fig_DO.savefig('features_DO.png')




if __name__ == "__main__":
    main()
    """
    #Data normalisation

    for i in range(dsnp.shape[3]):
        for j in range(11):
            dsnp[:,:,i,j] = (dsnp[:,:,i,j] - np.min(dsnp[:,:,i,j]))/(np.max(dsnp[:,:,i,j]) - np.min(dsnp[:,:,i,j]))
            dsnp_c[:,:,i,j] = (dsnp_c[:,:,i,j] - np.min(dsnp_c[:,:,i,j]))/(np.max(dsnp_c[:,:,i,j]) - np.min(dsnp_c[:,:,i,j]))

            #Split into training and test set


    train_ds = tf.data.Dataset.from_tensor_slices((dsnp_c[:800,:,:,:], dsnp[:800,:,:,:]))
    test_ds = tf.data.Dataset.from_tensor_slices((dsnp_c[800:,:,:,:], dsnp[800:,:,:,:]))

#Split into training and test set
"""