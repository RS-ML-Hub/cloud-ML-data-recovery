import pickle

import keras


class VGG_fromTorch(keras.Model):
    def __init__(self):
        super().__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.net = keras.Sequential([keras.Input(shape=(224, 224, 3))])
        for v in self.cfg:
            if v == 'M':
                self.net.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            else:
                self.net.add(keras.layers.Conv2D(v, kernel_size=3, padding='same'))
                self.net.add(keras.layers.BatchNormalization())
                self.net.add(keras.layers.ReLU())
        self.net.trainable = False
        self.extractor = keras.Model(inputs=self.net.layers[0].input, outputs=[self.net.layers[i].output for i in [9,12,16]])
        self.extractor.trainable = False
    def make_layers(self):
        for v in self.cfg:
            if v == 'M':
                self.net.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            else:
                self.net.add(keras.layers.Conv2D(v, kernel_size=3, padding='same'))
                self.net.add(keras.layers.BatchNormalization())
                self.net.add(keras.layers.ReLU())
        self.net.build((None, 224, 224, 3))
    def load_weights_from_pytorch(self, weight_file):
        with open(weight_file, 'rb') as f:
            layer_weights = pickle.load(f)
        
        conv_idx = 0
        bn_idx = 0
        for layer in self.net.layers:
            if isinstance(layer, keras.layers.MaxPooling2D):
                conv_idx += 1
                bn_idx += 1
            if isinstance(layer, keras.layers.Conv2D):
                layer_name = f'features.{conv_idx}'
                layer.set_weights([layer_weights[layer_name]['weights'].transpose(2, 3, 1, 0), layer_weights[layer_name]['bias']])
                conv_idx += 3
            elif isinstance(layer, keras.layers.BatchNormalization):
                layer_name = f'features.{bn_idx+1}'
                layer.set_weights([layer_weights[layer_name]['weights'], layer_weights[layer_name]['bias'], layer_weights[layer_name]['running_mean'], layer_weights[layer_name]['running_var']])
                bn_idx += 3
    
    def call(self, x):
        return self.net(x)
    
"""model=VGG_fromTorch()

model.load_weights_from_pytorch('vgg16_bn_weights.pkl')

# Save the Keras model
model.net.save('vgg16_bn.h5')"""