import pickle
import keras
from keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, Softmax, Input, Flatten
from keras.models import Model

def import_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def define_architecture():
    wavelet_inputs = Input(shape=(248, 16, 1), name='wavelet_input')
    
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(wavelet_inputs)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 1), strides=None)(x)
    x = Flatten()(x)
    classified_outputs = Dense(6, activation='softmax')(x)
    
    model = Model(inputs=[wavelet_inputs], outputs=classified_outputs)

    return model

nn_data = import_data('waveletdata.pkl');