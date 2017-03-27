import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, \
    Flatten, Dropout, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import relu


class EMA(Recurrent):
    def __init__(self, tao=1.5, **kwargs):
        self.tao = tao
        self.prev_output = None
        super(EMA, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x, training=None):
        """
        Preprocess first time-step to be true to the EMA equation of EMA(0) = input(0)
        Rather than EMA(0) = 1.0/tao * input(0)
        """
        # change_first_input = np.ones(shape=x.get_shape().as_list())
        # change_first_input[:, 0, :] *= self.tao
        return x

    def step(self, x, states):
        self.prev_output = prev_output = states[0]

        output = (1.0 - 1.0/self.tao) * prev_output + 1.0/self.tao * x

        return output, [output]


class ED_EMA(Recurrent):
    def __init__(self, tao=1.5, v_threshold=1.0, beta=1.0, **kwargs):
        self.tao = tao
        self.prev_output = None
        self.v_threshold = v_threshold
        self.beta = beta
        super(ED_EMA, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x, training=None):
        """
        Preprocess first time-step to be true to the EMA equation of EMA(0) = input(0)
        Rather than EMA(0) = 1.0/tao * input(0)
        """
        # change_first_input = np.ones(shape=x.get_shape().as_list())
        # change_first_input[:, 0, :] *= self.tao
        return x

    def step(self, x, states):
        self.prev_output = prev_output = states[0]
        self.prev_ema = prev_ema = states[1]

        ema = (1.0 - 1.0/self.tao) * prev_ema + 1.0/self.tao * x
        output = (x / ema)**self.beta - (1.0 + self.v_threshold)
        if self.v_threshold > 0.0:
            output = max(output, 0.0)
        else:
            output = max(-output, 0.0)

        return output, [output, ema]


def Lenet(input_layer):
    x = TimeDistributed(Convolution2D(filters=6, kernel_size=(5, 5),
                                      strides=(1, 1), activation='relu'))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    # TODO: Need to have relu after the batch norm?
    # x = relu(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Convolution2D(filters=16, kernel_size=(5, 5),
                                      strides=(1, 1), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    # x = relu(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    return x


def Block(input_layer, num_filters, conv_size, conv_strides):
    x = TimeDistributed(Convolution2D(filters=num_filters, kernel_size=conv_size, strides=conv_strides,
                                      activation="relu"))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    # x = relu(x)
    return x


def NiN(input_layer):
    x = Block(input_layer, 192, (5, 5), (1, 1))
    x = Block(x, 160, (1, 1), (1, 1))
    x = Block(x, 96, (1, 1), (1, 1))
    x = TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = Block(x, 192, (5, 5), (1, 1))
    x = Block(x, 192, (1, 1), (1, 1))
    x = Block(x, 192, (1, 1), (1, 1))
    x = TimeDistributed(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = Block(x, 192, (3, 3), (1, 1))
    x = Block(x, 192, (1, 1), (1, 1))
    x = TimeDistributed(Flatten())(x)
    return x
