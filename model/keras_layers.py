import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, \
    Flatten, Dropout, AveragePooling2D, Activation, InputSpec
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import relu


class EMA(Recurrent):
    def __init__(self, tao=1.5, **kwargs):
        self.tao = tao
        self.prev_output = None
        self.input_spec = False
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
    def __init__(self, tao=1.5, v_threshold=0.05, beta=2.0, **kwargs):
        self.tao = tao
        self.v_threshold = v_threshold
        self.beta = beta
        super(ED_EMA, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=5)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.states = [None, None]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        in_shape = K.int_shape(inputs)
        initial_state = K.zeros(shape=(in_shape[0], in_shape[2], in_shape[3], in_shape[4]))
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states_value=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')
        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if states_value is not None:
            if not isinstance(states_value, (list, tuple)):
                states_value = [states_value]
            if len(states_value) != len(self.states):
                raise ValueError('The layer has ' + str(len(self.states)) +
                                 ' states, but the `states_value` '
                                 'argument passed '
                                 'only has ' + str(len(states_value)) +
                                 ' entries')
        if self.states[0] is None:
            self.states = [K.zeros(np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4])))
                           for _ in self.states]
            if not states_value:
                return
        for i, state in enumerate(self.states):
            if states_value:
                value = states_value[i]
                if value.shape != np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4])):
                    raise ValueError(
                        'Expected state #' + str(i) +
                        ' to have shape ' + str(np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4]))) +
                        ' but got array with shape ' + str(value.shape))
            else:
                value = np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4]))
            K.set_value(state, value)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[2], input_shape[3], input_shape[4]

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
            output = relu(output, 0.0)
        else:
            output = relu(-output, 0.0)

        return output, [output, ema]


def Lenet(input_layer):
    x = TimeDistributed(Convolution2D(filters=6, kernel_size=(5, 5),
                                      strides=(1, 1), activation='linear'))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    # TODO: Need to have relu after the batch norm?
    x = Activation("relu")(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Convolution2D(filters=16, kernel_size=(5, 5),
                                      strides=(1, 1), activation='linear'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Activation("relu")(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    return x


def Block(input_layer, num_filters, conv_size, conv_strides):
    x = TimeDistributed(Convolution2D(filters=num_filters, kernel_size=conv_size, strides=conv_strides,
                                      activation="linear"))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    x = Activation("relu")(x)
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
