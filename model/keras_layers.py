import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent


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