import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, \
    Flatten, Dropout, AveragePooling2D, Activation, InputSpec, initializers, regularizers, constraints, Layer
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import relu


class LayerLambdas:
    @staticmethod
    def KTHSlice(x):
        return x[:, :, :, :, 0:1]

    @staticmethod
    def KTHSlice_Output_Shape(x):
        return x[0], x[1], x[2], x[3], 1

    @staticmethod
    def ChannelizedLSTM(input_layer, num_layers, rnn_size):
        """ This model assumes that the channel dim is the fifth dimension """
        slices = []
        for i in range(K.int_shape(input_layer)[4]):
            single_slice = tf.slice(input_layer, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1])
            x = TimeDistributed(Flatten())(single_slice)
            for _ in range(num_layers):
                x = LSTM(rnn_size, return_sequences=True)(x)
            expanded = K.expand_dims(x, axis=-1)
            slices.append(expanded)
            print(expanded)
        stacked = K.concatenate(slices, axis=-1)
        print(stacked)
        return stacked

    @staticmethod
    def OnOffThreshold(x, pos_threshold, neg_threshold):
        pos_output = x + (1.0 + pos_threshold)
        pos_output = relu(pos_output, 0.0)
        neg_output = x - (1.0 - neg_threshold)
        neg_output = relu(-neg_output, 0.0)

        output = K.concatenate([pos_output, neg_output], axis=3)

    @staticmethod
    def OnOffThreshold_OutputShape(x):
        """ 5 dimension? Batch x Time x l x w x c """
        return x[0], x[1], x[2], x[3], x[4] * 2



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

class ScaledLogReturn(Recurrent):
    def __init__(self, shape, tao=1.5, **kwargs):
        self.shape = shape
        self.tao = tao
        self.prev_output = None
        self.input_spec = False
        super(ScaledLogReturn, self).__init__(**kwargs)

    def tao_init(self, shape, dtype=None):
        """ From EDEMA """
        return K.variable(np.ones(shape=shape, dtype=dtype) * self.tao)

    def beta_init(self, shape, dtype=None):
        """ From EDEMA """
        return K.variable(np.ones(shape=shape, dtype=dtype) * self.beta)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.beta = self.add_weight(shape=self.shape, initializer=self.beta_init)
        self.tao_mat = self.add_weight(shape=self.shape, initializer=self.tao_init)
        self.states = [None]

        if self.stateful:
            self.reset_states()

        self.built = True

    def preprocess_input(self, x, training=None):
        """
        Preprocess first time-step to be true to the EMA equation of EMA(0) = beta * log(input(0))
        Rather than EMA(0) = beta * log(1.0/tao * input(0))
        """
        # change_first_input = np.ones(shape=x.get_shape().as_list())
        # change_first_input[:, 0, :] *= self.tao
        # TODO: What happens if beta x is 0? or negative?
        return self.beta * np.log(x)

    def get_initial_states(self, inputs):
        """ From the ED_EMA """
        # build an all-zero tensor of shape (samples, output_dim)
        in_shape = K.int_shape(inputs)
        initial_state = K.zeros(shape=(in_shape[0], in_shape[2], in_shape[3], in_shape[4]))
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states_value=None):
        """ Copied from ED-EMA """
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
            self.states = [
                K.zeros(np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4])))
                for _ in self.states]
            if not states_value:
                return
        for i, state in enumerate(self.states):
            if states_value:
                value = states_value[i]
                if value.shape != np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4])):
                    raise ValueError(
                        'Expected state #' + str(i) +
                        ' to have shape ' + str(
                            np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4]))) +
                        ' but got array with shape ' + str(value.shape))
            else:
                value = np.zeros((batch_size, self.input_shape[2], self.input_shape[3], self.input_shape[4]))
            K.set_value(state, value)

    def compute_output_shape(self, input_shape):
        """ Copied from ED-EMA """
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.return_sequences:
            return input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        else:
            return input_shape[0], input_shape[2], input_shape[3], input_shape[4]

    def step(self, x, states):
        self.prev_output = prev_output = states[0]

        # TODO: What happens if tao_mat goes to zero? o.O
        ema = (1.0 - 1.0/self.tao_mat) * prev_output + 1.0/self.tao_mat * x
        output = self.beta * tf.log(ema)

        return output, [output]


class ED_EMA(Recurrent):
    def __init__(self, shape, tao=1.5, v_pos_threshold=0.003, v_neg_threshold=-0.003, beta=2.0, **kwargs):
        self.shape = shape
        self.tao = tao
        self.v_pos_threshold = v_pos_threshold
        self.v_neg_threshold = v_neg_threshold
        self.beta = beta
        super(ED_EMA, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=5)

    def tao_init(self, shape, dtype=None):
        return K.variable(np.ones(shape=shape, dtype=dtype) * self.tao)

    def beta_init(self, shape, dtype=None):
        return K.variable(np.ones(shape=shape, dtype=dtype) * self.beta)

    def build(self, input_shape):
        input_dim = input_shape[2]

        # TODO: Are there constraints on tao and beta?
        self.tao_mat = self.add_weight(shape=self.shape, initializer=self.tao_init, name="tao")
        self.beta_mat = self.add_weight(shape=self.shape, initializer=self.beta_init, name="beta")

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.states = [None]
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
            return input_shape[0], input_shape[1], input_shape[2], input_shape[3], 2*input_shape[4]
        else:
            return input_shape[0], input_shape[2], input_shape[3], 2*input_shape[4]

    def preprocess_input(self, x, training=None):
        """
        Preprocess first time-step to be true to the EMA equation of EMA(0) = input(0)
        Rather than EMA(0) = 1.0/tao * input(0)
        """
        # change_first_input = np.ones(shape=x.get_shape().as_list())
        # change_first_input[:, 0, :] *= self.tao
        return x

    def step(self, x, states):
        self.prev_ema = prev_ema = states[0]

        ema = (1.0 - 1.0/self.tao_mat) * prev_ema + 1.0/self.tao_mat * x

        pos_output = (x / ema)**self.beta_mat - (1.0 + self.v_pos_threshold)
        pos_output = relu(pos_output, 0.0)
        neg_output = (x / ema)**self.beta_mat - (1.0 - self.v_neg_threshold)
        neg_output = relu(-neg_output, 0.0)

        output = K.concatenate([pos_output, neg_output], axis=3)

        return output, [ema]


class CLSTM(Recurrent):
    WIDTH = 0
    HEIGHT = 1
    CHANNELS = 2

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        super(CLSTM, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=5)

    def build(self, input_shape):
        # input_dim = input_shape[2]
        # self.input_dim = input_dim
        # self.output_dim = input_dim
        # self.states = [None, None]
        # if self.stateful:
        #     self.reset_states()
        # self.built = True
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2:]
        self.channels = self.input_dim[self.CHANNELS]
        self.input_spec = InputSpec(shape=(([batch_size, None].extend(self.input_dim))))
        self.state_spec = [InputSpec(shape=((batch_size,) + ())),
                           InputSpec(shape=([batch_size].extend(self.units)))]
        self.output_dim = (self.channels, self.units)

        self.states = [None, None]
        if self.stateful:
            self.reset_states()
        kernel_shape = self.input_dim + (self.units * 4,)
        self.kernel = self.add_weight(kernel_shape,
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        recurrent_shape = (self.channels, self.units, self.units * 4)
        self.recurrent_kernel = self.add_weight(
            recurrent_shape,
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        bias_shape = (self.channels, self.units * 4)
        if self.use_bias:
            self.bias = self.add_weight((self.channels, self.units * 4),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = np.zeros(bias_shape)
                bias_value[:, self.units: self.units * 2] = 1.
                K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :, :, :self.units]
        self.kernel_f = self.kernel[:, :, :, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, :, :, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, :, :, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, :, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, :, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:, self.units]
            self.bias_f = self.bias[:, self.units: self.units * 2]
            self.bias_c = self.bias[:, self.units * 2: self.units * 3]
            self.bias_o = self.bias[:, self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    # TODO: Modify the compute_output_shape, get_initial_states, and reset_states
    # TODO: appropriately for this Channelized LSTM
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[2], input_shape[3], input_shape[4]

    def get_initial_states(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.kernel_shape)
        shape[-1] = self.filters
        initial_state = self.input_conv(initial_state,
                                        K.zeros(tuple(shape)),
                                        padding=self.padding)

        initial_states = [initial_state for _ in range(2)]
        return initial_states

    def reset_states(self):
        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')
        input_shape = self.input_spec.shape
        output_shape = self.compute_output_shape(input_shape)
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))

        if self.return_sequences:
            out_row, out_col, out_filter = output_shape[2:]
        else:
            out_row, out_col, out_filter = output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
        else:
            self.states = [K.zeros((input_shape[0],
                                    out_row, out_col, out_filter)),
                           K.zeros((input_shape[0],
                                    out_row, out_col, out_filter))]

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation == 0 and 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:
            shape = list(self.kernel_shape)
            shape[-1] = self.filters
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones = self.input_conv(ones, K.zeros(shape),
                                   padding=self.padding)
            ones += 1.

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def preprocess_input(self, x, training=None):
        """
        Preprocess first time-step to be true to the EMA equation of EMA(0) = input(0)
        Rather than EMA(0) = 1.0/tao * input(0)
        """
        # change_first_input = np.ones(shape=x.get_shape().as_list())
        # change_first_input[:, 0, :] *= self.tao
        return x

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        # Inputs: Batchsize x width x height x channels
        # input kernel: width x height x channels x units*4
        z = tf.tensordot(inputs * dp_mask[0], self.kernel, ((1, 2), (0, 1)))

        # h_tm1: channels x units
        # recurrent kernel: channels x units x units*4
        z += tf.tensordot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel, ((1), (1)))

        z0 = z[:, :, :, :self.units]
        z1 = z[:, :, :, self.units: self.units * 2]
        z2 = z[:, :, :, self.units * 2: self.units * 3]
        z3 = z[:, :, :, self.units * 3:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]


class PerChannelLSTM(Recurrent):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.channel_i = 3  # Expecting row x channels
        self.length_i = 2
        super(PerChannelLSTM, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2:] # Default will be length x channel
        self.channels = self.input_dim[self.channel_i]

        # NB: Not sure about the shapes here. Is keras using these specs
        # NB: Somewhere internally?
        self.input_spec = InputSpec(shape=(([batch_size, None].extend(self.input_dim))))
        self.state_spec = [InputSpec(shape=((batch_size,) + ())),
                           InputSpec(shape=([batch_size].extend(self.units)))]
        self.output_dim = (self.channels, self.units)

        self.states = [None, None]
        if self.stateful:
            self.reset_states()
        self.kernel_shape = self.input_dim + (self.units * 4,)
        self.kernel = self.add_weight(self.kernel_shape,
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_shape = (self.channels, self.units, self.units * 4)
        self.recurrent_kernel = self.add_weight(
            self.recurrent_shape,
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        bias_shape = (self.channels, self.units * 4)
        if self.use_bias:
            self.bias = self.add_weight((self.channels, self.units * 4),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = np.zeros(bias_shape)
                bias_value[:, self.units: self.units * 2] = 1.
                K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :, :self.units]
        self.kernel_f = self.kernel[:, :, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, :, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, :, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, :, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, :, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:, self.units]
            self.bias_f = self.bias[:, self.units: self.units * 2]
            self.bias_c = self.bias[:, self.units * 2: self.units * 3]
            self.bias_o = self.bias[:, self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    # TODO: Modify the compute_output_shape, get_initial_states, and reset_states
    # TODO: appropriately for this Channelized LSTM
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[self.channel_i], self.units

    def get_initial_states(self, inputs):
        # Build output layer of samples x output_dim
        # (samples, timesteps, rows, channels)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, channels)
        initial_state = K.sum(initial_state, axis=1)
        initial_state = K.sum(initial_state, axis=1)
        initial_state = K.expand_dims(initial_state, axis=-1)
        initial_states = [initial_state for _ in self.states]
        return initial_states

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2:]
            timesteps = input_shape[1]

            x_i = _time_distributed_dense(inputs, self.kernel_i, self.bias_i,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_c = _time_distributed_dense(inputs, self.kernel_c, self.bias_c,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_o = _time_distributed_dense(inputs, self.kernel_o, self.bias_o,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return inputs

    def reset_states(self, states_value=None):
        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')
        input_shape = self.input_spec.shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))


        # NB: This might be units * 4
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0],
                                  self.channels, self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0],
                                  self.channels, self.units)))
        else:
            self.states = [K.zeros((input_shape[0],
                                    self.channels, self.units)),
                           K.zeros((input_shape[0],
                                    self.channels, self.units))]

    def get_constants(self, inputs, training=None):
        constants = []
        if 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:

            ones = K.ones(shape=self.recurrent_shape)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants


    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        # Inputs: Batchsize x length x channel
        # input kernel: length x channels x units*4
        # for each channel_i, we want batchsize x length_i dot length_i x unit * 4
        # resulting calculation: batchsize x units*4 x channel

        z = tf.tensordot(inputs * dp_mask[0], self.kernel, ((self.length_i,), (self.length_i - 1,)))

        # h_tm1: channels x units
        # recurrent kernel: channels x units x units*4
        z += tf.tensordot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel, ((1), (1)))

        z0 = z[:, :, :self.units]
        z1 = z[:, :, self.units: self.units * 2]
        z2 = z[:, :, self.units * 2: self.units * 3]
        z3 = z[:, :, self.units * 3:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]


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
    return x


def Block(input_layer, num_filters, conv_size, conv_strides):
    x = TimeDistributed(Convolution2D(filters=num_filters, kernel_size=conv_size, strides=conv_strides,
                                      activation="linear", padding="same"))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    x = Activation("relu")(x)
    return x


def NiN(input_layer):
    x = Block(input_layer, 192, (5, 5), (2, 2))
    x = Block(x, 160, (1, 1), (1, 1))
    x = Block(x, 96, (1, 1), (1, 1))
    x = TimeDistributed(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = Block(x, 192, (5, 5), (2, 2))
    x = Block(x, 192, (1, 1), (1, 1))
    x = Block(x, 192, (1, 1), (1, 1))
    x = TimeDistributed(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = Block(x, 192, (3, 3), (1, 1))
    x = Block(x, 2, (1, 1), (1, 1))
    return x


def axis_element_wise_multiplication(t1, t2, which_axis):
    """
    Gets each tensor along a particular axis of t1, perform element-wise multiplication of those
    slices with t2 and returns the resulting matricies stacked.
    :param t1: First tensor. Should be 1 rank greater than t2
    :param t2: Tensor to element-wise multiply each slice of t1 by
    :param which_axis: Axis of t1 to slice along
    :return: Element-wise slice multiplication of t1 and t2. Should have shape of t1
    """
    # assert len(K.int_shape(t1)) == len(K.int_shape(t2)) + 1, "rank(t1) should be rank(t2) + 1"
    slices = tf.unstack(t1, axis=which_axis)
    # assert K.int_shape(slices[0]) == K.int_shape(t2), "Slices of t1 were not the same shape as t2"
    multiplies = []
    for s in slices:
        multiplies.append(t2 * s)
    return tf.stack(multiplies, axis=2)
