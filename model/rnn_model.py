from keras import backend as K
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, Flatten
from keras.models import Model
from keras.optimizers import Adam

CPU = -1

# Small net settings for testing
NUM_CONV_LAYERS = 2
CONV_SIZES = [(13, 13), (5, 5)]
CONV_FILTERS = [48, 128]
CONV_STRIDES = [(4, 4), (2, 2)]
CONV_POOL = [(2, 2), (2, 2)]

# # AlexNet settings
# NUM_CONV_LAYERS = 5
# CONV_SIZES = [(13, 13), (5, 5), (3, 3), (3, 3), (3, 3)]
# CONV_FILTERS = [48, 128, 192, 192, 128]

class ED_RNN:
    def __init__(self, opt):
        self.opt = opt

        device = '/cpu:0'

        if self.opt.gpu != CPU:
            device = '/gpu:' + str(self.opt.gpu)
        with K.tf.device(device):
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
            self.learning_rate = opt.learning_rate

            self.num_rnn_layers = opt.num_rnn_layers
            self.rnn_size = opt.rnn_size
            self.num_steps = opt.num_steps

            self.batch_size = opt.batch_size
            self.run_id = opt.run_id + '-conv_layers' + str(NUM_CONV_LAYERS) + '-rnn_layers' + \
                          str(self.num_rnn_layers) + "-rnn_size" + str(self.rnn_size)

            self.num_classes = opt.num_classes
            self.data_dir = opt.data_dir

            self.inputs = Input(batch_shape=(opt.batch_size, opt.num_steps, opt.input_height, opt.input_width, opt.input_channels),
                                dtype='float32', name='video')
            x = self.inputs

            if opt.batch_norm:
                x = BatchNormalization()(x)

            for cl in range(NUM_CONV_LAYERS):
                x = TimeDistributed(Convolution2D(filters=CONV_FILTERS[cl], kernel_size=CONV_SIZES[cl],
                                                  strides=CONV_STRIDES[cl], activation='relu'))(x)
                x = TimeDistributed(MaxPool2D(pool_size=(CONV_POOL[cl])))(x)

            x = TimeDistributed(Flatten())(x)
            for rl in range(opt.num_rnn_layers):
                x = LSTM(self.rnn_size, return_sequences=True, stateful=True)(x)

            self.output = TimeDistributed(Dense(input_dim=self.rnn_size,
                                                output_dim=opt.num_classes, activation='softmax'))(x)
            self.model = Model(input=self.inputs, output=self.output)
            self.model.summary()
            self.optimizer = Adam(lr=opt.learning_rate)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)
            self.write_dir = '../results_data/'

    def train(self, train_loader):
        self.model.fit_generator(generator=train_loader.generator(),
                                 steps_per_epoch=train_loader.num_batches)

