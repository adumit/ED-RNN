from keras import backend as K
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from .keras_layers import Lenet, NiN, ED_EMA

CPU = -1

# Small net settings for testing and debugging
NUM_CONV_LAYERS = 2
CONV_SIZES = [(13, 13), (5, 5)]
CONV_FILTERS = [48, 128]
CONV_STRIDES = [(4, 4), (2, 2)]
CONV_POOL = [(2, 2), (2, 2)]

class ED_RNN:
    def __init__(self, opt):
        self.opt = opt

        device = '/cpu:0'

        # Dirty workaround for BatchNormalization layers. https://github.com/fchollet/keras/issues/5975
        K.set_learning_phase(1)

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

            self.inputs = Input(batch_shape=(opt.batch_size, opt.num_steps, opt.height, opt.width, opt.num_channels),
                                dtype='float32', name='video')
            x = self.inputs

            if opt.use_edema == 1:
                x = ED_EMA(tao=1.5, v_threshold=0.05, beta=2.0, return_sequences=True)(x)

            if opt.network.lower() == "lenet":
                x = Lenet(x)
            elif opt.network.lower() == "nin":
                x = NiN(x)
            else:
                # General conv-net with parameters defined at top of file
                for cl in range(NUM_CONV_LAYERS):
                    x = TimeDistributed(Convolution2D(filters=CONV_FILTERS[cl], kernel_size=CONV_SIZES[cl],
                                                      strides=CONV_STRIDES[cl], activation='relu'))(x)
                    x = TimeDistributed(MaxPool2D(pool_size=(CONV_POOL[cl])))(x)

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

