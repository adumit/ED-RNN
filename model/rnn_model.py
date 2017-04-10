from keras import backend as K
from keras.layers import Input, TimeDistributed, Dense, LSTM, Convolution2D, BatchNormalization, MaxPool2D, \
    Flatten, Reshape, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_accuracy
from .keras_layers import Lenet, NiN, ED_EMA, CLSTM, PerChannelLSTM, LayerLambdas
from .Callbacks import *
from PIL import Image

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
            self.max_epochs = opt.max_epochs

            self.num_rnn_layers = opt.num_rnn_layers
            self.rnn_size = opt.rnn_size
            self.num_steps = opt.num_steps

            self.batch_size = opt.batch_size
            self.run_id = opt.run_id + '-conv_layers' + str(NUM_CONV_LAYERS) + '-rnn_layers' + \
                          str(self.num_rnn_layers) + "-rnn_size" + str(self.rnn_size)

            self.num_classes = opt.num_classes
            self.data_dir = opt.data_dir

            self.inputs = x = Input(
                batch_shape=(opt.batch_size, opt.num_steps, opt.height, opt.width, opt.num_channels),
                dtype='float32', name='video')
            # TODO: For non-KTH dataset, make sure this doesn't go to only 1 channel
            if "kth" in opt.data_dir.lower():
                x = Lambda(LayerLambdas.KTHSlice, output_shape=LayerLambdas.KTHSlice_Output_Shape)(x)

            if opt.use_edema == 1:
                x = ED_EMA(shape=(opt.height, opt.width, K.int_shape(x)[4]),
                           tao=1.5, v_pos_threshold=0.003, v_neg_threshold=0.003, beta=2.0, return_sequences=True)(x)


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

            # For regular LSTM
            print("X Shape: ", K.int_shape(x))
            # x = TimeDistributed(Flatten())(x)
            # For the PerChannelLSTM user created layer:
            # x = TimeDistributed(Reshape(target_shape=(K.int_shape(x)[2] * K.int_shape(x)[3], K.int_shape(x)[4])))(x)
            print("X SHAPE:", x.shape)
            # for rl in range(opt.num_rnn_layers):
            #     # Don't return states for the last layer
            #     if rl == opt.num_rnn_layers - 1:
            #         x = LSTM(self.rnn_size, return_sequences=False)(x)
            #     else:
            #         x = LSTM(self.rnn_size, return_sequences=True)(x)
                # x = PerChannelLSTM(self.rnn_size, return_sequences=True)(x)
            x = Lambda(LayerLambdas.ChannelizedLSTM)(x, arguments=[opt.num_rnn_layers, opt.rnn_size])
            # x = LayerLambdas.ChannelizedLSTM(x, opt.num_rnn_layers, opt.rnn_size)
            # x = TimeDistributed(Flatten())(x)
            self.output = Dense(input_dim=self.rnn_size, units=opt.num_classes, activation='softmax')(x)
            self.model = Model(input=self.inputs, output=self.output)
            self.model.summary()
            self.optimizer = Adam(lr=opt.learning_rate, decay=0.05)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer,
                               metrics=[sparse_categorical_accuracy])
            self.write_dir = '../results_data/'

    def train(self, train_loader):
        self.model.fit_generator(generator=train_loader.generator(), steps_per_epoch=train_loader.num_batches,
                                 validation_data=train_loader.validation_generator(),
                                 validation_steps=train_loader.num_valid_batches,
                                 epochs=self.max_epochs,
                                 callbacks=[TestCallback(train_loader, train_loader.num_valid_batches)])
        x_ = train_loader.next_batch()[0]
        self.plot_KTH(x_, 2, 0)
        self.plot_KTH(x_, 2, 1)

    def get_activations(self, x_, layer_index):
        funcs = []
        for i in range(layer_index + 1):
            funcs.append(K.function([self.model.layers[i].input], [self.model.layers[i].output]))
        next_result = funcs[0]([x_])
        for f in funcs[1:]:
            next_result = f(next_result)
        return next_result

    def plot_KTH(self, x_, layer_index, which_channel=0):
        """
        :param x_: Batch of inputs that would normally be passed to the model 
        :param layer_index: Index of the layer to show activations for
        :param which_channel: 0 for show positive channel, 1 for show negative channel
        :return: 
        """
        activations = self.get_activations(x_, layer_index)
        img = Image.fromarray(activations[0][0][-1][:, :, which_channel], "L")
        img.save("./test" + str(which_channel) + ".png")