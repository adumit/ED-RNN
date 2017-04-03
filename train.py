import argparse
import numpy as np
from Utils.BatchLoader import KTHDataLoader
from model.rnn_model import ED_RNN
from datetime import datetime

def main(opt):
    print("Loading in data...")
    start = datetime.now()
    train_loader = KTHDataLoader(opt.data_dir, opt.batch_size, opt.num_steps)
    print("Completed loading data. It took ", (datetime.now() - start).seconds, "seconds.")

    opt.num_classes = train_loader.num_classes
    opt.height = train_loader.height
    opt.width = train_loader.width
    opt.num_channels = train_loader.num_channels

    model = ED_RNN(opt)
    model.train(train_loader)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a ED-RNN on video data')
    # data
    parser.add_argument('--data_dir', type=str, default='./KTHData/', help='KTHData directory')
    # model params
    parser.add_argument('--rnn_size', type=int, default=8, help='size of RNN cell internal state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of layers in the LSTM')
    parser.add_argument('--network', type=str, default="nin", help='Type of network. Either lenet, nin, or anything '
                                                               'else to use conv-net params in rnn_model.py')
    parser.add_argument('--use_edema', type=int, default=1, help='Use the ED_EMA layer or not')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='starting learning rate')
    parser.add_argument('--batch_norm', type=int, default=0, help='use batch normalization over input (1=yes)')
    parser.add_argument('--num_steps', type=int, default=80, help='number of timesteps to unroll for')
    parser.add_argument('--batch_size', type=int, default=30, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, default=30, help='number of full passes through the training data')
    parser.add_argument('--gpu', type=int, default=0, help='Which gpu are you running on? -1 for cpu, you lame')
    # bookkeeping
    parser.add_argument('--seed', type=int, default=3435, help='manual random number generator seed')
    parser.add_argument('--run_id', default="default", help="Name the run")

    # parse input params
    params = parser.parse_args()
    np.random.seed(params.seed)

    main(params)