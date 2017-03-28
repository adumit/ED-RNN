"""
Must have openCV for python, downloaded here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
"""

import cv2
import numpy as np
import os
import math
import random
from .Video_Processing import process_video


def read_kth_splits():
    split_dict = {}
    for f in os.listdir("./doc/KTHSplit"):
        which_class = f.split("_")[0]  # Which of the 6 classes the split file is for
        class_dict = {}  # Dictionary of file_name : which set (training or test) the file is in

        split_file = open("./doc/KTHSplit/" + f, "r")
        for line in split_file:
            file_name, which_set = line.split(" ")
            class_dict[file_name] = int(which_set)
        split_dict[which_class] = class_dict
    return split_dict



class KTHDataLoader:
    def __init__(self, data_path, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        video_data = []
        video_labels = []

        seen_labels = {}
        if not os.path.isdir("./Processed_KTH_Data"):
            os.mkdir("./Processed_KTH_Data")
            print("No processed data found. Processing KTH data.")
            for vid_file in os.listdir(data_path):
                process_video("./Processed_KTH_Data/", data_path, vid_file, 24, 24)
            print("Done processing data.")
        for npfile in os.listdir("./Processed_KTH_Data/"):
            vid_label = npfile.split("_")[1]
            if vid_label not in seen_labels:
                if len(seen_labels) == 0:
                    seen_labels[vid_label] = 0
                else:
                    seen_labels[vid_label] = max(seen_labels.values()) + 1
            vid = np.load("./Processed_KTH_Data/" + npfile)
            frameCount = vid.shape[0]

            number_of_slices = math.floor(frameCount / num_steps)
            split_vid_data = [np.array(vid[i * num_steps:(i + 1) * num_steps]) for i in range(number_of_slices)]
            video_data += split_vid_data
            video_labels += [np.ones(num_steps) * seen_labels[vid_label]] * len(split_vid_data)

        # Get the width, height, and channel properties from the videos
        self.width = video_data[0].shape[2]
        self.height = video_data[0].shape[1]
        self.num_channels = video_data[0].shape[3]

        shuffle_indices = list(range(len(video_data)))
        random.shuffle(shuffle_indices)
        shuffled_video_data = [video_data[i] for i in shuffle_indices]
        shuffled_video_labels = [video_labels[i] for i in shuffle_indices]

        self.num_classes = len(seen_labels)
        self.num_batches = math.floor(len(shuffled_video_data) / batch_size)
        self.batched_data = [
            (np.concatenate(np.expand_dims(shuffled_video_data[i * batch_size:(i + 1) * batch_size], axis=0)),
             np.array(shuffled_video_labels[i * batch_size:(i + 1) * batch_size]))
            for i in range(self.num_batches)]
        standard_shape = self.batched_data[0][0].shape
        self.batched_data = [x for x in self.batched_data if x[0].shape == standard_shape]
        self.num_batches = len(self.batched_data)
        self.batch_index = 0

    def generator(self):
        while True:
            if self.batch_index >= self.num_batches:
                self.batch_index = 0
                random.shuffle(self.batched_data)

            xdata = self.batched_data[self.batch_index][0]
            sparse_ydata = self.batched_data[self.batch_index][1]
            ydata = np.expand_dims(sparse_ydata, axis=2)
            self.batch_index += 1
            yield (xdata, ydata)

    def next_batch(self):
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            np.random.shuffle(self.batched_data)

        xdata = self.batched_data[self.batch_index][0]
        sparse_ydata = self.batched_data[self.batch_index][1]
        ydata = np.expand_dims(sparse_ydata, axis=2)
        self.batch_index += 1
        return xdata, ydata
