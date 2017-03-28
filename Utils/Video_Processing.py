import numpy as np
import cv2


def process_video(save_path, data_path, vid_name, new_x, new_y):
    vid_read = cv2.VideoCapture(data_path + vid_name)
    vid_data = []
    ret, frame = vid_read.read()
    while ret:
        vid_data.append(cv2.resize(frame, (new_x, new_y), cv2.INTER_LINEAR))
        ret, frame = vid_read.read()
    np.save(save_path + vid_name, np.array(vid_data))
