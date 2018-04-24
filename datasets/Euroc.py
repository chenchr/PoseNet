from __future__ import division
import torch.utils.data as data
import os.path
import glob
from scipy.ndimage import imread
import numpy as np
import pandas as pd
import torch
import util
from .base_dataset import base_dataset

def Euroc(root, train_sequence, test_sequence, split, stride, transform):
    train_seq = ['V2_01_easy', 'MH_02_easy', 'V1_03_difficult', 'V1_01_easy', 'V1_02_medium', 'V2_02_medium', 'MH_04_difficult', 'MH_03_medium']
    test_seq  = ['MH_05_difficult', 'V2_03_difficult']
    # vector_w = T * vector_b
    #read csv
    train_image_list = []
    train_pose_list = []
    for seq in train_seq:
        path_csv = os.path.join(root, seq, 'data.csv')
        df = pd.DataFrame(pd.read_csv(path_csv))
        image_list = df['filename'].values.tolist()
        image_list = [os.path.join(root, seq, 'image', name) for name in image_list]
        pose_list = []
        for row in df.iterrows():
            index, data = row
            pose_list.append(data.tolist()[1:-1])
        pose_list = [util.vector_to_transform(item) for item in pose_list]
        train_image_list.extend(image_list)
        train_pose_list.extend(pose_list)
    # train_dataset = base_dataset(image_list[0:int(len(image_list)*split)], pose_list[0:int(len(image_list)*split)], stride, transform)
    train_dataset = base_dataset(train_image_list, train_pose_list, stride, transform)

    test_image_list = []
    test_pose_list = []
    for seq in test_seq:
        path_csv = os.path.join(root, seq, 'data.csv')
        df = pd.DataFrame(pd.read_csv(path_csv))
        image_list = df['filename'].values.tolist()
        image_list = [os.path.join(root, seq, 'image', name) for name in image_list]
        pose_list = []
        for row in df.iterrows():
            index, data = row
            pose_list.append(data.tolist()[1:-1])
        pose_list = [util.vector_to_transform(item) for item in pose_list]
        test_image_list.extend(image_list)
        test_pose_list.extend(pose_list)
    # test_dataset = base_dataset(image_list[int(len(image_list)*split):], pose_list[int(len(image_list)*split):], stride, transform)
    test_dataset = base_dataset(test_image_list, test_pose_list, stride, transform)
    return train_dataset, test_dataset