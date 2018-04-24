from __future__ import division
import torch.utils.data as data
import os.path
import glob
from scipy.ndimage import imread
import numpy as np
import torch
import util
from .base_dataset import base_dataset

def Kitti(root, train_sequence, test_sequence, split=0.9, stride=2, transform=None):
    #get image path
    im_folder = os.path.join(root, "sequences", train_sequence, "image_0")
    im_num = len(os.listdir(im_folder))
    formater = "{:0>6d}.png"
    image_list = []
    for i in range(im_num):
        image_list.append(os.path.join(im_folder, formater.format(i)))
    #get pose
    pose_path = os.path.join(root, "poses", "{}.txt".format(train_sequence))
    pose_list = []
    with open(pose_path, 'r') as ff:
        pose_list = ff.readlines()
    pose_list = [i.strip("\n") for i in pose_list]
    temp = []
    row = np.array([0, 0, 0, 1])
    for i in pose_list:
        temp.append(np.row_stack((np.array(i.split(' ')).reshape(3, 4).astype(np.float32), row)))
    pose_list = temp
    # train_dataset = base_dataset(image_list[0:int(len(image_list)*split)], pose_list[0:int(len(image_list)*split)], stride, transform)
    train_dataset = base_dataset(image_list, pose_list, stride, transform)

    #get image path
    im_folder = os.path.join(root, "sequences", test_sequence, "image_0")
    im_num = len(os.listdir(im_folder))
    formater = "{:0>6d}.png"
    image_list = []
    for i in range(im_num):
        image_list.append(os.path.join(im_folder, formater.format(i)))
    #get pose
    pose_path = os.path.join(root, "poses", "{}.txt".format(test_sequence))
    pose_list = []
    with open(pose_path, 'r') as ff:
        pose_list = ff.readlines()
    pose_list = [i.strip("\n") for i in pose_list]
    temp = []
    row = np.array([0, 0, 0, 1])
    for i in pose_list:
        temp.append(np.row_stack((np.array(i.split(' ')).reshape(3, 4).astype(np.float32), row)))
    pose_list = temp
    test_dataset = base_dataset(image_list, pose_list, stride, transform)
    return train_dataset, test_dataset