from __future__ import division
import torch.utils.data as data
import os.path
import glob
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import torch
import util

from matplotlib import pyplot as plt
class base_dataset(data.Dataset):
    def __init__(self, image_list, pose_list, stride, transform):
        self.image_list = image_list
        self.pose_list = pose_list
        self.stride = stride
        self.transform = transform

    def __getitem__(self, index):
        # vector_w = T_1 * vector_b_1
        # vector_w = T_2 * vector_b_2
        # vector_b_2 = T_relative * vector_b_1
        # T_relative = inv(T_2) * T_1
        im1, im2 = [imread(self.image_list[i]) for i in [index, index+self.stride]]
        h, w = im1.shape[0:2]
        if w > 1000:
            h, w = 370, 1226
        im1, im2 = [imresize(i, (h//2, w//2)).astype(np.float32) for i in [im1, im2]]
        pose1, pose2 = self.pose_list[index].astype(np.float32), self.pose_list[index+self.stride].astype(np.float32)
        relative_pose = np.linalg.inv(pose2).dot(pose1) #TODO need check
        # print('relative_pose: {}'.format(relative_pose))
        rotation = relative_pose[0:3, 0:3]
        translation = relative_pose[0:3, 3]
        quaternion = util.rotation_matrix_to_quaternion(rotation)
        vector = np.append(quaternion, translation)
        if(len(im1.shape) == 2):
            im1, im2 = [np.expand_dims(im, axis=2) for im in [im1, im2]]
            im1, im2 = [np.repeat(im, 3, axis=2) for im in [im1, im2]]
            
        im1, im2 = [torch.from_numpy(np.transpose(im, [2,0,1])) for im in [im1, im2]]
        if(self.transform is not None):
            im1, im2 = [self.transform(im) for im in [im1, im2]]
        im_all = torch.cat([im1, im2], 0)
        vector = torch.from_numpy(vector)

        return im_all, vector

    def __len__(self):
        return len(self.image_list) - self.stride