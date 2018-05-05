# adapt from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/models/PoseExpNet.py
from __future__ import division
import torch
import torch.nn as nn
import matplotlib as mpl
import cv2
mpl.use('Agg')

__all__ = [
    'posenets', 'posenets_bn'
]

def conv(in_planes, out_planes, kernel_size=3, bn=False):
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

import numpy as np
from matplotlib import pyplot as plt
class PoseNetS(nn.Module):
    def __init__(self, bn=False):
        super(PoseNetS, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.bn = bn
        self.conv1 = conv(6,              conv_planes[0], kernel_size=7, bn=bn)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5, bn=bn)
        self.conv3 = conv(conv_planes[1], conv_planes[2], bn=bn)
        self.conv4 = conv(conv_planes[2], conv_planes[3], bn=bn)
        self.conv5 = conv(conv_planes[3], conv_planes[4], bn=bn)
        self.conv6 = conv(conv_planes[4], conv_planes[5], bn=bn)
        self.conv7 = conv(conv_planes[5], conv_planes[6], bn=bn)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)

        self.qua_weight = nn.Parameter(torch.ones(1))
        self.t_weight = nn.Parameter(torch.ones(1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.qua_weight.data.fill_(-3.0)
        self.t_weight.data.zero_()

    def trainable_parameters(self):
        ll = [param for name, param in self.named_parameters() if param.requires_grad == True]
        # print("ll:")
        # print(ll)
        return ll

    def forward(self, im):
        #test image
        print("tensor shape: {}".format(im.data.shape))
        im_test1, im_test2 = im[0, 0:3, :, :].data.cpu().numpy(), im[0, 3:6, :, :].data.cpu().numpy()
        im_test1, im_test2 = [np.transpose(im1, (1,2,0)) for im1 in [im_test1, im_test2]]
        im_test1 = im_test1.astype(np.uint8)
        cv2.imwrite('/data/chenchr/123.png', im_test1)
        # im_test1, im_test2 = [(im1 + 1)/2 for im1 in [im_test1, im_test2]]
        # print('max: {}, min: {}'.format(np.max(im_test1), np.min(im_test1)))
        # print('image shape: {}'.format(im_test1.shape))
        # plt.subplot(2,1,1)
        # plt.imshow(im_test2)
        # plt.subplot(2,1,2)
        # plt.imshow(im_test1)
        # plt.savefig('/data/chenchr/123.png')
        # plt.show()
        out_conv1 = self.conv1(im)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)

        return pose, self.qua_weight, self.t_weight

def posenets(data=None):
    model = PoseNetS(bn=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

def posenets_bn(data=None):
    model = PoseNetS(bn=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model