import torch
import torch.nn as nn
import math
from corr.functions.corr import CorrFunction

__all__ = [
    'PoseNetC', 'posenetc', 'posenetc_bn'
]

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

import numpy as np
from matplotlib import pyplot as plt

class PoseNetC(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(PoseNetC,self).__init__()

        self.batchNorm = batchNorm
        self.conv1  = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2  = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3  = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        # self.conv_redir  = conv(self.batchNorm, 256,  32, kernel_size=1, stride=1)
        self.corr = CorrFunction(20,1,20,1,2)
        self.LU_after_corr = nn.LeakyReLU(0.1,inplace=True)
        # self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv3_1 = conv(self.batchNorm, 441,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.pose_pred = nn.Conv2d(1024, 7, kernel_size=3, padding=0)

        self.qua_weight = nn.Parameter(torch.ones(1))
        self.t_weight = nn.Parameter(torch.ones(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data) # default mode is 'fan_in'
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def trainable_parameters(self):
        ll = [param for name, param in self.named_parameters() if param.requires_grad == True]
        # print("ll:")
        # print(ll)
        return ll


    def forward(self, x):
        ima, imb= x[:,0:3,:,:], x[:,3:6,:,:]

        # print('batch shape: {}'.format(x.data.shape))
        # im_test1, im_test2 = x[0, 0:3, :, :].data.cpu().numpy(), x[0, 3:6, :, :].data.cpu().numpy()
        # im_test1, im_test2 = [np.transpose(im1, (1,2,0)) for im1 in [im_test1, im_test2]]
        # im_test1, im_test2 = [(im1 + 1)/2 for im1 in [im_test1, im_test2]]
        # print('max: {}, min: {}'.format(np.max(im_test1), np.min(im_test1)))
        # print('image shape: {}'.format(im_test1.shape))
        # plt.subplot(2,1,1)
        # plt.imshow(im_test2)
        # plt.subplot(2,1,2)
        # plt.imshow(im_test1)
        # plt.show()

        out_conv2a = self.conv2(self.conv1(ima))
        out_conv2b = self.conv2(self.conv1(imb))
        out_conv3_1a = self.conv3(out_conv2a)
        out_conv3_1b = self.conv3(out_conv2b)

        out_corr_lu = self.LU_after_corr(self.corr(out_conv3_1a, out_conv3_1b))
        # out_redir = self.conv_redir(out_conv3_1a)
        # concat473 = torch.cat((out_corr_lu, out_redir), 1)

        # out_conv3 = self.conv3_1(concat473)
        out_conv3 = self.conv3_1(out_corr_lu)

        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        pose = self.pose_pred(out_conv6)
        pose = pose.mean(3).mean(2)

        return pose, self.qua_weight, self.t_weight




def posenetc(path=None):
    model = PoseNetC(batchNorm=False)
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model

def posenetc_bn(path=None):
    model = PoseNetC(batchNorm=True)
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model