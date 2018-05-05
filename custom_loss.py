import torch
import torch.nn as nn

def normalize(vec):
    divided = torch.sqrt(vec.pow(2).sum(dim=1, keepdim=True)) + 1e-8
    return vec/divided

def custom_loss(estimate, target, qua_weight, t_weight, test=False):
    # print("estimete: {}\ntarget: {}".format(estimate[0,:], target[0,:]))
    #if test:
     #   qua_estimate, qua_target = normalize(estimate[:, 0:3]), normalize(target[:, 0:3])
    #else:
    rVec_estimate, rVec_target = estimate[:, 0:3], target[:, 0:3]
    # we, xe, ye, ze = qua_estimate[0,:].data.cpu().numpy()
    # wt, xt, yt, zt = qua_target[0,:].data.cpu().numpy()
    # print('qua estimate: {}, target: {}'.format([we,xe,ye,ze], [wt,xt,yt,zt]))
    if test:
        t_estimate, t_target = normalize(estimate[:, 3:6]), normalize(target[:, 3:6])
    else:
        t_estimate, t_target = estimate[:, 3:6], target[:, 3:6]
    # xe, ye, ze = t_estimate[0,:].data.cpu().numpy()
    # xt, yt, zt = t_target[0,:].data.cpu().numpy()
    # print('t estimate: {}, target: {}'.format([xe,ye,ze],[xt,yt,zt]))
    rVec_error = (rVec_estimate - rVec_target).pow(2).sum(dim=1).mean()
    t_error = (t_estimate - t_target).pow(2).sum(dim=1).mean()
    # print('t error: {}'.format(t_error.data[0]))
    # all_error = qua_error * torch.exp(-qua_weight) + qua_weight + t_error * torch.exp(-t_weight) + t_weight
    all_error = rVec_error + t_error
    return all_error, rVec_error, t_error
