import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
from custom_loss import custom_loss
import datetime
from tensorboardX import SummaryWriter
import numpy as np

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("_"))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch PoseNet Training on Kitti, Euroc',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('--train-seq', required=True,
                    help='train sequence')
parser.add_argument('--test-seq', required=True,
                    help='test sequence')

parser.add_argument('--dataset', metavar='DATASET', default='Kitti',
                    choices=dataset_names,
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--stride', default = 2, type=int,
                    help='number of frame between 2 images')
group = parser.add_mutually_exclusive_group()
group.add_argument('--split-value', default=0.9, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')
parser.add_argument('--arch', '-a', metavar='ARCH', default='posenets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--qua-weight', default=-100, type=float,
                    metavar='W', help='weight for loss of qua, use Homoscedastic Uncertainty if not specified')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

lowest_error = -1
n_iter = 0


def main():
    global args, lowest_error, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{},stride{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
        args.stride)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join('LOG_'+args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))

    # Data loading code
    input_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("=> fetching img pairs in '{}'".format(args.root))
    train_set, test_set = datasets.__dict__[args.dataset](args.root, args.train_seq, args.test_seq, args.split_value, args.stride, input_transform)
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    print('train set: {}\t test set: {}'.format(args.train_seq, args.test_seq))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    # param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
    #                 {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    param_groups = [{'params': model.module.trainable_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        lowest_error = validate(val_loader, model, 0)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.epochs):
        scheduler.step()

        # train for one epoch
        all_error, qua_error, t_error = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('all_error', qua_error, epoch)
        train_writer.add_scalar('qua_error', qua_error, epoch)
        train_writer.add_scalar('t_error', t_error, epoch)

        # evaluate on validation set

        all_error, qua_error, t_error = validate(val_loader, model, epoch)
        test_writer.add_scalar('all_error', all_error, epoch)
        test_writer.add_scalar('qua_error', qua_error, epoch)
        test_writer.add_scalar('t_error', t_error, epoch)

        if lowest_error < 0:
            best_EPE = all_error

        is_best = all_error < lowest_error
        lowest_error = min(all_error, lowest_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'lowest_error': lowest_error
        }, is_best)


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    all_error_meter = AverageMeter()
    qua_error_meter = AverageMeter()
    t_error_meter = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input_im, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_im = input_im.cuda()
        input_var = torch.autograd.Variable(input_im)
        target_var = torch.autograd.Variable(target)

        # compute output
        output_var, qua_weight, t_weight = model(input_var)
        if args.qua_weight != -100:
            qua_weight = torch.autograd.Variable(torch.ones(1).fill_(args.qua_weight)).type_as(target_var)
            t_weight = torch.autograd.Variable(torch.zeros(1)).type_as(target_var)
        all_error, qua_error, t_error = custom_loss(output_var, target_var, qua_weight, t_weight)
            
        # record three error
        all_error_meter.update(all_error.data[0])
        qua_error_meter.update(qua_error.data[0])
        t_error_meter.update(t_error.data[0])
        train_writer.add_scalar('train_loss', all_error.data[0], n_iter)
        train_writer.add_scalar('train_qua_loss', qua_error.data[0], n_iter)
        train_writer.add_scalar('train_t_loss', t_error.data[0], n_iter)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        all_error.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t Qua {6}\t Transl {7}'
                  .format(epoch, i, epoch_size, batch_time, data_time, 
                  all_error_meter, qua_error_meter, t_error_meter))
        n_iter += 1
        if i >= epoch_size:
            break

    return all_error_meter.avg, qua_error_meter.avg, t_error_meter.avg


def validate(val_loader, model, epoch):
    global args

    batch_time = AverageMeter()
    all_error_meter = AverageMeter()
    qua_error_meter = AverageMeter()
    t_error_meter = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_im, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_im.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output_var, qua_weight, t_weight = model(input_var)
        if args.qua_weight != -100:
            qua_weight = torch.autograd.Variable(torch.ones(1).fill_(args.qua_weight)).type_as(target_var)
            t_weight = torch.autograd.Variable(torch.zeros(1)).type_as(target_var)
        all_error, qua_error, t_error = custom_loss(output_var, target_var, qua_weight, t_weight)
        # record EPE
        all_error_meter.update(all_error.data[0])
        qua_error_meter.update(qua_error.data[0])
        t_error_meter.update(t_error.data[0])


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t Loss {3}\t Qua {4}\t Transl {5}'
                  .format(i, len(val_loader), batch_time, all_error_meter, qua_error_meter, t_error_meter))

    print(' * Loss {:.3f}\t Qua {:.3f}\t Transl {:.3f}'.format(all_error_meter.avg, qua_error_meter.avg, t_error_meter.avg))

    return all_error_meter.avg, qua_error_meter.avg, t_error_meter.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


if __name__ == '__main__':
    main()
