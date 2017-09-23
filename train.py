from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import accuracy_score

from utils import DataLoader
from model import FineTuneNet

TRAIN_FOLDER = os.path.abspath('./train_emb')
TEST_FOLDER = os.path.abspath('./test_emb')


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


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


def gen_accuracy(output, target):
    output_np = output.cpu().squeeze(1).data.numpy()
    output_np = np.argmax(output_np, axis=1)
    target_np = target.cpu().data.numpy()
    return accuracy_score(target_np, output_np)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_folder', type=str, help='where to store trained model')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.cuda and torch.cuda.is_available()


    train_loader = DataLoader(TRAIN_FOLDER, batch_size=args.batch_size)
    test_loader = DataLoader(TEST_FOLDER, batch_size=args.batch_size)


    model = FineTuneNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        model.train()
        while True:
            out_of_data, batch_idx, (data, target) = train_loader.load()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = Variable(data)
            target = Variable(target, requires_grad=False)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            acc = gen_accuracy(torch.exp(output), target)
            
            loss_meter.update(loss.data[0], len(data)) 
            acc_meter.update(acc, len(data))

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx * len(data), train_loader.size,
                    100. * batch_idx * len(data) / train_loader.size, 
                    loss_meter.avg, acc_meter.avg))

            if out_of_data:
                break

        train_loader.reset()  # restarts from top


    def test(epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        model.eval()
        while True:
            out_of_data, batch_idx, (data, target) = test_loader.load()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = Variable(data, volatile=True)
            target = Variable(target, requires_grad=False)

            output = model(data)
            loss = F.nll_loss(output, target)
            acc = gen_accuracy(torch.exp(output), target)
            
            loss_meter.update(loss.data[0], len(data)) 
            acc_meter.update(acc, len(data))

            if out_of_data:
                break

        print('\nTest Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}\n'.format(
              epoch, loss_meter.avg, acc_meter.avg))

        test_loader.reset()
        return acc_meter.avg


    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
        }, is_best, folder=args.out_folder)
