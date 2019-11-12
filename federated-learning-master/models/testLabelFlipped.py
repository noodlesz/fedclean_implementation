#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss_2 = 0
    correct_2 = 0
    counter_2_target = 0
    counter_2_bool = False
    counter_2_pred = 0
    total_2 = 0
    correct_2 = 0

    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        counter_2_target,counter_2_pred = 0,0
        for i in target:
            if i == 2:
                counter_2_target+=1
        total_2+=counter_2_target
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        for i in y_pred:
            if i[0] == 2:
                counter_2_pred+=1
        correct_2+=counter_2_pred
        #print(y_pred)
        #print(target.data.view_as(y_pred))
        #print(y_pred.eq(target.data.view_as(y_pred)))
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_2 = 100.00 * correct_2/total_2
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss,accuracy_2
