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
    test_loss = 0
    correct = 0


    correct_0,correct_1,correct_2,correct_3,correct_4,correct_5,correct_6,correct_7,correct_8,correct_9 = 0,0,0,0,0,0,0,0,0,0
    total_0,total_1,total_2,total_3,total_4,total_5,total_6,total_7,total_8,total_9 = 0,0,0,0,0,0,0,0,0,0


    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        #print(len(target))
        #print(y_pred)

        for i in range(len(target)):
            if target[i] == 0:
                total_0 += 1
                if y_pred[i][0] == 0:
                    correct_0 +=1

            if target[i] == 1:
                total_1 += 1
                if y_pred[i][0] == 1:
                    correct_1 += 1

            if target[i] == 2:
                total_2 += 1
                if y_pred[i][0] == 2:
                    correct_2 += 1

            if target[i] == 3:
                total_3 += 1
                if y_pred[i][0] == 3:
                    correct_3 += 1

            if target[i] == 4:
                total_4 += 1
                if y_pred[i][0] == 4:
                    correct_4 += 1

            if target[i] == 5:
                total_5 += 1
                if y_pred[i][0] == 5:
                    correct_5 += 1

            if target[i] == 6:
                total_6 += 1
                if y_pred[i][0] == 6:
                    correct_6 += 1

            if target[i] == 7:
                total_7 += 1
                if y_pred[i][0] == 7:
                    correct_7 += 1

            if target[i] == 8:
                total_8 += 1
                if y_pred[i][0] == 8:
                    correct_8 += 1

            if target[i] == 9:
                total_9 += 1
                if y_pred[i][0] == 9:
                    correct_9 += 1

        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        #print(correct)
    target_freq = {0:total_0,1:total_1,2:total_2,3:total_3,4:total_4,5:total_5,6:total_6,7:total_7,8:total_8,9:total_9}
    correct_freq = {0:correct_0,1:correct_1,2:correct_2,3:correct_3,4:correct_4,5:correct_5,6:correct_6,7:correct_7,8:correct_8,9:correct_9}
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_0 = 100.00 * correct_0 / total_0
    accuracy_1 = 100.00 * correct_1 / total_1
    accuracy_2 = 100.00 * correct_2 / total_2
    accuracy_3 = 100.00 * correct_3 / total_3
    accuracy_4 = 100.00 * correct_4 / total_4
    accuracy_5 = 100.00 * correct_5 / total_5
    accuracy_6 = 100.00 * correct_6 / total_6
    accuracy_7 = 100.00 * correct_7 / total_7
    accuracy_8 = 100.00 * correct_8 / total_8
    accuracy_9 = 100.00 * correct_9 / total_9

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, accuracy_0, accuracy_1, accuracy_2, accuracy_3,accuracy_4,accuracy_5,accuracy_6,accuracy_7,accuracy_8,accuracy_9,target_freq,correct_freq
