#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from collections import OrderedDict,defaultdict


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        net_glob1 = CNNMnist(args=args).to(args.device)
        net_glob5 = CNNMnist(args=args).to(args.device)
        net_glob7 = CNNMnist(args=args).to(args.device)
        net_glob10 = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    net_glob1.train()
    net_glob5.train()
    net_glob7.train()
    net_glob10.train()

    # copy weights
    w_glob = net_glob.state_dict()
    w_glob1 = net_glob1.state_dict()
    w_glob5 = net_glob5.state_dict()
    w_glob7 = net_glob7.state_dict()
    w_glob10 = net_glob10.state_dict()
    # training - NO ATTACK
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    #VIVEK constant attack experiment - 1 MALICIOUS
    loss_train_1 = []
    fixed_agent_1 = random.randint(0,10)  #random agent between 0 and 31 is fixed
    updates_recorded_1 = False
    fixed_agent_storage_1 = None
    count_array_1 = []

    #VIVEK constant attack experiment - 20 MALICIOUS
    loss_train_5 = []
    fixed_agent_5 = random.sample(range(10),5)
    updates_recorded_mapping_5 = defaultdict(bool)
    for i in fixed_agent_5:
        updates_recorded_mapping_5[i] = False
    fixed_agent_storage_mapping_5 = {}
    count_array_5 = []


    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_7 = []
    fixed_agent_7 = random.sample(range(10),7)
    updates_recorded_mapping_7 = defaultdict(bool)
    for i in fixed_agent_7:
        updates_recorded_mapping_7[i] = False
    fixed_agent_storage_mapping_7 = {}
    count_array_7 = []


    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_10 = []
    fixed_agent_10 = random.sample(range(10),10)
    updates_recorded_mapping_10 = defaultdict(bool)
    for i in fixed_agent_10:
        updates_recorded_mapping_10[i] = False
    fixed_agent_storage_mapping_10 = {}
    count_array_10 = []



    for iter in range(args.epochs):
        #agent_found_count = 0
        w_locals, loss_locals = [], []          #w_locals = array of local_weights
        w_locals_1, loss_locals_1 = [],[]
        w_locals_5, loss_locals_5 = [],[]
        w_locals_7, loss_locals_7 = [],[]
        w_locals_10, loss_locals_10 = [],[]
        m = max(int(args.frac * args.num_users), 1)     #m = number of users used in one ROUND/EPOCH, check utils.options for more clarity on this
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #Randomly selecting m users out of 32 users. NEED TO REPLACE THIS WITH OUR SAMPLING MECHANISM

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local1 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local5 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local7 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local10 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w1, loss1 = local1.train(net=copy.deepcopy(net_glob1).to(args.device))
            w5, loss5 = local5.train(net=copy.deepcopy(net_glob5).to(args.device))
            w7, loss7 = local7.train(net=copy.deepcopy(net_glob7).to(args.device))
            w10, loss10 = local10.train(net=copy.deepcopy(net_glob10).to(args.device))
            print("***BLAH BLAH BLAH***")


            if idx==fixed_agent_1:
                if updates_recorded_1:
                    w1 = copy.deepcopy(fixed_agent_storage_1)
                elif not updates_recorded_1:
                    fixed_agent_storage_1 = copy.deepcopy(w1)
                    updates_recorded_1 = True

            if idx in fixed_agent_5:
                if updates_recorded_mapping_5[idx]:
                    w5 = copy.deepcopy(fixed_agent_storage_mapping_5[idx])
                elif not updates_recorded_mapping_5[idx]:
                    fixed_agent_storage_mapping_5[idx] = copy.deepcopy(w5)
                    updates_recorded_mapping_5[idx] = True

            if idx in fixed_agent_7:
                if updates_recorded_mapping_7[idx]:
                    w7 = copy.deepcopy(fixed_agent_storage_mapping_7[idx])
                elif not updates_recorded_mapping_7[idx]:
                    fixed_agent_storage_mapping_7[idx] = copy.deepcopy(w7)
                    updates_recorded_mapping_7[idx] = True

            if idx in fixed_agent_10:
                if updates_recorded_mapping_10[idx]:
                    w10 = copy.deepcopy(fixed_agent_storage_mapping_10[idx])
                elif not updates_recorded_mapping_10[idx]:
                    fixed_agent_storage_mapping_10[idx] = copy.deepcopy(w10)
                    updates_recorded_mapping_10[idx] = True


            #NO ATTACK
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            #1 MALICIOUS
            w_locals_1.append(copy.deepcopy(w1))
            loss_locals_1.append(copy.deepcopy(loss1))

            #20 MALICIOUS
            w_locals_5.append(copy.deepcopy(w5))
            loss_locals_5.append(copy.deepcopy(loss5))


            #30 MALICIOUS
            w_locals_7.append(copy.deepcopy(w7))
            loss_locals_7.append(copy.deepcopy(loss7))


            #30 MALICIOUS
            w_locals_10.append(copy.deepcopy(w10))
            loss_locals_10.append(copy.deepcopy(loss10))

        # update global weights
        w_glob = FedAvg(w_locals)
        w_glob1 = FedAvg(w_locals_1)
        w_glob5 = FedAvg(w_locals_5)
        w_glob7 = FedAvg(w_locals_7)
        w_glob10 = FedAvg(w_locals_10)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob1.load_state_dict(w_glob1)
        net_glob5.load_state_dict(w_glob5)
        net_glob7.load_state_dict(w_glob7)
        net_glob10.load_state_dict(w_glob10)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_avg_1 = sum(loss_locals_1) / len(loss_locals_1)
        loss_avg_5 = sum(loss_locals_5) / len(loss_locals_5)
        loss_avg_7 = sum(loss_locals_7) / len(loss_locals_7)
        loss_avg_10 = sum(loss_locals_10) / len(loss_locals_10)


        print('NO ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('C1 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_1))
        print('C5 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_5))
        print('C7 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_7))
        print('C10 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_10))
        

        #count_array.append(agent_found_count)
        loss_train.append(loss_avg)
        loss_train_1.append(loss_avg_1)
        loss_train_5.append(loss_avg_5)
        loss_train_7.append(loss_avg_7)
        loss_train_10.append(loss_avg_10)

    # plot loss curve
    plt.figure()
    attack_no = plt.plot(range(len(loss_train)), loss_train, label="0 malicious")
    attack_no = plt.plot(range(len(loss_train_1)), loss_train_1, label="1 malicious")
    attack_no = plt.plot(range(len(loss_train_5)), loss_train_5, label="5 malicious")
    attack_no = plt.plot(range(len(loss_train_7)), loss_train_7, label="7 malicious")
    attack_no = plt.plot(range(len(loss_train_10)), loss_train_10, label="10 malicious")
    plt.xlabel('epochs')
    plt.ylabel('train_loss')
    plt.savefig('log/fed_{}_{}_{}_C{}_iid{}_train_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close()

    #print("COUNT DATA",str(count_array))
    


    print("NO ATTACK DATA=",loss_train)
    print("1 ATTACK DATA=",loss_train_1)
    print("5 ATTACK DATA=",loss_train_5)
    print("7 ATTACK DATA=",loss_train_7)
    print("10 ATTACK DATA=",loss_train_10)


    # testing
    net_glob.eval()
    #print("Agent_Found_Count",agent_found_count)
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy (NO ATTACK): {:.2f}".format(acc_train))
    print("Testing accuracy (NO ATTACK): {:.2f}".format(acc_test))

    net_glob1.eval()
    acc_train1, loss_train_1 = test_img(net_glob1, dataset_train, args)
    acc_test1, loss_test_1 = test_img(net_glob1, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 1): {:.2f}".format(acc_train1))
    print("Testing accuracy (CONSTANT ATTACK 1): {:.2f}".format(acc_test1))


    net_glob5.eval()
    acc_train5, loss_train_5 = test_img(net_glob5, dataset_train, args)
    acc_test5, loss_test_5 = test_img(net_glob5, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 5): {:.2f}".format(acc_train5))
    print("Testing accuracy (CONSTANT ATTACK 5): {:.2f}".format(acc_test5))


    net_glob7.eval()
    acc_train7, loss_train_7 = test_img(net_glob7, dataset_train, args)
    acc_test7, loss_test_7 = test_img(net_glob7, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 7): {:.2f}".format(acc_train7))
    print("Testing accuracy (CONSTANT ATTACK 7): {:.2f}".format(acc_test7))


    net_glob10.eval()
    acc_train10, loss_train_10 = test_img(net_glob10, dataset_train, args)
    acc_test10, loss_test_10 = test_img(net_glob10, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_train10))
    print("Testing accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_test10))

    plt.figure()
    objects = ("no_mal","1_mal","5_mal","7_mal","10_mal")
    y_pos = np.arange(len(objects))
    performance = [acc_test,acc_test1,acc_test5,acc_test7,acc_test10]
    plt.bar(y_pos,performance,align='center',alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Analysis")
    plt.savefig('log/fed_{}_{}_{}_C{}_iid{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close
