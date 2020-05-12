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
        net_glob2 = CNNMnist(args=args).to(args.device)
        net_glob3 = CNNMnist(args=args).to(args.device)
        net_glob4 = CNNMnist(args=args).to(args.device)
        net_glob5 = CNNMnist(args=args).to(args.device)
        net_glob6 = CNNMnist(args=args).to(args.device)
        net_glob7 = CNNMnist(args=args).to(args.device)
        net_glob8 = CNNMnist(args=args).to(args.device)
        net_glob9 = CNNMnist(args=args).to(args.device)
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
    net_glob2.train()
    net_glob3.train()
    net_glob4.train()
    net_glob5.train()
    net_glob6.train()
    net_glob7.train()
    net_glob8.train()
    net_glob9.train()
    net_glob10.train()

    # copy weights
    w_glob = net_glob.state_dict()
    w_glob1 = net_glob1.state_dict()
    w_glob2 = net_glob2.state_dict()
    w_glob3 = net_glob3.state_dict()
    w_glob4 = net_glob4.state_dict()
    w_glob5 = net_glob5.state_dict()
    w_glob6 = net_glob6.state_dict()
    w_glob7 = net_glob7.state_dict()
    w_glob8 = net_glob8.state_dict()
    w_glob9 = net_glob9.state_dict()
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

    #VIVEK constant attack experiment - 5 MALICIOUS
    loss_train_2 = []
    fixed_agent_2 = random.sample(range(10),2)
    updates_recorded_mapping_2 = defaultdict(bool)
    for i in fixed_agent_2:
        updates_recorded_mapping_2[i] = False  #KEY = agent no. & VAL = boolean
    fixed_agent_storage_mapping_2 = {} #KEY = agent no. & VAL = Fixed Updates
    count_array_2 = []

    #VIVEK constant attack experiment - 10 MALICIOUS
    loss_train_3 = []
    fixed_agent_3 = random.sample(range(10),3)
    updates_recorded_mapping_3 = defaultdict(bool)
    for i in fixed_agent_3:
        updates_recorded_mapping_3[i] = False
    fixed_agent_storage_mapping_3 = {}
    count_array_3 = []

    #VIVEK constant attack experiment - 15 MALICIOUS
    loss_train_4 = []
    fixed_agent_4 = random.sample(range(10),4)
    updates_recorded_mapping_4 = defaultdict(bool)
    for i in fixed_agent_4:
        updates_recorded_mapping_4[i] = False
    fixed_agent_storage_mapping_4 = {}
    count_array_4 = []

    #VIVEK constant attack experiment - 20 MALICIOUS
    loss_train_5 = []
    fixed_agent_5 = random.sample(range(10),5)
    updates_recorded_mapping_5 = defaultdict(bool)
    for i in fixed_agent_5:
        updates_recorded_mapping_5[i] = False
    fixed_agent_storage_mapping_5 = {}
    count_array_5 = []

    #VIVEK constant attack experiment - 25 MALICIOUS
    loss_train_6 = []
    fixed_agent_6 = random.sample(range(10),6)
    updates_recorded_mapping_6 = defaultdict(bool)
    for i in fixed_agent_6:
        updates_recorded_mapping_6[i] = False
    fixed_agent_storage_mapping_6 = {}
    count_array_6 = []

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_7 = []
    fixed_agent_7 = random.sample(range(10),7)
    updates_recorded_mapping_7 = defaultdict(bool)
    for i in fixed_agent_7:
        updates_recorded_mapping_7[i] = False
    fixed_agent_storage_mapping_7 = {}
    count_array_7 = []

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_8 = []
    fixed_agent_8 = random.sample(range(10),8)
    updates_recorded_mapping_8 = defaultdict(bool)
    for i in fixed_agent_8:
        updates_recorded_mapping_8[i] = False
    fixed_agent_storage_mapping_8 = {}
    count_array_8 = []

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_9 = []
    fixed_agent_9 = random.sample(range(10),9)
    updates_recorded_mapping_9 = defaultdict(bool)
    for i in fixed_agent_9:
        updates_recorded_mapping_9[i] = False
    fixed_agent_storage_mapping_9 = {}
    count_array_9 = []

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
        w_locals_2, loss_locals_2 = [],[]
        w_locals_3, loss_locals_3 = [],[]
        w_locals_4, loss_locals_4 = [],[]
        w_locals_5, loss_locals_5 = [],[]
        w_locals_6, loss_locals_6 = [],[]
        w_locals_7, loss_locals_7 = [],[]
        w_locals_8, loss_locals_8 = [],[]
        w_locals_9, loss_locals_9 = [],[]
        w_locals_10, loss_locals_10 = [],[]
        m = max(int(args.frac * args.num_users), 1)     #m = number of users used in one ROUND/EPOCH, check utils.options for more clarity on this
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #Randomly selecting m users out of 32 users. NEED TO REPLACE THIS WITH OUR SAMPLING MECHANISM

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local1 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local2 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local3 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local4 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local5 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local6 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local7 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local8 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local9 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local10 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w1, loss1 = local1.train(net=copy.deepcopy(net_glob1).to(args.device))
            w2, loss2 = local2.train(net=copy.deepcopy(net_glob2).to(args.device))
            w3, loss3 = local3.train(net=copy.deepcopy(net_glob3).to(args.device))
            w4, loss4 = local4.train(net=copy.deepcopy(net_glob4).to(args.device))
            w5, loss5 = local5.train(net=copy.deepcopy(net_glob5).to(args.device))
            w6, loss6 = local6.train(net=copy.deepcopy(net_glob6).to(args.device))
            w7, loss7 = local7.train(net=copy.deepcopy(net_glob7).to(args.device))
            w8, loss8 = local8.train(net=copy.deepcopy(net_glob8).to(args.device))
            w9, loss9 = local9.train(net=copy.deepcopy(net_glob9).to(args.device))
            w10, loss10 = local10.train(net=copy.deepcopy(net_glob10).to(args.device))
            print("***BLAH BLAH BLAH***")


            if idx==fixed_agent_1:
                if updates_recorded_1:
                    w1 = copy.deepcopy(fixed_agent_storage_1)
                elif not updates_recorded_1:
                    fixed_agent_storage_1 = copy.deepcopy(w1)
                    updates_recorded_1 = True

            if idx in fixed_agent_2:
                if updates_recorded_mapping_2[idx]:
                    w2 = copy.deepcopy(fixed_agent_storage_mapping_2[idx])
                elif not updates_recorded_mapping_2[idx]:
                    fixed_agent_storage_mapping_2[idx] = copy.deepcopy(w2)
                    updates_recorded_mapping_2[idx] = True

            if idx in fixed_agent_3:
                if updates_recorded_mapping_3[idx]:
                    w3 = copy.deepcopy(fixed_agent_storage_mapping_3[idx])
                elif not updates_recorded_mapping_3[idx]:
                    fixed_agent_storage_mapping_3[idx] = copy.deepcopy(w3)
                    updates_recorded_mapping_3[idx] = True

            if idx in fixed_agent_4:
                if updates_recorded_mapping_4[idx]:
                    w4 = copy.deepcopy(fixed_agent_storage_mapping_4[idx])
                elif not updates_recorded_mapping_4[idx]:
                    fixed_agent_storage_mapping_4[idx] = copy.deepcopy(w4)
                    updates_recorded_mapping_4[idx] = True

            if idx in fixed_agent_5:
                if updates_recorded_mapping_5[idx]:
                    w5 = copy.deepcopy(fixed_agent_storage_mapping_5[idx])
                elif not updates_recorded_mapping_5[idx]:
                    fixed_agent_storage_mapping_5[idx] = copy.deepcopy(w5)
                    updates_recorded_mapping_5[idx] = True

            if idx in fixed_agent_6:
                if updates_recorded_mapping_6[idx]:
                    w6 = copy.deepcopy(fixed_agent_storage_mapping_6[idx])
                elif not updates_recorded_mapping_6[idx]:
                    fixed_agent_storage_mapping_6[idx] = copy.deepcopy(w6)
                    updates_recorded_mapping_6[idx] = True

            if idx in fixed_agent_7:
                if updates_recorded_mapping_7[idx]:
                    w7 = copy.deepcopy(fixed_agent_storage_mapping_7[idx])
                elif not updates_recorded_mapping_7[idx]:
                    fixed_agent_storage_mapping_7[idx] = copy.deepcopy(w7)
                    updates_recorded_mapping_7[idx] = True

            if idx in fixed_agent_8:
                if updates_recorded_mapping_8[idx]:
                    w8 = copy.deepcopy(fixed_agent_storage_mapping_8[idx])
                elif not updates_recorded_mapping_8[idx]:
                    fixed_agent_storage_mapping_8[idx] = copy.deepcopy(w8)
                    updates_recorded_mapping_8[idx] = True

            if idx in fixed_agent_9:
                if updates_recorded_mapping_9[idx]:
                    w9 = copy.deepcopy(fixed_agent_storage_mapping_9[idx])
                elif not updates_recorded_mapping_9[idx]:
                    fixed_agent_storage_mapping_9[idx] = copy.deepcopy(w9)
                    updates_recorded_mapping_9[idx] = True

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

            #5 MALICIOUS
            w_locals_2.append(copy.deepcopy(w2))
            loss_locals_2.append(copy.deepcopy(loss2))

            #10 MALICIOUS
            w_locals_3.append(copy.deepcopy(w3))
            loss_locals_3.append(copy.deepcopy(loss3))

            #15 MALICIOUS
            w_locals_4.append(copy.deepcopy(w4))
            loss_locals_4.append(copy.deepcopy(loss4))

            #20 MALICIOUS
            w_locals_5.append(copy.deepcopy(w5))
            loss_locals_5.append(copy.deepcopy(loss5))

            #25 MALICIOUS
            w_locals_6.append(copy.deepcopy(w6))
            loss_locals_6.append(copy.deepcopy(loss6))

            #30 MALICIOUS
            w_locals_7.append(copy.deepcopy(w7))
            loss_locals_7.append(copy.deepcopy(loss7))

            #30 MALICIOUS
            w_locals_8.append(copy.deepcopy(w8))
            loss_locals_8.append(copy.deepcopy(loss8))

            #30 MALICIOUS
            w_locals_9.append(copy.deepcopy(w9))
            loss_locals_9.append(copy.deepcopy(loss9))

            #30 MALICIOUS
            w_locals_10.append(copy.deepcopy(w10))
            loss_locals_10.append(copy.deepcopy(loss10))

        # update global weights
        w_glob = FedAvg(w_locals)
        w_glob1 = FedAvg(w_locals_1)
        w_glob2 = FedAvg(w_locals_2)
        w_glob3 = FedAvg(w_locals_3)
        w_glob4 = FedAvg(w_locals_4)
        w_glob5 = FedAvg(w_locals_5)
        w_glob6 = FedAvg(w_locals_6)
        w_glob7 = FedAvg(w_locals_7)
        w_glob8 = FedAvg(w_locals_8)
        w_glob_9 = FedAvg(w_locals_9)
        w_glob30 = FedAvg(w_locals_10)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob1.load_state_dict(w_glob1)
        net_glob2.load_state_dict(w_glob2)
        net_glob3.load_state_dict(w_glob3)
        net_glob4.load_state_dict(w_glob4)
        net_glob5.load_state_dict(w_glob5)
        net_glob6.load_state_dict(w_glob6)
        net_glob7.load_state_dict(w_glob7)
        net_glob8.load_state_dict(w_glob8)
        net_glob9.load_state_dict(w_glob9)
        net_glob10.load_state_dict(w_glob10)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_avg_1 = sum(loss_locals_1) / len(loss_locals_1)
        loss_avg_2 = sum(loss_locals_2) / len(loss_locals_2)
        loss_avg_3 = sum(loss_locals_3) / len(loss_locals_3)
        loss_avg_4 = sum(loss_locals_4) / len(loss_locals_4)
        loss_avg_5 = sum(loss_locals_5) / len(loss_locals_5)
        loss_avg_6 = sum(loss_locals_6) / len(loss_locals_6)
        loss_avg_7 = sum(loss_locals_7) / len(loss_locals_7)
        loss_avg_8 = sum(loss_locals_8) / len(loss_locals_8)
        loss_avg_9 = sum(loss_locals_9) / len(loss_locals_9)
        loss_avg_10 = sum(loss_locals_10) / len(loss_locals_10)


        print('NO ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('C1 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_1))
        print('C2 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_2))
        print('C3 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_3))
        print('C4 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_4))
        print('C5 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_5))
        print('C6 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_6))
        print('C7 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_7))
        print('C8 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_8))
        print('C9 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_9))
        print('C10 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_10))
        

        #count_array.append(agent_found_count)
        loss_train.append(loss_avg)
        loss_train_1.append(loss_avg_1)
        loss_train_2.append(loss_avg_2)
        loss_train_3.append(loss_avg_3)
        loss_train_4.append(loss_avg_4)
        loss_train_5.append(loss_avg_5)
        loss_train_6.append(loss_avg_6)
        loss_train_7.append(loss_avg_7)
        loss_train_8.append(loss_avg_8)
        loss_train_9.append(loss_avg_9)
        loss_train_10.append(loss_avg_10)

    # plot loss curve
    plt.figure()
    attack_no = plt.plot(range(len(loss_train)), loss_train, label="0 malicious")
    attack_no = plt.plot(range(len(loss_train_1)), loss_train_1, label="1 malicious")
    attack_no = plt.plot(range(len(loss_train_2)), loss_train_2, label="2 malicious")
    attack_no = plt.plot(range(len(loss_train_3)), loss_train_3, label="3 malicious")
    attack_no = plt.plot(range(len(loss_train_4)), loss_train_4, label="4 malicious")
    attack_no = plt.plot(range(len(loss_train_5)), loss_train_5, label="5 malicious")
    attack_no = plt.plot(range(len(loss_train_6)), loss_train_6, label="6 malicious")
    attack_no = plt.plot(range(len(loss_train_7)), loss_train_7, label="7 malicious")
    attack_no = plt.plot(range(len(loss_train_8)), loss_train_8, label="8 malicious")
    attack_no = plt.plot(range(len(loss_train_9)), loss_train_9, label="9 malicious")
    attack_no = plt.plot(range(len(loss_train_10)), loss_train_10, label="10 malicious")
    plt.xlabel('epochs')
    plt.ylabel('train_loss')
    plt.savefig('log/fed_{}_{}_{}_C{}_iid{}_train_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close()

    #print("COUNT DATA",str(count_array))
    


    print("NO ATTACK DATA=",loss_train)
    print("1 ATTACK DATA=",loss_train_1)
    print("2 ATTACK DATA=",loss_train_2)
    print("3 ATTACK DATA=",loss_train_3)
    print("4 ATTACK DATA=",loss_train_4)
    print("5 ATTACK DATA=",loss_train_5)
    print("6 ATTACK DATA=",loss_train_6)
    print("7 ATTACK DATA=",loss_train_7)
    print("8 ATTACK DATA=",loss_train_8)
    print("9 ATTACK DATA=",loss_train_9)
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


    net_glob2.eval()
    acc_train2, loss_train_2 = test_img(net_glob2, dataset_train, args)
    acc_test2, loss_test_2 = test_img(net_glob2, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 2): {:.2f}".format(acc_train2))
    print("Testing accuracy (CONSTANT ATTACK 2): {:.2f}".format(acc_test2))

    net_glob3.eval()
    acc_train3, loss_train_3 = test_img(net_glob3, dataset_train, args)
    acc_test3, loss_test_3 = test_img(net_glob3, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 3): {:.2f}".format(acc_train3))
    print("Testing accuracy (CONSTANT ATTACK 3): {:.2f}".format(acc_test3))

    net_glob4.eval()
    acc_train4, loss_train_4 = test_img(net_glob4, dataset_train, args)
    acc_test4, loss_test_4 = test_img(net_glob4, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 4): {:.2f}".format(acc_train4))
    print("Testing accuracy (CONSTANT ATTACK 4): {:.2f}".format(acc_test4))

    net_glob5.eval()
    acc_train5, loss_train_5 = test_img(net_glob5, dataset_train, args)
    acc_test5, loss_test_5 = test_img(net_glob5, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 5): {:.2f}".format(acc_train5))
    print("Testing accuracy (CONSTANT ATTACK 5): {:.2f}".format(acc_test5))

    net_glob6.eval()
    acc_train6, loss_train_6 = test_img(net_glob6, dataset_train, args)
    acc_test6, loss_test_6 = test_img(net_glob6, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 6): {:.2f}".format(acc_train6))
    print("Testing accuracy (CONSTANT ATTACK 6): {:.2f}".format(acc_test6))

    net_glob7.eval()
    acc_train7, loss_train_7 = test_img(net_glob7, dataset_train, args)
    acc_test7, loss_test_7 = test_img(net_glob7, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 7): {:.2f}".format(acc_train7))
    print("Testing accuracy (CONSTANT ATTACK 7): {:.2f}".format(acc_test7))

    net_glob8.eval()
    acc_train8, loss_train_8 = test_img(net_glob8, dataset_train, args)
    acc_test8, loss_test_8 = test_img(net_glob8, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 8): {:.2f}".format(acc_train8))
    print("Testing accuracy (CONSTANT ATTACK 8): {:.2f}".format(acc_test8))

    net_glob9.eval()
    acc_train9, loss_train_9 = test_img(net_glob9, dataset_train, args)
    acc_test9, loss_test_9 = test_img(net_glob9, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 9): {:.2f}".format(acc_train9))
    print("Testing accuracy (CONSTANT ATTACK 9): {:.2f}".format(acc_test9))

    net_glob10.eval()
    acc_train10, loss_train_10 = test_img(net_glob10, dataset_train, args)
    acc_test10, loss_test_10 = test_img(net_glob10, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_train10))
    print("Testing accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_test10))

    plt.figure()
    objects = ("no_mal","1_mal","2_mal","3_mal","4_mal","5_mal","6_mal","7_mal","8_mal","9_mal","10_mal")
    y_pos = np.arange(len(objects))
    performance = [acc_test,acc_test1,acc_test2,acc_test3,acc_test4,acc_test5,acc_test6,acc_test7,acc_test8,acc_test9,acc_test10]
    plt.bar(y_pos,performance,align='center',alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Analysis")
    plt.savefig('log/fed_{}_{}_{}_C{}_iid{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close
