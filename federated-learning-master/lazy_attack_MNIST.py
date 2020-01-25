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
import csv

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from collections import OrderedDict,defaultdict

from torch.autograd import Variable

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
        net_glob10 = CNNMnist(args=args).to(args.device)
        net_glob15 = CNNMnist(args=args).to(args.device)
        net_glob20 = CNNMnist(args=args).to(args.device)
        net_glob25 = CNNMnist(args=args).to(args.device)
        net_glob30 = CNNMnist(args=args).to(args.device)
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
    net_glob10.train()
    net_glob15.train()
    net_glob20.train()
    net_glob25.train()
    net_glob30.train()

    # copy weights
    w_glob = net_glob.state_dict()
    w_glob1 = net_glob1.state_dict()
    w_glob5 = net_glob5.state_dict()
    w_glob10 = net_glob10.state_dict()
    w_glob15 = net_glob15.state_dict()
    w_glob20 = net_glob20.state_dict()
    w_glob25 = net_glob25.state_dict()
    w_glob30 = net_glob30.state_dict()

    # training - NO ATTACK
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    #VIVEK constant attack experiment - 1 MALICIOUS
    loss_train_1 = []
    fixed_agent_1 = random.randint(0,31)  #random agent between 0 and 31 is fixed
    updates_recorded_1 = False
    fixed_agent_storage_1 = None
    count_array_1 = []
    selection_count_1 = 0

    #VIVEK constant attack experiment - 5 MALICIOUS
    loss_train_5 = []
    fixed_agent_5 = random.sample(range(32),5)
    updates_recorded_mapping_5 = defaultdict(bool)
    selection_count_5 = defaultdict(int)
    for i in fixed_agent_5:
        updates_recorded_mapping_5[i] = False  #KEY = agent no. & VAL = boolean
        selection_count_5[i] = 0
    fixed_agent_storage_mapping_5 = {} #KEY = agent no. & VAL = Fixed Updates
    count_array_5 = []


    #VIVEK constant attack experiment - 10 MALICIOUS
    loss_train_10 = []
    fixed_agent_10 = random.sample(range(32),10)
    updates_recorded_mapping_10 = defaultdict(bool)
    selection_count_10 = defaultdict(int)
    for i in fixed_agent_10:
        updates_recorded_mapping_10[i] = False
        selection_count_10[i] = 0
    fixed_agent_storage_mapping_10 = {}
    count_array_10 = []

    #VIVEK constant attack experiment - 15 MALICIOUS
    loss_train_15 = []
    fixed_agent_15 = random.sample(range(32),15)
    updates_recorded_mapping_15 = defaultdict(bool)
    selection_count_15 = defaultdict(int)
    for i in fixed_agent_15:
        updates_recorded_mapping_15[i] = False
        selection_count_15[i] = 0
    fixed_agent_storage_mapping_15 = {}
    count_array_15 = []

    #VIVEK constant attack experiment - 20 MALICIOUS
    loss_train_20 = []
    fixed_agent_20 = random.sample(range(32),20)
    updates_recorded_mapping_20 = defaultdict(bool)
    selection_count_20 = defaultdict(int)
    for i in fixed_agent_20:
        updates_recorded_mapping_20[i] = False
        selection_count_20[i] = 0
    fixed_agent_storage_mapping_20 = {}
    count_array_20 = []

    #VIVEK constant attack experiment - 25 MALICIOUS
    loss_train_25 = []
    fixed_agent_25 = random.sample(range(32),25)
    updates_recorded_mapping_25 = defaultdict(bool)
    selection_count_25 = defaultdict(int)
    for i in fixed_agent_25:
        updates_recorded_mapping_25[i] = False
        selection_count_25[i] = 0
    fixed_agent_storage_mapping_25 = {}
    count_array_25 = []

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_30 = []
    fixed_agent_30 = random.sample(range(32),30)
    updates_recorded_mapping_30 = defaultdict(bool)
    selection_count_30 = defaultdict(int)
    for i in fixed_agent_30:
        updates_recorded_mapping_30[i] = False
        selection_count_30[i] = 0
    fixed_agent_storage_mapping_30 = {}
    count_array_30 = []

    malicious_dict = {1:[fixed_agent_1],5:fixed_agent_5,10:fixed_agent_10,15:fixed_agent_15,20:fixed_agent_20,25:fixed_agent_25,30:fixed_agent_30}
    malicious_count ={1:0,5:0,10:0,15:0,20:0,25:0,30:0}

    for iter in range(args.epochs):
        #agent_found_count = 0
        w_locals, loss_locals = [], []          #w_locals = array of local_weights
        w_locals_1, loss_locals_1 = [],[]
        w_locals_5, loss_locals_5 = [],[]
        w_locals_10, loss_locals_10 = [],[]
        w_locals_15, loss_locals_15 = [],[]
        w_locals_20, loss_locals_20 = [],[]
        w_locals_25, loss_locals_25 = [],[]
        w_locals_30, loss_locals_30 = [],[]
        m = max(int(args.frac * args.num_users), 1)     #m = number of users used in one ROUND/EPOCH, check utils.options for more clarity on this
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #Randomly selecting m users out of 32 users. NEED TO REPLACE THIS WITH OUR SAMPLING MECHANISM

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local1 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local5 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local10 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local15 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local20 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local25 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local30 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w1, loss1 = local1.train(net=copy.deepcopy(net_glob1).to(args.device))
            w5, loss5 = local5.train(net=copy.deepcopy(net_glob5).to(args.device))
            w10, loss10 = local10.train(net=copy.deepcopy(net_glob10).to(args.device))
            w15, loss15 = local15.train(net=copy.deepcopy(net_glob15).to(args.device))
            w20, loss20 = local20.train(net=copy.deepcopy(net_glob20).to(args.device))
            w25, loss25 = local25.train(net=copy.deepcopy(net_glob25).to(args.device))
            w30, loss30 = local30.train(net=copy.deepcopy(net_glob30).to(args.device))
            print("***BLAH BLAH BLAH***")


            if idx==fixed_agent_1:
                if w_glob1:
                    conv_1_weight_mean = w_glob1['conv1.weight'].mean()
                    conv_1_weight_std = w_glob1['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob1['conv1.bias'].mean()
                    conv_1_bias_std = w_glob1['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob1['conv2.weight'].mean()
                    conv_2_weight_std = w_glob1['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob1['conv2.bias'].mean()
                    conv_2_bias_std = w_glob1['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob1['fc1.weight'].mean()
                    fc1_weight_std = w_glob1['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob1['fc1.bias'].mean()
                    fc1_bias_std = w_glob1['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob1['fc2.weight'].mean()
                    fc2_weight_std = w_glob1['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob1['fc2.bias'].mean()
                    fc2_bias_std = w_glob1['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob1['conv1.weight'].data.new(w_glob1['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob1['conv1.bias'].data.new(w_glob1['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob1['conv2.weight'].data.new(w_glob1['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob1['conv2.bias'].data.new(w_glob1['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob1['fc1.weight'].data.new(w_glob1['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob1['fc1.bias'].data.new(w_glob1['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob1['fc2.weight'].data.new(w_glob1['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob1['fc2.bias'].data.new(w_glob1['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w1['conv1.weight']=w_glob1['conv1.weight']+noise_conv1_weight
                    w1['conv1.bias']=w_glob1['conv1.bias']+noise_conv1_bias
                    w1['conv2.weight']=w_glob1['conv2.weight']+noise_conv2_weight
                    w1['conv2.bias']=w_glob1['conv2.bias']+noise_conv2_bias
                    w1['fc1.weight']=w_glob1['fc1.weight']+noise_fc1_weight
                    w1['fc1.bias']=w_glob1['fc1.bias']+noise_fc1_bias
                    w1['fc2.weight']=w_glob1['fc2.weight']+noise_fc2_weight
                    w1['fc2.bias']=w_glob1['fc2.bias']+noise_fc2_bias

            if idx in fixed_agent_5:
                malicious_count[5]+=1
                if w_glob5:
                    conv_1_weight_mean = w_glob5['conv1.weight'].mean()
                    conv_1_weight_std = w_glob5['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob5['conv1.bias'].mean()
                    conv_1_bias_std = w_glob5['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob5['conv2.weight'].mean()
                    conv_2_weight_std = w_glob5['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob5['conv2.bias'].mean()
                    conv_2_bias_std = w_glob5['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob5['fc1.weight'].mean()
                    fc1_weight_std = w_glob5['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob5['fc1.bias'].mean()
                    fc1_bias_std = w_glob5['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob5['fc2.weight'].mean()
                    fc2_weight_std = w_glob5['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob5['fc2.bias'].mean()
                    fc2_bias_std = w_glob5['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob5['conv1.weight'].data.new(w_glob5['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob5['conv1.bias'].data.new(w_glob5['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob5['conv2.weight'].data.new(w_glob5['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob5['conv2.bias'].data.new(w_glob5['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob5['fc1.weight'].data.new(w_glob5['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob5['fc1.bias'].data.new(w_glob5['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob5['fc2.weight'].data.new(w_glob5['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob5['fc2.bias'].data.new(w_glob5['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w5['conv1.weight']=w_glob5['conv1.weight']+noise_conv1_weight
                    w5['conv1.bias']=w_glob5['conv1.bias']+noise_conv1_bias
                    w5['conv2.weight']=w_glob5['conv2.weight']+noise_conv2_weight
                    w5['conv2.bias']=w_glob5['conv2.bias']+noise_conv2_bias
                    w5['fc1.weight']=w_glob5['fc1.weight']+noise_fc1_weight
                    w5['fc1.bias']=w_glob5['fc1.bias']+noise_fc1_bias
                    w5['fc2.weight']=w_glob5['fc2.weight']+noise_fc2_weight
                    w5['fc2.bias']=w_glob5['fc2.bias']+noise_fc2_bias

            if idx in fixed_agent_10:
                malicious_count[10]+=1
                if w_glob10:
                    conv_1_weight_mean = w_glob10['conv1.weight'].mean()
                    conv_1_weight_std = w_glob10['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob10['conv1.bias'].mean()
                    conv_1_bias_std = w_glob10['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob10['conv2.weight'].mean()
                    conv_2_weight_std = w_glob10['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob10['conv2.bias'].mean()
                    conv_2_bias_std = w_glob10['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob10['fc1.weight'].mean()
                    fc1_weight_std = w_glob10['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob10['fc1.bias'].mean()
                    fc1_bias_std = w_glob10['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob10['fc2.weight'].mean()
                    fc2_weight_std = w_glob10['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob10['fc2.bias'].mean()
                    fc2_bias_std = w_glob10['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob10['conv1.weight'].data.new(w_glob10['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob10['conv1.bias'].data.new(w_glob10['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob10['conv2.weight'].data.new(w_glob10['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob10['conv2.bias'].data.new(w_glob10['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob10['fc1.weight'].data.new(w_glob10['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob10['fc1.bias'].data.new(w_glob10['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob10['fc2.weight'].data.new(w_glob10['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob10['fc2.bias'].data.new(w_glob10['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w10['conv1.weight']=w_glob10['conv1.weight']+noise_conv1_weight
                    w10['conv1.bias']=w_glob10['conv1.bias']+noise_conv1_bias
                    w10['conv2.weight']=w_glob10['conv2.weight']+noise_conv2_weight
                    w10['conv2.bias']=w_glob10['conv2.bias']+noise_conv2_bias
                    w10['fc1.weight']=w_glob10['fc1.weight']+noise_fc1_weight
                    w10['fc1.bias']=w_glob10['fc1.bias']+noise_fc1_bias
                    w10['fc2.weight']=w_glob10['fc2.weight']+noise_fc2_weight
                    w10['fc2.bias']=w_glob10['fc2.bias']+noise_fc2_bias

            if idx in fixed_agent_15:
                malicious_count[15]+=1
                if w_glob15:
                    conv_1_weight_mean = w_glob15['conv1.weight'].mean()
                    conv_1_weight_std = w_glob15['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob15['conv1.bias'].mean()
                    conv_1_bias_std = w_glob15['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob15['conv2.weight'].mean()
                    conv_2_weight_std = w_glob15['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob15['conv2.bias'].mean()
                    conv_2_bias_std = w_glob15['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob15['fc1.weight'].mean()
                    fc1_weight_std = w_glob15['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob15['fc1.bias'].mean()
                    fc1_bias_std = w_glob15['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob15['fc2.weight'].mean()
                    fc2_weight_std = w_glob15['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob15['fc2.bias'].mean()
                    fc2_bias_std = w_glob15['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob15['conv1.weight'].data.new(w_glob15['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob15['conv1.bias'].data.new(w_glob15['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob15['conv2.weight'].data.new(w_glob15['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob15['conv2.bias'].data.new(w_glob15['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob15['fc1.weight'].data.new(w_glob15['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob15['fc1.bias'].data.new(w_glob15['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob15['fc2.weight'].data.new(w_glob15['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob15['fc2.bias'].data.new(w_glob15['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w15['conv1.weight']=w_glob15['conv1.weight']+noise_conv1_weight
                    w15['conv1.bias']=w_glob15['conv1.bias']+noise_conv1_bias
                    w15['conv2.weight']=w_glob15['conv2.weight']+noise_conv2_weight
                    w15['conv2.bias']=w_glob15['conv2.bias']+noise_conv2_bias
                    w15['fc1.weight']=w_glob15['fc1.weight']+noise_fc1_weight
                    w15['fc1.bias']=w_glob15['fc1.bias']+noise_fc1_bias
                    w15['fc2.weight']=w_glob15['fc2.weight']+noise_fc2_weight
                    w15['fc2.bias']=w_glob15['fc2.bias']+noise_fc2_bias

            if idx in fixed_agent_20:
                malicious_count[20]+=1
                if w_glob20:
                    conv_1_weight_mean = w_glob20['conv1.weight'].mean()
                    conv_1_weight_std = w_glob20['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob20['conv1.bias'].mean()
                    conv_1_bias_std = w_glob20['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob20['conv2.weight'].mean()
                    conv_2_weight_std = w_glob20['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob20['conv2.bias'].mean()
                    conv_2_bias_std = w_glob20['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob20['fc1.weight'].mean()
                    fc1_weight_std = w_glob20['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob20['fc1.bias'].mean()
                    fc1_bias_std = w_glob20['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob20['fc2.weight'].mean()
                    fc2_weight_std = w_glob20['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob20['fc2.bias'].mean()
                    fc2_bias_std = w_glob20['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob20['conv1.weight'].data.new(w_glob20['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob20['conv1.bias'].data.new(w_glob20['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob20['conv2.weight'].data.new(w_glob20['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob20['conv2.bias'].data.new(w_glob20['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob20['fc1.weight'].data.new(w_glob20['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob20['fc1.bias'].data.new(w_glob20['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob20['fc2.weight'].data.new(w_glob20['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob20['fc2.bias'].data.new(w_glob20['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w20['conv1.weight']=w_glob20['conv1.weight']+noise_conv1_weight
                    w20['conv1.bias']=w_glob20['conv1.bias']+noise_conv1_bias
                    w20['conv2.weight']=w_glob20['conv2.weight']+noise_conv2_weight
                    w20['conv2.bias']=w_glob20['conv2.bias']+noise_conv2_bias
                    w20['fc1.weight']=w_glob20['fc1.weight']+noise_fc1_weight
                    w20['fc1.bias']=w_glob20['fc1.bias']+noise_fc1_bias
                    w20['fc2.weight']=w_glob20['fc2.weight']+noise_fc2_weight
                    w20['fc2.bias']=w_glob20['fc2.bias']+noise_fc2_bias

            if idx in fixed_agent_25:
                malicious_count[25]+=1
                if w_glob25:
                    conv_1_weight_mean = w_glob25['conv1.weight'].mean()
                    conv_1_weight_std = w_glob25['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob25['conv1.bias'].mean()
                    conv_1_bias_std = w_glob25['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob25['conv2.weight'].mean()
                    conv_2_weight_std = w_glob25['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob25['conv2.bias'].mean()
                    conv_2_bias_std = w_glob25['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob25['fc1.weight'].mean()
                    fc1_weight_std = w_glob25['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob25['fc1.bias'].mean()
                    fc1_bias_std = w_glob25['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob25['fc2.weight'].mean()
                    fc2_weight_std = w_glob25['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob25['fc2.bias'].mean()
                    fc2_bias_std = w_glob25['fc2.bias'].std(unbiased=False)
                    
                    noise_conv1_weight = Variable(w_glob25['conv1.weight'].data.new(w_glob25['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob25['conv1.bias'].data.new(w_glob25['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob25['conv2.weight'].data.new(w_glob25['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob25['conv2.bias'].data.new(w_glob25['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob25['fc1.weight'].data.new(w_glob25['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob25['fc1.bias'].data.new(w_glob25['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob25['fc2.weight'].data.new(w_glob25['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob25['fc2.bias'].data.new(w_glob25['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w25['conv1.weight']=w_glob25['conv1.weight']+noise_conv1_weight
                    w25['conv1.bias']=w_glob25['conv1.bias']+noise_conv1_bias
                    w25['conv2.weight']=w_glob25['conv2.weight']+noise_conv2_weight
                    w25['conv2.bias']=w_glob25['conv2.bias']+noise_conv2_bias
                    w25['fc1.weight']=w_glob25['fc1.weight']+noise_fc1_weight
                    w25['fc1.bias']=w_glob25['fc1.bias']+noise_fc1_bias
                    w25['fc2.weight']=w_glob25['fc2.weight']+noise_fc2_weight
                    w225['fc2.bias']=w_glob25['fc2.bias']+noise_fc2_bias


            if idx in fixed_agent_30:
                malicious_count[30]+=1
                #orig_glob = copy.deepcopy(w_glob)
                if w_glob30:
                    conv_1_weight_mean = w_glob30['conv1.weight'].mean()
                    conv_1_weight_std = w_glob30['conv1.weight'].std(unbiased=False)
                    conv_1_bias_mean = w_glob30['conv1.bias'].mean()
                    conv_1_bias_std = w_glob30['conv1.bias'].std(unbiased=False)

                    conv_2_weight_mean = w_glob30['conv2.weight'].mean()
                    conv_2_weight_std = w_glob30['conv2.weight'].std(unbiased=False)
                    conv_2_bias_mean = w_glob30['conv2.bias'].mean()
                    conv_2_bias_std = w_glob30['conv2.bias'].std(unbiased=False)

                    fc1_weight_mean = w_glob30['fc1.weight'].mean()
                    fc1_weight_std = w_glob30['fc1.weight'].std(unbiased=False)
                    fc1_bias_mean = w_glob30['fc1.bias'].mean()
                    fc1_bias_std = w_glob30['fc1.bias'].std(unbiased=False)

                    fc2_weight_mean = w_glob30['fc2.weight'].mean()
                    fc2_weight_std = w_glob30['fc2.weight'].std(unbiased=False)
                    fc2_bias_mean = w_glob30['fc2.bias'].mean()
                    fc2_bias_std = w_glob30['fc2.bias'].std(unbiased=False)
                    

                    noise_conv1_weight = Variable(w_glob30['conv1.weight'].data.new(w_glob30['conv1.weight'].size()).normal_(conv_1_weight_mean,conv_1_weight_std))
                    noise_conv1_bias = Variable(w_glob30['conv1.bias'].data.new(w_glob30['conv1.bias'].size()).normal_(conv_1_bias_mean,conv_1_bias_std))
                    noise_conv2_weight =  Variable(w_glob30['conv2.weight'].data.new(w_glob30['conv2.weight'].size()).normal_(conv_2_weight_mean,conv_2_weight_std))
                    noise_conv2_bias =  Variable(w_glob30['conv2.bias'].data.new(w_glob30['conv2.bias'].size()).normal_(conv_2_bias_mean,conv_2_bias_std))
                    noise_fc1_weight = Variable(w_glob30['fc1.weight'].data.new(w_glob30['fc1.weight'].size()).normal_(fc1_weight_mean,fc1_weight_std))
                    noise_fc1_bias = Variable(w_glob30['fc1.bias'].data.new(w_glob30['fc1.bias'].size()).normal_(fc1_bias_mean,fc1_bias_std))
                    noise_fc2_weight = Variable(w_glob30['fc2.weight'].data.new(w_glob30['fc2.weight'].size()).normal_(fc2_weight_mean,fc2_weight_std))
                    noise_fc2_bias = Variable(w_glob30['fc2.bias'].data.new(w_glob30['fc2.bias'].size()).normal_(fc2_bias_mean,fc2_bias_std))

                    w30['conv1.weight']=w_glob30['conv1.weight']+noise_conv1_weight
                    w30['conv1.bias']=w_glob30['conv1.bias']+noise_conv1_bias
                    w30['conv2.weight']=w_glob30['conv2.weight']+noise_conv2_weight
                    w30['conv2.bias']=w_glob30['conv2.bias']+noise_conv2_bias
                    w30['fc1.weight']=w_glob30['fc1.weight']+noise_fc1_weight
                    w30['fc1.bias']=w_glob30['fc1.bias']+noise_fc1_bias
                    w30['fc2.weight']=w_glob30['fc2.weight']+noise_fc2_weight
                    w30['fc2.bias']=w_glob30['fc2.bias']+noise_fc2_bias




            #NO ATTACK
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            #1 MALICIOUS
            w_locals_1.append(copy.deepcopy(w1))
            loss_locals_1.append(copy.deepcopy(loss1))

            #5 MALICIOUS
            w_locals_5.append(copy.deepcopy(w5))
            loss_locals_5.append(copy.deepcopy(loss5))

            #10 MALICIOUS
            w_locals_10.append(copy.deepcopy(w10))
            loss_locals_10.append(copy.deepcopy(loss10))

            #15 MALICIOUS
            w_locals_15.append(copy.deepcopy(w15))
            loss_locals_15.append(copy.deepcopy(loss15))

            #20 MALICIOUS
            w_locals_20.append(copy.deepcopy(w20))
            loss_locals_20.append(copy.deepcopy(loss20))

            #25 MALICIOUS
            w_locals_25.append(copy.deepcopy(w25))
            loss_locals_25.append(copy.deepcopy(loss25))

            #30 MALICIOUS
            w_locals_30.append(copy.deepcopy(w30))
            loss_locals_30.append(copy.deepcopy(loss30))

        # update global weights
        w_glob = FedAvg(w_locals)
        w_glob1 = FedAvg(w_locals_1)
        w_glob5 = FedAvg(w_locals_5)
        w_glob10 = FedAvg(w_locals_10)
        w_glob15 = FedAvg(w_locals_15)
        w_glob20 = FedAvg(w_locals_20)
        w_glob25 = FedAvg(w_locals_25)
        w_glob30 = FedAvg(w_locals_30)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob1.load_state_dict(w_glob1)
        net_glob5.load_state_dict(w_glob5)
        net_glob10.load_state_dict(w_glob10)
        net_glob15.load_state_dict(w_glob15)
        net_glob20.load_state_dict(w_glob20)
        net_glob25.load_state_dict(w_glob25)
        net_glob30.load_state_dict(w_glob30)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_avg_1 = sum(loss_locals_1) / len(loss_locals_1)
        loss_avg_5 = sum(loss_locals_5) / len(loss_locals_5)
        loss_avg_10 = sum(loss_locals_10) / len(loss_locals_10)
        loss_avg_15 = sum(loss_locals_15) / len(loss_locals_15)
        loss_avg_20 = sum(loss_locals_20) / len(loss_locals_20)
        loss_avg_25 = sum(loss_locals_25) / len(loss_locals_25)
        loss_avg_30 = sum(loss_locals_30) / len(loss_locals_30)


        print('NO ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('C1 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_1))
        print('C5 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_5))
        print('C10 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_10))
        print('C15 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_15))
        print('C20 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_20))
        print('C25 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_25))
        print('C30 ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_30))
        #count_array.append(agent_found_count)
        loss_train.append(loss_avg)
        loss_train_1.append(loss_avg_1)
        loss_train_5.append(loss_avg_5)
        loss_train_10.append(loss_avg_10)
        loss_train_15.append(loss_avg_15)
        loss_train_20.append(loss_avg_20)
        loss_train_25.append(loss_avg_25)
        loss_train_30.append(loss_avg_30)

    # plot loss curve
    #plt.figure()
    #plt.subplots()
    #attack_no = plt.plot(range(len(loss_train)), loss_train)
    #attack_1 = plt.plot(range(len(loss_train_1)),loss_train_1)
    #plt.ylabel('train_loss')
    #plt.savefig('log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    #print("COUNT DATA",str(count_array))
    print("NO ATTACK DATA=",loss_train)
    print("1 ATTACK DATA=",loss_train_1)
    print("5 ATTACK DATA=",loss_train_5)
    print("10 ATTACK DATA=",loss_train_10)
    print("15 ATTACK DATA=",loss_train_15)
    print("20 ATTACK DATA=",loss_train_20)
    print("25 ATTACK DATA=",loss_train_25)
    print("30 ATTACK DATA=",loss_train_30)

    print(malicious_dict)
    print(malicious_count)
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
    print("Training accuracy (LAZY ATTACK 1): {:.2f}".format(acc_train1))
    print("Testing accuracy (LAZY ATTACK 1): {:.2f}".format(acc_test1))

    net_glob5.eval()
    acc_train5, loss_train_5 = test_img(net_glob5, dataset_train, args)
    acc_test5, loss_test_5 = test_img(net_glob5, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 5): {:.2f}".format(acc_train5))
    print("Testing accuracy (LAZY ATTACK 5): {:.2f}".format(acc_test5))

    net_glob10.eval()
    acc_train10, loss_train_10 = test_img(net_glob10, dataset_train, args)
    acc_test10, loss_test_10 = test_img(net_glob10, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 10): {:.2f}".format(acc_train10))
    print("Testing accuracy (LAZY ATTACK 10): {:.2f}".format(acc_test10))

    net_glob15.eval()
    acc_train15, loss_train_15 = test_img(net_glob15, dataset_train, args)
    acc_test15, loss_test_15 = test_img(net_glob15, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 15): {:.2f}".format(acc_train15))
    print("Testing accuracy (LAZY ATTACK 15): {:.2f}".format(acc_test15))

    net_glob20.eval()
    acc_train20, loss_train_20 = test_img(net_glob20, dataset_train, args)
    acc_test20, loss_test_20 = test_img(net_glob20, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 20): {:.2f}".format(acc_train20))
    print("Testing accuracy (LAZY ATTACK 20): {:.2f}".format(acc_test20))

    net_glob25.eval()
    acc_train25, loss_train_25 = test_img(net_glob25, dataset_train, args)
    acc_test25, loss_test_25 = test_img(net_glob25, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 25): {:.2f}".format(acc_train25))
    print("Testing accuracy (LAZY ATTACK 25): {:.2f}".format(acc_test25))

    net_glob30.eval()
    acc_train30, loss_train_30 = test_img(net_glob30, dataset_train, args)
    acc_test30, loss_test_30 = test_img(net_glob30, dataset_test, args)
    print("Training accuracy (LAZY ATTACK 30): {:.2f}".format(acc_train30))
    print("Testing accuracy (LAZY ATTACK 30): {:.2f}".format(acc_test30))

    #write to csv
    with open("../MNIST_Results/benchmarks/lazy_attack_data/lazy_n_data_1.csv","w+") as csv_file:
        my_writer = csv.writer(csv_file,delimiter=',')
        my_writer.writerow(("NUM_ATTACKS","TRAIN_ACCURACY","TEST_ACCURACY"))
        my_writer.writerow((0,acc_train.item(),acc_test.item()))
        my_writer.writerow((1,acc_train1.item(),acc_test1.item()))
        my_writer.writerow((5,acc_train5.item(),acc_test5.item()))
        my_writer.writerow((10,acc_train10.item(),acc_test10.item()))
        my_writer.writerow((15,acc_train15.item(),acc_test15.item()))
        my_writer.writerow((20,acc_train20.item(),acc_test20.item()))
        my_writer.writerow((25,acc_train25.item(),acc_test25.item()))
        my_writer.writerow((30,acc_train30.item(),acc_test30.item()))

