#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


#system of 10 agents.
#every agent sampled for 50 epochs
#At every epoch, for each Layer (weights):
	#Calculate mean, SD, variance, max_val, min_val
	#Put this data in descriptive.csv
	#CSV - 
		#epoch | conv1 | conv2 | fc1 | fc2

#After calculating descriptive stats shown above,
	#Flatten all weights in one layer.
	#use autoencoders for dimensionality reduction.
	#calculate the difference compared to previous layer.
	#record the difference in differences.csv.

#----------

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
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training - NO ATTACK
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []


    descriptive_stats = defaultdict()

    for iter in range(args.epochs):
        #agent_found_count = 0
        w_locals, loss_locals = [], []          #w_locals = array of local_weights

        m = max(int(args.frac * args.num_users), 1)     #m = number of users used in one ROUND/EPOCH, check utils.options for more clarity on this
        users_list = [0,1,2,3,4,5,6,7,8,9]
        idxs_users = np.asarray(users_list)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            print("***BLAH BLAH BLAH***")

            #NO ATTACK
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


        # update global weights
        w_glob = FedAvg(w_locals)
        #dict_structure = {epoch: {layer:[size,mean,std,min,max]}}
        conv1 = w_glob['conv1.weight']
        conv2 = w_glob['conv2.weight']
        fc1 = w_glob['fc1.weight']
        fc2 = w_glob['fc2.weight']

        descriptive_stats[str(iter+1)] = {'conv1':[conv1.size(),conv1.mean().item(),conv1.std().item(),conv1.min().item(),conv1.max().item()],\
        								'conv2':[conv2.size(),conv2.mean().item(),conv2.std().item(),conv2.min().item(),conv2.max().item()],\
        								'fc1': [fc1.size(),fc1.mean().item(),fc1.std().item(),fc1.min().item(),fc1.max().item()],\
        								'fc2': [fc2.size(),fc2.mean().item(),fc2.std().item(),fc2.min().item(),fc2.max().item()]}
       	#print(descriptive_stats)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('NO ATTACK ---> Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
 
        #count_array.append(agent_found_count)
        loss_train.append(loss_avg)


    # plot loss curve
    #plt.figure()
    #plt.subplots()
    #attack_no = plt.plot(range(len(loss_train)), loss_train)
    #attack_1 = plt.plot(range(len(loss_train_1)),loss_train_1)
    #plt.ylabel('train_loss')
    #plt.savefig('log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    #print("COUNT DATA",str(count_array))
    print("NO ATTACK DATA=",loss_train)

    #descriptive.csv
    with open("../MNIST_Results/benchmarks/weight_differences/descriptive_1.csv","w+") as csv_file:
    	my_writer = csv.writer(csv_file,delimiter=',')
    	my_writer.writerow(("EPOCH","LAYER_NAME","MEAN","STD","MIN","MAX","LAYER_NAME","MEAN","STD","MIN","MAX","LAYER_NAME","MEAN","STD","MIN","MAX","LAYER_NAME","MEAN","STD","MIN","MAX"))

    	for items in descriptive_stats.keys():
    		my_writer.writerow((items,"conv1",descriptive_stats[items]['conv1'][1],descriptive_stats[items]['conv1'][2],descriptive_stats[items]['conv1'][3],descriptive_stats[items]['conv1'][4],"conv2",descriptive_stats[items]['conv2'][1],descriptive_stats[items]['conv2'][2],descriptive_stats[items]['conv2'][3],descriptive_stats[items]['conv2'][4],"fc1",descriptive_stats[items]['fc1'][1],descriptive_stats[items]['fc1'][2],descriptive_stats[items]['fc1'][3],descriptive_stats[items]['fc1'][4],"fc2",descriptive_stats[items]['fc2'][1],descriptive_stats[items]['fc2'][2],descriptive_stats[items]['fc2'][3],descriptive_stats[items]['fc2'][4]))

    		

    # testing
    net_glob.eval()
    #print("Agent_Found_Count",agent_found_count)
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy (NO ATTACK): {:.2f}".format(acc_train))
    print("Testing accuracy (NO ATTACK): {:.2f}".format(acc_test))

