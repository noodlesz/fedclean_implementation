#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import csv

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.MNIST_biasing_updates import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.MNIST_biasing_test import test_img
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

        ###
        #ANALYZING DATA FOR BIASING ATTACK
        train_count_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        test_count_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

        train_length = len(dataset_train)
        test_length = len(dataset_test)

        print("TRAIN LENGTH",train_length)
        print("TEST LENGTH",test_length)

        for items in dataset_train:
            train_count_dict[items[1]]+=1
        for items in dataset_test:
            test_count_dict[items[1]]+=1

        print(train_count_dict)
        train_sum = 0
        for items in train_count_dict.keys():
            train_sum+=train_count_dict[items]
        print("TRAIN SUM",train_sum)
        print(test_count_dict)
        test_sum = 0
        for items in test_count_dict.keys():
            test_sum+=test_count_dict[items]
        print("TEST SUM",test_sum)
        ###ANALYZING END
        ###

        

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

    #VIVEK constant attack experiment - 5 MALICIOUS
    loss_train_5 = []
    fixed_agent_5 = random.sample(range(32),5)
 
    #VIVEK constant attack experiment - 10 MALICIOUS
    loss_train_10 = []
    fixed_agent_10 = random.sample(range(32),10)

    #VIVEK constant attack experiment - 15 MALICIOUS
    loss_train_15 = []
    fixed_agent_15 = random.sample(range(32),15)

    #VIVEK constant attack experiment - 20 MALICIOUS
    loss_train_20 = []
    fixed_agent_20 = random.sample(range(32),20)

    #VIVEK constant attack experiment - 25 MALICIOUS
    loss_train_25 = []
    fixed_agent_25 = random.sample(range(32),25)

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_30 = []
    fixed_agent_30 = random.sample(range(32),30)


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


            if idx == fixed_agent_1:
                w1,loss1 = local1.train_biased(net=copy.deepcopy(net_glob1).to(args.device))
                malicious_count[1]+=1

            if idx != fixed_agent_1:
                w1, loss1 = local1.train(net=copy.deepcopy(net_glob1).to(args.device))

            if idx in fixed_agent_5:
                w5,loss5 = local5.train_biased(net=copy.deepcopy(net_glob5).to(args.device))
                malicious_count[5]+=1

            if idx not in fixed_agent_5:
                w5, loss5 = local5.train(net=copy.deepcopy(net_glob5).to(args.device))

            if idx in fixed_agent_10:
                w10,loss10 = local10.train_biased(net=copy.deepcopy(net_glob10).to(args.device))
                malicious_count[10]+=1

            if idx not in fixed_agent_10:
                w10, loss10 = local10.train(net=copy.deepcopy(net_glob10).to(args.device))

            if idx in fixed_agent_15:
                w15,loss15 = local15.train_biased(net=copy.deepcopy(net_glob15).to(args.device))
                malicious_count[15]+=1

            if idx not in fixed_agent_15:
                w15, loss15 = local15.train(net=copy.deepcopy(net_glob15).to(args.device))

            if idx in fixed_agent_20:
                w20,loss20 = local20.train_biased(net=copy.deepcopy(net_glob20).to(args.device))
                malicious_count[20]+=1

            if idx not in fixed_agent_20:
                w20, loss20 = local20.train(net=copy.deepcopy(net_glob20).to(args.device))

            if idx in fixed_agent_25:
                w25,loss25 = local25.train_biased(net=copy.deepcopy(net_glob25).to(args.device))
                malicious_count[25]+=1

            if idx not in fixed_agent_25:
                w25, loss25 = local25.train(net=copy.deepcopy(net_glob25).to(args.device))

            if idx in fixed_agent_30:
                w30,loss30 = local30.train_biased(net=copy.deepcopy(net_glob30).to(args.device))
                malicious_count[30]+=1

            if idx not in fixed_agent_30:
                w30, loss30 = local30.train(net=copy.deepcopy(net_glob30).to(args.device))



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
        w_glob_1 = FedAvg(w_locals_1)
        w_glob_5 = FedAvg(w_locals_5)
        w_glob_10 = FedAvg(w_locals_10)
        w_glob_15 = FedAvg(w_locals_15)
        w_glob_20 = FedAvg(w_locals_20)
        w_glob_25 = FedAvg(w_locals_25)
        w_glob_30 = FedAvg(w_locals_30)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob1.load_state_dict(w_glob_1)
        net_glob5.load_state_dict(w_glob_5)
        net_glob10.load_state_dict(w_glob_10)
        net_glob15.load_state_dict(w_glob_15)
        net_glob20.load_state_dict(w_glob_20)
        net_glob25.load_state_dict(w_glob_25)
        net_glob30.load_state_dict(w_glob_30)

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
    acc_train, loss_train, acc_train0,acc_train1,acc_train2,acc_train3,acc_train4,acc_train5,acc_train6,acc_train7,acc_train8,acc_train9,target_train_freq,correct_train_freq = test_img(net_glob, dataset_train, args)
    acc_test, loss_test, acc_test0,acc_test1,acc_test2,acc_test3,acc_test4,acc_test5,acc_test6,acc_test7,acc_test8,acc_test9,target_test_freq,correct_test_freq = test_img(net_glob, dataset_test, args)
    print("Training accuracy (NO ATTACK): {:.2f}".format(acc_train))
    print("Testing accuracy (NO ATTACK): {:.2f}".format(acc_test))
    print("Training accuracy-5 (NO ATTACK): {:.2f}".format(acc_train5))
    print("Testing accuracy-5 (NO ATTACK): {:.2f}".format(acc_test5))
    print("\n")

    net_glob1.eval()
    acc_train_1, loss_train_1, acc_train0_1,acc_train1_1,acc_train2_1,acc_train3_1,acc_train4_1,acc_train5_1,acc_train6_1,acc_train7_1,acc_train8_1,acc_train9_1,target_train_freq_1,correct_train_freq_1 = test_img(net_glob1, dataset_train, args)
    acc_test_1, loss_test_1, acc_test0_1,acc_test1_1,acc_test2_1,acc_test3_1,acc_test4_1,acc_test5_1,acc_test6_1,acc_test7_1,acc_test8_1,acc_test9_1,target_test_freq_1,correct_test_freq_1 = test_img(net_glob1, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 1): {:.2f}".format(acc_train_1))
    print("Testing accuracy (LABEL FLIPPED 1): {:.2f}".format(acc_test_1))
    print("Training accuracy-5 (LABEL FLIPPED 1): {:.2f}".format(acc_train5_1))
    print("Testing accuracy-5 (LABEL FLIPPED 1): {:.2f}".format(acc_test5_1))
    print("\n")

    net_glob5.eval()
    acc_train_5, loss_train_5, acc_train0_5,acc_train1_5,acc_train2_5,acc_train3_5,acc_train4_5,acc_train5_5,acc_train6_5,acc_train7_5,acc_train8_5,acc_train9_5,target_train_freq_5,correct_train_freq_5 = test_img(net_glob5, dataset_train, args)
    acc_test_5, loss_test_5, acc_test0_5,acc_test1_5,acc_test2_5,acc_test3_5,acc_test4_5,acc_test5_5,acc_test6_5,acc_test7_5,acc_test8_5,acc_test9_5,target_test_freq_5,correct_test_freq_5 = test_img(net_glob5, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 5): {:.2f}".format(acc_train_5))
    print("Testing accuracy (LABEL FLIPPED 5): {:.2f}".format(acc_test_5))
    print("Training accuracy-5 (LABEL FLIPPED 5): {:.2f}".format(acc_train5_5))
    print("Testing accuracy-5 (LABEL FLIPPED 5): {:.2f}".format(acc_test5_5))
    print("\n")

    net_glob10.eval()
    acc_train_10, loss_train_10, acc_train0_10,acc_train1_10,acc_train2_10,acc_train3_10,acc_train4_10,acc_train5_10,acc_train6_10,acc_train7_10,acc_train8_10,acc_train9_10,target_train_freq_10,correct_train_freq_10 = test_img(net_glob10, dataset_train, args)
    acc_test_10, loss_test_10, acc_test0_10,acc_test1_10,acc_test2_10,acc_test3_10,acc_test4_10,acc_test5_10,acc_test6_10,acc_test7_10,acc_test8_10,acc_test9_10,target_test_freq_10,correct_test_freq_10 = test_img(net_glob10, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 10): {:.2f}".format(acc_train_10))
    print("Testing accuracy (LABEL FLIPPED 10): {:.2f}".format(acc_test_10))
    print("Training accuracy-5 (LABEL FLIPPED 10): {:.2f}".format(acc_train5_10))
    print("Testing accuracy-5 (LABEL FLIPPED 10): {:.2f}".format(acc_test5_10))
    print("\n")

    net_glob15.eval()
    acc_train_15, loss_train_15, acc_train0_15,acc_train1_15,acc_train2_15,acc_train3_15,acc_train4_15,acc_train5_15,acc_train6_15,acc_train7_15,acc_train8_15,acc_train9_15,target_train_freq_15,correct_train_freq_15 = test_img(net_glob15, dataset_train, args)
    acc_test_15, loss_test_15, acc_test0_15,acc_test1_15,acc_test2_15,acc_test3_15,acc_test4_15,acc_test5_15,acc_test6_15,acc_test7_15,acc_test8_15,acc_test9_15,target_test_freq_15,correct_test_freq_15 = test_img(net_glob15, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 15): {:.2f}".format(acc_train_15))
    print("Testing accuracy (LABEL FLIPPED 15): {:.2f}".format(acc_test_15))
    print("Training accuracy-5 (LABEL FLIPPED 15): {:.2f}".format(acc_train5_15))
    print("Testing accuracy-5 (LABEL FLIPPED 15): {:.2f}".format(acc_test5_15))
    print("\n")

    net_glob20.eval()
    acc_train_20, loss_train_20, acc_train0_20,acc_train1_20,acc_train2_20,acc_train3_20,acc_train4_20,acc_train5_20,acc_train6_20,acc_train7_20,acc_train8_20,acc_train9_20,target_train_freq_20,correct_train_freq_20 = test_img(net_glob20, dataset_train, args)
    acc_test_20, loss_test_20, acc_test0_20,acc_test1_20,acc_test2_20,acc_test3_20,acc_test4_20,acc_test5_20,acc_test6_20,acc_test7_20,acc_test8_20,acc_test9_20,target_test_freq_20,correct_test_freq_20 = test_img(net_glob20, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 20): {:.2f}".format(acc_train_20))
    print("Testing accuracy (LABEL FLIPPED 20): {:.2f}".format(acc_test_20))
    print("Training accuracy-5 (LABEL FLIPPED 20): {:.2f}".format(acc_train5_20))
    print("Testing accuracy-5 (LABEL FLIPPED 20): {:.2f}".format(acc_test5_20))
    print("\n")

    net_glob25.eval()
    acc_train_25, loss_train_25, acc_train0_25,acc_train1_25,acc_train2_25,acc_train3_25,acc_train4_25,acc_train5_25,acc_train6_25,acc_train7_25,acc_train8_25,acc_train9_25,target_train_freq_25,correct_train_freq_25 = test_img(net_glob25, dataset_train, args)
    acc_test_25, loss_test_25, acc_test0_25,acc_test1_25,acc_test2_25, acc_test3_25,acc_test4_25,acc_test5_25,acc_test6_25,acc_test7_25,acc_test8_25,acc_test9_25,target_test_freq_25,correct_test_freq_25 = test_img(net_glob25, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 25): {:.2f}".format(acc_train_25))
    print("Testing accuracy (LABEL FLIPPED 25): {:.2f}".format(acc_test_25))
    print("Training accuracy-5 (LABEL FLIPPED 25): {:.2f}".format(acc_train5_25))
    print("Testing accuracy-5 (LABEL FLIPPED 25): {:.2f}".format(acc_test5_25))
    print("\n")

    net_glob30.eval()
    acc_train_30, loss_train_30, acc_train0_30,acc_train1_30,acc_train2_30,acc_train3_30,acc_train4_30,acc_train5_30,acc_train6_30,acc_train7_30,acc_train8_30,acc_train9_30,target_train_freq_30,correct_train_freq_30 = test_img(net_glob30, dataset_train, args)
    acc_test_30, loss_test_30, acc_test0_30,acc_test1_30,acc_test2_30,acc_test3_30,acc_test4_30,acc_test5_30,acc_test6_30,acc_test7_30,acc_test8_30,acc_test9_30,target_test_freq_30,correct_test_freq_30 = test_img(net_glob30, dataset_test, args)
    print("Training accuracy (LABEL FLIPPED 30): {:.2f}".format(acc_train_30))
    print("Testing accuracy (LABEL FLIPPED 30): {:.2f}".format(acc_test_30))
    print("Training accuracy-5 (LABEL FLIPPED 30): {:.2f}".format(acc_train5_30))
    print("Testing accuracy-5 (LABEL FLIPPED 30): {:.2f}".format(acc_test5_30))
    print("\n")

    with open("label_flipped_final_data.csv", "w") as csv_file:
        writer = csv.writer(csv_file,delimiter=',')
        writer.writerow(("OVERALL_TRAIN_ACCURACY","OVERALL_TEST_ACCURACY","TRAIN_ACC_0","TEST_ACC_0","TRAIN_ACC_1","TEST_ACC_1","TRAIN_ACC_2","TEST_ACC_2","TRAIN_ACC_3","TEST_ACC_3","TRAIN_ACC_4","TEST_ACC_4","TRAIN_ACC_5","TEST_ACC_5","TRAIN_ACC_6","TEST_ACC_6","TRAIN_ACC_7","TEST_ACC_7","TRAIN_ACC_8","TEST_ACCURACY_8","TRAIN_ACCURACY_9","TEST_ACCURACY_9"))
        writer.writerow((acc_train.item(),acc_test.item(),acc_train0,acc_test0,acc_train1,acc_test1,acc_train2,acc_test2,acc_train3,acc_test3,acc_train4,acc_test4,acc_train5,acc_test5,acc_train6,acc_test6,acc_train7,acc_test7,acc_train8,acc_test8,acc_train9,acc_test9))
        writer.writerow((acc_train_1.item(),acc_test_1.item(),acc_train0_1,acc_test0_1,acc_train1_1,acc_test1_1,acc_train2_1,acc_test2_1,acc_train3_1,acc_test3_1,acc_train4_1,acc_test4_1,acc_train5_1,acc_test5_1,acc_train6_1,acc_test6_1,acc_train7_1,acc_test7_1,acc_train8_1,acc_test8_1,acc_train9_1,acc_test9_1))
        writer.writerow((acc_train_5.item(),acc_test_5.item(),acc_train0_5,acc_test0_5,acc_train1_5,acc_test1_5,acc_train2_5,acc_test2_5,acc_train3_5,acc_test3_5,acc_train4_5,acc_test4_5,acc_train5_5,acc_test5_5,acc_train6_5,acc_test6_5,acc_train7_5,acc_test7_5,acc_train8_5,acc_test8_5,acc_train9_5,acc_test9_5))
        writer.writerow((acc_train_10.item(),acc_test_10.item(),acc_train0_10,acc_test0_10,acc_train1_10,acc_test1_10,acc_train2_10,acc_test2_10,acc_train3_10,acc_test3_10,acc_train4_10,acc_test4_10,acc_train5_10,acc_test5_10,acc_train6_10,acc_test6_10,acc_train7_10,acc_test7_10,acc_train8_10,acc_test8_10,acc_train9_10,acc_test9_10))
        writer.writerow((acc_train_15.item(),acc_test_15.item(),acc_train0_15,acc_test0_15,acc_train1_15,acc_test1_15,acc_train2_15,acc_test2_15,acc_train3_15,acc_test3_15,acc_train4_15,acc_test4_15,acc_train5_15,acc_test5_15,acc_train6_15,acc_test6_15,acc_train7_15,acc_test7_15,acc_train8_15,acc_test8_15,acc_train9_15,acc_test9_15))
        writer.writerow((acc_train_20.item(),acc_test_20.item(),acc_train0_20,acc_test0_20,acc_train1_20,acc_test1_20,acc_train2_20,acc_test2_20,acc_train3_20,acc_test3_20,acc_train4_20,acc_test4_20,acc_train5_20,acc_test5_20,acc_train6_20,acc_test6_20,acc_train7_20,acc_test7_20,acc_train8_20,acc_test8_20,acc_train9_20,acc_test9_20))
        writer.writerow((acc_train_25.item(),acc_test_25.item(),acc_train0_25,acc_test0_25,acc_train1_25,acc_test1_25,acc_train2_25,acc_test2_25,acc_train3_25,acc_test3_25,acc_train4_25,acc_test4_25,acc_train5_25,acc_test5_25,acc_train6_25,acc_test6_25,acc_train7_25,acc_test7_25,acc_train8_25,acc_test8_25,acc_train9_25,acc_test9_25))
        writer.writerow((acc_train_30.item(),acc_test_30.item(),acc_train0_30,acc_test0_30,acc_train1_30,acc_test1_30,acc_train2_30,acc_test2_30,acc_train3_30,acc_test3_30,acc_train4_30,acc_test4_30,acc_train5_30,acc_test5_30,acc_train6_30,acc_test6_30,acc_train7_30,acc_test7_30,acc_train8_30,acc_test8_30,acc_train9_30,acc_test9_30))


    with open("label_flipped_frequency_data.csv","w") as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(("TOTAL_TRAIN_0","CORRECT_TRAIN_0","TOTAL_TEST_0","CORRECT_TEST_0","TOTAL_TRAIN_1","CORRECT_TRAIN_1","TOTAL_TEST_1","CORRECT_TEST_1","TOTAL_TRAIN_2","CORRECT_TRAIN_2","TOTAL_TEST_2","CORRECT_TEST_2","TOTAL_TRAIN_3","CORRECT_TRAIN_3","TOTAL_TEST_3","CORECT_TEST_3","TOTAL_TRAIN_4","CORRECT_TRAIN_4","TOTAL_TEST_4","CORRECT_TEST_4","TOTAL_TRAIN_5","CORRECT_TRAIN_5","TOTAL_TEST_5","CORRECT_TEST_5","TOTAL_TRAIN_6","CORRECT_TRAIN_6","TOTAL_TEST_6","CORRECT_TEST_6","TOTAL_TRAIN_7","CORRECT_TRAIN_7","TOTAL_TEST_7","CORRECT_TEST_7","TOTAL_TRAIN_8","CORRECT_TRAIN_8","TOTAL_TEST_8","CORRECT_TEST_8","TOTAL_TRAIN_9","CORRECT_TRAIN_9","TOTAL_TEST_9","CORRECT_TEST_9"))
        writer.writerow((target_train_freq[0],correct_train_freq[0],target_test_freq[0],correct_test_freq[0],target_train_freq[1],correct_train_freq[1],target_test_freq[1],correct_test_freq[1],target_train_freq[2],correct_train_freq[2],target_test_freq[2],correct_test_freq[2],target_train_freq[3],correct_test_freq[3],target_test_freq[3],correct_test_freq[3],target_train_freq[4],correct_train_freq[4],target_test_freq[4],correct_test_freq[4],target_train_freq[5],correct_train_freq[5],target_test_freq[5],correct_test_freq[5],target_train_freq[6],correct_train_freq[6],target_test_freq[6],correct_test_freq[6],target_train_freq[7],correct_train_freq[7],target_test_freq[7],correct_test_freq[7],target_train_freq[8],correct_train_freq[8],target_test_freq[8],correct_test_freq[8],target_train_freq[9],correct_train_freq[9],target_test_freq[9],correct_test_freq[9]))
        writer.writerow((target_train_freq_1[0],correct_train_freq_1[0],target_test_freq_1[0],correct_test_freq_1[0],target_train_freq_1[1],correct_train_freq_1[1],target_test_freq_1[1],correct_test_freq_1[1],target_train_freq_1[2],correct_train_freq_1[2],target_test_freq_1[2],correct_test_freq_1[2],target_train_freq_1[3],correct_test_freq_1[3],target_test_freq_1[3],correct_test_freq_1[3],target_train_freq_1[4],correct_train_freq_1[4],target_test_freq_1[4],correct_test_freq_1[4],target_train_freq_1[5],correct_train_freq_1[5],target_test_freq_1[5],correct_test_freq_1[5],target_train_freq_1[6],correct_train_freq_1[6],target_test_freq_1[6],correct_test_freq_1[6],target_train_freq_1[7],correct_train_freq_1[7],target_test_freq_1[7],correct_test_freq_1[7],target_train_freq_1[8],correct_train_freq_1[8],target_test_freq_1[8],correct_test_freq_1[8],target_train_freq_1[9],correct_train_freq_1[9],target_test_freq_1[9],correct_test_freq_1[9]))
        writer.writerow((target_train_freq_5[0],correct_train_freq_5[0],target_test_freq_5[0],correct_test_freq_5[0],target_train_freq_5[1],correct_train_freq_5[1],target_test_freq_5[1],correct_test_freq_5[1],target_train_freq_5[2],correct_train_freq_5[2],target_test_freq_5[2],correct_test_freq_5[2],target_train_freq_5[3],correct_test_freq_5[3],target_test_freq_5[3],correct_test_freq_5[3],target_train_freq_5[4],correct_train_freq_5[4],target_test_freq_5[4],correct_test_freq_5[4],target_train_freq_5[5],correct_train_freq_5[5],target_test_freq_5[5],correct_test_freq_5[5],target_train_freq_5[6],correct_train_freq_5[6],target_test_freq_5[6],correct_test_freq_5[6],target_train_freq_5[7],correct_train_freq_5[7],target_test_freq_5[7],correct_test_freq_5[7],target_train_freq_5[8],correct_train_freq_5[8],target_test_freq_5[8],correct_test_freq_5[8],target_train_freq_5[9],correct_train_freq_5[9],target_test_freq_5[9],correct_test_freq_5[9]))
        writer.writerow((target_train_freq_10[0],correct_train_freq_10[0],target_test_freq_10[0],correct_test_freq_10[0],target_train_freq_10[1],correct_train_freq_10[1],target_test_freq_10[1],correct_test_freq_10[1],target_train_freq_10[2],correct_train_freq_10[2],target_test_freq_10[2],correct_test_freq_10[2],target_train_freq_10[3],correct_test_freq_10[3],target_test_freq_10[3],correct_test_freq_10[3],target_train_freq_10[4],correct_train_freq_10[4],target_test_freq_10[4],correct_test_freq_10[4],target_train_freq_10[5],correct_train_freq_10[5],target_test_freq_10[5],correct_test_freq_10[5],target_train_freq_10[6],correct_train_freq_10[6],target_test_freq_10[6],correct_test_freq_10[6],target_train_freq_10[7],correct_train_freq_10[7],target_test_freq_10[7],correct_test_freq_10[7],target_train_freq_10[8],correct_train_freq_10[8],target_test_freq_10[8],correct_test_freq_10[8],target_train_freq_10[9],correct_train_freq_10[9],target_test_freq_10[9],correct_test_freq_10[9]))
        writer.writerow((target_train_freq_15[0],correct_train_freq_15[0],target_test_freq_15[0],correct_test_freq_15[0],target_train_freq_15[1],correct_train_freq_15[1],target_test_freq_15[1],correct_test_freq_15[1],target_train_freq_15[2],correct_train_freq_15[2],target_test_freq_15[2],correct_test_freq_15[2],target_train_freq_15[3],correct_test_freq_15[3],target_test_freq_15[3],correct_test_freq_15[3],target_train_freq_15[4],correct_train_freq_15[4],target_test_freq_15[4],correct_test_freq_15[4],target_train_freq_15[5],correct_train_freq_15[5],target_test_freq_15[5],correct_test_freq_15[5],target_train_freq_15[6],correct_train_freq_15[6],target_test_freq_15[6],correct_test_freq_15[6],target_train_freq_15[7],correct_train_freq_15[7],target_test_freq_15[7],correct_test_freq_15[7],target_train_freq_15[8],correct_train_freq_15[8],target_test_freq_15[8],correct_test_freq_15[8],target_train_freq_15[9],correct_train_freq_15[9],target_test_freq_15[9],correct_test_freq_15[9]))
        writer.writerow((target_train_freq_20[0],correct_train_freq_20[0],target_test_freq_20[0],correct_test_freq_20[0],target_train_freq_20[1],correct_train_freq_20[1],target_test_freq_20[1],correct_test_freq_20[1],target_train_freq_20[2],correct_train_freq_20[2],target_test_freq_20[2],correct_test_freq_20[2],target_train_freq_20[3],correct_test_freq_20[3],target_test_freq_20[3],correct_test_freq_20[3],target_train_freq_20[4],correct_train_freq_20[4],target_test_freq_20[4],correct_test_freq_20[4],target_train_freq_20[5],correct_train_freq_20[5],target_test_freq_20[5],correct_test_freq_20[5],target_train_freq_20[6],correct_train_freq_20[6],target_test_freq_20[6],correct_test_freq_20[6],target_train_freq_20[7],correct_train_freq_20[7],target_test_freq_20[7],correct_test_freq_20[7],target_train_freq_20[8],correct_train_freq_20[8],target_test_freq_20[8],correct_test_freq_20[8],target_train_freq_20[9],correct_train_freq_20[9],target_test_freq_20[9],correct_test_freq_20[9]))
        writer.writerow((target_train_freq_25[0],correct_train_freq_25[0],target_test_freq_25[0],correct_test_freq_25[0],target_train_freq_25[1],correct_train_freq_25[1],target_test_freq_25[1],correct_test_freq_25[1],target_train_freq_25[2],correct_train_freq_25[2],target_test_freq_25[2],correct_test_freq_25[2],target_train_freq_25[3],correct_test_freq_25[3],target_test_freq_25[3],correct_test_freq_25[3],target_train_freq_25[4],correct_train_freq_25[4],target_test_freq_25[4],correct_test_freq_25[4],target_train_freq_25[5],correct_train_freq_25[5],target_test_freq_25[5],correct_test_freq_25[5],target_train_freq_25[6],correct_train_freq_25[6],target_test_freq_25[6],correct_test_freq_25[6],target_train_freq_25[7],correct_train_freq_25[7],target_test_freq_25[7],correct_test_freq_25[7],target_train_freq_25[8],correct_train_freq_25[8],target_test_freq_25[8],correct_test_freq_25[8],target_train_freq_25[9],correct_train_freq_25[9],target_test_freq_25[9],correct_test_freq_25[9]))
        writer.writerow((target_train_freq_30[0],correct_train_freq_30[0],target_test_freq_30[0],correct_test_freq_30[0],target_train_freq_30[1],correct_train_freq_30[1],target_test_freq_30[1],correct_test_freq_30[1],target_train_freq_30[2],correct_train_freq_30[2],target_test_freq_30[2],correct_test_freq_30[2],target_train_freq_30[3],correct_test_freq_30[3],target_test_freq_30[3],correct_test_freq_30[3],target_train_freq_30[4],correct_train_freq_30[4],target_test_freq_30[4],correct_test_freq_30[4],target_train_freq_30[5],correct_train_freq_30[5],target_test_freq_30[5],correct_test_freq_30[5],target_train_freq_30[6],correct_train_freq_30[6],target_test_freq_30[6],correct_test_freq_30[6],target_train_freq_30[7],correct_train_freq_30[7],target_test_freq_30[7],correct_test_freq_30[7],target_train_freq_30[8],correct_train_freq_30[8],target_test_freq_30[8],correct_test_freq_30[8],target_train_freq_30[9],correct_train_freq_30[9],target_test_freq_30[9],correct_test_freq_30[9]))
