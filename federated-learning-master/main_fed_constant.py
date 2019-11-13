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
        trusted_glob = net_glob
        trusted_malicious_glob = net_glob
        net_glob1 = net_glob
        net_glob5 = net_glob
        net_glob10 = net_glob
        net_glob15 = net_glob
        net_glob20 = net_glob
        net_glob25 = net_glob
        net_glob30 = net_glob
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    trusted_glob.train()
    trusted_malicious_glob.train()
    net_glob.train()
    net_glob1.train()
    net_glob5.train()
    net_glob10.train()
    net_glob15.train()
    net_glob20.train()
    net_glob25.train()
    net_glob30.train()

    # copy weights
    w_glob_trusted = trusted_glob.state_dict()
    w_glob_trusted_malicious = trusted_malicious_glob.state_dict()
    w_glob = net_glob.state_dict()
    w_glob1 = net_glob1.state_dict()
    w_glob5 = net_glob5.state_dict()
    w_glob10 = net_glob10.state_dict()
    w_glob15 = net_glob15.state_dict()
    w_glob20 = net_glob20.state_dict()
    w_glob25 = net_glob25.state_dict()
    w_glob30 = net_glob30.state_dict()

    #A LIST OF BAD AGENTS
    #bad_agents = []

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

    #VIVEK constant attack experiment - 5 MALICIOUS
    loss_train_5 = []
    fixed_agent_5 = random.sample(range(32),5)
    updates_recorded_mapping_5 = defaultdict(bool)
    for i in fixed_agent_5:
        updates_recorded_mapping_5[i] = False  #KEY = agent no. & VAL = boolean
    fixed_agent_storage_mapping_5 = {} #KEY = agent no. & VAL = Fixed Updates
    count_array_5 = []

    #VIVEK constant attack experiment - 10 MALICIOUS
    loss_train_10 = []
    fixed_agent_10 = random.sample(range(32),10)
    updates_recorded_mapping_10 = defaultdict(bool)
    for i in fixed_agent_10:
        updates_recorded_mapping_10[i] = False
    fixed_agent_storage_mapping_10 = {}
    count_array_10 = []

    #VIVEK constant attack experiment - 15 MALICIOUS
    loss_train_15 = []
    fixed_agent_15 = random.sample(range(32),15)
    updates_recorded_mapping_15 = defaultdict(bool)
    for i in fixed_agent_15:
        updates_recorded_mapping_15[i] = False
    fixed_agent_storage_mapping_15 = {}
    count_array_15 = []

    #VIVEK constant attack experiment - 20 MALICIOUS
    loss_train_20 = []
    fixed_agent_20 = random.sample(range(32),20)
    updates_recorded_mapping_20 = defaultdict(bool)
    for i in fixed_agent_20:
        updates_recorded_mapping_20[i] = False
    fixed_agent_storage_mapping_20 = {}
    count_array_20 = []

    #VIVEK constant attack experiment - 25 MALICIOUS
    loss_train_25 = []
    fixed_agent_25 = random.sample(range(32),25)
    updates_recorded_mapping_25 = defaultdict(bool)
    for i in fixed_agent_25:
        updates_recorded_mapping_25[i] = False
    fixed_agent_storage_mapping_25 = {}
    count_array_25 = []

    #VIVEK constant attack experiment - 30 MALICIOUS
    loss_train_30 = []
    fixed_agent_30 = random.sample(range(32),30)
    updates_recorded_mapping_30 = defaultdict(bool)
    for i in fixed_agent_30:
        updates_recorded_mapping_30[i] = False
    fixed_agent_storage_mapping_30 = {}
    count_array_30 = []


    def find_good_agent(list_bad):
        good_agent = None
        for i in range(args.num_users):
            if i not in list_bad:
                return i

    def find_good_agents(list_bad):
        good_agent_list = []
        for i in range(args.num_users):
            if i not in list_bad:
                good_agent_list.append(i)
        return good_agent_list

    def trustedFilterModel(good_agent_index):
        local_trusted = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[good_agent_index])
        w_trusted,loss_trusted = local.train(net=copy.deepcopy(trusted_glob).to(args.device))
        w_locals_trusted,loss_locals_trusted = [w_trusted],[loss_trusted]

        w_glob_trusted = FedAvg(w_locals_trusted)
        trusted_glob.load_state_dict(w_glob_trusted)
        trusted_loss_avg = sum(loss_locals_trusted)/len(loss_locals_trusted)  #THIS IS THE LOSS OF THE TRUSTED MODEL AFTER EACH EPOCH

        return w_trusted,loss_trusted,trusted_loss_avg

    def trustedPlusMaliciousModel(w_trusted,loss_trusted,w_malicious,loss_malicious,trusted_loss_average):
        w_locals_trusted_malicious,loss_locals_trusted_malicious = [w_trusted,w_malicious],[loss_trusted,loss_malicious]

        w_glob_trusted_malicious = FedAvg(w_locals_trusted_malicious)
        trusted_malicious_glob.load_state_dict(w_glob_trusted_malicious)
        trusted_malicious_loss_avg = sum(loss_locals_trusted_malicious)/len(loss_locals_trusted_malicious)

        #print("TRUSTED_MAL",str(trusted_malicious_loss_avg))
        #print("TRUSTED",str(trusted_loss_average))
        #print("DIFFERENCE",str(trusted_malicious_loss_avg-trusted_loss_average))
        if trusted_malicious_loss_avg <= (trusted_loss_average + (0.05*trusted_loss_average)):
            #print("GOOD")
            return True         #TRUE implies GOOD UPDATE
        #print("BAD")
        return False            #FALSE implies BAD UPDATE

    for iter in range(args.epochs):
        if iter > 15:
            w_locals, loss_locals = [], []          
            w_locals_1, loss_locals_1 = [],[]
            w_locals_5, loss_locals_5 = [],[]
            w_locals_10, loss_locals_10 = [],[]
            w_locals_15, loss_locals_15 = [],[]
            w_locals_20, loss_locals_20 = [],[]
            w_locals_25, loss_locals_25 = [],[]
            w_locals_30, loss_locals_30 = [],[]
            m = max(int(args.frac * args.num_users), 1)     #m = number of users used in one ROUND/EPOCH, check utils.options for more clarity on this
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            good_agents = find_good_agents(fixed_agent_30)

            my_list = list(idxs_users)
            for items in good_agents:
                if items not in my_list:
                    my_list.append(items)
                    my_list.pop(0)
            idxs_users = np.asarray(my_list)

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

                if idx in fixed_agent_10:
                    if updates_recorded_mapping_10[idx]:
                        w10 = copy.deepcopy(fixed_agent_storage_mapping_10[idx])
                    elif not updates_recorded_mapping_10[idx]:
                        fixed_agent_storage_mapping_10[idx] = copy.deepcopy(w10)
                        updates_recorded_mapping_10[idx] = True

                if idx in fixed_agent_15:
                    if updates_recorded_mapping_15[idx]:
                        w15 = copy.deepcopy(fixed_agent_storage_mapping_15[idx])
                    elif not updates_recorded_mapping_15[idx]:
                        fixed_agent_storage_mapping_15[idx] = copy.deepcopy(w15)
                        updates_recorded_mapping_15[idx] = True

                if idx in fixed_agent_20:
                    if updates_recorded_mapping_20[idx]:
                        w20 = copy.deepcopy(fixed_agent_storage_mapping_20[idx])
                    elif not updates_recorded_mapping_20[idx]:
                        fixed_agent_storage_mapping_20[idx] = copy.deepcopy(w20)
                        updates_recorded_mapping_20[idx] = True

                if idx in fixed_agent_25:
                    if updates_recorded_mapping_25[idx]:
                        w25 = copy.deepcopy(fixed_agent_storage_mapping_25[idx])
                    elif not updates_recorded_mapping_25[idx]:
                        fixed_agent_storage_mapping_25[idx] = copy.deepcopy(w25)
                        updates_recorded_mapping_25[idx] = True

                if idx in fixed_agent_30:
                    if updates_recorded_mapping_30[idx]:
                        w30 = copy.deepcopy(fixed_agent_storage_mapping_30[idx])
                    elif not updates_recorded_mapping_30[idx]:
                        fixed_agent_storage_mapping_30[idx] = copy.deepcopy(w30)
                        updates_recorded_mapping_30[idx] = True


            #NO ATTACK
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

                #1 MALICIOUS
                good_val = find_good_agent([fixed_agent_1])
                while (good_val == idx) or (good_val == fixed_agent_1):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w1_malicious = copy.deepcopy(w1)
                ll1_malicious = copy.deepcopy(loss1)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w1_malicious,ll1_malicious,trusted_accuracy):
                    w_locals_1.append(w1_malicious)
                    loss_locals_1.append(ll1_malicious)

                    #5 MALICIOUS
                good_val = find_good_agent(fixed_agent_5)
                while (good_val == idx) or (good_val in fixed_agent_5):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w5_malicious = copy.deepcopy(w5)
                ll5_malicious = copy.deepcopy(loss5)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w5_malicious,ll5_malicious,trusted_accuracy):
                    w_locals_5.append(w5_malicious)
                    loss_locals_5.append(ll5_malicious)

            #10 MALICIOUS
                good_val = find_good_agent(fixed_agent_10)
                while (good_val == idx) or (good_val in fixed_agent_10):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w10_malicious = copy.deepcopy(w10)
                ll10_malicious = copy.deepcopy(loss10)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w10_malicious,ll10_malicious,trusted_accuracy):
                    w_locals_10.append(w10_malicious)
                    loss_locals_10.append(ll10_malicious)

            #15 MALICIOUS
                good_val = find_good_agent(fixed_agent_15)
                while (good_val == idx) or (good_val in fixed_agent_15):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w15_malicious = copy.deepcopy(w15)
                ll15_malicious = copy.deepcopy(loss15)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w15_malicious,ll15_malicious,trusted_accuracy):
                    w_locals_15.append(w15_malicious)
                    loss_locals_15.append(ll15_malicious)


            #20 MALICIOUS
                good_val = find_good_agent(fixed_agent_20)
                while (good_val == idx) or (good_val in fixed_agent_20):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w20_malicious = copy.deepcopy(w20)
                ll20_malicious = copy.deepcopy(loss20)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w20_malicious,ll20_malicious,trusted_accuracy):
                    w_locals_20.append(w20_malicious)
                    loss_locals_20.append(ll20_malicious)

            #25 MALICIOUS
                good_val = find_good_agent(fixed_agent_25)
                while (good_val == idx) or (good_val in fixed_agent_25):
                    if good_val == 31:
                        break
                    good_val+=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w25_malicious = copy.deepcopy(w25)
                ll25_malicious = copy.deepcopy(loss25)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w25_malicious,ll25_malicious,trusted_accuracy):
                    w_locals_25.append(w25_malicious)
                    loss_locals_25.append(ll25_malicious)

            #30 MALICIOUS
                good_val = find_good_agent(fixed_agent_30)
                while (good_val == idx) or (good_val in fixed_agent_30):
                    if good_val == 31:
                        breal
                    good_val +=1
                weight_trusted,losss_trusted,trusted_accuracy = trustedFilterModel(good_val)
                w30_malicious = copy.deepcopy(w30)
                ll30_malicious = copy.deepcopy(loss30)
                if trustedPlusMaliciousModel(weight_trusted,losss_trusted,w30_malicious,ll30_malicious,trusted_accuracy):
                    w_locals_30.append(w30_malicious)
                    loss_locals_30.append(ll30_malicious)

        elif iter <= 15:
            #agent_found_count = 0
                w_locals, loss_locals = [], []          #w_locals = array of local_weights
                w_locals_1, loss_locals_1 = [],[]
                w_locals_5, loss_locals_5 = [],[]
                w_locals_10, loss_locals_10 = [],[]
                w_locals_15, loss_locals_15 = [],[]
                w_locals_20, loss_locals_20 = [],[]
                w_locals_25, loss_locals_25 = [],[]
                w_locals_30, loss_locals_30 = [],[]
                m = max(int(args.frac * args.num_users), 1)
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
                    print("***HONEST HONEST HONEST***")


                #NO ATTACK
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))

                    w_locals_1.append(copy.deepcopy(w1))
                    loss_locals_1.append(copy.deepcopy(loss1))

                    w_locals_5.append(copy.deepcopy(w5))
                    loss_locals_5.append(copy.deepcopy(loss5))


                    w_locals_10.append(copy.deepcopy(w10))
                    loss_locals_10.append(copy.deepcopy(loss10))

                    w_locals_15.append(copy.deepcopy(w15))
                    loss_locals_15.append(copy.deepcopy(loss15))

                    w_locals_20.append(copy.deepcopy(w20))
                    loss_locals_20.append(copy.deepcopy(loss20))


                    w_locals_25.append(copy.deepcopy(w25))
                    loss_locals_25.append(copy.deepcopy(loss25))

                    w_locals_30.append(copy.deepcopy(w30))
                    loss_locals_30.append(loss30)




        # update global weights
        if w_locals:
            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)

        if w_locals_1:
            w_glob_1 = FedAvg(w_locals_1)
            net_glob1.load_state_dict(w_glob_1)

        if w_locals_5:
            w_glob_5 = FedAvg(w_locals_5)
            net_glob5.load_state_dict(w_glob_5)

        if w_locals_10:
            w_glob_10 = FedAvg(w_locals_10)
            net_glob10.load_state_dict(w_glob_10)

        if w_locals_15:
            w_glob_15 = FedAvg(w_locals_15)
            net_glob15.load_state_dict(w_glob_15)

        if w_locals_20:
            w_glob_20 = FedAvg(w_locals_20)
            net_glob20.load_state_dict(w_glob_20)

        if w_locals_25:
            w_glob_25 = FedAvg(w_locals_25)
            net_glob25.load_state_dict(w_glob_25)

        if w_locals_30:
            w_glob_30 = FedAvg(w_locals_30)
            net_glob30.load_state_dict(w_glob_30)



        # print loss
        if loss_locals:
            loss_avg = sum(loss_locals) / len(loss_locals)
        if loss_locals_1:
            loss_avg_1 = sum(loss_locals_1) / len(loss_locals_1)
        if loss_locals_5:
            loss_avg_5 = sum(loss_locals_5) / len(loss_locals_5)
        if loss_locals_10:
            loss_avg_10 = sum(loss_locals_10) / len(loss_locals_10)
        if loss_locals_15:
            loss_avg_15 = sum(loss_locals_15) / len(loss_locals_15)
        if loss_locals_20:
            loss_avg_20 = sum(loss_locals_20) / len(loss_locals_20)
        if loss_locals_25:
            loss_avg_25 = sum(loss_locals_25) / len(loss_locals_25)
        if loss_locals_30:
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

    net_glob10.eval()
    acc_train10, loss_train_10 = test_img(net_glob10, dataset_train, args)
    acc_test10, loss_test_10 = test_img(net_glob10, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_train10))
    print("Testing accuracy (CONSTANT ATTACK 10): {:.2f}".format(acc_test10))

    net_glob15.eval()
    acc_train15, loss_train_15 = test_img(net_glob15, dataset_train, args)
    acc_test15, loss_test_15 = test_img(net_glob15, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 15): {:.2f}".format(acc_train15))
    print("Testing accuracy (CONSTANT ATTACK 15): {:.2f}".format(acc_test15))

    net_glob20.eval()
    acc_train20, loss_train_20 = test_img(net_glob20, dataset_train, args)
    acc_test20, loss_test_20 = test_img(net_glob20, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 20): {:.2f}".format(acc_train20))
    print("Testing accuracy (CONSTANT ATTACK 20): {:.2f}".format(acc_test20))

    net_glob25.eval()
    acc_train25, loss_train_25 = test_img(net_glob25, dataset_train, args)
    acc_test25, loss_test_25 = test_img(net_glob25, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 25): {:.2f}".format(acc_train25))
    print("Testing accuracy (CONSTANT ATTACK 25): {:.2f}".format(acc_test25))

    net_glob30.eval()
    acc_train30, loss_train_30 = test_img(net_glob30, dataset_train, args)
    acc_test30, loss_test_30 = test_img(net_glob30, dataset_test, args)
    print("Training accuracy (CONSTANT ATTACK 30): {:.2f}".format(acc_train30))
    print("Testing accuracy (CONSTANT ATTACK 30): {:.2f}".format(acc_test30))
