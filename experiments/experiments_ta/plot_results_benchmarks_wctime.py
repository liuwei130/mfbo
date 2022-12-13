# -*- coding: utf-8 -*-
import os
import sys
import pickle
import itertools

import numpy as np
import numpy.matlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myBO.scripts.test_functions import test_functions

plt.rcParams['pdf.fonttype'] = 42 # Type3font回避
plt.rcParams['ps.fonttype'] = 42 # Type3font回避
plt.rcParams['font.family'] = 'sans-serif' # font familyの設定

plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
plt.rcParams["axes.titlesize"]=21
plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます
plt.rcParams['legend.fontsize'] = 22

plt.rcParams['figure.figsize'] = (6.5, 6.5)
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['errorbar.capsize'] = 4.0
plt.rcParams['lines.markersize'] = 12

plt.rcParams['lines.linewidth'] = 2.
plt.rcParams['lines.markeredgewidth'] = 1.5

plt.rcParams['legend.borderaxespad'] = 0.15
plt.rcParams['legend.borderpad'] = 0.2
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.5

def main(q=1, Parallel=False, wctime=False):
    STR_NUM_WORKER = 'Q='+str(q)
    seeds_num = 10
    seeds = np.arange(seeds_num)

    color_dict = {"TAMFMES": 'red', "MFMES": 'red', "KG": 'black', "MES": 'green', 'MFPES': 'blue', 'BOCA': 'orange', 'MFSKO': 'purple', 'GIBBON': 'black', 'Batch_MES': 'olive', 'LP': 'brown' , 'GP_UCB_PE': 'gray', 'TS': 'deepskyblue'}
    line_dict = {'MF-para': 'solid', 'MF-seq': 'solid', 'SF-seq': 'solid', 'SF-para': 'solid'}
    marker_dict ={'1': 'o', '10': None, '50': 'x'}

    func_names = ['cnn_mnist', 'cnn_cifar10']

    models = ['MFGP_', '', 'MTGP_']
    if Parallel:
        BO_methods = ['MFMES_RFM', 'Parallel_MFMES_RFM_'+STR_NUM_WORKER, 'Parallel_MES_RFM_'+STR_NUM_WORKER, 'MES_LP_'+STR_NUM_WORKER, 'GP_UCB_PE_'+STR_NUM_WORKER, 'AsyncTS_'+STR_NUM_WORKER]
    else:
        # BO_methods = ['MFMES_RFM_res', 'MFMES_RFM', 'TAMFMES_RFM', 'TA_KG']
        BO_methods = ['MFMES_RFM', 'TAMFMES_RFM', 'TA_KG']
        markers_iterator = itertools.cycle(('o', 's', 'x'))

    # fig = plt.figure(figsize=(10, len(func_names)*2.5))
    for i, func_name in enumerate(func_names):
        cost_ratio = 1

        func = eval('test_functions.'+func_name)()


        if 'cifar10' in func_name:
            COST_INI = 65
        else:
            COST_INI = 85
        COST_MAX = 150 + np.int(COST_INI)
        input_dim = func.d
        M = func.M
        GLOBAL_MAX = np.max(func.Y[-1])

        if 'mnist' in func_name:
            plot_y_max = 3*1e-2
            plot_y_min = 3*1e-4
        elif 'cifar10' in func_name:
            plot_y_max = 4*1e-1
            plot_y_min = 1e-4

        COST_MAX += 1
        errorevery = 5
        errorevery *= 180

        plot_cost = np.arange(COST_MAX * 60 * cost_ratio)
        result_path = func_name + '_results/'
        markers, markers_iterator = itertools.tee(markers_iterator, 2)
        for method in BO_methods:
            for model in models:
                if  (method != 'BOCA') and (func_name=='Material') and (model=='MTGP_'):
                    break
                plot=True
                SimReg_all = np.ones((seeds_num, COST_MAX * 60 * cost_ratio)) * np.inf
                for seed in seeds:
                    temp_path = result_path + method + '/' + model + 'seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = np.array(pickle.load(f))

                            with open(temp_path + 'model_computation_time.pickle', 'rb') as f:
                                model_computation_time = np.array(pickle.load(f))

                            with open(temp_path + 'preprocessing_time.pickle', 'rb') as f:
                                preprocessing_time = np.array(pickle.load(f))

                            with open(temp_path + 'optimize_acquisition_time.pickle', 'rb') as f:
                                optimize_acquisition_time = np.array(pickle.load(f))

                            if wctime:
                                cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            else:
                                cost = np.round(cost * cost_ratio * 60)


                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)

                            for j in range(np.size(cost)):
                                # if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    if cost[j] == COST_MAX-1:
                                        SimReg_all[seed, int(cost[j])] = SimReg[j]
                                else:
                                    SimReg_all[seed, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                    else:
                        plot=False


                if 'cnn' in func_name:
                    SimReg_all = 1. / (1 + np.exp(- SimReg_all))
                    SimReg_all[SimReg_all == 1] = np.inf
                    SimReg_all = 1. / (1 + np.exp(- GLOBAL_MAX)) - SimReg_all
                else:
                    SimReg_all = GLOBAL_MAX - SimReg_all

                if plot:
                    SimReg_ave = np.mean(SimReg_all, axis=0)
                    SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                    # if GLOBAL_MAX is None:
                    #     SimReg_ave = np.abs(SimReg_ave)
                    # else:
                    #     SimReg_ave = GLOBAL_MAX - SimReg_ave

                    index = SimReg_ave != -np.inf
                    linestyle = None
                    marker = None
                    color = None

                    if 'TAMFMES' in method:
                        color = color_dict["TAMFMES"]
                    elif 'MFMES' in method:
                        color = color_dict["MFMES"]
                    elif 'KG' in method:
                        color = color_dict["KG"]
                    elif 'MFPES' in method:
                        color = color_dict["MFPES"]
                    elif 'GIBBON' in method:
                        color = color_dict["GIBBON"]
                    elif 'LP' in method:
                        color = color_dict["LP"]
                    elif 'Batch_MES' in method:
                        color = color_dict["Batch_MES"]
                    elif 'MES' in method:
                        color = color_dict["MES"]
                    elif 'TS' in method:
                        color = color_dict["TS"]
                    elif 'GP_UCB_PE' in method:
                        color = color_dict["GP_UCB_PE"]
                    elif 'BOCA' in method:
                        color = color_dict["BOCA"]
                    elif 'MFSKO' in method:
                        color = color_dict["MFSKO"]

                    if Parallel:
                        linestyle = line_dict['SF-para']
                        if 'Elastic_Parallel_MFMES' in method:
                            label='Async MF-MES + EGP'
                            linestyle = "--"
                        elif 'gradient' in method:
                            if 'Sync' in method:
                                # label='Sync MF-MES + grad'
                                label='Sync MF-MES'
                            else:
                                # label='Async MF-MES + grad'
                                label='Async MF-MES'
                            linestyle = line_dict['MF-para']
                            # marker = marker_dict['gradient']
                        elif 'Parallel_MFMES' in method:
                            label='Async MF-MES + DIRECT'
                            label='Async MF-MES'
                            linestyle = line_dict['MF-para']
                        elif 'MFMES' in method:
                            label = 'MF-MES'
                            linestyle = line_dict['MF-seq']
                            linestyle = 'dashed'
                        elif 'Parallel_MES' in method:
                            label='Async MES'
                        elif 'Batch_MES' in method:
                            label='Batch MES'
                        elif 'AsyncTS' in method:
                            label = 'Async TS'
                        elif 'MES_RFM' in method:
                            label = 'MES'
                            linestyle = line_dict['SF-seq']
                            linestyle = 'dashed'
                        else:
                            label=(method.replace('_'+STR_NUM_WORKER, '')).replace('_', '-')
                    else:
                        linestyle = line_dict['MF-seq']
                        if 'TAMFMES' in method:
                            label='TA-MF-MES'
                            label = label
                        elif 'TA_KG' in method:
                            label='TA-KG'
                            label = label
                        elif 'MFMES_RFM_res' in method:
                            label='MF-MES-res'
                            label = label
                            color=None
                        elif 'MFMES' in method:
                            label='MF-MES'
                            linestyle='dashed'
                        elif 'MFPES' in method:
                            label='MF-PES'
                        elif 'MES' in method:
                            label='MES'
                            linestyle = line_dict['SF-seq']
                        else:
                            label=method
                            if 'MFSKO' in method:
                                label = 'MF-SKO'

                    error_mark_every = (int(BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)

                    plt.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, SimReg_ave[index], yerr=SimReg_se[index], errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle, color=color, markerfacecolor="None")


                    # plt.plot(plot_cost[index], SimReg_ave[index], label=(method+'-'+model.rstrip('_')).rstrip('-'))
                    # plt.fill_between(plot_cost[index], SimReg_ave[index] - SimReg_se[index], SimReg_ave[index] + SimReg_se[index], alpha=0.3)
                    # plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=errorevery, capsize=3, elinewidth=1, label=label)

                    # if np.max(SimReg_ave[index] + SimReg_se[index]/2.) < plot_y_max:
                    #     plot_y_max = np.max(SimReg_ave[index] + SimReg_se[index]/2.)

        if 'mnist' in func_name:
            func_name = 'CNN MNIST'
        elif 'cifar10' in func_name:
            func_name = 'CNN CIFAR10'

        if Parallel:
            plt.title(func_name + ' (M='+str(M)+', ' + 'd='+str(input_dim)+', ' +'Q=4)')
        else:
            plt.title(func_name + ' (M='+str(M)+', ' + 'd='+str(input_dim)+')')

        if 'MNIST' in func_name:
            func_name = 'cnn_mnist'
        elif 'CIFAR10' in func_name:
            func_name = 'cnn_cifar10'

        plt.xlim(0, (COST_MAX - COST_INI) * cost_ratio)
        # plt.ylim(plot_y_min, plot_y_max)
        plt.yscale('log')

        if wctime:
            plt.xlabel('Wall-clock time (min)')
        else:
            plt.xlabel('Total cost')

        plt.ylabel('Simple regret')
        plt.grid(which='major')
        plt.grid(which='minor')
        # plt.legend(loc='best')
        # plt.tight_layout()
        # if wctime and 'mnist' in func_name:
        #     plt.legend(loc='best', ncol=1)

        if wctime:
            if Parallel:
                plt.savefig('plots/Results_wctime_SimMax_Parallel_'+str(func_name)+'_log'+STR_NUM_WORKER+'.pdf')
            else:
                plt.savefig('plots/Results_wctime_SimMax_'+str(func_name)+'_log.pdf')
        else:
            if Parallel:
                plt.savefig('plots/Results_SimMax_Parallel_'+str(func_name)+'_log'+STR_NUM_WORKER+'.pdf')
            else:
                plt.savefig('plots/Results_SimMax_'+str(func_name)+'_log.pdf')
        plt.close()

        markers, markers_iterator = itertools.tee(markers_iterator, 2)
        for method in BO_methods:
            for model in models:
                if  (method != 'BOCA') and (func_name=='Material') and (model=='MTGP_'):
                    break
                plot=True
                InfReg_all = np.ones((seeds_num, COST_MAX * 60 * cost_ratio)) * np.inf
                for seed in seeds:
                    temp_path = result_path + method + '/' + model + 'seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)

                            with open(temp_path + 'model_computation_time.pickle', 'rb') as f:
                                model_computation_time = np.array(pickle.load(f))

                            with open(temp_path + 'preprocessing_time.pickle', 'rb') as f:
                                preprocessing_time = np.array(pickle.load(f))

                            with open(temp_path + 'optimize_acquisition_time.pickle', 'rb') as f:
                                optimize_acquisition_time = np.array(pickle.load(f))

                            if wctime:
                                cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            else:
                                cost = np.round(cost * cost_ratio * 60)

                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)
                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)
                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        if cost[j] == COST_MAX-1:
                                            InfReg_all[seed, int(cost[j])] = np.max([InfReg[j], SimReg[j]])
                                        break
                                    else:
                                        InfReg_all[seed, int(cost[j]) : int(cost[j+1])] = np.max([InfReg[j], SimReg[j]])
                                        # InfReg_all[seed, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False

                if 'cnn' in func_name:
                    InfReg_all = 1. / (1 + np.exp(- InfReg_all))
                    InfReg_all[InfReg_all == 1] = np.inf
                    InfReg_all = 1. / (1 + np.exp(- GLOBAL_MAX)) - InfReg_all
                else:
                    InfReg_all = GLOBAL_MAX - InfReg_all

                if plot:
                    InfReg_ave = np.mean(InfReg_all, axis=0)
                    InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                    # if GLOBAL_MAX is None:
                    #     InfReg_ave = np.abs(InfReg_ave)
                    # else:
                    #     InfReg_ave = GLOBAL_MAX - InfReg_ave

                    index = InfReg_ave != -np.inf
                    linestyle = None
                    marker = None
                    color = None

                    if 'TAMFMES' in method:
                        color = color_dict["TAMFMES"]
                    elif 'MFMES' in method:
                        color = color_dict["MFMES"]
                    elif 'KG' in method:
                        color = color_dict["KG"]
                    elif 'MFPES' in method:
                        color = color_dict["MFPES"]
                    elif 'GIBBON' in method:
                        color = color_dict["GIBBON"]
                    elif 'LP' in method:
                        color = color_dict["LP"]
                    elif 'Batch_MES' in method:
                        color = color_dict["Batch_MES"]
                    elif 'MES' in method:
                        color = color_dict["MES"]
                    elif 'TS' in method:
                        color = color_dict["TS"]
                    elif 'GP_UCB_PE' in method:
                        color = color_dict["GP_UCB_PE"]
                    elif 'BOCA' in method:
                        color = color_dict["BOCA"]
                    elif 'MFSKO' in method:
                        color = color_dict["MFSKO"]

                    if Parallel:
                        linestyle = line_dict['SF-para']
                        if 'Elastic_Parallel_MFMES' in method:
                            label='Async MF-MES + EGP'
                            linestyle = "--"
                        elif 'gradient' in method:
                            if 'Sync' in method:
                                # label='Sync MF-MES + grad'
                                label='Sync MF-MES'
                            else:
                                # label='Async MF-MES + grad'
                                label='Async MF-MES'
                            linestyle = line_dict['MF-para']
                            # marker = marker_dict['gradient']
                        elif 'Parallel_MFMES' in method:
                            label='Async MF-MES + DIRECT'
                            label='Async MF-MES'
                            linestyle = line_dict['MF-para']
                        elif 'MFMES' in method:
                            label = 'MF-MES'
                            linestyle = line_dict['MF-seq']
                            linestyle = 'dashed'
                        elif 'Parallel_MES' in method:
                            label='Async MES'
                        elif 'Batch_MES' in method:
                            label='Batch MES'
                        elif 'AsyncTS' in method:
                            label = 'Async TS'
                        elif 'MES_RFM' in method:
                            label = 'MES'
                            linestyle = line_dict['SF-seq']
                            linestyle = 'dashed'
                        else:
                            label=(method.replace('_'+STR_NUM_WORKER, '')).replace('_', '-')
                    else:
                        linestyle=line_dict['MF-seq']
                        if 'TAMFMES' in method:
                            label='TA-MF-MES'
                            label = label
                        elif 'TA_KG' in method:
                            label='TA-KG'
                            label = label
                        elif 'MFMES_RFM_res' in method:
                            label='MF-MES-res'
                            label = label
                            color=None
                        elif 'MFMES' in method:
                            label='MF-MES'
                            linestyle='dashed'
                        elif 'MFPES' in method:
                            label='MF-PES'
                        elif 'MES' in method:
                            label='MES'
                            linestyle = line_dict['SF-seq']
                        else:
                            label=method
                            if 'MFSKO' in method:
                                label = 'MF-SKO'


                    # plt.plot(plot_cost[index], InfReg_ave[index], label=(method+'-'+model.rstrip('_')).rstrip('-'))
                    # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)
                    # plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=errorevery, capsize=3, elinewidth=1, label=label)
                    error_mark_every = (int( BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)

                    plt.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, InfReg_ave[index], yerr=InfReg_se[index], errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle, color=color, markerfacecolor="None")

        if 'mnist' in func_name:
            func_name = 'CNN MNIST'
        elif 'cifar10' in func_name:
            func_name = 'CNN CIFAR10'

        if Parallel:
            plt.title(func_name + ' (M='+str(M)+', ' + 'd='+str(input_dim)+', ' +'Q=4)')
        else:
            plt.title(func_name + ' (M='+str(M)+', ' + 'd='+str(input_dim)+')')

        if 'MNIST' in func_name:
            func_name = 'cnn_mnist'
        elif 'CIFAR10' in func_name:
            func_name = 'cnn_cifar10'

        plt.xlim(0, (COST_MAX -  COST_INI)* cost_ratio)
        plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor')
        if 'mnist' in func_name:
            plt.ylim(1e-3, 1e-2)
            plt.yticks([1e-3, 1e-2, 6*1e-3, 4*1e-3, 3*1e-3, 2*1e-3], [r'$10^{-3}$', r'$10^{-2}$', '', '', '', ''])

        # plt.ylim(plot_y_min, plot_y_max)

        if wctime:
            plt.xlabel('Wall-clock time (min)')
        else:
            plt.xlabel('Total cost')

        plt.ylabel('Inference regret')

        # if wctime and 'mnist' in func_name:
        #     plt.legend(loc='best', ncol=1)
        # plt.legend(loc='best')
        # plt.tight_layout()

        if wctime:
            if Parallel:
                plt.savefig('plots/Results_wctime_InfMax_Parallel_'+str(func_name)+'_log'+STR_NUM_WORKER+'.pdf')
            else:
                plt.savefig('plots/Results_wctime_InfMax_'+str(func_name)+'_log.pdf')
        else:
            if Parallel:
                plt.savefig('plots/Results_InfMax_Parallel_'+str(func_name)+'_log'+STR_NUM_WORKER+'.pdf')
            else:
                plt.savefig('plots/Results_InfMax_'+str(func_name)+'_log.pdf')
        plt.close()

if __name__ == '__main__':
    # for q in [2,4,8]:
    #     main(q=q, Parallel=True)
    # main(q=4, Parallel=True)

    wc_flag = True
    main(Parallel=False, wctime=wc_flag)

    # wc_flag = False
    # main(Parallel=False, wctime=wc_flag)