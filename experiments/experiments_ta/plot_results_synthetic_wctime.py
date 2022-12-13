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

plt.rcParams['legend.borderaxespad'] = 0.2
plt.rcParams['legend.borderpad'] = 0.25
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams["legend.handletextpad"] = 0.25
plt.rcParams['legend.handlelength'] = 2.
plt.rcParams['legend.handleheight'] = 1.


def plot_synthetic(q=1, Parallel=False, Sync=False, num_sample=False):
    STR_NUM_WORKER = 'Q='+str(q)
    plot_max = 2
    plot_min = 1e-3

    if Parallel or Sync:
        NUM_COL = 1
    else:
        NUM_COL = 1
    if Parallel:
        plot_min=1e-5

    color_dict = {"TAMFMES": 'red', "MFMES": 'red', "KG": 'black', "MES": 'green', 'MFPES': 'blue', 'BOCA': 'orange', 'MFSKO': 'purple', 'GIBBON': 'black', 'Batch_MES': 'olive', 'LP': 'brown' , 'GP_UCB_PE': 'gray', 'TS': 'deepskyblue'}
    line_dict = {'MF-para': 'solid', 'MF-seq': 'solid', 'SF-seq': 'solid', 'SF-para': 'solid'}
    marker_dict ={'1': 'o', '10': None, '50': 'x'}

    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = ['SynFun_for_ta_ell=0.1-d=3-seed0', 'SynFun_for_ta_ell=0.1-d=3-seed1', 'SynFun_for_ta_ell=0.1-d=3-seed2', 'SynFun_for_ta_ell=0.1-d=3-seed3', 'SynFun_for_ta_ell=0.1-d=3-seed4']
    func_names.extend(['SynFun_for_ta_ell=0.1-d=3-seed5', 'SynFun_for_ta_ell=0.1-d=3-seed6', 'SynFun_for_ta_ell=0.1-d=3-seed7', 'SynFun_for_ta_ell=0.1-d=3-seed8', 'SynFun_for_ta_ell=0.1-d=3-seed9'])

    models = ['MFGP_', '', 'MTGP_']
    BO_methods = ['MFMES_RFM', 'MFMES_RFM_res', 'TAMFMES_RFM', 'TA_KG']
    BO_methods = ['MFMES_RFM', 'TAMFMES_RFM', 'TA_KG']
    markers_iterator = itertools.cycle(('o', 's', 'x'))

    if 'SynFun' in func_names[0]:
        test_func = eval('test_functions.SynFun_for_ta')()
        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3
        elif 'd=4' in func_names[0]:
            input_dim = 4
        COST_INI = 5*input_dim * np.max(test_func.cost)
        COST_MAX = 150*input_dim
        COST_MAX = COST_INI + input_dim * 60

    cost_ratio = 1
    COST_MAX += 1
    plot_cost = np.arange(COST_MAX * 60 * cost_ratio)

    # fig = plt.figure(figsize=(7, 5))
    # Simple regret plot
    # plt.subplot(1, 2, 1)
    markers, markers_iterator = itertools.tee(markers_iterator, 2)
    for method in BO_methods:
        for model in models:
            SimReg_all = np.ones((seeds_num*len(func_names), COST_MAX * 60 * cost_ratio)) * np.inf
            for i, func_name in enumerate(func_names):
                GLOBAL_MAX = None
                # 最大値設定 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed0':
                    GLOBAL_MAX = 2.7517570150971675
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed1':
                    GLOBAL_MAX = 2.7840396458698535
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed2':
                    GLOBAL_MAX = 2.8352314743334937
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed3':
                    GLOBAL_MAX = 3.007286994446803
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed4':
                    GLOBAL_MAX = 3.178208878540384
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed5':
                    GLOBAL_MAX = 2.701028226956206
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed6':
                    GLOBAL_MAX = 2.6622675496563164
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed7':
                    GLOBAL_MAX = 2.478251813988675
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed8':
                    GLOBAL_MAX = 2.43375297656076
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed9':
                    GLOBAL_MAX = 3.1436128114460002
                # 最大値設定 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                result_path = func_name + '_results/'

                plot=True
                for seed in seeds:
                    temp_path = result_path + method + '/' + model + 'seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                if Sync and ( 'Parallel_MES' in method or 'Batch_MES' in method ):
                                    cost = cost * q - np.min(cost)*(q-1)

                            with open(temp_path + 'model_computation_time.pickle', 'rb') as f:
                                model_computation_time = np.array(pickle.load(f))

                            with open(temp_path + 'preprocessing_time.pickle', 'rb') as f:
                                preprocessing_time = np.array(pickle.load(f))

                            with open(temp_path + 'optimize_acquisition_time.pickle', 'rb') as f:
                                optimize_acquisition_time = np.array(pickle.load(f))

                            # cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            cost = np.round(cost * cost_ratio * 60)


                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(SimReg):
                                    if j+1 == np.size(cost):
                                        # SimReg_all[seed+i*10, -1] = SimReg[j]
                                        break
                                    else:
                                        SimReg_all[seed+i*10, int(cost[j]) : int(cost[j+1])] = SimReg[j]

                    else:
                        plot=False

                # 各関数でregretに変換
                SimReg_all[i*10:(i+1)*10, :] = GLOBAL_MAX - SimReg_all[i*10:(i+1)*10, :]
            if plot:
                SimReg_ave = np.mean(SimReg_all, axis=0)
                SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))
                linestyle = None
                marker = None
                color = None
                label = method

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

                if Parallel or Sync:
                    linestyle = line_dict['SF-para']

                    if 'Sync' in method:
                        label='Sync MF-MES'
                    elif 'Parallel_MFMES' in method:
                        label='Async MF-MES'
                        linestyle = line_dict['MF-para']
                    elif 'MFMES' in method:
                        label = 'MF-MES'
                        linestyle = line_dict['MF-seq']
                        linestyle = 'dashed'
                    elif 'Parallel_MES' in method:
                        label='Async MES'
                        if Sync:
                            label='Sync MES'
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
                    if '1' in method:
                        sample_size = 1
                        marker = marker_dict['1']
                    elif '50' in method:
                        sample_size = 50
                        marker = marker_dict['50']
                    else:
                        sample_size = 10

                    linestyle=line_dict['MF-seq']
                    if 'TAMFMES' in method:
                        label='TA-MF-MES'
                        # label = label + ' ' + str(sample_size)
                    elif 'TA_KG' in method:
                            label='TA-KG'
                            label = label
                    elif 'MFMES_RFM_res' in method:
                        label='MF-MES-res'
                        # label = label + ' ' + str(sample_size)
                        color=None
                    elif 'MFMES' in method:
                        label='MF-MES'
                        # label = label + ' ' + str(sample_size)
                        linestyle = 'dashed'
                    elif 'MFPES' in method:
                        label='MF-PES'
                        label = label + ' ' + str(sample_size)
                    elif 'MES' in method:
                        label='MES'
                        label = label + ' ' + str(sample_size)
                        linestyle = line_dict['SF-seq']
                    else:
                        label=method
                        if 'MFSKO' in method:
                            label = 'MF-SKO'


                index = SimReg_ave != -np.inf
                # plt.plot(plot_cost[index], SimReg_ave[index], label=label)
                # plt.fill_between(plot_cost[index], SimReg_ave[index] - SimReg_se[index], SimReg_ave[index] + SimReg_se[index], alpha=0.3)
                # plt.plot(plot_cost[index], SimReg_ave[index], label=(method+'-'+model.rstrip('_')).rstrip('-'))

                error_bar_plot = SimReg_se[index]
                if Parallel:
                    errorevery = 20
                else:
                    errorevery = 50
                errorevery *= 20
                # if Sync:
                #     errorevery *= q
                error_mark_every = (int(BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)
                plt.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, SimReg_ave[index], yerr=error_bar_plot, errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle,color=color, markerfacecolor="None")
    # plt.title(func_name + ' (d='+str(input_dim)+')')
    if Parallel or Sync:
        plt.title('Synthetic function (M='+str(test_func.M)+', d='+str(test_func.d)+', '+STR_NUM_WORKER+')')
    else:
        plt.title('Synthetic function (M='+str(test_func.M)+', d='+str(test_func.d)+')')

    plt.xlim(0, (COST_MAX - COST_INI) * cost_ratio)
    plt.ylim(plot_min, plot_max)
    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    # plt.xlabel('Wall-clock time (min)')
    plt.xlabel('Total cost')
    plt.ylabel('Simple regret')
    # if q==2 or (not Parallel):
    plt.legend(loc='best', ncol=NUM_COL)
    # plt.legend(loc='best', ncol=NUM_COL)
    # plt.tight_layout()

    if Parallel:
        plt.savefig('plots/Results_SimMax_Syn_log_'+STR_NUM_WORKER+'.pdf')
    elif Sync:
        plt.savefig('plots/Results_SimMax_Syn_log_Sync.pdf')
    elif num_sample:
        plt.savefig('plots/Results_SimMax_Syn_log_samplesize.pdf')
    else:
        plt.savefig('plots/Results_SimMax_Syn_log_trace.pdf')
    plt.close()

    # fig = plt.figure(figsize=(7, 5))
    # Inference regret plot
    # plt.subplot(1, 2, 2)
    markers, markers_iterator = itertools.tee(markers_iterator, 2)
    for method in BO_methods:
        for model in models:
            InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX * 60 * cost_ratio)) * np.inf
            for i, func_name in enumerate(func_names):
                GLOBAL_MAX = None
                # 最大値設定 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed0':
                    GLOBAL_MAX = 2.7517570150971675
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed1':
                    GLOBAL_MAX = 2.7840396458698535
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed2':
                    GLOBAL_MAX = 2.8352314743334937
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed3':
                    GLOBAL_MAX = 3.007286994446803
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed4':
                    GLOBAL_MAX = 3.178208878540384
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed5':
                    GLOBAL_MAX = 2.701028226956206
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed6':
                    GLOBAL_MAX = 2.6622675496563164
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed7':
                    GLOBAL_MAX = 2.478251813988675
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed8':
                    GLOBAL_MAX = 2.43375297656076
                if func_name=='SynFun_for_ta_ell=0.1-d=3-seed9':
                    GLOBAL_MAX = 3.1436128114460002
                # 最大値設定 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                result_path = func_name + '_results/'

                plot=True
                for seed in seeds:
                    temp_path = result_path + method + '/' + model + 'seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                if Sync and ( 'Parallel_MES' in method or 'Batch_MES' in method):
                                    cost = cost * q - np.min(cost)*(q-1)

                            with open(temp_path + 'model_computation_time.pickle', 'rb') as f:
                                model_computation_time = np.array(pickle.load(f))

                            with open(temp_path + 'preprocessing_time.pickle', 'rb') as f:
                                preprocessing_time = np.array(pickle.load(f))

                            with open(temp_path + 'optimize_acquisition_time.pickle', 'rb') as f:
                                optimize_acquisition_time = np.array(pickle.load(f))

                            # cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            cost = np.round(cost * cost_ratio * 60)


                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        # InfReg_all[seed+i*10, -1] = InfReg[j]
                                        break
                                    else:
                                        InfReg_all[seed+i*10, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False
                # 各関数でregretに変換
                InfReg_all[i*10:(i+1)*10, :] = GLOBAL_MAX - InfReg_all[i*10:(i+1)*10, :]
            if plot:
                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))
                linestyle = None
                marker = None
                color = None
                label = method

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

                if Parallel or Sync:
                    linestyle = line_dict['SF-para']
                    if 'Sync' in method:
                        label='Sync MF-MES'
                    elif 'Parallel_MFMES' in method:
                        label='Async MF-MES'
                        linestyle = line_dict['MF-para']
                    elif 'MFMES' in method:
                        label = 'MF-MES'
                        linestyle = line_dict['MF-seq']
                        linestyle = 'dashed'
                    elif 'Parallel_MES' in method:
                        label='Async MES'
                        if Sync:
                            label='Sync MES'
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
                    if '1' in method:
                        sample_size = 1
                        marker = marker_dict['1']
                    elif '50' in method:
                        sample_size = 50
                        marker = marker_dict['50']
                    else:
                        sample_size = 10

                    linestyle=line_dict['MF-seq']
                    if 'TAMFMES' in method:
                        label='TA-MF-MES'
                        # label = label + ' ' + str(sample_size)
                    elif 'TA_KG' in method:
                            label='TA-KG'
                            label = label
                    elif 'MFMES_RFM_res' in method:
                        label='MF-MES-res'
                        # label = label + ' ' + str(sample_size)
                        color=None
                    elif 'MFMES' in method:
                        label='MF-MES'
                        # label = label + ' ' + str(sample_size)
                        linestyle = 'dashed'
                    elif 'MFPES' in method:
                        label='MF-PES'
                        label = label + ' ' + str(sample_size)
                    elif 'MES' in method:
                        label='MES'
                        label = label + ' ' + str(sample_size)
                        linestyle = line_dict['SF-seq']
                    else:
                        label=method
                        if 'MFSKO' in method:
                            label = 'MF-SKO'


                index = InfReg_ave != -np.inf
                # plt.plot(plot_cost[index], InfReg_ave[index], label=label)
                # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)

                error_bar_plot = InfReg_se[index]
                if Parallel:
                    errorevery = 20
                else:
                    errorevery = 50
                errorevery *= 20

                error_mark_every = (int(BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)
                plt.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, InfReg_ave[index], yerr=error_bar_plot, errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle,color=color, markerfacecolor="None")
    # plt.title(func_name + ' (d='+str(input_dim)+')')
    if Parallel or Sync:
        plt.title('Synthetic function (M='+str(test_func.M)+', d='+str(test_func.d)+', '+STR_NUM_WORKER+')')
    else:
        plt.title('Synthetic function (M='+str(test_func.M)+', d='+str(test_func.d)+')')
    plt.xlim(0, (COST_MAX - COST_INI) * cost_ratio)
    plt.ylim(plot_min, plot_max)
    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    # plt.xlabel('Wall-clock time (min)')
    plt.xlabel('Total cost')
    plt.ylabel('Inference regret')
    # if q==2 or (not Parallel and not Sync):
    #     plt.legend(loc='best', ncol=NUM_COL)
    plt.legend(loc='best', ncol=NUM_COL)
    # plt.tight_layout()

    if Parallel:
        plt.savefig('plots/Results_InfMax_Syn_log_'+STR_NUM_WORKER+'.pdf')
    elif Sync:
        plt.savefig('plots/Results_InfMax_Syn_log_Sync.pdf')
    elif num_sample:
        plt.savefig('plots/Results_InfMax_Syn_log_samplesize.pdf')
    else:
        plt.savefig('plots/Results_InfMax_Syn_log_trace.pdf')
    plt.close()


if __name__ == '__main__':
    # plot_synthetic(num_sample=True)
    plot_synthetic(Parallel=False)
    # plot_synthetic(q=4, Sync=True)
    # for q in [2,4,8]:
    #     plot_synthetic(q=q, Parallel=True)
