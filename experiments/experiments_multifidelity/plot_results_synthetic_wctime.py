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

def plot_synthetic(q=1, Parallel=False, Sync=False, num_sample=False, wctime=False):
    STR_NUM_WORKER = 'Q='+str(q)
    plot_max = 2
    plot_min = 1e-5

    if Parallel or Sync:
        NUM_COL = 1
    else:
        NUM_COL = 2
    if Parallel:
        plot_min=1e-5

    color_dict = {"MFMES": 'red', "MES": 'green', 'MFPES': 'blue', 'BOCA': 'orange', 'MFSKO': 'purple', 'GIBBON': 'black', 'Batch_MES': 'olive', 'LP': 'brown' , 'GP_UCB_PE': 'gray', 'TS': 'deepskyblue'}
    line_dict = {'MF-para': 'solid', 'MF-seq': 'solid', 'SF-seq': 'solid', 'SF-para': 'solid'}
    marker_dict ={'1': 'o', '10': None, '50': 'x'}

    seeds_num = 10
    seeds = np.arange(seeds_num)
    func_names = ['SynFun_ell=0.05-d=3-seed0', 'SynFun_ell=0.05-d=3-seed1', 'SynFun_ell=0.05-d=3-seed2', 'SynFun_ell=0.05-d=3-seed3', 'SynFun_ell=0.05-d=3-seed4']
    func_names.extend(['SynFun_ell=0.05-d=3-seed5', 'SynFun_ell=0.05-d=3-seed6', 'SynFun_ell=0.05-d=3-seed7', 'SynFun_ell=0.05-d=3-seed8', 'SynFun_ell=0.05-d=3-seed9'])
    # func_names = ['SynFun_ell=0.1-d=3-seed0', 'SynFun_ell=0.1-d=3-seed1', 'SynFun_ell=0.1-d=3-seed2', 'SynFun_ell=0.1-d=3-seed3', 'SynFun_ell=0.1-d=3-seed4']
    # func_names.extend(['SynFun_ell=0.1-d=3-seed5', 'SynFun_ell=0.1-d=3-seed6', 'SynFun_ell=0.1-d=3-seed7', 'SynFun_ell=0.1-d=3-seed8', 'SynFun_ell=0.1-d=3-seed9'])
    func_names = ['SynFun_ell=0.2-d=3-seed0', 'SynFun_ell=0.2-d=3-seed1', 'SynFun_ell=0.2-d=3-seed2', 'SynFun_ell=0.2-d=3-seed3', 'SynFun_ell=0.2-d=3-seed4']
    func_names.extend(['SynFun_ell=0.2-d=3-seed5', 'SynFun_ell=0.2-d=3-seed6', 'SynFun_ell=0.2-d=3-seed7', 'SynFun_ell=0.2-d=3-seed8', 'SynFun_ell=0.2-d=3-seed9'])
    models = ['MFGP_', '', 'MTGP_']
    if Parallel:
        BO_methods = ['MFMES_RFM', 'Parallel_MFMES_RFM_'+STR_NUM_WORKER, 'Parallel_MES_RFM_'+STR_NUM_WORKER, 'MES_LP_'+STR_NUM_WORKER, 'GP_UCB_PE_'+STR_NUM_WORKER, 'AsyncTS_'+STR_NUM_WORKER]
        markers_iterator = itertools.cycle(('o', 's', '>', '<', 'd', 'x'))
    elif Sync:
        # BO_methods = ['MFMES_RFM', 'MES_RFM', 'Sync_MFMES_RFM_'+STR_NUM_WORKER, 'Parallel_MES_RFM_'+STR_NUM_WORKER, 'Batch_MES_RFM_'+STR_NUM_WORKER]
        BO_methods = ['MFMES_RFM', 'Sync_MFMES_RFM_'+STR_NUM_WORKER, 'MES_RFM', 'Parallel_MES_RFM_'+STR_NUM_WORKER, 'Batch_MES_RFM_'+STR_NUM_WORKER]
        markers_iterator = itertools.cycle(('o', 's', '>', 'x', 'd'))
        if q != 4:
            print('sync experiments with q not equal 4 have not done.')
    elif num_sample:
        BO_methods = ['MFMES_RFM1', 'MFMES_RFM', 'MFMES_RFM50', 'MFPES1', 'MFPES', 'MFPES50', 'MES_RFM1', 'MES_RFM', 'MES_RFM50']
    else:
        BO_methods = ['MFMES_RFM1', 'MFMES_RFM', 'MFMES_RFM50', 'MFPES1', 'MFPES', 'MFPES50', 'MES_RFM1', 'MES_RFM', 'MES_RFM50', 'BOCA', 'MFSKO', 'GIBBON']
        markers_iterator = itertools.cycle(('v', 'o', '^', 'v', 's', '^', 'v', '>', '^', '<', 'd', 'x'))

    if 'SynFun' in func_names[0]:
        test_func = eval('test_functions.SynFun')()
        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3
        elif 'd=4' in func_names[0]:
            input_dim = 4
        COST_INI = 5*5*input_dim
        COST_MAX = 150*input_dim
        COST_MAX = 450

    cost_ratio = 1
    COST_MAX += 1
    plot_cost = np.arange(COST_MAX * 60 * cost_ratio)

    fig = plt.figure()
    # Simple regret plot
    ax = plt.subplot(1, 1, 1)
    markers, markers_iterator = itertools.tee(markers_iterator, 2)
    for method in BO_methods:
        for model in models:
            SimReg_all = np.ones((seeds_num*len(func_names), COST_MAX * 60 * cost_ratio)) * np.inf
            for i, func_name in enumerate(func_names):
                GLOBAL_MAX = None
                # 0125dhirのパラメータ設定%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if func_name=='SynFun_ell=0.05-d=3-seed0':
                    GLOBAL_MAX = 5.031059493345406
                if func_name=='SynFun_ell=0.05-d=3-seed1':
                    GLOBAL_MAX = 4.374313613017984
                if func_name=='SynFun_ell=0.05-d=3-seed2':
                    GLOBAL_MAX = 4.00052359472389
                if func_name=='SynFun_ell=0.05-d=3-seed3':
                    GLOBAL_MAX = 4.244909566041141
                if func_name=='SynFun_ell=0.05-d=3-seed4':
                    GLOBAL_MAX = 4.5309594023802955
                if func_name=='SynFun_ell=0.05-d=3-seed5':
                    GLOBAL_MAX = 4.047994024791497
                if func_name=='SynFun_ell=0.05-d=3-seed6':
                    GLOBAL_MAX = 4.682209280768367
                if func_name=='SynFun_ell=0.05-d=3-seed7':
                    GLOBAL_MAX = 3.5508995357341497
                if func_name=='SynFun_ell=0.05-d=3-seed8':
                    GLOBAL_MAX = 4.003416602329548
                if func_name=='SynFun_ell=0.05-d=3-seed9':
                    GLOBAL_MAX = 4.448605011574502

                # # feature size = 1000
                # if func_name=='SynFun_ell=0.1-d=3-seed0':
                #     GLOBAL_MAX =3.4941486117314584
                # if func_name=='SynFun_ell=0.1-d=3-seed1':
                #     GLOBAL_MAX = 3.777271511426723
                # if func_name=='SynFun_ell=0.1-d=3-seed2':
                #     GLOBAL_MAX = 3.399427844596842
                # if func_name=='SynFun_ell=0.1-d=3-seed3':
                #     GLOBAL_MAX = 3.7830556644126045
                # if func_name=='SynFun_ell=0.1-d=3-seed4':
                #     GLOBAL_MAX = 3.3143127191397013
                # if func_name=='SynFun_ell=0.1-d=3-seed5':
                #     GLOBAL_MAX = 3.3993847652431355
                # if func_name=='SynFun_ell=0.1-d=3-seed6':
                #     GLOBAL_MAX = 3.3348217029037017
                # if func_name=='SynFun_ell=0.1-d=3-seed7':
                #     GLOBAL_MAX = 3.2469869851581996
                # if func_name=='SynFun_ell=0.1-d=3-seed8':
                #     GLOBAL_MAX = 3.617150751397999
                # if func_name=='SynFun_ell=0.1-d=3-seed9':
                #     GLOBAL_MAX = 3.773216957061331

                # feature size = 2000
                if func_name=='SynFun_ell=0.1-d=3-seed0':
                    GLOBAL_MAX = 3.4469825359421264
                if func_name=='SynFun_ell=0.1-d=3-seed1':
                    GLOBAL_MAX = 4.135979467669553
                if func_name=='SynFun_ell=0.1-d=3-seed2':
                    GLOBAL_MAX = 3.6217187759634966
                if func_name=='SynFun_ell=0.1-d=3-seed3':
                    GLOBAL_MAX = 3.7085719078940502
                if func_name=='SynFun_ell=0.1-d=3-seed4':
                    GLOBAL_MAX = 3.5080916354252203
                if func_name=='SynFun_ell=0.1-d=3-seed5':
                    GLOBAL_MAX = 3.173353637468107
                if func_name=='SynFun_ell=0.1-d=3-seed6':
                    GLOBAL_MAX = 4.682209280768467
                if func_name=='SynFun_ell=0.1-d=3-seed7':
                    GLOBAL_MAX = 3.506638604648495
                if func_name=='SynFun_ell=0.1-d=3-seed8':
                    GLOBAL_MAX = 4.003416602329574
                if func_name=='SynFun_ell=0.1-d=3-seed9':
                    GLOBAL_MAX = 3.92568706382066

                # feature size = 2000
                if func_name=='SynFun_ell=0.2-d=3-seed0':
                    GLOBAL_MAX = 3.446982535942134
                if func_name=='SynFun_ell=0.2-d=3-seed1':
                    GLOBAL_MAX = 2.8805623378775342
                if func_name=='SynFun_ell=0.2-d=3-seed2':
                    GLOBAL_MAX = 3.621718775963503
                if func_name=='SynFun_ell=0.2-d=3-seed3':
                    GLOBAL_MAX = 2.082905983171184
                if func_name=='SynFun_ell=0.2-d=3-seed4':
                    GLOBAL_MAX = 3.488175503464955
                if func_name=='SynFun_ell=0.2-d=3-seed5':
                    GLOBAL_MAX = 3.1577579649103553
                if func_name=='SynFun_ell=0.2-d=3-seed6':
                    GLOBAL_MAX = 2.274376458147822
                if func_name=='SynFun_ell=0.2-d=3-seed7':
                    GLOBAL_MAX = 3.145705623171234
                if func_name=='SynFun_ell=0.2-d=3-seed8':
                    GLOBAL_MAX = 3.140006941078865
                if func_name=='SynFun_ell=0.2-d=3-seed9':
                    GLOBAL_MAX = 2.437838037052898


                # 0125dhirのパラメータ設定%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

                            # if not Sync:
                            #     cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            # else:
                            #     cost = np.round(cost * cost_ratio * 60)
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
                # marker = None
                color = None

                if 'MFMES' in method:
                    color = color_dict["MFMES"]
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
                        # marker = marker_dict['1']
                        linestyle='dashed'
                    elif '50' in method:
                        sample_size = 50
                        # marker = marker_dict['50']
                        linestyle='dotted'
                    else:
                        sample_size = 10

                    # linestyle=line_dict['MF-seq']
                    if 'MFMES' in method:
                        label='MF-MES'
                        label = label + ' ' + str(sample_size)
                    elif 'MFPES' in method:
                        label='MF-PES'
                        label = label + ' ' + str(sample_size)
                    elif 'MES' in method:
                        label='MES'
                        label = label + ' ' + str(sample_size)
                        # linestyle = line_dict['SF-seq']
                    else:
                        label=method
                        if 'MFSKO' in method:
                            label = 'MF-SKO'


                index = SimReg_ave != -np.inf
                # plt.plot(plot_cost[index], SimReg_ave[index], label=label)
                # plt.fill_between(plot_cost[index], SimReg_ave[index] - SimReg_se[index], SimReg_ave[index] + SimReg_se[index], alpha=0.3)
                # plt.plot(plot_cost[index], SimReg_ave[index], label=(method+'-'+model.rstrip('_')).rstrip('-'))

                error_bar_plot = SimReg_se[index]
                if Sync:
                    errorevery = 20
                elif Parallel:
                    errorevery = 20
                else:
                    errorevery = 30
                errorevery *= 100
                # if Sync:
                #     errorevery *= q
                error_mark_every = (int(BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)
                ax.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, SimReg_ave[index], yerr=error_bar_plot, errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle,color=color, markerfacecolor="None")
    # ax.title(func_name + ' (d='+str(input_dim)+')')
    if Parallel or Sync:
        ax.set_title('Synthetic function (M=2, d=3, '+STR_NUM_WORKER+')', x = 0.45, y = 1.0)
    else:
        ax.set_title('Synthetic function (M=2, d=3)')

    ax.set_xlim(0, (COST_MAX - COST_INI) * cost_ratio)
    ax.set_ylim(plot_min, plot_max)
    ax.set_yscale('log')
    ax.grid(which='major')
    ax.grid(which='minor')
    if not Sync:
        # ax.set_xlabel('Wall-clock time (min)')
        ax.set_xlabel('Total cost')
    else:
        ax.set_xlabel('Sum of costs')
    ax.set_ylabel('Simple regret')
    if q==8:
        ax.legend(loc='upper right', ncol=NUM_COL)
    # ax.legend(loc='best', ncol=NUM_COL)
    # ax.tight_layout()

    if Parallel:
        fig.savefig('plots/Results_SimMax_Syn_log_'+STR_NUM_WORKER+'.pdf')
    elif Sync:
        fig.savefig('plots/Results_SimMax_Syn_log_Sync.pdf')
    elif num_sample:
        fig.savefig('plots/Results_SimMax_Syn_log_samplesize.pdf')
    else:
        fig.savefig('plots/Results_SimMax_Syn_log.pdf')

    handles, labels = ax.get_legend_handles_labels()
    plt.close()

    legend_fig = plt.figure()
    legend_fig_ax = legend_fig.add_subplot(1,1,1)

    legend_fig_ax.legend(handles, labels, ncol=1, loc='upper left')
    legend_fig_ax.axis("off")
    if Parallel:
        legend_fig.savefig('plots/Results_Syn_'+STR_NUM_WORKER+'_legend.pdf')
    elif Sync:
        legend_fig.savefig('plots/Results_Syn_Sync_legend.pdf')
    elif num_sample:
        legend_fig.savefig('plots/Results_Syn_samplesize_legend.pdf')
    else:
        legend_fig.savefig('plots/Results_Syn_legend.pdf')

    plt.close()

    # fig = plt.figure(figsize=(7, 5))
    # Inference regret plot
    # plt.subplot(1, 2, 2)
    plot_min = 5*1e-7
    markers, markers_iterator = itertools.tee(markers_iterator, 2)
    for method in BO_methods:
        for model in models:
            InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX * 60 * cost_ratio)) * np.inf
            for i, func_name in enumerate(func_names):
                GLOBAL_MAX = None
                # 0125dhirのパラメータ設定%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if func_name=='SynFun_ell=0.05-d=3-seed0':
                    GLOBAL_MAX = 5.031059493345406
                if func_name=='SynFun_ell=0.05-d=3-seed1':
                    GLOBAL_MAX = 4.374313613017984
                if func_name=='SynFun_ell=0.05-d=3-seed2':
                    GLOBAL_MAX = 4.00052359472389
                if func_name=='SynFun_ell=0.05-d=3-seed3':
                    GLOBAL_MAX = 4.244909566041141
                if func_name=='SynFun_ell=0.05-d=3-seed4':
                    GLOBAL_MAX = 4.5309594023802955
                if func_name=='SynFun_ell=0.05-d=3-seed5':
                    GLOBAL_MAX = 4.047994024791497
                if func_name=='SynFun_ell=0.05-d=3-seed6':
                    GLOBAL_MAX = 4.682209280768367
                if func_name=='SynFun_ell=0.05-d=3-seed7':
                    GLOBAL_MAX = 3.5508995357341497
                if func_name=='SynFun_ell=0.05-d=3-seed8':
                    GLOBAL_MAX = 4.003416602329548
                if func_name=='SynFun_ell=0.05-d=3-seed9':
                    GLOBAL_MAX = 4.448605011574502

                # # feature size = 1000
                # if func_name=='SynFun_ell=0.1-d=3-seed0':
                #     GLOBAL_MAX =3.4941486117314584
                # if func_name=='SynFun_ell=0.1-d=3-seed1':
                #     GLOBAL_MAX = 3.777271511426723
                # if func_name=='SynFun_ell=0.1-d=3-seed2':
                #     GLOBAL_MAX = 3.399427844596842
                # if func_name=='SynFun_ell=0.1-d=3-seed3':
                #     GLOBAL_MAX = 3.7830556644126045
                # if func_name=='SynFun_ell=0.1-d=3-seed4':
                #     GLOBAL_MAX = 3.3143127191397013
                # if func_name=='SynFun_ell=0.1-d=3-seed5':
                #     GLOBAL_MAX = 3.3993847652431355
                # if func_name=='SynFun_ell=0.1-d=3-seed6':
                #     GLOBAL_MAX = 3.3348217029037017
                # if func_name=='SynFun_ell=0.1-d=3-seed7':
                #     GLOBAL_MAX = 3.2469869851581996
                # if func_name=='SynFun_ell=0.1-d=3-seed8':
                #     GLOBAL_MAX = 3.617150751397999
                # if func_name=='SynFun_ell=0.1-d=3-seed9':
                #     GLOBAL_MAX = 3.773216957061331

                # feature size = 2000
                if func_name=='SynFun_ell=0.1-d=3-seed0':
                    GLOBAL_MAX = 3.4469825359421264
                if func_name=='SynFun_ell=0.1-d=3-seed1':
                    GLOBAL_MAX = 4.135979467669553
                if func_name=='SynFun_ell=0.1-d=3-seed2':
                    GLOBAL_MAX = 3.6217187759634966
                if func_name=='SynFun_ell=0.1-d=3-seed3':
                    GLOBAL_MAX = 3.7085719078940502
                if func_name=='SynFun_ell=0.1-d=3-seed4':
                    GLOBAL_MAX = 3.5080916354252203
                if func_name=='SynFun_ell=0.1-d=3-seed5':
                    GLOBAL_MAX = 3.173353637468107
                if func_name=='SynFun_ell=0.1-d=3-seed6':
                    GLOBAL_MAX = 4.682209280768467
                if func_name=='SynFun_ell=0.1-d=3-seed7':
                    GLOBAL_MAX = 3.506638604648495
                if func_name=='SynFun_ell=0.1-d=3-seed8':
                    GLOBAL_MAX = 4.003416602329574
                if func_name=='SynFun_ell=0.1-d=3-seed9':
                    GLOBAL_MAX = 3.92568706382066

                # feature size = 2000
                if func_name=='SynFun_ell=0.2-d=3-seed0':
                    GLOBAL_MAX = 3.446982535942134
                if func_name=='SynFun_ell=0.2-d=3-seed1':
                    GLOBAL_MAX = 2.8805623378775342
                if func_name=='SynFun_ell=0.2-d=3-seed2':
                    GLOBAL_MAX = 3.621718775963503
                if func_name=='SynFun_ell=0.2-d=3-seed3':
                    GLOBAL_MAX = 2.082905983171184
                if func_name=='SynFun_ell=0.2-d=3-seed4':
                    GLOBAL_MAX = 3.488175503464955
                if func_name=='SynFun_ell=0.2-d=3-seed5':
                    GLOBAL_MAX = 3.1577579649103553
                if func_name=='SynFun_ell=0.2-d=3-seed6':
                    GLOBAL_MAX = 2.274376458147822
                if func_name=='SynFun_ell=0.2-d=3-seed7':
                    GLOBAL_MAX = 3.145705623171234
                if func_name=='SynFun_ell=0.2-d=3-seed8':
                    GLOBAL_MAX = 3.140006941078865
                if func_name=='SynFun_ell=0.2-d=3-seed9':
                    GLOBAL_MAX = 2.437838037052898

                # 0125dhirのパラメータ設定%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

                            # if not Sync:
                            #     cost = np.round(cost * cost_ratio * 60 + model_computation_time + preprocessing_time + optimize_acquisition_time)
                            # else:
                            #     cost = np.round(cost * cost_ratio * 60)
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
                # marker = None
                color = None

                if 'MFMES' in method:
                    color = color_dict["MFMES"]
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
                        # marker = marker_dict['1']
                        linestyle='dashed'
                    elif '50' in method:
                        sample_size = 50
                        # marker = marker_dict['50']
                        linestyle='dotted'
                    else:
                        sample_size = 10

                    # linestyle=line_dict['MF-seq']
                    if 'MFMES' in method:
                        label='MF-MES'
                        label = label + ' ' + str(sample_size)
                    elif 'MFPES' in method:
                        label='MF-PES'
                        label = label + ' ' + str(sample_size)
                    elif 'MES' in method:
                        label='MES'
                        label = label + ' ' + str(sample_size)
                        # linestyle = line_dict['SF-seq']
                    else:
                        label=method
                        if 'MFSKO' in method:
                            label = 'MF-SKO'



                index = InfReg_ave != -np.inf
                # plt.plot(plot_cost[index], InfReg_ave[index], label=label)
                # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)

                error_bar_plot = InfReg_se[index]
                if Sync:
                    errorevery = 20
                elif Parallel:
                    errorevery = 20
                else:
                    errorevery = 30
                errorevery *= 100

                error_mark_every = (int(BO_methods.index(method) * errorevery / len(BO_methods)), errorevery)
                plt.errorbar(plot_cost[index] / 60. - COST_INI*cost_ratio, InfReg_ave[index], yerr=error_bar_plot, errorevery=error_mark_every, capsize=4, elinewidth=2, label=label, marker=next(markers), markevery=error_mark_every, linestyle=linestyle,color=color, markerfacecolor="None")
    # plt.title(func_name + ' (d='+str(input_dim)+')')
    if Parallel or Sync:
        plt.title('Synthetic function (M=2, d=3, '+STR_NUM_WORKER+')', x = 0.45, y = 1.0)
    else:
        plt.title('Synthetic function (M=2, d=3)')
    plt.xlim(0, (COST_MAX - COST_INI) * cost_ratio)
    plt.ylim(plot_min, plot_max)
    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    if not Sync:
        # plt.xlabel('Wall-clock time (min)')
        plt.xlabel('Total cost')
    else:
        plt.xlabel('Sum of costs')
    plt.ylabel('Inference regret')
    # if q==2 or (not Parallel and not Sync):
    #     plt.legend(loc='best', ncol=NUM_COL)
    if q==8:
        plt.legend(loc='upper right', ncol=NUM_COL)
    # plt.legend(loc='best', ncol=NUM_COL)
    # plt.tight_layout()

    if Parallel:
        plt.savefig('plots/Results_InfMax_Syn_log_'+STR_NUM_WORKER+'.pdf')
    elif Sync:
        plt.savefig('plots/Results_InfMax_Syn_log_Sync.pdf')
    elif num_sample:
        plt.savefig('plots/Results_InfMax_Syn_log_samplesize.pdf')
    else:
        plt.savefig('plots/Results_InfMax_Syn_log.pdf')
    plt.close()


if __name__ == '__main__':
    # plot_synthetic(num_sample=True)
    # wc_flag = True
    # plot_synthetic(Parallel=False, wctime=wc_flag)
    # for q in [2,4,8]:
    #     plot_synthetic(q=q, Parallel=True, wctime=wc_flag)
    # plot_synthetic(q=4, Sync=True)

    wc_flag = False
    plot_synthetic(Parallel=False, wctime=wc_flag)
    for q in [2,4,8]:
        plot_synthetic(q=q, Parallel=True, wctime=wc_flag)
    plot_synthetic(q=4, Sync=True, wctime=wc_flag)