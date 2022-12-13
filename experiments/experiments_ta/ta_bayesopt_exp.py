# -*- coding: utf-8 -*-
import os
import sys
import signal
import glob
import time
import random
import concurrent.futures
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import GPy

# add my modules
# from myBO.scripts.vanillaBO import bayesian_opt as BO
# from myBO.scripts.vanillaBO import parallel_bayesian_opt as PBO

# from myBO.scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO
from myBO.scripts.MultiFidelityBO import traceaware_multifidelity_bayesian_opt as TAMFBO


from myBO.scripts.test_functions import test_functions
from myBO.scripts.myutils import myutils

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(params):
    (func_name, BO_method, initial_seed, function_seed, MAX_NUM_WORKER) = params
    print(params)
    if not ((BO_method == 'BOCA') or ('MF' in BO_method) or (BO_method == 'GIBBON') or ('TA' in BO_method)):
        BO_type = 'Single'
    else:
        BO_type = 'MultiFidelity'

    # settings for test functions
    if func_name == 'Material':
        test_func = eval('test_functions.'+func_name)()
        M = test_func.M
        bounds = test_func.bounds
        input_dim = test_func.d
        interval_size = bounds[1] - bounds[0]
        kernel_bounds = np.c_[np.array([interval_size / 25., interval_size*4]), np.c_[[1e1, 1e3]]]
        if BO_type == 'Single':
            pool_X = test_func.X[-1].copy()
            X_all = pool_X.copy()
        else:
            pool_X = test_func.X.copy()
            X_all = pool_X[M-1].copy()
    elif 'cnn' in func_name:
        test_func = eval('test_functions.'+func_name)()
        M = test_func.M
        bounds = test_func.bounds
        input_dim = test_func.d
        interval_size = bounds[1] - bounds[0]
        kernel_bounds = np.c_[np.array([interval_size * 1e-2, interval_size * 1e2]), np.c_[[1, 1e2]]]
        if BO_type == 'Single':
            pool_X = test_func.X[-1].copy()
            X_all = pool_X.copy()
        else:
            pool_X = test_func.X.copy()
            X_all = pool_X[M-1].copy()
    else:
        test_func = eval('test_functions.'+func_name)()
        if 'SynFun' in func_name:
            test_func.func_sampling(seed=function_seed)
        M = test_func.M
        bounds = test_func.bounds
        input_dim = test_func.d
        interval_size = bounds[1] - bounds[0]
        kernel_bounds = np.array([interval_size / 10., interval_size])
        if BO_type == 'Single':
            pool_X = test_func.X[-1].copy()
            X_all = pool_X.copy()
        else:
            pool_X = test_func.X.copy()
            X_all = pool_X[M-1].copy()

    if func_name == 'Material':
        if BO_type == 'Single':
            cost = 60
        else:
            cost = np.array([5, 10, 60])
    elif 'cnn' in func_name:
        if BO_type == 'Single':
            cost = test_func.fidelity_features[-1]
        else:
            cost = test_func.fidelity_features.copy()
    else:
        if BO_type == 'Single':
            cost = 5
        elif M == 2:
            if func_name == 'SynFun_for_diffcost':
                cost = np.array([1, 1000])
            else:
                cost = np.array([1, 5])
        elif M == 3:
            cost = np.array([1, 3, 5])
        elif M == 4:
            cost = test_func.cost

    if BO_type == 'MultiFidelity':
        if func_name == 'Material':
            fidelity_features = cost.copy()
        elif 'cnn' in func_name:
            fidelity_features = test_func.fidelity_features
        else:
            fidelity_features = None

    if BO_type == 'MultiFidelity':
        if BO_method == 'BOCA':
            model_name = 'MTGP'
        elif func_name == 'Material':
            kernel_bounds = np.array([interval_size / 25., interval_size*4])
            model_name = 'MFGP'
        else:
            model_name = 'MFGP'
    else:
        model_name = 'GP'

    # set the seed
    np.random.seed(initial_seed)
    random.seed(initial_seed)

    if func_name == 'Material':
        ITR_MAX = 200
        COST_MAX = 1200
    elif 'cnn' in func_name:
        ITR_MAX = 1000
        COST_MAX = 1000
    elif 'SynFun' in func_name:
        ITR_MAX = 100*input_dim*cost[-1]
        COST_MAX = 100*input_dim*cost[-1]
    else:
        ITR_MAX = 50 * input_dim
        COST_MAX = 50 * input_dim

    # Latin Hypercube Sampling
    if BO_type == 'Single':
        if 'cnn' in func_name:
            FIRST_NUM = test_func.first_num_forBO
        else:
            FIRST_NUM = 5 * input_dim
        if X_all is None:
            training_input = myutils.initial_design(FIRST_NUM, input_dim, bounds)
            training_output = test_func.values(training_input)
        else:
            training_input, pool_X = myutils.initial_design_pool(FIRST_NUM, input_dim, bounds, pool_X)
            training_output = test_func.values(training_input)
        eval_num = FIRST_NUM
        if 'cnn' in func_name:
            first_cost = np.sum(np.array(test_func.costs(training_input)))
    else:
        if func_name == 'Material':
            FIRST_NUM = [10*input_dim, 7*input_dim, 3*input_dim]
        elif 'cnn' in func_name:
            FIRST_NUM = test_func.first_num
        elif M == 2:
            FIRST_NUM = [5*input_dim, 4*input_dim]
        elif M == 3:
            FIRST_NUM = [6*input_dim, 3*input_dim, 2*input_dim]
        elif M >= 4:
            FIRST_NUM = [5*input_dim, 5*input_dim, 5*input_dim, 5*input_dim]

        if X_all is None:
            training_input = myutils.initial_design(FIRST_NUM, input_dim, bounds)
            training_output = test_func.mf_values(training_input)
        else:
            training_input, pool_X = myutils.initial_design_pool(FIRST_NUM, input_dim, bounds, pool_X)
            training_output = test_func.mf_values(training_input)
        eval_num = FIRST_NUM.copy()
        if 'cnn' in func_name:
            first_cost = np.sum(np.array(test_func.costs(training_input[-1])))

    if 'cnn' in func_name:
        current_cost = first_cost
    else:
        current_cost = eval_num[-1] * cost[-1]
    selected_inputs = []
    remain_cost = None
    gp_regressor = None
    # if test function is synthetic, add parameter ell information
    if 'SynFun' in func_name:
        func_name = func_name+'_ell='+str(test_func.ell)+'-d='+str(test_func.d)+'-seed'+str(test_func.seed)
        optimize=False

        if model_name == 'MFGP':
            gp_regressor = myutils.set_mfgpy_regressor(None, training_input, training_output, eval_num, kernel_bounds, M, noise_var=1e-6, optimize=False, normalizer=False)
            gp_regressor['.*mul.rbf.variance'].constrain_fixed(1)
            gp_regressor['.*mul_1.rbf.variance'].constrain_fixed(1)
            gp_regressor['.*mul.rbf.lengthscale'].constrain_fixed(test_func.ell)
            gp_regressor['.*mul_1.rbf.lengthscale'].constrain_fixed(test_func.ell)
            gp_regressor['.*mul.coregion.W'].constrain_fixed(test_func.w_1)
            gp_regressor['.*mul.coregion.kappa'].constrain_fixed(test_func.kappa_1)
            gp_regressor['.*mul_1.coregion.W'].constrain_fixed(test_func.w_2)
            gp_regressor['.*mul_1.coregion.kappa'].constrain_fixed(test_func.kappa_2)
        elif model_name == 'GP':
            gp_regressor = myutils.set_gpy_regressor(None, training_input, training_output, kernel_bounds, noise_var=1e-6, optimize=False, normalizer=False)
            gp_regressor['.*rbf.variance'].constrain_fixed(1)
            gp_regressor['.*rbf.lengthscale'].constrain_fixed(test_func.ell)
        elif model_name == 'MTGP':
            gp_regressor = myutils.set_mtgpy_regressor(None, training_input, training_output, eval_num, np.c_[kernel_bounds, np.c_[[2., (M-1)*10]]], M, np.c_[np.arange(M)], 1, noise_var=1e-6, optimize=False, normalizer=False)
            gp_regressor['.*mul.rbf.variance'].constrain_fixed(1)
            gp_regressor['.*mul.rbf_1.variance'].constrain_fixed(1)
            gp_regressor['.*mul.rbf.lengthscale'].constrain_fixed(test_func.ell)
            # ell value of function kernel of MTmodel
            gp_regressor['.*mul.rbf_1.lengthscale'].constrain_fixed(2.178442285330266)
    else:
        optimize=True


    if MAX_NUM_WORKER == 1:
        if BO_type == 'Single':
            results_path = func_name+'_results/'+BO_method+'/seed='+str(initial_seed)+'/'
        else:
            results_path = func_name+'_results/'+BO_method+'/'+model_name+'_seed='+str(initial_seed)+'/'
    elif MAX_NUM_WORKER > 1:
        if BO_type == 'Single':
            results_path = func_name+'_results/'+BO_method+'_Q='+str(MAX_NUM_WORKER)+'/seed='+str(initial_seed)+'/'
        else:
            results_path = func_name+'_results/'+BO_method+'_Q='+str(MAX_NUM_WORKER)+'/'+model_name+'_seed='+str(initial_seed)+'/'

    if '1' in BO_method:
        NUM_SAMPLING = 1
        BO_method = BO_method.replace('1', '')
    elif '50' in BO_method:
        NUM_SAMPLING = 50
        BO_method = BO_method.replace('50', '')
    else:
        NUM_SAMPLING = 10


    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(results_path+'optimizer_log/'):
        os.makedirs(results_path+'optimizer_log/')
    cost_list = list()
    InfReg_list = list()
    SimReg_list = list()
    model_computation_time_list = list()
    preprocessing_time_list = list()
    optimize_acquisition_time_list = list()
    remain_cost = np.array([])
    data_num = np.sum(eval_num)
    diff_data_num = 0

    start = time.time()
    # bayesian optimizer
    if 'TAMFMES_RFM' == BO_method:
        optimizer = TAMFBO.TA_MultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, sampling_method='RFM', GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif 'MFMES_RFM' == BO_method:
        optimizer = TAMFBO.MultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, sampling_method='RFM', GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif 'TA_KG' == BO_method:
        optimizer = TAMFBO.TA_KG(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    else:
        print('Coresponding BO method is not implemented')
        exit()


    tmp_time = time.time() - start
    model_computation_time_list.append(tmp_time - optimizer.preprocessing_time)
    preprocessing_time_list.append(0)
    optimize_acquisition_time_list.append(0)


    for i in range(ITR_MAX):
        print('-------------------------------------')
        print(str(i)+'th iteration')
        print('-------------------------------------')
        gp_regressor = optimizer.GPmodel
        if (diff_data_num >= 5) or i == 0:
            if model_name=='MFGP':
                print(gp_regressor.mean, gp_regressor.std)
                print(gp_regressor['.*Gaussian_noise.variance'])
                print(gp_regressor['.*mul.rbf.variance'])
                print(gp_regressor['.*mul.rbf.lengthscale'])
                print(gp_regressor['.*mul.coregion.W'])
                print(gp_regressor['.*mul.coregion.kappa'])
                print(gp_regressor['.*mul_1.rbf.variance'])
                print(gp_regressor['.*mul_1.rbf.lengthscale'])
                print(gp_regressor['.*mul_1.coregion.W'])
                print(gp_regressor['.*mul_1.coregion.kappa'])
            elif model_name=='MTGP':
                print(gp_regressor.mean, gp_regressor.std)
                print(gp_regressor['.*Gaussian_noise.variance'])
                print(gp_regressor['.*mul.rbf.variance'])
                print(gp_regressor['.*mul.rbf.lengthscale'])
                print(gp_regressor['.*mul.rbf_1.variance'])
                print(gp_regressor['.*mul.rbf_1.lengthscale'])
            else:
                print(gp_regressor.mean, gp_regressor.std)
                print(gp_regressor['.*Gaussian_noise.variance'])
                print(gp_regressor['.*rbf.variance'])
                print(gp_regressor['.*rbf.lengthscale'])


        if X_all is None:
            inference_point, _ = optimizer.posteriori_maximum()
        else:
            if BO_type == 'Single':
                mean, _ = gp_regressor.predict(X_all)
                inference_point = X_all[np.argmax(mean.ravel())]
            else:
                if model_name == 'MTGP':
                    mean, _ = gp_regressor.predict(np.c_[X_all, np.matlib.repmat(fidelity_features[-1], np.shape(X_all)[0], 1)])
                    inference_point = X_all[np.argmax(mean.ravel())]
                elif model_name == 'MFGP':
                    mean, _ = gp_regressor.predict(np.c_[X_all, np.matlib.repmat([M-1], np.shape(X_all)[0], 1)])
                    inference_point = X_all[np.argmax(mean.ravel())]

        cost_list.append(current_cost)
        InfReg_list.append(test_func.values(np.atleast_2d(inference_point)))
        if BO_type == 'Single':
            SimReg_list.append(np.max(training_output))
        else:
            SimReg_list.append(np.max(training_output[M-1]))


        with open(results_path + 'cost.pickle', 'wb') as f:
            pickle.dump(np.array(cost_list), f)

        with open(results_path + 'InfReg.pickle', 'wb') as f:
            pickle.dump(np.array(InfReg_list), f)

        with open(results_path + 'SimReg.pickle', 'wb') as f:
            pickle.dump(np.array(SimReg_list), f)

        with open(results_path + 'EvalNum.pickle', 'wb') as f:
            pickle.dump(np.array(eval_num), f)

        with open(results_path + 'model_computation_time.pickle', 'wb') as f:
            pickle.dump(np.array(model_computation_time_list), f)

        with open(results_path + 'preprocessing_time.pickle', 'wb') as f:
            pickle.dump(np.array(preprocessing_time_list), f)

        with open(results_path + 'optimize_acquisition_time.pickle', 'wb') as f:
            pickle.dump(np.array(optimize_acquisition_time_list), f)

        print('Cost, eval_num, InfMax, SimMax :', cost_list[-1], eval_num, InfReg_list[-1], SimReg_list[-1])
        print('Average model computation time:', np.mean(model_computation_time_list))
        print('Average preprocessing time:', np.mean(preprocessing_time_list))
        print('Average acquisition function maximization time:', np.mean(optimize_acquisition_time_list))

        if (i % 10) == 0:
            with open(results_path + 'optimizer_log/' + 'optimizer'+str(int(i))+'.pickle', 'wb') as f:
                pickle.dump(optimizer, f)

        if cost_list[-1] >= COST_MAX:
            # print(cost_list[-1], COST_MAX)
            with open(results_path + 'optimizer_log/' + 'optimizer'+str(int(i))+'.pickle', 'wb') as f:
                pickle.dump(optimizer, f)
            break

        if X_all is not None:
            if BO_type == 'Single':
                print('remained X shape:', np.shape(pool_X))
            else:
                print('remained X shape:', [np.shape(X) for X in pool_X])

        preprocessing_time_list.append(optimizer.preprocessing_time)
        # add new input
        if BO_type == 'Single':
            start = time.time()
            if X_all is None:
                new_inputs = optimizer.next_input()
            else:
                new_inputs, pool_X = optimizer.next_input_pool(pool_X)
            tmp_time = time.time() - start
            optimize_acquisition_time_list.append(tmp_time)

            new_output = test_func.values(new_inputs)
            eval_num += MAX_NUM_WORKER
            iter_cost = cost
            current_cost += iter_cost

            training_input = np.r_[training_input, np.atleast_2d(new_inputs)]
            training_output = np.r_[training_output, np.atleast_2d(new_output)]
            print("new_inputs : ", new_inputs)
            print("its prediction:", optimizer.GPmodel.predict_noiseless(new_inputs))
            print("new_output : ", new_output)

            diff_data_num = eval_num - data_num
            start = time.time()
            if diff_data_num >= 5 and optimize:
                optimizer.update(np.atleast_2d(new_inputs), np.atleast_2d(new_output), optimize=True)
                data_num = eval_num
            else:
                optimizer.update(np.atleast_2d(new_inputs), np.atleast_2d(new_output), optimize=False)
            tmp_time = time.time() - start
            model_computation_time_list.append(tmp_time - optimizer.preprocessing_time)

        else:
            # if all evaluations are finished, new x is selected
            if np.size(selected_inputs) == 0:
                start = time.time()
                if X_all is None:
                    selected_inputs, remain_cost = optimizer.next_input()
                else:
                    selected_inputs, remain_cost, pool_X = optimizer.next_input_pool(pool_X)

                if 'cnn' in func_name:
                    remain_cost = np.array([
                            test_func.costs(selected_inputs[selected_inputs[:,-1] == m, :-1], fidelity=int(selected_inputs[selected_inputs[:,-1] == m, -1])).ravel()[0]
                            for m in range(M)
                            if np.size(selected_inputs[selected_inputs[:,-1] == m, :-1]) > 0
                            ])

                    if np.min(selected_inputs[:,-1]) > 0:
                        remain_cost -= test_func.costs(selected_inputs[0,:-1], fidelity = int(np.min(selected_inputs[:,-1]) - 1)).ravel()[0]
                tmp_time = time.time() - start
                optimize_acquisition_time_list.append(tmp_time)
            else:
                optimize_acquisition_time_list.append(0)

            print(selected_inputs)
            print(remain_cost)
            if MAX_NUM_WORKER > 1:
                '''
                Parallelization is not implemented
                '''
                exit()
            elif MAX_NUM_WORKER == 1:
                new_inputs = selected_inputs[remain_cost <= np.min(remain_cost)]
                selected_inputs = selected_inputs[remain_cost > np.min(remain_cost)]

                current_cost += np.min(remain_cost)
                if np.any(remain_cost > np.min(remain_cost)):
                    remain_cost = remain_cost[remain_cost > np.min(remain_cost)] - np.min(remain_cost)

            new_input_list = [new_inputs[new_inputs[:,-1] == m, :-1] for m in range(M)]
            new_output_list = test_func.mf_values(new_input_list)
            eval_num += np.array([np.size(output) for output in new_output_list])

            training_input = [np.r_[training_input[m], new_input_list[m]] if np.size(new_input_list[m]) > 0 else training_input[m] for m in range(M)]
            training_output = [np.r_[training_output[m], new_output_list[m]] if np.size(new_output_list[m]) > 0 else training_output[m] for m in range(M)]
            print("new_input :", new_inputs)
            print("its prediction:", optimizer.GPmodel.predict_noiseless(np.atleast_2d(new_inputs)))
            print("new_output :", new_output_list)

            diff_data_num = np.sum(eval_num) - data_num
            preparing_flag = np.size(selected_inputs) == 0
            start = time.time()
            if diff_data_num >= 5 and optimize:
                optimizer.update(new_input_list, new_output_list, optimize=True, prepare_new_acq=preparing_flag)
                data_num = np.sum(eval_num)
            elif not optimize:
                optimizer.update(new_input_list, new_output_list, optimize=False, prepare_new_acq=preparing_flag)
                data_num = np.sum(eval_num)
            else:
                optimizer.update(new_input_list, new_output_list, optimize=False, prepare_new_acq=preparing_flag)
            tmp_time = time.time() - start
            model_computation_time_list.append(tmp_time - optimizer.preprocessing_time)





if __name__ == '__main__':
    args = sys.argv
    BO_method = args[1]
    test_func = args[2]
    initial_seed = np.int(args[3])
    function_seed = np.int(args[4])
    parallel_num = np.int(args[5])
    options = [option for option in args if option.startswith('-')]

    test_funcs = ['SynFun_for_ta', 'cnn_mnist', 'cnn_cifar10']
    BO_methods = ['TAMFMES_RFM', 'MFMES_RFM', 'TA_KG']

    if BO_method in BO_methods and parallel_num > 1:
        print('BO method is sequential, but parallel num is larger than 1')
        exit()

    parallel_methods = []
    BO_methods.extend(parallel_methods)

    if BO_method in parallel_methods and parallel_num <= 1:
        print('BO method for parallel querying, but parallel num is smaller than 2')
        exit()


    if not(test_func in test_funcs):
        print(test_func + ' is not implemented!')
        exit(1)
    if not(BO_method in BO_methods):
        print(BO_method + ' is not implemented!')
        exit(1)


    os.environ["OMP_NUM_THREADS"] = "1"
    NUM_WORKER = 10

    # When seed = -1, experiments of seed of 0-9 is done for parallel
    # When other seed is set, experiments of the seed is done
    if function_seed >= 0:
        if initial_seed >= 0:
            main((test_func, BO_method, initial_seed, function_seed, parallel_num))
            exit()

        function_seeds = [function_seed]
        initial_seeds = np.arange(10).tolist()
    else:
        function_seeds = np.arange(10).tolist()
        if initial_seed < 0:
            initial_seeds = np.arange(10).tolist()
        else:
            initial_seeds = [initial_seed]

    params = list()
    for f_seed in function_seeds:
        for i_seed in initial_seeds:
            params.append((test_func, BO_method, i_seed, f_seed, parallel_num))
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        results = executor.map(main, params)
