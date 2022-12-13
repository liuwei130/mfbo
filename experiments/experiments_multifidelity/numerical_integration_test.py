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
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import GPy
from sklearn.linear_model import LinearRegression

# add my modules
from myBO.scripts.MultiFidelityBO import multifidelity_bayesian_opt as MFBO
from myBO.scripts.MultiFidelityBO import parallel_multifidelity_bayesian_opt as PMFBO

from myBO.scripts.vanillaBO import bayesian_opt as BO
from myBO.scripts.vanillaBO import parallel_bayesian_opt as PBO

from myBO.scripts.test_functions import test_functions
from myBO.scripts.myutils import myutils

signal.signal(signal.SIGINT, signal.SIG_DFL)


plt.rcParams["font.size"] = 15 #
# plt.rcParams["axes.titlesize"]=20
# plt.rcParams['xtick.labelsize'] = 20 #
# plt.rcParams['ytick.labelsize'] = 20 #
# plt.rcParams['legend.fontsize'] = 20

plt.rcParams['figure.figsize'] = (6., 4.)
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['errorbar.capsize'] = 4.0

plt.rcParams['lines.linewidth'] = 2.
plt.rcParams['lines.markeredgewidth'] = 1.5

plt.rcParams['legend.borderaxespad'] = 0.1
plt.rcParams['legend.borderpad'] = 0.1
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.5

def integrand_MC(params):
    i = params[0]
    mu = params[1]
    cov = params[2]
    f_star = params[3]
    base_sample_num = 10**7
    np.random.seed(i)

    f =  np.c_[np.random.randn(base_sample_num) * np.sqrt(cov[0,0]) + mu[0]]

    Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
    temp = cov[1, 0] / cov[0,0]
    conditional_mean = mu[1] + temp * (f - mu[0])
    del f

    conditional_var = cov[1,1] - temp*cov[0,1]
    conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )
    del conditional_mean

    conditional_cdf[conditional_cdf<=0] = 1
    first_term = conditional_cdf / Z
    integrands = np.sum(first_term * np.log(conditional_cdf)) / (10 * base_sample_num )

    del first_term
    del conditional_cdf
    return integrands


def integration_test(f_star, mu, cov, func_name, initial_seed):
    func_evaluate_num_list = np.linspace(10, 100, 19).astype(np.int).tolist()
    # func_evaluate_num_list.append(500)
    print(func_evaluate_num_list)
    f_star = np.atleast_2d(f_star)

    cons = 6
    # cdf(-6) \approx 10^-9
    rho = cov[0, 1] / np.sqrt(cov[0,0] * cov[1, 1])


    first_entropy = 1/2. + 1/2.*np.log(2*np.pi*cov[0,0])
    # print('low first entropy :', first_entropy)

    gamma = (f_star - mu[1]) / np.sqrt(cov[1, 1])
    print('high acquisition :', np.mean(gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))))
    MES = gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))

    legendre_before = list()
    legendre_before_interval = list()
    legendre_analytical = list()
    legendre_analytical_wointerval = list()
    hermite_analytical = list()
    hermite = list()

    cdf_central = mu[0] + cov[0,0] / cov[1,0] * (f_star - mu[1])
    cdf_width = np.abs(cons*cov[0,0] / cov[1,0] * np.sqrt(cov[1,1] - cov[0,1]**2 / cov[0,0]))
    pdf_central = mu[0]
    pdf_width = cons * np.sqrt(cov[0,0])

    # points_min = np.max([cdf_central-cdf_width, pdf_central-pdf_width])
    # points_max = np.min([cdf_central+cdf_width, pdf_central+pdf_width])
    logsumexp_const = 1e3
    points_min = np.logaddexp(logsumexp_const*(cdf_central-cdf_width), logsumexp_const*(pdf_central-pdf_width)) / logsumexp_const
    points_max = - np.logaddexp(-logsumexp_const*(cdf_central+cdf_width), -logsumexp_const*(pdf_central+pdf_width)) / logsumexp_const


    points_min_before = pdf_central-pdf_width
    # points_max_before = np.copy(points_max)
    points_max_before = pdf_central+pdf_width

    # print(points_min, points_max)
    # print(points_min_before, points_max_before)


    '''
    MC approximation start
    '''
    # def integrand_MC(f):
    #     Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
    #     temp = cov[1, 0] / cov[0,0]
    #     conditional_mean = mu[1] + temp * (f - mu[0])
    #     conditional_var = cov[1,1] - temp*cov[0,1]
    #     conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )

    #     conditional_cdf[conditional_cdf<=0] = 1
    #     first_term = conditional_cdf / Z
    #     return first_term * np.log(conditional_cdf)

    # base_sample_num = 10**8
    # MC_samples = np.random.randn(base_sample_num) * np.sqrt(cov[0,0]) + mu[0]
    # integrands = integrand_MC(np.c_[MC_samples])
    # MC_approx = np.mean(rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))) + np.mean(integrands)

    # with open('integral_plots/'+func_name+'_integration_MC.pickle', 'rb') as f:
    #     MC_approx = pickle.load(f)


    print('start parallel processing')
    NUM_WORKER=10
    results = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        for i in range(10):
            future = executor.submit(integrand_MC, (i, mu, cov, f_star))
            results.append(future)
    integrands = np.array([x.result() for x in results])
    print(integrands)
    MC_approx = np.mean(rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))) + np.sum(integrands) / 10
    # with open('acquisition_approx_MC_rho='+str(rho)+'.pickle', 'wb') as f:
    #     pickle.dump(MC_approx, f)
    print('end parallel processing')
    '''
    MC approximation end
    '''
    print('end MC estimation')
    # with open('integral_plots/'+func_name+'_integration_MC.pickle', 'rb') as f:
    #     MC_approx = pickle.load(f)


    for func_evaluate_num  in func_evaluate_num_list:
        def integrand_before(f):
            pdf = norm.pdf(f, loc=mu[0], scale=np.sqrt(cov[0,0]))
            Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
            temp = cov[1, 0] / cov[0,0]
            conditional_mean = mu[1] + temp * (f - mu[0])
            conditional_var = cov[1,1] - temp*cov[0,1]
            conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )
            conditional_cdf[conditional_cdf<=0] = 1

            ESG_pdf = conditional_cdf * pdf / Z
            return - ESG_pdf * np.log(ESG_pdf)

        def integrand_analytical(f):
            pdf = norm.pdf(f, loc=mu[0], scale=np.sqrt(cov[0,0]))
            Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
            temp = cov[1, 0] / cov[0,0]
            conditional_mean = mu[1] + temp * (f - mu[0])
            conditional_var = cov[1,1] - temp*cov[0,1]
            conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )

            conditional_cdf[conditional_cdf<=0] = 1
            ESG_pdf = conditional_cdf * pdf / Z
            return ESG_pdf * np.log(conditional_cdf)
        '''
        Gauss-Legendre
        '''
        points, weights = np.polynomial.legendre.leggauss(func_evaluate_num)

        legendre_approx_entropy = first_entropy - np.mean(np.sum(np.c_[weights] * (points_max_before-points_min_before)/2. * integrand_before( (points_max_before+points_min_before)/2. + (points_max_before-points_min_before)/2.*np.c_[points]), axis=0))
        legendre_before.append(legendre_approx_entropy)

        # legendre_approx_entropy = first_entropy - np.sum(np.c_[weights] * integrand_before( (points_max+points_min_before)/2. + (points_max-points_min_before)/2.*np.c_[points])) * (points_max-points_min_before)/2.
        # legendre_before_interval.append(legendre_approx_entropy)


        legendre_approx_entropy = np.mean(rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.atleast_2d(np.sum(np.c_[weights] * (points_max-points_min)/2. * integrand_analytical( (points_max+points_min)/2. + (points_max-points_min)/2.*np.c_[points]), axis=0)))
        legendre_analytical.append(legendre_approx_entropy)


        legendre_approx_entropy = np.mean(rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.sum(np.c_[weights] * (points_max_before-points_min_before)/2. * integrand_analytical( (points_max_before+points_min_before)/2. + (points_max_before-points_min_before)/2.*np.c_[points]), axis=0))
        legendre_analytical_wointerval.append(legendre_approx_entropy)

        '''
        Gauss-Hermite
        '''

        def hermite_integrand_analytical(f):
            # pdf = norm.pdf(f, loc=mu[0], scale=np.sqrt(cov[0,0]))
            Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
            temp = cov[1, 0] / cov[0,0]
            conditional_mean = mu[1] + temp * (f - mu[0])
            conditional_var = cov[1,1] - temp*cov[0,1]
            conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )
            conditional_cdf[conditional_cdf <= 0] = 1

            return conditional_cdf / Z * np.log(conditional_cdf)

        points, weights = np.polynomial.hermite.hermgauss(func_evaluate_num)
        hermite_approx_entropy = np.mean(rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.sum(np.c_[weights] * hermite_integrand_analytical( mu[0] + np.sqrt(2*cov[0,0])*np.c_[points]), axis=0) / np.sqrt(np.pi))
        hermite_analytical.append(hermite_approx_entropy)

        def hermite_integrand(f):
            pdf = norm.pdf(f, loc=mu[0], scale=np.sqrt(cov[0,0]))
            Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
            temp = cov[1, 0] / cov[0,0]
            conditional_mean = mu[1] + temp * (f - mu[0])
            conditional_var = cov[1,1] - temp*cov[0,1]
            conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )
            conditional_cdf[conditional_cdf <= 0] = 1
            return conditional_cdf / Z * (np.log(conditional_cdf) + np.log(pdf) - np.log(Z) )

        hermite_approx_entropy = first_entropy + np.mean( np.sum(np.c_[weights] * hermite_integrand( mu[0] + np.sqrt(2*cov[0,0])*np.c_[points]), axis=0) / np.sqrt(np.pi))
        hermite.append(hermite_approx_entropy)




    ground_truth = MC_approx
    # plt.plot(func_evaluate_num_list, np.abs(legendre_before_interval - ground_truth), label=r'naive with $[l_{\rm pdf}, u]$')
    plt.plot(func_evaluate_num_list, np.abs(legendre_before - ground_truth).ravel(), label=r'naive with $[l_{\rm pdf}, u_{\rm pdf}]$')
    plt.plot(func_evaluate_num_list, np.abs(legendre_analytical_wointerval - ground_truth).ravel(), label=r'(12) with $[l_{\rm pdf}, u_{\rm pdf}]$')
    plt.plot(func_evaluate_num_list, np.abs(hermite - ground_truth).ravel(), label=r'naive with G-H quad.')
    plt.plot(func_evaluate_num_list, np.abs(hermite_analytical - ground_truth).ravel(), label=r'(12) with G-H quad.')
    plt.plot(func_evaluate_num_list, np.abs(legendre_analytical - ground_truth).ravel(), label=r'(12) with $[l, u]$ (proposed)')


    # plt.plot(func_evaluate_num_list, legendre_before, label=r'naive with $[l_{\rm pdf}, u_{\rm pdf}]$')
    # plt.plot(func_evaluate_num_list, legendre_analytical_wointerval, label=r'(12) with $[l_{\rm pdf}, u_{\rm pdf}]$')
    # plt.plot(func_evaluate_num_list, hermite, label=r'naive with G-H quad.')
    # plt.plot(func_evaluate_num_list, hermite_analytical, label=r'(12) with G-H quad.')
    # plt.plot(func_evaluate_num_list, legendre_analytical, label=r'(12) with $[l, u]$ (proposed)')

    with open('integral_plots/'+func_name+'_integration_naive.pickle', 'wb') as f:
        pickle.dump(legendre_before, f)
    with open('integral_plots/'+func_name+'_integration_expansion.pickle', 'wb') as f:
        pickle.dump(legendre_analytical_wointerval, f)
    with open('integral_plots/'+func_name+'_integration_proposed.pickle', 'wb') as f:
        pickle.dump(legendre_analytical, f)
    with open('integral_plots/'+func_name+'_integration_hermite.pickle', 'wb') as f:
        pickle.dump(hermite_analytical, f)
    with open('integral_plots/'+func_name+'_integration_hermite_naive.pickle', 'wb') as f:
        pickle.dump(hermite, f)
    with open('integral_plots/'+func_name+'_integration_MC.pickle', 'wb') as f:
        pickle.dump(MC_approx, f)

    plt.yscale('log')
    plt.xlabel(r'\# integrand evaluations')
    plt.ylabel(r'Diff. to MC')

    # if rho==0.9:
    plt.title(func_name[:8]+':MC = {:.8f}'.format(MC_approx))
    plt.legend()
    plt.savefig('integral_plots/check_accuracy'+func_name+'_seed='+str(initial_seed)+'.pdf')
    plt.close()
    # plt.show()


def main(params):
    (func_name, BO_method, initial_seed, function_seed, MAX_NUM_WORKER) = params
    print(params)
    if not ((BO_method == 'BOCA') or ('MF' in BO_method) or (BO_method == 'GIBBON')):
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
    elif func_name == 'SVM':
        test_func = eval('test_functions.'+func_name)()
        M = test_func.M
        bounds = test_func.bounds
        input_dim = test_func.d
        interval_size = bounds[1] - bounds[0]
        kernel_bounds = np.c_[np.array([interval_size / 1e1, interval_size / 2.]), np.c_[[1e1, 1e3]]]
        # kernel_bounds = np.c_[np.array([interval_size / 1e2, interval_size * 1e2]), np.c_[[1e1, 1e3]]]
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
        X_all = None


    # model setting
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
    elif func_name == 'SVM':
        ITR_MAX = 500
        COST_MAX = 300
    elif 'SynFun' in func_name:
        ITR_MAX = 150*input_dim
        COST_MAX = 150*input_dim
    else:
        ITR_MAX = 50 * input_dim
        COST_MAX = 50 * input_dim

    # Latin Hypercube Sampling
    if BO_type == 'Single':
        if func_name=='SVM':
            FIRST_NUM = test_func.first_num_forBO
        else:
            FIRST_NUM = 5 * input_dim

        if X_all is None:
            training_input = myutils.initial_design(FIRST_NUM, input_dim, bounds)
            training_output = test_func.values(training_input)
        else:
            training_input, pool_X = myutils.initial_design_pool(FIRST_NUM, input_dim, bounds, pool_X)
            training_output = test_func.values(training_input)
            if 'SVM' in func_name:
                training_costs = test_func.costs(training_input)
        eval_num = FIRST_NUM
    else:
        if func_name == 'Material':
            FIRST_NUM = [10*input_dim, 7*input_dim, 3*input_dim]
        elif func_name == 'SVM':
            FIRST_NUM = test_func.first_num
        elif M == 2:
            FIRST_NUM = [5*input_dim, 4*input_dim]
        elif M == 3:
            FIRST_NUM = [6*input_dim, 3*input_dim, 2*input_dim]

        if X_all is None:
            training_input = myutils.initial_design(FIRST_NUM, input_dim, bounds)
            training_output = test_func.mf_values(training_input)
        else:
            training_input, pool_X = myutils.initial_design_pool(FIRST_NUM, input_dim, bounds, pool_X)
            training_output = test_func.mf_values(training_input)
            if 'SVM' in func_name:
                training_costs = test_func.mf_costs(training_input)
        eval_num = FIRST_NUM.copy()

    # cost setting
    if func_name == 'Material':
        if BO_type == 'Single':
            cost = 60
        else:
            cost = np.array([5, 10, 60])
    elif func_name == 'SVM':
        # minute
        if BO_type == 'Single':
            cost = 1
        else:
            cost = np.array(test_func.data_num_list)**2 / np.max(test_func.data_num_list)**2
            # cost = test_func.fidelity_features  / 60.
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

    if BO_type == 'MultiFidelity':
        if func_name == 'Material':
            fidelity_features = cost.copy()
        elif func_name == 'SVM':
            fidelity_features = cost.copy()
        else:
            fidelity_features = None


    if 'SVM' in func_name:
        current_cost = np.sum(np.vstack(training_costs))
    else:
        current_cost = np.sum(eval_num * cost)

    selected_inputs = None
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
            gp_regressor['.*mul.coregion.W'].constrain_fixed(np.sqrt(0.8))
            gp_regressor['.*mul.coregion.kappa'].constrain_fixed(0.05)
            gp_regressor['.*mul_1.coregion.W'].constrain_fixed(np.sqrt(0.1))
            gp_regressor['.*mul_1.coregion.kappa'].constrain_fixed(0.05)
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
    if BO_method == 'MFSKO':
        optimizer = MFBO.MultiFidelitySuquentialKrigingOptimization(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif 'GIBBON' == BO_method:
        optimizer = MFBO.GIBBON(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, sampling_method='Gumbel', GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif 'MFPES' in BO_method:
        optimizer = MFBO.MultifidelityPredictiveEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif BO_method == 'BOCA':
        optimizer = MFBO.MultiFidelityBayesianOptimizationWithContinuousApproximations(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, iteration=1, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, optimize=optimize)
    # elif 'MFMES_Gumbel' in BO_method:
    #     optimizer = MFBO.MultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif 'MFMES_RFM' == BO_method:
        optimizer = MFBO.MultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, sampling_num=NUM_SAMPLING, sampling_method='RFM', GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    # elif 'MES_Gumbel' in BO_method:
    #     optimizer = BO.MaxvalueEntropySearch(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, sampling_num=NUM_SAMPLING, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
    elif 'MES_RFM' == BO_method:
        optimizer = BO.MaxvalueEntropySearch(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, sampling_num=NUM_SAMPLING, GPmodel=gp_regressor, sampling_method='RFM', pool_X=X_all, optimize=optimize)
    elif BO_method == 'Sync_MFMES_RFM':
        optimizer = PMFBO.SyncMultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif BO_method == 'Parallel_MFMES_RFM':
        optimizer = PMFBO.ParallelMultiFidelityMaxvalueEntropySearch(X_list=training_input, Y_list=training_output, eval_num=eval_num, bounds = bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, model_name=model_name, fidelity_features=fidelity_features, pool_X=X_all, optimize=optimize)
    elif BO_method == 'Parallel_MES_RFM' or BO_method == 'Sync_MES_RFM':
        optimizer = PBO.ParallelMaxvalueEntropySearch(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
    elif BO_method == 'Batch_MES_RFM':
        optimizer = PBO.BatchMaxvalueEntropySearch(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
    elif BO_method == 'MES_LP':
        optimizer = PBO.MaxvalueEntropySearch_LP(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
    elif BO_method == 'GP_UCB_PE':
        optimizer = PBO.GaussianProcessUCB_PE(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, iteration=1, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
    elif BO_method == 'AsyncTS':
        optimizer = PBO.AsynchronousThompsonSampling(X=training_input, Y=training_output, bounds = bounds, kernel_bounds=kernel_bounds, selected_inputs=selected_inputs, num_worker=MAX_NUM_WORKER, GPmodel=gp_regressor, pool_X=X_all, optimize=optimize)
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
        if ((diff_data_num >= 5) and optimize) or i == 0:
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
        # if BO_type == 'Single':
        #     start = time.time()
        #     if X_all is None:
        #         new_inputs = optimizer.next_input()
        #     else:
        #         new_inputs, pool_X = optimizer.next_input_pool(pool_X)
        #     tmp_time = time.time() - start
        #     optimize_acquisition_time_list.append(tmp_time)

        #     new_output = test_func.values(new_inputs)
        #     eval_num += MAX_NUM_WORKER
        #     if 'SVM' in func_name:
        #         if MAX_NUM_WORKER > 1:
        #             iter_cost = []
        #             for j in range(np.shape(new_inputs)[0]):
        #                 iter_cost.append(test_func.costs(new_inputs[i]).ravel()[0])
        #             iter_cost = np.max()
        #         else:
        #             iter_cost = test_func.costs(new_inputs)

        #         '''
        #         if 'Sync_MFMES' in BO_method:
        #             #
        #             iter_cost = np.max(remain_cost)
        #             current_cost += np.sum(remain_cost)
        #         else:
        #             iter_cost = np.min(remain_cost)
        #             current_cost += iter_cost
        #         '''
        #     else:
        #         iter_cost = cost
        #     current_cost += iter_cost

        #     training_input = np.r_[training_input, np.atleast_2d(new_inputs)]
        #     training_output = np.r_[training_output, np.atleast_2d(new_output)]
        #     print("new_inputs : ", new_inputs)
        #     print("its prediction:", optimizer.GPmodel.predict_noiseless(new_inputs))
        #     print("new_output : ", new_output)

        #     diff_data_num = eval_num - data_num
        #     start = time.time()
        #     if diff_data_num >= 5 and optimize:
        #         optimizer.update(np.atleast_2d(new_inputs), np.atleast_2d(new_output), optimize=True)
        #         data_num = eval_num
        #     else:
        #         optimizer.update(np.atleast_2d(new_inputs), np.atleast_2d(new_output), optimize=False)
        #     tmp_time = time.time() - start
        #     model_computation_time_list.append(tmp_time - optimizer.preprocessing_time)
        # else:
        if BO_type == 'Single':
            start = time.time()
            if X_all is None:
                new_selected_inputs = optimizer.next_input()
            else:
                new_selected_inputs, pool_X = optimizer.next_input_pool(pool_X)

            new_selected_inputs = np.c_[new_selected_inputs, (M-1) * np.c_[np.ones(np.shape(new_selected_inputs)[0])]]
            tmp_time = time.time() - start
            optimize_acquisition_time_list.append(tmp_time)
        else:
            start = time.time()
            if X_all is None:
                new_selected_inputs = optimizer.next_input()
            else:
                new_selected_inputs, pool_X = optimizer.next_input_pool(pool_X)
            tmp_time = time.time() - start
            optimize_acquisition_time_list.append(tmp_time)

        if MAX_NUM_WORKER > 1:
            if selected_inputs is None:
                selected_inputs = new_selected_inputs
                if 'SVM' in func_name:
                    remain_cost = []
                    for j in range(np.shape(selected_inputs)[0]):
                        remain_cost.append(
                            test_func.costs(selected_inputs[j, :-1], fidelity=selected_inputs[j, -1].astype(np.int)).ravel()[0]
                        )
                    remain_cost = np.ravel(np.array(remain_cost))
                else:
                    remain_cost = cost[new_selected_inputs[:,-1].astype(np.int)]
            else:
                selected_inputs = np.r_[selected_inputs, new_selected_inputs]

                if 'SVM' in func_name:
                    remain_cost = remain_cost.tolist()
                    for j in range(np.shape(new_selected_inputs)[0]):
                        remain_cost.append(
                            test_func.costs(new_selected_inputs[j, :-1], fidelity=new_selected_inputs[j, -1].astype(np.int)).ravel()[0]
                        )
                    remain_cost = np.ravel(np.array(remain_cost))
                else:
                    remain_cost = np.r_[remain_cost, cost[new_selected_inputs[:,-1].astype(np.int)]]
            # print('selected_inputs:', selected_inputs)
            # print('remain_cost:', remain_cost)

            assert np.shape(selected_inputs)[0] <= MAX_NUM_WORKER, 'NUM_WORKER is lager than MAX_NUM_WORKER'

            if 'Sync' in BO_method:
                #
                iter_cost = np.max(remain_cost)
                current_cost += np.sum(remain_cost)
            else:
                iter_cost = np.min(remain_cost)
                current_cost += iter_cost

            # print(selected_inputs)
            # print(remain_cost)
            # print(iter_cost)

            new_inputs = selected_inputs[remain_cost <= iter_cost]
            selected_inputs = selected_inputs[remain_cost > iter_cost]
            if np.size(selected_inputs) == 0:
                selected_inputs = None
                remain_cost = None
            else:
                remain_cost = remain_cost[remain_cost > iter_cost] - iter_cost
        elif MAX_NUM_WORKER == 1:
            new_inputs = new_selected_inputs

            if 'SVM' in func_name:
                iter_cost = test_func.costs(new_selected_inputs[0, :-1], fidelity=new_selected_inputs[0, -1].astype(np.int)).ravel()[0]
            else:
                iter_cost = cost[new_selected_inputs[0, -1].astype(np.int)]
            current_cost += iter_cost


        if BO_type=='Single':
            new_input = new_inputs[new_inputs[:,-1] == (M-1), :-1]
            new_output = test_func.values(new_input)
            eval_num += np.size(new_output)

            training_input = np.r_[training_input, np.atleast_2d(new_input)]
            training_output = np.r_[training_output, np.atleast_2d(new_output)]
            print("new_inputs : ", new_inputs)
            print("its prediction:", optimizer.GPmodel.predict_noiseless(new_input))
            print("new_output : ", new_output)

            diff_data_num = eval_num - data_num
            start = time.time()
            if diff_data_num >= 5 and optimize:
                optimizer.update(np.atleast_2d(new_input), np.atleast_2d(new_output), optimize=True)
                data_num = eval_num
            else:
                optimizer.update(np.atleast_2d(new_input), np.atleast_2d(new_output), optimize=False)
            tmp_time = time.time() - start
            model_computation_time_list.append(tmp_time - optimizer.preprocessing_time)
        else:
            new_input_list = [new_inputs[new_inputs[:,-1] == m, :-1] for m in range(M)]
            new_output_list = test_func.mf_values(new_input_list)
            eval_num += np.array([np.size(output) for output in new_output_list])

            training_input = [np.r_[training_input[m], new_input_list[m]] if np.size(new_input_list[m]) > 0 else training_input[m] for m in range(M)]
            training_output = [np.r_[training_output[m], new_output_list[m]] if np.size(new_output_list[m]) > 0 else training_output[m] for m in range(M)]
            print("new_input :", new_inputs)
            print("its prediction:", optimizer.GPmodel.predict_noiseless(new_inputs))
            print("new_output :", new_output_list)


            tmp_inputs = np.vstack([new_inputs, new_inputs])
            tmp_inputs[1, -1] = M-1
            print(tmp_inputs)
            mu, cov = optimizer.GPmodel.predict_noiseless(tmp_inputs, full_cov=True)
            print("its prediction:", mu, cov)
            integration_test(optimizer.maximums, mu, cov, func_name, initial_seed)
            exit()

            diff_data_num = np.sum(eval_num) - data_num
            start = time.time()
            if diff_data_num >= 5 and optimize:
                optimizer.update(new_input_list, new_output_list, optimize=True)
                data_num = np.sum(eval_num)
            else:
                optimizer.update(new_input_list, new_output_list, optimize=False)

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

    if BO_method == 'PMES':
        BO_method = 'Parallel_MES_RFM'
    elif BO_method == 'PMFMES':
        BO_method = 'Parallel_MFMES_RFM'
    elif BO_method == 'SMFMES':
        BO_method = 'Sync_MFMES_RFM'
    elif BO_method == 'SMES':
        BO_method = 'Sync_MFMES_RFM'
    elif BO_method == 'BMES':
        BO_method = 'Batch_MES_RFM'

    test_funcs = ['SynFun', 'Material', 'Forrester', 'Beale', 'HartMann3', 'HartMann4', 'HartMann6', 'Borehole', 'Branin', 'Colville', 'CurrinExp', 'Styblinski_tang', 'Powell', 'Shekel', 'SVM']
    BO_methods = ['MFMES_Gumbel', 'MES_Gumbel', 'MFMES_RFM', 'MES_RFM', 'BOCA', 'MFSKO', 'MFPES']
    BO_methods.extend(['MFMES_Gumbel1', 'MES_Gumbel1', 'MFMES_RFM1', 'MES_RFM1', 'MFPES1',  'MFMES_Gumbel50', 'MES_Gumbel50', 'MFMES_RFM50', 'MES_RFM50', 'MFPES50', 'GIBBON'])

    if BO_method in BO_methods and parallel_num > 1:
        print('BO method is sequential, but parallel num is larger than 1')
        exit()

    parallel_methods = ['Batch_MES_RFM', 'Parallel_MES_RFM', 'Parallel_MFMES_RFM', 'Sync_MFMES_RFM', 'Sync_MES_RFM', 'MES_LP', 'GP_UCB_PE', 'AsyncTS']
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


