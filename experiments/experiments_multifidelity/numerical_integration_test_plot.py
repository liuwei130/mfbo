# -*- coding: utf-8 -*-


import time
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import concurrent.futures
import signal
import pickle

from scipy.stats import norm
from scipy.stats import mvn
import GPy
# plt.rcParams['pdf.fonttype'] = 42 # Type3font
# plt.rcParams['ps.fonttype'] = 42 # Type3font
plt.rcParams['font.family'] = 'sans-serif' # font family
matplotlib.rcParams['text.usetex'] = True # Type1 font
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # Type1 font

plt.rcParams["font.size"] = 20 #
plt.rcParams["axes.titlesize"]=21
plt.rcParams['xtick.labelsize'] = 20 #
plt.rcParams['ytick.labelsize'] = 20 #
plt.rcParams['legend.fontsize'] = 22

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

signal.signal(signal.SIGINT, signal.SIG_DFL)



def integrand_MC(params):
    i = params[0]
    rho = params[1]
    mu = np.c_[np.array([1, 1])]
    cov = 3*np.array([[1, 1*rho], [1*rho, 1]])
    f_star = 3
    base_sample_num = 10**8
    np.random.seed(i)
    f = np.random.randn(base_sample_num) * np.sqrt(cov[0,0]) + mu[0]

    Z = norm.cdf( (f_star - mu[1]) / np.sqrt(cov[1,1]) )
    temp = cov[1, 0] / cov[0,0]
    conditional_mean = mu[1] + temp * (f - mu[0])
    conditional_var = cov[1,1] - temp*cov[0,1]
    conditional_cdf = norm.cdf( (f_star - conditional_mean) / np.sqrt(conditional_var) )

    conditional_cdf[conditional_cdf<=0] = 1
    first_term = conditional_cdf / Z
    return first_term * np.log(conditional_cdf)



def integration_test():
    # func_evaluate_num_list = [10, 20, 30, 40, 50, 75, 100, 1000]
    func_evaluate_num_list = np.linspace(10, 60, 11).astype(np.int).tolist()

    f_star = 3.
    cons = 6
    # cdf(-6) \approx 10^-9
    rho_list = [0.9, 0.99, 0.999, 0.9999]
    for rho in rho_list:
        mu = np.c_[np.array([1, 1])]
        cov = 3*np.array([[1, 1*rho], [1*rho, 1]])
        # rho = cov[1, 0] / (np.sqrt(cov[0,0]*cov[1,1]))
        first_entropy = 1/2. + 1/2.*np.log(2*np.pi*cov[0,0])
        # print('low first entropy :', first_entropy)

        gamma = (f_star - mu[1]) / np.sqrt(cov[1, 1])
        print('high acquisition :', gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)))
        MES = gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))

        legendre_before = list()
        legendre_before_interval = list()
        legendre_analytical = list()
        legendre_analytical_wointerval = list()

        cdf_central = mu[0] + cov[0,0] / cov[1,0] * (f_star - mu[1])
        cdf_width = np.abs(cons*cov[0,0] / cov[1,0] * np.sqrt(cov[1,1] - cov[0,1]**2 / cov[0,0]))
        pdf_central = mu[0]
        pdf_width = cons * np.sqrt(cov[0,0])

        # points_min = np.max([cdf_central-cdf_width, pdf_central-pdf_width])
        # points_max = np.min([cdf_central+cdf_width, pdf_central+pdf_width])
        logsumexp_const = 1e3
        points_min = np.logaddexp(logsumexp_const*(cdf_central-cdf_width), logsumexp_const*(pdf_central-pdf_width)) / logsumexp_const
        points_max = - np.logaddexp(-logsumexp_const*(cdf_central+cdf_width), -logsumexp_const*(pdf_central+pdf_width)) / logsumexp_const
        print(points_min, points_max)

        points_min_before = pdf_central-pdf_width
        # points_max_before = np.copy(points_max)
        points_max_before = pdf_central+pdf_width


        '''
        MC approximation start
        '''
        # base_sample_num = 10**8
        # # MC_samples = np.random.randn(100 * base_sample_num) * np.sqrt(cov[0,0]) + mu[0]
        # NUM_WORKER=30
        # params=list()
        # for i in range(100):
        #     params.append((i, rho))

        # print('start parallel processing')
        # results = list()
        # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        #     for i in range(100):
        #         future = executor.submit(integrand_MC, (i, rho))
        #         results.append(future)
        # integrands = np.array([x.result() for x in results])
        # print(integrands)
        # MC_approx = rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.mean(integrands)
        # with open('acquisition_approx_MC_rho='+str(rho)+'.pickle', 'wb') as f:
        #     pickle.dump(MC_approx, f)
        # print('end parallel processing')
        '''
        MC approximation end
        '''
        with open('acquisition_approx_MC_rho='+str(rho)+'.pickle', 'rb') as f:
           MC_approx = pickle.load(f)


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

            legendre_approx_entropy = first_entropy - np.sum(np.c_[weights] * integrand_before( (points_max_before+points_min_before)/2. + (points_max_before-points_min_before)/2.*np.c_[points])) * (points_max_before-points_min_before)/2.
            legendre_before.append(legendre_approx_entropy[0])

            legendre_approx_entropy = first_entropy - np.sum(np.c_[weights] * integrand_before( (points_max+points_min_before)/2. + (points_max-points_min_before)/2.*np.c_[points])) * (points_max-points_min_before)/2.
            legendre_before_interval.append(legendre_approx_entropy[0])



            legendre_approx_entropy = rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.sum(np.c_[weights] * integrand_analytical( (points_max+points_min)/2. + (points_max-points_min)/2.*np.c_[points])) * (points_max-points_min)/2.
            legendre_analytical.append(legendre_approx_entropy[0])



            legendre_approx_entropy = rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma)) + np.sum(np.c_[weights] * integrand_analytical( (points_max_before+points_min_before)/2. + (points_max_before-points_min_before)/2.*np.c_[points])) * (points_max_before-points_min_before)/2.
            legendre_analytical_wointerval.append(legendre_approx_entropy[0])

            info_upper_bound = rho**2 * gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))



        print('legendre_before rule:', legendre_before)
        print('legendre_analytical rule:', legendre_analytical)

        plt.plot(func_evaluate_num_list, np.abs(legendre_before - MC_approx), label=r'naive with $[l_{\rm pdf}, u_{\rm pdf}]$')
        plt.plot(func_evaluate_num_list, np.abs(legendre_before_interval - MC_approx), label=r'naive with $[l_{\rm pdf}, u]$')
        plt.plot(func_evaluate_num_list, np.abs(legendre_analytical_wointerval - MC_approx), label=r'(12) with $[l_{\rm pdf}, u_{\rm pdf}]$')
        plt.plot(func_evaluate_num_list, np.abs(legendre_analytical - MC_approx), label=r'(12) with $[l, u]$ (proposed)')

        # y_width = 1.1*np.max(np.abs(legendre_analytical_wointerval - legendre_analytical_wointerval[-1]))
        # plt.ylim(legendre_analytical_wointerval[-1]-y_width, legendre_analytical_wointerval[-1]+y_width)
        # if rho >= 0.99:
        #     plt.hlines(MES, np.min(func_evaluate_num_list), np.max(func_evaluate_num_list), label=r'highest (MES)')
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'\# integrand evaluations')
        plt.ylabel(r'Difference to MC approximation')
        plt.title(r'$\rho^{(mM)}_{x}=$'+str(rho))
        if rho==0.9:
            plt.legend()
        plt.savefig('check_accuracy_rho='+str(rho)+'.pdf')
        plt.close()
        # plt.show()


def plot_result():
    func_evaluate_num_list = np.linspace(10, 100, 10).astype(np.int).tolist()
    func_evaluate_num_list = np.linspace(10, 60, 11).astype(np.int).tolist()
    func_evaluate_num_list = np.linspace(10, 100, 19).astype(np.int).tolist()
    func_name_list = ['SVM', 'Material']

    for func_name in func_name_list:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        with open('integral_plots/'+func_name+'_integration_naive.pickle', 'rb') as f:
            legendre_before = pickle.load(f)
        with open('integral_plots/'+func_name+'_integration_expansion.pickle', 'rb') as f:
            legendre_analytical_wointerval = pickle.load(f)
        with open('integral_plots/'+func_name+'_integration_hermite.pickle', 'rb') as f:
            hermite_analytical = pickle.load(f)
        with open('integral_plots/'+func_name+'_integration_hermite_naive.pickle', 'rb') as f:
            hermite = pickle.load(f)
        with open('integral_plots/'+func_name+'_integration_proposed.pickle', 'rb') as f:
            legendre_analytical = pickle.load(f)
        with open('integral_plots/'+func_name+'_integration_MC.pickle', 'rb') as f:
            MC_approx = pickle.load(f)

        ground_truth = MC_approx
        max_idx = 19

        # ax.plot(func_evaluate_num_list[:max_idx], np.abs(legendre_before - ground_truth)[:max_idx], label=r'non-expanded with $[l_{\rm pdf}, u_{\rm pdf}]$')
        # ax.plot(func_evaluate_num_list[:max_idx], np.abs(hermite - ground_truth)[:max_idx], label=r'non-expanded with G-H quad.')
        ax.plot(func_evaluate_num_list[:max_idx], np.abs(hermite_analytical - ground_truth)[:max_idx], label=r'G-H quad.')
        ax.plot(func_evaluate_num_list[:max_idx], np.abs(legendre_analytical_wointerval - ground_truth)[:max_idx], label=r'G-L quad. with $[l_{\rm pdf}, u_{\rm pdf}]$')
        ax.plot(func_evaluate_num_list[:max_idx], np.abs(legendre_analytical - ground_truth)[:max_idx], label=r'G-L quad. with $[l, u]$ (proposed)')

        ax.set_yscale('log')
        ax.set_xlabel(r'\# integrand evaluations')
        ax.set_ylabel(r'Difference to MC estimation')

        ax.set_title(func_name)
        # if 'Material' in func_name:
        #     ax.legend()
        fig.savefig('integral_plots/check_accuracy_'+func_name+'.pdf')
        # plt.close()

        handles, labels = ax.get_legend_handles_labels()
        plt.close()

        legend_fig = plt.figure()
        legend_fig_ax = legend_fig.add_subplot(1,1,1)

        legend_fig_ax.legend(handles, labels, ncol=1, loc='upper left')
        legend_fig_ax.axis("off")
        legend_fig.savefig('check_accuracy_legend.pdf')

        plt.close()


if __name__ == '__main__':
    # integration_test()
    plot_result()