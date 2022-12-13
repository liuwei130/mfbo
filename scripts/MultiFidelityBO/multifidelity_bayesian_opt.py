# -*- coding: utf-8 -*-

import os
import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import mvn
import time


from ..myutils import myutils as utils
from ..myutils.BO_core import MFBO_core

class MFBO(MFBO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features=fidelity_features, optimize=optimize)

    # X
    def next_input_pool(self, X_list):
        self.max_acq = list()
        self.max_acq_idx = list()
        evaluate_X = list()

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            # start_m = time.time()
            if X_list[m].size != 0:
                upper_bound = self._upper_bound(X_list[m])
                if np.size(X_list[m][(upper_bound >= max_lower_bounds).ravel()]) > 0:
                    evaluate_X.append(X_list[m][(upper_bound >= max_lower_bounds).ravel()])
                else:
                    evaluate_X.append(X_list[m])

                # evaluate_X.append(X_list[m][(self._upper_bound(X_list[m]) >= self.y_max).ravel()])
                if m < self.M-1:
                    acq = -1*self.acq_low(evaluate_X[-1])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[-1]])
                else:
                    acq = -1*self.acq_high(evaluate_X[-1])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[-1]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append([- np.inf])
            # print(str(m)+'-th fidelity time:', time.time() - start_m)

        print('optimized_acquisition function values:', self.max_acq)
        self.fidelity_index = np.argmax(np.array(self.max_acq) / np.array(self.cost))
        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        # print('pool selection time: ', time.time() - start)
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list

    def next_input(self):
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.high_minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        if self.max_inputs is not None:
            x0s = np.r_[x0s, self.max_inputs]

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        x0s = x0s[(self._upper_bound(x0s) >= max_lower_bounds).ravel()]

        f_min_list = list()
        x_min_list = list()
        for m in range(self.M-1):
            self.acq_m = self.fidelity_features[m]

            x_min, f_min = utils.minimize(self.acq_low_onepoint, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            f_min_list.append(f_min)
            x_min_list.append(x_min)

        x_min, f_min = utils.minimize(self.acq_high, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        f_min_list.append(f_min)
        x_min_list.append(x_min)

        print('optimized_acquisition function values:', np.array(f_min_list).ravel())
        new_fidelity = np.argmin(np.array(f_min_list).ravel() / np.array(self.cost))
        new_input = x_min_list[new_fidelity]
        return np.atleast_2d(np.r_[new_input, new_fidelity])


class MultiFidelityMaxvalueEntropySearch(MFBO):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, sampling_num=10, EntropyApproxNum=50, sampling_method='Gumbel', GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        start = time.time()
        self.sampling_num = sampling_num
        self.EntropyApproxNum = EntropyApproxNum
        # norm_cdf(-6) \approx 10^{-9}
        self.cons = 6.
        self.logsumexp_const = 1e3
        self.sampling_method = sampling_method
        self.pool_X = pool_X

        self.gauss_legendre_points, self.gauss_legendre_weights = np.polynomial.legendre.leggauss(EntropyApproxNum)
        self.gauss_legendre_points = self.gauss_legendre_points[None, None, :]
        self.gauss_legendre_weights = self.gauss_legendre_weights[None, None, :]


        if sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(pool_X)
        elif sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        start = time.time()
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()

    # (
    def acq_low(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        low_x = np.c_[x, np.matlib.repmat(self.acq_m, x_size, 1)]
        high_x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], x_size, 1)]

        # For fast computation, calculate parameters of low and high fidelity predictive distributions
        mean, var = self.GPmodel.predict_noiseless(np.r_[low_x, high_x])

        low_mean = mean[:x_size, :]
        low_var = var[:x_size, :]
        low_std = np.sqrt(low_var)
        high_mean = mean[x_size:, :]
        high_var = var[x_size:, :]
        high_std = np.sqrt(high_var)

        # compute the covariances between [x, m] amd [x, M]
        cov_low_high = self.GPmodel.diag_covariance_between_points(low_x, high_x)
        rho = cov_low_high / (low_std * high_std)

        normalized_max = (self.maximums - high_mean) / high_std
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        acquisition_values = np.mean(rho**2 * (normalized_max * high_pdf) / (2*high_cdf) - np.log(high_cdf), axis=1)
        conditional_high_std = np.sqrt(high_var - cov_low_high**2 / low_var)

        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = low_mean + low_var / cov_low_high * (self.maximums[None, :] - high_mean)
        cdf_width = np.abs(self.cons*low_var / cov_low_high * conditional_high_std)
        cdf_central[np.isnan(cdf_central)] = 0
        cdf_width[np.isnan(cdf_width)] = np.inf

        pdf_central = low_mean * np.atleast_2d(np.ones(self.sampling_num))
        pdf_width = self.cons * low_std * np.atleast_2d(np.ones(self.sampling_num))

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        remained_index = np.any(tmp_index, axis=1)
        # print(np.shape(np.where(remained_index==True)[0]))
        if np.any(remained_index):
            points_min = points_min[remained_index]
            points_max = points_max[remained_index]
            tmp_index = ~tmp_index[remained_index]
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)[:,:,None]/2. + (points_max-points_min)[:,:,None]/2.*self.gauss_legendre_points
            # until here

            low_pdf = norm.pdf(integrate_points, loc=low_mean[remained_index, :, np.newaxis], scale=low_std[remained_index, :, np.newaxis])
            conditional_high_mean = high_mean[remained_index, :, np.newaxis] + cov_low_high[remained_index, :, np.newaxis] / low_var[remained_index, :, np.newaxis] * (integrate_points - low_mean[remained_index, :, np.newaxis])
            conditional_high_cdf = norm.cdf((self.maximums[np.newaxis, :, np.newaxis] - conditional_high_mean) / conditional_high_std[remained_index, :, np.newaxis])
            # avoid the numerical error with log0 (By substituting 1, integrand = 0 log 0 = log1 = 0)
            conditional_high_cdf[conditional_high_cdf <= 0] = 1

            numerical_integration_term = np.sum( self.gauss_legendre_weights*low_pdf*conditional_high_cdf * np.log(conditional_high_cdf), axis=2) / high_cdf[remained_index] * (points_max-points_min)/2.
            numerical_integration_term = np.mean(numerical_integration_term, axis=1).ravel()
            numerical_integration_term[np.abs(numerical_integration_term) < 1e-10] = 0
            acquisition_values[remained_index] = acquisition_values[remained_index] + numerical_integration_term

        return - acquisition_values

    # 1 \times sampling_num \times Entropy approx num
    def acq_low_onepoint(self, x):
        x = np.atleast_2d(np.c_[np.matlib.repmat(x, 2, 1), np.array([self.acq_m, self.fidelity_features[-1]])])

        mean, cov = self.GPmodel.predict_noiseless(x, full_cov=True)
        std = np.sqrt(np.diag(cov))
        rho = cov[1, 0] / (std[0]*std[1])

        normalized_max = (self.maximums[None, :] - mean[1]) / std[1]
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        acquisition_values = np.array([np.mean(rho**2 * (normalized_max * high_pdf) / (2*high_cdf) - np.log(high_cdf))])

        conditional_high_std = np.sqrt(cov[1,1] - cov[1,0]**2 / cov[0,0])

        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = mean[0] + cov[0,0] / cov[1,0] * (self.maximums[None, :] - mean[1])
        cdf_width = np.abs(self.cons*cov[0,0] / cov[1,0] * conditional_high_std)

        cdf_central[np.isnan(cdf_central)] = 0
        if np.isnan(cdf_width):
            cdf_width = np.inf

        pdf_central = mean[0] * np.atleast_2d(np.ones(self.sampling_num))
        pdf_width = self.cons * std[0]

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        # remained_index = np.any(points_max > points_min, axis=1)
        # print(np.shape(np.where(remained_index==True)[0]))
        if np.any(tmp_index, axis=1):
            tmp_index = ~tmp_index
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)[:,:,None]/2. + (points_max-points_min)[:,:,None]/2.*self.gauss_legendre_points
            # until here

            low_pdf = norm.pdf(integrate_points, loc=mean[0], scale=std[0])
            conditional_high_mean = mean[1] + cov[1,0] / cov[0,0] * (integrate_points - mean[0])
            conditional_high_cdf = norm.cdf((self.maximums[np.newaxis, :, np.newaxis] - conditional_high_mean) / conditional_high_std)
            # avoid the numerical error with log0 (By substituting 1, integrand = 0 log 0 = log1 = 0)
            conditional_high_cdf[conditional_high_cdf <= 0] = 1

            numerical_integration_term = np.sum( self.gauss_legendre_weights*low_pdf*conditional_high_cdf * np.log(conditional_high_cdf), axis=2) / high_cdf * (points_max-points_min)/2.
            numerical_integration_term = np.mean(numerical_integration_term, axis=1).ravel()
            numerical_integration_term[np.abs(numerical_integration_term) < 1e-10] = 0
            acquisition_values = acquisition_values + numerical_integration_term

        return - acquisition_values



class GIBBON(MFBO):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, sampling_num=10, sampling_method='Gumbel', GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        start = time.time()
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.pool_X = pool_X


        if sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(pool_X)
        elif sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        start = time.time()
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        pdf_cdf = pdf / cdf
        return np.sum(np.log(1 - pdf_cdf * (normalized_max + pdf_cdf)), axis=1).ravel()
        # return np.mean(np.log(1 - pdf_cdf * (normalized_max + pdf_cdf)) / 2., axis=1).ravel()


    # (
    def acq_low(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        low_x = np.c_[x, np.matlib.repmat(self.acq_m, x_size, 1)]
        high_x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], x_size, 1)]

        # For fast computation, calculate parameters of low and high fidelity predictive distributions
        mean, var = self.GPmodel.predict_noiseless(np.r_[low_x, high_x])

        # low_mean = mean[:x_size, :]
        low_var = var[:x_size, :]
        low_std = np.sqrt(low_var)
        high_mean = mean[x_size:, :]
        high_var = var[x_size:, :]
        high_std = np.sqrt(high_var)

        # compute the covariances between [x, m] amd [x, M]
        cov_low_high = self.GPmodel.diag_covariance_between_points(low_x, high_x)
        rho = cov_low_high / (low_std * high_std)

        normalized_max = (self.maximums - high_mean) / high_std
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        pdf_cdf = high_pdf / high_cdf

        return np.sum(np.log(1 - rho**2 * pdf_cdf * (normalized_max + pdf_cdf)), axis=1).ravel()
        # return np.mean(np.log(1 - rho**2 * pdf_cdf * (normalized_max + pdf_cdf)) / 2., axis=1).ravel()


    # 1 \times sampling_num \times Entropy approx num
    def acq_low_onepoint(self, x):
        x = np.atleast_2d(np.c_[np.matlib.repmat(x, 2, 1), np.array([self.acq_m, self.fidelity_features[-1]])])

        mean, cov = self.GPmodel.predict_noiseless(x, full_cov=True)
        std = np.sqrt(np.diag(cov))
        rho = cov[1, 0] / (std[0]*std[1])

        normalized_max = (self.maximums[None, :] - mean[1]) / std[1]
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        pdf_cdf = high_pdf / high_cdf

        return np.sum(np.log(1 - rho**2 * pdf_cdf * (normalized_max + pdf_cdf)), axis=1).ravel()
        # return np.mean(np.log(1 - rho**2 * pdf_cdf * (normalized_max + pdf_cdf)) / 2., axis=1).ravel()

    def next_input_pool(self, X_list):
        self.max_acq = list()
        self.max_acq_idx = list()
        evaluate_X = list()

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            # start_m = time.time()
            if X_list[m].size != 0:
                upper_bound = self._upper_bound(X_list[m])
                evaluate_X.append(X_list[m][(upper_bound >= max_lower_bounds).ravel()])
                # evaluate_X.append(X_list[m][(self._upper_bound(X_list[m]) >= self.y_max).ravel()])
                if m < self.M-1:
                    acq = -1*self.acq_low(evaluate_X[-1])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[-1]])
                else:
                    acq = -1*self.acq_high(evaluate_X[-1])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[-1]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append([- np.inf])
            # print(str(m)+'-th fidelity time:', time.time() - start_m)

        # '''
        # set the ordinal scale by log
        # '''
        # self.max_acq = -1 * np.log(-1 * np.array(self.max_acq))
        print('optimized_acquisition function values:', self.max_acq)
        self.fidelity_index = np.argmax(np.array(self.max_acq) / np.array(self.cost))
        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        # print('pool selection time: ', time.time() - start)
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list

    def next_input(self):
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.high_minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        if self.max_inputs is not None:
            x0s = np.r_[x0s, self.max_inputs]

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        x0s = x0s[(self._upper_bound(x0s) >= max_lower_bounds).ravel()]

        f_min_list = list()
        x_min_list = list()
        for m in range(self.M-1):
            self.acq_m = self.fidelity_features[m]

            x_min, f_min = utils.minimize(self.acq_low_onepoint, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            f_min_list.append(f_min)
            x_min_list.append(x_min)

        x_min, f_min = utils.minimize(self.acq_high, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        f_min_list.append(f_min)
        x_min_list.append(x_min)

        # '''
        # set the ordinal scale by log
        # '''
        # f_min_list = np.log(np.array(f_min_list))
        print('optimized_acquisition function values:', np.array(f_min_list).ravel())

        if np.min(f_min_list) < 0:
            new_fidelity = np.argmin(np.array(f_min_list).ravel() / np.array(self.cost))
            new_input = x_min_list[new_fidelity]
        else:
            print('Warnings: Acquisition function value is zero anywhere, then next input is chosen by usual MES with RFM.')
            new_fidelity = int(self.M - 1)
            # new_input = np.random.rand(self.input_dim) * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)
            x_min, f_min = utils.minimize(self.acq_high_MES, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            new_input = x_min

        return np.atleast_2d(np.r_[new_input, new_fidelity])

    def acq_high_MES(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()




class MultiFidelitySuquentialKrigingOptimization(MFBO):
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)

        # c
        #
        start = time.time()
        self.c = 1
        obtained_X = np.c_[self.unique_X, np.matlib.repmat(self.fidelity_features[-1], np.shape(self.unique_X)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(obtained_X)
        self.y_max = mean[np.argmax(mean - self.c * np.sqrt(var))]

        self.preprocessing_time = time.time() - start

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        start = time.time()
        obtained_X = np.c_[self.unique_X, np.matlib.repmat(self.fidelity_features[-1], np.shape(self.unique_X)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(obtained_X)
        self.y_max = mean[np.argmax(mean - self.c * np.sqrt(var))]
        self.preprocessing_time = time.time() - start


    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        Z = (mean - self.y_max) / std
        EI = (Z * std)*norm.cdf(Z) + std*norm.pdf(Z)

        alpha1 = 1.
        noise_var = self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2
        alpha2 = 1 - np.sqrt(noise_var) / np.sqrt(var + noise_var)
        return -1 *(EI * alpha1 * alpha2).ravel()

    # (
    def acq_low(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        low_x = np.c_[x, np.matlib.repmat(self.acq_m, x_size, 1)]
        high_x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], x_size, 1)]

        # GPy
        # low_mean, low_var = self.GPmodel.predict_noiseless(low_x)
        # high_mean, high_var = self.GPmodel.predict_noiseless(high_x)
        mean, var = self.GPmodel.predict_noiseless(np.r_[low_x, high_x])
        # low_mean = mean[:x_size, :]
        low_var = var[:x_size, :]
        high_mean = mean[x_size:, :]
        high_var = var[x_size:, :]

        #
        cov_low_high = self.GPmodel.diag_covariance_between_points(low_x, high_x)

        high_std = np.sqrt(high_var)
        Z = (high_mean - self.y_max) / high_std
        EI = (Z * high_std)*norm.cdf(Z) + high_std*norm.pdf(Z)

        alpha1 = np.abs(cov_low_high / np.sqrt(low_var * high_var))
        noise_var = self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2
        alpha2 = 1 - np.sqrt(noise_var) / np.sqrt(low_var + noise_var)
        return -1 * (EI * alpha1 * alpha2).ravel()

    #
    def acq_low_onepoint(self, x):
        x = np.atleast_2d(np.c_[np.matlib.repmat(x, 2, 1), np.array([self.acq_m, self.fidelity_features[-1]])])
        mean, cov = self.GPmodel.predict_noiseless(x, full_cov=True)

        high_std = np.sqrt(cov[1, 1])
        Z = (mean[1] - self.y_max) / high_std
        EI = (Z * high_std)*norm.cdf(Z) + high_std*norm.pdf(Z)

        alpha1 = np.abs(cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1]))
        noise_var = self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2
        alpha2 = 1 - np.sqrt(noise_var) / np.sqrt(cov[0, 0] + noise_var)
        return -1 * (EI * alpha1 * alpha2)


class MultiFidelityBayesianOptimizationWithContinuousApproximations(MFBO):
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, iteration, GPmodel=None, model_name='MTGP', fidelity_features=None, optimize=True):
        if model_name != 'MTGP':
            print('model is incorrect')
            exit(1)
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features=fidelity_features, optimize=optimize)
        start = time.time()
        self.iteration = iteration
        self.beta = 0.5 * self.input_dim * np.log(2. * self.iteration + 1)
        self.q = 1. / (self.input_dim + self.fidelity_feature_dim + 2)
        self.preprocessing_time = time.time() - start

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        start = time.time()
        self.iteration = self.iteration + 1
        self.beta = 0.5 * self.input_dim * np.log(2. * self.iteration + 1)
        self.q = 1. / (self.input_dim + self.fidelity_feature_dim + 2)
        self.preprocessing_time = time.time() - start


    def xi(self, fidelity_index):
        return np.sqrt(1 - self.GPmodel.kern.rbf_1.K(np.atleast_2d(np.r_[np.zeros(self.input_dim), self.fidelity_features[fidelity_index]]),
            np.atleast_2d(np.r_[np.zeros(self.input_dim), self.fidelity_features[-1, :]]))**2)

    def gamma(self, fidelity_index):
        return self.GPmodel.std * self.xi(fidelity_index) * np.power(self.cost[fidelity_index] / self.cost[-1], self.q)

    def acq(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        return - (mean + np.sqrt(self.beta * var)).ravel()

    def select_fidelity(self, new_input, X_list=None):
        # BOCA
        #
        features_min = np.min(self.fidelity_features, axis=0)
        fidelity_p = features_min + (np.max(self.fidelity_features, axis=0) - features_min) * np.sqrt(np.shape(self.fidelity_features)[1])
        xi_p = np.sqrt(1 - self.GPmodel.kern.rbf_1.K(np.atleast_2d(np.r_[np.zeros(self.input_dim), fidelity_p]),
            np.atleast_2d(np.r_[np.zeros(self.input_dim), self.fidelity_features[-1, :]]))**2)
        for m in range(self.M):
            if X_list is None:
                if self.xi(m) > xi_p / np.sqrt(self.beta):
                    _, var = self.GPmodel.predict_noiseless(np.c_[np.r_[np.array(new_input),self.fidelity_features[m]]].T)
                    if np.sqrt(var) > self.gamma(m):
                        return m
            elif new_input.tolist() in X_list[m].tolist():
                if self.xi(m) > xi_p / np.sqrt(self.beta):
                    _, var = self.GPmodel.predict_noiseless(np.c_[np.r_[np.array(new_input),self.fidelity_features[m]]].T)
                    if np.sqrt(var) > self.gamma(m):
                        return m
        return self.M - 1

    def next_input(self):
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.high_minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]


        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        upper_bound = self._upper_bound(x0s)
        x0s = x0s[(upper_bound >= max_lower_bounds).ravel()]

        x_min, _ = utils.minimize(self.acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)

        new_input = x_min
        new_fidelity = self.select_fidelity(new_input)
        # print(res)
        return np.atleast_2d(np.r_[new_input, new_fidelity])

    def next_input_pool(self, X_list):
        self.acquistion_values = self.acq(X_list[-1])

        max_index = np.argmin(self.acquistion_values)
        new_input = X_list[-1][max_index]
        self.fidelity_index = self.select_fidelity(new_input, X_list)
        # print(res)
        X_list[self.fidelity_index] = np.delete(X_list[self.fidelity_index], X_list[self.fidelity_index].tolist().index(new_input.tolist()), axis=0)
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list

class MultiInformationSourceKnowledgeGradient(MFBO):
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        start = time.time()
        #
        num_discrete_points = np.min([100 ** self.input_dim, 3000*self.input_dim, 10000])
        #
        # self.discrete_points = np.c_[np.random.rand(num_discrete_points, self.input_dim)] * (bounds[1]- bounds[0]) + bounds[0]
        self.discrete_points = lhs(self.input_dim, samples=num_discrete_points, criterion='maximin', iterations=5)  * (bounds[1]- bounds[0]) + bounds[0]
        self.discrete_points = np.c_[self.discrete_points, np.matlib.repmat(self.fidelity_features[-1], np.shape(self.discrete_points)[0], 1)]
        self.mean, _ = self.GPmodel.predict_noiseless(self.discrete_points)
        self.mean = self.mean.ravel()
        # if num_discrete_points > 3000:
        #     argsort_mean = np.argsort(self.mean.ravel())
        #     self.discrete_points = self.discrete_points[argsort_mean[-3000:]]
        #     self.mean = self.mean[argsort_mean[-3000:]]
        self.preprocessing_time = time.time() - start

    def compute_d_A(self, a_in, b_in):
        """Algorithm 1 in Frazier 2009 paper
        :param a_in:
        :param b_in:
        :return:
        """
        M = np.size(a_in)
        # Use the same subscripts as in Algorithm 1, therefore, a[0] and b[0] are dummy values with no meaning
        a = numpy.r_[[numpy.inf], a_in]
        b = numpy.r_[[numpy.inf], b_in]
        d = numpy.zeros(M+1)

        d[0] = -numpy.inf
        d[1] = numpy.inf
        A = [1]
        for i in range(1, M):
            d[i+1] = numpy.inf
            while True:
                j = A[-1]
                d[j] = (a[j] - a[i+1]) / (b[i+1] - b[j])
                if len(A) != 1 and d[j] <= d[A[-2]]:
                    del A[-1]
                else:
                    break
            A.append(i+1)
        return d, A

    def compute_kg(self, x):
        _, var = self.GPmodel.predict(x)
        cov_x_dispoints = self.GPmodel.posterior_covariance_between_points(x, self.discrete_points)

        a = self.mean.copy()
        b = cov_x_dispoints.ravel() / np.sqrt(var.ravel() + self.GPmodel['.*Gaussian_noise.variance'].values)

        ab = np.c_[a, b]
        sort_index = np.lexsort(ab.T)
        a = a[sort_index]
        b = b[sort_index]

        dominated_index = list()
        for i in range(np.size(b)-1):
            if b[i] == b[i+1]:
                dominated_index.append(i)

        a = np.delete(a, dominated_index)
        b = np.delete(b, dominated_index)

        d, A = self.compute_d_A(a, b)
        A_1 = [i-1 for i in A]
        b = b[A_1]
        d = d[A]

        diff_b = (b[1:] - b[:-1])

        d = d[:-1]
        acq_value = np.log(diff_b) - 0.5*np.log(2. * np.pi) - 0.5*d**2
        d_abs = np.abs(d)

        cutoff = 10.0
        temp_index = d_abs < cutoff
        acq_value[temp_index] += np.log1p(-d_abs[temp_index]*norm.cdf(-d_abs[temp_index]) / norm.pdf(d_abs[temp_index]))
        temp_index = d_abs >= cutoff
        acq_value[temp_index] += np.log1p(-d[temp_index]**2 / (d[temp_index]**2 + 1))

        acq_value = np.sum(np.exp(acq_value))
        return -1*acq_value

    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        return self.compute_kg(x)

    def acq_low_onepoint(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.acq_m, np.shape(x)[0], 1)]
        return self.compute_kg(x)


class MultifidelityPredictiveEntropySearch(MFBO):
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, sampling_num=10, GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        self.sampling_num = sampling_num
        self.pool_X = pool_X

        start = time.time()
        if self.model_name == 'MFGP':
            _ , self.max_inputs, self.c = self.sampling_MFRFM(pool_X, slack=True)
        elif self.model_name == 'MTGP':
            _ , self.max_inputs, self.c = self.sampling_MTRFM(pool_X, slack=True)

        self.y_max_ep = np.array([self.GPmodel.mean if i==np.array([]) else np.max(i) for i in Y_list])
        self.mean_new, self.var_new = self.ep_maximums_params()

        self.preprocessing_time = time.time() - start
        print('max_inputs :', self.max_inputs)
        print('slack variable :', self.c)

    def update(self, add_X_list, add_Y_list, optimize=False):
        self.y_max_ep[self.y_max_ep==self.GPmodel.mean] = -np.inf

        super().update(add_X_list, add_Y_list, optimize=optimize)
        start = time.time()
        if self.model_name == 'MFGP':
            _, self.max_inputs, self.c = self.sampling_MFRFM(self.pool_X, slack=True)
        elif self.model_name == 'MTGP':
            _, self.max_inputs, self.c = self.sampling_MTRFM(self.pool_X, slack=True)

        add_y_max = np.array([ -np.inf if np.size(add_Y_list[m])==0 else np.max(add_Y_list[m]) for m in range(self.M)])
        self.y_max_ep = np.array([np.max([self.y_max_ep[m], add_y_max[m]]) for m in range(self.M)])
        self.y_max_ep[np.isinf(self.y_max_ep)] = self.GPmodel.mean
        self.mean_new, self.var_new = self.ep_maximums_params()
        self.preprocessing_time = time.time() - start
        print(self.y_max_ep)
        print('max_inputs :', self.max_inputs)
        print('slack variable :', self.c)


    def ep_maximums_params(self):
        mean_tilde = np.zeros((self.M, self.sampling_num))
        var_tilde = np.ones((self.M, self.sampling_num)) * np.inf
        mean_new = np.zeros((self.M, self.sampling_num))
        var_new = np.ones((self.M, self.M, self.sampling_num)) * np.inf
        for j in range(self.sampling_num):
            x_temp = np.c_[np.matlib.repmat(np.c_[self.max_inputs[j]].T, self.M, 1), self.fidelity_features]
            mean0, var0 = self.GPmodel.predict_noiseless(x_temp, full_cov=True)

            #
            mean = mean0.ravel()
            var = var0

            # ep
            for i in range(100000):
                v_bar = 1./(1./np.diag(var) - 1./var_tilde[:, j])
                m_bar = v_bar*(mean/np.diag(var) - mean_tilde[:, j]/var_tilde[:, j])

                alpha = (m_bar - self.y_max_ep) / np.sqrt(v_bar + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)
                cdf = norm.cdf(alpha)
                # 0
                cdf[cdf<=0] = 1e-12
                temp = norm.pdf(alpha) / cdf

                beta = temp * (temp + alpha) / (v_bar + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)
                kappa = (temp + alpha) / np.sqrt(v_bar + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)

                old_mean_tilde = mean_tilde[:, j]
                old_var_tilde = var_tilde[:, j]

                mean_tilde[:, j] = m_bar + 1./kappa
                var_tilde[:, j] = 1./beta - v_bar

                var_new[:,:,j] = np.linalg.inv(np.linalg.inv(var0) + np.linalg.inv(np.diag(var_tilde[:, j])))
                mean_new[:,j] = var_new[:,:,j].dot(np.linalg.inv(np.diag(var_tilde[:, j])).dot(np.c_[mean_tilde[:, j]]) + np.linalg.inv(var0).dot(np.c_[mean0])).ravel()

                #
                #
                change = np.max(np.c_[np.abs( mean_new[:,j] - mean ), np.abs( np.diag(var_new[:,:,j]) - np.diag(var) )])

                if np.isnan(change):
                    print('Error : convergence value is NAN')
                    print('mean0:', mean0)
                    print('var0 :', var0)
                    print('y_max :', self.y_max_ep)
                    print('v_bar', v_bar)
                    print('m_bar', m_bar)
                    print('temp', temp)
                    print('alpha', alpha)
                    print('kappa', kappa)
                    print('var_tilde :', var_tilde)
                    print('mean_tilde:', mean_tilde)
                    print('old_var_tilde :', old_var_tilde)
                    print('old_mean_tilde:', old_mean_tilde)
                    print('var :', var)
                    print('mean:', mean)
                    exit()

                mean = mean_new[:,j]
                var = var_new[:,:,j]
                if change < 1e-10:
                    # print('EP convergence')
                    break
                if i >= (100000-1):
                    print('Error : EP don\'t converge, convergence value is', change)

        return mean_new, var_new

    # def acq_high(self, x):
    #     conditional_entropy = 0
    #     for j in range(self.sampling_num):
    #         x_temp = np.c_[np.c_[self.max_inputs[j]].T, self.fidelity_features[-1]]
    #         x_temp = np.r_[np.c_[np.r_[np.array(x), self.fidelity_features[-1]]].T, x_temp]
    #         pre_mean, pre_var = self.GPmodel.predict_noiseless(x_temp, full_cov=True)
    #         if j==0:
    #             pre_entropy = np.log(2*np.pi*np.e*(pre_var[0, 0] + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.

    #         var_tilde_inv = np.linalg.inv(np.diag(self.var_tilde[[0, -1], j]))
    #         pre_var_inv = np.linalg.inv(pre_var)

    #         var = np.linalg.inv(pre_var_inv + var_tilde_inv)
    #         mean = var.dot(pre_var_inv.dot(pre_mean) + var_tilde_inv.dot(np.c_[self.mean_tilde[[0, -1], j]]))

    #         # mean = mean[[0, -1]]
    #         # var = var[np.ix_([0, -1], [0, -1])]

    #         # PES
    #         s = var[0, 0] + var[1, 1] - 2*var[1, 0]
    #         if s <= 0:
    #             s = 1e-10
    #         mu = - mean[0] + mean[1]
    #         alpha = mu / np.sqrt(s)
    #         cdf_alpha = norm.cdf(alpha)
    #         if cdf_alpha <= 0:
    #             beta = - alpha
    #         else:
    #             beta = norm.pdf(alpha) / cdf_alpha

    #         conditional_var = var[0, 0] - beta * (beta+alpha) * (var[0, 0] - var[0, 1])**2 / s
    #         conditional_entropy += np.log(2*np.pi*np.e*(conditional_var + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.
    #     return -1 * (pre_entropy - conditional_entropy / self.sampling_num)

    # def acq_low(self, x):
    #     conditional_entropy = 0
    #     for j in range(self.sampling_num):
    #         x_temp = np.c_[np.matlib.repmat(np.c_[self.max_inputs[j]].T, self.M, 1), self.fidelity_features]
    #         x_temp = np.r_[np.atleast_2d(np.r_[np.array(x), self.acq_m]), x_temp]
    #         pre_mean, pre_var = self.GPmodel.predict_noiseless(x_temp, full_cov=True)
    #         if j==0:
    #             pre_entropy = np.log(2*np.pi*np.e*(pre_var[0, 0] + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.

    #         var_tilde_inv = np.linalg.inv(np.diag(self.var_tilde[:, j]))
    #         pre_var_inv = np.linalg.inv(pre_var)

    #         var = np.linalg.inv(pre_var_inv + var_tilde_inv)
    #         mean = var.dot(pre_var_inv.dot(pre_mean) + var_tilde_inv.dot(np.c_[self.mean_tilde[:, j]]))

    #         mean = mean[[0, self.acq_m+1]]
    #         var = var[np.ix_([0, self.acq_m+1], [0, self.acq_m+1])]

    #         # PES
    #         s = var[0, 0] + var[1, 1] - 2*var[1, 0]
    #         if s <= 0:
    #             s = 1e-10
    #         mu = - mean[0] + mean[1] + self.c[self.acq_m]
    #         alpha = mu / np.sqrt(s)
    #         cdf_alpha = norm.cdf(alpha)
    #         if cdf_alpha <= 0:
    #             beta = - alpha
    #         else:
    #             beta = norm.pdf(alpha) / cdf_alpha

    #         conditional_var = var[0, 0] - beta * (beta+alpha) * (var[0, 0] - var[0, 1])**2 / s
    #         conditional_entropy += np.log(2*np.pi*np.e*(conditional_var + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.
    #     return -1 * (pre_entropy - conditional_entropy / self.sampling_num)

    def acq_high(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        x = np.r_[x, self.max_inputs]
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]

        mean, var = self.GPmodel.predict_noiseless(x)

        #
        #

        pre_mean = np.c_[mean[:x_size,:]]
        pre_max_mean = np.c_[mean[x_size:, :]].T
        pre_var = np.c_[var[:x_size,:]]
        pre_max_var = np.c_[var[x_size:, :]].T
        pre_cov = self.GPmodel.posterior_covariance_between_points(np.atleast_2d(x[:x_size,:]), x[x_size:,:])
        ep_max_mean = np.atleast_2d(self.mean_new[-1, :])
        ep_max_var = np.atleast_2d(self.var_new[-1, -1, :])

        cond_mean = pre_mean + (ep_max_mean - pre_max_mean) * pre_cov / pre_max_var
        cond_max_mean = ep_max_mean
        cond_var = pre_var - pre_cov**2 / pre_max_var * (1 - ep_max_var / pre_max_var)
        cond_cov = pre_cov / pre_max_var * ep_max_var
        cond_max_var = ep_max_var

        s = cond_var + cond_max_var - 2*cond_cov
        s[s<=0] = 1e-10
        mu = -cond_mean + cond_max_mean + self.c[-1]
        alpha = mu / np.sqrt(s)
        cdf_alpha = norm.cdf(alpha)
        beta = -alpha
        beta[cdf_alpha > 0] = norm.pdf(alpha[cdf_alpha > 0]) / cdf_alpha[cdf_alpha > 0]

        truncated_cond_var = cond_var - beta * (beta+alpha) * (cond_var - cond_cov)**2 / s

        pre_entropy = np.log(2*np.pi*np.e*(pre_var[:, 0] + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.
        cond_entropy = np.mean(np.log(2*np.pi*np.e*(truncated_cond_var + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2., axis=1)

        return -1 * (pre_entropy - cond_entropy)

    def acq_low(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        x = np.r_[x, self.max_inputs]
        x = np.c_[x, np.matlib.repmat(self.acq_m, np.shape(x)[0], 1)]

        mean, var = self.GPmodel.predict_noiseless(x)

        #
        #

        pre_mean = np.c_[mean[:x_size,:]]
        pre_max_mean = np.c_[mean[x_size:, :]].T
        pre_var = np.c_[var[:x_size,:]]
        pre_max_var = np.c_[var[x_size:, :]].T
        pre_cov = self.GPmodel.posterior_covariance_between_points(np.atleast_2d(x[:x_size,:]), x[x_size:,:])
        fidelity_index = self.fidelity_features.ravel() == self.acq_m[0]
        ep_max_mean = np.atleast_2d(self.mean_new[fidelity_index, :])
        ep_max_var = np.atleast_2d(self.var_new[fidelity_index, fidelity_index, :])

        cond_mean = pre_mean + (ep_max_mean - pre_max_mean) * pre_cov / pre_max_var
        cond_max_mean = ep_max_mean
        cond_var = pre_var - pre_cov**2 / pre_max_var * (1 - ep_max_var / pre_max_var)
        cond_cov = pre_cov / pre_max_var * ep_max_var
        cond_max_var = ep_max_var

        s = cond_var + cond_max_var - 2*cond_cov
        s[s<=0] = 1e-10
        mu = -cond_mean + cond_max_mean + self.c[fidelity_index]
        alpha = mu / np.sqrt(s)
        cdf_alpha = norm.cdf(alpha)
        beta = -alpha
        beta[cdf_alpha > 0] = norm.pdf(alpha[cdf_alpha > 0]) / cdf_alpha[cdf_alpha > 0]

        truncated_cond_var = cond_var - beta * (beta+alpha) * (cond_var - cond_cov)**2 / s

        pre_entropy = np.log(2*np.pi*np.e*(pre_var[:, 0] + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2.
        cond_entropy = np.mean(np.log(2*np.pi*np.e*(truncated_cond_var + self.GPmodel['.*Gaussian_noise.variance'].values * self.GPmodel.std**2)) / 2., axis=1)
        return -1 * (pre_entropy - cond_entropy)

    def acq_low_onepoint(self, x):
        return self.acq_low(x)

