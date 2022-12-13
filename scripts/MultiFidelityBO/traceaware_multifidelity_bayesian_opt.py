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
import itertools

from numba import jit

from ..myutils import myutils as utils
from ..myutils.BO_core import MFBO_core

from .multifidelity_bayesian_opt import MultiFidelityMaxvalueEntropySearch


class TA_MFBO(MFBO_core):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        start = time.time()

        self.pool_X = pool_X
        self.freeze_m = -1
        input, count = np.unique(np.vstack([X for X in X_list if np.size(X) > 0]), return_counts=True, axis=0)
        # delete highest-fidelity
        input = input[count < M]
        count = count[count < M]
        self.freeze_inputs = np.c_[np.atleast_2d(input), np.c_[count]-1]
        print('freeze_input:', self.freeze_inputs)


    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        self.preprocessing_time = 0
        print('freeze inputs:', self.freeze_inputs)

    @abstractmethod
    def prepare_new_acq(self):
        pass

    def next_input_pool(self, X_list):
        freeze_fidelity_list = []
        freeze_acq_list = []
        for i in range(np.shape(self.freeze_inputs)[0]):
            fidelity, acq = self.acq_for_freeze(self.freeze_inputs[i])
            freeze_fidelity_list.append(fidelity)
            freeze_acq_list.append(acq)

        self.min_acq = list()
        self.min_x = list()
        evaluate_X = list()

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m][0]

            if X_list[m].size != 0:
                upper_bound = self._upper_bound(X_list[m])
                if np.size(X_list[m][(upper_bound >= max_lower_bounds).ravel()]) > 0:
                    evaluate_X.append(X_list[m][(upper_bound >= max_lower_bounds).ravel()])
                else:
                    evaluate_X.append(X_list[m])

                if m < self.M-1:
                    acq = self.acq_low(evaluate_X[-1])
                    min_idx = np.argmin(acq)
                    self.min_x.append(evaluate_X[-1][min_idx])
                    self.min_acq.append(acq[min_idx])
                else:
                    acq = self.acq_high(evaluate_X[-1])
                    self.min_x.append(evaluate_X[-1][min_idx])
                    self.min_acq.append(acq[min_idx])
            else:
                evaluate_X.append([])
                self.min_x.append([])
                self.min_acq.append(- np.inf)

        acq_list = (np.hstack(self.min_acq) / np.array(self.cost)).tolist()
        acq_list.extend(freeze_acq_list)
        fidelity_list = np.arange(self.M).tolist()
        fidelity_list.extend(freeze_fidelity_list)

        temp_index = np.argmin(acq_list)
        new_fidelity = fidelity_list[temp_index]

        if temp_index < self.M:
            x = self.min_x[new_fidelity]
            new_input = np.c_[np.matlib.repmat(x, new_fidelity+1, 1), np.c_[self.fidelity_features[:new_fidelity+1]]]
            start_fidelity = -1
            remain_cost = self.cost[:new_fidelity+1]

            if new_fidelity < self.M-1:
                self.freeze_inputs = np.r_[self.freeze_inputs, np.c_[np.atleast_2d(self.min_x[new_fidelity]), new_fidelity]]
        else:
            start_fidelity = int(self.freeze_inputs[temp_index - self.M][-1])

            x = self.freeze_inputs[temp_index - self.M, :-1]
            new_input = np.c_[np.matlib.repmat(x, new_fidelity - start_fidelity, 1), np.c_[self.fidelity_features[start_fidelity+1:new_fidelity+1]]]

            remain_cost = self.cost[start_fidelity+1 : new_fidelity+1] - self.cost[start_fidelity]

            if new_fidelity==self.M-1:
                self.freeze_inputs = np.delete(self.freeze_inputs, temp_index - self.M, 0)
            else:
                self.freeze_inputs[temp_index - self.M, -1] = new_fidelity

        for m in range(self.M):
            if start_fidelity < m and m <= new_fidelity:
                X_list[m] = X_list[m][~np.all(X_list[m]==x, axis=1)]
        return new_input, remain_cost, X_list



    def next_input(self):
        freeze_fidelity_list = []
        freeze_acq_list = []
        for i in range(np.shape(self.freeze_inputs)[0]):
            fidelity, acq = self.acq_for_freeze(self.freeze_inputs[i])
            freeze_fidelity_list.append(fidelity)
            freeze_acq_list.append(acq)


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
            self.acq_m = self.fidelity_features[m][0]

            x_min, f_min = utils.minimize(self.acq_low_onepoint, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            f_min_list.append(f_min)
            x_min_list.append(x_min)

        x_min, f_min = utils.minimize(self.acq_high, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        f_min_list.append(f_min)
        x_min_list.append(x_min)


        acq_list = (np.hstack(f_min_list) / np.array(self.cost)).tolist()
        acq_list.extend(freeze_acq_list)
        fidelity_list = np.arange(self.M).tolist()
        fidelity_list.extend(freeze_fidelity_list)

        temp_index = np.argmin(acq_list)
        new_fidelity = fidelity_list[temp_index]

        if temp_index < self.M:
            x = x_min_list[new_fidelity]
            new_input = np.c_[np.matlib.repmat(x, new_fidelity+1, 1), np.c_[self.fidelity_features[:new_fidelity+1]]]
            start_fidelity = -1
            remain_cost = self.cost[:new_fidelity+1]

            if new_fidelity < self.M-1:
                self.freeze_inputs = np.r_[self.freeze_inputs, np.c_[np.atleast_2d(x_min_list[new_fidelity]), new_fidelity]]
        else:
            start_fidelity = int(self.freeze_inputs[temp_index - self.M][-1])

            x = self.freeze_inputs[temp_index - self.M, :-1]
            new_input = np.c_[np.matlib.repmat(x, new_fidelity - start_fidelity, 1), np.c_[self.fidelity_features[start_fidelity+1:new_fidelity+1]]]

            remain_cost = self.cost[start_fidelity+1 : new_fidelity+1] - self.cost[start_fidelity]

            if new_fidelity==self.M-1:
                self.freeze_inputs = np.delete(self.freeze_inputs, temp_index - self.M, 0)
            else:
                self.freeze_inputs[temp_index - self.M, -1] = new_fidelity
        return new_input, remain_cost

    @abstractmethod
    def acq_for_freeze(self,x):
        pass


class TA_MultiFidelityMaxvalueEntropySearch(TA_MFBO):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, sampling_num=10, EntropyApproxNum=50, sampling_method='RFM', GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
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


    def max_value_sampling(self):
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)

    def prepare_new_acq(self):
        self.max_value_sampling()

    def update(self, add_X_list, add_Y_list, optimize=False, prepare_new_acq=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        if prepare_new_acq:
            start = time.time()
            self.prepare_new_acq()
            self.preprocessing_time = time.time() - start
        else:
            self.preprocessing_time = 0
        print('sampled maximums:', self.maximums)

    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()

    def acq_low_naive(self, X):
        X = np.atleast_2d(X)
        acq = list()
        for i in range(np.shape(X)[0]):
            acq.append(self.acq_low_onepoint(X[i]))
        return np.array(acq)

    # max sample num \times candidate x num \times M \times M
    def acq_low(self, X):
        X = np.atleast_2d(X)
        X_size = np.shape(X)[0]
        cov_each_x = np.empty(shape=(1,X_size,(self.acq_m+1)+1,(self.acq_m+1)+1))
        tmp_index = list(range(self.acq_m+1)) + [self.M-1]

        X_multi_fidelity = np.vstack( [ np.c_[X, (m)*np.c_[np.ones(X_size)]] for m in tmp_index ] )
        mean, var = self.GPmodel.predict_noiseless(X_multi_fidelity)
        self.high_mean = mean[-X_size:].ravel()
        for i, m in enumerate(tmp_index):
            cov_each_x[:,:,i,i] = var[i*X_size:(i+1)*X_size].T

        for i in itertools.combinations(range(len(tmp_index)), 2):
            X_1 = np.c_[X, (tmp_index[i[0]])*np.c_[np.ones(X_size)]]
            X_2 = np.c_[X, (tmp_index[i[1]])*np.c_[np.ones(X_size)]]
            cov = self.GPmodel.diag_covariance_between_points(X_1, X_2).T
            cov_each_x[:,:,i[0],i[1]] = cov
            cov_each_x[:,:,i[1],i[0]] = cov

        #####################################################################################################
        high_var = cov_each_x[:,:,-1, -1][:,:,None,None]
        gamma = (self.maximums[:,None, None, None] - self.high_mean[None, :, None, None]) / np.sqrt(high_var)
        Z = norm.cdf(gamma)
        F = norm.pdf(gamma) / Z

        infogain = (self.acq_m+1) / 2. - np.log(Z).reshape(self.sampling_num, X_size)
        #####################################################################################################
        cov_between_highest = cov_each_x[:,:,-1,:self.acq_m+1].reshape(1, X_size, self.acq_m+1, 1)
        # d = F * cov_between_highest.reshape(1, X_size, self.acq_m+1, 1) / np.sqrt(high_var)
        truncated_cov_plus_dd = cov_each_x[:,:,:self.acq_m+1,:self.acq_m+1] - F * gamma / high_var * np.einsum( '...ij,...jk->...ik', cov_between_highest, cov_between_highest.reshape(1, X_size, 1, self.acq_m+1))
        lower_cov_inv = np.linalg.inv(cov_each_x[:,:,:self.acq_m+1,:self.acq_m+1])
        tmp_product = np.einsum('...ij,...jk->...ik', lower_cov_inv, truncated_cov_plus_dd)

        infogain -= np.einsum('ijkk->ij', tmp_product) / 2.
        #####################################################################################################
        # max sample num \times candidate x num \times EntropyApproxNum
        conditional_mean_mean = self.high_mean.reshape(1,X_size,1)
        conditional_mean_var = np.einsum('...ij,...jk->...ik', cov_between_highest.reshape(1, X_size, 1, self.acq_m+1), np.einsum('...ij,...jk->...ik', lower_cov_inv, cov_between_highest)).reshape(1,X_size,1)
        conditional_mean_std = np.sqrt(conditional_mean_var)

        conditional_high_std = np.sqrt(high_var.reshape(1,X_size,1) - conditional_mean_var)
        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = self.maximums[:,None, None]
        cdf_width = np.abs(self.cons * conditional_high_std)

        pdf_central = conditional_mean_mean * np.ones(shape=(self.sampling_num, 1, 1))
        pdf_width = self.cons * conditional_mean_std * np.ones(shape=(self.sampling_num, 1, 1))

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        remained_index = np.any(tmp_index, axis=0).ravel()
        # if integration region is not empty
        if np.any(remained_index):
            points_min = points_min[:,remained_index]
            points_max = points_max[:,remained_index]

            # if remained x has some empty integration region w.r.t. max-value sample, fix to have non-empty region
            tmp_index = ~tmp_index[:,remained_index]
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)/2. + (points_max-points_min)/2.*self.gauss_legendre_points

            conditional_high_cdf = norm.cdf((self.maximums[:,None, None] - integrate_points) / conditional_high_std[:,remained_index,:])
            numerical_integration_term = np.sum( self.gauss_legendre_weights * norm.pdf(integrate_points, loc=conditional_mean_mean[:,remained_index,:], scale=conditional_mean_std[:,remained_index,:]) * conditional_high_cdf * np.log(conditional_high_cdf), axis=2) * (points_max-points_min)[:,:,0] / 2. / Z.reshape(self.sampling_num, X_size)[:,remained_index]
            infogain[:,remained_index] = infogain[:,remained_index] + numerical_integration_term

        infogain = np.mean(infogain, axis=0)
        return - infogain

    def acq_low_onepoint(self,x):
        x = np.atleast_2d(x)
        trace_x = np.c_[np.matlib.repmat(x, self.acq_m+1, 1), np.c_[self.fidelity_features[:self.acq_m+1]]]
        high_x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], 1, 1)]

        mean, var = self.GPmodel.predict_noiseless(np.r_[trace_x, high_x], full_cov=True)

        high_gamma = ((self.maximums - mean.ravel()[-1]) / np.sqrt(var[-1, -1]))[:, None, None]
        Z = norm.cdf(high_gamma)
        F = norm.pdf(high_gamma) / Z

        # truncated_mean = mean[None, :, :] - F * np.c_[var[-1, :]][None, :, :] / np.sqrt(var[-1, -1])
        d = F * np.c_[var[-1, :]][None, :, :] / np.sqrt(var[-1, -1])
        truncated_var = var[None, :, :] - F * (high_gamma + F) / var[-1, -1] * (np.c_[var[-1, :]].dot(np.c_[var[-1, :]].T))[None, :, :]

        lower_var_inv = np.linalg.inv(var[:-1,:-1])
        d = d[:,:-1,:]
        truncated_var = truncated_var[:,:-1,:-1]
        return -1*self.infogain(mean.ravel()[-1], var[-1, -1], Z, var[-1, :-1], d, truncated_var, lower_var_inv)


    def infogain(self, high_mean, high_var, Z, cov_high_lower, d, truncated_var, lower_var_inv):
        trace_term = np.einsum('ijk,ikl->ijl', d, d.reshape((self.sampling_num, 1, np.shape(d)[1])))
        trace_term = trace_term + truncated_var
        trace_term = np.einsum('ij,...jl->...il', lower_var_inv, trace_term)
        trace_term = np.einsum('ijj->i', trace_term) / 2.

        infogain = np.shape(d)[1] / 2. - np.mean(trace_term+np.log(Z.ravel()))

        conditional_mean_mean = high_mean
        conditional_mean_var = cov_high_lower.dot(lower_var_inv).dot(np.c_[cov_high_lower])
        conditional_mean_std = np.sqrt(conditional_mean_var).ravel()

        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = np.c_[self.maximums].T
        cdf_width = np.abs(self.cons* np.sqrt(high_var - conditional_mean_var))
        # cdf_central[np.isnan(cdf_central)] = 0
        # cdf_width[np.isnan(cdf_width)] = np.inf

        pdf_central = conditional_mean_mean * np.atleast_2d(np.ones(self.sampling_num))
        pdf_width = self.cons * conditional_mean_std * np.atleast_2d(np.ones(self.sampling_num))

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        remained_index = np.any(tmp_index, axis=1)
        # if integration region is not empty
        if np.any(remained_index):
            points_min = points_min[remained_index]
            points_max = points_max[remained_index]

            # if remained x has some empty integration region w.r.t. max-value sample, fix to have non-empty region
            tmp_index = ~tmp_index[remained_index]
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)[:,:,None]/2. + (points_max-points_min)[:,:,None]/2.*self.gauss_legendre_points

            conditional_high_cdf = norm.cdf((np.c_[self.maximums] - integrate_points) / np.sqrt(high_var - conditional_mean_var))
            numerical_integration_term = np.sum( self.gauss_legendre_weights * norm.pdf(integrate_points, loc=conditional_mean_mean, scale=conditional_mean_std) * conditional_high_cdf * np.log(conditional_high_cdf), axis=2) * (points_max-points_min)/2.
            numerical_integration_term = np.mean(numerical_integration_term / np.c_[Z.ravel()].T, axis=1)
            numerical_integration_term[np.abs(numerical_integration_term) < 1e-10] = 0
            infogain = infogain + numerical_integration_term[0]

        return infogain

    def acq_for_freeze(self,x):
        freeze_m = int(x[-1])
        x = np.atleast_2d(x[:-1])
        trace_x = np.c_[np.matlib.repmat(x, self.M - freeze_m - 1, 1), np.c_[self.fidelity_features[freeze_m+1:]]]

        mean, var = self.GPmodel.predict(trace_x, full_cov=True)
        high_gamma = ((self.maximums - mean.ravel()[-1]) / np.sqrt(var[-1, -1]))[:, None, None]
        Z = norm.cdf(high_gamma)
        F = norm.pdf(high_gamma) / Z

        d = F * np.c_[var[-1, :]][None, :, :] / np.sqrt(var[-1, -1])
        truncated_var = var[None, :, :] - F * (high_gamma + F) / var[-1, -1] * (np.c_[var[-1, :]].dot(np.c_[var[-1, :]].T))[None, :, :]

        acq_list=[]
        # until (M-1) fidelity
        for i in range(self.M - freeze_m - 2):
            index = np.arange(i+1)
            lower_var_inv = np.linalg.inv(var[np.ix_(index, index)])
            d_i = d[:,index,:]
            truncated_var_i = truncated_var[np.ix_(np.arange(self.sampling_num), index, index)]
            infogain = self.infogain(mean.ravel()[-1], var[-1, -1], Z, var[-1, index], d_i, truncated_var_i, lower_var_inv)
            acq_list.append(infogain / (self.cost[freeze_m+i+1] - self.cost[freeze_m]))

        high_acq = self.acq_high(x) / (self.cost[-1] - self.cost[freeze_m])
        acq_list.append(-1*high_acq[0])
        return freeze_m + np.argmax(acq_list) + 1, -1 * np.max(acq_list)


class MultiFidelityMaxvalueEntropySearch(TA_MFBO):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, sampling_num=10, EntropyApproxNum=50, sampling_method='RFM', GPmodel=None, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
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

    def next_input_pool(self, X_list):
        self.min_acq = list()
        self.min_x = list()
        evaluate_X = list()

        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m][0]

            if X_list[m].size != 0:
                upper_bound = self._upper_bound(X_list[m])
                if np.size(X_list[m][(upper_bound >= max_lower_bounds).ravel()]) > 0:
                    evaluate_X.append(X_list[m][(upper_bound >= max_lower_bounds).ravel()])
                else:
                    evaluate_X.append(X_list[m])

                if m < self.M-1:
                    acq = self.acq_low(evaluate_X[-1])
                    min_idx = np.argmin(acq)
                    self.min_x.append(evaluate_X[-1][min_idx])
                    self.min_acq.append(acq[min_idx])
                else:
                    acq = self.acq_high(evaluate_X[-1])
                    self.min_x.append(evaluate_X[-1][min_idx])
                    self.min_acq.append(acq[min_idx])
            else:
                evaluate_X.append([])
                self.min_x.append([])
                self.min_acq.append(- np.inf)

        acq_list = (np.hstack(self.min_acq) / np.array(self.cost)).tolist()
        fidelity_list = np.arange(self.M).tolist()

        temp_index = np.argmin(acq_list)
        new_fidelity = fidelity_list[temp_index]

        x = self.min_x[new_fidelity]
        new_input = np.c_[np.matlib.repmat(x, new_fidelity+1, 1), np.c_[self.fidelity_features[:new_fidelity+1]]]
        start_fidelity = -1
        remain_cost = self.cost[:new_fidelity+1]

        if new_fidelity < self.M-1:
            self.freeze_inputs = np.r_[self.freeze_inputs, np.c_[np.atleast_2d(self.min_x[new_fidelity]), new_fidelity]]
        # if temp_index < self.M:
        # else:
        #     start_fidelity = int(self.freeze_inputs[temp_index - self.M][-1])

        #     x = self.freeze_inputs[temp_index - self.M, :-1]
        #     new_input = np.c_[np.matlib.repmat(x, new_fidelity - start_fidelity, 1), np.c_[self.fidelity_features[start_fidelity+1:new_fidelity+1]]]

        #     remain_cost = self.cost[start_fidelity+1 : new_fidelity+1] - self.cost[start_fidelity]

        #     if new_fidelity==self.M-1:
        #         self.freeze_inputs = np.delete(self.freeze_inputs, temp_index - self.M, 0)
        #     else:
        #         self.freeze_inputs[temp_index - self.M, -1] = new_fidelity

        for m in range(self.M):
            if start_fidelity < m and m <= new_fidelity:
                X_list[m] = X_list[m][~np.all(X_list[m]==x, axis=1)]
        return new_input, remain_cost, X_list

    def max_value_sampling(self):
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)

    def prepare_new_acq(self):
        self.max_value_sampling()

    def update(self, add_X_list, add_Y_list, optimize=False, prepare_new_acq=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        if prepare_new_acq:
            start = time.time()
            self.prepare_new_acq()
            self.preprocessing_time = time.time() - start
        else:
            self.preprocessing_time = 0
        print('sampled maximums:', self.maximums)

    def acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()

    # using tensor whose size is (number of candidate points × number of max-vale sampling × number for approximating G-L quadrature)
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
        x = np.atleast_2d(np.c_[np.matlib.repmat(x, 2, 1), np.array([self.acq_m, self.fidelity_features[-1][0]])])

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

    def acq_for_freeze(self,x):
        freeze_m = int(x[-1])
        x = np.atleast_2d(x[:-1])
        acq_list=[]
        # until (M-1) fidelity
        for i in range(self.M - freeze_m - 2):
            self.acq_m = i + 1 + freeze_m
            acq_list.append(-1 * self.acq_low_onepoint(x)[0] / (self.cost[self.acq_m] - self.cost[freeze_m]))

        high_acq = self.acq_high(x) / (self.cost[-1] - self.cost[freeze_m])
        acq_list.append(-1*high_acq[0])
        return freeze_m + np.argmax(acq_list) + 1, -1 * np.max(acq_list)


class TA_KG(TA_MFBO):
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, ExpectationApproxNum = 100, model_name='MFGP', fidelity_features = None, pool_X=None, optimize=True):
        super().__init__(X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        self.ExpectationApproxNum = ExpectationApproxNum

        if pool_X is None:
            print("TA_KG for continuous input is not implemented")
            exit()
        self.pool_X = pool_X

        start = time.time()
        self.posterior_mean, _ = self.GPmodel.predict( np.c_[self.pool_X, (self.M-1)*np.c_[np.ones(np.shape(self.pool_X)[0])]], full_cov=False)
        self.posterior_mean_max = np.max(self.posterior_mean)
        self.Z = np.atleast_2d(np.random.randn(self.ExpectationApproxNum))
        self.preprocessing_time = time.time() - start

    def prepare_new_acq(self):
        self.posterior_mean, _ = self.GPmodel.predict( np.c_[self.pool_X, (self.M-1)*np.c_[np.ones(np.shape(self.pool_X)[0])]], full_cov=False)
        self.posterior_mean_max = np.max(self.posterior_mean)
        self.Z = np.atleast_2d(np.random.randn(self.ExpectationApproxNum))

    def update(self, add_X_list, add_Y_list, optimize=False, prepare_new_acq=False):
        super().update(add_X_list, add_Y_list, optimize=optimize)
        if prepare_new_acq:
            start = time.time()
            self.prepare_new_acq()
            self.preprocessing_time = time.time() - start
        else:
            self.preprocessing_time = 0

    def acq_low(self, X):
        return self.acq(X)

    def acq_high(self, X):
        return self.acq(X)

    # Monte Carlo sample num \times candidate x num \times M \times M
    def acq(self, X):
        X = np.atleast_2d(X)
        X_size = np.shape(X)[0]
        cov_each_x = np.empty(shape=(1,X_size,(self.acq_m+1),(self.acq_m+1)))
        tmp_index = list(range(self.acq_m+1))

        X_multi_fidelity = np.vstack( [ np.c_[X, (m)*np.c_[np.ones(X_size)]] for m in tmp_index ] )
        _, var = self.GPmodel.predict(X_multi_fidelity)

        for i, m in enumerate(tmp_index):
            cov_each_x[:,:,i,i] = var[i*X_size:(i+1)*X_size].T

        for i in itertools.combinations(range(len(tmp_index)), 2):
            X_1 = np.c_[X, (tmp_index[i[0]])*np.c_[np.ones(X_size)]]
            X_2 = np.c_[X, (tmp_index[i[1]])*np.c_[np.ones(X_size)]]
            cov = self.GPmodel.diag_covariance_between_points(X_1, X_2).T
            cov_each_x[:,:,i[0],i[1]] = cov
            cov_each_x[:,:,i[1],i[0]] = cov

        cov_each_x_chol = np.linalg.cholesky(cov_each_x)
        cov_each_x_chol_inv = np.linalg.inv(cov_each_x_chol)
        cov_each_x_chol_inv_T = cov_each_x_chol_inv.transpose((0,1,3,2))

        cov_allX_x = self.GPmodel.posterior_covariance_between_points(np.c_[self.pool_X, (self.M-1)*np.c_[np.ones(np.shape(self.pool_X)[0])]], X_multi_fidelity)
        cov_allX_x = np.array([ cov_allX_x[:,i*X_size:(i+1)*X_size] for i in range(len(tmp_index))]).transpose((2,1,0))[None,:,:,:]

        tmp_product = np.sqrt( np.sum( np.einsum('...ij,...jk->...ik', cov_allX_x, cov_each_x_chol_inv_T)**2, axis=3))

        updated_posterior_mean = self.posterior_mean.reshape(1,1,np.shape(self.pool_X)[0]) + self.Z.reshape(self.ExpectationApproxNum, 1, 1) * tmp_product
        max_updated_posterior_mean = np.max(updated_posterior_mean, axis=2)
        acq_value = np.mean(max_updated_posterior_mean, axis=0) - self.posterior_mean_max
        return -acq_value


    def acq_naive(self, X):
        X = np.atleast_2d(X)
        acq = list()
        self.start_fidelity = 0
        for i in range(np.shape(X)[0]):
            acq.append(self.acq_onepoint(X[i]))
        return np.array(acq)

    def acq_onepoint(self, x):
        x = np.c_[ np.matlib.repmat(np.atleast_2d(x), self.acq_m - self.start_fidelity + 1, 1), np.c_[ np.arange(self.start_fidelity, self.acq_m+1)]]
        _, cov = self.GPmodel.predict(x, full_cov=True)
        cov_chol = np.linalg.cholesky(cov)
        cov_chol_inv = np.linalg.inv(cov_chol)
        cov_allX_x = self.GPmodel.posterior_covariance_between_points(np.c_[self.pool_X, (self.M-1)*np.c_[np.ones(np.shape(self.pool_X)[0])]], x)

        updated_posterior_mean = self.posterior_mean + np.c_[np.sqrt(np.sum(cov_allX_x.dot(cov_chol_inv.T)**2, axis=1))] * self.Z
        max_updated_posterior_mean = np.max(updated_posterior_mean, axis=0)
        return - (np.mean(max_updated_posterior_mean) - self.posterior_mean_max)


    def acq_for_freeze(self,x):
        freeze_m = int(x[-1])
        x = np.atleast_2d(x[:-1])
        acq_list=[]
        # until (M-1) fidelity
        self.start_fidelity = freeze_m + 1
        for i in range(self.M - freeze_m - 1):
            self.acq_m = i + 1 + freeze_m
            acq_list.append(-1 * self.acq_onepoint(x) / (self.cost[self.acq_m] - self.cost[freeze_m]))

        return freeze_m + np.argmax(acq_list) + 1, -1 * np.max(acq_list)
