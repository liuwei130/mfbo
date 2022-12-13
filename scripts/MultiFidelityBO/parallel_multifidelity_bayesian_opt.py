# -*- coding: utf-8 -*-

import os
import sys
from abc import ABCMeta, abstractmethod
import time

import numpy as np
import numpy.matlib
from scipy.stats import norm
from scipy import optimize


from ..myutils import myutils as utils
from ..myutils.BO_core import MFBO_core
from .multifidelity_bayesian_opt import MultiFidelityMaxvalueEntropySearch


class ParallelMFBO(MFBO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features=None, selected_inputs=None, num_worker=1, optimize=True):
        MFBO_core.__init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=GPmodel, model_name=model_name, fidelity_features=fidelity_features, optimize=optimize)
        self.selected_inputs = selected_inputs
        self.num_worker = num_worker

    @abstractmethod
    def parallel_acq_high(self, x):
        pass

    @abstractmethod
    def parallel_acq_low(self, x):
        pass

    @abstractmethod
    def parallel_acq_low_onepoint(self, x):
        pass

    # parallel
    @abstractmethod
    def preparation(self):
        pass

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

        if self.selected_inputs is not None:
            tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            tmp_num_worker = self.num_worker

        for q in range(tmp_num_worker):
            print(str(q)+'-th selection---------------------')
            if self.selected_inputs is None:
                new_input = self.first_next_input(x0s)
                self.selected_inputs = new_input
            else:
                self.preparation()
                new_input = self.parallel_next_input(x0s)

                self.selected_inputs = np.r_[self.selected_inputs, new_input]
        return self.selected_inputs[-tmp_num_worker:, :]

    def first_next_input(self, x0s):
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

    def parallel_next_input(self, x0s):
        f_min_list = list()
        x_min_list = list()

        for m in range(self.M-1):
            self.acq_m = self.fidelity_features[m]
            x_min, f_min = utils.minimize(self.parallel_acq_low_onepoint, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            f_min_list.append(f_min)
            x_min_list.append(x_min)

        x_min, f_min = utils.minimize(self.parallel_acq_high, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        f_min_list.append(f_min)
        x_min_list.append(x_min)

        print('optimized_acquisition function values:', np.array(f_min_list).ravel())
        new_fidelity = np.argmin(np.array(f_min_list).ravel() / np.array(self.cost))
        new_input = x_min_list[new_fidelity]
        return np.atleast_2d(np.r_[new_input, new_fidelity])



    def next_input_pool(self, X_list):
        evaluate_X = list()
        observed_lower_bound = self._lower_bound(self.unique_X)
        max_lower_bounds = np.max(observed_lower_bound)
        for m in range(self.M):
            if X_list[m].size != 0:
                upper_bound = self._upper_bound(X_list[m])
                evaluate_X.append(X_list[m][(upper_bound >= max_lower_bounds).ravel()])
            else:
                evaluate_X.append([])

        if self.selected_inputs is not None:
            tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            tmp_num_worker = self.num_worker

        for q in range(tmp_num_worker):
            print(str(q)+'-th selection---------------------')
            if self.selected_inputs is None:
                new_input, X_list, evaluate_X = self.first_next_input_pool(X_list, evaluate_X)
                self.selected_inputs = new_input
            else:
                self.preparation()
                new_input, X_list, evaluate_X = self.parallel_next_input_pool(X_list, evaluate_X)
                self.selected_inputs = np.r_[self.selected_inputs, new_input]

        return self.selected_inputs[-tmp_num_worker:, :], X_list

    def first_next_input_pool(self, X_list, evaluate_X):
        self.max_acq = list()
        self.max_acq_idx = list()
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            if evaluate_X[m].size != 0:
                if m < self.M-1:
                    acq = -1*self.acq_low(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
                else:
                    acq = -1*self.acq_high(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append(- np.inf)

        print('optimized_acquisition function values:', self.max_acq)
        self.fidelity_index = np.argmax(np.array(self.max_acq) / np.array(self.cost))
        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        evaluate_X[self.fidelity_index] = evaluate_X[self.fidelity_index][~np.all(evaluate_X[self.fidelity_index]==new_input, axis=1)]
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list, evaluate_X

    def parallel_next_input_pool(self, X_list, evaluate_X):
        self.max_acq = list()
        self.max_acq_idx = list()
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            if evaluate_X[m].size != 0:
                if m < self.M-1:
                    acq = -1*self.parallel_acq_low(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
                else:
                    acq = -1*self.parallel_acq_high(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append(- np.inf)

        print('optimized_acquisition function values:', self.max_acq)
        self.fidelity_index = np.argmax(np.array(self.max_acq) / np.array(self.cost))
        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        evaluate_X[self.fidelity_index] = evaluate_X[self.fidelity_index][~np.all(evaluate_X[self.fidelity_index]==new_input, axis=1)]
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list, evaluate_X



class ParallelMultiFidelityMaxvalueEntropySearch(ParallelMFBO, MultiFidelityMaxvalueEntropySearch):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, selected_inputs=None, num_worker=1, sampling_num=10, EntropyApproxNum=50, sampling_method='RFM', model_name='MFGP', GPmodel=None, fidelity_features = None, pool_X=None, optimize=True):
        ParallelMFBO.__init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, model_name=model_name, fidelity_features = fidelity_features, optimize=optimize)
        self.sampling_num = sampling_num
        self.EntropyApproxNum = EntropyApproxNum
        self.cons = 6.
        self.logsumexp_const = 1e3
        self.sampling_method = sampling_method
        self.pool_X = pool_X

        self.gauss_legendre_points, self.gauss_legendre_weights = np.polynomial.legendre.leggauss(EntropyApproxNum)
        self.gauss_legendre_points = self.gauss_legendre_points[None, None, :]
        self.gauss_legendre_weights = self.gauss_legendre_weights[None, None, :]

        start = time.time()
        if sampling_method == 'RFM':
            if model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(pool_X=pool_X)
            elif model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(pool_X=pool_X)
            else:
                print('RFM of input model_name did not implemented')
                exit(1)
        elif sampling_method == 'Gumbel':
            print('Gumbel sampling for Parallel MF-MES is not implemented')
            exit(1)

        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)

    def update(self, add_X_list, add_Y_list, optimize=False):
        if self.model_name=='MFGP':
            fidelity_features = np.hstack([i*np.ones(np.shape(add_X_list[i])[0]) for i in range(self.M) if np.size(add_X_list[i] > 0)])
        elif self.model_name=='MTGP':
            fidelity_features = np.vstack([np.matlib.repmat(self.fidelity_features.ravel()[i], np.size(add_Y_list[i]), 1) for i in range(len(add_Y_list))])
        add_X = np.c_[np.vstack([X for X in add_X_list if np.size(X) > 0]), fidelity_features]
        add_Y = np.vstack([Y for Y in add_Y_list if np.size(Y) > 0])


        self.eval_num = [self.eval_num[m] + np.size(add_Y_list[m]) for m in range(len(add_Y_list))]
        self.GPmodel.add_XY(add_X, add_Y)
        if optimize:
            self.GPmodel.my_optimize()

        self.unique_X = np.unique(self.GPmodel.X[:,:-1], axis=0)
        if np.size(add_Y_list[-1]) > 0:
            self.y_max = np.max([self.y_max, np.max(add_Y_list[-1])])

        for i in range(np.shape(add_X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==add_X[i], axis=1) == True)[0][0], axis=0)
            # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==add_X[i], axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None


        start = time.time()
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            if self.model_name == 'MFGP':
                self.maximums, self.max_inputs = self.sampling_MFRFM(self.pool_X)
            elif self.model_name == 'MTGP':
                self.maximums, self.max_inputs = self.sampling_MTRFM(self.pool_X)
        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)
        print('self selected inputs', self.selected_inputs)

    def preparation(self):
        # self.mean, self.cov = self.GPmodel.predict_noiseless(self.selected_inputs, full_cov=True)
        self.mean, self.cov = self.GPmodel.predict(self.selected_inputs, full_cov=True)

        # Q × sampling_num
        if self.model_name == 'MFGP':
            sampled_selected_outputs = self.sample_path_MFRFM(self.selected_inputs)
        elif self.model_name == 'MTGP':
            sampled_selected_outputs = self.sample_path_MTRFM(self.selected_inputs)
        else:
            print('RFM of input model_name did not implemented')
            exit(1)

        try:
            self.cov_chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as e:
            print('In preparation method,', e)
            self.cov = self.cov + 1e-8 * np.eye(np.shape(self.selected_inputs)[0])
            self.cov_chol = np.linalg.cholesky(self.cov)
        self.alpha = np.linalg.solve(self.cov_chol.T, np.linalg.solve(self.cov_chol, sampled_selected_outputs - self.mean))

        tmp_sampled_selected_outputs =  sampled_selected_outputs + 3 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        temp_max = np.max(np.r_[self.y_max*np.c_[np.ones(self.sampling_num)].T, tmp_sampled_selected_outputs], axis=0)
        correction_index = self.maximums < temp_max
        self.maximums[correction_index] = temp_max[correction_index]


    def parallel_acq_high(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        cov_x_selected = self.GPmodel.posterior_covariance_between_points(x, self.selected_inputs)

        sampled_mean = mean + cov_x_selected.dot(self.alpha)
        v = np.linalg.solve(self.cov_chol, cov_x_selected.T)
        sampled_var = var.ravel() - np.einsum('ij,ji->i', v.T, v)
        sampled_std = np.sqrt(sampled_var)

        maximums = self.maximums[np.newaxis, :]
        normalized_max = (maximums - sampled_mean) / sampled_std[:, np.newaxis]
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()


    def parallel_acq_low_onepoint(self, x):
        x = np.atleast_2d(np.c_[np.matlib.repmat(x, 2, 1), np.array([self.acq_m, self.fidelity_features[-1]])])

        mean, cov = self.GPmodel.predict_noiseless(x, full_cov=True)
        cov_x_selected = self.GPmodel.posterior_covariance_between_points(x, self.selected_inputs)

        sampled_mean = mean + cov_x_selected.dot(self.alpha)
        v = np.linalg.solve(self.cov_chol, cov_x_selected.T)
        sampled_cov = cov - v.T.dot(v)
        sampled_std = np.sqrt(np.diag(sampled_cov))
        sampled_rho = sampled_cov[1,0] / (sampled_std[0] * sampled_std[1])
        conditional_high_std = np.sqrt(sampled_cov[1,1] - sampled_cov[1,0]**2 / sampled_cov[0,0])

        normalized_max = (self.maximums[None, :] - sampled_mean[1]) / sampled_std[1]
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        acquisition_values = np.array([np.mean(sampled_rho**2 * (normalized_max * high_pdf) / (2*high_cdf) - np.log(high_cdf))])

        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = np.atleast_2d(sampled_mean[0]) + sampled_cov[0,0] / sampled_cov[1,0] * (self.maximums[None, :] - np.atleast_2d(sampled_mean[1]))
        cdf_width = np.abs(self.cons*sampled_cov[0,0] / sampled_cov[1,0] * conditional_high_std)

        cdf_central[np.isnan(cdf_central)] = 0
        if np.isnan(cdf_width):
            cdf_width = np.inf

        pdf_central = np.atleast_2d(sampled_mean[0])
        pdf_width = self.cons * sampled_std[0]

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        if np.any(tmp_index):
            tmp_index = ~tmp_index
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)[:,:,None]/2. + (points_max-points_min)[:,:,None]/2.*self.gauss_legendre_points
            # until here

            low_pdf = norm.pdf(integrate_points, loc=sampled_mean[0][None,:,None], scale=sampled_std[0])
            conditional_high_mean = sampled_mean[1][None,:,None] + sampled_cov[1,0] / sampled_cov[0,0] * (integrate_points - sampled_mean[0][None,:,None])
            conditional_high_cdf = norm.cdf((self.maximums[np.newaxis, :, np.newaxis] - conditional_high_mean) / conditional_high_std)
            # avoid the numerical error with log0 (By substituting 1, integrand = 0 log 0 = log1 = 0)
            conditional_high_cdf[conditional_high_cdf <= 0] = 1

            numerical_integration_term = np.sum( self.gauss_legendre_weights*low_pdf*conditional_high_cdf * np.log(conditional_high_cdf), axis=2) / high_cdf * (points_max-points_min)/2.
            numerical_integration_term = np.mean(numerical_integration_term, axis=1).ravel()
            numerical_integration_term[np.abs(numerical_integration_term) < 1e-10] = 0
            acquisition_values = acquisition_values + numerical_integration_term

        return - acquisition_values


    def parallel_acq_low(self, x):
        x = np.atleast_2d(x)
        x_size = np.shape(x)[0]
        low_x = np.c_[x, np.matlib.repmat(self.acq_m, x_size, 1)]
        high_x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], x_size, 1)]

        mean, var = self.GPmodel.predict(np.r_[low_x, high_x])
        cov_low_high = self.GPmodel.diag_covariance_between_points(low_x, high_x)
        cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.r_[low_x, high_x], self.selected_inputs)

        sampled_mean = mean + cov_x_selected.dot(self.alpha)
        v = np.linalg.solve(self.cov_chol, cov_x_selected.T)
        sampled_var = var.ravel() - np.einsum('ij,ji->i', v.T, v)
        # compute the covariances between [x, m] amd [x, M]
        sampled_cov_low_high = np.c_[cov_low_high.ravel() - np.einsum('ij,ji->i', v.T[x_size:, :], v[:, :x_size]).ravel()]

        sampled_low_mean = sampled_mean[:x_size, :]
        sampled_low_var = np.c_[sampled_var[:x_size]]
        sampled_low_std = np.sqrt(sampled_low_var)

        sampled_high_mean = sampled_mean[x_size:, :]
        sampled_high_var = np.c_[sampled_var[x_size:]]
        sampled_high_std = np.sqrt(sampled_high_var)

        sampled_rho = sampled_cov_low_high / (sampled_low_std * sampled_high_std)
        normalized_max = (self.maximums - sampled_high_mean) / sampled_high_std
        high_pdf = norm.pdf(normalized_max)
        high_cdf = norm.cdf(normalized_max)
        acquisition_values = np.mean(sampled_rho**2 * (normalized_max * high_pdf) / (2*high_cdf) - np.log(high_cdf), axis=1)
        conditional_high_std = np.sqrt(sampled_high_var - sampled_cov_low_high**2 / sampled_low_var)

        # make the integrated range of Gauss-Legendre quadrature
        cdf_central = sampled_low_mean + sampled_low_var / sampled_cov_low_high * (self.maximums[None, :] - sampled_high_mean)
        cdf_width = np.abs(self.cons*sampled_low_var / sampled_cov_low_high * conditional_high_std)
        cdf_central[np.isnan(cdf_central)] = 0
        cdf_width[np.isnan(cdf_width)] = np.inf

        pdf_central = sampled_low_mean
        pdf_width = self.cons * sampled_low_std

        points_min = np.logaddexp(self.logsumexp_const*(cdf_central-cdf_width), self.logsumexp_const*(pdf_central-pdf_width)) / self.logsumexp_const
        points_max = - np.logaddexp(-self.logsumexp_const*(cdf_central+cdf_width), -self.logsumexp_const*(pdf_central+pdf_width)) / self.logsumexp_const

        tmp_index = points_max > points_min
        remained_index = np.any(tmp_index, axis=1)
        if np.any(remained_index):
            points_min = points_min[remained_index]
            points_max = points_max[remained_index]
            tmp_index = ~tmp_index[remained_index]
            if np.any(tmp_index):
                points_max[tmp_index] = points_min[tmp_index] + 1e-2*self.GPmodel.std
            integrate_points = (points_max+points_min)[:,:,None]/2. + (points_max-points_min)[:,:,None]/2.*self.gauss_legendre_points
            # until here

            low_pdf = norm.pdf(integrate_points, loc=sampled_low_mean[remained_index, :, np.newaxis], scale=sampled_low_std[remained_index, :, np.newaxis])
            conditional_high_mean = sampled_high_mean[remained_index, :, np.newaxis] + sampled_cov_low_high[remained_index, :, np.newaxis] / sampled_low_var[remained_index, :, np.newaxis] * (integrate_points - sampled_low_mean[remained_index, :, np.newaxis])
            conditional_high_cdf = norm.cdf((self.maximums[np.newaxis, :, np.newaxis] - conditional_high_mean) / conditional_high_std[remained_index, :, np.newaxis])
            # avoid the numerical error with log0 (By substituting 1, integrand = 0 log 0 = log1 = 0)
            conditional_high_cdf[conditional_high_cdf <= 0] = 1

            numerical_integration_term = np.sum( self.gauss_legendre_weights*low_pdf*conditional_high_cdf * np.log(conditional_high_cdf), axis=2) / high_cdf[remained_index] * (points_max-points_min)/2.
            numerical_integration_term = np.mean(numerical_integration_term, axis=1).ravel()
            numerical_integration_term[np.abs(numerical_integration_term) < 1e-10] = 0
            acquisition_values[remained_index] = acquisition_values[remained_index] + numerical_integration_term
        return -acquisition_values




class SyncMultiFidelityMaxvalueEntropySearch(ParallelMultiFidelityMaxvalueEntropySearch):
    # sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, selected_inputs=None, num_worker=1, sampling_num=10, EntropyApproxNum=50, sampling_method='RFM', model_name='MFGP', GPmodel=None, fidelity_features = None, pool_X=None, optimize=True):
        ParallelMultiFidelityMaxvalueEntropySearch.__init__(self, X_list=X_list, Y_list=Y_list, eval_num=eval_num, bounds=bounds, kernel_bounds=kernel_bounds, M=M, cost=cost, selected_inputs=selected_inputs, num_worker=num_worker, sampling_num=sampling_num, EntropyApproxNum=EntropyApproxNum, sampling_method=sampling_method, model_name=model_name, GPmodel=GPmodel, fidelity_features = fidelity_features, pool_X=pool_X, optimize=optimize)


    def first_next_input(self, x0s):
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

        self.cumulative_gain = f_min_list[new_fidelity]
        self.cumulative_cost = self.cost[new_fidelity]

        new_input = x_min_list[new_fidelity]
        return np.atleast_2d(np.r_[new_input, new_fidelity])

    def parallel_next_input(self, x0s):
        f_min_list = list()
        x_min_list = list()

        for m in range(self.M-1):
            self.acq_m = self.fidelity_features[m]
            x_min, f_min = utils.minimize(self.parallel_acq_low_onepoint, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
            f_min_list.append(f_min)
            x_min_list.append(x_min)

        x_min, f_min = utils.minimize(self.parallel_acq_high, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        f_min_list.append(f_min)
        x_min_list.append(x_min)

        print('optimized_acquisition function values:', np.array(f_min_list).ravel())
        new_fidelity = np.argmin( (np.array(f_min_list).ravel() + self.cumulative_gain) / (np.array(self.cost) + self.cumulative_cost))

        self.cumulative_gain += f_min_list[new_fidelity]
        self.cumulative_cost += self.cost[new_fidelity]

        new_input = x_min_list[new_fidelity]
        return np.atleast_2d(np.r_[new_input, new_fidelity])

    def first_next_input_pool(self, X_list, evaluate_X):
        self.max_acq = list()
        self.max_acq_idx = list()
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            if evaluate_X[m].size != 0:
                if m < self.M-1:
                    acq = -1*self.acq_low(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
                else:
                    acq = -1*self.acq_high(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append([- np.inf])

        print('optimized_acquisition function values:', self.max_acq)
        self.fidelity_index = np.argmax(np.array(self.max_acq) / np.array(self.cost))

        self.cumulative_gain = self.max_acq[self.fidelity_index]
        self.cumulative_cost = self.cost[self.fidelity_index]

        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        evaluate_X[self.fidelity_index] = evaluate_X[self.fidelity_index][~np.all(evaluate_X[self.fidelity_index]==new_input, axis=1)]
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list, evaluate_X

    def parallel_next_input_pool(self, X_list, evaluate_X):
        self.max_acq = list()
        self.max_acq_idx = list()
        for m in range(self.M):
            self.acq_m = self.fidelity_features[m]
            if evaluate_X[m].size != 0:
                if m < self.M-1:
                    acq = -1*self.parallel_acq_low(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
                else:
                    acq = -1*self.parallel_acq_high(evaluate_X[m])
                    self.max_acq_idx.append(np.argmax(acq))
                    self.max_acq.append(acq[self.max_acq_idx[m]])
            else:
                evaluate_X.append([])
                self.max_acq_idx.append([])
                self.max_acq.append([- np.inf])

        print('optimized_acquisition function values:', self.max_acq)

        self.fidelity_index = np.argmax( (np.array(self.max_acq).ravel() + self.cumulative_gain) / (np.array(self.cost) + self.cumulative_cost))

        self.cumulative_gain += self.max_acq[self.fidelity_index]
        self.cumulative_cost += self.cost[self.fidelity_index]

        max_index = self.max_acq_idx[self.fidelity_index]
        new_input = evaluate_X[self.fidelity_index][max_index]
        X_list[self.fidelity_index] = X_list[self.fidelity_index][~np.all(X_list[self.fidelity_index]==new_input, axis=1)]
        evaluate_X[self.fidelity_index] = evaluate_X[self.fidelity_index][~np.all(evaluate_X[self.fidelity_index]==new_input, axis=1)]
        return np.atleast_2d(np.r_[new_input, self.fidelity_index]), X_list, evaluate_X



