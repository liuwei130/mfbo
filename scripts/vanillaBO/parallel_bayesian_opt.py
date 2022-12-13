import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mvn
from scipy.stats import norm
from scipy import special

from ..myutils import myutils as utils
from ..myutils.BO_core import BO_core

from .bayesian_opt import MaxvalueEntropySearch
from .bayesian_opt import ExpectedImprovement

class ParallelBO(BO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, GPmodel=None, optimize=True):
        BO_core.__init__(self, X, Y, bounds, kernel_bounds, GPmodel, optimize=optimize)
        self.selected_inputs = selected_inputs
        self.num_worker = num_worker

    @abstractmethod
    def parallel_acq(self, x):
        pass


    @abstractmethod
    def preparation(self):
        pass

    def next_input(self):
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.GPmodel.minus_predict(self.unique_X)
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
            self.q = q
            print(str(q)+'-th selection---------------------')
            if self.selected_inputs is None:
                x_min, f_min = utils.minimize(self.acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
                self.selected_inputs = np.atleast_2d(x_min)
            else:
                self.preparation()
                x_min, f_min = utils.minimize(self.parallel_acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
                self.selected_inputs = np.r_[self.selected_inputs, np.atleast_2d(x_min)]
            print('optimized acquisition function value:', -1*f_min)
        return self.selected_inputs[-tmp_num_worker:, :]


    def next_input_pool(self, X):
        upper_bound = self._upper_bound(X)
        evaluate_X = X[(upper_bound >= self.y_max).ravel()]

        if self.selected_inputs is not None:
            tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            tmp_num_worker = self.num_worker


        for q in range(tmp_num_worker):
            self.q = q
            print(str(q)+'-th selection---------------------')
            if self.selected_inputs is None:
                self.acquisition_values = self.acq(evaluate_X)
                new_input = np.atleast_2d(evaluate_X[np.argmin(self.acquisition_values)])
                X = X[~np.all(X==new_input, axis=1)]
                evaluate_X = evaluate_X[~np.all(evaluate_X==new_input, axis=1)]
                self.selected_inputs = new_input
            else:
                self.preparation()
                self.acquisition_values = self.parallel_acq(evaluate_X)
                new_input = np.atleast_2d(evaluate_X[np.argmin(self.acquisition_values)])
                X = X[~np.all(X==new_input, axis=1)]
                evaluate_X = evaluate_X[~np.all(evaluate_X==new_input, axis=1)]
                self.selected_inputs = np.r_[self.selected_inputs, new_input]

        return self.selected_inputs[-tmp_num_worker:, :], X




class BatchMaxvalueEntropySearch(ParallelBO, MaxvalueEntropySearch):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, sampling_num=10, sampling_method='RFM', GPmodel=None, pool_X=None, optimize=True):
        if not(selected_inputs is None):
            print('ERROR: In batch BO, selected_inputs must be None')
            exit(1)

        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.pool_X = pool_X


        start = time.time()
        if sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        elif sampling_method == 'Gumbel':
            print('Gumbel pattern is not implemented for parallel MES')
            exit(1)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def update(self, X, Y, optimize=False):
        # This update called update in MES in bayesopt_exp.py (not BO_core)
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None


    def next_input(self):
        return self.next_input_batch()

    def next_input_pool(self):
        print('Batch MES for pool setting is not implemented')
        exit()


    def next_input_batch(self):
        if self.selected_inputs is not None:
            self.tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            self.tmp_num_worker = self.num_worker
        self.lower = -np.inf * np.ones(self.tmp_num_worker)

        num_start = 100
        x0s = utils.lhs(self.input_dim*self.tmp_num_worker, samples=num_start, criterion='maximin') * (np.matlib.repmat(self.bounds[1], 1, self.tmp_num_worker) - np.matlib.repmat(self.bounds[0], 1, self.tmp_num_worker)) + np.matlib.repmat(self.bounds[0], 1, self.tmp_num_worker)
        x_min, f_min = utils.minimize(self.parallel_acq, x0s, self.bounds_list*self.tmp_num_worker)

        print('optimized acquisition function value:', -1*f_min)
        self.selected_inputs = np.reshape(x_min, (self.tmp_num_worker, self.input_dim))
        return self.selected_inputs

    def parallel_acq(self, x):
        x = np.reshape(x, (self.tmp_num_worker, self.input_dim))
        mean, cov = self.GPmodel.predict(x, full_cov=True)
        mean = mean.ravel()
        cov_inv = np.linalg.inv(cov)

        infogain = self.tmp_num_worker / 2.
        for f in self.maximums:
            f_star = f * np.ones(self.tmp_num_worker)
            Z, _ = mvn.mvnun(self.lower, f_star, mean, cov, maxpts=self.tmp_num_worker*1e4, abseps=1e-5, releps=1e-3)
            cov_TN_plus_dd = utils.for_BMES(f_star, mean, cov, Z, self.tmp_num_worker)
            infogain -= (np.trace(cov_inv.dot(cov_TN_plus_dd)) / 2. + np.log(Z)) / (self.sampling_num)
        return - infogain


class BatchMaxvalueEntropySearch_IBO(ParallelBO, MaxvalueEntropySearch):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, sampling_num=10, sampling_method='RFM', GPmodel=None, pool_X=None, optimize=True):
        if not(selected_inputs is None):
            print('ERROR: In batch BO, selected_inputs must be None')
            exit(1)

        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.pool_X = pool_X

        start = time.time()
        if sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        elif sampling_method == 'Gumbel':
            print('Gumbel pattern is not implemented for parallel MES')
            exit(1)
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def update(self, X, Y, optimize=False):
        # This update called update in MES in bayesopt_exp.py (not BO_core)
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None


    def next_input(self):
        return self.next_input_batch()

    def next_input_pool(self):
        print('Batch MES for pool setting is not implemented')
        exit()


    def next_input_batch(self):
        if self.selected_inputs is not None:
            self.tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            self.tmp_num_worker = self.num_worker
        self.lower = -np.inf * np.ones(self.tmp_num_worker)

        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim*self.tmp_num_worker, samples=num_start, criterion='maximin') * (np.matlib.repmat(self.bounds[1], 1, self.tmp_num_worker) - np.matlib.repmat(self.bounds[0], 1, self.tmp_num_worker)) + np.matlib.repmat(self.bounds[0], 1, self.tmp_num_worker)
        x_min, f_min = utils.minimize(self.parallel_acq, x0s, self.bounds_list*self.tmp_num_worker)

        print('optimized acquisition function value:', -1*f_min)
        self.selected_inputs = np.reshape(x_min, (self.tmp_num_worker, self.input_dim))
        return self.selected_inputs

    def parallel_acq(self, x):
        x = np.reshape(x, (self.tmp_num_worker, self.input_dim))
        mean, cov = self.GPmodel.predict(x, full_cov=True)
        mean = mean.ravel()

        infogain = 0
        for f in self.maximums:
            f_star = f * np.ones(self.tmp_num_worker)
            Z, _ = mvn.mvnun(self.lower, f_star, mean, cov, maxpts=self.tmp_num_worker*1e4, abseps=1e-5, releps=1e-3)
            infogain += -np.log(Z)
        return - infogain



class ParallelMaxvalueEntropySearch(ParallelBO, MaxvalueEntropySearch):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, sampling_num=10, sampling_method='RFM', GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.pool_X = pool_X

        start = time.time()
        if sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        elif sampling_method == 'Gumbel':
            print('Gumbel pattern is not implemented for parallel MES')
            exit(1)


        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time = time.time() - start
        print('sampled maximums', self.maximums)


    def update(self, X, Y, optimize=False):
        # This update called update in MES in bayesopt_exp.py (not BO_core)
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None

        start = time.time()
        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time += time.time() - start

    def preparation(self):
        # self.mean, self.cov = self.GPmodel.predict_noiseless(self.selected_inputs, full_cov=True)
        self.mean, self.cov = self.GPmodel.predict(self.selected_inputs, full_cov=True)
        try:
            self.cov_chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as e:
            print('In preparation method,', e)
            self.cov = self.cov + 1e-8 * np.eye(np.shape(self.selected_inputs)[0])
            self.cov_chol = np.linalg.cholesky(self.cov)

        # Q × sampling_num
        sampled_selected_outputs = self.sample_path(self.selected_inputs)
        self.alpha = np.linalg.solve(self.cov_chol.T, np.linalg.solve(self.cov_chol, sampled_selected_outputs - self.mean))

        temp_sampled_selected_outputs = sampled_selected_outputs + 3 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        temp_max = np.max(np.r_[self.y_max*np.c_[np.ones(self.sampling_num)].T, temp_sampled_selected_outputs], axis=0)
        # correction_term = temp_max + 3 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        correction_index = self.maximums < temp_max
        self.maximums[correction_index] = temp_max[correction_index]


    def parallel_acq(self, x):
        x = np.atleast_2d(x)
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


class GaussianProcessUCB_PE(ParallelBO):
    def __init__(self, X, Y, bounds, kernel_bounds, iteration, selected_inputs=None, num_worker=1, GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        start = time.time()
        self.iteration = iteration
        self.beta = np.sqrt(2. * self.input_dim * np.log(2 * iteration))
        def lower_bound(x):
            x = np.atleast_2d(x)
            mean, var = self.GPmodel.predict_noiseless(x)
            return -1 * (mean - self.beta * np.sqrt(var)).ravel()

        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.GPmodel.minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]
        _, f_min = utils.minimize(lower_bound, x0s, self.bounds_list)
        self.func_lower_bound = -1 * f_min
        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time = time.time() - start

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None

        start = time.time()
        self.iteration += 1
        if self.selected_inputs is None:
            self.cov = np.array([[]])
        else:
            _, self.cov = self.GPmodel.predict(self.selected_inputs, full_cov=True)
            self.cov_inv = np.linalg.inv(self.cov)

        self.beta = np.sqrt(2. * self.input_dim * np.log(2 * self.iteration))
        def lower_bound(x):
            x = np.atleast_2d(x)
            mean, var = self.GPmodel.predict_noiseless(x)
            return -1 * (mean - self.beta * np.sqrt(var)).ravel()

        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.GPmodel.minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]
        _, f_min = utils.minimize(lower_bound, x0s, self.bounds_list)
        self.func_lower_bound = -1 * f_min
        if self.selected_inputs is not None:
            self.preparation()
        self.preprocessing_time = time.time() - start

    def preparation(self):
        _, self.cov = self.GPmodel.predict(self.selected_inputs, full_cov=True)
        self.cov_inv = np.linalg.inv(self.cov)

    def parallel_acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        upper_bound = (mean + self.beta*np.sqrt(var)).ravel()
        cov_x_selected = self.GPmodel.posterior_covariance_between_points(x, self.selected_inputs)

        conditional_var = var.ravel() - np.einsum('ij,jk,ki->i', cov_x_selected, self.cov_inv, cov_x_selected.T)
        conditional_var[conditional_var <= 0] = 1e-32
        conditional_var[upper_bound < self.func_lower_bound] = 0
        return -1 * conditional_var

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        return - (mean + self.beta * std).ravel()


class AsynchronousThompsonSampling(ParallelBO):
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        if selected_inputs is None:
            self.sampling_num = num_worker
        else:
            self.sampling_num = num_worker - np.shape(selected_inputs)[0]

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None

    def next_input(self):
        if self.selected_inputs is not None:
            self.sampling_num = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            self.sampling_num = self.num_worker


        _, max_inputs = self.sampling_RFM()
        if self.selected_inputs is None:
            self.selected_inputs = max_inputs
        else:
            self.selected_inputs = np.r_[self.selected_inputs, max_inputs]
        return max_inputs

    def next_input_pool(self, X):
        if self.selected_inputs is not None:
            tmp_num_worker = self.num_worker - np.shape(self.selected_inputs)[0]
        else:
            tmp_num_worker = self.num_worker

        self.sampling_num = 1
        for _ in range(tmp_num_worker):
            _, max_inputs = self.sampling_RFM(X)
            max_inputs = np.atleast_2d(max_inputs)
            if self.selected_inputs is None:
                self.selected_inputs = max_inputs
            else:
                self.selected_inputs = np.r_[self.selected_inputs, max_inputs]
            X = X[~np.all(X==max_inputs, axis=1)]

        return self.selected_inputs[-tmp_num_worker:, :], X


class Asynchronous_randomized_ProbabilityImprovement(ParallelBO):
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.pool_X = pool_X

        if selected_inputs is None:
            self.sampling_num = num_worker
        else:
            self.sampling_num = num_worker - np.shape(selected_inputs)[0]

        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None
            self.sampling_num = self.num_worker
        else:
            self.sampling_num = self.num_worker - np.shape(self.selected_inputs)[0]


        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return (self.maximums[self.q] - mean) / std

    def parallel_acq(self, x):
        return self.acq(x)




class MaxvalueEntropySearch_LP(ParallelBO, MaxvalueEntropySearch):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, sampling_num=10, sampling_method='RFM', GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.pool_X = pool_X

        start = time.time()
        if sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(pool_X)
        elif sampling_method == 'RFM':
            self.maximums, _ = self.sampling_RFM(pool_X)
        else:
            print('This sampling method '+ sampling_method +' is not implemented for MES-LP')
            exit(1)
        print('sampled maximums:', self.maximums)

        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        x0s = np.r_[x0s, self.unique_X]

        self.M = self.y_max
        _, f_min = utils.minimize(self._predict_norm_gradients, x0s, self.bounds_list)
        self.GP_LCA = -1 * f_min
        self.preprocessing_time = time.time() - start

    def _predict_norm_gradients(self, x):
        x = np.atleast_2d(x)
        mean_gradient, _ = self.GPmodel.predict_jacobian(x)
        norm_gradients = np.sqrt(np.sum(mean_gradient**2))
        return -1 * norm_gradients

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        for i in range(np.shape(X)[0]):
            self.selected_inputs = np.delete(self.selected_inputs, np.where(np.all(self.selected_inputs==X[i], axis=1) == True)[0][0], axis=0)
        # self.selected_inputs = self.selected_inputs[~np.all(self.selected_inputs==X, axis=1)]
        if np.size(self.selected_inputs) == 0:
            self.selected_inputs = None

        start = time.time()
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        x0s = np.r_[x0s, self.unique_X]

        self.M = self.y_max
        _, f_min = utils.minimize(self._predict_norm_gradients, x0s, self.bounds_list)
        self.GP_LCA = -1 * f_min
        self.preprocessing_time += time.time() - start


    def parallel_acq(self, x):
        # MES
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        acq_MES = np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()

        dist = utils._two_inputs_dist(x, self.selected_inputs)
        z = (self.GP_LCA * dist - self.M + mean) / np.sqrt(2 * var)
        penalization = np.prod(special.erfc(-z), axis=1).ravel() / 2.
        return -1 * acq_MES * penalization

class EI_LP(ParallelBO, ExpectedImprovement):
    def __init__(self, X, Y, bounds, kernel_bounds, selected_inputs=None, num_worker=1, xi=1e-2, GPmodel=None, pool_X=None, optimize=True):
        ParallelBO.__init__(self, X, Y, bounds, kernel_bounds, selected_inputs=selected_inputs, num_worker=num_worker, GPmodel=GPmodel, optimize=optimize)
        self.xi = xi

        self.M = self.y_max
        def predict_gradient(x):
            x = np.atleast_2d(x)
            mean_gradient, _ = self.GPmodel.predict_jacobian(x)
            norm_gradients = np.sqrt(np.sum(mean_gradient**2))
            return -1 * norm_gradients

        res = utils.minimize(predict_gradient, self.bounds_list, inputs_dim=self.input_dim)
        self.GP_LCA = -1 * res['fun']

    def next_input(self):
        return super().next_input_greedy()

    def next_input_pool(self, X):
        return super().next_input_pool_greedy(X)

    def parallel_acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        Z = (mean - self.y_max - self.xi) / std
        acq_EI = ((Z * std)*norm.cdf(Z) + std*norm.pdf(Z)).ravel()

        dist = utils._two_inputs_dist(x, self.selected_inputs)
        z = (self.GP_LCA * dist - self.M + mean) / np.sqrt(2 * var)
        penalization = np.prod(special.erfc(-z), axis=1).ravel() / 2.
        return -1 * acq_EI * penalization
