import os
import sys
import time
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
# from scipydirect import minimize

from ..myutils import myutils as utils
from ..myutils.BO_core import BO_core

class BO(BO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel, optimize=optimize)


    def next_input_pool(self, X):
        tmp_X = X[(self._upper_bound(X) >= self.y_max).ravel()]
        self.acquisition_values = self.acq(tmp_X)
        next_input = np.atleast_2d(tmp_X[np.argmin(self.acquisition_values)])
        X = X[~np.all(X == next_input, axis=1), :]
        return next_input, X

    def next_input(self):
        num_start = 100 * self.input_dim
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean, _ = self.GPmodel.predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]


        f_min = np.inf
        x_min = x0s[0]
        if self.max_inputs is not None:
            x0s = np.r_[x0s, self.max_inputs]

        x0s = x0s[(self._upper_bound(x0s) >= self.y_max).ravel()]

        x_min, f_min = utils.minimize(self.acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        print('optimized acquisition function value:', -1*f_min)
        return np.atleast_2d(x_min)


class ProbabilityImprovement(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, xi=1e-2, GPmodel=None, pool_X=None, optimize=True):
        self.xi = xi
        self.preprocessing_time = 0
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

    # For minimize, multiply minus
    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        Z = (mean - self.y_max - self.xi) / std
        return - norm.cdf(Z).ravel()

class ExpectedImprovement(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, xi=1e-2, GPmodel=None, pool_X=None, optimize=True):
        self.xi = xi
        self.preprocessing_time = 0
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

    # For minimize, multiply minus
    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        Z = (mean - self.y_max - self.xi) / std
        return - ((Z * std)*norm.cdf(Z) + std*norm.pdf(Z)).ravel()


class UncertaintySampling(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        self.preprocessing_time = 0
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

    # For minimize, multiply minus
    def acq(self, x):
        x = np.atleast_2d(x)
        _, var = self.GPmodel.predict_noiseless(x)
        return - var


class GP_UCB(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, iteration=1, GPmodel=None, pool_X=None, optimize=True):
        self.iteration = iteration
        self.preprocessing_time = 0
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

    def update(self, X, Y, optimize=False):
        self.iteration += 1
        super().update(X, Y, optimize=optimize)

    # For minimize, multiply minus
    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        return - (mean + np.sqrt(2. * self.input_dim * np.log(2 * self.iteration)) * std).ravel()

class ThompsonSampling(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1
        self.preprocessing_time = 0


    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def next_input(self):
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        return np.atleast_2d(self.max_inputs[0])

    def next_input_pool(self, X):
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        next_input = np.atleast_2d(self.max_inputs[0])
        X = X[~np.all(X == next_input, axis=1), :]
        return next_input, X




class MaxvalueEntropySearch(BO):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, sampling_num=10, sampling_method='Gumbel', GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_method = sampling_method
        start = time.time()
        if sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(pool_X)
        elif sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        # mean, var = self.GPmodel.predict(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        if np.any(cdf <= 0):
            print(mean, var)
        return - np.mean((normalized_max * pdf) / (2*cdf) - np.log(cdf), axis=1).ravel()


class MaxvalueEntropySearch_IBO(BO):
    #sampling_method = Gumbel / RFM (RandomFeatureMap)
    def __init__(self, X, Y, bounds, kernel_bounds, sampling_num=10, sampling_method='Gumbel', GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)
        self.sampling_num = sampling_num
        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_method = sampling_method
        start = time.time()
        if sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(pool_X)
        elif sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        if self.sampling_method == 'Gumbel':
            self.maximums = self.sampling_gumbel(self.pool_X)
        elif self.sampling_method == 'RFM':
            self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)
        normalized_max = (self.maximums - mean) / std
        # pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        # acq = np.sum(np.log(cdf), axis=1).ravel()
        acq = np.product(cdf, axis=1)
        return acq



class randomized_ProbabilityImprovement(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1

        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print('sampled maximums:', self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return (self.maximums - mean) / std


class randomized_ProbabilityImprovement_maxmu(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1

        start = time.time()
        self.maximums, self.max_inputs, self.mu_flag = self.sampling_RFM_max_mean_sample(pool_X)
        self.preprocessing_time = time.time() - start
        print('mu_flag, sampled maximums:', self.mu_flag, self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.maximums, self.max_inputs, self.mu_flag = self.sampling_RFM_max_mean_sample(self.pool_X)
        self.preprocessing_time = time.time() - start
        print('mu_flag, sampled maximums:', self.mu_flag, self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return (self.maximums - mean) / std

    def next_input(self):
        if self.mu_flag:
            return np.atleast_2d(self.max_inputs[0])
        else:
            return super().next_input()

    def next_input_pool(self, X):
        if self.mu_flag:
            next_input = np.atleast_2d(self.max_inputs[0])
            X = X[~np.all(X == next_input, axis=1), :]
            return next_input, X
        else:
            return super().next_input_pool(X)

class randomized_ProbabilityImprovement_truncation(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1

        start = time.time()
        self.mu_flag = True
        num_rejection = 0
        while self.mu_flag:
            self.maximums, self.max_inputs, self.mu_flag = self.sampling_RFM_max_mean_sample(pool_X)
            num_rejection += 1
        self.preprocessing_time = time.time() - start
        print('num_rejection, regesampled maximums:', num_rejection, self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.mu_flag = True
        num_rejection = 0
        while self.mu_flag:
            self.maximums, self.max_inputs, self.mu_flag = self.sampling_RFM_max_mean_sample(self.pool_X)
            num_rejection += 1
        self.preprocessing_time = time.time() - start
        print('num_rejection, regesampled maximums:', num_rejection, self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return (self.maximums - mean) / std
