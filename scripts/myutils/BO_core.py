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
from scipy import optimize
import time
import nlopt
# from scipy.optimize import minimize as scipyminimize

# My modules
from . import myutils as utils


class BO_core(object):
    __metaclass__ = ABCMeta

    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, optimize=True):
        self.GPmodel = utils.set_gpy_regressor(GPmodel, X, Y, kernel_bounds, optimize=optimize)
        self.y_max = np.max(Y)
        self.unique_X = np.unique(X, axis=0)
        self.input_dim = np.shape(X)[1]
        self.bounds = bounds
        self.bounds_list = bounds.T.tolist()
        self.sampling_num = 10
        self.inference_point = None
        self.top_number=50
        self.preprocessing_time = 0
        self.max_inputs = None

    def update(self, X, Y, optimize=False):
        self.GPmodel.add_XY(X, Y)
        if optimize:
            self.GPmodel.my_optimize()

        self.y_max = np.max(self.GPmodel.Y)
        self.unique_X = np.unique(self.GPmodel.X, axis=0)

    @abstractmethod
    def acq(self, x):
        pass

    @abstractmethod
    def next_input_pool(self, X):
        pass

    @abstractmethod
    def next_input(self):
        pass

    def _upper_bound(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean + 5*np.sqrt(var)

    def _lower_bound(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean - 5*np.sqrt(var)

    def posteriori_maximum_direct(self):
        res = utils.minimize(self.GPmodel.minus_predict, self.bounds_list, self.input_dim)
        # print(res)
        return np.atleast_2d(res['x']), -1 * res['fun']

    def posteriori_maximum(self):
        num_start = 100
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean, _ = self.GPmodel.predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        if self.inference_point is not None:
            x0s = np.r_[x0s, self.inference_point]


        x_min, f_min = utils.minimize(self.GPmodel.minus_predict, x0s,self.bounds_list, jac=self.GPmodel.minus_predict_gradients)

        # f_min = np.inf
        # x_min = x0s[0]
        # for x0 in x0s:
        #     res = optimize.minimize(self.GPmodel.minus_predict, x0=x0, bounds=self.bounds_list, method='L-BFGS-B', options={'ftol': 1e-3}, jac=self.GPmodel.minus_predict_gradients)
        #     if f_min > res['fun']:
        #         x_min = res['x']
        #         f_min = res['fun']

        self.inference_point = np.atleast_2d(x_min)
        return x_min, -1 * f_min

    def sampling_RFM(self, pool_X=None, MES_correction=True):

        basis_dim = 100 + np.shape(self.unique_X)[0]
        self.rbf_features = utils.RFM_RBF(lengthscales=self.GPmodel['.*rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim = basis_dim)
        X_train_features = self.rbf_features.transform(self.GPmodel.X)

        max_sample = np.zeros(self.sampling_num)
        max_inputs = list()

        A_inv = np.linalg.inv((X_train_features.T).dot(
            X_train_features) + np.eye(self.rbf_features.basis_dim)* self.GPmodel['.*Gaussian_noise.variance'].values)
        weights_mean = A_inv.dot(X_train_features.T).dot((self.GPmodel.Y - self.GPmodel.mean) / self.GPmodel.std)
        weights_var = A_inv * self.GPmodel['.*Gaussian_noise.variance'].values


        try:
            L = np.linalg.cholesky(weights_var)
        except np.linalg.LinAlgError as e:
            print('In RFM-based sampling,', e)
            L = np.linalg.cholesky(weights_var + 1e-8 * np.eye(np.shape(weights_var)[0]))


        standard_normal_rvs = np.random.normal(0, 1, size=(np.size(weights_mean), self.sampling_num))
        self.weights_sample = np.c_[weights_mean] + L.dot(standard_normal_rvs)

        if pool_X is None:
            num_start = 100 * self.input_dim
            x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

            if np.shape(self.unique_X)[0] <= self.top_number:
                x0s = np.r_[x0s, self.unique_X]
            else:
                mean, _ = self.GPmodel.predict(self.unique_X)
                mean = mean.ravel()
                top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
                x0s = np.r_[x0s, self.unique_X[top_idx]]
        else:
            if np.size(pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]) > 0:
                pool_X = pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]


        for j in range(self.sampling_num):
            def BLR(x):
                X_features = self.rbf_features.transform(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:,j]])
                return - (sampled_value * self.GPmodel.std + self.GPmodel.mean).ravel()

            def BLR_gradients(x):
                X_features = self.rbf_features.transform_grad(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:,j]])
                return - (sampled_value * self.GPmodel.std).ravel()


            if pool_X is None:
                f_min = np.inf
                x_min = x0s[0]

                x_min, f_min = utils.minimize(BLR, x0s,self.bounds_list, jac=BLR_gradients)
                max_sample[j] = -1 * f_min
                max_inputs.append(x_min)
            else:
                pool_Y = BLR(pool_X)
                min_index = np.argmin(pool_Y)
                max_sample[j] = -1 * pool_Y[min_index]
                max_inputs.append(pool_X[min_index])



        if MES_correction:
            correction_value = self.y_max + 3 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
            max_sample[max_sample < correction_value] = correction_value
        return max_sample, np.array(max_inputs)

    def sampling_RFM_max_mean_sample(self, pool_X=None):

        basis_dim = 100 + np.shape(self.unique_X)[0]
        self.rbf_features = utils.RFM_RBF(lengthscales=self.GPmodel['.*rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim = basis_dim)
        X_train_features = self.rbf_features.transform(self.GPmodel.X)

        max_sample = np.zeros(self.sampling_num)
        max_inputs = list()

        A_inv = np.linalg.inv((X_train_features.T).dot(
            X_train_features) + np.eye(self.rbf_features.basis_dim)* self.GPmodel['.*Gaussian_noise.variance'].values)
        weights_mean = A_inv.dot(X_train_features.T).dot((self.GPmodel.Y - self.GPmodel.mean) / self.GPmodel.std)
        weights_var = A_inv * self.GPmodel['.*Gaussian_noise.variance'].values


        try:
            L = np.linalg.cholesky(weights_var)
        except np.linalg.LinAlgError as e:
            print('In RFM-based sampling,', e)
            L = np.linalg.cholesky(weights_var + 1e-8 * np.eye(np.shape(weights_var)[0]))


        standard_normal_rvs = np.random.normal(0, 1, size=(np.size(weights_mean), self.sampling_num))
        self.weights_sample = np.c_[weights_mean] + L.dot(standard_normal_rvs)

        if pool_X is None:
            num_start = 100 * self.input_dim
            x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

            if np.shape(self.unique_X)[0] <= self.top_number:
                x0s = np.r_[x0s, self.unique_X]
            else:
                mean, _ = self.GPmodel.predict(self.unique_X)
                mean = mean.ravel()
                top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
                x0s = np.r_[x0s, self.unique_X[top_idx]]
        else:
            pool_X = pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]


        for j in range(self.sampling_num):
            def BLR(x):
                mu, _ = self.GPmodel.predict(np.atleast_2d(x))

                X_features = self.rbf_features.transform(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:,j]])
                sampled_value = sampled_value * self.GPmodel.std + self.GPmodel.mean

                return - np.max(np.c_[mu, sampled_value], axis=1).ravel()

            def BLR_gradients(x):
                mu, _ = self.GPmodel.predict(np.atleast_2d(x))

                X_features = self.rbf_features.transform(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:,j]])
                sampled_value = sampled_value * self.GPmodel.std + self.GPmodel.mean

                if sampled_value > mu:
                    X_features = self.rbf_features.transform_grad(x)
                    sampled_value = X_features.dot(np.c_[self.weights_sample[:,j]])
                    return - (sampled_value * self.GPmodel.std).ravel()
                else:
                    mu_grad, _ = self.GPmodel.predictive_gradients(np.atleast_2d(x))
                    return - mu_grad.ravel()


            if pool_X is None:
                f_min = np.inf
                x_min = x0s[0]

                x_min, f_min = utils.minimize(BLR, x0s,self.bounds_list, jac=BLR_gradients)
                max_sample[j] = -1 * f_min
                max_inputs.append(x_min)
            else:
                pool_Y = BLR(pool_X)
                min_index = np.argmin(pool_Y)
                max_sample[j] = -1 * pool_Y[min_index]
                max_inputs.append(pool_X[min_index])

        mu_flag = False
        mu_x_min, _ = self.GPmodel.predict(np.c_[max_inputs[0]].T)
        # if BLR(max_inputs[0]) == mu_x_min:
        if max_sample[0] == mu_x_min:
            mu_flag = True

        return max_sample, np.array(max_inputs), mu_flag

    def sample_path(self, X):
        '''

        Parameter
        -----------------------
        X: numpy array
            inputs (N \times input_dim)

        Retrun
        -----------------------
        sampled_outputs: numpy array
            sample_path f_s(X) (N \times sampling_num)
        '''
        X_features = self.rbf_features.transform(X)
        sampled_outputs = X_features.dot(np.c_[self.weights_sample]) * self.GPmodel.std + self.GPmodel.mean
        return sampled_outputs

    def _find_r(self, val, cdf_func, R, Y, thres):
        current_r_pos = np.argmin(np.abs(val - R))
        if (np.abs(val - R[current_r_pos])) < thres:
            return Y[current_r_pos]

        # print(np.shape(Y))
        if R[current_r_pos] > val:
            left = Y[current_r_pos - 1]
            right = Y[current_r_pos]
        else:
            left = Y[current_r_pos]
            right = Y[current_r_pos + 1]

        for _ in range(10000):
            mid = (left + right)/2.
            mid_r = cdf_func(mid)

            if (np.abs(val - mid_r)) < thres:
                return mid

            if mid_r > val:
                right = mid
            else:
                left = mid

        print('r='+str(val)+': error over')

        return mid

    def sampling_gumbel(self, pool_X=None):
        if pool_X is None:
            x = self.bounds[0, :] + (self.bounds[1, :] - self.bounds[0, :])*np.random.rand(10000, self.input_dim)
            x = np.r_[x, self.GPmodel.X]
        else:
            x = np.atleast_2d(pool_X)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        def approx_gumbel_cdf(y):
            return np.prod(norm.cdf((y-np.c_[mean])/np.c_[std]), axis=0)

        left = self.y_max
        if approx_gumbel_cdf(left) < 0.25:
            right = np.max(mean + 5*std)
            while (approx_gumbel_cdf(right) < 0.75):
                right = 2*right - left

            Y = np.c_[np.linspace(left, right, 100)].T
            R = approx_gumbel_cdf(Y)
            Y = np.ravel(Y)

            med = self._find_r(0.5, approx_gumbel_cdf, R, Y, 0.01)
            y1 = self._find_r(0.25, approx_gumbel_cdf, R, Y, 0.01)
            y2 = self._find_r(0.75, approx_gumbel_cdf, R, Y, 0.01)

            self.b = (y1 - y2)/(np.log(np.log(4/3)) - np.log(np.log(4)))
            self.a = med + self.b*np.log(np.log(2))

            # print('y1 and y2 is:'+str(y1)+','+str(y2))
            # print('gumbel parameters:a='+str(self.a) + ', b=' + str(self.b))

            max_samples = np.array(np.random.gumbel(
                self.a, self.b, self.sampling_num))
            max_samples[max_samples < left + 5 *
                        np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = left + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        else:
            print('Error: observation is larger than maximum!?')
            max_samples = (left + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values)
                           ) * np.ones(self.sampling_num)

        return max_samples





class ConstrainedBO_core(object):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=None, GPmodel=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        self.model=model
        if GPmodel is None:
            if model=='independent':
                self.GPmodel = utils.GPy_independent_model(X_list, Y_list, kernel_name=kernel_name)
            if model=='correlated':
                self.GPmodel = utils.GPy_correlated_model(X_list, Y_list, kernel_name=kernel_name)
            self.GPmodel.set_hyperparameters_bounds(kernel_bounds)
        else:
            self.GPmodel = GPmodel

        # self.GPmodel.add_XY(X_list, Y_list)

        if optimize:
            self.GPmodel.my_optimize()

        self.y_max = None
        self.x_max = None
        self.feasible_points = list()
        unique_X, count = np.unique(np.vstack(X_list), return_counts=True, axis=0)
        allObj_X = unique_X[count>=C+1]

        for i in range(np.shape(allObj_X)[0]):
            feasible_flag = True
            for c in range(C):
                if Y_list[c+1][np.all(X_list[c+1]==allObj_X[i], axis=1)][0] < thresholds[c]:# - 1e-3:
                    feasible_flag = False
                    break

            if feasible_flag:
                self.feasible_points.append(allObj_X[i])
                temp_y = Y_list[0][np.all(X_list[0]==allObj_X[i], axis=1)]

                if np.size(temp_y) > 1:
                    temp_y = temp_y[0]
                if self.y_max is None:
                    self.y_max = temp_y
                    self.x_max = allObj_X[i]
                elif self.y_max < temp_y:
                    self.y_max = temp_y
                    self.x_max = allObj_X[i]
        if self.y_max is not None:
            self.y_max = self.y_max.ravel()


        quantile_val = np.power(0.95, 1/C)
        self.prod_cons = norm.ppf(quantile_val)
        means = list()
        vars = list()
        for c in range(C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[X_list[0], np.c_[(c+1)*np.ones(np.shape(X_list[0])[0])]])
            means.append(mean.ravel())
            vars.append(var.ravel())
        means = np.array(means).T
        vars = np.array(vars).T
        lower_bound = means - self.prod_cons * np.sqrt(vars)
        high_prob_index = np.all(lower_bound > thresholds, axis=1)
        if np.any(high_prob_index):
            high_prob_observations = Y_list[0][high_prob_index]
            self.max_samples_lower = np.max(high_prob_observations)
        else:
            self.max_samples_lower = None

        if self.y_max is not None:
            if self.max_samples_lower is None:
                self.max_samples_lower = self.y_max
            elif self.y_max > self.max_samples_lower:
                self.max_samples_lower = self.y_max

        if np.size(self.feasible_points) > 0:
            self.feasible_points = np.array(self.feasible_points)
        else:
            self.feasible_points = None

        self.input_dim = np.shape(X_list[0])[1]
        self.bounds = bounds
        self.bounds_list = bounds.T.tolist()
        self.sampling_num = 10
        if cost is None:
            self.cost = np.ones(C+1)
        self.C = C
        self.thresholds = thresholds
        self.max_samples = None
        self.max_inputs = None


        print('y max:', self.y_max)
        print('x max:', self.x_max)
        print('max lower:', self.max_samples_lower)

    def update(self, add_X_list, add_Y_list, optimize=False):
        self.GPmodel.add_XY(add_X_list, add_Y_list)
        if optimize:
            self.GPmodel.my_optimize()

        Y_size = np.array([np.size(Y) for Y in add_Y_list])
        if Y_size[0] > 0 and np.all(Y_size == Y_size[0]):
            X_temp = np.all([np.all(add_X_list[0] == X) for X in add_X_list])

            if X_temp:
                Y_temp = np.hstack(add_Y_list)
                feasible_index = np.all(Y_temp[:,1:] >= np.c_[self.thresholds].T, axis=1)

                if np.any(feasible_index):
                    if self.feasible_points is None:
                        self.feasible_points = add_X_list[0][feasible_index,:]
                    else:
                        self.feasible_points = np.r_[self.feasible_points, add_X_list[0][feasible_index,:]]
                    add_arg_max = np.argmax(add_Y_list[0][feasible_index])
                    add_Y_feasible_max = add_Y_list[0][feasible_index][add_arg_max]

                    if self.y_max is None or self.y_max < add_Y_feasible_max:
                            self.y_max = add_Y_feasible_max
                            self.x_max = add_X_list[0][feasible_index][add_arg_max]

        means = list()
        vars = list()
        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[add_X_list[0], np.c_[(c+1)*np.ones(np.shape(add_X_list[0])[0])]])
            means.append(mean.ravel())
            vars.append(var.ravel())
        means = np.array(means).T
        vars = np.array(vars).T
        lower_bound = means - self.prod_cons * np.sqrt(vars)
        high_prob_index = np.all(lower_bound > self.thresholds, axis=1)
        if np.any(high_prob_index):
            high_prob_observations = add_Y_list[0][high_prob_index]
            if self.max_samples_lower is None or self.max_samples_lower < np.max(high_prob_observations):
                self.max_samples_lower = np.max(high_prob_observations)

        if self.y_max is not None:
            if self.max_samples_lower is None:
                self.max_samples_lower = self.y_max
            elif self.y_max > self.max_samples_lower:
                self.max_samples_lower = self.y_max

        print('y max:', self.y_max)
        print('x max:', self.x_max)
        print('max lower:', self.max_samples_lower)
        # print('feasible points:', self.feasible_points)


    @abstractmethod
    def acq(self, x):
        pass

    @abstractmethod
    def acq_correlated(self, x):
        pass

    @abstractmethod
    def next_input_pool(self, X):
        pass

    @abstractmethod
    def next_input(self):
        pass

    def posteriori_const_optimize(self, lower_p, additional_x=None):
        quantile_val = np.power(lower_p, 1/self.C)
        prod_cons = norm.ppf(quantile_val)

        def func_nlopt(x, grad):
            x = np.atleast_2d(x)
            if grad.size > 0:
                f_mean_grad, _ = self.GPmodel.predictive_gradients(np.c_[x, [0]])
                grad[:] = - f_mean_grad.ravel()
            f_mean, _ = self.GPmodel.predict(np.c_[x, [0]])
            return - f_mean.ravel()[0]

        def constraint_maker_for_nlopt(c=0):  # c MUST be an optional keyword argument, else it will not work
            def const_nlopt(x, grad):
                x = np.atleast_2d(x)
                g_mean, g_var = self.GPmodel.predict_noiseless(np.c_[x, [c+1]])
                g_std = np.sqrt(g_var.ravel())

                if grad.size > 0:
                    g_mean_grad, g_var_grad = self.GPmodel.predictive_gradients(np.c_[x, [c+1]])
                    grad[:] = - g_mean_grad.ravel() + prod_cons * g_var_grad.ravel() / (2*g_std)
                return (- g_mean.ravel() + prod_cons * g_std + self.thresholds[c])[0]
            return const_nlopt

        NUM_START = 100 * np.min([self.input_dim, 10])
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        if additional_x is not None:
            x0 = np.r_[x0, np.atleast_2d(additional_x)]

        opt = nlopt.opt(nlopt.LD_MMA, self.input_dim)
        opt.set_lower_bounds(self.bounds[0].tolist())
        opt.set_upper_bounds(self.bounds[1].tolist())
        opt.set_min_objective(func_nlopt)
        for c in range(self.C):
            opt.add_inequality_constraint(constraint_maker_for_nlopt(c), 0)
        opt.set_xtol_rel(1e-3)
        opt.set_ftol_rel(1e-3)
        opt.set_maxtime(60*2 / NUM_START)
        # '''
        # for debug
        # '''
        # opt.set_maxtime(1 / NUM_START)
        opt.set_maxeval(1000)

        # check the mean value at the observed points and set the defalt inference point if it satisfy the constraints
        if self.x_max is not None:
            high_prob_means, _ = self.GPmodel.predict_noiseless(np.c_[np.atleast_2d(self.x_max), [0]])
            min_fun = - high_prob_means.ravel()[0]
            min_x = self.x_max.copy()
        else:
            min_fun = np.inf
            min_x = None
        print('First inference point:', min_x, min_fun)

        grad = np.array([])
        for i in range(np.shape(x0)[0]):
            x = opt.optimize(x0[i])
            minf = opt.last_optimum_value()
            res = opt.last_optimize_result()
            if res > 0:
                success_flag = True
                for c in range(self.C):
                    const = constraint_maker_for_nlopt(c)
                    if const(x, grad) > 0:
                        success_flag = False
                        break
                if min_fun > minf and success_flag:
                    min_fun = minf
                    min_x = x

        if min_x is not None:
            opt.set_xtol_rel(1e-16)
            opt.set_ftol_rel(1e-16)
            opt.set_maxeval(0)
            opt.set_maxtime(5)
            x = opt.optimize(min_x)
            minf = opt.last_optimum_value()
            res = opt.last_optimize_result()
            if res > 0:
                # res
                success_flag = True
                for c in range(self.C):
                    const = constraint_maker_for_nlopt(c)
                    if const(x, grad) > 0:
                        success_flag = False
                        break
                if min_fun > minf and success_flag:
                    min_fun = minf
                    min_x = x
        print(min_x, - min_fun)
        return min_x

    def sampling_RFM(self, pool_X=None, sampling_approach='inf'):
        if self.model=='independent':
            return self._sampling_RFM_independent(pool_X=pool_X, sampling_approach=sampling_approach)
        if self.model=='correlated':
            return self._sampling_RFM_correlated(pool_X=pool_X, sampling_approach=sampling_approach)


    def _sampling_RFM_independent(self, pool_X=None, sampling_approach='inf'):
        #
        basis_dim = 1000
        self.random_features_list = list()
        self.weights_mean_list = list()
        self.weights_varL_list = list()
        self.weights_sample_list = list()
        max_sample_nlopt = -np.inf * np.ones(self.sampling_num)
        max_inputs_nlopt = [None for i in range(self.sampling_num)]

        for c in range(self.C+1):
            if self.GPmodel.kernel_name == 'rbf':
                self.random_features_list.append(utils.RFM_RBF(lengthscales=self.GPmodel.model_list[c]['.*rbf.lengthscale'].values, input_dim=self.input_dim, variance=self.GPmodel.model_list[c]['.*rbf.variance'].values, basis_dim = basis_dim))
                X_train_features = self.random_features_list[c].transform(self.GPmodel.model_list[c].X)
            elif self.GPmodel.kernel_name == 'linear+rbf':
                self.random_features_list.append(utils.RFM_Linear_RBF(lengthscales=self.GPmodel.model_list[c]['.*rbf.lengthscale'].values, input_dim=self.input_dim, RBF_variance=self.GPmodel.model_list[c]['.*rbf.variance'].values, Linear_variance=self.GPmodel.model_list[c]['.*linear.variances'].values, basis_dim_for_RBF = basis_dim))
                X_train_features = self.random_features_list[c].transform(self.GPmodel.model_list[c].X[:,:-1])
            else:
                print('not implemented RFM for the kernel:', self.GPmodel.kernel_name)
                exit(1)

            A_inv = np.linalg.inv((X_train_features.T).dot(
                X_train_features) + np.eye(self.random_features_list[c].basis_dim)* self.GPmodel.model_list[c]['.*Gaussian_noise.variance'].values)
            self.weights_mean_list.append(A_inv.dot(X_train_features.T).dot(self.GPmodel.model_list[c].Y))
            weights_var = A_inv * self.GPmodel.model_list[c]['.*Gaussian_noise.variance'].values

            try:
                self.weights_varL_list.append(np.linalg.cholesky(weights_var))
            except np.linalg.LinAlgError:
                print('Cholesky decomposition in RFM output error, thus add the 1e-3 to diagonal elements.')
                weights_var = weights_var + 1e-3*np.eye(self.random_features_list[c].basis_dim)
                self.weights_varL_list.append(np.linalg.cholesky(weights_var))


            #
            standard_normal_rvs = np.random.normal(0, 1, size=(self.random_features_list[c].basis_dim, self.sampling_num))
            self.weights_sample_list.append(np.c_[self.weights_mean_list[c]] + self.weights_varL_list[c].dot(standard_normal_rvs))

        if pool_X is None:
            NUM_START = 100 * np.min([self.input_dim, 10])
            x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
            if self.feasible_points is not None:
                x0 = np.r_[x0, self.feasible_points]

        # NUM_GRID = 100
        # x1 = np.linspace(0, 1, NUM_GRID)
        # x2 = np.linspace(0, 1, NUM_GRID)
        # X1, X2 = np.meshgrid(x1, x2)
        # X = np.c_[np.c_[X1.ravel()], np.c_[X2.ravel()]]
        # for c in range(self.C+1):
        #     for j in range(self.sampling_num):
        #         X_features = self.random_features_list[c].transform(X)
        #         value = (X_features.dot(np.c_[self.weights_sample_list[c][:,j]]) * self.GPmodel.stds[c] + self.GPmodel.means[c]).ravel()
        #         plt.pcolor(X1, X2, value.reshape(NUM_GRID, NUM_GRID))
        #         plt.colorbar()
        #         # plt.show()
        #         plt.savefig('sampling_tests/test_Gramacy_c='+str(c)+'_j='+str(j)+'.pdf')
        #         plt.close()



        j=0
        while j < self.sampling_num:
            if pool_X is None:
                def sampled_obj_func_for_nlopt(x, grad):
                    x = np.atleast_2d(x)
                    X_features = self.random_features_list[0].transform(x)
                    if grad.size > 0:
                        X_features_grad = self.random_features_list[0].transform_grad(x)
                        grad[:] = - (X_features_grad.dot(np.c_[self.weights_sample_list[0][:,j]]).ravel() * self.GPmodel.stds[0])
                    value = - (X_features.dot(np.c_[self.weights_sample_list[0][:,j]]) * self.GPmodel.stds[0] + self.GPmodel.means[0]).ravel()[0]
                    return value

                def constraint_maker_for_nlopt(c=0):  # c MUST be an optional keyword argument, else it will not work
                    def sampled_const_func(x, grad):
                        x = np.atleast_2d(x)
                        X_features = self.random_features_list[c+1].transform(x)
                        if grad.size > 0:
                            X_features_grad = self.random_features_list[c+1].transform_grad(x)
                            grad[:] = - (X_features_grad.dot(np.c_[self.weights_sample_list[c+1][:,j]]).ravel() * self.GPmodel.stds[c+1] )
                        value = - (X_features.dot(np.c_[self.weights_sample_list[c+1][:,j]]) * self.GPmodel.stds[c+1] + self.GPmodel.means[c+1] - self.thresholds[c]).ravel()[0]
                        return value
                    return sampled_const_func

                opt = nlopt.opt(nlopt.LD_MMA, self.input_dim)
                opt.set_lower_bounds(self.bounds[0].tolist())
                opt.set_upper_bounds(self.bounds[1].tolist())
                opt.set_min_objective(sampled_obj_func_for_nlopt)
                for c in range(self.C):
                    opt.add_inequality_constraint(constraint_maker_for_nlopt(c), 0)
                opt.set_xtol_rel(1e-2)
                opt.set_ftol_rel(1e-2)
                opt.set_maxeval(1000)
                opt.set_maxtime(30 / NUM_START)

                for i in range(np.shape(x0)[0]):
                    x = opt.optimize(x0[i])
                    minf = opt.last_optimum_value()
                    res = opt.last_optimize_result()
                    if res > 0:
                        # res
                        success_flag = True
                        for c in range(self.C):
                            const = constraint_maker_for_nlopt(c)
                            if const(x, np.array([])) > 0:
                                success_flag = False
                                break
                        if max_sample_nlopt[j] < - minf and success_flag:
                            max_sample_nlopt[j] = - minf
                            max_inputs_nlopt[j] = np.array(x)
            else:
                # Note that sign of the functions are different to nlopt
                def sampled_obj_func(x):
                    x = np.atleast_2d(x)
                    X_features = self.random_features_list[0].transform(x)
                    value = (X_features.dot(np.c_[self.weights_sample_list[0][:,j]]) * self.GPmodel.stds[0] + self.GPmodel.means[0]).ravel()
                    return value

                def constraint_maker(c=0):  # c MUST be an optional keyword argument, else it will not work
                    def sampled_const_func(x):
                        x = np.atleast_2d(x)
                        X_features = self.random_features_list[c+1].transform(x)
                        value = (X_features.dot(np.c_[self.weights_sample_list[c+1][:,j]]) * self.GPmodel.stds[c+1] + self.GPmodel.means[c+1]).ravel()
                        return value
                    return sampled_const_func


                pool_obj = sampled_obj_func(pool_X)
                pool_const = list()
                for c in range(self.C):
                    const = constraint_maker(c)
                    pool_const.append(np.c_[const(pool_X)].T)
                pool_const = np.vstack(pool_const)


                feasible_index = (pool_const - np.c_[self.thresholds]) > 0
                feasible_index = np.all(feasible_index, axis=0)
                if np.any(feasible_index):
                    max_index = np.argmax(pool_obj[feasible_index])
                    max_sample_nlopt[j] = pool_obj[feasible_index][max_index]
                    max_inputs_nlopt[j] = pool_X[feasible_index][max_index]


            if max_inputs_nlopt[j] is None:
                if sampling_approach == 'inf':
                    max_sample_nlopt[j] = - np.infty
                elif sampling_approach == 'MinViolation':
                    max_inputs_nlopt[j], max_sample_nlopt[j] = self._minimize_violation(sample_dim=j, pool_X=pool_X)
                j += 1
            else:
                if pool_X is None:
                    opt.set_xtol_rel(1e-16)
                    opt.set_ftol_rel(1e-16)
                    opt.set_maxeval(0)
                    opt.set_maxtime(1)
                    x = opt.optimize(max_inputs_nlopt[j])
                    minf = opt.last_optimum_value()

                    res = opt.last_optimize_result()
                    if res > 0:
                        success_flag = True
                        for c in range(self.C):
                            const = constraint_maker_for_nlopt(c)
                            if const(x, np.array([])) > 0:
                                success_flag = False
                                break
                        if max_sample_nlopt[j] < - minf and success_flag:
                            max_sample_nlopt[j] = - minf
                            max_inputs_nlopt[j] = np.array(x)
                # print(-minf, max_sample_nlopt[j], -minf - max_sample_nlopt[j])
                j += 1


        del self.weights_mean_list
        del self.weights_varL_list
        print('sampled maximums before cut-off:', max_sample_nlopt)
        #
        if self.max_samples_lower is not None:
            correction_term = self.max_samples_lower + 5 * np.sqrt(self.GPmodel.model_list[0]['.*Gaussian_noise.variance'].values) * self.GPmodel.stds[0]
            print(correction_term)
            max_sample_nlopt[max_sample_nlopt < correction_term] = correction_term
        return max_sample_nlopt, max_inputs_nlopt


    # GPy
    # def RBF_kernel_correlated(self, X1, X2):
    #     X1 = np.atleast_2d(X1)
    #     X2 = np.atleast_2d(X2)
    #     dist_original_scale = (X1[:,:-1]**2)[:,None,:] + (X2[:,:-1]**2)[None,:,:] - 2*X1[:,:-1][:,None,:]*X2[:,:-1][None,:,:]

    #     index = np.meshgrid(X1[:, -1], X2[:, -1])
    #     index_1 = index[0].astype(int)
    #     index_2 = index[1].astype(int)

    #     ell_1 = self.GPmodel['.*mul.rbf.lengthscale'].values
    #     B_1 = self.GPmodel['.*mul.coregion.W'].values.dot(self.GPmodel['.*mul.coregion.W'].values.T) + np.diag(self.GPmodel['.*mul.coregion.kappa'].values)
    #     ell_2 = self.GPmodel['.*mul_1.rbf.lengthscale'].values
    #     B_2 = self.GPmodel['.*mul_1.coregion.W'].values.dot(self.GPmodel['.*mul_1.coregion.W'].values.T) + np.diag(self.GPmodel['.*mul_1.coregion.kappa'].values)

    #     dist = np.sum( dist_original_scale / (2 * ell_1[None,None,:]**2), axis=2)
    #     kernel_1 = B_1[index_1, index_2].T * np.exp( - dist)
    #     dist = np.sum( dist_original_scale / (2 * ell_2[None,None,:]**2), axis=2)
    #     kernel_2 = B_2[index_1, index_2].T * np.exp( - dist)

    #     return np.c_[kernel_1 + kernel_2]


    def _sampling_RFM_correlated(self, pool_X=None, sampling_approach='inf'):
        #
        if self.GPmodel.kernel_name == 'linear+rbf':
            # basis_dim = 4000//(2*(self.C+1))
            basis_dim = 100

            self.rbf_features_1 = utils.RFM_Linear_RBF(lengthscales=self.GPmodel['.*mul.sum.rbf.lengthscale'].values, input_dim=self.input_dim, RBF_variance=self.GPmodel['.*mul.sum.rbf.variance'].values, Linear_variance=self.GPmodel['.*mul.sum.linear.variances'].values, basis_dim_for_RBF = basis_dim)
            self.rbf_features_2 = utils.RFM_Linear_RBF(lengthscales=self.GPmodel['.*mul_1.sum.rbf.lengthscale'].values, input_dim=self.input_dim, RBF_variance=self.GPmodel['.*mul_1.sum.rbf.variance'].values, Linear_variance=self.GPmodel['.*mul_1.sum.linear.variances'].values, basis_dim_for_RBF = basis_dim)
        elif self.GPmodel.kernel_name == 'rbf':
            # basis_dim = 4000//(2*(self.C+1))
            basis_dim = 100

            # print('check RFM model:', basis_dim, np.shape(self.GPmodel.X))
            self.rbf_features_1 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf.lengthscale'].values, input_dim=self.input_dim, variance=self.GPmodel['.*mul.rbf.variance'].values, basis_dim = basis_dim)
            self.rbf_features_2 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul_1.rbf.lengthscale'].values, input_dim=self.input_dim, variance=self.GPmodel['.*mul_1.rbf.variance'].values, basis_dim = basis_dim)

        W = self.GPmodel['.*mul.coregion.W'].values
        kappa = self.GPmodel['.*mul.coregion.kappa'].values
        # print(W)
        # print(kappa)
        # print(W.dot(W.T) + np.diag(kappa))
        if not all(kappa == np.zeros(self.C+1)):
            self.L_1 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
        else:
            self.L_1 = W
        W = self.GPmodel['.*mul_1.coregion.W'].values
        kappa = self.GPmodel['.*mul_1.coregion.kappa'].values
        # print(W)
        # print(kappa)
        # print(W.dot(W.T) + np.diag(kappa))
        if not all(kappa == np.zeros(self.C+1)):
            self.L_2 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
        else:
            self.L_2 = W

        X_train_features_1 = list([])
        X_train_features_2 = list([])
        for c in range(self.C+1):
            if self.GPmodel.kernel_name=='rbf':
                X_train_features_1_m = self.rbf_features_1.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==c, : -1])
                X_train_features_2_m = self.rbf_features_2.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==c, : -1])
            elif self.GPmodel.kernel_name=='linear+rbf':
                X_train_features_1_m = self.rbf_features_1.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==c, : -2])
                X_train_features_2_m = self.rbf_features_2.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==c, : -2])
            X_train_features_1.append(np.kron(self.L_1[c, :], X_train_features_1_m))
            X_train_features_2.append(np.kron(self.L_2[c, :], X_train_features_2_m))
        X_train_features_1 = np.vstack(X_train_features_1)
        X_train_features_2 = np.vstack(X_train_features_2)

        X_train_features = np.c_[X_train_features_1, X_train_features_2]
        del X_train_features_1
        del X_train_features_2
        del X_train_features_1_m
        del X_train_features_2_m

        #
        # A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(np.shape(X_train_features)[1]) * self.GPmodel['.*Gaussian_noise.variance'].values)
        # weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot( self.GPmodel.Y ))
        # weights_var = A_inv * self.GPmodel['.*Gaussian_noise.variance']

        #
        # L = np.linalg.cholesky(weights_var)
        # standard_normal_rvs = np.random.normal(0, 1, size=(np.size(X_train_features.shape[1]), self.sampling_num))

        # sampling_num
        # self.weights_sample = np.c_[weights_mean] + L.dot(np.c_[standard_normal_rvs])
        self.weights_sample = np.random.normal(0, 1, size=(X_train_features.shape[1], self.sampling_num))

        # N × #sampling
        self.v = self.GPmodel.posterior.woodbury_inv.dot(self.GPmodel.Y - X_train_features.dot(self.weights_sample) - np.sqrt(self.GPmodel['.*Gaussian_noise.variance'])*np.random.normal(0, 1, size=(np.shape(self.GPmodel.Y)[0], self.sampling_num)))

        # print(self.GPmodel.X)
        # K_inv = np.linalg.inv(self.GPmodel.kern.K(self.GPmodel.X, self.GPmodel.X) + self.GPmodel['.*Gaussian_noise.variance'] * np.eye(np.shape(self.GPmodel.X)[0]))
        # self.v = K_inv.dot(self.GPmodel.Y - X_train_features.dot(self.weights_sample) - np.sqrt(self.GPmodel['.*Gaussian_noise.variance'])*np.random.normal(0, 1, size=(np.shape(self.GPmodel.Y)[0], 1)))

        # del weights_var
        # del weights_mean
        # del A_inv
        # del L
        del X_train_features
        # del standard_normal_rvs

        if pool_X is None:
            NUM_START = 100 * np.min([self.input_dim, 10])
            x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
            if self.feasible_points is not None:
                x0 = np.r_[x0, self.feasible_points]

        max_sample_nlopt = -np.inf * np.ones(self.sampling_num)
        max_inputs_nlopt = [None for i in range(self.sampling_num)]


        # for debug plot
        # NUM_GRID = 101
        # x1 = np.linspace(0, 1, NUM_GRID)
        # x2 = np.linspace(0, 1, NUM_GRID)
        # X1, X2 = np.meshgrid(x1, x2)
        # X = np.c_[np.c_[X1.ravel()], np.c_[X2.ravel()]]
        # for c in range(self.C+1):
        #     for i in range(self.sampling_num):
        #         X_features_1 = self.rbf_features_1.transform(X)
        #         X_features_2 = self.rbf_features_2.transform(X)
        #         X_features = np.c_[np.kron(self.L_1[c, :], X_features_1), np.kron(self.L_2[c, :], X_features_2)]

        #         prior = X_features.dot(np.c_[self.weights_sample[:,i]]).ravel()

        #         update = self.v[:,i].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[X, c*np.c_[np.ones(np.shape(X)[0])]]))

        #         fig = plt.figure(figsize = (4,7))
        #         plt.subplot(2,1,1)
        #         plt.pcolor(X1, X2, prior.reshape(NUM_GRID, NUM_GRID), label='prior')
        #         plt.colorbar()
        #         plt.scatter(self.GPmodel.X[:,0][self.GPmodel.X[:,-1]==0], self.GPmodel.X[:,1][self.GPmodel.X[:,-1]==0], marker='x')

        #         plt.subplot(2,1,2)
        #         plt.pcolor(X1, X2, update.reshape(NUM_GRID, NUM_GRID), label='update')
        #         plt.colorbar()
        #         plt.scatter(self.GPmodel.X[:,0][self.GPmodel.X[:,-1]==0], self.GPmodel.X[:,1][self.GPmodel.X[:,-1]==0], marker='x')
        #         # plt.show()
        #         plt.savefig('sampling_tests/test_SynFun_c='+str(c)+'_j='+str(i)+'_datasize='+str(self.GPmodel.X.shape[0])+'.png')
        #         plt.close()

        #         x = np.atleast_2d([1./2., 1./2.])
        #         start = time.time()
        #         X_features_grad_1 = self.rbf_features_1.transform_grad(x)
        #         X_features_grad_2 = self.rbf_features_2.transform_grad(x)
        #         X_features_grad = np.c_[np.kron(self.L_1[c, :], X_features_grad_1), np.kron(self.L_2[c, :], X_features_grad_2)]
        #         grad_test = X_features_grad.dot(np.c_[self.weights_sample[:,i]]).ravel()
        #         grad_test += self.GPmodel.kern.gradients_X(np.c_[self.v[:,i]].T, np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]], self.GPmodel.X).ravel()[:-1]
        #         grad_test *= - self.GPmodel.stds[c]
        #         print('grad:', time.time() - start)

        #         h = 1e-6
        #         X = np.array([[0.5, 0.5], [0.5 + h, 0.5], [0.5, 0.5 + h], [0.55, 0.5], [0.55 + h, 0.5], [0.55, 0.5 + h]])
        #         start = time.time()
        #         X_features_1 = self.rbf_features_1.transform(X)
        #         X_features_2 = self.rbf_features_2.transform(X)
        #         X_features = np.c_[np.kron(self.L_1[c, :], X_features_1), np.kron(self.L_2[c, :], X_features_2)]
        #         value = - (X_features.dot(np.c_[self.weights_sample[:,i]]).ravel() + self.v[:,i].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[X, c*np.c_[np.ones(np.shape(X)[0])]])) ) * self.GPmodel.stds[c] + self.GPmodel.means[c]
        #         print('grad_app:', time.time() - start)

        #         print(grad_test)
        #         print((value[1] - value[0]) / h)
        #         print((value[2] - value[0]) / h)
        #         exit()

        # if self.GPmodel.X.shape[0] > 100:
        #     exit()
        # end for debug

        j=0
        while j < self.sampling_num:
            if pool_X is None:
                def sampled_obj_func_for_nlopt(x, grad):
                    x = np.atleast_2d(x)
                    X_features_1 = self.rbf_features_1.transform(x)
                    X_features_2 = self.rbf_features_2.transform(x)
                    X_features = np.c_[np.kron(self.L_1[0, :], X_features_1), np.kron(self.L_2[0, :], X_features_2)]
                    if grad.size > 0:
                        X_features_grad_1 = self.rbf_features_1.transform_grad(x)
                        X_features_grad_2 = self.rbf_features_2.transform_grad(x)
                        X_features_grad = np.c_[np.kron(self.L_1[0, :], X_features_grad_1), np.kron(self.L_2[0, :], X_features_grad_2)]
                        grad[:] = X_features_grad.dot(np.c_[self.weights_sample[:,j]]).ravel()
                        if self.GPmodel.kernel_name=='rbf':
                            grad[:] += self.GPmodel.kern.gradients_X(np.c_[self.v[:,j]].T, np.c_[x, [0]], self.GPmodel.X).ravel()[:-1]
                        elif self.GPmodel.kernel_name=='linear+rbf':
                            # print(self.GPmodel.kern.gradients_X(np.c_[self.v[:,j]].T, np.c_[x, [[1, 0]]], self.GPmodel.X).ravel()[:-2])
                            grad[:] += self.GPmodel.kern.gradients_X(np.c_[self.v[:,j]].T, np.c_[x, [[1, 0]]], self.GPmodel.X).ravel()[:-2]
                        grad[:] *= - self.GPmodel.stds[0]
                    value = X_features.dot(np.c_[self.weights_sample[:,j]]).ravel()
                    if self.GPmodel.kernel_name=='rbf':
                        value += self.v[:,j].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [0]]))
                    elif self.GPmodel.kernel_name=='linear+rbf':
                        value += self.v[:,j].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [[1, 0]]]))
                    value = - (value * self.GPmodel.stds[0] + self.GPmodel.means[0]).ravel()[0]
                    return value

                def constraint_maker_for_nlopt(c=0):  # c MUST be an optional keyword argument, else it will not work
                    def sampled_const_func(x, grad):
                        x = np.atleast_2d(x)
                        X_features_1 = self.rbf_features_1.transform(x)
                        X_features_2 = self.rbf_features_2.transform(x)
                        X_features = np.c_[np.kron(self.L_1[c+1, :], X_features_1), np.kron(self.L_2[c+1, :], X_features_2)]
                        if grad.size > 0:
                            X_features_grad_1 = self.rbf_features_1.transform_grad(x)
                            X_features_grad_2 = self.rbf_features_2.transform_grad(x)
                            X_features_grad = np.c_[np.kron(self.L_1[c+1, :], X_features_grad_1), np.kron(self.L_2[c+1, :], X_features_grad_2)]
                            grad[:] = X_features_grad.dot(np.c_[self.weights_sample[:,j]]).ravel()
                            if self.GPmodel.kernel_name=='rbf':
                                grad[:] += self.GPmodel.kern.gradients_X(np.c_[self.v[:,j]].T, np.c_[x, [c+1]], self.GPmodel.X).ravel()[:-1]
                            elif self.GPmodel.kernel_name=='linear+rbf':
                                grad[:] += self.GPmodel.kern.gradients_X(np.c_[self.v[:,j]].T, np.c_[x, [[1, c+1]]], self.GPmodel.X).ravel()[:-2]
                            grad[:] *= - self.GPmodel.stds[c+1]
                        value = X_features.dot(np.c_[self.weights_sample[:,j]]).ravel()
                        if self.GPmodel.kernel_name=='rbf':
                            value += self.v[:,j].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [c+1]]))
                        elif self.GPmodel.kernel_name=='linear+rbf':
                            value += self.v[:,j].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [[1, c+1]]]))
                        value = - (value * self.GPmodel.stds[c+1] + self.GPmodel.means[c+1] - self.thresholds[c]).ravel()[0]
                        # value = - (X_features.dot(np.c_[self.weights_sample[:,j]]) * self.GPmodel.stds[c+1] + self.GPmodel.means[c+1] - self.thresholds[c]).ravel()[0]
                        return value
                    return sampled_const_func

                opt = nlopt.opt(nlopt.LD_MMA, self.input_dim)
                opt.set_lower_bounds(self.bounds[0].tolist())
                opt.set_upper_bounds(self.bounds[1].tolist())
                opt.set_min_objective(sampled_obj_func_for_nlopt)
                for c in range(self.C):
                    opt.add_inequality_constraint(constraint_maker_for_nlopt(c), 0)
                opt.set_xtol_rel(1e-2)
                opt.set_ftol_rel(1e-2)
                opt.set_maxeval(1000)
                opt.set_maxtime(10 / NUM_START)


                for i in range(np.shape(x0)[0]):
                    x = opt.optimize(x0[i])
                    minf = opt.last_optimum_value()
                    res = opt.last_optimize_result()
                    if res > 0:
                        # res
                        success_flag = True
                        for c in range(self.C):
                            const = constraint_maker_for_nlopt(c)
                            if const(x, np.array([])) > 0:
                                success_flag = False
                                break
                        if max_sample_nlopt[j] < - minf and success_flag:
                            max_sample_nlopt[j] = - minf
                            max_inputs_nlopt[j] = np.array(x)
            else:
                # Note that sign of the functions are different to nlopt
                def sampled_obj_func(x):
                    x = np.atleast_2d(x)
                    X_features_1 = self.rbf_features_1.transform(x)
                    X_features_2 = self.rbf_features_2.transform(x)
                    X_features = np.c_[np.kron(self.L_1[0, :], X_features_1), np.kron(self.L_2[0, :], X_features_2)]
                    value = - (X_features.dot(np.c_[self.weights_sample[:,j]]) * self.GPmodel.stds[0] + self.GPmodel.means[0]).ravel()[0]
                    return value

                def constraint_maker(c=0):  # c MUST be an optional keyword argument, else it will not work
                    def sampled_const_func(x):
                        x = np.atleast_2d(x)
                        X_features_1 = self.rbf_features_1.transform(x)
                        X_features_2 = self.rbf_features_2.transform(x)
                        X_features = np.c_[np.kron(self.L_1[c+1, :], X_features_1), np.kron(self.L_2[c+1, :], X_features_2)]
                        value = - (X_features.dot(np.c_[self.weights_sample[:,j]]) * self.GPmodel.stds[c+1] + self.GPmodel.means[c+1] - self.thresholds[c]).ravel()[0]
                        return value
                    return sampled_const_func

                pool_obj = sampled_obj_func(pool_X)
                pool_const = list()
                for c in range(self.C):
                    const = constraint_maker(c)
                    pool_const.append(np.c_[const(pool_X)].T)
                pool_const = np.vstack(pool_const)
                feasible_index = (pool_const - np.c_[self.thresholds]) > 0
                feasible_index = np.all(feasible_index, axis=0)
                if np.any(feasible_index):
                    max_index = np.argmax(pool_obj[feasible_index])
                    max_sample_nlopt[j] = pool_obj[feasible_index][max_index]
                    max_inputs_nlopt[j] = pool_X[feasible_index][max_index]


            if max_inputs_nlopt[j] is None:
                if sampling_approach == 'inf':
                    max_sample_nlopt[j] = - np.infty
                elif sampling_approach == 'MinViolation':
                    max_inputs_nlopt[j], max_sample_nlopt[j] = self._minimize_violation(sample_dim=j, pool_X=pool_X)
                j += 1
            else:
                if pool_X is None:
                    opt.set_xtol_rel(1e-16)
                    opt.set_ftol_rel(1e-16)
                    opt.set_maxeval(0)
                    opt.set_maxtime(1)
                    x = opt.optimize(max_inputs_nlopt[j])
                    minf = opt.last_optimum_value()

                    res = opt.last_optimize_result()
                    if res > 0:
                        success_flag = True
                        for c in range(self.C):
                            const = constraint_maker_for_nlopt(c)
                            if const(x, np.array([])) > 0:
                                success_flag = False
                                break
                        if max_sample_nlopt[j] < - minf and success_flag:
                            max_sample_nlopt[j] = - minf
                            max_inputs_nlopt[j] = np.array(x)
                # print(-minf, max_sample_nlopt[j], -minf - max_sample_nlopt[j])
                j += 1

        print('sampled maximums before cut-off:', max_sample_nlopt)
        #
        if self.max_samples_lower is not None:
            correction_term = self.max_samples_lower + 5 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.stds[0]
            max_sample_nlopt[max_sample_nlopt < correction_term] = correction_term
        return max_sample_nlopt, max_inputs_nlopt


    def _minimize_violation(self, sample_dim=0, pool_X=None):
        if pool_X is None:
            def sum_violate(x):
                x = np.atleast_2d(x)
                X_list = [[]]
                X_list.extend([x for c in range(self.C)])
                outputs = np.vstack(self.sample_path(X_list, sample_dim=sample_dim))
                outputs[outputs > np.c_[self.thresholds]] = 0
                return -np.sum(outputs)

            NUM_START = 100 * np.min([self.input_dim, 10])
            x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
            min_violation = np.inf
            min_x = None
            for i in range(np.shape(x0)[0]):
                res = scipyminimize(sum_violate, x0=x0[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-2})
                if min_violation > res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                    min_violation = res['fun']
                    min_x = res['x']
        else:
            X_list = [[]]
            X_list.extend([pool_X for c in range(self.C)])
            sum_violate = np.hstack(self.sample_path(X_list, sample_dim=sample_dim))
            sum_violate[sum_violate > self.thresholds] = 0
            sum_violate = np.sum(sum_violate, axis=1)
            min_index = np.argmin(sum_violate)
            min_violation = sum_violate[min_index]
            min_x = pool_X[min_index]

        print('min violation:', min_violation)
        X_list = [[] for c in range(self.C+1)]
        X_list[0] = min_x
        return np.array(min_x), self.sample_path(X_list)[0].ravel()[0]

    def sample_path(self, X_list, sample_dim=None):
        '''


        Parameter
        -----------------------
        X_list: list of numpy array
            inputs (C+1 \times N \times input_dim)

        Retrun
        -----------------------
        sampled_outputs: numpy array
            sample_path_list [f_s(X), ..., g_s_C(X) ] (C+1 \times N \times sampling_num)
        '''
        sampled_outputs_list = list()
        if self.model=='independent':
            if sample_dim is None:
                for c in range(self.C+1):
                    if len(X_list[c]) > 0:
                        X_features = self.random_features_list[c].transform(X_list[c])
                        sampled_outputs_list.append(X_features.dot(np.c_[self.weights_sample_list[c]]) * self.GPmodel.stds[c] + self.GPmodel.means[c])
                return sampled_outputs_list
            else:
                for c in range(self.C+1):
                    if len(X_list[c]) > 0:
                        X_features = self.random_features_list[c].transform(X_list[c])
                        sampled_outputs_list.append(X_features.dot(np.c_[self.weights_sample_list[c][:,sample_dim]]) * self.GPmodel.stds[c] + self.GPmodel.means[c])
                return sampled_outputs_list
        elif self.model=='correlated':
            if sample_dim is None:
                for c in range(self.C+1):
                    if len(X_list[c]) > 0:
                        x = np.atleast_2d(X_list[c])
                        X_features_1 = self.rbf_features_1.transform(x)
                        X_features_2 = self.rbf_features_2.transform(x)
                        X_features = np.c_[np.kron(self.L_1[c, :], X_features_1), np.kron(self.L_2[c, :], X_features_2)]
                        value = X_features.dot(self.weights_sample)
                        if self.GPmodel.kernel_name=='rbf':
                            value += self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [c]]).T.dot(self.v)
                        elif self.GPmodel.kernel_name=='linear+rbf':
                            value += self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [[1, c]]]).T.dot(self.v)
                        sampled_outputs_list.append(value * self.GPmodel.stds[c] + self.GPmodel.means[c])
                return sampled_outputs_list
            else:
                for c in range(self.C+1):
                    if len(X_list[c]) > 0:
                        x = np.atleast_2d(X_list[c])
                        X_features_1 = self.rbf_features_1.transform(x)
                        X_features_2 = self.rbf_features_2.transform(x)
                        X_features = np.c_[np.kron(self.L_1[c, :], X_features_1), np.kron(self.L_2[c, :], X_features_2)]
                        value = X_features.dot(np.c_[self.weights_sample[:,sample_dim]]).ravel()
                        if self.GPmodel.kernel_name=='rbf':
                            value += self.v[:,sample_dim].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [c]]))
                        elif self.GPmodel.kernel_name=='linear+rbf':
                            value += self.v[:,sample_dim].dot(self.GPmodel.kern.K(self.GPmodel.X, np.c_[x, [[1, c]]]))
                        sampled_outputs_list.append(value * self.GPmodel.stds[c] + self.GPmodel.means[c])
                return sampled_outputs_list



class MFBO_core(object):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, eval_num, bounds, kernel_bounds, M, cost, GPmodel=None, model_name='MFGP', fidelity_features=None, optimize=True):
        self.model_name = model_name
        if model_name=='MFGP':
            self.fidelity_features = np.c_[np.arange(M)]
            self.fidelity_feature_dim = 1
            self.GPmodel = utils.set_mfgpy_regressor(GPmodel, X_list, Y_list, eval_num, kernel_bounds, M, optimize=optimize)
        elif model_name=='MTGP':
            if fidelity_features is None:
                self.fidelity_features = np.c_[np.arange(M)]
                kernel_bounds = np.c_[kernel_bounds, np.c_[[2., (M-1)*10]]]
            else:
                self.fidelity_features = np.c_[fidelity_features]
            self.fidelity_feature_dim = np.shape(self.fidelity_features)[1]
            self.GPmodel = utils.set_mtgpy_regressor(GPmodel, X_list, Y_list, eval_num, kernel_bounds, M, fidelity_features=self.fidelity_features, fidelity_feature_dim=self.fidelity_feature_dim, optimize=optimize)
        else:
            # if you want to add model, implement here.
            print('Model name is incorrect.')
            exit(1)

        self.M = M
        self.cost = cost
        if np.size(Y_list[-1]) == 0:
            print('There is no observation on highest-fidelity function')
            exit()
            # self.y_max = np.max(np.vstack([Y for Y in Y_list if np.size(Y) > 0]))
        else:
            self.y_max = np.max(Y_list[-1])
        self.input_dim = np.shape(X_list[0])[1]
        self.bounds = bounds
        self.bounds_list = bounds.T.tolist()
        self.sampling_num = 10

        self.inference_point = None
        self.max_inputs = None
        self.preprocessing_time = 0
        self.eval_num = eval_num
        self.unique_X = np.unique(self.GPmodel.X[:,:-1], axis=0)
        self.top_number = 50

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


    @abstractmethod
    def acq_high(self, x):
        pass

    @abstractmethod
    def acq_low(self, x):
        pass

    @abstractmethod
    def acq_low_onepoint(self, x):
        pass

    @abstractmethod
    def next_input(self):
        pass

    def high_minus_predict(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        return self.GPmodel.minus_predict(x)

    def high_minus_predict_gradients(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        return self.GPmodel.minus_predict_gradients(x)[:self.input_dim]

    def _upper_bound(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean + 5*np.sqrt(var)

    def _lower_bound(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean - 5*np.sqrt(var)

    def _bounds(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.matlib.repmat(self.fidelity_features[-1], np.shape(x)[0], 1)]
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean - 5*np.sqrt(var), mean + 5*np.sqrt(var)

    def posteriori_maximum(self):
        num_start = 100
        x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean = -1 * self.high_minus_predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        if self.inference_point is not None:
            x0s = np.r_[x0s, self.inference_point]

        x_min, f_min = utils.minimize(self.high_minus_predict, x0s,self.bounds_list, jac=self.high_minus_predict_gradients)
        self.inference_point = np.atleast_2d(x_min)
        return x_min, -1 * f_min

    def posteriori_maximum_direct(self):
        res = utils.minimize(self.high_minus_predict, self.bounds_list, self.input_dim)
        # print(res)
        return np.atleast_2d(res['x']), -1 * res['fun']

    def sampling_MFRFM(self, pool_X=None, slack = False):
        C = 2
        basis_dim = 100 + np.max(self.eval_num)
        # feature_size = basis_dim * self.M * C
        self.rbf_features_1 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)
        self.rbf_features_2 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul_1.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)

        W = self.GPmodel['.*mul.coregion.W'].values[::-1]
        kappa = self.GPmodel['.*mul.coregion.kappa'].values[::-1]
        rbf_std_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])
        if not all(kappa == np.zeros(self.M)):
            self.L_1 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
        else:
            self.L_1 = W

        W = self.GPmodel['.*mul_1.coregion.W'].values[::-1]
        kappa = self.GPmodel['.*mul_1.coregion.kappa'].values[::-1]
        rbf_std_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])
        if not all(kappa == np.zeros(self.M)):
            self.L_2 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
        else:
            self.L_2 = W

        X_train_features_1 = list([])
        X_train_features_2 = list([])
        #
        for m in range(self.M):
            X_train_features_1_m = rbf_std_1*self.rbf_features_1.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==m, : -1])
            X_train_features_1.append(np.kron(self.L_1[self.M-1-m, :], X_train_features_1_m))
            X_train_features_2_m = rbf_std_2*self.rbf_features_2.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==m, : -1])
            X_train_features_2.append(np.kron(self.L_2[self.M-1-m, :], X_train_features_2_m))
        X_train_features_1 = np.vstack(X_train_features_1)
        X_train_features_2 = np.vstack(X_train_features_2)

        X_train_features = np.c_[X_train_features_1, X_train_features_2]
        del X_train_features_1
        del X_train_features_2
        del X_train_features_1_m
        del X_train_features_2_m

        max_samples = np.zeros(self.sampling_num)
        max_inputs = list([])
        #
        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(np.shape(X_train_features)[1]) * self.GPmodel['.*Gaussian_noise.variance'].values)
        weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot(self.GPmodel.Y_normalized))
        weights_var = A_inv * self.GPmodel['.*Gaussian_noise.variance']

        c = np.zeros(self.M)
        #
        try:
            L = np.linalg.cholesky(weights_var)
        except np.linalg.LinAlgError as e:
            print('In RFM-based sampling,', e)
            L = np.linalg.cholesky(weights_var + 1e-8 * np.eye(np.shape(weights_var)[0]))

        standard_normal_rvs = np.random.normal(0, 1, size=(np.size(weights_mean), self.sampling_num))
        # sampling_num
        self.weights_sample = np.c_[weights_mean] + L.dot(np.c_[standard_normal_rvs])
        del weights_var
        del weights_mean
        del A_inv
        del L
        del X_train_features
        del standard_normal_rvs

        if pool_X is None:
            # num_start = 100 * np.min([self.input_dim, 5])
            num_start = 100 * self.input_dim
            x0s = utils.lhs(self.input_dim, samples=num_start, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]

            if np.shape(self.unique_X)[0] <= self.top_number:
                x0s = np.r_[x0s, self.unique_X]
            else:
                mean = -1 * self.high_minus_predict(self.unique_X)
                mean = mean.ravel()
                top_idx = np.argpartition(mean, -self.top_number)[-self.top_number:]
                x0s = np.r_[x0s, self.unique_X[top_idx]]
        else:
            if np.size(pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]) > 0:
                pool_X = pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]



        self.L_1_highest = self.L_1[0, 0]
        self.L_2_highest = self.L_2[0, 0]
        tmp_index = int(basis_dim*self.M)
        for j in range(self.sampling_num):
            weight_jth = np.r_[self.weights_sample[0:basis_dim, j], self.weights_sample[tmp_index:tmp_index+basis_dim, j]]
            def BLR(x):
                X_features_1 = rbf_std_1*self.rbf_features_1.transform(x)
                X_features_2 = rbf_std_2*self.rbf_features_2.transform(x)
                X_features = np.c_[self.L_1_highest * X_features_1, self.L_2_highest * X_features_2]
                return -1 * (X_features.dot(weight_jth) * self.GPmodel.std + self.GPmodel.mean)

            def BLR_gradients(x):
                X_features_1 = rbf_std_1*self.rbf_features_1.transform_grad(x)
                X_features_2 = rbf_std_2*self.rbf_features_2.transform_grad(x)
                X_features = np.c_[self.L_1_highest * X_features_1, self.L_2_highest * X_features_2]
                return -1 * (X_features.dot(weight_jth) * self.GPmodel.std + self.GPmodel.mean)

            if pool_X is None:
                # start = time.time()
                x_min, f_min = utils.minimize(BLR, x0s, self.bounds_list, jac=BLR_gradients)
                max_samples[j] = -1 * f_min
                max_inputs.append(x_min)
                # print(time.time() - start)
            else:
                pool_Y = BLR(pool_X)
                max_index = np.argmin(pool_Y)
                max_samples[j] = -1 * pool_Y[max_index]
                max_inputs.append(pool_X[max_index])

            if slack:
                for m in range(self.M-1):
                    def BLR_low(x):
                        X_features_1 = rbf_std_1*self.rbf_features_1.transform(x)
                        X_features_2 = rbf_std_2*self.rbf_features_2.transform(x)
                        X_features = np.c_[np.kron(self.L_1[self.M-1-m, :], X_features_1), np.kron(self.L_2[self.M-1-m, :], X_features_2)]
                        return -1 * (X_features.dot(self.weights_sample[:,j]) * self.GPmodel.std + self.GPmodel.mean)

                    if pool_X is None:
                        _, f_min = utils.minimize(BLR, x0s, self.bounds_list, jac=BLR_gradients)
                        c[m] += (-1 * f_min) - (-1 * BLR_low(max_inputs[len(max_inputs)-1]))
                    else:
                        pool_Y = BLR_low(pool_X)
                        c[m] += (-1 * np.min(pool_Y)) - (-1 * BLR_low(max_inputs[len(max_inputs)-1]))
        #
        correction_term = self.y_max + 3 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        max_samples[max_samples < correction_term] = correction_term
        if slack:
            return max_samples, np.array(max_inputs), c.ravel() / self.sampling_num
        else:
            return max_samples, np.array(max_inputs)

    def sample_path_MFRFM(self, X):
        '''


        Parameter
        -----------------------
        X: numpy array
            inputs (N \times input_dim + 1)

        Retrun
        -----------------------
        sampled_outputs: numpy array
            sample_path f_s(X) (N \times sampling_num)
        '''
        X = np.atleast_2d(X)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(X[i,:-1])
            X_features_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(X[i,:-1])
            X_features.append(np.c_[np.kron(self.L_1[self.M-1-np.int(X[i,-1]), :], X_features_1), np.kron(self.L_2[self.M-1-np.int(X[i,-1]), :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.weights_sample) * self.GPmodel.std + self.GPmodel.mean



    # def sampling_MFRFM_dsgp_for_largeM(self, pool_X=None):
    #     feature_size = 500 #
    #     basis_dim = feature_size // 2 # //(2*self.M)
    #     self.rbf_features_1 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)
    #     self.rbf_features_2 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul_1.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)

    #     W = self.GPmodel['.*mul.coregion.W'].values[::-1]
    #     kappa = self.GPmodel['.*mul.coregion.kappa'].values[::-1]
    #     # print(W, kappa)
    #     if not all(kappa == np.zeros(self.M)):
    #         # print(np.linalg.cholesky(W.dot(W.T) + np.diag(kappa)))
    #         self.L_1 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
    #     else:
    #         self.L_1 = W
    #     W = self.GPmodel['.*mul_1.coregion.W'].values[::-1]
    #     kappa = self.GPmodel['.*mul_1.coregion.kappa'].values[::-1]
    #     # print(W, kappa)
    #     if not all(kappa == np.zeros(self.M)):
    #         # print(W.dot(W.T) + np.diag(kappa))
    #         # print(np.linalg.cholesky(W.dot(W.T) + np.diag(kappa)))
    #         self.L_2 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
    #     else:
    #         self.L_2 = W

    #     #
    #     # print(self.L_1, self.L_2)
    #     self.L_1 = self.L_1[::-1]
    #     self.L_2 = self.L_2[::-1]
    #     # print(self.L_1, self.L_2)

    #     X_train_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(self.GPmodel.X[:, : -1])
    #     approx_K_1 = X_train_features_1.dot(X_train_features_1.T)
    #     X_train_features_1 = np.c_[self.L_1[:,0][self.GPmodel.X[:,-1].astype(np.int)]] * X_train_features_1

    #     X_train_features_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(self.GPmodel.X[:, : -1])
    #     approx_K_2 = X_train_features_2.dot(X_train_features_2.T)
    #     X_train_features_2 = np.c_[self.L_2[:,0][self.GPmodel.X[:,-1].astype(np.int)]] * X_train_features_2

    #     X_train_features = np.c_[X_train_features_1, X_train_features_2]
    #     # print(np.shape(X_train_features))

    #     B_1 = self.L_1.dot(self.L_1.T)
    #     B_2 = self.L_2.dot(self.L_2.T)

    #     fide_num_1, fide_num_2 = np.meshgrid(self.GPmodel.X[:,-1], self.GPmodel.X[:,-1])
    #     fide_num_1 = fide_num_1.astype(np.int)
    #     fide_num_2 = fide_num_2.astype(np.int)
    #     approx_K = B_1[fide_num_1.ravel(), fide_num_2.ravel()].reshape(np.shape(approx_K_1)) * approx_K_1
    #     approx_K += B_2[fide_num_1.ravel(), fide_num_2.ravel()].reshape(np.shape(approx_K_1)) * approx_K_2


    #     low_fidelity_num = np.size(np.where(self.GPmodel.X[:,-1] < self.M-1)[0])
    #     X_train_cov = approx_K + self.GPmodel['.*Gaussian_noise.variance'] * np.eye(np.shape(self.GPmodel.X)[0])


    #     self.sampled_weights_for_highest = np.random.normal(0, 1, size=(feature_size, self.sampling_num))
    #     low_train_cov = X_train_cov[:low_fidelity_num, :low_fidelity_num] - X_train_features[:low_fidelity_num,:].dot(X_train_features[:low_fidelity_num,:].T)
    #     low_train_mean = X_train_features[:low_fidelity_num,:].dot(self.sampled_weights_for_highest)

    #     # # non-singular check
    #     # np.set_printoptions(precision=3)
    #     # print(low_train_cov)
    #     # print(X_train_cov[:low_fidelity_num, :low_fidelity_num])
    #     # print(X_train_features[:low_fidelity_num,:].dot(X_train_features[:low_fidelity_num,:].T))
    #     # print(np.sort(np.linalg.eigvals(low_train_cov)))
    #     # exit()

    #     low_train_cov_chol = np.linalg.cholesky(low_train_cov)
    #     sampled_low_train_observations = low_train_mean + low_train_cov_chol.dot(np.random.normal(0, 1, size=(low_fidelity_num, self.sampling_num)))
    #     sampled_high_train_observations = X_train_features[low_fidelity_num:,:].dot(self.sampled_weights_for_highest) + np.sqrt(self.GPmodel['.*Gaussian_noise.variance']) * np.random.normal(0, 1, size=(np.shape(self.GPmodel.X)[0] - low_fidelity_num, self.sampling_num))

    #     sampled_train_observations = np.r_[sampled_low_train_observations, sampled_high_train_observations]
    #     self.observations = self.GPmodel.Y_normalized.copy()

    #     max_samples = np.zeros(self.sampling_num)
    #     max_inputs = list([])
    #     self.X_train_cov_inv = np.linalg.inv(self.GPmodel.kern.K(self.GPmodel.X) + self.GPmodel['.*Gaussian_noise.variance'] * np.eye(np.shape(self.GPmodel.X)[0]))
    #     # plot_x = np.linspace(0,1,100)
    #     for j in range(self.sampling_num):
    #         def sampled_function(x):
    #             x = np.atleast_2d(x)
    #             X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(x)
    #             X_features_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(x)
    #             X_features = np.c_[self.L_1[self.M-1,0] * X_features_1, self.L_2[self.M-1,0] * X_features_2]
    #             prior = X_features.dot(self.sampled_weights_for_highest[:,j])
    #             update = self.GPmodel.kern.K(np.c_[x, np.matlib.repmat([self.M-1],np.shape(x)[0], 1)], self.GPmodel.X).dot(self.X_train_cov_inv).dot(self.observations - np.c_[sampled_train_observations[:,j]]).ravel()
    #             return -1 * ((prior + update)* self.GPmodel.std + self.GPmodel.mean)

    #         # print(sampled_function([0.5,0.5]))
    #         if pool_X is None:
    #             # DIRECT
    #             result = utils.minimize(sampled_function, self.bounds_list, self.input_dim)
    #             max_samples[j] = -1 * result['fun']
    #             max_inputs.append(result['x'])
    #         else:
    #             pool_Y = sampled_function(pool_X)
    #             max_index = np.argmin(pool_Y)
    #             max_samples[j] = -1 * pool_Y[max_index]
    #             max_inputs.append(pool_X[max_index])
    #     # # for debug
    #     #     plt.plot(plot_x, - sampled_function(np.c_[plot_x]))
    #     # mean, var = self.GPmodel.predict_noiseless(np.c_[np.c_[plot_x], np.matlib.repmat([self.M-1],np.size(plot_x), 1)])
    #     # mean = mean.ravel()
    #     # var = var.ravel()
    #     # plt.plot(plot_x,mean, label='mean')
    #     # plt.fill_between(plot_x, mean - 2*np.sqrt(var), mean + 2*np.sqrt(var), alpha=0.3)
    #     # plt.legend(loc='best')
    #     # plt.savefig('test.png')

    #     #
    #     max_samples[max_samples < self.y_max + 5 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = self.y_max + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
    #     return max_samples, np.array(max_inputs)


    # def sampling_MFRFM_dsgp(self, pool_X=None):
    #     feature_size = 1000
    #     KERNEL_NUM = 2
    #     basis_dim = feature_size //(KERNEL_NUM*self.M)
    #     self.rbf_features_1 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)
    #     self.rbf_features_2 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul_1.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim)

    #     W = self.GPmodel['.*mul.coregion.W'].values[::-1]
    #     kappa = self.GPmodel['.*mul.coregion.kappa'].values[::-1]
    #     # print(W, kappa)
    #     if not all(kappa == np.zeros(self.M)):
    #         # print(np.linalg.cholesky(W.dot(W.T) + np.diag(kappa)))
    #         self.L_1 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
    #     else:
    #         self.L_1 = W
    #     W = self.GPmodel['.*mul_1.coregion.W'].values[::-1]
    #     kappa = self.GPmodel['.*mul_1.coregion.kappa'].values[::-1]
    #     # print(W, kappa)
    #     if not all(kappa == np.zeros(self.M)):
    #         # print(W.dot(W.T) + np.diag(kappa))
    #         # print(np.linalg.cholesky(W.dot(W.T) + np.diag(kappa)))
    #         self.L_2 = np.linalg.cholesky(W.dot(W.T) + np.diag(kappa))
    #     else:
    #         self.L_2 = W
    #     #
    #     self.L_1 = self.L_1[::-1]
    #     self.L_2 = self.L_2[::-1]


    #     X_train_features_1 = list([])
    #     X_train_features_2 = list([])
    #     #
    #     for m in range(self.M):
    #         X_train_features_1_m = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==m, : -1])
    #         X_train_features_1.append(np.kron(self.L_1[m, :], X_train_features_1_m))
    #         X_train_features_2_m = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(self.GPmodel.X[self.GPmodel.X[:,-1]==m, : -1])
    #         X_train_features_2.append(np.kron(self.L_2[m, :], X_train_features_2_m))
    #     X_train_features_1 = np.vstack(X_train_features_1)
    #     X_train_features_2 = np.vstack(X_train_features_2)
    #     X_train_features = np.c_[X_train_features_1, X_train_features_2]

    #     max_samples = np.zeros(self.sampling_num)
    #     max_inputs = list([])
    #     self.X_train_cov_inv = np.linalg.inv(self.GPmodel.posterior._K + self.GPmodel['.*Gaussian_noise.variance'] * np.eye(np.shape(self.GPmodel.X)[0]))
    #     self.X_train_cov_inv_chol = np.linalg.cholesky(self.X_train_cov_inv)

    #     self.sampled_weights = np.random.normal(0, 1, size=(self.M*KERNEL_NUM*basis_dim, self.sampling_num))
    #     diff_observation_blr = self.GPmodel.Y_normalized - X_train_features.dot(self.sampled_weights) + np.sqrt(self.GPmodel['.*Gaussian_noise.variance'])*np.c_[np.random.normal(0, 1, size=(np.shape(self.GPmodel.Y)[0], self.sampling_num))]
    #     self.update_latter_vectors = self.X_train_cov_inv_chol.T.dot(diff_observation_blr)
    #     # plot_x = np.linspace(0,1,100)
    #     for j in range(self.sampling_num):
    #         def sampled_function(x):
    #             x = np.atleast_2d(x)
    #             X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(x)
    #             X_features_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(x)
    #             X_features = np.c_[self.L_1[self.M-1,0] * X_features_1, self.L_2[self.M-1,0] * X_features_2]
    #             prior = X_features.dot(self.sampled_weights[:KERNEL_NUM*basis_dim,j])

    #             x = np.c_[x, np.matlib.repmat([self.M-1],np.shape(x)[0], 1)]
    #             KxX = self.GPmodel.kern.K(x, self.GPmodel.X)
    #             temp = np.dot(KxX, self.X_train_cov_inv_chol)
    #             update = temp.dot(self.update_latter_vectors[:,j])
    #             # update = self.GPmodel.kern.K(x, self.GPmodel.X).dot(self.X_train_cov_inv).dot(diff_observation_blr[:,j]).ravel()
    #             return -1 * ((prior + update)* self.GPmodel.std + self.GPmodel.mean)

    #         # start = time.time()
    #         # print(sampled_function([0.5,0.5,0.5]))
    #         # print("time:", time.time()-start)
    #         # exit()
    #         if pool_X is None:
    #             # DIRECT
    #             result = utils.minimize(sampled_function, self.bounds_list, self.input_dim)
    #             max_samples[j] = -1 * result['fun']
    #             max_inputs.append(result['x'])
    #         else:
    #             pool_Y = sampled_function(pool_X)
    #             max_index = np.argmin(pool_Y)
    #             max_samples[j] = -1 * pool_Y[max_index]
    #             max_inputs.append(pool_X[max_index])
    #     # # for debug
    #     #     plt.plot(plot_x, - sampled_function(np.c_[plot_x]))
    #     # mean, var = self.GPmodel.predict_noiseless(np.c_[np.c_[plot_x], np.matlib.repmat([self.M-1],np.size(plot_x), 1)])
    #     # mean = mean.ravel()
    #     # var = var.ravel()
    #     # plt.plot(plot_x,mean, label='mean')
    #     # plt.fill_between(plot_x, mean - 2*np.sqrt(var), mean + 2*np.sqrt(var), alpha=0.3)
    #     # plt.legend(loc='best')
    #     # plt.savefig('test.png')

    #     #
    #     max_samples[max_samples < self.y_max + 5 * np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = self.y_max + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
    #     return max_samples, np.array(max_inputs)


    # def sample_path_MFRFM_dsgp(self, X):
    #     '''
    #

    #     Parameter
    #     -----------------------
    #     X: numpy array
    #         inputs (N \times input_dim + 1)

    #     Retrun
    #     -----------------------
    #     sampled_outputs: numpy array
    #         sample_path f_s(X) (N \times sampling_num)
    #     '''
    #     X = np.atleast_2d(X)
    #     X_features = list()
    #     for i in range(np.shape(X)[0]):
    #         X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(X[i,:-1])
    #         X_features_2 = np.sqrt(self.GPmodel['.*mul_1.rbf.variance'])*self.rbf_features_2.transform(X[i,:-1])
    #         X_features.append(np.c_[np.kron(self.L_1[np.int(X[i,-1]), :], X_features_1), np.kron(self.L_2[np.int(X[i,-1]), :], X_features_2)])
    #     X_features = np.vstack(X_features)
    #     return X_features.dot(self.weights_sample) * self.GPmodel.std + self.GPmodel.mean




    # def sampling_MTRFM(self, pool_X=None, slack = False):
    #     basis_dim = [200, 5]
    #     feature_size = np.product(basis_dim)
    #     self.rbf_features_1 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf.lengthscale'].values, input_dim=self.input_dim, basis_dim=basis_dim[0])
    #     self.rbf_features_2 = utils.RFM_RBF(lengthscales=self.GPmodel['.*mul.rbf_1.lengthscale'].values, input_dim=self.fidelity_feature_dim, basis_dim=basis_dim[1])

    #     X_train_features = np.empty(shape=(1,feature_size))
    #     X_train_features = list([])
    #     for m in range(self.M):
    #         X_train_features_1_m = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(self.GPmodel.X[np.all(self.GPmodel.X[:,self.input_dim:]==self.fidelity_features[m], axis=1), :self.input_dim])
    #         X_train_features_2_m = np.sqrt(self.GPmodel['.*mul.rbf_1.variance'])*self.rbf_features_2.transform(np.atleast_2d(self.fidelity_features[m]))
    #         X_train_features.append(np.kron(X_train_features_1_m, X_train_features_2_m))
    #     X_train_features = np.vstack(X_train_features)

    #     max_samples = np.zeros(self.sampling_num)
    #     max_inputs = list([])
    #     #
    #     A_inv = np.linalg.inv((X_train_features.T).dot(
    #         X_train_features) + np.eye(np.shape(X_train_features)[1]) * self.GPmodel['.*Gaussian_noise.variance'])
    #     weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot(self.GPmodel.Y_normalized))
    #     weights_var = A_inv * self.GPmodel['.*Gaussian_noise.variance'].values

    #     c = np.zeros(self.M)
    #     #
    #     L = np.linalg.cholesky(weights_var)
    #     standard_normal_rvs = np.random.normal(0, 1, size=(np.size(weights_mean), self.sampling_num))
    #     self.weights_sample = np.c_[weights_mean] + L.dot(np.c_[standard_normal_rvs])
    #     for j in range(self.sampling_num):
    #         def bayesian_linear_regression(x):
    #             X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(x)
    #             X_features_2 = np.sqrt(self.GPmodel['.*mul.rbf_1.variance'])*self.rbf_features_2.transform(self.fidelity_features[-1])
    #             X_features = np.kron(X_features_1, X_features_2)
    #             return -1 * (X_features.dot(self.weights_sample[:,j]) * self.GPmodel.std + self.GPmodel.mean)

    #         if pool_X is None:
    #             # DIRECT
    #             result = utils.minimize(bayesian_linear_regression, self.bounds_list, self.input_dim)
    #             max_samples[j] = -1 * result['fun']
    #             max_inputs.append(result['x'])
    #         else:
    #             pool_Y = bayesian_linear_regression(pool_X)
    #             max_index = np.argmin(pool_Y)
    #             max_samples[j] = -1 * pool_Y[max_index]
    #             max_inputs.append(pool_X[max_index])

    #         if slack:
    #             for m in range(self.M-1):
    #                 def bayesian_linear_regression_low(x):
    #                     X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(x)
    #                     X_features_2 = np.sqrt(self.GPmodel['.*mul.rbf_1.variance'])*self.rbf_features_2.transform(self.fidelity_features[m])
    #                     X_features = np.kron(X_features_1, X_features_2)
    #                     return -1 * (X_features.dot(self.weights_sample[:,j]) * self.GPmodel.std + self.GPmodel.mean)

    #                 if pool_X is None:
    #                     # DIRECT
    #                     result = utils.minimize(bayesian_linear_regression_low, self.bounds_list, self.input_dim)
    #                     c[m] += (-1 * result['fun']) - (-1 * bayesian_linear_regression_low(max_inputs[len(max_inputs)-1]))
    #                 else:
    #                     pool_Y = bayesian_linear_regression_low(pool_X)
    #                     c[m] += (-1 * np.min(pool_Y)) - (-1 * bayesian_linear_regression_low(max_inputs[len(max_inputs)-1]))

    #     #
    #     max_samples[max_samples < self.y_max + 5 *
    #                 np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = self.y_max + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
    #     if slack:
    #         return max_samples, np.array(max_inputs), c.ravel() / self.sampling_num
    #     else:
    #         return max_samples, np.array(max_inputs)

    # def sample_path_MTRFM(self, X):
    #     '''
    #

    #     Parameter
    #     -----------------------
    #     X: numpy array
    #         inputs (N \times input_dim + fidelity_feature_dim)

    #     Retrun
    #     -----------------------
    #     sampled_outputs: numpy array
    #         sample_path f_s(X) (N \times sampling_num)
    #     '''
    #     X = np.atleast_2d(X)
    #     X_features = list()
    #     for i in range(np.shape(X)[0]):
    #         X_features_1 = np.sqrt(self.GPmodel['.*mul.rbf.variance'])*self.rbf_features_1.transform(X[i,:-1])
    #         X_features_2 = np.sqrt(self.GPmodel['.*mul.rbf_1.variance'])*self.rbf_features_2.transform(self.fidelity_features[np.int(X[i,-1])])
    #         X_features.append(np.kron(X_features_1, X_features_2))
    #     X_features = np.vstack(X_features)
    #     return -1 * (X_features.dot(self.weights_sample) * self.GPmodel.std + self.GPmodel.mean)

    #cdf(Y) = val
    #thres
    def _find_r(self, val, cdf_func, R, Y, thres):
        current_r_pos = np.argmin(np.abs(val - R))
        if (np.abs(val - R[current_r_pos])) < thres:
            return Y[current_r_pos]

        if R[current_r_pos] > val:
            left = Y[current_r_pos - 1]
            right = Y[current_r_pos]
        else:
            left = Y[current_r_pos]
            right = Y[current_r_pos + 1]

        for _ in range(10000):
            mid = (left + right)/2.
            mid_r = cdf_func(mid)

            if (np.abs(val - mid_r)) < thres:
                return mid

            if mid_r > val:
                right = mid
            else:
                left = mid

        print('r='+str(val)+': error over')

        return mid

    #
    def sampling_gumbel(self, pool_X=None):
        if pool_X is None:
            # for GIBBON
            RANDOM_INPUT_NUM = 10000 * self.input_dim
            # if self.input_dim == 1:
            #     RANDOM_INPUT_NUM = 100
            # else:
            #     RANDOM_INPUT_NUM = 10000
            x = self.bounds[0, :] + (self.bounds[1, :] - self.bounds[0, :])*np.random.rand(RANDOM_INPUT_NUM, self.input_dim)
            x = np.r_[np.c_[x, np.c_[(self.fidelity_features[-1])*np.ones((RANDOM_INPUT_NUM, self.fidelity_feature_dim))]], self.GPmodel.X]
        else:
            pool_X = np.atleast_2d(pool_X)
            pool_X = np.c_[pool_X, np.matlib.repmat(self.fidelity_features[-1], np.shape(pool_X)[0], 1)]
            x = pool_X
        mean, var = self.GPmodel.predict(x)
        std = np.sqrt(var)

        def approx_gumbel_cdf(y):
            return np.prod(norm.cdf((y-np.c_[mean])/np.c_[std]), axis=0)

        #
        left = self.y_max
        if approx_gumbel_cdf(left) < 0.25:
            right = np.max(mean + 5*std)
            while (approx_gumbel_cdf(right) < 0.75):
                right = 2*right - left

            Y = np.c_[np.linspace(left, right, 100)].T
            R = approx_gumbel_cdf(Y)
            Y = np.ravel(Y)

            # r = 0.25, 0.5, 0.75
            med = self._find_r(0.5, approx_gumbel_cdf, R, Y, 0.01)
            y1 = self._find_r(0.25, approx_gumbel_cdf, R, Y, 0.01)
            y2 = self._find_r(0.75, approx_gumbel_cdf, R, Y, 0.01)

            self.b = (y1 - y2)/(np.log(np.log(4/3)) - np.log(np.log(4)))
            self.a = med + self.b*np.log(np.log(2))

            # print('y1 and y2 is:'+str(y1)+','+str(y2))
            # print('gumbel parameters:a='+str(self.a) + ', b=' + str(self.b))

            max_samples = np.array(np.random.gumbel(self.a, self.b, self.sampling_num))
            max_samples[max_samples < left + 3 *
                        np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = left + 3*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
        else:
            print('Error: observation is larger than maximum!?')
            max_samples = (left + 3*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
                           ) * np.ones(self.sampling_num)
        return max_samples


    # # Random
    # def sampling_test(self, pool_X=None):
    #     if pool_X is None:
    #         RANDOM_INPUT_NUM = 10000
    #         x = self.bounds[0, :] + (self.bounds[1, :] - self.bounds[0, :])*np.random.rand(RANDOM_INPUT_NUM, self.input_dim)
    #         x = np.r_[np.c_[x, np.c_[(self.fidelity_features[-1])*np.ones((RANDOM_INPUT_NUM, self.fidelity_feature_dim))]], self.GPmodel.X]
    #     else:
    #         pool_X = np.atleast_2d(pool_X)
    #         pool_X = np.c_[pool_X, np.matlib.repmat(self.fidelity_features[-1], np.shape(pool_X)[0], 1)]
    #         x = pool_X
    #     mean, cov = self.GPmodel.predict(x, full_cov=True)
    #     cov_chol = np.linalg.cholesky(cov)
    #     random_normal = np.random.normal(size=(RANDOM_INPUT_NUM+np.shape(self.GPmodel.X)[0], self.sampling_num))

    #     max_samples = mean + cov_chol.dot(random_normal)
    #     max_samples = np.max(max_samples, axis=0)
    #     max_samples[max_samples < self.y_max + 5 *
    #                 np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std] = self.y_max + 5*np.sqrt(self.GPmodel['.*Gaussian_noise.variance'].values) * self.GPmodel.std
    #     return max_samples
