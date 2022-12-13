import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.stats import mvn
from scipy.stats import norm

def onesided_marginal_density_tmvn(x, mu, sigma, Z, L, upper):
    marginal_dim = int(np.size(x))

    if marginal_dim == L:
        return multivariate_normal.pdf(x, mean=mu, cov=sigma) / Z
    elif np.all(np.isinf(upper[marginal_dim:])):
        return multivariate_normal.pdf(x, mean=mu[:marginal_dim], cov=sigma[:marginal_dim, :marginal_dim]) / Z
    else:
        cov_inv = np.linalg.inv(sigma[:marginal_dim, :marginal_dim])
        temp = sigma[marginal_dim:, :marginal_dim].dot(cov_inv)
        conditional_sigma = sigma[marginal_dim:, marginal_dim:] - \
            temp.dot(sigma[marginal_dim:, :marginal_dim].T)
        conditional_mu = mu[marginal_dim:] + \
            temp.dot(np.c_[x - mu[:marginal_dim]]).ravel()

        pdf = multivariate_normal.pdf(x, mean=mu[:marginal_dim], cov=sigma[:marginal_dim, :marginal_dim])
        index = np.logical_not(np.isinf(upper[marginal_dim:]))
        num_notinf = np.count_nonzero(index)

        conditional_Z_m, _ = mvn.mvnun(-np.infty*np.ones(num_notinf), upper[marginal_dim:][index], conditional_mu.ravel()[index], conditional_sigma[np.ix_(index, index)], maxpts=(L - marginal_dim)*1e4, abseps=1e-5, releps=1e-3)

        if np.isnan(conditional_Z_m):
            print('conditional_Z_m is NANs')
            print(conditional_sigma)
            print(np.linalg.inv(conditional_sigma))
            print(upper[marginal_dim:])
            print(conditional_mu.ravel())
    return pdf * conditional_Z_m / Z

def onesided_tmvn_params(upper, mu, sigma, Z, L):
    '''
    return means and covariance matrices of truncated multi-variate normal distribution truncated by upper vector.
    if there is a element which is not truncated, please put np.inf in upper.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimensional multi-variate normal (L)
    sigma : numpy array
        covariance matrix of L-dimensional original multi-variate normal (L \times L)
    upper : numpy array
        upper of truncated normal distribution (L)
    Z : float
        normalizer constant of truncated normal
    L : int
        dimension

    Retruns
    -------
    # mu_TN : numpy array
    #     means of truncated multi-variate normal (L)
    # sigma_TN : numpy array
    #     covariance matrices of truncated multi-variate normal (L \times L)
    # d   : numpy array (L)
    #     difference of original mean and mean of truncated multi-variate normal

    sigma_plus_dd : numpy array
        sigma_TN + d d^\top
    '''
    upper = upper - mu
    zero_mean = np.zeros(L)
    # d = np.zeros(L)
    sigma_plus_dd = sigma.copy()
    F_k_q_bk_bq = np.empty(shape=(L,L))

    for k in range(L):
        if np.isinf(upper[k]):
            continue
        else:
            dims = [k]
            index_list = dims + [i for i in range(L) if i not in dims]
            temp_sigma = sigma[index_list, :][:, index_list]
            temp_upper = upper[index_list]
            F_k_bk = onesided_marginal_density_tmvn(upper[k], zero_mean, temp_sigma, Z, L, temp_upper)
            # d -= F_k_bk * sigma[:, k]
            sigma_plus_dd -= upper[k]*F_k_bk * np.c_[sigma[:, k]] * sigma[:, k] / sigma[k, k]
            latter_sum = 0
            for q in range(L):
                if q != k:
                    if np.isinf(upper[q]):
                        continue
                    else:
                        fir_term = sigma[:, q] - sigma[k, q] * sigma[:, k] / sigma[k, k]

                        dims = [k, q]
                        index_list = dims + [i for i in range(L) if i not in dims]
                        temp_sigma = sigma[index_list, :][:, index_list]
                        temp_upper = upper[index_list]

                        if q > k:
                            F_k_q_bk_bq[k, q] = onesided_marginal_density_tmvn(upper[dims], zero_mean, temp_sigma, Z, L, temp_upper)
                            F_k_q_bk_bq[q, k] = F_k_q_bk_bq[k, q]

                        latter_sum += fir_term * F_k_q_bk_bq[k,q]
        sigma_plus_dd += np.c_[sigma[:, k]] * latter_sum

    #
    # mu_TN = mu + d
    # d = np.c_[d]
    # sigma_TN = sigma_plus_dd - d.dot(d.T)
    # return mu_TN, sigma_TN, d

    return sigma_plus_dd



def main():
    np.random.seed(0)
    f_stars = 4.0*np.ones(4) + 0.1*np.random.randn(4)
    f_stars = [4.]
    L = 2

    mu = np.array([2., 3.])
    sigma = np.diag(np.ones(L))
    f_star_truncated_entropy = 0
    f_star_q_truncated_entropy = 0

    #--------------------------------------------------------------------------------------------------------------
    L = 3
    C = np.c_[[1, -1]].T
    c_mu = C.dot(mu)
    c_sigma = C.dot(sigma).dot(C.T)
    Z_1 = norm.cdf((0 - c_mu) / np.sqrt(c_sigma))[0,0]
    # print(Z_1)
    H_q = - Z_1 * np.log(Z_1) - (1-Z_1) * np.log((1-Z_1))

    #--------------------------------------------------------------------------------------------------------------

    for f_star in f_stars:
        L=2
        upper = np.array([f_star, f_star])

        NotInfIndex = np.logical_not(np.isinf(upper))
        Z_f_star, _ = mvn.mvnun((-np.infty*np.ones(L))[NotInfIndex], upper[NotInfIndex], mu[NotInfIndex], sigma[np.ix_(NotInfIndex, NotInfIndex)], maxpts=L * 1e4, abseps=1e-8, releps=1e-6)
        sigma_plus_dd = onesided_tmvn_params(upper, mu, sigma, Z_f_star, L)

        f_star_truncated_entropy += np.trace(sigma.dot(sigma_plus_dd))/2. + np.log(Z_f_star)
        print(f_star_truncated_entropy)

        #--------------------------------------------------------------------------------------------------------------
        L=3
        upper = np.array([f_star, f_star, 0])

        C = np.array([[1, 0],[0, 1],[1, -1]])
        c_mu = C.dot(mu)
        c_sigma = C.dot(sigma).dot(C.T)

        NotInfIndex = np.logical_not(np.isinf(upper))
        Z, _ = mvn.mvnun((-np.infty*np.ones(L))[NotInfIndex], upper[NotInfIndex], c_mu[NotInfIndex], c_sigma[np.ix_(NotInfIndex, NotInfIndex)], maxpts=L * 1e4, abseps=1e-8, releps=1e-6)
        sigma_plus_dd = onesided_tmvn_params(upper, c_mu, c_sigma, Z, L)
        f_star_q1_truncated_entropy = np.trace(sigma.dot(sigma_plus_dd[:2,:2]))/2. + np.log(Z)
        print(f_star_q1_truncated_entropy, Z_1)
        f_star_q_truncated_entropy += (Z_1) * f_star_q1_truncated_entropy

        #--------------------------------------------------
        C = np.array([[1, 0],[0, 1],[-1, 1]])
        c_mu = C.dot(mu)
        c_sigma = C.dot(sigma).dot(C.T)

        NotInfIndex = np.logical_not(np.isinf(upper))
        Z, _ = mvn.mvnun((-np.infty*np.ones(L))[NotInfIndex], upper[NotInfIndex], c_mu[NotInfIndex], c_sigma[np.ix_(NotInfIndex, NotInfIndex)], maxpts=L * 1e4, abseps=1e-8, releps=1e-6)
        sigma_plus_dd = onesided_tmvn_params(upper, c_mu, c_sigma, Z, L)

        f_star_q0_truncated_entropy = np.trace(sigma.dot(sigma_plus_dd[:2,:2]))/2. + np.log(Z)
        print(f_star_q0_truncated_entropy, 1-Z_1)
        f_star_q_truncated_entropy += (1-Z_1) * f_star_q0_truncated_entropy


    f_star_truncated_entropy /= np.size(f_stars)
    f_star_q_truncated_entropy /= np.size(f_stars)
    print(H_q)
    print(f_star_truncated_entropy)
    print(f_star_q_truncated_entropy)
    print('entropy', H_q - (f_star_truncated_entropy - f_star_q_truncated_entropy))


if __name__ == '__main__':
    main()
