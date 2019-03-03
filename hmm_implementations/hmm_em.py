import numpy as np
import copy
from alpha_beta import q2_initialize, k_gaussians, alpha_recur_stable, beta_recur_stable


class hmm_gaussian:

    # def __init__(self, k):
    # self.k = k
    # self.A = np.zeros(k, k)
    # self.pi = np.zeros(k, 1)

    def initialize(self):
        """ shapes :
                self.mus : (k, d)
                self.sigmas : (k, d, d)
                self.pi : (k,)
                self.A : (k, k)
        """
        self.mus, self.sigmas, self.pi, self.A = q2_initialize()
        self.k = self.mus.shape[0]
        self.d = self.mus.shape[1]

    def alpha_beta_recursion(self, X, is_training_data=True):
        """ Use numerically stable alpha-beta recursion to 
            calculate the smoothings and pair_marginals values
        """
        pi, A, mu, sigma = (self.pi, self.A, self.mus, self.sigmas)
        n = X.shape[0]
        k = mu.shape[0]

        alpha_norm, _ = alpha_recur_stable(X, pi, A, mu, sigma)
        beta_norm, _ = beta_recur_stable(X, A, mu, sigma)

        log_smooting_distr = (
            alpha_norm
            + beta_norm
            - np.log(np.sum(np.exp(alpha_norm + beta_norm), axis=1)).reshape(-1, 1)
        )

        pair_marginals = np.zeros((n - 1, k, k))

        for t in range(n - 1):
            o = k_gaussians(X[t + 1], mu, sigma)
            pair_marginals_ = np.log(
                np.exp(alpha_norm[t].reshape(-1, 1)).dot(
                    (np.exp(beta_norm[t + 1].reshape(-1, 1)) * o).T
                )
                * A
            )
            pair_marginals[t] = pair_marginals_ - np.log(
                np.sum(np.exp(pair_marginals_))
            )

        if is_training_data:
            self.taus = np.exp(log_smooting_distr)
            self.zeta = np.exp(pair_marginals)

        return np.exp(log_smooting_distr), np.exp(pair_marginals)

    def log_lh(self, data):
        n, d = data.shape
        k = self.k

        taus, zeta = self.alpha_beta_recursion(data, is_training_data=False)

        # for pi
        llh1 = (taus[0] * np.log(self.pi)).sum()

        # gaussian
        o = np.zeros((n, k))
        for t in range(n):
            o[t] = k_gaussians(data[t], self.mus, self.sigmas).reshape(-1)
        llh2 = (taus * np.log(o)).sum()

        # for A related
        llh3 = (zeta * np.log(self.A)).sum()

        return llh1 + llh2 + llh3

    def e_step(self, data):
        n, d = data.shape
        self.alpha_beta_recursion(data, is_training_data=True)

    def m_step(self, data):
        n, d = data.shape

        self.pi = self.taus[0]  # .reshape(-1, 1)

        self.mus = np.array(
            [np.expand_dims(tau, axis=1) * data for tau in self.taus.T]
        ).sum(axis=1) / self.taus.T.sum(axis=1, keepdims=True)

        data_c = np.array([data - mu for mu in self.mus])

        s = np.array(
            [
                [
                    np.expand_dims(xc_ij, axis=1) @ np.expand_dims(xc_ij, axis=1).T
                    for xc_ij in xc_i
                ]
                for xc_i in data_c
            ]
        )
        self.sigmas = (self.taus.T.reshape(self.k, n, 1, 1) * s).sum(
            axis=1
        ) / self.taus.sum(axis=0).reshape(self.k, 1, 1)

        A_denom = np.array(
            [np.sum(np.sum(self.zeta, axis=0), axis=1) for _ in range(self.k)]
        ).T
        self.A = np.sum(self.zeta, axis=0) / A_denom

    def run_em(self, data, test_data):

        self.initialize()

        prev_llh, llh = (-1e7, 0)
        train_llhs, test_llhs = ([], [])
        for i in range(10000):
            self.e_step(data)

            llh = self.log_lh(data)
            train_llhs.append(llh)
            test_llhs.append(self.log_lh(test_data))

            if llh - prev_llh < 1e-6:
                self.train_llhs = train_llhs
                self.test_llhs = test_llhs
                break
            prev_llh = llh

            self.m_step(data)

        print(
            "EM converged after {} iterations. \
                Final train log_lh = {} and test log_lh = {}".format(
                i, llh, self.test_llhs[-1]
            )
        )

    def viterbi_decode(self, data):
        """ viterbi algorithm for finding posterior
            Using dynamic programming - 
                dp (k,): store max values of sub problem
                dp_chains (list of k lists, each of length t): store the 4 chain of most-likely states
        """
        n, d = data.shape
        k = self.k

        # o - gaussian emission probabilities for 0th time
        o = k_gaussians(data[0], self.mus, self.sigmas).reshape(-1)

        # DP initialization
        dp = np.log(self.pi) + np.log(o)
        dp_chains = [[i] for i in range(k)]

        for t in range(1, n):
            # o (4,): gaussian emission probabilities
            o = k_gaussians(data[t], self.mus, self.sigmas).reshape(-1)

            argmax_states = np.argmax(np.log(self.A.T) + dp, axis=1)

            dp_chains = [copy.deepcopy(dp_chains[ami]) for ami in argmax_states]
            for i in range(k):
                dp_chains[i].append(i)

            dp = np.log(o) + np.max(np.log(self.A.T) + dp, axis=1)

        # print(dp)

        self.viterbi_decoding = np.array(dp_chains[np.argmax(dp)])
        return self.viterbi_decoding

