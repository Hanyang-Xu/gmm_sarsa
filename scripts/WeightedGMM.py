import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import scipy.stats as stats

class WeightedGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, sample_weights=None):
        n_samples, n_features = X.shape
        sample_weights = np.ones(n_samples) if sample_weights is None else sample_weights
        sample_weights /= np.sum(sample_weights)

        np.random.seed(42)
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        for _ in range(self.max_iter):
            # E-step
            log_resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_resp[:, k] = np.log(self.weights_[k] + 1e-10) + self._log_gaussian_pdf(X, self.means_[k], self.covariances_[k])

            log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(np.clip(log_resp, -700, 700))

            # M-step
            for k in range(self.n_components):
                gamma_w = sample_weights * resp[:, k]
                Nk = np.sum(gamma_w)

                self.means_[k] = np.sum(gamma_w[:, None] * X, axis=0) / Nk
                diff = X - self.means_[k]
                self.covariances_[k] = (diff.T * gamma_w) @ diff / Nk + 1e-6 * np.eye(n_features)
                self.weights_[k] = Nk

    def _log_gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        
        cov += 1e-6 * np.eye(cov.shape[0])
        
        cov_inv = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov) + 1e-10)
        diff = X - mean
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + np.sum(diff @ cov_inv * diff, axis=1))

    def conditional_sample(self, states):
        state_dim = states.shape[1]
        action_dim = self.means_.shape[1] - state_dim
        n_samples = states.shape[0]

        log_component_probs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            mu_s = self.means_[k, :state_dim]
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6 * np.eye(state_dim)
            Sigma_ss_inv = np.linalg.inv(Sigma_ss)

            log_det = np.log(np.linalg.det(Sigma_ss) + 1e-10)
            log_norm = 0.5 * (state_dim * np.log(2 * np.pi) + log_det)
            diff = states - mu_s
            mahalanobis = np.sum(diff @ Sigma_ss_inv * diff, axis=1)

            log_p_sk = -0.5 * mahalanobis - log_norm
            log_component_probs[:, k] = np.log(self.weights_[k] + 1e-10) + log_p_sk

        log_component_probs -= logsumexp(log_component_probs, axis=1, keepdims=True)
        component_probs = np.exp(np.clip(log_component_probs, -700, 700))
        component_probs /= np.sum(component_probs, axis=1, keepdims=True)

        actions = np.zeros((n_samples, action_dim))
        for i in range(n_samples):
            mu_cond_all = np.zeros((self.n_components, action_dim))
            sigma_cond_all = np.zeros((self.n_components, action_dim, action_dim))

            for k in range(self.n_components):
                mu_s = self.means_[k, :state_dim]
                mu_a = self.means_[k, state_dim:]
                Sigma_sa = self.covariances_[k, :state_dim, state_dim:]
                Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6 * np.eye(state_dim)
                Sigma_aa = self.covariances_[k, state_dim:, state_dim:] + 1e-6 * np.eye(action_dim)

                mu_cond = mu_a + (Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ (states[i] - mu_s).T).flatten()
                sigma_cond = Sigma_aa - Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ Sigma_sa
                sigma_cond = (sigma_cond + sigma_cond.T) / 2 + 1e-6 * np.eye(action_dim)

                mu_cond_all[k] = mu_cond
                sigma_cond_all[k] = sigma_cond

            mu_mixed = np.sum(component_probs[i, :, None] * mu_cond_all, axis=0)
            sigma_mixed = np.sum(
                component_probs[i, :, None, None] * (sigma_cond_all + np.einsum('ki,kj->kij', mu_cond_all, mu_cond_all)),
                axis=0
            ) - np.outer(mu_mixed, mu_mixed)

            actions[i] = np.random.multivariate_normal(mu_mixed, sigma_mixed)

        return actions
