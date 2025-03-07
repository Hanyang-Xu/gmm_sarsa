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

        # 初始化参数
        np.random.seed(42)
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        for _ in range(self.max_iter):
            # E-step
            log_resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_resp[:, k] = np.log(self.weights_[k] + 1e-10) + self._log_gaussian_pdf(X, self.means_[k], self.covariances_[k])

            # 处理溢出问题
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
        
        # 确保协方差矩阵正定
        cov += 1e-6 * np.eye(cov.shape[0])
        
        cov_inv = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov) + 1e-10)
        diff = X - mean
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + np.sum(diff @ cov_inv * diff, axis=1))

    def conditional_sample(self, states):
        state_dim = states.shape[1]
        action_dim = self.means_.shape[1] - state_dim
        n_samples = states.shape[0]

        # 计算每个组件的后验概率
        log_component_probs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            mu_s = self.means_[k, :state_dim]
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6 * np.eye(state_dim)
            Sigma_ss_inv = np.linalg.inv(Sigma_ss)

            # 完整的高斯对数概率
            log_det = np.log(np.linalg.det(Sigma_ss) + 1e-10)
            log_norm = 0.5 * (state_dim * np.log(2 * np.pi) + log_det)
            diff = states - mu_s
            mahalanobis = np.sum(diff @ Sigma_ss_inv * diff, axis=1)

            log_p_sk = -0.5 * mahalanobis - log_norm
            log_component_probs[:, k] = np.log(self.weights_[k] + 1e-10) + log_p_sk

        # 归一化处理
        log_component_probs -= logsumexp(log_component_probs, axis=1, keepdims=True)
        component_probs = np.exp(np.clip(log_component_probs, -700, 700))

        # 处理 NaN
        if np.isnan(component_probs).any():
            print("NaN detected in component_probs:", component_probs)
            component_probs = np.nan_to_num(component_probs, nan=1.0 / self.n_components)

        # 确保概率和为 1
        component_probs /= np.sum(component_probs, axis=1, keepdims=True)
        if not np.allclose(np.sum(component_probs, axis=1), 1.0):
            print("Warning: Probabilities do not sum to 1, fixing...")
            component_probs = np.nan_to_num(component_probs, nan=1.0 / self.n_components)
            component_probs /= np.sum(component_probs, axis=1, keepdims=True)

        # 采样
        actions = np.zeros((n_samples, action_dim))
        for i in range(n_samples):
            k = np.random.choice(self.n_components, p=component_probs[i])

            # 条件分布参数
            mu_s = self.means_[k, :state_dim]
            mu_a = self.means_[k, state_dim:]
            Sigma_sa = self.covariances_[k, :state_dim, state_dim:]
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6 * np.eye(state_dim)
            Sigma_aa = self.covariances_[k, state_dim:, state_dim:] + 1e-6 * np.eye(action_dim)

            # 条件均值和协方差
            mu_cond = mu_a + (Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ (states[i] - mu_s).T).flatten()
            sigma_cond = Sigma_aa - Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ Sigma_sa

            # 确保条件协方差正定
            sigma_cond = (sigma_cond + sigma_cond.T) / 2 + 1e-6 * np.eye(action_dim)

            # 采样动作
            actions[i] = np.random.multivariate_normal(mu_cond, sigma_cond)

        return actions



# # 生成双峰测试数据 (s,a)
# np.random.seed(42)
# true_means = [
#     np.array([1, 0]),    # 组件1: s=1时a~N(0,0.5)
#     np.array([1, 3])     # 组件2: s=1时a~N(3,0.5)
# ]
# true_covs = [
#     np.array([[0.1, 0.05], [0.05, 0.5]]),
#     np.array([[0.1, -0.1], [-0.1, 0.5]])
# ]
# true_weights = np.array([0.5, 0.5])

# # 生成样本数据
# n_samples = 2000
# samples = []
# for _ in range(n_samples):
#     k = np.random.choice(2, p=true_weights)
#     samples.append(np.random.multivariate_normal(true_means[k], true_covs[k]))
# samples = np.array(samples)

# # 训练GMM模型
# gmm = WeightedGMM(n_components=2)
# gmm.fit(samples)

# # 验证条件采样（当s=1时，理论分布应为双峰）
# s_test = 1.0
# states_test = np.array([[s_test]] * 5000)  # 固定s=1，生成5000个样本

# # 生成条件样本
# actions = gmm.conditional_sample(states_test)

# # 理论分布计算
# def theoretical_pdf(a):
#     pdf = 0
#     for k in range(2):
#         # 计算 p(k|s=1)
#         mu_s = true_means[k][0]
#         sigma_ss = true_covs[k][0, 0]
#         p_sk = stats.norm.pdf(s_test, loc=mu_s, scale=np.sqrt(sigma_ss))
#         # 计算 p(a|s=1,k)
#         mu_a_cond = true_means[k][1] + (true_covs[k][0,1]/sigma_ss)*(s_test - mu_s)
#         sigma_a_cond = true_covs[k][1,1] - (true_covs[k][0,1]**2)/sigma_ss
#         pdf += true_weights[k] * p_sk * stats.norm.pdf(a, mu_a_cond, np.sqrt(sigma_a_cond))
#     # 归一化
#     pdf /= stats.norm.pdf(s_test, loc=true_means[0][0], scale=np.sqrt(true_covs[0][0,0]))*true_weights[0] + \
#            stats.norm.pdf(s_test, loc=true_means[1][0], scale=np.sqrt(true_covs[1][0,0]))*true_weights[1]
#     return pdf

# # 可视化对比
# a_grid = np.linspace(-2, 5, 500)
# pdf_theory = theoretical_pdf(a_grid)

# plt.figure(figsize=(10, 6))
# plt.hist(actions, bins=50, density=True, alpha=0.6, label='实际采样')
# plt.plot(a_grid, pdf_theory, 'r-', lw=2, label='理论分布')
# plt.title(f"条件分布 p(a|s={s_test})（双峰验证）")
# plt.xlabel("Action")
# plt.ylabel("概率密度")
# plt.legend()
# plt.show()

# # 打印组件后验概率
# print("组件1的后验概率:", np.mean(actions < 1.5))  # 第一个峰在a=0附近
# print("组件2的后验概率:", np.mean(actions >= 1.5)) # 第二个峰在a=3附近