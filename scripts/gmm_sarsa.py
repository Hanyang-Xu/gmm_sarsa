import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class WeightedGMM():
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, sample_weights=None):
        """训练 GMM"""
        n_samples, n_features = X.shape
        sample_weights = np.ones(n_samples) if sample_weights is None else sample_weights
        sample_weights /= np.sum(sample_weights)

        # 初始化 GMM 参数
        np.random.seed(42)
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx]
        self.covariances_ = np.array([np.cov(X.T) + np.eye(n_features) * 1e-2 for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        for _ in range(self.max_iter):
            # E 步：计算责任度
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
            
            # 对责任度进行归一化
            responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True)
            responsibilities_sum = np.maximum(responsibilities_sum, 1e-6)  # 防止除零
            responsibilities /= responsibilities_sum

            # 修复 NaN 和 Inf
            responsibilities = np.nan_to_num(responsibilities, nan=0.0, posinf=1.0, neginf=0.0)

            # M 步：更新参数
            for k in range(self.n_components):
                gamma_w = sample_weights * responsibilities[:, k]  # 加权责任度
                Nk = np.sum(gamma_w)

                self.means_[k] = np.sum(gamma_w[:, np.newaxis] * X, axis=0) / Nk
                diff = X - self.means_[k]
                self.covariances_[k] = (diff.T @ (gamma_w[:, np.newaxis] * diff)) / Nk + np.eye(n_features) * 1e-6
                self.weights_[k] = Nk / np.sum(sample_weights)

    def _gaussian_pdf(self, X, mean, cov):
        """计算高斯分布的概率密度"""
        d = X.shape[1]
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(d))  # 避免奇异矩阵
        det_cov = np.linalg.det(cov + 1e-6 * np.eye(d))
        norm_factor = 1 / (np.sqrt((2 * np.pi) ** d * det_cov))
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm_factor * exp_term

    def conditional_sample(self, states):
        """给定多个状态，从条件分布 P(a | s) 采样"""
        state_dim = states.shape[1] 
        num_samples = states.shape[0]
        action_dim = self.means_.shape[1] - state_dim

        conditional_means = np.zeros((num_samples, action_dim))
        conditional_covs = np.zeros((num_samples, action_dim, action_dim))
        component_probs = np.zeros((num_samples, self.n_components))

        for k in range(self.n_components):
            mu_s, mu_a = self.means_[k, :state_dim], self.means_[k, state_dim:]
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim]
            Sigma_sa = self.covariances_[k, :state_dim, state_dim:]
            Sigma_as = self.covariances_[k, state_dim:, :state_dim]
            Sigma_aa = self.covariances_[k, state_dim:, state_dim:]
            
            # 避免奇异矩阵
            Sigma_ss_inv = np.linalg.inv(Sigma_ss + 1e-6 * np.eye(state_dim))
            mu_cond = mu_a.reshape(-1,1) + Sigma_as @ Sigma_ss_inv @ (states.T - mu_s.reshape(-1, 1))
            mu_cond = mu_cond.T  # 变回 (num_samples, action_dim)

            Sigma_cond = Sigma_aa - Sigma_as @ Sigma_ss_inv @ Sigma_sa
            conditional_means += self.weights_[k] * mu_cond
            conditional_covs += self.weights_[k] * Sigma_cond

            # 使用对数变换计算概率（避免数值溢出）
            log_component_probs = np.log(self.weights_[k] * np.exp(-0.5 * np.sum((states - mu_s) @ Sigma_ss_inv * (states - mu_s), axis=1)) + 1e-10)

            component_probs[:, k] = log_component_probs

        # 使用对数概率进行归一化
        log_component_probs_sum = np.sum(component_probs, axis=1, keepdims=True)
        log_component_probs_sum = np.maximum(log_component_probs_sum, 1e-6)  # 防止除零
        component_probs -= log_component_probs_sum  # 对数空间中减去总和
        component_probs = np.exp(component_probs)  # 转回原始空间

        # 修复 NaN 和 Inf
        component_probs = np.nan_to_num(component_probs, nan=0.0, posinf=1.0, neginf=0.0)

        # 确保概率和为 1
        component_probs /= np.sum(component_probs, axis=1, keepdims=True)
        # 采样
        chosen_components = [np.random.choice(self.n_components, p=component_probs[i]) for i in range(num_samples)]
        actions = np.array([np.random.multivariate_normal(conditional_means[i], conditional_covs[i]) for i in range(num_samples)])

        return actions




class gmmsarsa():
    def __init__(self,n_components, state_dim, action_dim, poly_degree=3, alpha=0.1,lambda_reg=0.1, gamma=0.99, init_data=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.error_list = []

        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.feature_dim = self.poly.fit_transform(np.zeros((1, self.state_dim + self.action_dim))).shape[1]
        self.theta = np.zeros(self.feature_dim)

        self.gmm = WeightedGMM(n_components)
        if init_data is None:
            dummy_data = np.random.randn(10, self.state_dim + self.action_dim)
        else: dummy_data = init_data
        print(f"init_data:{dummy_data.shape}")
        self.gmm.fit(dummy_data)


    def _get_phi(self, state, action):
        sa = np.concatenate((state, action)).reshape(1, -1)
        phi = self.poly.transform(sa)[0]
        return phi
    
    def _Q_func(self, state, action):
        phi = self._get_phi(state, action)
        return np.dot(self.theta, phi)

    def _update_Q_func(self, state, action, reward, next_state, next_action):
        Q_old = self._Q_func(state, action)
        Q_new = self._Q_func(next_state, next_action)
        phi = self._get_phi(state, action)

        td_error = reward + self.gamma * Q_new - Q_old
        self.theta += self.alpha * td_error * phi
        self.error_list.append(td_error)
        print(f"new theta: {self.theta}")
    
    def plot_Q_eval_error(self):
        plt.figure()
        plt.plot(self.error_list)
        plt.title('td_error')
        plt.show


    def _update_policy(self, state, num_samples, tau, epsilon):
        
        perturbation = np.random.uniform(-epsilon, epsilon, (num_samples, self.state_dim))
        state_samples = np.repeat(state[None, :], num_samples, axis=0) + perturbation
        actions = np.atleast_2d(self.gmm.conditional_sample(state_samples))

        Q_values = np.array([self._Q_func(state_samples[i], actions[i]) for i in range(num_samples)])
        Q_values -= np.max(Q_values)  # 防止溢出
        weights = np.exp(Q_values / tau)
        weights /= np.sum(weights)

        state_action_data = np.hstack([state_samples, actions])
        self.gmm.fit(state_action_data, sample_weights=weights)

    def train(self, env, num_samples=100, max_iter=100 ,tau=0.1, epsilon=2, tol=0.00001, axs=None):
        state = env.reset()
        action = np.atleast_2d(self.gmm.conditional_sample(state.reshape(1, -1))).flatten()

        iter = 0
        reward_list = []
        total_reward = 0
        total_reward_list=[]
        last_reward = 0
        while (iter < max_iter):
            next_state, reward = env.step(action, axs, iter)
            next_action = np.atleast_2d(self.gmm.conditional_sample(next_state.reshape(1, -1))).flatten()

            self._update_Q_func(state, action, reward, next_state, next_action)
            self._update_policy(state, num_samples, tau, epsilon)

            state, action = next_state, next_action
            
            iter += 1
            print(f"------------{iter}-------------")
            print(f"reward:{reward}")
            reward_diff = np.abs(reward - last_reward)
            reward_list.append(reward)
            total_reward+=reward
            total_reward_list.append(total_reward)
            last_reward = reward


            if reward_diff < tol:
                break
        self.plot_Q_eval_error()
        return reward_list, total_reward_list
        
