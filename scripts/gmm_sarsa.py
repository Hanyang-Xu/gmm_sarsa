import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.special import logsumexp

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
        self.covariances_ = np.array([np.cov(X.T) + 1e-6*np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        for _ in range(self.max_iter):
            # E-step
            log_resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_resp[:, k] = np.log(self.weights_[k] + 1e-10) + self._log_gaussian_pdf(X, self.means_[k], self.covariances_[k])
            
            log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp)
            
            # M-step
            for k in range(self.n_components):
                gamma_w = sample_weights * resp[:, k]
                Nk = np.sum(gamma_w)
                
                self.means_[k] = np.sum(gamma_w[:, None] * X, axis=0) / Nk
                diff = X - self.means_[k]
                self.covariances_[k] = (diff.T * gamma_w) @ diff / Nk + 1e-6*np.eye(n_features)
                self.weights_[k] = Nk
        
    def _log_gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
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
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6*np.eye(state_dim)
            Sigma_ss_inv = np.linalg.inv(Sigma_ss)
            
            # 完整的高斯对数概率
            log_det = np.log(np.linalg.det(Sigma_ss))
            log_norm = 0.5 * (state_dim * np.log(2 * np.pi) + log_det)
            diff = states - mu_s
            mahalanobis = np.sum(diff @ Sigma_ss_inv * diff, axis=1)
            
            log_p_sk = -0.5 * mahalanobis - log_norm
            log_component_probs[:, k] = np.log(self.weights_[k] + 1e-10) + log_p_sk
        
        # 归一化
        log_component_probs -= logsumexp(log_component_probs, axis=1, keepdims=True)
        component_probs = np.exp(log_component_probs)
        component_probs /= component_probs.sum(axis=1, keepdims=True)
        
        # 采样
        actions = np.zeros((n_samples, action_dim))
        for i in range(n_samples):
            k = np.random.choice(self.n_components, p=component_probs[i])
            
            # 条件分布参数
            mu_s = self.means_[k, :state_dim]
            mu_a = self.means_[k, state_dim:]
            Sigma_sa = self.covariances_[k, :state_dim, state_dim:]
            Sigma_ss = self.covariances_[k, :state_dim, :state_dim] + 1e-6*np.eye(state_dim)
            Sigma_aa = self.covariances_[k, state_dim:, state_dim:] + 1e-6*np.eye(action_dim)
            
            # 条件均值
            mu_cond = mu_a + (Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ (states[i] - mu_s).T).flatten()
            # 条件协方差
            sigma_cond = Sigma_aa - Sigma_sa.T @ np.linalg.inv(Sigma_ss) @ Sigma_sa
            
            # 确保协方差正定
            sigma_cond = (sigma_cond + sigma_cond.T) / 2 + 1e-6*np.eye(action_dim)
            
            actions[i] = np.random.multivariate_normal(mu_cond, sigma_cond)
        
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
        state = env.reset('init_state')
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
        
