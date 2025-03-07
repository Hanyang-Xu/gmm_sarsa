import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from WeightedGMM import WeightedGMM

class SarsaGMM:
    def __init__(self, 
                 gmm_components=2, 
                 poly_degree=2, 
                 state_dim=5,
                 action_dim=5,
                 gamma=0.99, 
                 alpha=0.1, 
                 init_data=None):
        
        self.gamma = gamma
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._init_gmm(init_data, gmm_components)
        self._init_poly(poly_degree)
    
    def _init_gmm(self, init_data, gmm_components):
        gmm = WeightedGMM(gmm_components)
        gmm.fit(init_data)
        self.gmm = gmm
    def _init_poly(self, poly_degree):
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.feature_dim = self.poly.fit_transform(np.zeros((1, self.state_dim + self.action_dim))).shape[1]
        self.theta = np.zeros(self.feature_dim)
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
        # 梯度裁剪防止数值溢出
        update = self.alpha * td_error * phi
        max_grad_norm = 1.0  # 可调整的阈值
        update_norm = np.linalg.norm(update)
        if update_norm > max_grad_norm:
            update = update * (max_grad_norm / update_norm)
        self.theta += update
        print(f"theta:{self.theta}")
        self.td_error = td_error
    # def _update_policy(self, state, num_samples, tau, epsilon):
        
    #     perturbation = np.random.uniform(-epsilon, epsilon, (num_samples, self.state_dim))
    #     state_samples = np.repeat(state[None, :], num_samples, axis=0) + perturbation
    #     actions = np.atleast_2d(self.gmm.conditional_sample(state_samples))
    #     Q_values = np.array([self._Q_func(state_samples[i], actions[i]) for i in range(num_samples)])
    #     Q_values -= np.max(Q_values)  # 防止溢出
    #     weights = np.exp(Q_values / tau)
    #     weights /= np.sum(weights)
    #     print(f"weights:{weights}")
    #     state_action_data = np.hstack([state_samples, actions])
    #     self.gmm.fit(state_action_data, sample_weights=weights)

    def _update_policy(self, state, num_samples, tau, epsilon):
        perturbation = np.random.uniform(-epsilon, epsilon, (num_samples, self.state_dim))
        state_samples = np.repeat(state[None, :], num_samples, axis=0) + perturbation
        actions = np.atleast_2d(self.gmm.conditional_sample(state_samples))
        
        # 计算Q值时增加数值稳定性
        Q_values = np.array([self._Q_func(state_samples[i], actions[i]) for i in range(num_samples)])
        
        # 双重保护：减最大值 + 安全除法
        max_q = np.max(Q_values)
        Q_shifted = Q_values - max_q
        exp_values = np.exp(Q_shifted / np.clip(tau, 1e-8, None))  # 防止tau为0
        
        # 增加溢出保护逻辑
        sum_exp = np.sum(exp_values)
        if sum_exp == 0:  # 所有exp值都下溢为0时（理论上不会出现）
            exp_values = np.ones_like(exp_values)
            sum_exp = num_samples
        
        weights = exp_values / sum_exp
        print(f"weights:{weights}")
        print(f"weights sum: {np.sum(weights):.4f}")  # 验证权重和
        
        state_action_data = np.hstack([state_samples, actions])
        self.gmm.fit(state_action_data, sample_weights=weights)


    def train(self, 
          env, 
          num_samples=100, 
          max_iter=100,
          tau=0.1, 
          epsilon=2, 
          tol=0.00001,
          target_angle=None):
    
        state = env.reset()
        action = np.atleast_2d(self.gmm.conditional_sample(state.reshape(1, -1))).flatten()
        iter = 0
        last_reward = 0
        # 创建绘图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.ion()  # 打开交互模式 (关键点)
        ax1.set_title('Angle curve')
        # ax1.plot(target_angle, label='Target', linestyle='-.')
        ax1.legend()
        ax2.set_title('Reward')
        ax3.set_title('TD Error')
        
        while (iter < max_iter):
            next_state, reward, actual_angle = env.step(action)
            next_action = np.atleast_2d(self.gmm.conditional_sample(next_state.reshape(1, -1))).flatten()
            self._update_Q_func(state, action, reward, next_state, next_action)
            self._update_policy(state, num_samples, tau, epsilon)
            state, action = next_state, next_action
            
            iter += 1
            print(f"------------{iter}-------------")
            print(f"reward:{reward}")
            reward_diff = np.abs(reward - last_reward)
            last_reward = reward
            # 实时更新绘图
            ax1.clear()
            ax1.plot(target_angle, label='Target', linestyle='-.')
            ax1.plot(actual_angle)
            ax2.plot(iter, reward, 'or')
            ax3.plot(iter, self.td_error, 'ob')
            plt.pause(0.001)  # 暂停一会儿，以便更新图像 (关键点)
            fig.canvas.flush_events()  # 刷新绘图事件 (关键点)
            if reward_diff < tol:
                print("policy has been trained")
                break
        plt.ioff()  # 关闭交互模式
        plt.show()  # 最终显示绘图