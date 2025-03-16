import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from collections import deque

from WeightedGMM import WeightedGMM
from utils import interpolate

class SarsaGMM:
    def __init__(self, 
                 gmm_components=2, 
                 poly_degree=2, 
                 state_space=None,
                 action_space=None,
                 gamma=0.99, 
                 alpha=0.1, 
                 init_data=None):
        
        self.gamma = gamma
        self.alpha = alpha
        self.state_dim = state_space.shape[1]
        self.action_dim = action_space.shape[1]
        self.state_space = state_space
        self.action_space = action_space
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
        if np.abs(td_error) < 100:
            self.theta += self.alpha * td_error * phi
            self.td_error = td_error

    def _update_policy(self, num_samples):
        interpolated_states = interpolate(self.state_space[0], self.state_space[1], num_samples)
        actions = self.gmm.conditional_sample(interpolated_states)
        Q_values = [self._Q_func(s, a) for s, a in zip(interpolated_states, actions)]
        
        Q_shifted = Q_values - np.max(Q_values)
        exp_values = np.exp(Q_shifted)         
        
        sum_exp = np.sum(exp_values) + 1e-12  
        weights = exp_values / sum_exp
        
        state_action_data = np.hstack([
            np.array(interpolated_states).reshape(num_samples, -1),  
            np.array(actions).reshape(num_samples, -1)          
        ])
        
        self.gmm.fit(state_action_data, sample_weights=weights)

    def train(self, 
            env, 
            num_samples=100, 
            max_iter=100,
            tol=0.00001,
            target_angle=None):
        
        state = env.reset()
        action = np.atleast_2d(self.gmm.conditional_sample(state.reshape(1, -1))).flatten()
        iter = 0
        last_reward = 0

        # 创建用于早停判断的滑动窗口，存储最近 10 次的 reward
        reward_window = deque(maxlen=10)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.ion() 
        ax1.set_title('Angle curve')
        # ax1.plot(target_angle, label='Target', linestyle='-.')
        ax2.set_title('Reward')
        ax3.set_title('TD Error')
        
        while (iter < max_iter):
            next_state, reward, actual_angle = env.step(action)
            next_action = np.atleast_2d(self.gmm.conditional_sample(next_state.reshape(1, -1))).flatten()
            self._update_Q_func(state, action, reward, next_state, next_action)
            self._update_policy(num_samples)
            state, action = next_state, next_action
            
            iter += 1
            print(f"------------{iter}-------------")
            print(f"reward: {reward}")

            # 更新滑动窗口
            reward_window.append(reward)

            # 计算最近 10 次的标准差
            if len(reward_window) == 10:
                reward_std = np.std(reward_window)
                print(f"Reward Std (last 10 steps): {reward_std}")
                # 判断早停条件
                if reward_std < tol:
                    print("Early stopping: Reward standard deviation within tolerance.")
                    break

            reward_diff = np.abs(reward - last_reward)
            last_reward = reward

            if iter == 1:
                init_angle = actual_angle
            # 实时更新绘图
            ax1.clear()
            ax1.plot(target_angle, label='Target', linestyle='-.', color='r')
            ax1.plot(init_angle, label='Initial', linestyle='-.', color='gray')
            ax1.plot(actual_angle, label='Actual', color='b')
            ax1.legend()
            ax2.plot(iter, reward, 'or')
            ax3.plot(iter, self.td_error, 'ob')
            plt.pause(0.001)  
            fig.canvas.flush_events() 

        plt.ioff() 
        plt.show()  
