import numpy as np
import torch
from env_model import TorqueToAngleLSTM  # 导入模型类
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

class Prosthesis:
    def __init__(self, promps, target_angle, s_reducer, a_reducer):
        self.promps = promps
        self.target_angle = target_angle
        self.s_reducer = s_reducer
        self.a_reducer = a_reducer
        self.current_state = None
        # 只加载一次模型
        self.model = TorqueToAngleLSTM()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()  # 设置模型为评估模式

    def reset(self, mode='zero'):
        # 初始化状态为零或某个初始状态
        if mode == 'zero':
            init_state = np.zeros(self.promps.num_basis)
        if mode == 'init_state':
            init_state = np.load('trajectory_reps/init_state.npy')
        self.current_state = init_state
        init_state = self.s_reducer.transform(init_state.reshape(1,-1)).reshape(-1)
        return init_state

    def reward(self, current_angle):
        # 计算当前角度和目标角度的差异并返回奖励
        # all + end point
        # k = 0.9
        # max_idx = np.argmax(self.target_angle)
        # max_target = self.target_angle[max_idx]
        # max_angle = current_angle[max_idx]
        # diff = (1-k)*np.sum(np.abs(self.target_angle - current_angle)) + k*np.abs(self.target_angle[-1]-current_angle[-1])+ k*np.abs(max_angle-max_target)
        # diff = (1-k)*np.sum(np.abs(self.target_angle - current_angle)) + k*np.abs(max_angle-max_target)
        # diff = (1-k)*np.sum(np.abs(self.target_angle - current_angle))
        # print(f"diff:{diff}")

        # four points
        # k = 0.5
        max_idxs = argrelextrema(self.target_angle, np.greater)[0]
        min_idxs = argrelextrema(self.target_angle, np.less)[0]
        diff1 = np.abs(self.target_angle[max_idxs[0]]-current_angle[max_idxs[0]])
        diff2 = np.abs(self.target_angle[min_idxs[0]]-current_angle[min_idxs[0]])
        diff3 = np.abs(self.target_angle[max_idxs[1]]-current_angle[max_idxs[1]])
        diff4 = np.abs(self.target_angle[-1]-current_angle[-1])
        weight = [1, 1, 1, 1]
        diff = diff1*weight[0]+diff2*weight[1]+diff3*weight[2]+diff4*weight[3]

        # diff based on time
        # tar_max_idx = argrelextrema(self.target_angle, np.greater)[0][1]
        # max_idx = argrelextrema(current_angle[40:90], np.greater)[0][0]+40
        # diff = (tar_max_idx - max_idx)*10

        print(f"diff:{diff}") 

        # 使用负二次函数计算奖励
        sigma = 50
        reward = np.exp(- (diff ** 2) / (2 * sigma ** 2))
        return reward

    def step(self, action_w, axs, iter):
        action_w = self.a_reducer.inverse_trans(action_w.reshape(1,-1))
        action = self.promps.sample(action_w)
        # action here means the moment curve
        action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1, 1)  # 调整为 (batch_size, sequence_length, 1)

        # 使用模型预测角度或力矩
        with torch.no_grad():  # 不需要计算梯度
            predicted_angle = self.model(action_tensor).numpy().flatten()  # 假设预测是角度

        # 更新当前状态
        self.current_state = predicted_angle# 根据预测结果更新状态
        axs[0].plot(predicted_angle)  # 更新曲线的y数据
        
        # 计算奖励
        reward = self.reward(predicted_angle)
        state_w, _= self.promps.traj2w(predicted_angle)
        axs[1].plot(iter, reward, 'or')
        plt.draw()  # 重新绘制图像
        plt.pause(0.1)

        state_w = self.s_reducer.transform(state_w.reshape(1,-1)).reshape(-1)
        return state_w, reward