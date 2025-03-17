import numpy as np
import torch
from env_model import TorqueToAngleLSTM 
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

class Prosthesis:
    def __init__(self, promps, target_angle, s_reducer, a_reducer):
        self.promps = promps
        self.target_angle = target_angle
        self.s_reducer = s_reducer
        self.a_reducer = a_reducer
        self.current_state = None

        self.model = TorqueToAngleLSTM()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()  

    def reset(self):
        init_state = np.load('init_state.npy').reshape(-1)
        self.current_state = init_state
        return init_state

    def reward(self, current_angle):
        # all + end point
        # k = 0
        # max_idx = np.argmax(self.target_angle)
        # max_target = self.target_angle[max_idx]
        # max_angle = current_angle[max_idx]
        # diff = (1-k)*np.sum(np.abs(self.target_angle - current_angle)) + k*np.abs(self.target_angle[-1]-current_angle[-1])+ k*np.abs(max_angle-max_target)
        # diff = (1-k)*np.sum(np.abs(self.target_angle - current_angle)) + k*np.abs(max_angle-max_target)
        # diff = 0.1*np.sum(np.abs(self.target_angle - current_angle))
        # print(f"diff:{diff}")

        # four points
        # k = 0.5
        max_idxs = argrelextrema(self.target_angle, np.greater)[0]
        min_idxs = argrelextrema(self.target_angle, np.less)[0]
        diff1 = np.abs(self.target_angle[max_idxs[0]]-current_angle[max_idxs[0]])
        diff2 = np.abs(self.target_angle[min_idxs[0]]-current_angle[min_idxs[0]])
        diff3 = np.abs(self.target_angle[max_idxs[1]]-current_angle[max_idxs[1]])
        diff4 = np.abs(self.target_angle[-1]-current_angle[-1])
        weight = [0, 1.5, 2.5, 0]
        diff = diff1*weight[0]+diff2*weight[1]+diff3*weight[2]+diff4*weight[3]

        # diff based on time
        # tar_max_idx = argrelextrema(self.target_angle, np.greater)[0][1]
        # max_idx = argrelextrema(current_angle[40:90], np.greater)[0][0]+40
        # diff = (tar_max_idx - max_idx)*10

        print(f"diff:{diff}") 
        sigma = 30
        reward = np.exp(- (diff ** 2) / (2 * sigma ** 2))
        return reward

    def step(self, action_w):
        print(f"action_w:{action_w.shape}")
        action_w = self.a_reducer.inverse_trans(action_w.reshape(1,-1))
        action = self.promps.sample(action_w)
        action_tensor = torch.tensor(action, dtype=torch.float32).view(1, -1, 1)  

        with torch.no_grad(): 
            predicted_angle = self.model(action_tensor).numpy().flatten() 

        self.current_state = predicted_angle
        reward = self.reward(predicted_angle)
        state_w, _= self.promps.traj2w(predicted_angle)

        state_w = self.s_reducer.transform(state_w.reshape(1,-1)).reshape(-1)
        return state_w, reward, predicted_angle