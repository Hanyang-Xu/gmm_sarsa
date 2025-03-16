import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from promps import ProMPs
from utils import dim_reducer
from env import Prosthesis

def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ == '__main__':
    angle_data, moment_data = load_data('Datasets/AB01/Left.xlsx')

    # train promps model
    promps = ProMPs(data=moment_data, basis_fun='gaussian', num_basis=10, sigma=0.1)
    promps.train(max_iter=20, threshold=1e-4)

    g_ws = []
    for angle in angle_data:
        g_w, _ = promps.traj2w(angle)
        g_ws.append(g_w)

    t_ws = []
    for torque in moment_data:
        t_w, _ = promps.traj2w(torque)
        t_ws.append(t_w)

    s_reducer = dim_reducer(g_ws, 5)
    a_reducer = dim_reducer(t_ws, 5)
    g_ws = s_reducer.transform(g_ws)
    t_ws = a_reducer.transform(t_ws)

    # 使用zip转置矩阵并遍历每一列
    min_values = [min(col) for col in zip(*t_ws)]
    max_values = [max(col) for col in zip(*t_ws)]

    print("最小值:", min_values) 
    print("最大值:", max_values) 
    