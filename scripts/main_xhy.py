import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from promps import ProMPs
from SarsaGMM import SarsaGMM
from env import Prosthesis
from utils import dim_reducer

def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ == '__main__':
    # parameters
    state_dim = 5
    action_dim = 5
    gmm_components = 2

    # train promps model
    file_path = 'Datasets/AB01/Left.xlsx' 
    angle_data, moment_data = load_data(file_path)
    promps = ProMPs(data = moment_data, basis_fun='gaussian', num_basis=10, sigma=0.1)
    promps.train(max_iter=20, threshold=1e-4)

    # choose target angle curve
    target_traj = angle_data[1]
    w, sampled_w = promps.traj2w(target_traj)
    target_angle = promps.sample(w)

    g_ws = []
    for angle in angle_data:
        g_w, _ = promps.traj2w(angle)
        g_ws.append(g_w)

    t_ws = []
    for torque in moment_data:
        t_w, _ = promps.traj2w(torque)
        t_ws.append(t_w)

    s_reducer = dim_reducer(g_ws, state_dim)
    a_reducer = dim_reducer(t_ws, action_dim)
    g_ws = s_reducer.transform(g_ws)
    t_ws = a_reducer.transform(t_ws)
    init_data = np.hstack((g_ws, t_ws))

    # set up environment
    env = Prosthesis(promps=promps, 
                     target_angle=target_angle,
                     s_reducer=s_reducer,
                     a_reducer=a_reducer)

    # set up the agent
    agent = SarsaGMM(gmm_components=gmm_components, 
                     state_dim=state_dim, 
                     action_dim=action_dim, 
                     poly_degree=2, 
                     gamma=0.99, 
                     alpha=0.0001,
                     init_data=init_data)
    
    agent.train(env, 
                num_samples=100, 
                max_iter=100, 
                tau=1, 
                epsilon=1, 
                tol=0,
                target_angle=target_angle)
