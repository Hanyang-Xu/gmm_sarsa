import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from promps import ProMPs
from WeightedGMM import WeightedGMM
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
    init_data = np.hstack((g_ws, t_ws))

    print(f"g_ws:{g_ws.shape}")
    print(f"t_ws:{t_ws.shape}")

    init_data = np.hstack((g_ws, t_ws))
    print(f"init_data:{init_data.shape}")

    gmm = WeightedGMM(n_components=2)
    gmm.fit(init_data)

    test_idx = 8
    target_angle = angle_data[test_idx]
    # init the env
    env = Prosthesis(promps=promps, target_angle=target_angle, s_reducer=s_reducer, a_reducer=a_reducer)

    # test conditional_sample

    test_state = g_ws[test_idx].reshape(1, -1)
    test_action = t_ws[test_idx].reshape(1, -1)

    sampled_action = gmm.conditional_sample(test_state)
    next_state,_,_ = env.step(test_action)
    next_state = next_state.reshape(1,-1)
    np.save('init_state.npy', next_state)
    print(f"next_state:{next_state.shape}")
    next_action = gmm.conditional_sample(next_state)

    print(f"sampled_action:{sampled_action.shape}")
    print(f"test_action:{test_action.shape}")
    sampled_action = a_reducer.inverse_trans(sampled_action)
    test_action = a_reducer.inverse_trans(test_action)
    next_state = s_reducer.inverse_trans(next_state)
    next_action = a_reducer.inverse_trans(next_action)

    sampled_action = promps.sample(sampled_action)
    test_action = promps.sample(test_action)
    next_state = promps.sample(next_state)
    next_action = promps.sample(next_action)

    plt.figure()
    plt.plot(test_action, label='Reconstracted from PCA', color='r')
    plt.plot(sampled_action, label='Sampled from GMM & Reconstructed', color='b')
    plt.plot(next_action, label='Next action', color='g')
    plt.plot(moment_data[test_idx], label='Original', linestyle='-.')
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(next_state, label='Sampled from GMM & Reconstructed', color='b')
    # plt.plot(angle_data[test_idx], label='Original', linestyle='-.')
    # plt.legend()
    # plt.show()