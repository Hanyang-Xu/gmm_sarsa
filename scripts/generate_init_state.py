import numpy as np
import pandas as pd

from promps import ProMPs

def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ == "__main__":
    file_path = 'Datasets/AB01/Left.xlsx' 
    angle_data, moment_data = load_data(file_path)
    promps = ProMPs(data = moment_data, basis_fun='gaussian', num_basis=10, sigma=0.1)
    promps.train(max_iter=20, threshold=1e-4)

    # choose target angle curve
    target_traj = angle_data[5]
    w, sampled_w = promps.traj2w(target_traj)
    np.save('init_state.npy', w)
