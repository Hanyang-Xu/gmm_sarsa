import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from promps import ProMPs


def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ == '__main__':
    # parameters
    state_dim = 10
    action_dim = 10
    gmm_components = 2

    # train promps model
    file_path = 'Datasets/AB01/Left.xlsx' 
    # file_path = 'Datasets/merged_data.xlsx'
    angle_data, moment_data = load_data(file_path)
    promps = ProMPs(data = moment_data, basis_fun='gaussian', num_basis=state_dim, sigma=0.1)
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

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(t_ws)
    print(X_scaled.shape)

    # 进行PCA
    pca = PCA()
    pca.fit(X_scaled)

    # 获取每个主成分的重要性（解释的方差比例）
    explained_variance = pca.explained_variance_ratio_

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='skyblue')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.show()

    pca2 = PCA(n_components=5)

    reduced_t_ws = pca2.fit_transform(X_scaled)
    print(reduced_t_ws.shape)
    X_reconstructed_scaled = pca2.inverse_transform(reduced_t_ws)
    X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
    pca_sampled_y = promps.sample(X_reconstructed[8])
    sampled_y = promps.sample(t_ws[8])
    origin_y = moment_data[8]

    plt.figure()
    plt.plot(origin_y, label='origin', color='red')
    plt.plot(sampled_y, label='before dimension reduce', color='b')
    plt.plot(pca_sampled_y, label='after dimension reduce', color='g')
    plt.legend()
    plt.show()


