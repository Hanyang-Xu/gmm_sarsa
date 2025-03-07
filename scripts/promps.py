import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt

class ProMPs:
    def __init__(self, data, basis_fun='gaussian', num_basis=10, sigma=0.1):
        self.basis_fun = basis_fun
        self.data = data
        self.num_basis = num_basis
        self.sigma = sigma
        self.T = data.shape[1]
        self.N = data.shape[0]

    def _basis_fun(self):
        if self.basis_fun == 'gaussian':
            return self.gaussian_rbf()

    def _gaussian_rbf(self, t, centers, sigma):
        centers = np.asarray(centers)
        dist_squared = (t - centers)**2
        result = np.exp(-dist_squared / (2 * sigma**2))
        return result
    
    def _get_phi(self, t):
        if self.basis_fun == 'gaussian':
            centers = np.linspace(0, 1, self.num_basis)
            return self._gaussian_rbf(t, centers=centers, sigma=self.sigma)
    
    def _get_Phi(self, traj=None):
        if traj is not None:
            times = np.linspace(0,1,len(traj))
            Phi = []
            for t in times:
                phi_t = self._get_phi(t)
                Phi.append(phi_t)
            return Phi
        else:
            times = np.linspace(0, 1, self.T)
            Phi = []
            for _ in range(self.N):
                phi_n = []
                for t in times:
                    phi_nt = self._get_phi(t)
                    phi_n.append(phi_nt)
                # print(np.array(phi_n).shape)
                Phi.append(phi_n)
            self.Phi = Phi
            return Phi

    def _E_step(self):
        #2) Load some variables
        inv_sig_w = np.linalg.inv(self.sig_w)
        # inv_sig_y = np.linalg.inv(self.sig_y)
        inv_sig_y = 1/self.sig_y
        Phi = self._get_Phi()
        #3) Compute expectations
        w_means = []
        w_covs = []
        for n in range(self.data.shape[0]):
            sum_cov = inv_sig_w
            sum_mean = np.dot(inv_sig_w, self.mu_w)
            for t in range(self.data.shape[1]):
                phi_nt = Phi[n][t].reshape(1,-1)
                # print(f"phi_nt:{phi_nt.shape}")
                tmp1 = np.dot(np.transpose(phi_nt),inv_sig_y)
                sum_cov = sum_cov + np.dot(tmp1, phi_nt)
                sum_mean = sum_mean + np.dot(tmp1, self.data[n][t])
            Swn = utils.force_sym(np.linalg.inv(sum_cov))
            wn = np.dot(Swn, sum_mean)
            w_means.append(wn)
            w_covs.append(Swn)
        return {'w_means': w_means, 'w_covs': w_covs}

    def _M_step(self, expectations):
        Phi = self._get_Phi()
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        K = self.num_basis

        #1) Optimize mu_w
        mu_w = sum(w_means)/self.N
        #2) Optimize Sigma_w
        sig_w = np.zeros((K,K))
        for n in range(self.N):
            tmp = w_covs[n] + np.dot((w_means[n]-mu_w), (w_means[n]-mu_w).T)
            sig_w = sig_w + tmp
        #3) Optimize Sigma_y
        diff_y = 0
        uncert_w_y = 0
        for n in range(self.N):
            for t in range(self.T):
                diff_y += (self.data[n][t] - np.dot(Phi[n][t], w_means[n]))**2
                uncert_w_y += np.dot(np.dot(Phi[n][t],w_covs[n]),Phi[n][t].T)
        sig_y = np.squeeze((diff_y + uncert_w_y) / (self.N * self.T))
        # print(f"sig_y:{sig_y}")
        #4) Update
        self.mu_w = mu_w
        self.sig_w = sig_w
        y_diff = self.sig_y - sig_y
        self.sig_y = sig_y
        if y_diff < self.threshold:
            print("Promps Converged")
            return True

    def _EM_training(self, max_iter):
        #1) Initialize
        self.mu_w = np.zeros((self.num_basis,1))
        self.sig_w = self.sig_w = np.eye(self.num_basis) * 1e-6
        self.sig_y = 1
        #2) Train
        for it in range(max_iter):
            print(f"================= Iteration {it} =================")
            # print(f"mu_w:{self.mu_w.shape}, sig_w:{self.sig_w.shape}, sig_y:{self.sig_y}")
            expectations = self._E_step()
            # self._M_step(expectations)
            if self._M_step(expectations):
                break


    def train(self, max_iter=50, threshold=1e-4):
        self.threshold = threshold
        return self._EM_training(max_iter)
    
    def _y2w_dist(self, traj):
        Phi = self._get_Phi(traj)
        traj = np.reshape(traj, (-1,1))
        w_means = self.mu_w
        w_covs = self.sig_w
        y_covs = self.sig_y
        sig_Y = np.eye(self.T) * y_covs
        inv_sig_w = np.linalg.inv(w_covs)
        inv_sig_Y = np.linalg.inv(sig_Y)

        tmp1 = inv_sig_w + np.dot(np.transpose(Phi), np.dot(inv_sig_Y, Phi))
        S = np.linalg.inv(tmp1)
        tmp2 = np.dot(inv_sig_w, w_means)
        tmp3 = np.dot(np.transpose(Phi), np.dot(inv_sig_Y, traj))
        m = np.dot(S, tmp2+tmp3)
        return m, S

    def traj2w(self, traj):
        m, S = self._y2w_dist(traj)
        m = m.flatten()
        sampled_w = np.random.multivariate_normal(m, S)
        return m, sampled_w
    
    def sample(self, w):
        Phi = np.copy(self.Phi)
        sampled_y = []
        for t in range(self.T):
            phi_t = np.array(Phi[0][t]).reshape(1, -1)
            w = w.reshape(-1, 1)
            y_t = np.dot(phi_t, w)
            sampled_y.append(y_t)
        sampled_y = np.array(sampled_y).flatten()
        return sampled_y

        
def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ == '__main__':
    # file_path = 'Datasets/AB01/Left.xlsx' 
    file_path = 'Datasets/merged_data.xlsx'
    angle_data, moment_data = load_data(file_path)
    promps = ProMPs(data = moment_data, basis_fun='gaussian', 
                    num_basis=8, sigma=0.1)
    Phi = promps._get_Phi()
    promps.train(max_iter=20, threshold=1e-4)
    traj = moment_data[8]
    w, sampled_w = promps.traj2w(traj)
    print(f"w:{w}")
    sampled_y = promps.sample(w)

    plt.figure()
    plt.plot(traj, label='Original Trajectory', color='blue')
    plt.plot(sampled_y, label='Sampled Trajectory', color='red')
    plt.legend()
    plt.show()

