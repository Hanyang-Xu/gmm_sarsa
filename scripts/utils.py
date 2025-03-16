import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def force_sym(A):
    """ Returns a symmetric matrix from the received matrix
    """
    return (A + A.T) / 2.0

def interpolate(start_state, end_state, num_samples):
    """5维状态空间的线性插值"""
    return np.linspace(
        start_state, 
        end_state,    
        num_samples,  
        axis=0        
    )

class dim_reducer():
    def __init__(self, init_data, latent_space):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(init_data)
        pca = PCA(n_components=latent_space)
        pca.fit(scaled_data)
        
        self.scaler = scaler
        self.pca = pca

    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        X_pca = self.pca.transform(scaled_X)
        return X_pca
    
    def inverse_trans(self, X_pca):
        X_scaled = self.pca.inverse_transform(X_pca)
        X = self.scaler.inverse_transform(X_scaled)
        return X