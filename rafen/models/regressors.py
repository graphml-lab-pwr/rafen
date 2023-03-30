import numpy as np
from scipy.linalg import orthogonal_procrustes


class OrthogonalProcrustesRegressor:
    def __init__(self):
        self.t_m = None

    def predict(self, v):
        return np.matmul(v, self.t_m)

    def fit(self, X, Y):
        self.t_m = orthogonal_procrustes(X, Y)[0]
