from .base_model import BaseModel
import numpy as np

class ScalarOnScalarModel(BaseModel):
    def __init__(self, Kx, order=1):
        self.Kx = Kx
        self.J_cb = 1
        self.order = order

    def calc_model_matrix(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        model_matrix = np.concatenate((ones, X), axis=1)

        return model_matrix
