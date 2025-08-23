from .base_model import BaseModel
import numpy as np
import tensorflow as tf

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
    
    def calc_covar_matrix(self, Model_mat):
        model_mat = self.calc_model_matrix(Model_mat)
        Covar = model_mat.T @ model_mat
        return Covar
    
    def compute_objective(self, Model_mat):
        Covar = self.calc_covar_matrix(Model_mat)

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)
        return value

    def compute_objective_tf(self, X, m, n):
        batch_size = tf.shape(X)[0]
        X = tf.reshape(X, (-1, m, n))  # shape: (batch_size, m, n)

        def build_model_matrix(X_single):
            Z = tf.ones((m, 1), dtype=X_single.dtype)
            Z = tf.concat([Z, X_single], axis=1)
            return Z

        model_matrices = tf.map_fn(build_model_matrix, X)  # shape: (batch_size, m, num_features_model_matrix)

        Z_t_Z = tf.matmul(model_matrices, model_matrices,
                          transpose_a=True)  # shape: (batch_size, num_features, num_features)

        det = tf.linalg.det(Z_t_Z)
        epsilon = 1e-6
        condition = tf.abs(det)[:, None, None] < epsilon

        diagonal = tf.linalg.diag_part(Z_t_Z) + epsilon
        Z_t_Z_epsilon = Z_t_Z + tf.linalg.diag(diagonal - tf.linalg.diag_part(Z_t_Z))
        Z_t_Z_regularized = tf.where(condition, Z_t_Z_epsilon, Z_t_Z)

        M = tf.linalg.inv(Z_t_Z_regularized)
        value = tf.linalg.trace(M)
        return tf.where(value < 0, tf.constant(1e10, dtype=value.dtype), value)

    def compute_objective_bo(self, X, m, n):
        try:
            X = np.array(X).reshape(m, -1)  # infer the correct number of columns
            model_mat = self.calc_model_matrix(X)
            ZtZ = model_mat.T @ model_mat
            M = np.linalg.inv(ZtZ)
            result = np.trace(M)
            return 1e10 if result < 0 else result
        except np.linalg.LinAlgError:
            return 1e10