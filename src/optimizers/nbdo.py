from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import clear_session
import gc
from scipy.stats import qmc
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from base_optimizer import BaseOptimizer
from models.sos import ScalarOnScalarModel

class NBDO(BaseOptimizer):
    def __init__(self, model, latent_dim, base=2, max_layers=None, alpha=0.0, latent_space_act='tanh', output_layer_act='tanh'):
        self.model = model
        self.runs = None
        self.latent_dim
        self.base = base
        self.max_layers = max_layers
        self.alpha = alpha
        self.latent_space_act = latent_space_act
        self.latent_layer_act = output_layer_act

        self.input_dim = None
        self.encoder = None
        self.latent = None
        self.decoder = None
        self.autoencoder = None
        self.num_layers = None

        self.input_layer = None
        self.output_layer = None

        self.train_set = None
        self.val_set = None
        self.history = None

        self.optimal_latent_var = None
        self.optimal_cr = None
        self.optimal_des = None
        self.search_history = None
        self.eval_history = None

    def compute_train_set(self, num_designs: int, runs: int, random_state: int | None =42) -> None:
        """
        Build flattened design vectors for AE training/validation.

        Parameters
        ----------
        num_designs : int
            Number of candidate designs to generate (L).
        runs : int
            Number of experimental runs (n).
        epsilon : float, optional
            Keep samples strictly within (-1, 1) by shrinking bounds.
        random_state : int | None, optional
            RNG seed for reproducibility.

        Side effects
        ------------
        Sets self.train_set, self.val_set, self.input_dim, and self.runs.
        """
        
        self.runs = int(runs)
        num_features = int(self.model.Kx)

        rng = np.random.default_rng(random_state)
        D = rng.uniform(low, high, size=(num_designs, self.runs, num_features)).astype(np.float32, copy=False)
        X = D.reshape(num_designs, self.runs * num_features)

        self.train_set, self.val_set = train_test_split(X,
         test_size=0.20,
         random_state=random_state, 
         shuffle=True)
        self.input_dim = X.shape[1]