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

from .base_optimizer import BaseOptimizer
from models.sos import ScalarOnScalarModel

class NBDO(BaseOptimizer):
    def __init__(self, model, latent_dim, max_layers=None, alpha=0.0, latent_space_act='tanh', output_layer_act='tanh'):
        self.model = model
        self.runs = None
        self.latent_dim = latent_dim
        self.max_layers = max_layers
        self.alpha = alpha
        self.latent_space_act = latent_space_act
        self.output_layer_act = output_layer_act

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
        self.optimal_criterion = None
        self.optimal_design = None
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

        epsilon = 1e-8
        low, high = -1 + epsilon, 1 - epsilon
        rng = np.random.default_rng(random_state)
        D = rng.uniform(-1, 1, size=(num_designs, self.runs, num_features)).astype(np.float32, copy=False)
        X = D.reshape(num_designs, self.runs * num_features)

        self.train_set, self.val_set = train_test_split(X,
         test_size=0.20,
         random_state=random_state, 
         shuffle=True)
        self.input_dim = X.shape[1]

    def _build_encoder(self):
        self.num_layers = int(np.log(self.input_dim / self.latent_dim) / np.log(2))
        self.num_layers = min(self.num_layers, self.max_layers) if self.max_layers is not None else self.num_layers
        
        self.input_layer = Input(shape=(self.input_dim,))
        encoder = self.input_layer
        for layer in range(self.num_layers):
            n_neurons = int(self.input_dim / 2 ** (layer + 1))
            encoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(encoder)

        latent = Dense(self.latent_dim, activation=self.latent_space_act, name='latent')(encoder)
        self.encoder = Model(self.input_layer, latent, name='encoder')
    
    def _build_decoder(self):

        latent_inputs = Input(shape=(self.latent_dim,))
        decoder = latent_inputs
        for layer in range(self.num_layers, 0, -1):
            n_neurons = int(self.input_dim / 2 ** layer)
            decoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(decoder)
        self.output_layer = Dense(self.input_dim, activation=self.output_layer_act)(decoder)
        self.decoder = Model(latent_inputs, self.output_layer, name='decoder')

    def _build_autoencoder(self):
        self._build_encoder()
        self._build_decoder()

        autoencoder_input = self.input_layer
        latent_representation = self.encoder(autoencoder_input)
        autoencoder_output = self.decoder(latent_representation)

        self.autoencoder = Model(inputs=autoencoder_input, outputs=autoencoder_output, name='autoencoder')

    def _get_custom_loss(self):
        if isinstance(self.model, ScalarOnScalarModel):
            def custom_loss(y_true, y_pred):
                objective_value = self.model.compute_objective_tf(y_pred, self.runs, self.model.Kx)
                return objective_value
            return custom_loss

    def fit(self, epochs, batch_size=32, patience=50, optimizer=RMSprop()):
        self._build_autoencoder()
        custom_loss = self._get_custom_loss()
        self.autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.autoencoder.build(input_shape=(None, self.input_dim))
        self.history = self.autoencoder.fit(self.train_set, self.train_set,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(self.val_set, self.val_set),
                                            callbacks=[early_stopping],
                                            verbose=0)
        return self.history

    def clear_memory(self):
        del self.autoencoder
        del self.encoder
        del self.decoder
        gc.collect()

    def encode(self, design):
        return self.encoder.predict(design.reshape(1, -1))
    
    def decode(self, latent):
        return self.decoder.predict(latent).reshape(self.runs, -1)
    
    def optimize(self, n_calls=10, acq_func='EI', acq_optimizer='sampling', n_random_starts=5, verbose=True):
        
        def objective(latent_var):
            latent_var = np.array(latent_var).reshape(1, -1)
            decoded = self.decoder.predict(latent_var)
            if isinstance(self.model, ScalarOnScalarModel):
                optimality = self.model.compute_objective_bo(X=decoded, m=self.runs, n=self.model.Kx)
                return optimality

        dimensions = [(-1, 1) for _ in range(self.latent_dim)]
        res = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42, verbose=verbose, n_jobs=-1,
                            n_random_starts=n_random_starts, acq_func=acq_func, acq_optimizer=acq_optimizer)
        self.optimal_latent_var = res.x
        self.optimal_criterion = res.fun
        self.optimal_design = self.decode(np.array(self.optimal_latent_var).reshape(1, -1))
        self.search_history = res.x_iters
        self.eval_history = res.func_vals
        clear_session()
        return self.optimal_criterion, self.optimal_design