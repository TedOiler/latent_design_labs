from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class GOptimality(BasePsi):
    """
    G-optimality criterion implementation.
    TODO: Implement the actual G-optimality logic.
    """
    
    def loss_from_M(self, M):
        # TODO: Implement G-optimality loss
        raise NotImplementedError("G-optimality not yet implemented")
    
    def report_from_M(self, M):
        # TODO: Implement G-optimality reporting
        raise NotImplementedError("G-optimality not yet implemented")
