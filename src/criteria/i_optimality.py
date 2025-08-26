from __future__ import annotations
import numpy as np
import tensorflow as tf
from .base_psi import BasePsi

class IOptimality(BasePsi):
    """
    I-optimality criterion implementation.
    TODO: Implement the actual I-optimality logic.
    """
    
    def loss_from_M(self, M):
        # TODO: Implement I-optimality loss
        raise NotImplementedError("I-optimality not yet implemented")
    
    def report_from_M(self, M):
        # TODO: Implement I-optimality reporting
        raise NotImplementedError("I-optimality not yet implemented")
