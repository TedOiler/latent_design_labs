from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Optimize the model's design matrix to meet specific criteria."""
        pass