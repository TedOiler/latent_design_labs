from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def compute_objective(self, *args, **kwargs):
        """Compute the objective function for the model."""
        pass