from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Protocol

class BaseOptimizer(ABC):
    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        """Optimize the model's design matrix to meet specific criteria."""
        ...