from latent_design.models.base_model import BaseModel
from latent_design.models.sos import ScalarOnScalarModel
from latent_design.models.sof import ScalarOnFunctionModel
from latent_design.models.fof import FunctionOnFunctionModel

# Create aliases for easier importing
SOS = ScalarOnScalarModel
SOF = ScalarOnFunctionModel
FOF = FunctionOnFunctionModel

__all__ = [
    "BaseModel",
    "ScalarOnScalarModel",
    "ScalarOnFunctionModel", 
    "FunctionOnFunctionModel",
    # Aliases
    "SOS",
    "SOF", 
    "FOF",
]