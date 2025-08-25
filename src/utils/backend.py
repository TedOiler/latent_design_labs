from typing import Union
import numpy as np
import tensorflow as tf

ArrayLike = Union[np.ndarray, tf.Tensor]

def to_tensor(x: ArrayLike, dtype=tf.float32) -> tf.Tensor:
    # No copy if already a Tensor. Ensures single dtype everywhere.
    return x if isinstance(x, tf.Tensor) else tf.convert_to_tensor(x, dtype=dtype)

def to_float(x: Union[tf.Tensor, float]) -> float:
    # Pulls a scalar tensor to a Python float (for BO, logging, etc.)
    if isinstance(x, tf.Tensor):
        # x should be rank-0 here
        return float(x.numpy())
    return float(x)