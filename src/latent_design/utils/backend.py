from typing import Union
import numpy as np
import tensorflow as tf

ArrayLike = Union[np.ndarray, tf.Tensor]

def to_tensor(x, dtype=None):
    # BEFORE (likely hard-coded): dtype = tf.float32
    if dtype is None:
        # Use the current global Keras float policy
        dtype = tf.as_dtype(tf.keras.backend.floatx())
    return tf.convert_to_tensor(x, dtype=dtype)

def to_float(x: Union[tf.Tensor, float]) -> float:
    # Pulls a scalar tensor to a Python float (for BO, logging, etc.)
    if isinstance(x, tf.Tensor):
        # x should be rank-0 here
        return float(x.numpy())
    return float(x)