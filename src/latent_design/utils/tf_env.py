# src/latent_design/utils/tf_env.py
import os
import random
import numpy as np
import tensorflow as tf

def initialize(
    seed: int = 42,
    *,
    floatx: str = "float64",
    inter_threads: int = 1,
    intra_threads: int = 1,
    deterministic: bool = True,
    quiet_tf: bool = True,
    cpp_log_level: str = "2",   # <— add this (0..3); 3 hides ERRORs too
) -> None:
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    if quiet_tf:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = cpp_log_level  # <— change this line
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) Seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 3) Threads
    try:
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    except Exception:
        # Some TF builds/platforms may not allow changing threads—ignore quietly.
        pass

    # 4) Global float policy (affects Keras layer creation & converts via backend.floatx)
    if floatx:
        tf.keras.backend.set_floatx(floatx)
