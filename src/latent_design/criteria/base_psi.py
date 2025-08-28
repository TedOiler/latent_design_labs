import tensorflow as tf
from latent_design.utils.backend import to_float  # <- uses your existing util

class BasePsi:
    def loss_from_M(self, M) -> tf.Tensor:
        raise NotImplementedError

    def report_from_M(self, M) -> tf.Tensor:
        raise NotImplementedError

    # --- numeric helpers, unified default implementations ---
    def loss_from_M_num(self, M) -> float:
        # Expect scalar or (B,) tensor; squeeze to scalar and convert
        return to_float(tf.reshape(self.loss_from_M(M), ()))

    def report_from_M_num(self, M) -> float:
        return to_float(tf.reshape(self.report_from_M(M), ()))
