from __future__ import annotations
from time import perf_counter
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TextColumn

from pyclbr import Function
from typing import Any, Callable, List, Optional, Tuple, Union
import warnings
import gc
import random
import numpy as np
import tensorflow as tf

def _policy_tf_dtype() -> tf.DType:
    return tf.as_dtype(tf.keras.backend.floatx())

def _policy_np_dtype():
    return np.float64 if tf.keras.backend.floatx() == "float64" else np.float32

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import RMSprop, Optimizer
from tensorflow.keras.backend import clear_session

from scipy.stats import qmc
from skopt import gp_minimize
from skopt.utils import OptimizeResult

from latent_design.optimizers.base_optimizer import BaseOptimizer
from latent_design.models.sos import ScalarOnScalarModel
from latent_design.models.sof import ScalarOnFunctionModel
from latent_design.models.fof import FunctionOnFunctionModel
from latent_design.models.base_model import BaseModel


# --- Tied decoder layer -------------------------------------------------------
class TiedDense(tf.keras.layers.Layer):
    def __init__(self, tied_to: Dense, activation: Optional[Union[str, Callable]] = None, use_bias: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.tied_to = tied_to
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.bias: Optional[tf.Variable] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        out_units = int(self.tied_to.kernel.shape[0])
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(out_units,), initializer="zeros", trainable=True
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        y = tf.linalg.matmul(inputs, tf.transpose(self.tied_to.kernel))
        if self.bias is not None:
            y = y + self.bias
        return self.activation(y) if self.activation is not None else y

# --- Progress bar for training ------------------------------------------------
class _RichEpochProgress(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, enable: bool):
        super().__init__()
        self.total_epochs = total_epochs
        self.enable = enable
        self._progress = None
        self._task = None
        self._completed = 0

    def on_train_begin(self, logs=None):
        if not self.enable:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]AE training[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )
        self._progress.start()
        self._task = self._progress.add_task("epochs", total=self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if self._progress:
            self._progress.update(self._task, advance=1)
        # Optional external hook
        try:
            if hasattr(self.model, "_nbdo_owner") and getattr(self.model._nbdo_owner, "progress_hook", None):
                getattr(self.model._nbdo_owner, "progress_hook")({
                    "phase": "fit",
                    "epoch": int(epoch) + 1,
                    "total_epochs": int(self.total_epochs),
                    "loss": None if logs is None else float(logs.get("loss", float("nan"))),
                })
        except Exception:
            pass

    def on_train_end(self, logs=None):
        if self._progress:
            self._progress.stop()

# --- NBDO ---------------------------------------------------------------------
class NBDO(BaseOptimizer):
    """Neural Bayesian Design Optimization using tied-weight autoencoders."""
    
    def __init__(
        self,
        model: BaseModel,
        latent_dim: int,
        max_layers: Optional[int] = None,
        alpha: float = 0.0,                  # kept for API compatibility (hidden are linear now)
        latent_space_act: str = "tanh",
        output_layer_act: str = "tanh",
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize NBDO optimizer with model and architecture parameters."""
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.model: BaseModel = model
        self.runs: Optional[int] = None
        self.latent_dim: int = latent_dim
        self.max_layers: Optional[int] = max_layers
        self.alpha: float = alpha
        self.latent_space_act: str = latent_space_act
        self.output_layer_act: str = output_layer_act
        self.seed: Optional[int] = seed
        self.verbose: bool = verbose

        self.input_dim: Optional[int] = None
        self.encoder: Optional[Model] = None
        self.latent: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.num_layers: Optional[int] = None

        self.input_layer: Optional[Input] = None
        self.output_layer: Optional[Any] = None

        self.train_set: Optional[np.ndarray] = None
        self.history: Optional[History] = None

        self.optimal_latent_var: Optional[List[float]] = None
        self.optimal_criterion: Optional[float] = None
        self.optimal_design: Optional[np.ndarray] = None
        self.optimal_report: Optional[float] = None  # positive, human-friendly criterion
        self.search_history: Optional[List[List[float]]] = None
        self.eval_history: Optional[List[float]] = None

        self.time_fit_s: Optional[float] = None
        self.time_bo_s: Optional[float] = None
        self.time_total_s: Optional[float] = None

        self.progress_hook: Optional[Callable[[dict], None]] = None

        if self.seed is not None:
            self._set_all_seeds(self.seed)

    # --- utils ----------------------------------------------------------------
    def _set_all_seeds(self, seed: int) -> None:
        """Set seeds for all random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def _get_random_state(self) -> Optional[int]:
        """Get the random state for functions that require it."""
        return self.seed

    # --- data -----------------------------------------------------------------
    def compute_train_set(self, num_designs: int, runs: int, random_state: Optional[int] = None) -> None:
        """Generate training data using Sobol sequences with boundary points."""
        if num_designs <= 0:
            raise ValueError("num_designs must be positive")
        if runs <= 0:
            raise ValueError("runs must be positive")

        effective_random_state = random_state if random_state is not None else self._get_random_state()

        self.runs = int(runs)
        num_features = int(self.model.Kx)
        D_flat = self.runs * num_features

        sampler = qmc.Sobol(d=D_flat, scramble=True, seed=effective_random_state)
        warnings.simplefilter("ignore", category=UserWarning)
        dtype_np = _policy_np_dtype()
        X_unit = sampler.random(num_designs).astype(dtype_np)
        X = 2.0 * X_unit - 1.0

        # include exact corners to ensure boundaries are reachable during training
        X = np.concatenate(
            [
                X,
                np.full((1, D_flat), -1.0, dtype=dtype_np),
                np.full((1, D_flat),  1.0, dtype=dtype_np),
            ],
            axis=0
        )

        self.train_set = X
        self.input_dim = D_flat

    # --- model ----------------------------------------------------------------
    def _build_encoder(self) -> None:
        """Build encoder with automatically computed layer widths."""
        if self.input_dim is None:
            raise RuntimeError("input_dim is not set. Call compute_train_set() first.")
        D, d = int(self.input_dim), int(self.latent_dim)

        ratio = max(D / float(d), 1.000001)
        S = int(np.ceil(np.log(ratio) / np.log(8.0)))
        if S < 1:
            S = 1
        if self.max_layers is not None:
            S = min(S, int(self.max_layers))

        widths = []
        for k in range(1, S + 1):
            t = k / (S + 1.0)
            w = int(np.round((D ** (1 - t)) * (d ** t)))
            widths.append(w)
        for i in range(len(widths)):
            prev = D if i == 0 else widths[i - 1]
            widths[i] = min(widths[i], prev - 1)
            widths[i] = max(widths[i], d + 1)
        for i in range(1, len(widths)):
            if widths[i] >= widths[i - 1]:
                widths[i] = max(d + 1, widths[i - 1] - 1)

        self.hidden_widths = widths
        self.num_layers = len(widths)

        self.input_layer = Input(shape=(D,))
        x = self.input_layer

        self._enc_dense_layers: List[Dense] = []
        for w in widths:
            layer = Dense(w, activation=None)   # linear hidden
            x = layer(x)
            self._enc_dense_layers.append(layer)

        latent_layer = Dense(d, activation=self.latent_space_act, name="latent")
        latent = latent_layer(x)
        self._enc_dense_layers.append(latent_layer)

        self.encoder = Model(self.input_layer, latent, name="encoder")

    def _build_decoder(self) -> None:
        """Build decoder with weights tied to encoder layers."""
        if not hasattr(self, "_enc_dense_layers") or not self._enc_dense_layers:
            raise RuntimeError("Encoder must be built before decoder for tied weights.")
        d = int(self.latent_dim)

        latent_inputs = Input(shape=(d,))
        x = latent_inputs
        for i, enc_layer in enumerate(reversed(self._enc_dense_layers)):
            is_last = (i == len(self._enc_dense_layers) - 1)
            act = self.output_layer_act if is_last else None
            x = TiedDense(tied_to=enc_layer, activation=act)(x)

        self.output_layer = x
        self.decoder = Model(latent_inputs, x, name="decoder")

    def _build_autoencoder(self) -> None:
        """Build complete autoencoder by connecting encoder and decoder."""
        self._build_encoder()
        self._build_decoder()
        autoencoder_input: Any = self.input_layer
        latent_representation: Any = self.encoder(autoencoder_input)
        autoencoder_output: Any = self.decoder(latent_representation)
        self.autoencoder = Model(inputs=autoencoder_input, outputs=autoencoder_output, name="autoencoder")

    # --- loss -----------------------------------------------------------------
    def _get_custom_loss(self) -> Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        """Get custom loss function for the current model."""
        if isinstance(self.model, ScalarOnScalarModel):
            def custom_loss(_y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                # y_pred: (B, runs*Kx) — SoS path (no dtype attribute on the model)
                return self.model.objective_from_flat(y_pred, self.runs, self.model.Kx)
            return custom_loss

        # --- SoF path: add a custom loss and cast to the model's dtype (float64 by default) ---
        if isinstance(self.model, ScalarOnFunctionModel):
            def custom_loss(_y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                target_dtype = getattr(self.model, "dtype", y_pred.dtype)
                if y_pred.dtype != target_dtype:
                    y_pred = tf.cast(y_pred, target_dtype)
                return self.model.objective_from_flat(y_pred, self.runs)
            return custom_loss
        
        # --- FoF path (explicit): identical to SoF today; will diverge later ---
        if isinstance(self.model, FunctionOnFunctionModel):
            def custom_loss(_y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                target_dtype = getattr(self.model, "dtype", y_pred.dtype)
                if y_pred.dtype != target_dtype:
                    y_pred = tf.cast(y_pred, target_dtype)
                return self.model.objective_from_flat(y_pred, self.runs)
            return custom_loss

        return None

    # --- training -------------------------------------------------------------
    def fit(
        self,
        epochs: int,
        batch_size: int = 32,
        patience: int = 50,
        optimizer: Optimizer = RMSprop(learning_rate=1e-3, rho=0.9, epsilon=1e-7, clipnorm=5.0),
    ) -> History:
        """Train autoencoder with early stopping and learning rate reduction."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if self.train_set is None:
            raise RuntimeError("Must call compute_train_set() before fit()")

        t0 = perf_counter()

        self._build_autoencoder()
        try:
            self.autoencoder._nbdo_owner = self
        except Exception:
            pass

        custom_loss = self._get_custom_loss()
        self.autoencoder.compile(optimizer=optimizer, loss=custom_loss)

        early_stopping = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5, verbose=0
        )
        rich_cb = _RichEpochProgress(total_epochs=epochs, enable=self.verbose)  # <<< rich progress bar

        self.autoencoder.build(input_shape=(None, self.input_dim))
        self.history = self.autoencoder.fit(
            self.train_set,
            self.train_set,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, rich_cb],
            shuffle=False,
            verbose=0,
        )
        self.time_fit_s = perf_counter() - t0
        self.time_total_s = (self.time_fit_s or 0.0) + (self.time_bo_s or 0.0)
        return self.history
        
    # --- memory ---------------------------------------------------------------
    def clear_memory(self) -> None:
        """Free TF/keras objects and clear graph state."""
        try:
            clear_session()
        finally:
            # Best-effort cleanup of references
            del self.autoencoder
            del self.encoder
            del self.decoder
            gc.collect()

    # --- encode/decode --------------------------------------------------------
    def encode(self, design: np.ndarray) -> np.ndarray:
        """Encode design vector to latent representation."""
        if self.encoder is None:
            raise RuntimeError("Encoder not available. Call fit() first.")
        return self.encoder.predict(design.reshape(1, -1), verbose=0)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent representation back to design space."""
        if self.decoder is None:
            raise RuntimeError("Decoder not available. Call fit() first.")
        return self.decoder.predict(latent, verbose=0).reshape(self.runs, -1)

    # --- BO -------------------------------------------------------------------
    def optimize(
        self,
        n_calls: int = 10,
        acq_func: str = "EI",
        acq_optimizer: str = "sampling",
        n_random_starts: int = 5,
    ) -> Tuple[float, np.ndarray]:
        """Optimize design using Bayesian optimization in latent space."""
        if n_calls <= 0:
            raise ValueError("n_calls must be positive")
        if n_random_starts <= 0:
            raise ValueError("n_random_starts must be positive")
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not available. Call fit() first.")

        def objective(latent_var: List[float]) -> float:
            # latent → decoded design (flat)
            z = tf.convert_to_tensor([latent_var], dtype=_policy_tf_dtype())   # (1, d)
            decoded = self.decoder(z, training=False)                  # (1, runs*Kx) float32

            # Align dtypes with the model (SoF uses float64)
            target_dtype = getattr(self.model, "dtype", decoded.dtype)
            if decoded.dtype != target_dtype:
                decoded = tf.cast(decoded, target_dtype)

            # Delegate to the appropriate model path
            if isinstance(self.model, ScalarOnScalarModel):
                loss_t = self.model.objective_from_flat(decoded, self.runs, self.model.Kx)  # (1,)
            elif isinstance(self.model, ScalarOnFunctionModel):
                loss_t = self.model.objective_from_flat(decoded, self.runs)                 # (1,)
            elif isinstance(self.model, FunctionOnFunctionModel):
                # identical to SoF today; explicit branch for future divergence
                loss_t = self.model.objective_from_flat(decoded, self.runs)
            else:
                raise TypeError(f"Unsupported model type: {type(self.model)}")

            # Return Python float for skopt
            return float(tf.reshape(loss_t, ()).numpy())

        dimensions: List[Tuple[float, float]] = [(-1.0, 1.0) for _ in range(self.latent_dim)]
        effective_random_state = self._get_random_state()

        # Warm-start BO from encoded training designs (deterministic)
        M = min(max(128, 16 * self.latent_dim), len(self.train_set))
        K = max(1, min(10, M // 8))
        rng = np.random.default_rng(effective_random_state)
        idx = rng.choice(len(self.train_set), size=M, replace=False)

        Z0 = self.encoder.predict(self.train_set[idx], verbose=0)

        seed_vals: List[float] = [objective(z.tolist()) for z in Z0]
        order = np.argsort(seed_vals)[:K]
        x0 = [Z0[i].tolist() for i in order]
        y0 = [float(seed_vals[i]) for i in order]

        t1 = perf_counter()  # <<< start timing BO

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Bayesian Optimization[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) if self.verbose else None

        def _bo_progress_cb(res_obj):
            # skopt calls after each iteration
            if progress:
                progress.update(task, advance=1)
            # Optional external hook for a web UI
            try:
                if self.progress_hook:
                    self.progress_hook({
                        "phase": "bo",
                        "iters_completed": len(res_obj.x_iters),
                        "iters_total": int(n_calls),
                        "best_so_far": float(res_obj.fun) if hasattr(res_obj, "fun") else None,
                    })
            except Exception:
                pass
        
        if progress:
            progress.start()
            task = progress.add_task("bo", total=n_calls)
        try:
            res: OptimizeResult = gp_minimize(
                objective,
                dimensions,
                n_calls=n_calls,
                random_state=effective_random_state,
                verbose=0,
                n_jobs=1,
                n_random_starts=n_random_starts,
                acq_func=acq_func,
                acq_optimizer=acq_optimizer,
                x0=x0,
                y0=y0,
                callback=[_bo_progress_cb]
            )
        finally:
            if progress:
                progress.stop()
        
        self.time_bo_s = perf_counter() - t1      # <<< stop timing
        self.time_total_s = (self.time_fit_s or 0.0) + (self.time_bo_s or 0.0)

        self.optimal_latent_var = res.x
        self.optimal_criterion = res.fun
        latent_arr = np.array(self.optimal_latent_var, dtype=_policy_np_dtype()).reshape(1, -1)
        self.optimal_design = self.decode(latent_arr)

        try:
            self.optimal_report = float(self.model.report_num(self.optimal_design))

        except Exception:
            self.optimal_report = None  # fallback if something goes wrong
        self.search_history = res.x_iters
        self.eval_history = res.func_vals

        return self.optimal_report, self.optimal_design