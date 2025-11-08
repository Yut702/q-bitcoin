import os
import math
import argparse
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


# ========= Configuration =========
# Set to a specific path or leave as None to auto-detect a local sample.
DATA_PATH: Optional[str] = None  # e.g., "Dataset_Simulated_Price_swaption.xlsx"

# PCA + PQC dimensions
N_PCA_COMPONENTS: int = 6
ENTANGLING_DEPTH: int = 2

# Train/test split as a fraction of timesteps (sequential split)
TEST_FRACTION: float = 0.2

# Optional fixed surface grid shape for heatmaps. If None, attempts to infer a square.
GRID_SHAPE: Optional[Tuple[int, int]] = None


# ========= Data Loading =========
def find_default_data_path() -> Optional[str]:
    candidates = [
        "sample_Simulated_Swaption_Price.xlsx",
        "Dataset_Simulated_Price_swaption.xlsx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        # Use first sheet by default
        df = pd.read_excel(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data extension: {ext}")
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def dataframe_to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[pd.Series]]:
    # Identify a time-like column by name; otherwise, use datetime index if present
    time_col = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("date", "time", "timestamp", "dt")):
            time_col = c
            break

    timeline: Optional[pd.Series]
    df_work = df.copy()
    if time_col is not None:
        timeline = pd.to_datetime(df_work[time_col], errors='coerce')
        df_work = df_work.drop(columns=[time_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        timeline = df.index.to_series()
    else:
        timeline = None

    # Coerce all remaining columns to numeric, dropping non-numeric columns
    df_numeric = df_work.apply(pd.to_numeric, errors='coerce')
    # Drop columns that are entirely NaN after coercion
    df_numeric = df_numeric.dropna(axis=1, how='all')
    # Replace inf and drop rows with any NaNs
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    if df_numeric.shape[0] < 3 or df_numeric.shape[1] < 2:
        raise ValueError(
            f"Dataset must have at least 3 timesteps and 2 numeric features after cleaning; got {df_numeric.shape}."
        )

    return df_numeric.to_numpy(dtype=float), timeline


def sequential_train_test_split(X: np.ndarray, test_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    test_size = max(1, int(round(n * test_fraction)))
    split = n - test_size
    X_train = X[:split]
    X_test = X[split:]
    return X_train, X_test


def make_next_step_pairs(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Create supervised pairs for next-step prediction: (x_t -> x_{t+1})
    return X[:-1], X[1:]


# ========= PCA Pipeline =========
def fit_pca_on_train(X_train: np.ndarray, n_components: int):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X_train)
    max_components = min(Xtr_s.shape[0], Xtr_s.shape[1])
    n_comp = min(n_components, max_components)
    if n_comp < 1:
        raise ValueError("PCA requires at least 1 component.")
    if n_comp != n_components:
        print(f"[Info] Reducing PCA components from {n_components} to {n_comp} to fit data.")
    pca = PCA(n_components=n_comp, svd_solver='auto', random_state=0)
    Xtr_pca = pca.fit_transform(Xtr_s)
    return scaler, pca, Xtr_pca


def transform_with_pca(scaler: StandardScaler, pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(scaler.transform(X))


def inverse_pca(scaler: StandardScaler, pca: PCA, Xp: np.ndarray) -> np.ndarray:
    X_s = pca.inverse_transform(Xp)
    return scaler.inverse_transform(X_s)


# ========= PQC Model =========
def angle_encoding_layer(qc: QuantumCircuit, x: np.ndarray):
    for i, val in enumerate(x):
        qc.ry(float(val), i)


def entangling_ansatz(qc: QuantumCircuit, thetas: np.ndarray, depth: int):
    n = qc.num_qubits
    idx = 0
    for _ in range(depth):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        # Parameterized single-qubit rotations
        for i in range(n):
            qc.ry(float(thetas[idx]), i); idx += 1
            qc.rz(float(thetas[idx]), i); idx += 1


def build_angle_circuit(x_angles: np.ndarray, thetas: np.ndarray, depth: int) -> QuantumCircuit:
    n_qubits = len(x_angles)
    qc = QuantumCircuit(n_qubits)
    for i, ang in enumerate(x_angles):
        qc.ry(float(ang), i)
    entangling_ansatz(qc, thetas, depth)
    return qc


def z_expectations_from_state(state: Statevector, n_qubits: int) -> np.ndarray:
    # Compute <Z> per qubit using probabilities and bit-parity trick (little-endian)
    probs = np.abs(state.data) ** 2
    exps = np.zeros(n_qubits, dtype=float)
    dim = 2 ** n_qubits
    basis = np.arange(dim, dtype=np.uint64)
    for k in range(n_qubits):
        signs = 1.0 - 2.0 * ((basis >> k) & 1).astype(float)
        exps[k] = float(np.dot(probs, signs))
    return exps


class PQCRegressor:
    def __init__(self, n_qubits: int, depth: int = 2, random_state: int = 0):
        self.n_qubits = n_qubits
        self.depth = depth
        self.rng = np.random.default_rng(random_state)
        # Each layer: for each qubit we have two params (ry, rz)
        self.theta = self.rng.normal(0, 0.1, size=(depth * n_qubits * 2,))
        self.loss_history: List[float] = []
        # Scalers to align ranges between PCA space and PQC outputs
        self.in_scaler: Optional[MinMaxScaler] = None
        self.out_scaler: Optional[MinMaxScaler] = None

    def _predict_single(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(self.n_qubits)
        angle_encoding_layer(qc, x)
        entangling_ansatz(qc, theta, self.depth)
        state = Statevector.from_instruction(qc)
        zexp = z_expectations_from_state(state, self.n_qubits)
        return zexp

    def predict(self, X: np.ndarray, theta: Optional[np.ndarray] = None, scaled: bool = False) -> np.ndarray:
        th = self.theta if theta is None else theta
        # Ensure input scaler is present (identity if absent)
        X_enc = X
        if self.in_scaler is not None:
            X_enc = self.in_scaler.transform(X)
        preds_scaled = [self._predict_single(x, th) for x in X_enc]
        preds_scaled = np.asarray(preds_scaled)
        if scaled:
            return preds_scaled
        # Inverse target scaling back to PCA space
        if self.out_scaler is not None:
            return self.out_scaler.inverse_transform(preds_scaled)
        return preds_scaled

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 200, verbose: bool = True, restarts: int = 0):
        # Fit scalers: inputs to angle range, targets to [-1, 1]
        self.in_scaler = MinMaxScaler(feature_range=(-math.pi / 2, math.pi / 2))
        self.in_scaler.fit(X)
        Xs = self.in_scaler.transform(X)

        self.out_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.out_scaler.fit(y)
        ys = self.out_scaler.transform(y)

        def obj(th):
            preds = [self._predict_single(x, th) for x in Xs]
            preds = np.asarray(preds)
            loss = float(np.mean((preds - ys) ** 2))
            self.loss_history.append(loss)
            return loss

        best_theta = self.theta.copy()
        best_val = obj(best_theta)

        def run_minimize(x0):
            res = minimize(
                obj,
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3, "disp": verbose},
            )
            return res.x, obj(res.x)

        # First run from current theta
        theta0, val0 = run_minimize(best_theta)
        if val0 < best_val:
            best_theta, best_val = theta0, val0

        # Optional random restarts
        for _ in range(max(0, int(restarts))):
            x0 = self.rng.normal(0, 0.2, size=best_theta.shape)
            th, val = run_minimize(x0)
            if val < best_val:
                best_theta, best_val = th, val

        self.theta = best_theta
        return self

    # Helper: build circuit for a single PCA input using this trained model
    def circuit_for_input(self, x_pca: np.ndarray) -> QuantumCircuit:
        x = x_pca.reshape(1, -1)
        if self.in_scaler is not None:
            x_angles = self.in_scaler.transform(x)[0]
        else:
            x_angles = x[0]
        return build_angle_circuit(x_angles, self.theta, self.depth)


# ========= Plotting =========
def guess_grid_shape(n_features: int, preferred: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    if preferred is not None:
        r, c = preferred
        if r * c == n_features:
            return preferred
    s = int(round(math.sqrt(n_features)))
    if s * s == n_features:
        return (s, s)
    # Try common grid widths
    for w in [8, 10, 12, 16, 20]:
        if n_features % w == 0:
            return (n_features // w, w)
    return None


def plot_surface_heatmaps(true_vec: np.ndarray, pred_vec: np.ndarray, grid_shape: Optional[Tuple[int, int]], title_prefix: str = ""):
    if grid_shape is None:
        # Fallback: plot as 1xF images
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].imshow(true_vec[np.newaxis, :], aspect='auto', cmap='viridis')
        axes[0].set_title(f"{title_prefix}True (flattened)")
        axes[1].imshow(pred_vec[np.newaxis, :], aspect='auto', cmap='viridis')
        axes[1].set_title(f"{title_prefix}Pred (flattened)")
        diff = pred_vec - true_vec
        axes[2].imshow(diff[np.newaxis, :], aspect='auto', cmap='coolwarm')
        axes[2].set_title(f"{title_prefix}Error (flattened)")
        for ax in axes:
            ax.set_yticks([])
            ax.set_xlabel("Feature Index")
        plt.show()
        return

    r, c = grid_shape
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(true_vec.reshape(r, c), cmap='viridis')
    axes[0].set_title(f"{title_prefix}True Surface")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_vec.reshape(r, c), cmap='viridis')
    axes[1].set_title(f"{title_prefix}Predicted Surface")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow((pred_vec - true_vec).reshape(r, c), cmap='coolwarm')
    axes[2].set_title(f"{title_prefix}Error Surface")
    plt.colorbar(im2, ax=axes[2])
    plt.show()


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Parity Plot"):
    t = y_true.flatten()
    p = y_pred.flatten()
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    plt.figure(figsize=(5, 5))
    plt.scatter(t, p, s=8, alpha=0.5)
    plt.plot(lims, lims, 'k--', lw=1)
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.show()


def loss_curve(losses: List[float]):
    if not losses:
        return
    plt.figure(figsize=(6, 3))
    plt.plot(losses, '-o', markersize=3)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.show()


def rbf_style_rollout_plot(
    full_true: np.ndarray,
    full_pred_train: np.ndarray,
    full_pred_test: np.ndarray,
    train_len: int,
    feature_index: int,
    title: str = "RBF-style Rollout",
):
    # Build aligned series
    n_total = full_true.shape[0]
    x_axis = np.arange(n_total)
    # Predictions are next-step; align them starting from index 1
    y_pred_aligned = np.full(n_total, np.nan)
    # Train predictions length is train_len - 1
    train_pred_len = int(full_pred_train.shape[0])
    # Assign exactly the available length to avoid shape mismatch
    y_pred_aligned[1:1 + train_pred_len] = full_pred_train[:, feature_index]
    # Test predictions length is test_len - 1; start at global index train_len + 1
    test_pred_len = int(full_pred_test.shape[0])
    test_start = (1 + train_pred_len) + 1  # == train_len + 1
    test_end = min(n_total, test_start + test_pred_len)
    assign_len = max(0, test_end - test_start)
    if assign_len > 0:
        y_pred_aligned[test_start:test_end] = full_pred_test[:assign_len, feature_index]

    # RMSE on test window only (use matched slices)
    test_true_series = full_true[test_start:test_end, feature_index]
    test_pred_series = full_pred_test[:assign_len, feature_index]
    rmse = math.sqrt(mean_squared_error(test_true_series, test_pred_series)) if assign_len > 0 else float('nan')

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis[:train_len], full_true[:train_len, feature_index], label="Train True", color="tab:blue")
    plt.plot(x_axis[train_len - 1 :], full_true[train_len - 1 :, feature_index], label="Test True", color="tab:orange")
    plt.plot(x_axis, y_pred_aligned, label="PQC Pred", color="tab:green", linestyle="--")
    plt.fill_between(x_axis, y_pred_aligned - rmse, y_pred_aligned + rmse, color="tab:green", alpha=0.1, label=f"±RMSE ({rmse:.4f})")
    plt.title(title + f" (feature {feature_index})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========= Main Pipeline =========
def main():
    parser = argparse.ArgumentParser(description="QML PCA+PQC next-step prediction pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel/CSV dataset")
    parser.add_argument("--n-components", type=int, default=None, help="Number of PCA components / qubits")
    parser.add_argument("--depth", type=int, default=None, help="Entangling depth of the PQC")
    parser.add_argument("--test-fraction", type=float, default=None, help="Test fraction for sequential split")
    parser.add_argument("--grid-rows", type=int, default=None, help="Rows for surface heatmap")
    parser.add_argument("--grid-cols", type=int, default=None, help="Cols for surface heatmap")
    parser.add_argument("--encoding", type=str, default="angle", choices=["angle", "amplitude"], help="Encoding type")
    parser.add_argument("--load-artifacts", type=str, default=None, help="Path to load cached model+PCA artifacts (joblib)")
    parser.add_argument("--save-artifacts", type=str, default=None, help="Path to save cached model+PCA artifacts (joblib)")
    args = parser.parse_args()

    # Resolve configuration
    data_path = args.data or DATA_PATH or find_default_data_path()
    if data_path is None:
        raise FileNotFoundError("No dataset found. Set DATA_PATH to your Excel/CSV path.")
    print(f"Using dataset: {data_path}")

    global N_PCA_COMPONENTS, ENTANGLING_DEPTH, TEST_FRACTION, GRID_SHAPE
    if args.n_components is not None:
        N_PCA_COMPONENTS = args.n_components
    if args.depth is not None:
        ENTANGLING_DEPTH = args.depth
    if args.test_fraction is not None:
        TEST_FRACTION = args.test_fraction
    if args.grid_rows is not None and args.grid_cols is not None:
        GRID_SHAPE = (int(args.grid_rows), int(args.grid_cols))

    df = load_dataset(data_path)
    X_raw, timeline = dataframe_to_matrix(df)
    n_total, n_features = X_raw.shape
    print(f"Loaded matrix shape: {X_raw.shape} (time x features)")

    # Split train/test sequentially
    X_train_raw, X_test_raw = sequential_train_test_split(X_raw, TEST_FRACTION)
    train_len = X_train_raw.shape[0]
    print(f"Train steps: {train_len}, Test steps: {X_test_raw.shape[0]}")

    # Build supervised next-step pairs in the original feature space
    Xtr_pairs_in, Xtr_pairs_out = make_next_step_pairs(X_train_raw)
    Xte_pairs_in, Xte_pairs_out = make_next_step_pairs(X_test_raw)

    # PCA on train
    scaler, pca, Xtr_pca = fit_pca_on_train(X_train_raw, N_PCA_COMPONENTS)
    Xte_pca = transform_with_pca(scaler, pca, X_test_raw)
    print(f"Explained variance (train PCA): {pca.explained_variance_ratio_.sum():.4f}")

    # Supervised pairs in PCA space
    Xtr_in_pca, Xtr_out_pca = make_next_step_pairs(Xtr_pca)
    Xte_in_pca, Xte_out_pca = make_next_step_pairs(Xte_pca)

    # PQC model selection
    def save_artifacts(path: str, scaler_obj, pca_obj, model_obj, encoding: str):
        payload = {"encoding": encoding, "scaler": scaler_obj, "pca": pca_obj}
        if encoding == "angle":
            payload.update({
                "depth": model_obj.depth,
                "n_qubits": model_obj.n_qubits,
                "theta": model_obj.theta,
                "in_scaler": model_obj.in_scaler,
                "out_scaler": model_obj.out_scaler,
            })
        else:
            payload.update({
                "depth": model_obj.depth,
                "n_features": model_obj.n_features,
                "n_qubits": model_obj.n_qubits,
                "theta": model_obj.theta,
                "norm_reg": model_obj.norm_reg,
            })
        joblib.dump(payload, path)
        print(f"[Cache] Saved artifacts to {path}")

    def load_artifacts(path: str):
        payload = joblib.load(path)
        encoding_loaded = payload.get("encoding", "angle")
        scaler_loaded = payload["scaler"]
        pca_loaded = payload["pca"]
        if encoding_loaded == "angle":
            mdl = PQCRegressor(n_qubits=payload["n_qubits"], depth=payload["depth"], random_state=0)
            mdl.theta = payload["theta"]
            mdl.in_scaler = payload.get("in_scaler")
            mdl.out_scaler = payload.get("out_scaler")
        else:
            mdl = PQCRegressorAmplitude(n_features=payload["n_features"], depth=payload["depth"], random_state=0)
            mdl.theta = payload["theta"]
            mdl.norm_reg = payload.get("norm_reg")
        return encoding_loaded, scaler_loaded, pca_loaded, mdl

    # Load-or-train
    model = None
    encoding_used = args.encoding
    if args.load_artifacts:
        try:
            enc_loaded, scaler, pca, model = load_artifacts(args.load_artifacts)
            encoding_used = enc_loaded
            print(f"[Cache] Loaded artifacts from {args.load_artifacts} (encoding={encoding_used})")
        except Exception as e:
            print(f"[Cache] Failed to load artifacts: {e}. Proceeding to train.")

    if model is None:
        if args.encoding == "angle":
            n_qubits = Xtr_in_pca.shape[1]
            model = PQCRegressor(n_qubits=n_qubits, depth=ENTANGLING_DEPTH, random_state=0)
            print(f"[Angle] Training PQC with {n_qubits} qubits, depth {ENTANGLING_DEPTH}...")
        else:
            model = PQCRegressorAmplitude(n_features=Xtr_in_pca.shape[1], depth=ENTANGLING_DEPTH, random_state=0)
            print(f"[Amplitude] Training PQC with {model.n_qubits} qubits (2^{model.n_qubits} >= {Xtr_in_pca.shape[1]}), depth {ENTANGLING_DEPTH}...")
        t0 = time.time()
        if args.encoding == "angle":
            model.fit(Xtr_in_pca, Xtr_out_pca, maxiter=300, verbose=False, restarts=2)
        else:
            model.fit(Xtr_in_pca, Xtr_out_pca, maxiter=300, verbose=False, restarts=2)
        dt = time.time() - t0
        print(f"[Timing] Training took {dt/60:.2f} minutes")
        if args.save_artifacts:
            save_artifacts(args.save_artifacts, scaler, pca, model, args.encoding)

    # Predictions in PCA space
    ytr_pred_pca = model.predict(Xtr_in_pca)
    yte_pred_pca = model.predict(Xte_in_pca)

    # Back to original feature space
    ytr_pred = inverse_pca(scaler, pca, ytr_pred_pca)
    yte_pred = inverse_pca(scaler, pca, yte_pred_pca)
    ytr_true = Xtr_pairs_out
    yte_true = Xte_pairs_out

    # Metrics
    train_rmse = math.sqrt(mean_squared_error(ytr_true, ytr_pred))
    test_rmse = math.sqrt(mean_squared_error(yte_true, yte_pred))
    print(f"Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")

    # Plots
    loss_curve(model.loss_history)

    # Heatmaps for last test step (if possible)
    grid = guess_grid_shape(n_features, GRID_SHAPE)
    try:
        last_true_vec = yte_true[-1]
        last_pred_vec = yte_pred[-1]
        plot_surface_heatmaps(last_true_vec, last_pred_vec, grid, title_prefix="Test Last • ")
    except Exception as e:
        print(f"Heatmap plotting skipped: {e}")

    # Parity plot on test set
    parity_plot(yte_true, yte_pred, title="Parity Plot • Test")

    # RBF-style rollout on a representative feature (center index)
    feat_idx = n_features // 2
    # Build full-length arrays for plotting
    # Note: ytr_pred aligns to 1..train_len-1; yte_pred aligns to train_len..n_total-1
    rbf_style_rollout_plot(
        full_true=X_raw,
        full_pred_train=ytr_pred,
        full_pred_test=yte_pred,
        train_len=train_len,
        feature_index=feat_idx,
        title="RBF-style Rollout",
    )

    print("Done.")


if __name__ == "__main__":
    main()
# ========= Amplitude-Encoding PQC =========
def next_pow2_geq(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def build_ansatz(n_qubits: int, thetas: np.ndarray, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    entangling_ansatz(qc, thetas, depth)
    return qc


class PQCRegressorAmplitude:
    def __init__(self, n_features: int, depth: int = 2, random_state: int = 0):
        # Number of qubits is minimal s.t. 2^n >= n_features
        self.n_features = int(n_features)
        m = next_pow2_geq(self.n_features)
        self.n_qubits = int(math.log2(m))
        self.depth = depth
        self.rng = np.random.default_rng(random_state)
        self.theta = self.rng.normal(0, 0.1, size=(depth * self.n_qubits * 2,))
        self.loss_history: List[float] = []
        # Separate regressor for output norm (magnitude)
        self.norm_reg: Optional[LinearRegression] = None

    def _encode_amplitude(self, vec: np.ndarray) -> np.ndarray:
        # vec: shape (d,). Return padded+normalized complex amplitude (length 2^n)
        d = self.n_features
        m = 1 << self.n_qubits
        pad = np.zeros(m, dtype=complex)
        pad[:d] = vec.astype(float)
        norm = np.linalg.norm(pad)
        if norm < 1e-12:
            # fallback to |0...0>
            pad[:] = 0.0
            pad[0] = 1.0
        else:
            pad /= norm
        return pad

    def _target_state(self, y_dir: np.ndarray) -> np.ndarray:
        return self._encode_amplitude(y_dir)

    def _apply_unitary(self, amp_in: np.ndarray, thetas: np.ndarray) -> Statevector:
        # Build unitary ansatz and evolve the input state
        qc = build_ansatz(self.n_qubits, thetas, self.depth)
        st = Statevector(amp_in)
        return st.evolve(qc)

    def _fidelity(self, psi: np.ndarray, phi: np.ndarray) -> float:
        # |<psi|phi>|^2
        return float(np.abs(np.vdot(psi, phi)) ** 2)

    def fit(self, X_in: np.ndarray, y_out: np.ndarray, maxiter: int = 300, verbose: bool = False, restarts: int = 2):
        # Train norm regressor on magnitudes
        y_norm = np.linalg.norm(y_out, axis=1)
        self.norm_reg = LinearRegression()
        self.norm_reg.fit(X_in, y_norm)

        # Directional targets (unit vectors)
        eps = 1e-12
        X_dir = X_in.copy()
        y_dir = y_out.copy()
        X_dir /= (np.linalg.norm(X_dir, axis=1, keepdims=True) + eps)
        y_dir /= (np.linalg.norm(y_dir, axis=1, keepdims=True) + eps)

        targets = np.asarray([self._target_state(v) for v in y_dir])
        inputs = np.asarray([self._encode_amplitude(v) for v in X_dir])

        def obj(th):
            # Average 1 - fidelity across samples
            vals = []
            for amp_in, target in zip(inputs, targets):
                st = self._apply_unitary(amp_in, th)
                vals.append(1.0 - self._fidelity(st.data, target))
            loss = float(np.mean(vals))
            self.loss_history.append(loss)
            return loss

        best_theta = self.theta.copy()
        best_val = obj(best_theta)

        def run_minimize(x0):
            res = minimize(
                obj,
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3, "disp": verbose},
            )
            return res.x, obj(res.x)

        th0, v0 = run_minimize(best_theta)
        if v0 < best_val:
            best_theta, best_val = th0, v0
        for _ in range(max(0, int(restarts))):
            x0 = self.rng.normal(0, 0.2, size=best_theta.shape)
            th, v = run_minimize(x0)
            if v < best_val:
                best_theta, best_val = th, v

        self.theta = best_theta
        return self

    def predict(self, X_in: np.ndarray) -> np.ndarray:
        # Predict directional vector via amplitude model, then scale by predicted norm
        eps = 1e-12
        X_dir = X_in / (np.linalg.norm(X_in, axis=1, keepdims=True) + eps)
        preds = []
        for v in X_dir:
            amp_in = self._encode_amplitude(v)
            st = self._apply_unitary(amp_in, self.theta)
            amp = st.data
            y_dir_pred = np.real(amp[: self.n_features])
            # Re-normalize direction
            y_dir_pred /= (np.linalg.norm(y_dir_pred) + eps)
            preds.append(y_dir_pred)
        preds = np.asarray(preds)
        # Predict norms and reconstruct
        if self.norm_reg is None:
            norms = np.ones((preds.shape[0],), dtype=float)
        else:
            norms = self.norm_reg.predict(X_in)
            norms = np.maximum(norms, 0.0)
        return preds * norms[:, None]

    # Helper: build circuit including state initialization and ansatz
    def circuit_for_input(self, x_pca: np.ndarray) -> QuantumCircuit:
        v = x_pca.astype(float)
        # Directional normalization and padding to 2^n
        eps = 1e-12
        v_dir = v / (np.linalg.norm(v) + eps)
        m = 1 << self.n_qubits
        amp = np.zeros(m, dtype=complex)
        amp[: self.n_features] = v_dir
        norm = np.linalg.norm(amp)
        if norm > 1e-12:
            amp /= norm
        else:
            amp[:] = 0.0
            amp[0] = 1.0
        qc = QuantumCircuit(self.n_qubits)
        try:
            qc.initialize(amp)
        except Exception:
            # Fallback: return just the ansatz if initialize not available
            pass
        entangling_ansatz(qc, self.theta, self.depth)
        return qc
