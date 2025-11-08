
"""
Quantum Forecasting Benchmark (Resource-Matched, Price-Level)
- Amplitude Encoding + VQA (3q)
- Angle Encoding + VQA (3q, 6q) x {LinearCX, Chain-RyRz, Efficient-SU2}
- Quantum Reservoir Computing (QR2; 3q, 6q)  --> now predicts PCA(Y) with ridge (ky components), not scalar mean

Key changes vs earlier draft:
- Target defaults to "price" (levels), not returns.
- ky defaults to 6; inverse-PCA used to reconstruct full grid.
- kx is decoupled from n_qubits. Default kx=6 via kx_map.
- Amplitude encoder uses L2-normalization with mild gain to avoid amplitude collapse.
- SPSA iterations default 250 (tunable).
- Reservoir now fits ridge to ky-dimensional Yb (was scalar mean).
- Metrics & plots assume price-level targets; QLIKE is meaningful again.
"""

from __future__ import annotations
import os, math, warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
RNG = np.random.default_rng(7)

# =======================
# Data I/O
# =======================

def resolve_csv() -> str:
    cands = [
        "Dataset_Simulated_Price_swaption.csv",
        "./data/swaption_features.csv",
        "Dataset_Simulated_Price_swaption.xlsx",
        "sample_Simulated_Swaption_Price.csv",
        "Dataset_Simulated_Price_swaption.xlsx",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No CSV/XLSX found next to this script. Put your Dataset_*.csv or sample_*.csv here.")

def read_prices_grid(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    date_col = df.columns[-1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    prices = df[df.columns[:-1]].apply(pd.to_numeric, errors="coerce")
    sorter = np.argsort(dates.values)
    dates = dates.iloc[sorter].reset_index(drop=True)
    prices = prices.iloc[sorter].reset_index(drop=True)
    prices.columns = [str(c) for c in prices.columns]
    return prices, dates

def make_windows(prices: pd.DataFrame, dates: pd.Series, window: int = 6,
                 target: str = "price") -> Tuple[np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    P = prices.values.astype(float)
    T, M = P.shape
    R = np.diff(P, axis=0)
    X_list, y_list, d_list = [], [], []
    for t in range(window, T-1):
        X_list.append(P[t-window:t, :].reshape(-1))
        if target.lower() == "ret":
            y_list.append(R[t-1, :])
        else:
            y_list.append(P[t, :])
        d_list.append(dates.iloc[t])
    X = np.vstack(X_list); y = np.vstack(y_list)
    return X, y, list(prices.columns), d_list

def tvt_split(X, y, d, train_ratio=0.8, val_ratio=0.0):
    N = X.shape[0]
    n_tr = int(N*train_ratio); n_va = int(N*val_ratio)
    parts = {
        "train": (X[:n_tr], y[:n_tr], d[:n_tr]),
        "val":   (X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va], d[n_tr:n_tr+n_va]),
        "test":  (X[n_tr+n_va:], y[n_tr+n_va:], d[n_tr+n_va:])
    }
    return parts

# =======================
# Transforms (Standardize + PCA)
# =======================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@dataclass
class Transforms:
    Xs: StandardScaler
    Ys: StandardScaler
    pcaX: PCA
    pcaY: PCA

def fit_transforms(X_tr, Y_tr, kx: int, ky: int) -> Transforms:
    Xs = StandardScaler().fit(X_tr)
    Ys = StandardScaler().fit(Y_tr)
    Xb = Xs.transform(X_tr)
    Yb = Ys.transform(Y_tr)
    pcaX = PCA(n_components=kx, random_state=7).fit(Xb)
    pcaY = PCA(n_components=ky, random_state=7).fit(Yb)
    return Transforms(Xs, Ys, pcaX, pcaY)

def tx(X, Y, tr: Transforms):
    Xb = tr.pcaX.transform(tr.Xs.transform(X))
    Yb = tr.pcaY.transform(tr.Ys.transform(Y))
    return Xb, Yb

def invY(Yb, tr: Transforms):
    Yb_arr = np.asarray(Yb)
    if Yb_arr.ndim == 1:
        Yb_arr = Yb_arr.reshape(-1, 1)
    ncomp = tr.pcaY.n_components_
    k = Yb_arr.shape[1]
    if k < ncomp:
        pad = np.zeros((Yb_arr.shape[0], ncomp - k), dtype=Yb_arr.dtype)
        Yb_mat = np.concatenate([Yb_arr, pad], axis=1)
    elif k > ncomp:
        Yb_mat = Yb_arr[:, :ncomp]
    else:
        Yb_mat = Yb_arr
    return tr.Ys.inverse_transform(tr.pcaY.inverse_transform(Yb_mat))

# =======================
# Qiskit VQA blocks
# =======================
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    try:
        from qiskit.primitives import Estimator as _Estimator
        _EST_MODE = "V2"
    except Exception:
        try:
            from qiskit_aer.primitives import Estimator as _Estimator
            _EST_MODE = "V1"
        except Exception:
            _Estimator = None
            _EST_MODE = "SV"
except Exception:
    QuantumCircuit = None
    SparsePauliOp = None
    Statevector = None
    _Estimator = None
    _EST_MODE = "SV"

def encoder_angle(n_qubits: int, xvec: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    xv = np.asarray(xvec, float).ravel()
    xv = np.clip(xv, -3.0, 3.0)
    angles = (math.pi/6.0) * xv
    for q, a in enumerate(angles[:n_qubits]):
        qc.ry(a, q)
    return qc

def encoder_amplitude(n_qubits: int, xvec: np.ndarray, gain: float = 1.5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    m = 1 << n_qubits
    v = np.zeros(m, dtype=complex)
    x = np.asarray(xvec, float).ravel()
    L = min(len(x), m)
    if L > 0:
        v[:L] = x[:L]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v[:] = 0.0
        v[0] = 1.0
    else:
        v = (v / norm) * min(gain, 2.0)
        v = v / np.linalg.norm(v)
    qc.initialize(v)
    return qc

def ansatz_linear_cx(n_qubits: int, depth: int, theta: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    ptr = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.rx(theta[ptr], q); ptr += 1
            qc.rz(theta[ptr], q); ptr += 1
        for q in range(n_qubits-1):
            qc.cx(q, q+1)
    return qc

def ansatz_chain_ryrz(n_qubits: int, depth: int, theta: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    ptr = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(theta[ptr], q); ptr += 1
            qc.rz(theta[ptr], q); ptr += 1
        for q in range(n_qubits-1):
            qc.cx(q, q+1)
    return qc

def ansatz_efficient_su2(n_qubits: int, depth: int, theta: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    ptr = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.rx(theta[ptr], q); ptr += 1
            qc.rz(theta[ptr], q); ptr += 1
        for q in range(n_qubits):
            qc.cz(q, (q+1) % n_qubits)
    return qc

def observables_z(n_qubits: int, n_readout: int) -> List[SparsePauliOp]:
    obs = []
    for q in range(n_readout):
        pauli = ['I']*n_qubits
        pauli[n_qubits-1-q] = 'Z'
        obs.append(SparsePauliOp.from_list([(''.join(pauli), 1.0)]))
    return obs

class VQARegressor:
    def __init__(self, n_qubits: int, ky: int, depth: int,
                 encoder: Callable[[int, np.ndarray], QuantumCircuit],
                 ansatz: Callable[[int, int, np.ndarray], QuantumCircuit],
                 amp_gain: float = 1.5,
                 shots: Optional[int] = None, seed: int = 7):
        if ky > n_qubits:
            print(f"[warn] ky={{ky}} exceeds n_qubits={{n_qubits}}; clamping ky to {{n_qubits}}.")
            ky = n_qubits
        self.nq = n_qubits; self.ky = ky; self.depth = depth
        self.encoder_fn = encoder; self.ansatz = ansatz
        self.amp_gain = amp_gain
        self.shots = shots; self.seed = seed
        self.obs = observables_z(n_qubits, ky)
        self.theta = RNG.normal(0.0, 0.1, size=depth*(2*n_qubits))
        self.est = _Estimator() if _Estimator is not None else None

    def _compose(self, x, theta):
        if self.encoder_fn == encoder_amplitude:
            enc = self.encoder_fn(self.nq, x, gain=self.amp_gain)
        else:
            enc = self.encoder_fn(self.nq, x)
        anz = self.ansatz(self.nq, self.depth, theta)
        return enc.compose(anz)

    def _predict_one(self, x, theta) -> np.ndarray:
        qc = self._compose(x, theta)
        if self.est is None:
            sv = Statevector.from_instruction(qc)
            vals = [float(np.real(sv.expectation_value(o))) for o in self.obs]
        else:
            try:
                circuits = [qc] * len(self.obs)
                res = self.est.run(circuits=circuits, observables=self.obs).result()
                vals = np.array(res.values, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                try:
                    res = self.est.run([(qc, self.obs)]).result()
                    vals = np.array(res.values, dtype=float).reshape(-1)
                except Exception:
                    circuits = [qc] * len(self.obs)
                    vals = self.est.run(circuits, self.obs).result().values
        return np.array(vals, dtype=float)

    def predict_batch(self, Xb, theta=None) -> np.ndarray:
        th = self.theta if theta is None else theta
        return np.vstack([self._predict_one(x, th) for x in Xb])

    def spsa_fit(self, Xb_tr, Yb_tr, Xb_va=None, Yb_va=None,
                 maxiter=250, a=0.05, c=0.1, alpha=0.602, gamma=0.101,
                 batch_size=32) -> Dict[str, List[float]]:
        rng = np.random.default_rng(self.seed)
        theta = self.theta.copy()
        N = Xb_tr.shape[0]
        hist = {"train": [], "val": []}
        def mse(th, xb, yb):
            yhat = self.predict_batch(xb, th)
            return float(np.mean((yhat - yb)**2))
        for k in range(1, maxiter+1):
            ak = a / (k**alpha); ck = c / (k**gamma)
            s = rng.integers(0, max(1, N-batch_size))
            xb = Xb_tr[s:s+batch_size]; yb = Yb_tr[s:s+batch_size]
            delta = rng.choice([-1.0, 1.0], size=theta.shape)
            Lp = mse(theta + ck*delta, xb, yb)
            Lm = mse(theta - ck*delta, xb, yb)
            gk = (Lp - Lm) / (2*ck) * delta
            theta = theta - ak * gk
            hist["train"].append(mse(theta, xb, yb))
            if Xb_va is not None and Yb_va is not None and len(Yb_va)>0:
                hist["val"].append(mse(theta, Xb_va, Yb_va))
        self.theta = theta
        return hist

# =======================
# Quantum Reservoir (QR2) predicting ky-dim PCA(Y) with Ridge
# =======================
import scipy.linalg as la
from sklearn.linear_model import Ridge

def apply_ry_inplace(psi: np.ndarray, theta: float, q: int, n_total: int):
    c = np.cos(theta/2.0); s = np.sin(theta/2.0)
    step = 1 << q; period = step << 1
    for base in range(0, psi.size, period):
        for off in range(step):
            i0 = base + off; i1 = i0 + step
            a0 = psi[i0]; a1 = psi[i1]
            psi[i0] = c*a0 - s*a1
            psi[i1] = s*a0 + c*a1

def expect_Z_all(psi: np.ndarray, n_total: int):
    exps = np.zeros(n_total)
    N = psi.size
    for j in range(n_total):
        step = 1 << j; period = step << 1; ssum = 0.0
        for base in range(0, N, period):
            b0 = psi[base:base+step]
            b1 = psi[base+step:base+period]
            ssum += (np.abs(b0)**2).sum() - (np.abs(b1)**2).sum()
        exps[j] = ssum.real
    return exps

@dataclass
class QRCConfig:
    n_total: int
    n_input: int
    tau: float = 1.0
    field_v: float = 1.0
    seed: int = 42
    mode: str = "QR2"

class QuantumReservoir:
    _cache = {}
    def __init__(self, cfg: QRCConfig):
        self.cfg = cfg
        key = (cfg.n_total, cfg.tau, cfg.field_v, cfg.seed)
        if key in QuantumReservoir._cache:
            self.H, self.U_tau, self.U_tau2 = QuantumReservoir._cache[key]
        else:
            np.random.seed(cfg.seed)
            n = cfg.n_total; dim = 1 << n
            J = np.triu(np.random.rand(n, n), 1)
            H = np.zeros((dim, dim), dtype=complex)
            for i in range(n):
                H += cfg.field_v * _single(n, 'Z', i)
            for i in range(n):
                for j in range(i+1, n):
                    if J[i,j] != 0.0:
                        H += J[i,j] * _two(n, 'X', i, 'X', j)
            U_tau  = la.expm(-1j * cfg.tau    * H)
            U_tau2 = la.expm(-1j * (cfg.tau/2) * H)
            QuantumReservoir._cache[key] = (H, U_tau, U_tau2)
            self.H, self.U_tau, self.U_tau2 = H, U_tau, U_tau2
        self.ridge = None
        self.n_total = cfg.n_total
        self.n_input = cfg.n_input

    def _forward_features(self, x_seq):
        n = self.n_total; dim = 1 << n
        psi = np.zeros(dim, dtype=complex); psi[0] = 1.0
        for step in range(2):
            x = x_seq[2-step]
            for q, th in enumerate(x[:self.n_input]):
                apply_ry_inplace(psi, th, q, n)
            psi = self.U_tau @ psi
        x = x_seq[0]
        for q, th in enumerate(x[:self.n_input]):
            apply_ry_inplace(psi, th, q, n)
        psi_tau  = self.U_tau  @ psi
        feat_tau = expect_Z_all(psi_tau, n)
        psi_tau2  = self.U_tau2 @ psi
        feat_tau2 = expect_Z_all(psi_tau2, n)
        return np.concatenate([feat_tau, feat_tau2]).astype(float)

    def fit(self, X_seq_list, Yb_tr, ridge=1e-6):
        M = np.array([self._forward_features(seq) for seq in X_seq_list])
        self.ridge = Ridge(alpha=ridge, fit_intercept=True).fit(M, Yb_tr)

    def predict(self, X_seq_list):
        M = np.array([self._forward_features(seq) for seq in X_seq_list])
        return self.ridge.predict(M)

def _pauli(op):
    if op == 'I': return np.eye(2, dtype=complex)
    if op == 'X': return np.array([[0,1],[1,0]], dtype=complex)
    if op == 'Z': return np.array([[1,0],[0,-1]], dtype=complex)
    raise ValueError

def _kron(ops):
    M = ops[0]
    for op in ops[1:]:
        M = np.kron(M, op)
    return M

def _single(n, which, qubit):
    ops = [_pauli('I')]*n; ops[qubit] = _pauli(which)
    return _kron(ops)

def _two(n, a, i, b, j):
    ops = [_pauli('I')]*n; ops[i] = _pauli(a); ops[j] = _pauli(b)
    return _kron(ops)

# =======================
# Feature builders (shared)
# =======================
def to_minuspi_pi_scaler(X_fit: np.ndarray):
    xmin = X_fit.min(axis=0); xmax = X_fit.max(axis=0)
    span = np.where((xmax-xmin)<1e-12, 1.0, xmax-xmin)
    def f(X):
        Z = (X - xmin)/span; Z = np.clip(Z,0.0,1.0)
        return (Z*2.0 - 1.0) * np.pi
    return f

def build_lagged_sequences(df_vals: np.ndarray, scaler, idx_arr, n_lags=3, n_input=3):
    X_scaled = scaler(df_vals)
    seqs = []
    for t in idx_arr:
        if t < n_lags: continue
        seq = []
        for lag in range(1, n_lags+1):
            seq.append(X_scaled[t-lag, :n_input])
        seqs.append(seq)
    return seqs

# =======================
# Metrics / Stats
# =======================
def mse(a,b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[ok]-b[ok])**2))

def mae(a,b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[ok]-b[ok])))

def rmse(a,b):
    return float(np.sqrt(mse(a,b)))

def r2(a,b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum()<3: return float('nan')
    ssr = np.sum((a[ok]-b[ok])**2)
    sst = np.sum((a[ok]-a[ok].mean())**2)
    return float(1 - ssr/max(sst,1e-12))

def corr(a,b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum()<3: return float('nan')
    return float(np.corrcoef(a[ok],b[ok])[0,1])

def qlike_from_levels(y, yhat, eps=1e-12):
    y = np.maximum(np.asarray(y,float), eps)
    f = np.maximum(np.asarray(yhat,float), eps)
    r = y/f
    return float(np.mean(r - np.log(r) - 1.0))

from math import erf

def dm_test(loss1, loss2, L=None):
    d = np.asarray(loss1) - np.asarray(loss2)
    d = d[np.isfinite(d)]
    T = len(d)
    if T<5: return float('nan'), float('nan')
    dbar = d.mean();
    if L is None: L = int(np.floor(T**(1/3)))
    gamma0 = np.mean((d-dbar)*(d-dbar))
    s = gamma0
    for k in range(1, L+1):
        w = 1.0 - k/(L+1)
        cov = np.mean((d[k:]-dbar)*(d[:-k]-dbar))
        s += 2*w*cov
    se = np.sqrt(max(s,1e-12)/T)
    stat = dbar/(se+1e-12)
    p = 2*(1 - 0.5*(1+erf(abs(stat)/np.sqrt(2))))
    return float(stat), float(p)

# =======================
# Benchmark runner
# =======================
@dataclass
class ModelSpec:
    name: str
    kind: str  # 'VQA' or 'QRC'
    n_qubits: int
    encoder: Optional[str] = None  # 'angle' | 'amplitude'
    ansatz: Optional[str] = None   # 'linear_cx' | 'chain_ryrz' | 'efficient_su2'
    depth: int = 2
    amp_gain: float = 1.5

def run_benchmark(csv_path: Optional[str] = None, window: int = 5, target: str = "price",
                  train_ratio: float = 0.8, val_ratio: float = 0.0,
                  kx_map: Optional[Dict[int, int]] = None, ky: int = 6,
                  maxiter: int = 250, batch: int = 32,
                  qrc_lags: int = 3):
    kx_map = kx_map or {3:6, 6:6}
    path = csv_path or resolve_csv()
    prices, dates = read_prices_grid(path)
    X, Y, series_names, d_list = make_windows(prices, dates, window=window, target=target)
    parts = tvt_split(X, Y, d_list, train_ratio=train_ratio, val_ratio=val_ratio)
    X_tr, Y_tr, d_tr = parts['train']; X_va, Y_va, d_va = parts['val']; X_te, Y_te, d_te = parts['test']

    model_grid: List[ModelSpec] = [
        ModelSpec("Amplitude+VQA (3q)", kind='VQA', n_qubits=3, encoder='amplitude', ansatz='linear_cx', depth=2, amp_gain=1.5),
        ModelSpec("Angle+VQA (LinearCX, 3q)", kind='VQA', n_qubits=3, encoder='angle', ansatz='linear_cx', depth=2),
        ModelSpec("Angle+VQA (LinearCX, 6q)", kind='VQA', n_qubits=6, encoder='angle', ansatz='linear_cx', depth=2),
        ModelSpec("Angle+VQA (Chain-RyRz, 3q)", kind='VQA', n_qubits=3, encoder='angle', ansatz='chain_ryrz', depth=2),
        ModelSpec("Angle+VQA (Chain-RyRz, 6q)", kind='VQA', n_qubits=6, encoder='angle', ansatz='chain_ryrz', depth=2),
        ModelSpec("Angle+VQA (Efficient-SU2, 3q)", kind='VQA', n_qubits=3, encoder='angle', ansatz='efficient_su2', depth=2),
        ModelSpec("Angle+VQA (Efficient-SU2, 6q)", kind='VQA', n_qubits=6, encoder='angle', ansatz='efficient_su2', depth=2),
        ModelSpec("QRC (QR2, 3q)", kind='QRC', n_qubits=3),
        ModelSpec("QRC (QR2, 6q)", kind='QRC', n_qubits=6),
    ]

    preds: Dict[str, np.ndarray] = {}
    results_rows = []

    for spec in model_grid:
        print(f"[RUN] {spec.name}")
        kx = kx_map.get(spec.n_qubits, 6)
        tr = fit_transforms(X_tr, Y_tr, kx=kx, ky=min(ky, spec.n_qubits) if spec.kind=='VQA' else ky)
        Xb_tr, Yb_tr = tx(X_tr, Y_tr, tr)
        Xb_te, Yb_te = tx(X_te, Y_te, tr)

        if spec.kind == 'VQA':
            encoder = encoder_amplitude if spec.encoder=='amplitude' else encoder_angle
            if spec.ansatz=='linear_cx':
                anz = ansatz_linear_cx
            elif spec.ansatz=='chain_ryrz':
                anz = ansatz_chain_ryrz
            else:
                anz = ansatz_efficient_su2
            ky_eff = min(ky, spec.n_qubits)
            vqa = VQARegressor(spec.n_qubits, ky=ky_eff, depth=spec.depth,
                               encoder=encoder, ansatz=anz, amp_gain=spec.amp_gain, seed=7)
            vqa.spsa_fit(Xb_tr, Yb_tr, Xb_va=None, Yb_va=None, maxiter=maxiter, batch_size=batch)
            Z_tr = vqa.predict_batch(Xb_tr)
            from sklearn.linear_model import Ridge
            cal  = Ridge(alpha=1e-6, fit_intercept=True).fit(Z_tr, Yb_tr[:, :ky_eff])
            Yb_hat_te_partial = cal.predict(vqa.predict_batch(Xb_te))
            if ky_eff < ky:
                pad = np.zeros((Yb_hat_te_partial.shape[0], ky-ky_eff))
                Yb_hat_te = np.hstack([Yb_hat_te_partial, pad])
            else:
                Yb_hat_te = Yb_hat_te_partial
            Y_hat_te  = invY(Yb_hat_te, tr)
            preds[spec.name] = Y_hat_te
        else:
            scaler = to_minuspi_pi_scaler(Xb_tr)
            idx_tr = np.arange(qrc_lags, Xb_tr.shape[0])
            idx_te = np.arange(qrc_lags, Xb_te.shape[0])
            seq_tr = build_lagged_sequences(Xb_tr, scaler, idx_tr, n_lags=qrc_lags, n_input=min(3, spec.n_qubits))
            Yb_tr_cut = Yb_tr[qrc_lags:qrc_lags+len(seq_tr), :ky]
            cfg = QRCConfig(n_total=spec.n_qubits, n_input=min(3, spec.n_qubits), tau=1.0, field_v=1.0, seed=42, mode='QR2')
            qrc = QuantumReservoir(cfg); qrc.fit(seq_tr, Yb_tr_cut, ridge=1e-6)
            seq_te = build_lagged_sequences(Xb_te, scaler, idx_te, n_lags=qrc_lags, n_input=min(3, spec.n_qubits))
            Yb_hat_te = qrc.predict(seq_te)
            n = min(Yb_hat_te.shape[0], Yb_te.shape[0])
            Yb_hat_te = Yb_hat_te[-n:, :]
            Y_hat_te = invY(Yb_hat_te, tr)
            if Y_hat_te.shape[0] != Y_te.shape[0]:
                pad_front = Y_te.shape[0] - Y_hat_te.shape[0]
                if pad_front > 0:
                    pad = np.tile(Y_hat_te[:1,:], (pad_front,1))
                    Y_hat_te = np.vstack([pad, Y_hat_te])
            preds[spec.name] = Y_hat_te

        y_true = Y_te
        y_pred = preds[spec.name]
        n = min(y_true.shape[0], y_pred.shape[0])
        y_true = y_true[-n:, :]; y_pred = y_pred[-n:, :]
        preds[spec.name] = y_pred

        m = dict(
            Model=spec.name,
            MSE = float(np.mean((y_true - y_pred)**2)),
            RMSE= float(np.sqrt(np.mean((y_true - y_pred)**2))),
            MAE = float(np.mean(np.abs(y_true - y_pred))),
            Bias= float(np.mean(y_true - y_pred)),
            Corr= float(np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]),
            R2  = float(1 - np.sum((y_true - y_pred)**2)/max(np.sum((y_true - y_true.mean())**2),1e-12)),
            QLIKE = qlike_from_levels(y_true, y_pred)
        )
        results_rows.append(m)

    metrics_df = pd.DataFrame(results_rows).sort_values("MSE").reset_index(drop=True)
    return metrics_df, preds, d_te, Y_te, series_names

# =======================
# Plot helpers
# =======================
def plot_overlay(dates, Y_true, pred_dict, series_names, limit_models: Optional[List[str]]=None, series_idx:int=0):
    sel = limit_models or list(pred_dict.keys())
    plt.figure(figsize=(12,5))
    plt.plot(dates, Y_true[:,series_idx], label=f'Actual ({{series_names[series_idx]}})', linewidth=1)
    for name in sel:
        plt.plot(dates, pred_dict[name][:,series_idx], label=name, linewidth=1)
    plt.title(f"Actual vs Predictions (series {{series_names[series_idx]}})"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_rolling_rmse(dates, Y_true, pred_dict, w=30, base_keys: Optional[List[str]]=None):
    plt.figure(figsize=(12,3))
    keys = base_keys or list(pred_dict.keys())[:4]
    for name in keys:
        rm = pd.Series(((Y_true - pred_dict[name])**2).mean(axis=1), index=dates).rolling(w).mean().pow(0.5)
        plt.plot(rm.index, rm.values, label=f"{{name}} RMSE (win={{w}})", linewidth=1)
    plt.title("Rolling RMSE (OOS)"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_parity(Y_true, Y_pred, title):
    plt.figure(figsize=(5,5))
    y = Y_true.flatten(); p = Y_pred.flatten()
    plt.scatter(y, p, s=8, alpha=0.6)
    lo, hi = np.nanpercentile(np.r_[y,p], [1,99])
    plt.plot([lo,hi],[lo,hi], linestyle='--')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(title); plt.tight_layout(); plt.show()

# =======================
# Main (CLI)
# =======================
if __name__ == "__main__":
    metrics, preds, d_te, Y_te, series_names = run_benchmark(
        train_ratio=0.8, val_ratio=0.0, maxiter=250, batch=32, target="price",
        kx_map={3:6, 6:6}, ky=6
    )
    print("\n=== METRICS (test) ===\n", metrics.to_string(index=False))

    first = list(preds.keys())[0]
    plot_overlay(d_te, Y_te, preds, series_names, limit_models=[first], series_idx=0)
    plot_parity(Y_te, preds[first], f"Parity: {{first}}")
