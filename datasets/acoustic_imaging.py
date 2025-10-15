"""
Unified acoustic imaging module for sparse source localization

Provides a single-file, dependency-light API for:
- Config loading (YAML with ${PROJECT_ROOT} expansion)
- Grid definition and flatten/index helpers
- Array geometry loading (.mat/.npy/.csv)
- Vectorized steering matrices (W, D) over full grid
- Implicit linear operator y = A x without materializing A
- Dataset for generating sparse x, blurred target, and y
- Dataloader builder returning dict or specific field tensors

Key relations (flattened vector view):
    y = A x,  where A = | W D^H |^2 / C^2

===============================
Dataset Return Format
===============================
Lazy Evaluation (Recommended):
   - Only compute and return keys specified in 'need' (set/list); unlisted keys are neither returned nor computed.
   - Default need={"source_map","target_map"}; can append 'observation_map' (implicit operator), 'psf_center_map' (not normalized), 'A_matrix' (explicit A, lazy cached), and metadata keys:
     {
       "source_map":             Tensor (1, H, W)  — Sparse source strength map (original x)
       "target_map":             Tensor (1, H, W)  — Blurred target map (original target)
       "observation_map":        Tensor (1, H, W)  — Observation map y = A x (implicit operator)
       "num_sources":           LongTensor ()      — Number of sources (original nos)
       "source_indices_xy":     LongTensor (max_nos, 2)   — Source grid indices (x_idx, y_idx)
       "source_positions_xy":   FloatTensor (max_nos, 2)  — Source physical coordinates (x, y) in meters
       "source_energies":       FloatTensor (max_nos,)    — Source energies
       "source_mask":           BoolTensor (max_nos,)     — Valid source mask (first num_sources are True)
       "analysis_frequency_hz": FloatTensor ()            — Actual analysis frequency (Hz)
       "source_frequency_hz":   FloatTensor ()            — Configured source frequency (Hz)
       "psf_center_map":        Tensor (1, H, W)          — PSF with observation point at (0,0,z), computed on demand
       "A_matrix":              Tensor (N, N)              — Explicit A matrix (on demand: disk cache .pt + memory cache)
     }

   - max_nos is taken from the configured source count upper limit (max of source_set.nos), ensuring stable batch stacking.
   - Image tensors are all float32; normalization strategy: if normalization is enabled, strictly use observation_map's max as denominator to uniformly normalize source/target/observation; PSF is not normalized.

3) Grid and Vectorization Convention:
   - H = x_grid, W = y_grid, flattening order follows idx = x_idx * y_grid + y_idx.
   - N = H * W is the flattened length, corresponding to A's size (N, N) (this implementation uses implicit operators to avoid explicit construction).

Usage Example (Minimal):
    from datasets.acoustic_imaging import build_dataloader
    dl = build_dataloader(cfg_path="configs/dataset_1.yaml",
                          batch_size=32,
                          need={"source_map", "target_map", "observation_map"},
                          dataset_device="cuda")
    ds = dl.dataset  # Optional: set random seed for reproducibility
    ds.set_seed(42)
    for batch in dl:
        ...

This module avoids cross-package imports and is self-contained.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal, Union, Set
from abc import ABC, abstractmethod
import random

import numpy as np
import torch


# --- Optional SciPy import for .mat files ---
try:
    import scipy.io as sio  # type: ignore
except Exception:  # pragma: no cover - provide runtime error when needed
    sio = None  # noqa: N816


# ===========================
# Config loading and helpers
# ===========================

def _expand_project_root_in_value(value: Any, project_root: str) -> Any:
    if isinstance(value, str):
        return value.replace("${PROJECT_ROOT}", project_root)
    if isinstance(value, list):
        return [_expand_project_root_in_value(v, project_root) for v in value]
    if isinstance(value, dict):
        return {k: _expand_project_root_in_value(v, project_root) for k, v in value.items()}
    return value


def load_yaml_config(path: str, validate_schema: bool = True) -> Dict[str, Any]:
    """Load YAML config with optional schema validation.
    
    Args:
        path: Path to YAML file
        validate_schema: If True, validate dataset config schema (scan_set, source_set, etc.)
                        Set to False for non-dataset configs like visualization.yaml
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML not installed. Please: pip install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # configs typically lives under <project_root>/configs/<file>.yaml
    # project_root should be the parent of the directory containing the yaml
    project_root = os.path.abspath(os.path.join(os.path.dirname(path), os.pardir))
    cfg = _expand_project_root_in_value(cfg, project_root)
    if validate_schema:
        _validate_config_schema(cfg, source_path=path)
    return cfg


def _validate_config_schema(cfg: Dict[str, Any], *, source_path: str = "<dict>") -> None:
    """Minimal YAML schema validation with clear messages."""
    path = source_path
    def _ensure(d: Dict[str, Any], k: str, typ: type):
        if k not in d:
            raise ValueError(f"Config missing key '{k}' in {path}")
        if not isinstance(d[k], typ):
            raise TypeError(f"Config key '{k}' expected {typ.__name__}, got {type(d[k]).__name__} in {path}")

    _ensure(cfg, 'scan_set', dict)
    _ensure(cfg, 'source_set', dict)
    _ensure(cfg, 'target_map_set', dict)

    ss = cfg['scan_set']
    for k, t in [('array_id', int), ('array_base_dir', str), ('fs', int), ('nfft', int), ('c', (int, float)),
                 ('x_range', list), ('y_range', list), ('z_scan', (int, float)), ('x_grid', int), ('y_grid', int)]:
        if k not in ss:
            raise ValueError(f"scan_set missing '{k}' in {path}")
    if len(ss['x_range']) != 2 or len(ss['y_range']) != 2:
        raise ValueError(f"x_range/y_range must be [min,max] in {path}")

    src = cfg['source_set']
    for k in ['freq', 'num_samples', 'nos', 'min_source_distance', 'minEnergy']:
        if k not in src:
            raise ValueError(f"source_set missing '{k}' in {path}")
    if not isinstance(src['freq'], list) or not src['freq']:
        raise ValueError(f"source_set.freq must be non-empty list in {path}")
    if not isinstance(src['nos'], list) or not src['nos']:
        raise ValueError(f"source_set.nos must be non-empty list in {path}")

    tm = cfg['target_map_set']
    for k in ['blur', 'lowk']:
        if k not in tm:
            raise ValueError(f"target_map_set missing '{k}' in {path}")


def fbin2freq(bin_index: int, fs: int, nfft: int) -> float:
    return round((bin_index * fs) / nfft, 1)


def freq2fbin(freq: float, fs: int, nfft: int) -> int:
    return round(freq * nfft / fs)


# ===========================
# Grid
# ===========================

@dataclass
class ScanGrid:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z: float
    x_grid: int
    y_grid: int

    def build_points(self, device: torch.device) -> torch.Tensor:
        x = torch.linspace(self.x_range[0], self.x_range[1], self.x_grid, device=device)
        y = torch.linspace(self.y_range[0], self.y_range[1], self.y_grid, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        P = torch.zeros((self.x_grid * self.y_grid, 3), device=device)
        P[:, 0] = X.reshape(-1)
        P[:, 1] = Y.reshape(-1)
        P[:, 2] = float(self.z)
        return P

    def index_of(self, x: float, y: float) -> int:
        xs = (self.x_range[1] - self.x_range[0]) / (self.x_grid - 1)
        ys = (self.y_range[1] - self.y_range[0]) / (self.y_grid - 1)
        xi = max(0, min(round((x - self.x_range[0]) / xs), self.x_grid - 1))
        yi = max(0, min(round((y - self.y_range[0]) / ys), self.y_grid - 1))
        return int(xi * self.y_grid + yi)


# ===========================
# Array geometry
# ===========================

class MicrophoneArray:
    def __init__(self, array_id: int, base_dir: str):
        self.array_id = int(array_id)
        self.base_dir = base_dir

    def _mat_first(self, path: str) -> np.ndarray:
        if sio is None:
            raise RuntimeError("SciPy is required for .mat files. pip install scipy")
        data = sio.loadmat(path)
        ks = [k for k in data.keys() if not k.startswith("__")]
        if not ks:
            raise ValueError(f"no variables in {path}")
        arr = data[ks[0]]
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return arr.astype(np.float32)

    def load(self) -> np.ndarray:
        # Build candidate search roots for array files
        candidates: List[str] = []
        # 1) configured base_dir
        candidates.append(self.base_dir)
        # 2) environment override
        env_dir = os.environ.get("ARRAYBOX_DIR")
        if env_dir:
            candidates.append(env_dir)
        # 3) common fallbacks
        candidates.extend([
            os.path.join(os.path.expanduser("~"), "Record_Dataset"),
            os.path.join(os.path.dirname(__file__), "dataset", "arraybox"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "arraybox"),
        ])

        def _first_existing(*relpaths: str) -> Optional[str]:
            for root in candidates:
                for rp in relpaths:
                    p = os.path.join(root, rp)
                    if os.path.isfile(p):
                        return p
            return None

        if self.array_id == 1:
            pass
        if self.array_id == 2:
            p = _first_existing("MIRACLE.npy")
            if p is None:
                raise FileNotFoundError("MIRACLE.npy not found in arraybox search paths")
            return np.load(p).astype(np.float32)
        raise ValueError(f"unsupported array_id={self.array_id}")


# ===========================
# Vectorized steering and operators
# ===========================

class SteeringOperators:
    @staticmethod
    def build_ws_ds(grid_points: torch.Tensor, mic_xyz: torch.Tensor, freq: float, c: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized steering matrices over all grid points and microphones.

        Returns:
            ws: (N,C) complex64
            ds: (N,C) complex64
        """
        device = grid_points.device
        gp = grid_points  # (N,3)
        mic = mic_xyz  # (C,3)
        # distances: (N,C)
        diff = mic.unsqueeze(0) - gp.unsqueeze(1)
        distances = torch.linalg.norm(diff, dim=2)
        tau = distances / float(c)
        phase = torch.exp(-2j * math.pi * float(freq) * tau)

        ref = torch.linalg.norm(gp, dim=1).clamp_min(1e-12)  # (N,)
        # dvecs: ref/dist * phase
        dvecs = ref.unsqueeze(1) * (1.0 / distances.clamp_min(1e-12)) * phase
        # wvecs: dist/ref * phase
        wvecs = distances * (1.0 / ref.unsqueeze(1)) * phase
        return wvecs.to(torch.complex64), dvecs.to(torch.complex64)

    @staticmethod
    def apply_Ax_implicit(W: torch.Tensor,
                          D: torch.Tensor,
                          x_image: torch.Tensor,
                          *,
                          x_grid: int,
                          y_grid: int,
                          sparse_threshold: int = 2048) -> torch.Tensor:
        """Compute y = A x without materializing A.

        A_{p,q} = | (W D^H)_{p,q} |^2 / C^2,  with C = num_mics.
        If x has limited non-zero entries (<= sparse_threshold), we accumulate columns.

        Args:
            W: (N,C) complex64
            D: (N,C) complex64
            x_image: (1,H,W) or (H,W) float32
            x_grid, y_grid: spatial grid sizes
            sparse_threshold: switch between sparse and dense path

        Returns:
            y_image: (1,H,W) float32
        """
        C = W.shape[1]
        N = x_grid * y_grid
        if x_image.dim() == 3:
            x_vec = x_image.reshape(-1)
        else:
            x_vec = x_image.reshape(-1)

        # identify non-zeros (treat >0 as active sources)
        nnz_idx = torch.nonzero(x_vec > 0, as_tuple=False).flatten()

        if nnz_idx.numel() == 0:
            return torch.zeros((1, x_grid, y_grid), dtype=torch.float32, device=W.device)

        # sparse path: accumulate selected columns
        if nnz_idx.numel() <= sparse_threshold and nnz_idx.numel() < N:
            y_acc = torch.zeros((W.shape[0],), dtype=torch.float32, device=W.device)
            D_conj_T = D.conj()  # (N,C)
            for q in nnz_idx.tolist():
                d_q_conj = D_conj_T[q]  # (C,)
                z = W @ d_q_conj  # (N,)
                a_col = (z.real * z.real + z.imag * z.imag) / (C ** 2)
                y_acc = y_acc + a_col * x_vec[q]
            return y_acc.reshape(1, x_grid, y_grid)

        # dense fallback: compute in manageable chunks to bound memory
        # We split columns of D into blocks and accumulate.
        block = max(1, 4096 // max(1, C // 8))  # heuristic
        y_acc = torch.zeros((W.shape[0],), dtype=torch.float32, device=W.device)
        D_conj_T = D.conj()  # (N,C)
        for start in range(0, N, block):
            end = min(N, start + block)
            cols = D_conj_T[start:end]  # (B,C)
            Z = W @ cols.T  # (N,B)
            A_cols = (Z.real * Z.real + Z.imag * Z.imag) / (C ** 2)
            y_acc = y_acc + (A_cols @ x_vec[start:end])
        return y_acc.reshape(1, x_grid, y_grid)

    @staticmethod
    def analytic_noise_map(W: torch.Tensor, x_grid: int, y_grid: int) -> torch.Tensor:
        C = W.shape[1]
        eye = torch.eye(C, dtype=torch.complex64, device=W.device)
        val = torch.real(torch.einsum("ij,jk,ik->i", W, eye, W.conj())) / C
        return val.reshape(x_grid, y_grid)

    @staticmethod
    def steering_at_point(point_xyz: torch.Tensor, freq: float, c: float,
                          mic_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (w, d) steering vectors for an arbitrary point (x,y,z).

        point_xyz: (3,) on same device as mic_xyz
        w, d: (C,) complex64
        """
        distances = torch.linalg.norm(mic_xyz - point_xyz, dim=1)
        tau = distances / float(c)
        phase = torch.exp(-2j * math.pi * float(freq) * tau)
        ref = torch.linalg.norm(point_xyz).clamp_min(1e-12)
        dvec = ref * (1.0 / distances.clamp_min(1e-12)) * phase
        wvec = distances * (1.0 / ref) * phase
        return wvec.to(torch.complex64), dvec.to(torch.complex64)


# ===========================
# Math operator units (structured)
# ===========================

class Operator(ABC):
    @abstractmethod
    def to(self, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def clear_cache(self):
        raise NotImplementedError


class BlurOperator(Operator):
    """Exponential radial blur on a grid.

    Kernel: mask = 10^(-lowk * r), r = sqrt(dx^2 + dy^2)
    """

    def __init__(self, x_grid: int, y_grid: int, step_x: float, step_y: float, lowk: float, device: torch.device):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.step_x = float(step_x)
        self.step_y = float(step_y)
        self.lowk = float(lowk)
        self.device = device
        self._xx = torch.arange(x_grid, device=device, dtype=torch.float32).unsqueeze(1).expand(x_grid, y_grid)
        self._yy = torch.arange(y_grid, device=device, dtype=torch.float32).unsqueeze(0).expand(x_grid, y_grid)

    def apply(self, indices_x: torch.Tensor, indices_y: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
        if energies.numel() == 0:
            return torch.zeros(self.x_grid, self.y_grid, dtype=torch.float32, device=self.device)
        dx = (self._xx.unsqueeze(0) - indices_x.view(-1, 1, 1)) * self.step_x
        dy = (self._yy.unsqueeze(0) - indices_y.view(-1, 1, 1)) * self.step_y
        dist = torch.sqrt(dx * dx + dy * dy)
        mask = torch.pow(10.0, -self.lowk * dist)
        return torch.sum(energies.view(-1, 1, 1) * mask, dim=0)

    def to(self, device: torch.device):
        if device == self.device:
            return self
        self.device = device
        self._xx = self._xx.to(device)
        self._yy = self._yy.to(device)
        return self

    def clear_cache(self):
        # BlurOperator has no heavy cache
        return None


class LinearForwardOperator(Operator):
    """Implicit linear forward y = A x, avoiding explicit A materialization.

    A = |W D^H|^2 / C^2, with W,D from steering.
    """

    def __init__(self, grid_points: torch.Tensor, mic_xyz: torch.Tensor, c: float, device: torch.device):
        self.grid_points = grid_points
        self.mic_xyz = mic_xyz
        self.c = float(c)
        self.device = device
        self._W_cache: Dict[float, torch.Tensor] = {}
        self._D_cache: Dict[float, torch.Tensor] = {}

    def ensure_freq(self, freq: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if freq in self._W_cache and freq in self._D_cache:
            return self._W_cache[freq], self._D_cache[freq]
        W, D = SteeringOperators.build_ws_ds(self.grid_points, self.mic_xyz, freq, self.c)
        self._W_cache[freq] = W
        self._D_cache[freq] = D
        return W, D

    def apply(self, freq: float, x_image: torch.Tensor, x_grid: int, y_grid: int) -> torch.Tensor:
        W, D = self.ensure_freq(freq)
        return SteeringOperators.apply_Ax_implicit(W, D, x_image, x_grid=x_grid, y_grid=y_grid)

    def psf_at_point(self, freq: float, point_xyz: torch.Tensor, x_grid: int, y_grid: int) -> torch.Tensor:
        """Compute PSF for an arbitrary observation point (x,y,z), shape (1,H,W).

        Uses W(point) and D(all) without materializing A.
        """
        # Ensure D is computed at freq
        W_all, D = self.ensure_freq(freq)
        # Build w for the arbitrary point
        w_point, _ = SteeringOperators.steering_at_point(point_xyz.to(self.mic_xyz.device), freq, self.c, self.mic_xyz)
        # Z = D conj() @ w (C,) => (N,)
        Z = D.conj() @ w_point
        C = self.mic_xyz.shape[0]
        psf = (Z.real * Z.real + Z.imag * Z.imag) / (C ** 2)
        return psf.reshape(1, x_grid, y_grid).to(W_all.device)

    def get_A_cached(self,
                     freq: float,
                     *,
                     cache_dir: str,
                     array_id: int,
                     fs: int,
                     nfft: int,
                     c: float,
                     x_grid: int,
                     y_grid: int,
                     x_range,
                     y_range,
                     z_scan: float,
                     device: Optional[torch.device] = None,
                     print_flag: bool = True) -> torch.Tensor:
        """Build/load full A matrix and cache to disk (.pt on CPU). Returns Tensor on 'device'.

        Path key includes array/physics/grid/frequency to avoid recomputation.
        """
        # ensure freq-aligned W/D
        W, D = self.ensure_freq(freq)
        act_freq = freq

        # deterministic cache path
        path = os.path.join(
            cache_dir,
            f'array_{array_id}',
            'A_matrix',
            f'c_{float(c)}',
            f'x_grid_{x_grid}',
            f'y_grid_{y_grid}',
            f'x_range_{float(x_range[0])}_{float(x_range[1])}',
            f'y_range_{float(y_range[0])}_{float(y_range[1])}',
            f'z_scan_{float(z_scan)}',
            f'fs_{fs}',
            f'nfft_{nfft}',
            f'freq_{act_freq:.1f}',
        )
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join(path, 'A.pt')

        target_device = device or W.device
        try:
            A = torch.load(fpath, map_location=target_device)
            if print_flag:
                print(f"Loaded A from cache: {fpath}")
            return A
        except Exception:
            if print_flag:
                print("Cache miss, building A ...")

        # build on current device
        C = W.shape[1]
        A = W @ D.conj().T
        A = (A.real * A.real + A.imag * A.imag) / (C ** 2)
        # save CPU copy
        torch.save(A.cpu(), fpath)
        if print_flag:
            print(f"Saved A to {fpath}, shape={tuple(A.shape)}")
        return A.to(target_device)

    def to(self, device: torch.device):
        if device == self.device:
            return self
        self.device = device
        # move cached tensors to new device lazily when accessed
        for f in list(self._W_cache.keys()):
            self._W_cache[f] = self._W_cache[f].to(device)
        for f in list(self._D_cache.keys()):
            self._D_cache[f] = self._D_cache[f].to(device)
        return self

    def clear_cache(self):
        self._W_cache.clear()
        self._D_cache.clear()


# ===========================
# Dataset
# ===========================

class SampleDict(TypedDict):
    source_map: torch.Tensor
    target_map: torch.Tensor
    observation_map: torch.Tensor
    is_observation_computed: torch.Tensor
    num_sources: torch.Tensor
    source_indices_xy: torch.Tensor
    source_positions_xy: torch.Tensor
    source_energies: torch.Tensor
    source_mask: torch.Tensor
    analysis_frequency_hz: torch.Tensor
    source_frequency_hz: torch.Tensor
    psf_center_map: torch.Tensor
    # Non-image metadata (returned on demand)
    # grid_hw: (H, W) for runner to reshape vectors back to images
    # A_matrix: (N, N) included if requested in 'need'; not a mandatory key in TypedDict


class LinearImagingDataset(torch.utils.data.Dataset):
    """Dataset generating sparse x, blurred target, and optional y via implicit operator.

    Return format (dict):
        {
          "x": (1,H,W) float32,
          "target": (1,H,W) float32,   # blurred
          "y": (1,H,W) float32,        # y = A x
          "nos": int,
          "analysis_freq": float,
          "source_freq": float,
          "indices": LongTensor[2,nos],
          "positions": FloatTensor[2,nos],
          "energies": FloatTensor[nos],
        }
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 device: torch.device,
                 cache_dir: Optional[str] = None,
                 *,
                 need: Optional[Union[Set[str], List[str]]] = None):
        super().__init__()
        self.device = device
        self.cfg = cfg
        # need keys (lazy computation); default minimal set
        default_need = {"source_map", "target_map", "num_sources", "source_indices_xy", "source_positions_xy", "source_energies", "source_mask", "analysis_frequency_hz", "source_frequency_hz"}
        self.need: Set[str] = set(need) if need is not None else default_need

        ss = cfg["scan_set"]
        self.grid = ScanGrid(
            x_range=tuple(ss["x_range"]),
            y_range=tuple(ss["y_range"]),
            z=float(ss["z_scan"]),
            x_grid=int(ss["x_grid"]),
            y_grid=int(ss["y_grid"]),
        )
        self.grid_points = self.grid.build_points(device)
        self.x_grid = int(ss["x_grid"])  # width-like
        self.y_grid = int(ss["y_grid"])  # height-like
        self.N = self.x_grid * self.y_grid
        self.fs = int(ss["fs"])
        self.nfft = int(ss["nfft"])
        self.c = float(ss["c"])

        arr = MicrophoneArray(int(ss["array_id"]), str(ss["array_base_dir"]))
        self.mic_xyz = torch.tensor(arr.load(), dtype=torch.float32, device=device)
        self.num_mics = int(self.mic_xyz.shape[0])

        src = cfg["source_set"]
        self.freq_list: List[float] = [float(f) for f in src["freq"]]
        self.nos_list: List[int] = [int(v) for v in src["nos"]]
        self.max_nos: int = int(max(self.nos_list)) if len(self.nos_list) > 0 else 0
        self.num_samples = int(src["num_samples"])
        self.min_energy = float(src["minEnergy"])
        self.min_dist = float(src["min_source_distance"])  # in grid index units

        tm = cfg["target_map_set"]
        self.blur = bool(tm["blur"])
        self.lowk = float(tm["lowk"])  # exponential decay coefficient

        self.norm_range = cfg.get("norm_range", [0, 1])
        self.noise_level = cfg.get("noise_level", [10, 30])
        self.cache_dir = cache_dir

        # operators
        xs = (self.grid.x_range[1] - self.grid.x_range[0]) / (self.x_grid - 1)
        ys = (self.grid.y_range[1] - self.grid.y_range[0]) / (self.y_grid - 1)
        self.blur_op = BlurOperator(self.x_grid, self.y_grid, xs, ys, self.lowk, self.device)
        self.forward_op = LinearForwardOperator(self.grid_points, self.mic_xyz, self.c, self.device)
        self._A_cache_mem: Dict[float, torch.Tensor] = {}

        # effective analysis frequencies (rounded to FFT bin)
        self.act_freqs: List[float] = []
        for f in self.freq_list:
            act = fbin2freq(freq2fbin(f, self.fs, self.nfft), self.fs, self.nfft)
            self.act_freqs.append(act)

        # Optional: prebuild A cache for all freqs if config enables it
        if bool(self.cfg.get('prebuild_A', False)):
            ss = self.cfg["scan_set"]
            for f in self.act_freqs:
                try:
                    A = self.forward_op.get_A_cached(
                        f,
                        cache_dir=self.cache_dir or os.path.join(os.path.dirname(__file__), "..", "cache_spDataset"),
                        array_id=int(ss["array_id"]),
                        fs=int(ss["fs"]),
                        nfft=int(ss["nfft"]),
                        c=float(ss["c"]),
                        x_grid=self.x_grid,
                        y_grid=self.y_grid,
                        x_range=ss["x_range"],
                        y_range=ss["y_range"],
                        z_scan=float(ss["z_scan"]),
                        device=self.device,
                        print_flag=True,
                    )
                    self._A_cache_mem[f] = A
                except Exception as e:
                    print(f"[A cache] Failed to build/load A for freq={f}: {e}")

        # noise map cache per freq (optional)
        self.noise_map_cache: Dict[float, torch.Tensor] = {}
        if "observation_map" in self.need or "psf_center_map" in self.need:
            for f in self.act_freqs:
                W, _ = self.forward_op.ensure_freq(f)
                self.noise_map_cache[f] = SteeringOperators.analytic_noise_map(W, self.x_grid, self.y_grid)

        # RNG (do not require explicit set_seed)
        self.rng: np.random.Generator = np.random.default_rng(int(time.time() * 1e6) % (2 ** 32))

    def __len__(self) -> int:
        return self.num_samples

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    # --------------------
    # sampling helpers
    # --------------------
    def _index_to_pos(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        xs = (self.grid.x_range[1] - self.grid.x_range[0]) / (self.x_grid - 1)
        ys = (self.grid.y_range[1] - self.grid.y_range[0]) / (self.y_grid - 1)
        x = self.grid.x_range[0] + x_idx * xs
        y = self.grid.y_range[0] + y_idx * ys
        return float(x), float(y)

    def _rand_sources(self, nos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.zeros((2, nos), dtype=np.int64)
        eng = np.zeros((nos,), dtype=np.float32)
        img = np.zeros((self.x_grid, self.y_grid), dtype=np.float32)
        chosen: List[Tuple[int, int]] = []
        for i in range(nos):
            while True:
                xi = int(self.rng.integers(0, self.x_grid))
                yi = int(self.rng.integers(0, self.y_grid))
                ok = True
                for px, py in chosen:
                    if abs(xi - px) < self.min_dist and abs(yi - py) < self.min_dist:
                        ok = False
                        break
                if ok:
                    chosen.append((xi, yi))
                    val = float(self.rng.uniform(self.min_energy, 1.0))
                    idx[0, i] = xi
                    idx[1, i] = yi
                    eng[i] = val
                    img[xi, yi] = val
                    break
        pos = np.zeros((2, nos), dtype=np.float32)
        for i in range(nos):
            pos[0, i], pos[1, i] = self._index_to_pos(int(idx[0, i]), int(idx[1, i]))
        return idx, eng, pos, img

    # --------------------
    # main sampling
    # --------------------
    def __getitem__(self, i: int) -> SampleDict | torch.Tensor:
        act_f = float(self.rng.choice(self.act_freqs))
        src_f = self.freq_list[self.act_freqs.index(act_f)]
        nos = int(self.rng.choice(self.nos_list))
        idx, eng, pos, img = self._rand_sources(nos)

        xi = torch.tensor(idx[0], dtype=torch.long, device=self.device)
        yi = torch.tensor(idx[1], dtype=torch.long, device=self.device)
        energies = torch.tensor(eng, dtype=torch.float32, device=self.device)

        # x (sparse image)
        x_img = torch.zeros(self.x_grid, self.y_grid, dtype=torch.float32, device=self.device)
        if energies.numel() > 0:
            x_img.index_put_((xi, yi), energies)

        # target (blurred) only if needed
        if "target_map" in self.need:
            target_img = self.blur_op.apply(xi, yi, energies) if self.blur else x_img
        else:
            target_img = None  # type: ignore[assignment]

        # observation map y = A x (lazy, implicit fast path)
        if "observation_map" in self.need:
            y_img = self.forward_op.apply(act_f, x_img, self.x_grid, self.y_grid)
        else:
            y_img = None  # type: ignore[assignment]

        # PSF at exact (0,0,z_scan) (lazy)
        if "psf_center_map" in self.need:
            point_xyz = torch.tensor([0.0, 0.0, float(self.grid.z)], dtype=torch.float32, device=self.device)
            psf_0_0 = self.forward_op.psf_at_point(act_f, point_xyz, self.x_grid, self.y_grid)  # (1,H,W)
        else:
            psf_0_0 = None  # type: ignore[assignment]

        # normalization (default: scale by max of reference field)
        if self.norm_range != "None" and y_img is not None:
            # Normalize strictly by observation_map (y). Exclude PSF from normalization.
            denom = torch.clamp(y_img.max(), min=1e-12)
            x_img = x_img / denom
            if target_img is not None:
                target_img = target_img / denom
            y_img = y_img / denom

        # pad variable-length metadata to fixed shapes
        pad_k = self.max_nos
        pad_mask = torch.zeros((pad_k,), dtype=torch.bool)
        pad_mask[:nos] = True
        pad_idx = torch.zeros((pad_k, 2), dtype=torch.long)
        pad_pos = torch.zeros((pad_k, 2), dtype=torch.float32)
        pad_eng = torch.zeros((pad_k,), dtype=torch.float32)
        if nos > 0:
            pad_idx[:nos, 0] = torch.tensor(idx[0], dtype=torch.long)
            pad_idx[:nos, 1] = torch.tensor(idx[1], dtype=torch.long)
            pad_pos[:nos, 0] = torch.tensor(pos[0], dtype=torch.float32)
            pad_pos[:nos, 1] = torch.tensor(pos[1], dtype=torch.float32)
            pad_eng[:nos] = torch.tensor(eng, dtype=torch.float32)

        sample: Dict[str, torch.Tensor] = {}
        # mandatory minimal outputs
        if "source_map" in self.need:
            sample["source_map"] = x_img.unsqueeze(0)
        if "target_map" in self.need and target_img is not None:
            sample["target_map"] = target_img.unsqueeze(0)
        if "observation_map" in self.need and y_img is not None:
            sample["observation_map"] = y_img
        if "psf_center_map" in self.need and psf_0_0 is not None:
            sample["psf_center_map"] = psf_0_0
        # lazy A exposure (disk+mem cache) if requested
        if "A_matrix" in self.need:
            if act_f not in self._A_cache_mem:
                ss = self.cfg["scan_set"]
                A = self.forward_op.get_A_cached(
                    act_f,
                    cache_dir=self.cache_dir or os.path.join(os.path.dirname(__file__), "..", "cache_spDataset"),
                    array_id=int(ss["array_id"]),
                    fs=int(ss["fs"]),
                    nfft=int(ss["nfft"]),
                    c=float(ss["c"]),
                    x_grid=self.x_grid,
                    y_grid=self.y_grid,
                    x_range=ss["x_range"],
                    y_range=ss["y_range"],
                    z_scan=float(ss["z_scan"]),
                    device=self.device,
                    print_flag=False,
                )
                self._A_cache_mem[act_f] = A
            sample["A_matrix"] = self._A_cache_mem[act_f]
        if "num_sources" in self.need:
            sample["num_sources"] = torch.tensor(nos, dtype=torch.long)
        if "source_indices_xy" in self.need:
            sample["source_indices_xy"] = pad_idx
        if "source_positions_xy" in self.need:
            sample["source_positions_xy"] = pad_pos
        if "source_energies" in self.need:
            sample["source_energies"] = pad_eng
        if "source_mask" in self.need:
            sample["source_mask"] = pad_mask
        if "analysis_frequency_hz" in self.need:
            sample["analysis_frequency_hz"] = torch.tensor(act_f, dtype=torch.float32)
        if "source_frequency_hz" in self.need:
            sample["source_frequency_hz"] = torch.tensor(src_f, dtype=torch.float32)
        # Grid shape for solver use
        if "grid_hw" in self.need:
            sample["grid_hw"] = torch.tensor([self.x_grid, self.y_grid], dtype=torch.long)
        
        # Add physical coordinate ranges (for visualization with real-world coordinates)
        sample["grid_x_range"] = torch.tensor(self.grid.x_range, dtype=torch.float32)
        sample["grid_y_range"] = torch.tensor(self.grid.y_range, dtype=torch.float32)

        return sample


# ===========================
# Dataloader builder
# ===========================

def build_dataloader(cfg_path: str,
                     *,
                     batch_size: int = 64,
                     num_samples: Optional[int] = None,
                     blur: Optional[bool] = None,
                     need: Optional[Union[Set[str], List[str]]] = None,
                     dataset_device: Optional[Union[str, torch.device]] = None,
                     num_workers: int = 4,
                     base_seed: Optional[int] = 12345) -> torch.utils.data.DataLoader:
    """Create dataloader based on unified module.

    Args:
        cfg_path: Path to YAML configuration file
        batch_size: Batch size
        num_samples: Override cfg.source_set.num_samples
        blur: Override cfg.target_map_set.blur
        need: Specify the set of fields to return, e.g., {"source_map", "observation_map", "target_map"}
        dataset_device: Device for dataset computation "cuda" | "cpu" | torch.device
        num_workers: Number of DataLoader worker processes
        base_seed: Random seed
    
    Returns:
        DataLoader instance
    """
    cfg = load_yaml_config(cfg_path)
    cfg = dict(cfg)
    if num_samples is not None:
        cfg.setdefault("source_set", {})
        cfg["source_set"]["num_samples"] = int(num_samples)
    if blur is not None:
        cfg.setdefault("target_map_set", {})
        cfg["target_map_set"]["blur"] = bool(blur)

    # Dataset device: default CPU for better throughput (multi-process)
    if dataset_device is None:
        dataset_device = torch.device("cpu")
    elif isinstance(dataset_device, str):
        dataset_device = torch.device(dataset_device)

    ds = LinearImagingDataset(cfg, device=dataset_device, cache_dir=cfg.get("cache_dir"), need=need)

    # Worker seeding for reproducibility
    def _seed_worker(worker_id: int):
        seed = (base_seed or 0) + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(base_seed or 12345)

    # If dataset is on GPU, pin_memory must be False and workers 0
    effective_num_workers = 0 if isinstance(dataset_device, torch.device) and dataset_device.type == "cuda" else int(num_workers)
    pin_flag = False if isinstance(dataset_device, torch.device) and dataset_device.type == "cuda" else True

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=pin_flag,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


# ===========================
# CLI example
# ===========================

def _example():  # pragma: no cover
    """Example: Load dataset and print one batch"""
    import sys
    import pathlib
    
    # Use configuration file from project root directory
    project_root = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "dataset_1.yaml"
    
    if not cfg_path.exists():
        print(f"Configuration file does not exist: {cfg_path}")
        sys.exit(1)
    
    print(f"Loading configuration: {cfg_path}")
    
    # Create DataLoader with specified fields
    dl = build_dataloader(
        str(cfg_path), 
        batch_size=4,
        need=["source_map", "target_map", "observation_map"],
        num_workers=0  # Use single process in example
    )
    
    ds = dl.dataset
    print(f"Dataset: {ds}")
    print(f"Dataset size: {len(ds)}")
    
    # Get one batch
    for batch in dl:
        print(f"\nBatch content:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        break


if __name__ == "__main__":  # pragma: no cover
    _example()


