"""
Unified loader for the four public single-cell datasets shipped (pre-processed)
with the PINN4SOH repository: XJTU, TJU, HUST, MIT.

Every CSV = one cell; one row = one cycle; 16 charging-curve features + capacity.

Design choices that differ from PINN4SOH (all deliberate, see docs):
  * NO per-cell normalisation of features or of the cycle index (per-cell
    min-max / z-score over a cell's *whole* life leaks its lifetime into the
    input, and is impossible online).  We offer:
      - 'global'      : z-score fitted on TRAINING cells only
      - 'first_cycle' : causal relative features (x - x_ref)/(|x_ref|+eps),
                        x_ref = mean of the first 3 cycles of the same cell,
                        followed by a global z-score fitted on training cells.
                        Deployable (needs only the cell's own first cycles),
                        chemistry-agnostic -> used for cross-dataset transfer.
      - 'per_cell_minmax' : PINN4SOH replica, ONLY for the leakage audit.
  * Cycle index is scaled by a global constant (1/1000), never per cell.
  * Splits are CELL-wise (no cell appears in two splits), stratified by
    protocol batch, fixed val/test for every labelled fraction.
"""
from __future__ import annotations
import os, re, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

FEATURES = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy']
CYCLE_SCALE = 1000.0          # cycle index / 1000  (global, never per cell)
R_GAS = 8.314                 # J / (mol K)

DATASET_INFO = {
    'XJTU': dict(chem='NCM', nominal=2.0, temp_c=25.0),
    'HUST': dict(chem='LFP', nominal=1.1, temp_c=30.0),
    'MIT':  dict(chem='LFP', nominal=1.1, temp_c=30.0),
    'TJU':  dict(chem='NCA/NCM', nominal=None, temp_c=None),   # per batch / per file
}
TJU_NOMINAL = {'Dataset_1_NCA_battery': 3.5, 'Dataset_2_NCM_battery': 3.5,
               'Dataset_3_NCM_NCA_battery': 2.5}


@dataclass
class Cell:
    cid: str                 # unique id  e.g. 'XJTU/2C_battery-1'
    dataset: str
    batch: str               # protocol batch used for stratification
    chem: str
    temp_k: float
    X: np.ndarray            # (n, 16) raw features (cleaned)
    cycle: np.ndarray        # (n,)   integer cycle index (0-based, after cleaning)
    soh: np.ndarray          # (n,)   capacity / nominal


def _clean_3sigma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-finite values and feature outliers (>3 sigma within the cell).
    Only *features* are used for the rule (never the capacity label)."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    f = df[FEATURES]
    z = (f - f.mean()) / (f.std() + 1e-12)
    keep = (z.abs() <= 3).all(axis=1)
    return df[keep]


def _read_cell(path: str, cid: str, dataset: str, batch: str, chem: str,
               nominal: float, temp_k: float) -> Cell:
    df = pd.read_csv(path)
    df['cycle'] = np.arange(len(df))
    df = _clean_3sigma_features(df)
    return Cell(cid=cid, dataset=dataset, batch=batch, chem=chem, temp_k=temp_k,
                X=df[FEATURES].values.astype(np.float32),
                cycle=df['cycle'].values.astype(np.int64),
                soh=(df['capacity'].values / nominal).astype(np.float32))


def load_dataset(root: str, name: str) -> List[Cell]:
    cells: List[Cell] = []
    if name == 'XJTU':
        d = os.path.join(root, 'XJTU data')
        for f in sorted(os.listdir(d)):
            batch = f.split('_battery')[0]           # 2C, 3C, R2.5, R3, RW, satellite
            cells.append(_read_cell(os.path.join(d, f), f'XJTU/{f[:-4]}', 'XJTU', batch,
                                    'NCM', 2.0, 273.15 + 25.0))
    elif name == 'HUST':
        d = os.path.join(root, 'HUST data')
        for f in sorted(os.listdir(d)):
            cells.append(_read_cell(os.path.join(d, f), f'HUST/{f[:-4]}', 'HUST', 'HUST',
                                    'LFP', 1.1, 273.15 + 30.0))
    elif name == 'MIT':
        d = os.path.join(root, 'MIT data')
        for b in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, b))):
                cells.append(_read_cell(os.path.join(d, b, f), f'MIT/{f[:-4]}', 'MIT', b,
                                        'LFP', 1.1, 273.15 + 30.0))
    elif name == 'TJU':
        d = os.path.join(root, 'TJU data')
        for b in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, b))):
                m = re.match(r'CY(\d+)-', f)
                temp_c = float(m.group(1)) if m else 25.0
                proto = f.split('_')[0]                 # e.g. CY25-05
                chem = {'Dataset_1_NCA_battery': 'NCA', 'Dataset_2_NCM_battery': 'NCM',
                        'Dataset_3_NCM_NCA_battery': 'NCM+NCA'}[b]
                cells.append(_read_cell(os.path.join(d, b, f), f'TJU/{b}/{f[:-4]}', 'TJU',
                                        f'{b}|{proto}', chem, TJU_NOMINAL[b], 273.15 + temp_c))
    else:
        raise ValueError(name)
    return cells


# --------------------------------------------------------------------------- splits
def split_cells(cells: List[Cell], seed: int, frac=(0.7, 0.15, 0.15)) -> Dict[str, List[str]]:
    """Cell-wise, stratified-by-batch 70/15/15 split. Returns dict of cell ids."""
    rng = np.random.RandomState(seed)
    by_batch: Dict[str, List[str]] = {}
    for c in cells:
        by_batch.setdefault(c.batch, []).append(c.cid)
    out = {'train': [], 'val': [], 'test': []}
    # tiny batches (<4 cells) are pooled so that val/test still get cells from them
    pooled: List[str] = []
    for b, ids in sorted(by_batch.items()):
        if len(ids) < 4:
            pooled.extend(ids)
            continue
        ids = list(ids); rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(frac[2] * n)))
        n_val = max(1, int(round(frac[1] * n)))
        out['test'] += ids[:n_test]
        out['val'] += ids[n_test:n_test + n_val]
        out['train'] += ids[n_test + n_val:]
    if pooled:
        rng.shuffle(pooled)
        n = len(pooled)
        n_test = int(round(frac[2] * n)); n_val = int(round(frac[1] * n))
        out['test'] += pooled[:n_test]; out['val'] += pooled[n_test:n_test + n_val]
        out['train'] += pooled[n_test + n_val:]
    assert not (set(out['train']) & set(out['test'])) and not (set(out['train']) & set(out['val'])) \
        and not (set(out['val']) & set(out['test'])), 'cell leakage between splits!'
    return out


def choose_labelled(train_ids: List[str], label_frac_of_all: float, n_all: int, seed: int,
                    cells_by_id: Dict[str, Cell]) -> Tuple[List[str], List[str]]:
    """Pick a subset of the training cells to carry labels.
    label_frac_of_all is the fraction of ALL cells (so 0.7 == full 70/15/15 protocol,
    0.3 == 30 % of all cells labelled, remaining training cells are unlabelled).
    Stratified by batch so every protocol keeps at least one labelled cell when possible."""
    rng = np.random.RandomState(seed + 1000)
    n_lab = int(round(label_frac_of_all * n_all))
    n_lab = max(1, min(n_lab, len(train_ids)))
    by_batch: Dict[str, List[str]] = {}
    for cid in train_ids:
        by_batch.setdefault(cells_by_id[cid].batch, []).append(cid)
    # round-robin over batches after shuffling each
    order = []
    lists = [list(v) for _, v in sorted(by_batch.items())]
    for l in lists:
        rng.shuffle(l)
    rng.shuffle(lists)
    while any(lists):
        for l in lists:
            if l:
                order.append(l.pop())
    labelled = order[:n_lab]
    unlabelled = [cid for cid in train_ids if cid not in set(labelled)]
    return labelled, unlabelled


# --------------------------------------------------------------------------- normalisation
class Normalizer:
    """Feature normaliser fitted on training cells only."""
    def __init__(self, mode: str = 'global', ref_cycles: int = 3):
        assert mode in ('global', 'first_cycle', 'per_cell_minmax')
        self.mode, self.ref_cycles = mode, ref_cycles
        self.mu = None; self.sd = None

    def _rel(self, c: Cell) -> np.ndarray:
        ref = c.X[:self.ref_cycles].mean(axis=0, keepdims=True)
        return (c.X - ref) / (np.abs(ref) + 1e-6)

    def _raw(self, c: Cell) -> np.ndarray:
        if self.mode == 'first_cycle':
            return self._rel(c)
        if self.mode == 'per_cell_minmax':      # PINN4SOH replica (leaky, audit only)
            mn, mx = c.X.min(0, keepdims=True), c.X.max(0, keepdims=True)
            return 2 * (c.X - mn) / (mx - mn + 1e-12) - 1
        return c.X

    def fit(self, cells: List[Cell]):
        Xs = np.concatenate([self._raw(c) for c in cells], axis=0)
        self.mu = Xs.mean(0, keepdims=True); self.sd = Xs.std(0, keepdims=True) + 1e-6
        return self

    def transform(self, c: Cell) -> np.ndarray:
        return ((self._raw(c) - self.mu) / self.sd).astype(np.float32)


def cycle_input(c: Cell, mode: str = 'global') -> np.ndarray:
    """Cycle-index input. 'global' = N/1000 (causal). 'per_cell_minmax' = PINN4SOH replica
    (maps the cell's own life to [-1,1] -> leaks lifetime; audit only)."""
    if mode == 'per_cell_minmax':
        n = c.cycle.astype(np.float32)
        return (2 * (n - n.min()) / (n.max() - n.min() + 1e-12) - 1).astype(np.float32)
    return (c.cycle.astype(np.float32) / CYCLE_SCALE).astype(np.float32)


# --------------------------------------------------------------------------- tensors
@dataclass
class PairArrays:
    """Consecutive-cycle pairs (N, N+1) of the same cell, flattened over cells."""
    x1: np.ndarray; x2: np.ndarray        # (m,16) normalised features
    t1: np.ndarray; t2: np.ndarray        # (m,1)  cycle inputs
    y1: np.ndarray; y2: np.ndarray        # (m,1)  SOH labels (may be unused)
    invT: np.ndarray                      # (m,1)  1/T  [1/K]
    cell_idx: np.ndarray                  # (m,)   integer id of the cell
    first: np.ndarray                     # (m,)   bool: pair starts at the cell's first cycle

def make_pairs(cells: List[Cell], norm: Normalizer, cyc_mode: str = 'global') -> PairArrays:
    xs1, xs2, ts1, ts2, ys1, ys2, invT, ci, first = [], [], [], [], [], [], [], [], []
    for k, c in enumerate(cells):
        X = norm.transform(c); t = cycle_input(c, cyc_mode)[:, None]; y = c.soh[:, None]
        if len(X) < 2:
            continue
        xs1.append(X[:-1]); xs2.append(X[1:]); ts1.append(t[:-1]); ts2.append(t[1:])
        ys1.append(y[:-1]); ys2.append(y[1:])
        invT.append(np.full((len(X) - 1, 1), 1.0 / c.temp_k, dtype=np.float32))
        ci.append(np.full(len(X) - 1, k, dtype=np.int64))
        f = np.zeros(len(X) - 1, dtype=bool); f[0] = True; first.append(f)
    cat = lambda l: np.concatenate(l, 0)
    return PairArrays(cat(xs1), cat(xs2), cat(ts1), cat(ts2), cat(ys1), cat(ys2),
                      cat(invT), cat(ci), cat(first))


def make_points(cells: List[Cell], norm: Normalizer, cyc_mode: str = 'global'):
    """All cycles (not pairs) -> (X, t, y, cell_idx) for evaluation."""
    X, T, Y, C = [], [], [], []
    for k, c in enumerate(cells):
        X.append(norm.transform(c)); T.append(cycle_input(c, cyc_mode)[:, None])
        Y.append(c.soh[:, None]); C.append(np.full(len(c.soh), k, dtype=np.int64))
    cat = lambda l: np.concatenate(l, 0)
    return cat(X), cat(T), cat(Y), cat(C)


@dataclass
class PointSet:
    """All cycles of a list of cells, flattened, with cell boundaries kept so that
    (N, N+h) pairs of the SAME cell can be sampled with a random horizon h."""
    X: np.ndarray; T: np.ndarray; Y: np.ndarray; invT: np.ndarray; cell: np.ndarray
    start: np.ndarray; end: np.ndarray        # per-point: first / one-past-last global index of its cell

    def __len__(self):
        return len(self.X)

    def sample_pairs(self, rng: np.random.RandomState, batch: int, h_min: int, h_max: int):
        """Return global indices (i, j) with j = i + h, h_min <= h <= h_max, same cell
        (h is clipped at the cell's last cycle)."""
        i = rng.randint(0, len(self.X), size=batch)
        h = rng.randint(h_min, h_max + 1, size=batch)
        j = np.minimum(i + h, self.end[i] - 1)
        # if i is the last point of its cell, step back one
        last = j == i
        i = np.where(last, np.maximum(i - 1, self.start[i]), i)
        return i, j


def make_pointset(cells: List[Cell], norm: Normalizer, cyc_mode: str = 'global') -> PointSet:
    X, T, Y, I, C, S, E = [], [], [], [], [], [], []
    off = 0
    for k, c in enumerate(cells):
        n = len(c.soh)
        if n < 2:
            continue
        X.append(norm.transform(c)); T.append(cycle_input(c, cyc_mode)[:, None]); Y.append(c.soh[:, None])
        I.append(np.full((n, 1), 1.0 / c.temp_k, dtype=np.float32)); C.append(np.full(n, k, dtype=np.int64))
        S.append(np.full(n, off, dtype=np.int64)); E.append(np.full(n, off + n, dtype=np.int64))
        off += n
    cat = lambda l: np.concatenate(l, 0)
    return PointSet(cat(X), cat(T), cat(Y), cat(I), cat(C), cat(S), cat(E))


def summary(cells: List[Cell]) -> pd.DataFrame:
    rows = [dict(cid=c.cid, dataset=c.dataset, batch=c.batch, chem=c.chem, temp_c=c.temp_k - 273.15,
                 n_cycles=len(c.soh), soh_first=float(c.soh[0]), soh_last=float(c.soh[-1])) for c in cells]
    return pd.DataFrame(rows)
