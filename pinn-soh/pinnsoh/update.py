"""
E32: mô hình học cập nhật từ lần đo dung lượng gần nhất (docs/de-cuong-E32.md).

Khi cell cần ước lượng có vài lần đo dung lượng (mức thông tin M1, M2 của analysis/analyze_e32.py), ước lượng ở
hàng n là

    ŷ_n = y_m + f(z_n, z_n − z_m, y_m, (c_n − c_m)/L, độ dốc của hai lần đo gần nhất),

với m là hàng được đo gần nhất trước n, z là đặc trưng đã chuẩn hoá, c là chỉ số chu kỳ, L là trung vị số chu kỳ của bộ
và f là GBDT học trên cell train có nhãn. Cặp (n, m) để học lấy theo đúng các lịch đo của phần đánh giá (đo ở chu kỳ
đầu và sau mỗi qL chu kỳ, q = 0,2; 0,1; 0,05, với pha 0 và hai pha ngẫu nhiên; và đo mọi chu kỳ trước), một mô hình
chung cho mọi lịch; số cây chọn theo MAE trên cặp của cell validation. Trên cell test, lịch đo có pha 0 nên hàng đánh
giá trùng hàng của bộ lọc Kalman (E25) và so ghép cặp được với bộ lọc Kalman, gần nhất, giữ nguyên.
"""
from __future__ import annotations
import json, os, time
import numpy as np

from .data import cv_split, choose_labelled, Normalizer

TIERS = {'q20': 0.2, 'q10': 0.1, 'q05': 0.05, 'M2': None}     # None: đo mọi chu kỳ trước (M2)
MAX_PAIRS, MAX_PAIRS_VAL = 300_000, 100_000
GBDT = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)


def schedule(cyc, q, L, phase=0.0):
    """Hàng được đo và hàng tính sai số; pha 0 trùng đúng lịch của analysis/analyze_e31.py. Pha > 0 (chỉ khi học): lần đo
    đầu vẫn ở chu kỳ đầu, các lần sau dời đi `phase` chu kỳ."""
    n = len(cyc); is_m = np.zeros(n, bool)
    if q is None:
        is_m[:] = True
        return is_m, np.arange(n) >= 1
    marks = np.arange(cyc[0] + phase, cyc[-1] + 1e-9, q * L)
    if phase > 0:
        marks = np.concatenate([[cyc[0]], marks])
    is_m[np.unique(np.searchsorted(cyc, marks, side='left'))] = True
    return is_m, ~is_m


def _last_measured(is_m):
    idx = np.where(is_m, np.arange(len(is_m)), -1)
    return np.concatenate([[-1], np.maximum.accumulate(idx)[:-1]])


def pairs(Z, cyc, y, L, q, phase=0.0):
    """Đầu vào u_n, mục tiêu y_n − y_m, chỉ số hàng n và m cho mọi hàng được đánh giá của một cell theo một lịch đo."""
    is_m, ev = schedule(cyc, q, L, phase)
    prev = _last_measured(is_m)
    rows = np.flatnonzero(ev & (prev >= 0))
    m = prev[rows]
    m2 = prev[m]; has = m2 >= 0; m2c = np.where(has, m2, m)
    dc, dc2 = (cyc[rows] - cyc[m]) / L, (cyc[m] - cyc[m2c]) / L
    slope = np.where(has, (y[m] - y[m2c]) / np.maximum(dc2, 1e-9), 0.0)
    F = np.hstack([Z[rows], Z[rows] - Z[m], y[m][:, None], dc[:, None],
                   np.where(has, y[m] - y[m2c], 0.0)[:, None], dc2[:, None], slope[:, None], (slope * dc)[:, None],
                   has[:, None].astype(float)])
    return F, y[rows] - y[m], rows, m


def _cell_arrays(c, norm):
    """Chỉ các hàng có nhãn, như dự đoán test của mọi mô hình khác (make_pointset(labelled_only=True))."""
    ok = np.isfinite(c.soh)
    return norm.transform(c)[ok].astype(np.float64), c.cycle[ok].astype(np.float64), c.soh[ok].astype(np.float64)


def train_set(cells, norm, L, rng, max_n=MAX_PAIRS):
    Fs, ts = [], []
    for c in cells:
        Z, cyc, y = _cell_arrays(c, norm)
        for q in TIERS.values():
            for ph in ([0.0] if q is None else [0.0] + list(rng.uniform(0, q * L, 2))):
                F, t, _, _ = pairs(Z, cyc, y, L, q, ph)
                Fs.append(F); ts.append(t)
    F, t = np.vstack(Fs), np.concatenate(ts)
    if len(t) > max_n:
        i = np.sort(rng.choice(len(t), max_n, replace=False)); F, t = F[i], t[i]
    return F, t


def fit(F, t, Fv, tv, seed):
    """Số cây theo MAE trên cặp validation (bước 50, dừng sau 3 lần không cải thiện, tối đa 1 000)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(**GBDT, max_iter=50, warm_start=True, early_stopping=False, random_state=seed)
    best, bad = None, 0
    for it in range(50, 1001, 50):
        m.set_params(max_iter=it); m.fit(F, t)
        v = float(np.abs(m.predict(Fv) - tv).mean())
        if best is None or v < best[0] - 1e-7:
            best, bad = (v, it), 0
        else:
            bad += 1
            if bad >= 3:
                break
    final = HistGradientBoostingRegressor(**GBDT, max_iter=best[1], early_stopping=False, random_state=seed).fit(F, t)
    return final, best


def predict_tiers(model, cells, norm, L):
    """Ước lượng trên cell test theo từng lịch đo (pha 0); kèm MAE theo cell để in khi chạy."""
    out, mae = {}, {}
    for name, q in TIERS.items():
        rec = dict(cell=[], cycle=[], y=[], est=[], base=[])
        per = []
        for i, c in enumerate(cells):
            Z, cyc, y = _cell_arrays(c, norm)
            F, _, rows, m = pairs(Z, cyc, y, L, q)
            if not len(rows):
                continue
            est = y[m] + model.predict(F)
            rec['cell'] += [i] * len(rows); rec['cycle'] += cyc[rows].tolist(); rec['y'] += y[rows].tolist()
            rec['est'] += est.tolist(); rec['base'] += y[m].tolist()
            per.append(float(np.abs(est - y[rows]).mean()))
        out[name] = rec
        mae[name] = float(np.mean(per))
    return out, mae


def _run(cells, tr_ids, va_ids, te_ids, norm_ids, seed, cfg, path):
    from .forecast import life_scale
    from .train import write_json
    t0 = time.time()
    by = {c.cid: c for c in cells}
    L = life_scale(cells)
    norm = Normalizer('global').fit([by[i] for i in norm_ids])
    rng = np.random.default_rng(seed)
    F, t = train_set([by[i] for i in tr_ids], norm, L, rng)
    Fv, tv = train_set([by[i] for i in va_ids], norm, L, rng, MAX_PAIRS_VAL)
    model, (v, n_trees) = fit(F, t, Fv, tv, seed)
    te = [by[i] for i in te_ids]
    tiers, mae = predict_tiers(model, te, norm, L)
    write_json(dict(config=cfg, n_cells=dict(train_labelled=len(tr_ids), val=len(va_ids), test=len(te_ids)),
                    hyper=dict(n_trees=n_trees, n_pairs=int(len(t)), n_pairs_val=int(len(tv)), L=L), val_MAE=v,
                    train_time_s=time.time() - t0, test=dict(MAE_cell=mae), test_tiers=tiers,
                    cell_ids=[c.cid for c in te]), path)
    return mae['q05']


def cv_path(ds, frac, r, k, out_dir='runs/E32'):
    return f'{out_dir}/update_{ds}_f{frac}_cv{r}-{k}of5.json'


def run_cv(ds, frac, r, k, out_dir='runs/E32'):
    """Một lượt (bộ, mức nhãn, lần lặp r, fold k): cùng cách chia, cùng cell có nhãn, cùng normaliser với E20 và E30."""
    from .baselines import cells_of
    path = cv_path(ds, frac, r, k, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']['q05']
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], frac, len(cells), seed, by)
    cfg = dict(tag='U', model='update', label_frac=frac, dataset=ds, repeat=r, fold=k, seed=seed, split='cv', n_folds=5)
    return _run(cells, lab, sp['val'], sp['test'], lab + unl, seed, cfg, path)
