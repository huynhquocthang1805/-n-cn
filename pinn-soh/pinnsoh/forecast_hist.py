"""
HiGBDT: GBDT dự báo có đặc trưng lịch sử SOH, lấy từ kho G-doan của đồ án (improvements/features.py và hàm
`forecast` trong improvements/study.py, ngày 30/10/2026), giữ nguyên cấu hình và cách chọn để kết quả trùng với bản đó.

Đặc trưng tại gốc k (chỉ dùng hàng <= k):
    z_k    23 đặc trưng gốc của E22 (pinnsoh/forecast.py: origin_features)
    rich   z_k cùng SOH tại k và độ lệch so với mức l_k; với cửa sổ 5, 10, 30, 100, 200 hàng: trung bình trừ SOH hiện
           tại, độ lệch chuẩn, độ dốc; với độ trễ 1, 3, 7, 15, 31, 63 hàng: thay đổi SOH và khoảng chu kỳ
Mô hình: HistGradientBoosting học độ giảm l_k − SOH(k+h) từ (đặc trưng, h), 6 cấu hình (CONFIGS) × 3 mức số cây
(STAGES); cấu hình, số cây và trọng số trộn với GBDT gốc của E22 (ALPHAS) chọn theo MAE trên cell validation.
Trọng số mẫu: mỗi cell (và mỗi gốc) đóng góp như nhau (`weights`).

Thêm so với bản G-doan: `features` nhận lịch sử SOH tường minh (`soh`), để dùng được với lịch sử ước lượng; khi
soh=None kết quả trùng từng bit với bản gốc.
"""
from __future__ import annotations
import numpy as np

CONFIGS = [
    dict(name='base_L2_15', rich=False, loss='squared_error', leaves=15, l2=10., mono=True, residual=False),
    dict(name='history_L2_15', rich=True, loss='squared_error', leaves=15, l2=10., mono=True, residual=False),
    dict(name='history_L1_15', rich=True, loss='absolute_error', leaves=15, l2=10., mono=True, residual=False),
    dict(name='history_L2_31', rich=True, loss='squared_error', leaves=31, l2=5., mono=True, residual=False),
    dict(name='history_free_15', rich=True, loss='squared_error', leaves=15, l2=10., mono=False, residual=False),
    dict(name='linear_residual_15', rich=True, loss='squared_error', leaves=15, l2=10., mono=False, residual=True)]
STAGES = [100, 250, 450]
ALPHAS = [0., .25, .5, .75, 1.]
CLIP = (.4, 1.15)


def weights(*groups):
    """Trọng số mẫu: mỗi giá trị của nhóm đầu (cell) có tổng trọng số như nhau, trong đó chia đều cho các tổ hợp nhóm sau."""
    g = np.column_stack(groups); w = np.zeros(len(g))
    for c in np.unique(g[:, 0]):
        ix = np.flatnonzero(g[:, 0] == c)
        u, inv, cnt = np.unique(g[ix, 1:], axis=0, return_inverse=True, return_counts=True)
        w[ix] = 1 / (len(u) * cnt[inv])
    return w / w.sum()


def features(c, x, soh=None):
    """(z, rich, mức l) cho mọi hàng của cell c; x là đặc trưng sạc đã chuẩn hoá; soh là lịch sử SOH (mặc định: đo)."""
    from .forecast import origin_features, causal_slope
    hist = c.soh if soh is None else soh
    z, lev = origin_features(hist, c.cycle, x, c.temp_k)
    s = np.asarray(hist).astype(float); t = c.cycle.astype(float); n = len(s); idx = np.arange(n)
    extras = [s, s - lev]
    for w in (5, 10, 30, 100, 200):
        start = np.maximum(0, idx - w + 1); cnt = idx - start + 1
        cs = np.r_[0., np.cumsum(s)]; cs2 = np.r_[0., np.cumsum(s * s)]
        mean = (cs[idx + 1] - cs[start]) / cnt
        std = np.sqrt(np.maximum(0., (cs2[idx + 1] - cs2[start]) / cnt - mean * mean))
        extras.extend([mean - s, std, causal_slope(s, t, w) * 1000.])
    for lag in (1, 3, 7, 15, 31, 63):
        j = np.maximum(0, idx - lag); extras.extend([s - s[j], (t - t[j]) / 1000.])
    return z.astype(float), np.column_stack([z, *extras]).astype(float), lev


def sample(ids, by, feat, origins=None):
    """Mẫu (gốc, tầm): origins=None -> 32 gốc × 32 tầm rải đều mỗi cell (để học); ngược lại gốc là các chu kỳ cho
    trước (hàng đầu tiên có chu kỳ >= gốc) và mọi hàng sau gốc (để chọn và chấm). Đích y luôn là SOH đo."""
    chunks = []
    for ci, cid in enumerate(ids):
        c = by[cid]; z, rich, lev = feat[cid]; n = len(c.soh)
        oo = (np.unique(np.linspace(5, n - 2, 32).round().astype(int)) if origins is None
              else np.unique([int(np.searchsorted(c.cycle, o)) for o in origins]))
        for i in oo:
            if i >= n - 1 or i < 0:
                continue
            jj = np.unique(np.linspace(i + 1, n - 1, 32).round().astype(int)) if origins is None else np.arange(i + 1, n)
            h = c.cycle[jj] - c.cycle[i]; m = len(h)
            chunks.append(dict(z=np.repeat(z[i:i + 1], m, axis=0), rich=np.repeat(rich[i:i + 1], m, axis=0), h=h,
                               lev=np.full(m, lev[i]), y=c.soh[jj], cell=np.full(m, ci), origin=np.full(m, c.cycle[i])))
    if not chunks:
        raise ValueError('No forecast samples')
    return {k: np.concatenate([q[k] for q in chunks]) for k in chunks[0]}


def matrix(d, cfg):
    return np.column_stack([d['rich' if cfg['rich'] else 'z'], d['h'] / 1000.])


def base(d, cfg):
    return d['lev'] + np.minimum(d['z'][:, 2], 0.) * d['h'] / 1000. if cfg['residual'] else d['lev']


def predict(m, cfg, d):
    return np.clip(base(d, cfg) - m.predict(matrix(d, cfg)), *CLIP)


def search(tr, va, seed, old_val):
    """Chọn cấu hình, số cây và trọng số trộn với GBDT gốc theo MAE có trọng số trên mẫu validation `va`.
    old_val: dự báo của GBDT gốc trên `va`. Trả (mô hình, cấu hình kèm số cây, trọng số mới a, vết tìm kiếm)."""
    import copy
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    wt = weights(tr['cell']); wv = weights(va['cell'], va['origin'])
    score = lambda p: float(wv @ np.abs(p - va['y']))
    trace = [dict(name='original_gbdt', val_MAE=score(old_val))]
    best = (np.inf, None, None, None)
    for cfg in CONFIGS:
        x = matrix(tr, cfg); target = base(tr, cfg) - tr['y']
        m = HGB(loss=cfg['loss'], learning_rate=.05, max_leaf_nodes=cfg['leaves'], l2_regularization=cfg['l2'],
                min_samples_leaf=25, early_stopping=False, warm_start=True, random_state=seed,
                monotonic_cst=[0] * (x.shape[1] - 1) + [int(cfg['mono'])])
        for n in STAGES:
            m.set_params(max_iter=n); m.fit(x, target, sample_weight=wt * len(wt))
            pv = predict(m, cfg, va); v = score(pv); trace.append(dict(**cfg, n_trees=n, val_MAE=v))
            if v < best[0]:
                best = (v, copy.deepcopy(m), dict(cfg, n_trees=n), pv)
    _, m, cfg, pv = best
    a = min(ALPHAS, key=lambda a: score(a * pv + (1 - a) * old_val))
    return m, cfg, a, trace, score
