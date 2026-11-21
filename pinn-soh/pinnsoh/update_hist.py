"""
E34 — HiGBDT cho bài toán cập nhật: ước lượng SOH ở các chu kỳ không đo từ lần đo dung lượng gần nhất, có hoặc không
có đặc trưng lịch sử. Mã gộp từ kho G-doan của đồ án (improvements/study.py, hàm `anchor`, ngày 30/10/2026), giữ nguyên
cấu hình, cách lấy mẫu và cách chọn trên validation.

Lịch đo của một cell: đo ở chu kỳ đầu rồi mỗi qL chu kỳ, q ∈ {0,2; 0,1; 0,05} (q20, q10, q05), hoặc biết mọi lần đo trước
(M2). Tại chu kỳ n chỉ dùng các lần đo trước n; m là lần đo gần nhất, m2 lần đo liền trước. Chấm trên các chu kỳ không
đo (với M2: mọi chu kỳ trừ chu kỳ đầu).

Mô hình (GBDT học SOH(n) − SOH(m), trọng số mẫu cân theo cell và lịch đo):
    GBDT-cập nhật  đặc trưng như E32: z_n, z_n − z_m, SOH(m), khoảng chu kỳ tới lần đo, độ dốc giữa hai lần đo gần nhất
    HiGBDT-Δ       thêm lịch sử nhân quả của đặc trưng sạc (hai trung bình trượt mũ, ρ = 0,1 và 0,03), log khoảng chu kỳ,
                   chỉ số chu kỳ
    HiGBDT-R       như HiGBDT-Δ nhưng học phần dư so với ngoại suy tuyến tính từ hai lần đo gần nhất
    HiGBDT-V       theo từng lịch đo, chọn trên validation một trong ba mô hình trên và trọng số α trộn nó với giữ lần đo
                   hoặc ngoại suy tuyến tính: α·mô hình + (1 − α)·mốc
Mỗi mô hình chọn số cây (100, 250, 450) theo MAE trên cell validation. Thang đời L và chuẩn hoá chỉ tính trên cell train.

    python run.py E34          # 60 lượt (4 bộ × 3 lần lặp × 5 fold) -> runs/E34/up_<bộ>_cv<r>-<k>of5.npz
"""
from __future__ import annotations
import copy, json, os, time
import numpy as np

ROOT = os.environ.get('E34_ROOT', 'runs/E34')
ALPHAS = [0., .25, .5, .75, 1.]
STAGES = [100, 250, 450]
TIERS = {'q20': .2, 'q10': .1, 'q05': .05, 'M2': None}
CONFIGS = [dict(name='e32_style', rich=False, residual=False, loss='squared_error', leaves=31, l2=1.),
           dict(name='history_delta', rich=True, residual=False, loss='squared_error', leaves=15, l2=10.),
           dict(name='history_residual', rich=True, residual=True, loss='absolute_error', leaves=15, l2=10.)]
COLUMNS = ['cell', 'tier', 'cycle', 'anchor_cycle', 'y', 'hold', 'linear', 'e32_style', 'history_delta',
           'history_residual', 'selected', 'n_measured']
CLIP = (.4, 1.15)


def out_path(ds, r, k, root=ROOT):
    return f'{root}/up_{ds}_cv{r}-{k}of5.npz'


def weights(*groups):
    """Trọng số mẫu: mỗi giá trị của nhóm đầu (cell) có tổng như nhau, chia đều cho các tổ hợp nhóm sau."""
    g = np.column_stack(groups); w = np.zeros(len(g))
    for c in np.unique(g[:, 0]):
        ix = np.flatnonzero(g[:, 0] == c)
        u, inv, cnt = np.unique(g[ix, 1:], axis=0, return_inverse=True, return_counts=True)
        w[ix] = 1 / (len(u) * cnt[inv])
    return w / w.sum()


def context(ds, r, k, root='PINN4SOH/data'):
    """Cùng fold với E13/E20/E33: 5 fold theo cell, 10 % validation; 30 % tổng số cell có nhãn; L và chuẩn hoá từ train."""
    from .data import load_dataset, cv_split, choose_labelled, Normalizer
    from .forecast import life_scale
    cells = load_dataset(root, ds, clean='causal'); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, .1); seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], .3, len(cells), seed, by); sp.update(labelled=lab, unlabelled=unl)
    norm = Normalizer('global').fit([by[i] for i in sp['train']])
    L = life_scale([by[i] for i in sp['train']])
    assert all(set(sp[a]).isdisjoint(sp[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    return by, sp, norm, L, seed


def schedule(cyc, q, L, phase=0.):
    """(đo, chấm): đo ở chu kỳ đầu và mỗi qL chu kỳ (lệch pha `phase`); q=None: đo mọi chu kỳ (M2)."""
    cyc = np.asarray(cyc, float); m = np.zeros(len(cyc), bool)
    if len(cyc) == 0:
        return m, m.copy()
    if not np.isfinite(cyc).all() or np.any(np.diff(cyc) <= 0) or L <= 0 or (q is not None and q <= 0):
        raise ValueError('Invalid cycle order or measurement interval')
    if q is None:
        m[:] = True
    else:
        marks = np.r_[cyc[0], np.arange(cyc[0] + phase, cyc[-1] + 1e-9, q * L)]
        idx = np.searchsorted(cyc, marks); m[np.unique(idx[idx < len(cyc)])] = True
    return m, (np.arange(len(cyc)) > 0 if q is None else ~m)


def inputs(z, cyc, observed, L):
    """Đặc trưng tại mọi hàng có ít nhất một lần đo trước nó; observed: SOH đo, NaN ở chu kỳ không đo."""
    from .data import causal_ema
    z = np.asarray(z, float); cyc = np.asarray(cyc, float); y = np.asarray(observed, float)
    if (z.ndim != 2 or len(z) != len(cyc) or len(y) != len(cyc) or not np.isfinite(z).all()
            or not np.isfinite(cyc).all() or np.any(np.diff(cyc) <= 0) or L <= 0):
        raise ValueError('Invalid input')
    prev = (np.r_[-1, np.maximum.accumulate(np.where(np.isfinite(y), np.arange(len(y)), -1))[:-1]] if len(y)
            else np.empty(0, int))
    rows = np.flatnonzero(prev >= 0); m = prev[rows]; m2 = prev[m]; has = m2 >= 0; m2c = np.where(has, m2, m)
    dc = (cyc[rows] - cyc[m]) / L; dc2 = (cyc[m] - cyc[m2c]) / L
    dy = np.where(has, y[m] - y[m2c], 0.); sl = dy / np.maximum(dc2, 1e-9)
    f = np.column_stack([z[rows], z[rows] - z[m], y[m], dc, dy, dc2, sl, sl * dc, has.astype(float)])
    extras = []
    for a in [.1, .03]:
        e = causal_ema(z, a).astype(float); extras.extend([e[rows], e[rows] - e[m]])
    rich = np.column_stack([f, *extras, np.log1p(cyc[rows] - cyc[m]), cyc[rows] / 1000.])
    slope = dy / np.maximum((cyc[m] - cyc[m2c]) / 1000., 1e-9)
    delta = np.clip(slope, -.5, .05) * (cyc[rows] - cyc[m]) / 1000.
    return dict(basic=f, rich=rich, rows=rows, anchor=m, hold=y[m], linear_delta=delta)


def anchor_data(ids, by, norm, L, rng, train=False):
    """Mẫu của mọi cell và mọi lịch đo; khi học thêm hai pha ngẫu nhiên mỗi lịch thưa và lấy tối đa ~60 nghìn cặp."""
    chunks = []; cap = max(20, 60000 // (len(ids) * 10))
    for ci, cid in enumerate(ids):
        c = by[cid]; ok = np.isfinite(c.soh); z = norm.transform(c)[ok]; cyc = c.cycle[ok]; y = c.soh[ok]
        for ti, (tier, q) in enumerate(TIERS.items()):
            phases = [0.] if not train or q is None else [0., *rng.uniform(0, q * L, 2)]
            for ph in phases:
                mask, ev = schedule(cyc, q, L, ph); d = inputs(z, cyc, np.where(mask, y, np.nan), L)
                ix = np.flatnonzero(ev[d['rows']])
                if train and len(ix) > cap:
                    ix = np.sort(rng.choice(ix, cap, replace=False))
                if not len(ix):
                    continue
                rows = d['rows'][ix]; m = d['anchor'][ix]; n = len(ix)
                chunks.append(dict(basic=d['basic'][ix], rich=d['rich'][ix], hold=d['hold'][ix],
                                   linear_delta=d['linear_delta'][ix], y=y[rows], cell=np.full(n, ci),
                                   tier=np.full(n, ti), cycle=cyc[rows], anchor_cycle=cyc[m],
                                   n_measured=np.full(n, mask.sum())))
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def apred(m, cfg, d):
    return np.clip(d['hold'] + (d['linear_delta'] if cfg['residual'] else 0.)
                   + m.predict(d['rich' if cfg['rich'] else 'basic']), *CLIP)


def online(bundle, z, cyc, observed, tier='q10'):
    """Ước lượng trực tuyến cho một cell: z đặc trưng đã chuẩn hoá, observed SOH đo (NaN ở chu kỳ không đo)."""
    d = inputs(z, cyc, observed, bundle['L_train']); c = bundle['choices'][tier]; mi = c['model_index']
    out = np.full(len(cyc), np.nan)
    if not len(d['rows']):
        return out
    p = apred(bundle['models'][mi], bundle['configs'][mi], d)
    prior = np.clip(d['hold'] + (d['linear_delta'] if c['prior'] == 'linear' else 0.), *CLIP)
    out[d['rows']] = c['alpha'] * p + (1 - c['alpha']) * prior
    return out


def fit(lab, val, by, norm, L, seed, rng):
    """Học GBDT-cập nhật, HiGBDT-Δ, HiGBDT-R trên cell `lab` và chọn số cây cùng HiGBDT-V (mô hình, mốc, α theo từng
    lịch đo) trên cell `val`. Trả (models, configs, choices, trace, số cặp học)."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    tr = anchor_data(lab, by, norm, L, rng, True); va = anchor_data(val, by, norm, L, rng)
    wt = weights(tr['cell'], tr['tier']); wv = weights(va['cell'], va['tier'])
    models, configs, vp, trace = [], [], [], []
    for cfg in CONFIGS:
        x = tr['rich' if cfg['rich'] else 'basic']
        target = tr['y'] - tr['hold'] - (tr['linear_delta'] if cfg['residual'] else 0.)
        m = HGB(loss=cfg['loss'], learning_rate=.05, max_leaf_nodes=cfg['leaves'], l2_regularization=cfg['l2'],
                min_samples_leaf=40, early_stopping=False, warm_start=True, random_state=seed)
        best = (np.inf, None, None, None)
        for n in STAGES:
            m.set_params(max_iter=n); m.fit(x, target, sample_weight=wt * len(wt)); p = apred(m, cfg, va)
            v = float(wv @ np.abs(p - va['y'])); trace.append(dict(**cfg, n_trees=n, val_MAE=v))
            if v < best[0]:
                best = (v, copy.deepcopy(m), dict(cfg, n_trees=n), p)
        models.append(best[1]); configs.append(best[2]); vp.append(best[3])
    choices = {}
    for ti, tier in enumerate(TIERS):
        ix = va['tier'] == ti; w = weights(va['cell'][ix]); opts = []
        for mi, p in enumerate(vp):
            for prior in ['hold', 'linear']:
                b = np.clip(va['hold'] + (va['linear_delta'] if prior == 'linear' else 0.), *CLIP)
                for a in ALPHAS:
                    opts.append(dict(model_index=mi, prior=prior, alpha=a,
                                     val_MAE=float(w @ np.abs(a * p[ix] + (1 - a) * b[ix] - va['y'][ix]))))
        choices[tier] = min(opts, key=lambda q: q['val_MAE'])
    return models, configs, choices, trace, int(len(tr['y']))


def run_fold(ds, r, k, root=ROOT, save_models=False):
    """Một lượt (bộ, lần lặp, fold): học ba mô hình, chọn trên validation, chấm trên cell test, lưu npz (data, info)."""
    path = out_path(ds, r, k, root)
    start = time.time(); by, sp, norm, L, seed = context(ds, r, k); rng = np.random.default_rng(seed)
    models, configs, choices, trace, n_train = fit(sp['labelled'], sp['val'], by, norm, L, seed, rng)
    te = anchor_data(sp['test'], by, norm, L, rng)
    ps = [apred(m, c, te) for m, c in zip(models, configs)]
    hold = np.clip(te['hold'], *CLIP); linear = np.clip(te['hold'] + te['linear_delta'], *CLIP); p = np.zeros(len(hold))
    for ti, tier in enumerate(TIERS):
        c = choices[tier]; ix = te['tier'] == ti; prior = hold if c['prior'] == 'hold' else linear
        p[ix] = c['alpha'] * ps[c['model_index']][ix] + (1 - c['alpha']) * prior[ix]
    data = np.column_stack([te['cell'], te['tier'], te['cycle'], te['anchor_cycle'], te['y'], hold, linear, *ps, p,
                            te['n_measured']])
    info = dict(dataset=ds, repeat=r, fold=k, L_train=L, split=sp, choices=choices, configs=configs, search=trace,
                n_train_pairs=n_train, fit_seconds=time.time() - start, columns=COLUMNS)
    os.makedirs(root, exist_ok=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, data=data, info=np.asarray(json.dumps(info)))
    os.replace(tmp, path)
    if save_models:
        import pickle
        pickle.dump(dict(models=models, configs=configs, choices=choices, norm=norm, L_train=L, info=info),
                    open(path[:-4] + '.pkl', 'wb'))
    sel = data[:, COLUMNS.index('selected')]; y = data[:, COLUMNS.index('y')]
    return float(np.abs(sel - y)[data[:, 1] == list(TIERS).index('q10')].mean())
