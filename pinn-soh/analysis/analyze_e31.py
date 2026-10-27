"""
E31: so với SambaMixer trên NASA theo lượng thông tin đo được, rồi kiểm tra lại trên mọi bộ (tai-lieu/de-cuong-E31.md).

    python analysis/analyze_e31.py    # -> results/e31_m0_samba.csv, e31_m0_cv.csv, e31_m0_tests.csv, e31_tuned.csv,
                                      #    e31_params.csv, e31_tiers.csv, e31_tiers_samba.csv, results/fig_e31_tiers.png|pdf

Ba mức thông tin, đều ít hơn SambaMixer (vốn nhận chuỗi đo của chính lần xả cần ước lượng):
    M0   không đo cell đánh giá: mọi mô hình của E24 và E30 trên NASA, NASAdt (lượt cũ) và NASAfc (E31: thêm 7 đặc trưng
         của cả lần sạc và log(1 + giờ nghỉ)); pinn_lai_t là PINN-lai tinh chỉnh, (ω, w) chọn trên validation như
         analysis/analyze_tuning.py (NASA-L: B0018 và B0055 của 3 hạt giống)
    M1   đo dung lượng ở chu kỳ đầu và mỗi qL chu kỳ (q20, q10, q05: q = 0,2; 0,1; 0,05; L = trung vị số chu kỳ của bộ);
         sai số tính trên các chu kỳ không được đo
    M2   đo mọi chu kỳ trước (dự báo một bước); sai số tính trên mọi chu kỳ trừ chu kỳ đầu
M1, M2: bộ lọc Kalman của E25 (độ lệch theo cell là quá trình Ornstein–Uhlenbeck) trên dự đoán của pinn_lai_t; μ0, R, P0
ước lượng trên cell validation của chính fold (NASA-L: của cùng hạt giống), ℓ chọn trên lưới của E25 riêng cho từng lịch
đo. Đối chứng: gan_nhat (dịch dự đoán theo sai số của lần đo gần nhất), giu_nguyen (lấy đúng SOH của lần đo gần nhất,
không dùng mô hình). Bốn bộ chính dùng lượt của E20 với (ω, w) của results/e20_tuned.csv; NASA và NASAdt lượt của E24,
NASAfc lượt của E31. So sánh ghép cặp theo cell (trung bình qua lần lặp), Wilcoxon, p hiệu chỉnh Holm qua bốn bộ chính.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.data import load_dataset
from pinnsoh.experiments import E30_MODELS, E30_L
from pinnsoh.forecast import life_scale
from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh import seqmodels

OUT = 'results'
MAIN = ['XJTU', 'TJU', 'MIT', 'HUST']
NASA_DS = ['NASA', 'NASAdt', 'NASAfc']
SRC = {'NASA': 'runs/E24', 'NASAdt': 'runs/E24', 'NASAfc': 'runs/E31'}
SEQ_SRC = {'NASA': 'runs/E30', 'NASAdt': 'runs/E30', 'NASAfc': 'runs/E31'}
FRACS = {**{ds: (0.1, 0.3, 0.5, 0.7) for ds in MAIN}, **{ds: (0.3, 0.7) for ds in NASA_DS}}
REPEATS = (0, 1, 2)
OMEGAS, WINDOWS = (0.3, 0.4, 0.5, 0.6, 0.7), (5, 10, 20, 40)
SCHED = {'q20': 0.2, 'q10': 0.1, 'q05': 0.05, 'M2': None}        # M1: ba lịch đo thưa; M2: đo mọi chu kỳ trước
TIERS = ['M0'] + list(SCHED)
ELL_GRID = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 3.0, np.inf])  # như E25, theo đơn vị L
METHODS = ['kalman', 'gan_nhat', 'giu_nguyen']
# SambaMixer (Olalde-Verano và cộng sự, 2024), Bảng 4: MAE và RMSE (SOH) trên ba cell đánh giá của NASA-L
SAMBA = {'NASA/B0006': (0.01173, 0.02068), 'NASA/B0007': (0.01197, 0.01285), 'NASA/B0047': (0.00512, 0.00645)}


# --------------------------------------------------------------------------- đọc dự đoán
def _read(path):
    """Dự đoán test và validation của một lượt; mã cell là mảng object trỏ tới cùng một chuỗi nên nhẹ (đọc thẳng JSON đủ
    nhanh, khoảng 30 giây cho mọi lượt, nên không cần bộ nhớ đệm)."""
    d = json.load(open(path))
    out = {}
    for part, key in (('test', 'test_predictions'), ('val', 'val_predictions')):
        if key in d:
            tp = d[key]
            out[part] = dict(cell=np.asarray(tp['cell_ids'], dtype=object)[np.asarray(tp['cell'], int)],
                             cyc=np.asarray(tp['cycle'], float), y=np.asarray(tp['y'], float),
                             p=np.asarray(tp['p'], float))
    return out


def _units():
    """Mọi đơn vị đánh giá: (cách chia, bộ, mức nhãn, lần lặp hay hạt giống, fold) -> các file của nó."""
    U = {}
    for ds in MAIN + NASA_DS:
        src = SRC.get(ds, 'runs/E20')
        for f, r, k in [(f, r, k) for f in FRACS[ds] for r in REPEATS for k in range(5)]:
            s = 100 * r + k
            u = dict(P=[f'{src}/P_{ds}_pinn_semi_f{f}_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}')
                        + '.json' for m in range(3)], G=f'{src}/gbdt_{ds}_f{f}_cv{r}-{k}of5.json')
            if ds in NASA_DS:
                u['M'] = f'{src}/M_{ds}_mlp_f{f}_s{s}_global_global_cv{r}-{k}of5.json'
                u['seq'] = {m: seqmodels.cv_path(m, E30_L[m], ds, f, r, k, SEQ_SRC[ds]) for m in E30_MODELS}
            U[('cv', ds, f, r, k)] = u
    for ds, s in [(ds, s) for ds in NASA_DS for s in REPEATS]:
        src = SRC[ds]
        U[('samba', ds, 0.7, s, 0)] = dict(
            P=[f'{src}/S_{ds}_pinn_semi_f0.7_s{s}_global_global' + ('' if m == 0 else f'_m{m}') + '.json' for m in range(3)],
            G=f'{src}/gbdt_{ds}_samba_s{s}.json', M=f'{src}/S_{ds}_mlp_f0.7_s{s}_global_global.json',
            seq={m: seqmodels.samba_path(m, E30_L[m], ds, s, SEQ_SRC[ds]) for m in E30_MODELS})
    return U


def _align(q, cell, cyc):
    if np.array_equal(q['cell'], cell) and np.array_equal(q['cyc'], cyc):
        return q['p']
    s = pd.Series(q['p'], index=pd.MultiIndex.from_arrays([q['cell'], q['cyc']]))
    out = s.reindex(pd.MultiIndex.from_arrays([cell, cyc])).values
    assert np.isfinite(out).all()
    return out


def build():
    """Dự đoán test và validation của mọi đơn vị đủ file: y, p3 (trung bình 3 PINN-semi), g (GBDT) và, ở NASA, các mô
    hình khác trên test."""
    U = _units()
    need = lambda u: u['P'] + [u['G']]
    files = sorted({p for u in U.values() for p in need(u) + [u.get('M')] + list(u.get('seq', {}).values())
                    if p and os.path.exists(p)})
    data = {p: _read(p) for p in files}
    recs = {}
    for key, u in U.items():
        if not all(p in data for p in need(u)):
            continue
        P = [data[p] for p in u['P']]; G = data[u['G']]
        rec = {}
        for part in ('test', 'val'):
            b = P[0][part]
            for o in P[1:] + [G]:
                assert np.array_equal(o[part]['cell'], b['cell']) and np.array_equal(o[part]['cyc'], b['cyc']), key
            t = dict(cell=b['cell'], cyc=b['cyc'], y=b['y'], p3=np.mean([o[part]['p'] for o in P], axis=0),
                     g=G[part]['p'], pinn=b['p'])
            if part == 'test' and 'M' in u:
                if u['M'] in data:
                    t['mlp'] = _align(data[u['M']]['test'], b['cell'], b['cyc'])
                for m, p in u['seq'].items():
                    if p in data:
                        t[m] = _align(data[p]['test'], b['cell'], b['cyc'])
            rec[part] = t
        recs[key] = rec
    return recs


# --------------------------------------------------------------------------- PINN-lai tinh chỉnh
def _cell_mae(y, p, cell):
    return [float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)]


def tune(recs):
    """(ω, w) cho mỗi (cách chia, bộ, mức nhãn): MAE theo cell trung bình trên validation của mọi lần lặp × fold (hay mọi
    hạt giống); bốn bộ chính lấy đúng kết quả đã chốt ở results/e20_tuned.csv và kiểm tra lại."""
    acc = {}
    for (split, ds, f, r, k), rec in recs.items():
        v = rec['val']
        for om in OMEGAS:
            raw = om * v['p3'] + (1 - om) * v['g']
            for w in WINDOWS:
                acc.setdefault((split, ds, f, om, w), []).extend(_cell_mae(v['y'], by_cell(causal_trend, raw, v['cell'],
                                                                                         v['cyc'], w=w), v['cell']))
    V = pd.DataFrame([dict(split=s, ds=ds, f=f, omega=om, w=w, val=np.mean(e)) for (s, ds, f, om, w), e in acc.items()])
    best = V.loc[V.groupby(['split', 'ds', 'f']).val.idxmin()].reset_index(drop=True)
    old = pd.read_csv(f'{OUT}/e20_tuned.csv').set_index(['ds', 'f'])
    for _, b in best[best.ds.isin(MAIN)].iterrows():
        o = old.loc[(b.ds, b.f)]
        assert np.isclose(o.omega, b.omega) and int(o.w) == int(b.w), (b.ds, b.f)
    return best


def lai(t, om, w):
    return by_cell(causal_trend, om * t['p3'] + (1 - om) * t['g'], t['cell'], t['cyc'], w=int(w))


# --------------------------------------------------------------------------- bộ lọc Kalman, nhiều chuỗi cùng lúc
def _pad(groups, key):
    n = max(len(g[key]) for g in groups)
    out = np.full((len(groups), n), np.nan)
    for i, g in enumerate(groups):
        out[i, :len(g[key])] = g[key]
    return out


def kalman_batch(cyc, r, meas, mu0, P0, R, ell):
    """Đúng bộ lọc của analyze_e25.kalman, chạy song song cho nhiều chuỗi: cyc, r, meas dạng (chuỗi, hàng), NaN ở hàng
    đệm; ell (chuỗi,) theo chu kỳ, có thể là inf. B[:, m] chỉ dùng các lần đo ở hàng < m."""
    S, N = cyc.shape
    b = np.full(S, mu0, float); P = np.full(S, P0, float); last = cyc[:, 0].copy()
    inv = np.where(np.isfinite(ell), 1.0 / np.where(np.isfinite(ell), ell, 1.0), 0.0)
    B = np.full((S, N), np.nan); V = np.full((S, N), np.nan)
    for m in range(N):
        c = cyc[:, m]; ok = np.isfinite(c)
        a = np.where(ok, np.exp(-(np.where(ok, c, last) - last) * inv), 1.0)
        last = np.where(ok, c, last)
        b = mu0 + a * (b - mu0); P = a * a * P + (1 - a * a) * P0
        B[:, m], V[:, m] = b, P
        u = meas[:, m] & ok
        K = np.where(u, P / (P + R), 0.0)
        b = b + K * (np.where(u, r[:, m], 0.0) - b); P = (1 - K) * P
    return B, V


def schedule(cyc, q, L):
    """Hàng được đo (bool) và hàng tính sai số. q None: mọi hàng được đo, tính sai số từ hàng thứ hai (M2). Ngược lại
    đo chu kỳ đầu và mỗi qL chu kỳ, lần đo rơi vào hàng đầu tiên có chu kỳ >= mốc (như E25), tính trên hàng không đo."""
    n = len(cyc); is_m = np.zeros(n, bool)
    if q is None:
        is_m[:] = True
        ev = np.arange(n) >= 1
    else:
        marks = np.arange(cyc[0], cyc[-1] + 1e-9, q * L)
        is_m[np.unique(np.searchsorted(cyc, marks, side='left'))] = True
        ev = ~is_m
    return is_m, ev


def _last_measured(is_m):
    """Chỉ số hàng được đo gần nhất trước mỗi hàng (−1 nếu chưa có)."""
    idx = np.where(is_m, np.arange(len(is_m)), -1)
    prev = np.maximum.accumulate(idx)
    return np.concatenate([[-1], prev[:-1]])


def fit_params(cells, L):
    """μ0, R, P0 trên cell validation như E25; ℓ chọn trên lưới bằng MAE theo cell riêng cho từng lịch đo."""
    r = np.concatenate([c['y'] - c['p'] for c in cells])
    mu0 = float(r.mean())
    dr = np.concatenate([np.diff(c['y'] - c['p']) for c in cells])
    R = max(float(dr.var(ddof=1)) / 2, 1e-8)
    P0 = max(float(((r - mu0) ** 2).mean()) - R, 1e-8)
    th = dict(mu0=mu0, P0=P0, R=R)
    for sc, q in SCHED.items():
        groups = []
        for c in cells:
            is_m, ev = schedule(c['cyc'], q, L)
            groups.append(dict(cyc=c['cyc'], r=c['y'] - c['p'], m=is_m, ev=ev))
        n_e = len(ELL_GRID)
        cyc = np.repeat(_pad(groups, 'cyc'), n_e, axis=0); res = np.repeat(_pad(groups, 'r'), n_e, axis=0)
        mm = np.repeat(_pad(groups, 'm') == 1, n_e, axis=0); ev = np.repeat(_pad(groups, 'ev') == 1, n_e, axis=0)
        ell = np.tile(ELL_GRID * L, len(groups))
        B, _ = kalman_batch(cyc, res, mm, mu0, P0, R, ell)
        e = np.where(ev, np.abs(res - B), np.nan)
        per = np.nanmean(e, axis=1).reshape(len(groups), n_e)          # (cell, ℓ)
        ok = ev.reshape(len(groups), n_e, -1)[:, 0].any(axis=1)
        j = int(np.argmin(per[ok].mean(axis=0)))
        th[f'ell_{sc}'] = float(ELL_GRID[j])
    return th


def tiers(cells, th, L):
    """MAE (và RMSE) theo cell ở M0 và ở mọi lịch đo, cho Kalman và hai đối chứng."""
    rows = []
    for c in cells:
        e0 = c['p'] - c['y']
        rows.append(dict(cell=c['cell'], tier='M0', n_meas=0, goc=np.abs(e0).mean(), kalman=np.abs(e0).mean(),
                         rmse_kalman=np.sqrt((e0 ** 2).mean())))
    for sc, q in SCHED.items():
        groups = []
        for c in cells:
            is_m, ev = schedule(c['cyc'], q, L)
            groups.append(dict(cyc=c['cyc'], r=c['y'] - c['p'], m=is_m, ev=ev))
        cyc, res, mm = _pad(groups, 'cyc'), _pad(groups, 'r'), _pad(groups, 'm') == 1
        ell = np.full(len(groups), th[f'ell_{sc}'] * L)
        B, _ = kalman_batch(cyc, res, mm, th['mu0'], th['P0'], th['R'], ell)
        for i, (c, g) in enumerate(zip(cells, groups)):
            ev = g['ev']
            if not ev.any():
                continue
            n = len(c['cyc']); prev = _last_measured(g['m'])
            have = prev >= 0
            est = dict(kalman=c['p'] + B[i, :n],
                       gan_nhat=c['p'] + np.where(have, g['r'][np.maximum(prev, 0)], 0.0),
                       giu_nguyen=np.where(have, c['y'][np.maximum(prev, 0)], c['p']))
            row = dict(cell=c['cell'], tier=sc, n_meas=int(g['m'].sum()), goc=np.abs(c['p'] - c['y'])[ev].mean())
            for m, p in est.items():
                d = (p - c['y'])[ev]
                row[m] = np.abs(d).mean(); row[f'rmse_{m}'] = np.sqrt((d ** 2).mean())
            rows.append(row)
    return rows


def split_cells(t, p):
    out = []
    for c in np.unique(t['cell']):
        i = np.flatnonzero(t['cell'] == c); i = i[np.argsort(t['cyc'][i], kind='stable')]
        out.append(dict(cell=c, cyc=t['cyc'][i], y=t['y'][i], p=p[i]))
    return out


def run_tiers(recs, best):
    Lof = {ds: life_scale(load_dataset('PINN4SOH/data', ds, clean='causal')) for ds in MAIN + NASA_DS}
    bt = best.set_index(['split', 'ds', 'f'])
    rows, params = [], []
    for key, rec in recs.items():
        split, ds, f, r, k = key
        om, w = bt.loc[(split, ds, f), ['omega', 'w']]
        L = Lof[ds]
        th = fit_params(split_cells(rec['val'], lai(rec['val'], om, w)), L)
        params.append(dict(split=split, ds=ds, f=f, r=r, k=k, L=L, **th))
        for row in tiers(split_cells(rec['test'], lai(rec['test'], om, w)), th, L):
            rows.append(dict(split=split, ds=ds, f=f, r=r, k=k, **row))
    return pd.DataFrame(rows), pd.DataFrame(params)


def summarize(T):
    """Theo cell (trung bình qua lần lặp / hạt giống), rồi theo (cách chia, bộ, mức nhãn, mức thông tin)."""
    cols = ['goc'] + METHODS + [f'rmse_{m}' for m in METHODS] + ['n_meas']
    pc = T.groupby(['split', 'ds', 'f', 'tier', 'cell'])[cols].mean().reset_index()
    rows = []
    for (split, ds, f, tier), g in pc.groupby(['split', 'ds', 'f', 'tier']):
        row = dict(split=split, ds=ds, f=f, tier=tier, n_cell=len(g), n_meas=g.n_meas.mean(), goc=g.goc.mean(),
                   **{m: g[m].mean() for m in METHODS}, **{f'rmse_{m}': g[f'rmse_{m}'].mean() for m in METHODS})
        if tier != 'M0' and split == 'cv':
            for ref in ('goc', 'gan_nhat', 'giu_nguyen'):
                d = (g.kalman - g[ref]).values
                row[f'better_{ref}'] = int((d < 0).sum())
                row[f'p_{ref}'] = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append(row)
    S = pd.DataFrame(rows)
    from analyze_e20 import holm
    main = (S.split == 'cv') & S.ds.isin(MAIN) & (S.tier != 'M0')
    for ref in ('goc', 'gan_nhat', 'giu_nguyen'):
        S.loc[main, f'p_holm_{ref}'] = S[main].groupby(['f', 'tier'])[f'p_{ref}'].transform(lambda p: holm(p.values))
    return S, pc


# --------------------------------------------------------------------------- M0: các mô hình trên NASA
M0_MODELS = ['mlp', 'cnn', 'lstm', 'transformer', 'gbdt', 'pinn', 'pinn3', 'pinn_lai', 'pinn_lai_t']


def m0_tables(recs, best):
    bt = best.set_index(['split', 'ds', 'f'])
    rows = []
    for (split, ds, f, r, k), rec in recs.items():
        if ds not in NASA_DS:
            continue
        t = rec['test']
        om, w = bt.loc[(split, ds, f), ['omega', 'w']]
        pr = dict(gbdt=t['g'], pinn=t['pinn'], pinn3=t['p3'], pinn_lai=lai(t, 0.5, 10), pinn_lai_t=lai(t, om, w))
        pr.update({m: t[m] for m in ['mlp'] + list(E30_MODELS) if m in t})
        for m, p in pr.items():
            for c in np.unique(t['cell']):
                i = t['cell'] == c; e = p[i] - t['y'][i]
                rows.append(dict(split=split, ds=ds, f=f, r=r, model=m, cell=c, mae=np.abs(e).mean(),
                                 rmse=np.sqrt((e ** 2).mean())))
    C = pd.DataFrame(rows)
    pc = C.groupby(['split', 'ds', 'f', 'model', 'cell'])[['mae', 'rmse']].mean().reset_index()
    tests = []
    for f in (0.3, 0.7):
        g = pc[(pc.split == 'cv') & (pc.f == f)]
        for m in M0_MODELS:
            a = g[(g.ds == 'NASA') & (g.model == m)].set_index('cell').mae
            for ds in ('NASAdt', 'NASAfc'):
                b = g[(g.ds == ds) & (g.model == m)].set_index('cell').mae
                j = a.index.intersection(b.index)
                if len(j) > 3:
                    d = b[j] - a[j]
                    tests.append(dict(kind=f'{ds}_vs_NASA', f=f, model=m, n=len(j), ratio=b[j].mean() / a[j].mean(),
                                      better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue))
        for ds in NASA_DS:
            h = g[g.ds == ds].pivot(index='cell', columns='model', values='mae')
            for m in [x for x in M0_MODELS if x in h and x != 'pinn_lai_t']:
                d = (h.pinn_lai_t - h[m]).dropna()
                tests.append(dict(kind=f'lai_t_vs_{m}', ds=ds, f=f, model=m, n=len(d),
                                  ratio=h.pinn_lai_t.mean() / h[m].mean(), better=int((d < 0).sum()),
                                  p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pc, pd.DataFrame(tests)


# --------------------------------------------------------------------------- hình
def figure(S, show_ds='NASAfc'):
    """MAE theo mức thông tin: bốn bộ chính và NASA (kiểm định chéo) ở 70 % nhãn, NASA-L với mức của SambaMixer."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, INK2, MUTED, panel_title, save
    apply_theme()
    panels = [('cv', ds, 0.7, ds) for ds in MAIN] + [('cv', show_ds, 0.7, 'NASA'), ('samba', show_ds, 0.7, 'NASA-L')]
    style = {'kalman': ('PINN-lai + Kalman', '#0b3d73', '-', 'o'), 'gan_nhat': ('gần nhất', '#2a78d6', '--', 's'),
             'giu_nguyen': ('giữ nguyên', '#eb6834', ':', '^')}
    xl = ['M0', 'q=0,2', 'q=0,1', 'q=0,05', 'M2']
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.1))
    for ax, (split, ds, f, title) in zip(axes.ravel(), panels):
        g = S[(S.split == split) & (S.ds == ds) & (S.f == f)].set_index('tier').reindex(TIERS)
        for m, (lab, col, ls, mk) in style.items():
            v = 100 * g[m].values.astype(float)
            ax.plot(range(len(TIERS)), v, color=col, ls=ls, marker=mk, ms=3.2, lw=1.3, label=lab)
        if split == 'samba':
            ax.axhline(100 * np.mean([v[0] for v in SAMBA.values()]), color=MUTED, lw=1.1, ls=(0, (4, 2)),
                       label='SambaMixer (dùng chính lần xả)')
        ax.set_yscale('log')
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if lo <= t <= hi])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks(range(len(TIERS))); ax.set_xticklabels(xl, fontsize=6.5)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        panel_title(ax, title, pad=4)
    for ax in axes[:, 0]:
        ax.set_ylabel('MAE theo cell (% SOH)')
    h, l = axes[1, 2].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=1.0, w_pad=0.8)
    save(fig, f'{OUT}/fig_e31_tiers')


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = build()
    best = tune(recs)
    best.to_csv(f'{OUT}/e31_tuned.csv', index=False)
    pc0, t0 = m0_tables(recs, best)
    pc0[pc0.split == 'samba'].to_csv(f'{OUT}/e31_m0_samba.csv', index=False)
    cv0 = pc0[pc0.split == 'cv'].groupby(['ds', 'f', 'model'])[['mae', 'rmse']].mean().reset_index()
    cv0.to_csv(f'{OUT}/e31_m0_cv.csv', index=False)
    t0.to_csv(f'{OUT}/e31_m0_tests.csv', index=False)
    T, params = run_tiers(recs, best)
    params.to_csv(f'{OUT}/e31_params.csv', index=False)
    S, pc = summarize(T)
    S.to_csv(f'{OUT}/e31_tiers.csv', index=False)
    pc[pc.split == 'samba'].to_csv(f'{OUT}/e31_tiers_samba.csv', index=False)
    with pd.option_context('display.width', 230, 'display.max_rows', 500, 'display.max_columns', 40):
        print(best.round(4).to_string(index=False))
        sb = pc0[pc0.split == 'samba'].assign(mae=lambda x: 100 * x.mae)
        print(sb.pivot_table(index=['ds', 'model'], columns='cell', values='mae').assign(
            mean=lambda d: d.mean(axis=1)).round(2))
        print(cv0.assign(mae=lambda x: 100 * x.mae).pivot_table(index=['ds', 'model'], columns='f', values='mae').round(2))
        print(t0.round(4).to_string(index=False))
        V = S.copy()
        for c in ['goc'] + METHODS:
            V[c] = (100 * V[c]).round(3)
        print(V.drop(columns=[c for c in V if c.startswith('rmse_')]).round(4).to_string(index=False))
        ps = pc[pc.split == 'samba'].assign(**{m: lambda x, m=m: 100 * x[m] for m in METHODS})
        print(ps.pivot_table(index=['ds', 'tier'], columns='cell', values='kalman').assign(
            mean=lambda d: d.mean(axis=1)).round(3))
        print(params.groupby(['split', 'ds'])[['mu0', 'P0', 'R'] + [f'ell_{s}' for s in SCHED]].median())
    figure(S)


if __name__ == '__main__':
    main()
