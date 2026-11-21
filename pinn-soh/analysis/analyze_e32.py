"""
E25 + E32: cập nhật ước lượng khi có vài lần đo dung lượng của chính cell, theo lượng thông tin đo được
(docs/de-cuong-E32.md).

    python analysis/analyze_e32.py    # -> results/e32_tuned.csv, e32_params.csv, e32_tiers.csv, e32_cells.csv,
                                      #    results/fig_e32_tiers.png|pdf

Ba mức thông tin về cell cần ước lượng:
    M0   không đo cell: PINN-lai tinh chỉnh, (ω, w) chọn trên validation như analysis/analyze_tuning.py
    M1   đo dung lượng ở chu kỳ đầu và mỗi qL chu kỳ (q20, q10, q05: q = 0,2; 0,1; 0,05; L = trung vị số chu kỳ của bộ);
         sai số tính trên các chu kỳ không được đo
    M2   đo mọi chu kỳ trước (dự báo một bước); sai số tính trên mọi chu kỳ trừ chu kỳ đầu
Ở M1, M2:
    kalman      bộ lọc Kalman của E25 (độ lệch theo cell là quá trình Ornstein–Uhlenbeck) trên dự đoán của PINN-lai; μ0,
                R, P0 ước lượng trên cell validation của chính fold, ℓ chọn trên lưới của E25 riêng cho từng lịch đo
    kalman_k    như kalman, ước lượng chỉ tin κ phần thay đổi của mô hình giữa hai chu kỳ; (ℓ, κ) chọn cùng lúc trên
                validation (thêm sau khi xem kết quả của kalman, nên là phân tích phụ)
    capnhat     mô hình học cập nhật từ lần đo gần nhất (E32, pinnsoh/update.py)
    gan_nhat    dịch dự đoán theo sai số của lần đo gần nhất
    giu_nguyen  lấy đúng SOH của lần đo gần nhất, không dùng mô hình
Dự đoán lấy từ lượt E20 với (ω, w) của results/e20_tuned.csv. So sánh ghép cặp theo cell (trung bình qua lần lặp),
Wilcoxon, p hiệu chỉnh Holm qua bốn bộ.
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
from pinnsoh.forecast import life_scale
from pinnsoh.postproc import by_cell, causal_trend

OUT = 'results'
MAIN = ['XJTU', 'TJU', 'MIT', 'HUST']
SRC = 'runs/E20'
FRACS = (0.1, 0.3, 0.5, 0.7)
REPEATS = (0, 1, 2)
OMEGAS, WINDOWS = (0.3, 0.4, 0.5, 0.6, 0.7), (5, 10, 20, 40)
SCHED = {'q20': 0.2, 'q10': 0.1, 'q05': 0.05, 'M2': None}        # M1: ba lịch đo thưa; M2: đo mọi chu kỳ trước
TIERS = ['M0'] + list(SCHED)
ELL_GRID = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 3.0, np.inf])  # như E25, theo đơn vị L
METHODS = ['kalman', 'kalman_k', 'capnhat', 'gan_nhat', 'giu_nguyen']
UPD_SRC = 'runs/E32'                                           # capnhat: mô hình học cập nhật (E32, pinnsoh/update.py)
KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0)                           # kalman_k: phần thay đổi của mô hình được tin


# --------------------------------------------------------------------------- đọc dự đoán
def _read(path):
    """Dự đoán test và validation của một lượt; mã cell là mảng object trỏ tới cùng một chuỗi nên nhẹ."""
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
    """Mọi đơn vị đánh giá: (bộ, mức nhãn, lần lặp, fold) -> các file của nó."""
    U = {}
    for ds, f, r, k in [(ds, f, r, k) for ds in MAIN for f in FRACS for r in REPEATS for k in range(5)]:
        s = 100 * r + k
        U[(ds, f, r, k)] = dict(P=[f'{SRC}/P_{ds}_pinn_semi_f{f}_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}')
                                   + '.json' for m in range(3)], G=f'{SRC}/gbdt_{ds}_f{f}_cv{r}-{k}of5.json')
    return U


def build():
    """Dự đoán test và validation của mọi đơn vị đủ file: y, p3 (trung bình 3 PINN-semi), g (GBDT)."""
    U = _units()
    need = lambda u: u['P'] + [u['G']]
    files = sorted({p for u in U.values() for p in need(u) if os.path.exists(p)})
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
            rec[part] = dict(cell=b['cell'], cyc=b['cyc'], y=b['y'], p3=np.mean([o[part]['p'] for o in P], axis=0),
                             g=G[part]['p'], pinn=b['p'])
        recs[key] = rec
    return recs


# --------------------------------------------------------------------------- PINN-lai tinh chỉnh
def _cell_mae(y, p, cell):
    return [float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)]


def tune(recs):
    """(ω, w) cho mỗi (bộ, mức nhãn): MAE theo cell trung bình trên validation của mọi lần lặp × fold; phải trùng
    kết quả đã chốt ở results/e20_tuned.csv."""
    acc = {}
    for (ds, f, r, k), rec in recs.items():
        v = rec['val']
        for om in OMEGAS:
            raw = om * v['p3'] + (1 - om) * v['g']
            for w in WINDOWS:
                acc.setdefault((ds, f, om, w), []).extend(_cell_mae(v['y'], by_cell(causal_trend, raw, v['cell'],
                                                                                    v['cyc'], w=w), v['cell']))
    V = pd.DataFrame([dict(ds=ds, f=f, omega=om, w=w, val=np.mean(e)) for (ds, f, om, w), e in acc.items()])
    best = V.loc[V.groupby(['ds', 'f']).val.idxmin()].reset_index(drop=True)
    old = pd.read_csv(f'{OUT}/e20_tuned.csv').set_index(['ds', 'f'])
    for _, b in best.iterrows():
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


def kalman_batch(cyc, r, meas, mu0, P0, R, ell, du=None, kappa=None):
    """Đúng bộ lọc của analyze_e25.kalman, chạy song song cho nhiều chuỗi: cyc, r, meas dạng (chuỗi, hàng), NaN ở hàng
    đệm; ell (chuỗi,) theo chu kỳ, có thể là inf. B[:, m] chỉ dùng các lần đo ở hàng < m.
    kalman_k: du = p_n − p_{n−1} (0 ở hàng đầu), kappa (chuỗi,); độ lệch nhận thêm −(1 − κ)·du ở mỗi bước, tức ước
    lượng chỉ tin κ phần thay đổi của mô hình. κ = 1 cho đúng từng bit bộ lọc gốc."""
    S, N = cyc.shape
    b = np.full(S, mu0, float); P = np.full(S, P0, float); last = cyc[:, 0].copy()
    inv = np.where(np.isfinite(ell), 1.0 / np.where(np.isfinite(ell), ell, 1.0), 0.0)
    drop = np.zeros(S) if kappa is None else 1.0 - np.asarray(kappa, float)
    B = np.full((S, N), np.nan); V = np.full((S, N), np.nan)
    for m in range(N):
        c = cyc[:, m]; ok = np.isfinite(c)
        a = np.where(ok, np.exp(-(np.where(ok, c, last) - last) * inv), 1.0)
        last = np.where(ok, c, last)
        b = mu0 + a * (b - mu0); P = a * a * P + (1 - a * a) * P0
        if du is not None:
            b = b - drop * np.where(ok, du[:, m], 0.0)
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


def _groups(cells, q, L):
    out = []
    for c in cells:
        is_m, ev = schedule(c['cyc'], q, L)
        out.append(dict(cyc=c['cyc'], r=c['y'] - c['p'], m=is_m, ev=ev, du=np.diff(c['p'], prepend=c['p'][0])))
    return out


def fit_params(cells, L):
    """μ0, R, P0 trên cell validation như E25; ℓ chọn trên lưới bằng MAE theo cell riêng cho từng lịch đo (bộ lọc chốt
    trước, κ = 1), và (ℓ, κ) chọn cùng lúc cho kalman_k."""
    r = np.concatenate([c['y'] - c['p'] for c in cells])
    mu0 = float(r.mean())
    dr = np.concatenate([np.diff(c['y'] - c['p']) for c in cells])
    R = max(float(dr.var(ddof=1)) / 2, 1e-8)
    P0 = max(float(((r - mu0) ** 2).mean()) - R, 1e-8)
    th = dict(mu0=mu0, P0=P0, R=R)
    grid = [(e, k) for k in KAPPAS for e in ELL_GRID]             # κ = 1 (bộ lọc chốt trước) ở cuối lưới
    n_g = len(grid)
    for sc, q in SCHED.items():
        groups = _groups(cells, q, L)
        rep = lambda key: np.repeat(_pad(groups, key), n_g, axis=0)
        cyc, res, du = rep('cyc'), rep('r'), rep('du')
        mm, ev = rep('m') == 1, rep('ev') == 1
        ell = np.tile([e * L for e, _ in grid], len(groups)); kap = np.tile([k for _, k in grid], len(groups))
        B, _ = kalman_batch(cyc, res, mm, mu0, P0, R, ell, du, kap)
        with np.errstate(invalid='ignore'):
            per = np.nanmean(np.where(ev, np.abs(res - B), np.nan), axis=1).reshape(len(groups), n_g)   # (cell, lưới)
        ok = ev.reshape(len(groups), n_g, -1)[:, 0].any(axis=1)
        v = per[ok].mean(axis=0)
        one = [i for i, (_, k) in enumerate(grid) if k == 1.0]
        j = one[int(np.argmin(v[one]))]
        th[f'ell_{sc}'] = float(grid[j][0])
        j = int(np.argmin(v))
        th[f'ellk_{sc}'], th[f'kappa_{sc}'] = float(grid[j][0]), float(grid[j][1])
    return th


def load_update(ds, f, r, k):
    """Ước lượng của mô hình cập nhật (E32) trên cell test của một đơn vị: {lịch đo: {cell: (chu kỳ, ước lượng)}}; None
    khi chưa có lượt E32 tương ứng."""
    from pinnsoh.update import cv_path
    p = cv_path(ds, f, r, k, UPD_SRC)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ids, out = d['cell_ids'], {}
    for sc, t in d['test_tiers'].items():
        cell, cyc, est = np.asarray(t['cell'], int), np.asarray(t['cycle'], float), np.asarray(t['est'], float)
        out[sc] = {ids[i]: (cyc[cell == i], est[cell == i]) for i in np.unique(cell)}
    return out


def tiers(cells, th, L, upd=None):
    """MAE (và RMSE) theo cell ở M0 và ở mọi lịch đo, cho Kalman, mô hình cập nhật (nếu có) và hai đối chứng."""
    rows = []
    for c in cells:
        e0 = c['p'] - c['y']
        rows.append(dict(cell=c['cell'], tier='M0', n_meas=0, goc=np.abs(e0).mean(), kalman=np.abs(e0).mean(),
                         rmse_kalman=np.sqrt((e0 ** 2).mean())))
    for sc, q in SCHED.items():
        groups = _groups(cells, q, L)
        cyc, res, mm, du = _pad(groups, 'cyc'), _pad(groups, 'r'), _pad(groups, 'm') == 1, _pad(groups, 'du')
        n_c = len(groups)
        B, _ = kalman_batch(cyc, res, mm, th['mu0'], th['P0'], th['R'], np.full(n_c, th[f'ell_{sc}'] * L))
        Bk, _ = kalman_batch(cyc, res, mm, th['mu0'], th['P0'], th['R'], np.full(n_c, th[f'ellk_{sc}'] * L), du,
                             np.full(n_c, th[f'kappa_{sc}']))
        for i, (c, g) in enumerate(zip(cells, groups)):
            ev = g['ev']
            if not ev.any():
                continue
            n = len(c['cyc']); prev = _last_measured(g['m'])
            have = prev >= 0
            est = dict(kalman=c['p'] + B[i, :n], kalman_k=c['p'] + Bk[i, :n],
                       gan_nhat=c['p'] + np.where(have, g['r'][np.maximum(prev, 0)], 0.0),
                       giu_nguyen=np.where(have, c['y'][np.maximum(prev, 0)], c['p']))
            if upd is not None:                    # cùng đúng các hàng được đánh giá (lịch đo pha 0, hàng có nhãn)
                uc, ue = upd[sc][c['cell']]
                assert np.array_equal(uc, c['cyc'][ev]), (c['cell'], sc)
                full = np.full(n, np.nan); full[ev] = ue
                est['capnhat'] = full
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
    Lof = {ds: life_scale(load_dataset('PINN4SOH/data', ds, clean='causal')) for ds in MAIN}
    bt = best.set_index(['ds', 'f'])
    rows, params = [], []
    for key, rec in recs.items():
        ds, f, r, k = key
        om, w = bt.loc[(ds, f), ['omega', 'w']]
        L = Lof[ds]
        th = fit_params(split_cells(rec['val'], lai(rec['val'], om, w)), L)
        params.append(dict(ds=ds, f=f, r=r, k=k, L=L, **th))
        for row in tiers(split_cells(rec['test'], lai(rec['test'], om, w)), th, L, load_update(*key)):
            rows.append(dict(ds=ds, f=f, r=r, k=k, **row))
    return pd.DataFrame(rows), pd.DataFrame(params)


PAIRS = ([(m, ref) for m in ('kalman', 'kalman_k') for ref in ('goc', 'gan_nhat', 'giu_nguyen')] + [('kalman_k', 'kalman')]
         + [('capnhat', ref) for ref in ('goc', 'kalman', 'kalman_k', 'gan_nhat', 'giu_nguyen')])


def summarize(T):
    """Theo cell (trung bình qua lần lặp), rồi theo (bộ, mức nhãn, mức thông tin)."""
    cols = ['goc'] + METHODS + [f'rmse_{m}' for m in METHODS] + ['n_meas']
    T = T.reindex(columns=list(T.columns) + [c for c in cols if c not in T.columns])   # chưa có lượt E32: cột NaN
    pc = T.groupby(['ds', 'f', 'tier', 'cell'])[cols].mean().reset_index()
    rows = []
    for (ds, f, tier), g in pc.groupby(['ds', 'f', 'tier']):
        row = dict(ds=ds, f=f, tier=tier, n_cell=len(g), n_meas=g.n_meas.mean(), goc=g.goc.mean(),
                   **{m: g[m].mean() for m in METHODS}, **{f'rmse_{m}': g[f'rmse_{m}'].mean() for m in METHODS})
        if tier != 'M0':
            for m, ref in PAIRS:
                d = (g[m] - g[ref]).dropna().values
                row[f'better_{m}_{ref}'] = int((d < 0).sum())
                row[f'p_{m}_{ref}'] = (stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0) if len(d) > 1 else np.nan
        rows.append(row)
    S = pd.DataFrame(rows)
    from analyze_e20 import holm
    up = S.tier != 'M0'
    for m, ref in PAIRS:
        S.loc[up, f'p_holm_{m}_{ref}'] = S[up].groupby(['f', 'tier'])[f'p_{m}_{ref}'].transform(lambda p: holm(p.values))
    return S, pc


# --------------------------------------------------------------------------- hình
def figure(S=None, f=0.7):
    """MAE theo mức thông tin trên bốn bộ, ở mức nhãn f."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme()
    S = pd.read_csv(f'{OUT}/e32_tiers.csv') if S is None else S
    style = {'kalman': ('PINN-lai + Kalman', '#0b3d73', '-', 'o'), 'kalman_k': ('PINN-lai + Kalman, κ', '#1baf7a', '-', 'D'),
             'capnhat': ('mô hình cập nhật', '#6250d6', '-', 'v'),
             'gan_nhat': ('gần nhất', '#2a78d6', '--', 's'), 'giu_nguyen': ('giữ nguyên', '#eb6834', ':', '^')}
    xl = ['M0', '0,2L', '0,1L', '0,05L', 'M2']
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2))
    for ax, ds in zip(axes.ravel(), MAIN):
        g = S[(S.ds == ds) & (S.f == f)].set_index('tier').reindex(TIERS)
        for m, (lab, col, ls, mk) in style.items():
            ax.plot(range(len(TIERS)), 100 * g[m].values.astype(float), color=col, ls=ls, marker=mk, ms=3.2, lw=1.3,
                    label=lab)
        ax.set_yscale('log')
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if lo <= t <= hi])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks(range(len(TIERS))); ax.set_xticklabels(xl, fontsize=6.5)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        panel_title(ax, ds, pad=4)
    for ax in axes[:, 0]:
        ax.set_ylabel('MAE theo cell (% SOH)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.0, w_pad=0.8)
    save(fig, f'{OUT}/fig_e32_tiers')


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = build()
    best = tune(recs)
    best.to_csv(f'{OUT}/e32_tuned.csv', index=False)
    T, params = run_tiers(recs, best)
    params.to_csv(f'{OUT}/e32_params.csv', index=False)
    S, pc = summarize(T)
    S.to_csv(f'{OUT}/e32_tiers.csv', index=False)
    pc.to_csv(f'{OUT}/e32_cells.csv', index=False)
    with pd.option_context('display.width', 230, 'display.max_rows', 500, 'display.max_columns', 40):
        print(best.round(4).to_string(index=False))
        V = S.copy()
        for c in ['goc'] + METHODS:
            V[c] = (100 * V[c]).round(3)
        print(V.drop(columns=[c for c in V if c.startswith('rmse_')]).round(4).to_string(index=False))
        print(params.groupby('ds')[['mu0', 'P0', 'R'] + [f'ell_{s}' for s in SCHED]].median())
        print(params.groupby('ds')[[f'kappa_{s}' for s in SCHED] + [f'ellk_{s}' for s in SCHED]].median())
    figure(S)


if __name__ == '__main__':
    main()
