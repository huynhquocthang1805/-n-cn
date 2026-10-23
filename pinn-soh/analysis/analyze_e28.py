"""
E28: ngân sách đo dung lượng — đo thưa trên mọi cell train so với đo đầy đủ trên một phần cell (E20).

    python analysis/analyze_e28.py      # -> results/e28_summary.csv, e28_cells.csv, e28_tests.csv, fig_e28_budget

Hai cách dùng cùng loại nhãn:
    theo cell   (E20) f = 10, 30, 50, 70 % số cell có nhãn ở mọi chu kỳ, các cell train còn lại không nhãn
    đo thưa     (E28) mọi cell train (70 % số cell) được đo ở chu kỳ đầu và sau mỗi q·L chu kỳ, q = 0,5; 0,2; 0,1; 0,05
Trục ngang chung: phần trăm số chu kỳ của các cell train có nhãn dung lượng (đếm trên dữ liệu sau lọc).
Mô hình ở chế độ đo thưa:
    pinn_lai    ½ (trung bình 3 PINN-semi học trên lần đo, vật lý trên mọi hàng) + ½ GBDT học nhãn nội suy, qua bộ lọc
                quỹ đạo w = 10 — đúng công thức PINN-lai của E20
    pinn3       trung bình 3 PINN-semi
    gbdt_meas   GBDT chỉ trên các hàng được đo;  gbdt_interp  GBDT trên nhãn nội suy
    mlp         MLP chỉ trên các hàng được đo;    pinn_interp  một PINN-semi học nhãn nội suy (q = 0,1; 0,05)
Sai số: MAE theo cell trên test, trung bình qua 3 lần lặp; Wilcoxon ghép cặp theo cell so với PINN-lai và GBDT của E20
ở 30 % và 70 % nhãn (cùng fold, cùng cell test).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import collections, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.experiments import E28, E28_QS
from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.runcache import cached_map

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
REPEATS = (0, 1, 2)
W = 10


def _pred(path, part='test'):
    d = json.load(open(path))
    tp = d[f'{part}_predictions']
    ids = tp['cell_ids']
    return dict(y=np.asarray(tp['y'], float), p=np.asarray(tp['p'], float),
                cell=np.asarray([ids[i] for i in tp['cell']]), cyc=np.asarray(tp['cycle'], float))


def _both(path):
    d = json.load(open(path))
    out = {'test': _pred(path)}
    if 'val_predictions' in d:
        out['val'] = _pred(path, 'val')
    if 'n_meas' in d:
        out['n_meas'], out['n_rows'] = d['n_meas'], d['n_rows']
    return out


def collect():
    G = collections.defaultdict(dict)
    for j in E28():
        a = j.args
        if j.kind == 'train':
            key = (a['dataset'], a['meas_every'], a['repeat'], a['fold'])
            name = ('pinn_interp' if a.get('meas_interp') else 'pinn') if a['model'] == 'pinn_semi' else 'mlp'
            if name == 'pinn':
                G[key].setdefault('pinn', [None] * 3)[a['member']] = j.out
            else:
                G[key][name] = j.out
        else:
            G[(a['ds'], a['q'], a['r'], a['k'])][f'gbdt_{a["mode"]}'] = j.out
    return G


def cell_errors(y, p, cell):
    return {c: float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)}


def build_sparse():
    G = collect()
    paths = sorted({p for g in G.values() for v in g.values() for p in (v if isinstance(v, list) else [v])
                    if p and os.path.exists(p)})
    data = dict(zip(paths, cached_map(paths, _both, 'e28_preds_v1')))
    rows, rowfrac, val = [], [], []
    for (ds, q, r, k), g in sorted(G.items()):
        need = g['pinn'] + [g['gbdt_interp'], g['gbdt_meas'], g['mlp']]
        if not all(p and p in data for p in need):
            continue
        rowfrac.append(dict(dataset=ds, q=q, r=r, k=k, frac_rows=data[g['gbdt_meas']]['n_meas'] / data[g['gbdt_meas']]['n_rows']))
        for part in ('test', 'val'):
            P = [data[p][part] for p in g['pinn']]
            y, cell, cyc = P[0]['y'], P[0]['cell'], P[0]['cyc']
            gi, gm = data[g['gbdt_interp']][part]['p'], data[g['gbdt_meas']][part]['p']
            pinn3 = np.mean([m['p'] for m in P], axis=0)
            pred = dict(pinn=P[0]['p'], pinn3=pinn3, gbdt_interp=gi, gbdt_meas=gm,
                        pinn_lai=by_cell(causal_trend, 0.5 * pinn3 + 0.5 * gi, cell, cyc, w=W),
                        pinn_lai_m=by_cell(causal_trend, 0.5 * pinn3 + 0.5 * gm, cell, cyc, w=W))
            if part == 'test':
                pred['mlp'] = data[g['mlp']]['test']['p']
                if g.get('pinn_interp') in data:
                    pred['pinn_interp'] = data[g['pinn_interp']]['test']['p']
            for m, p in pred.items():
                for c, e in cell_errors(y, p, cell).items():
                    (rows if part == 'test' else val).append(dict(dataset=ds, regime=f'q{q}', q=q, r=r, k=k, model=m,
                                                                  cell=c, mae=e))
    return pd.DataFrame(rows), pd.DataFrame(val), pd.DataFrame(rowfrac)


def build_bycell():
    """PINN-lai, GBDT, MLP của E20 ở 4 mức nhãn theo cell, và phần trăm số chu kỳ train có nhãn ở mỗi fold."""
    import analyze_e20 as A
    from pinnsoh.baselines import cells_of
    from pinnsoh.data import cv_split, choose_labelled
    P, _ = A.build()
    rows, rowfrac = [], []
    for (ds, f, r, k), rec in P.items():
        t = rec['test']
        for m in ('pinn_lai', 'gbdt', 'mlp', 'pinn3'):
            for c, e in cell_errors(t['y'], t['pred'][m], t['cell']).items():
                rows.append(dict(dataset=ds, regime=f'f{f}', frac=f, r=r, k=k, model=m, cell=c, mae=e))
    for ds in DS:
        cells = cells_of(ds); by = {c.cid: c for c in cells}
        n = {c.cid: int(np.isfinite(c.soh).sum()) for c in cells}
        for f, r, k in [(f, r, k) for f in FRACS for r in REPEATS for k in range(5)]:
            sp = cv_split(cells, k, r, 5, 0.10)
            lab, unl = choose_labelled(sp['train'], f, len(cells), 100 * r + k, by)
            rowfrac.append(dict(dataset=ds, frac=f, r=r, k=k, frac_rows=sum(n[i] for i in lab) / sum(n[i] for i in lab + unl)))
    return pd.DataFrame(rows), pd.DataFrame(rowfrac)


def per_cell_mean(C):
    """MAE theo cell trung bình qua lần lặp (mỗi cell nằm trong test đúng một lần mỗi lần lặp)."""
    return C.groupby(['dataset', 'regime', 'model', 'cell']).mae.mean().reset_index()


def same_repeats(Cs, Cb):
    """Chỉ giữ các (bộ, lần lặp) có ở cả hai phía để so ghép cặp trên cùng cell test, cùng số lần lặp."""
    keep = set(map(tuple, Cs[['dataset', 'r']].drop_duplicates().values))
    return Cb[[(ds, r) in keep for ds, r in zip(Cb.dataset, Cb.r)]]


def tests(Cs, Cb):
    Cb = same_repeats(Cs, Cb)
    a, b = per_cell_mean(Cs), per_cell_mean(Cb)
    out = []
    for ds in DS:
        for q in E28_QS:
            for m in ('pinn_lai', 'gbdt_interp', 'gbdt_meas', 'mlp', 'pinn3'):
                x = a[(a.dataset == ds) & (a.regime == f'q{q}') & (a.model == m)].set_index('cell').mae
                if not len(x):
                    continue
                for ref_m, f in (('pinn_lai', 0.3), ('pinn_lai', 0.7), ('gbdt', 0.7), ('gbdt', 0.3)):
                    yv = b[(b.dataset == ds) & (b.regime == f'f{f}') & (b.model == ref_m)].set_index('cell').mae
                    cells = sorted(set(x.index) & set(yv.index))
                    d = x.reindex(cells).values - yv.reindex(cells).values
                    out.append(dict(dataset=ds, q=q, model=m, ref=f'{ref_m}@{int(100 * f)}', n_cells=len(cells),
                                    mae=x.reindex(cells).mean(), mae_ref=yv.reindex(cells).mean(),
                                    ratio=x.reindex(cells).mean() / yv.reindex(cells).mean(), better=int((d < 0).sum()),
                                    p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pd.DataFrame(out)


def summary(Cs, Cb, Fs, Fb):
    Cb = same_repeats(Cs, Cb)
    a = per_cell_mean(Cs).groupby(['dataset', 'regime', 'model']).mae.mean().reset_index()
    b = per_cell_mean(Cb).groupby(['dataset', 'regime', 'model']).mae.mean().reset_index()
    fs = Fs.groupby(['dataset', 'q']).frac_rows.mean()
    fb = Fb.groupby(['dataset', 'frac']).frac_rows.mean()
    a['kind'] = 'do_thua'; b['kind'] = 'theo_cell'
    a['frac_rows'] = [fs[(r.dataset, float(r.regime[1:]))] for r in a.itertuples()]
    b['frac_rows'] = [fb[(r.dataset, float(r.regime[1:]))] for r in b.itertuples()]
    return pd.concat([b, a], ignore_index=True)


def figure(S):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    spec = [('theo_cell', 'pinn_lai', '#9cc3e6', '--', 's', 'PINN-lai, nhãn theo cell'),
            ('theo_cell', 'gbdt', '#f2c6a0', '--', 's', 'GBDT, nhãn theo cell'),
            ('do_thua', 'mlp', '#b9c2c9', '-', 'o', 'MLP, đo thưa'),
            ('do_thua', 'gbdt_interp', '#eb6834', '-', 'o', 'GBDT, đo thưa + nội suy'),
            ('do_thua', 'pinn_lai', '#0b3d73', '-', 'o', 'PINN-lai, đo thưa')]
    for ax, ds in zip(axes, DS):
        for kind, m, col, ls, mk, lab in spec:
            g = S[(S.dataset == ds) & (S.kind == kind) & (S.model == m)].sort_values('frac_rows')
            ax.plot(100 * g.frac_rows, 1e3 * g.mae, color=col, ls=ls, marker=mk, ms=3, lw=1.3, label=lab)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks([0.3, 1, 3, 10, 30, 100]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in (1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80) if lo <= t <= hi])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        panel_title(ax, ds, pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=6.8)
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.text(0.5, 0.17, '% số chu kỳ của cell train có đo dung lượng', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e28_budget')


def deploy(qs=E28_QS):
    """Đo thưa cả khi dùng: cell test cũng được đo dung lượng theo cùng lịch q, độ lệch theo cell được cập nhật bằng bộ
    lọc Kalman của E25 (tham số trên cell validation, độ dài tương quan chọn trên lưới bằng MAE validation). Sai số tính
    trên các chu kỳ KHÔNG được đo của cell test. Hai mô hình nền: PINN-lai học bằng nhãn thưa (E28) và PINN-lai học với
    70 % cell có nhãn đầy đủ (E20) — cùng lịch đo khi dùng."""
    import analyze_e25 as K
    from pinnsoh.baselines import cells_of
    from pinnsoh.forecast import life_scale
    G = collect()
    rows = []
    for ds in DS:
        L = life_scale(cells_of(ds))
        for q in qs:
            for r in REPEATS:
                for k in range(5):
                    g = G.get((ds, q, r, k))
                    if not g or not all(p and os.path.exists(p) for p in g['pinn'] + [g['gbdt_interp']]):
                        continue
                    base = {}
                    for part in ('test', 'val'):
                        P = [_pred(p, part) for p in g['pinn']]; Gi = _pred(g['gbdt_interp'], part)
                        raw = 0.5 * np.mean([m['p'] for m in P], axis=0) + 0.5 * Gi['p']
                        base[('sparse', part)] = pd.DataFrame(dict(cell=P[0]['cell'], cyc=P[0]['cyc'], y=P[0]['y'],
                                                                   p=by_cell(causal_trend, raw, P[0]['cell'], P[0]['cyc'], w=W)))
                    for part in ('test', 'val'):
                        base[('full70', part)] = K.pinn_lai(ds, 0.7, r, k, part)
                    for src in ('sparse', 'full70'):
                        Te, Va = base[(src, 'test')], base[(src, 'val')]
                        if Te is None or Va is None:
                            continue
                        Va = Va.sort_values(['cell', 'cyc'], kind='stable'); Te = Te.sort_values(['cell', 'cyc'], kind='stable')
                        Va = Va.assign(r=Va.y - Va.p)
                        mu0 = float(Va.r.mean()); R = max(float(Va.groupby('cell').r.diff().dropna().var()) / 2, 1e-8)
                        P0 = max(float(((Va.r - mu0) ** 2).mean()) - R, 1e-8)
                        best = None
                        for ell in K.ELL_GRID:
                            err = []
                            for c, gg in Va.groupby('cell', sort=False):
                                cyc, res = gg.cyc.values, gg.r.values
                                B, _, is_m = K.kalman(cyc, res, K.measured_rows(cyc, q, L), mu0, P0, R, ell * L)
                                if (~is_m).any():
                                    err.append(np.abs(res - B)[~is_m].mean())
                            v = float(np.mean(err))
                            if best is None or v < best[0]:
                                best = (v, ell)
                        for c, gg in Te.groupby('cell', sort=False):
                            cyc, res = gg.cyc.values, (gg.y - gg.p).values
                            meas = K.measured_rows(cyc, q, L)
                            B, _, is_m = K.kalman(cyc, res, meas, mu0, P0, R, best[1] * L)
                            ev = ~is_m
                            if ev.any():
                                rows.append(dict(dataset=ds, q=q, r=r, k=k, src=src, cell=c, n_meas=int(is_m.sum()),
                                                 n_rows=len(cyc), goc=float(np.abs(res)[ev].mean()),
                                                 kalman=float(np.abs(res - B)[ev].mean())))
    return pd.DataFrame(rows)


def main():
    Cs, Vs, Fs = build_sparse()
    Cb, Fb = build_bycell()
    os.makedirs(OUT, exist_ok=True)
    pd.concat([Cs.assign(kind='do_thua'), Cb.assign(kind='theo_cell')]).to_csv(f'{OUT}/e28_cells.csv.gz', index=False)
    S = summary(Cs, Cb, Fs, Fb); S.to_csv(f'{OUT}/e28_summary.csv', index=False)
    T = tests(Cs, Cb); T.to_csv(f'{OUT}/e28_tests.csv', index=False)
    V = Vs.groupby(['dataset', 'regime', 'model']).mae.mean().unstack('model')
    V.to_csv(f'{OUT}/e28_val.csv')
    with pd.option_context('display.width', 220, 'display.max_rows', 300):
        p = S.assign(mae=1e3 * S.mae, pr=100 * S.frac_rows).pivot_table(index=['dataset', 'kind', 'regime', 'pr'],
                                                                          columns='model', values='mae')
        print(p.round(2).to_string())
        print(T[T.model == 'pinn_lai'].round(4).to_string(index=False))
        print((1e3 * V).round(2).to_string())
    D = deploy()
    D.to_csv(f'{OUT}/e28_deploy.csv.gz', index=False)
    Dm = D.groupby(['dataset', 'q', 'src', 'cell'])[['goc', 'kalman', 'n_meas', 'n_rows']].mean().reset_index()
    with pd.option_context('display.width', 200):
        print((Dm.groupby(['dataset', 'q', 'src'])[['goc', 'kalman']].mean() * 1e3).round(2).unstack('src').to_string())
    figure(S)


if __name__ == '__main__':
    _sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
    main()
