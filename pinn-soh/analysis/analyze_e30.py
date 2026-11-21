"""
E30: CNN, LSTM, Transformer so với các mô hình của đồ án; bảng chỉ số đầy đủ (MAE, RMSE, MAPE, R²) cho mọi mô hình chính.

    python analysis/analyze_e30.py     # -> results/e30_metrics.csv, e30_tests.csv

Bốn bộ chính, 3 lần lặp × 5 fold × 4 mức nhãn, cùng fold và hạt giống với E20:
    mlp, gbdt, pinn           MLP, GBDT, PINN-semi (thành viên 0) — lượt E13 / E16 qua E20
    cnn, lstm, transformer    E30, độ dài cửa sổ chọn ở E30dev (pinnsoh.experiments.E30_L)
    pinn_lai                  PINN-lai tinh chỉnh: (ω, w) chọn trên validation (results/e20_tuned.csv)
    final                     PINN-lai tinh chỉnh, dùng hay không dùng lịch sử chọn trên validation (results/e29_select.csv)
Chỉ số: MAE_cell, RMSE_cell (mỗi cell trung bình qua lần lặp rồi trung bình theo cell), MAPE (gộp mọi chu kỳ của một lần
lặp), R² (gộp 5 fold của một lần lặp, bỏ 10 chu kỳ đầu mỗi cell, như Mục RMSE và R² của báo cáo); MAPE và R² trung bình qua
lần lặp. Kiểm định: Wilcoxon ghép cặp theo cell (trung bình 3 lần lặp), p hiệu chỉnh Holm qua bốn bộ.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.experiments import E30_MODELS, E30_L
from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.runcache import cached_map
from pinnsoh.seqmodels import cv_path

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
MODELS = ['mlp', 'cnn', 'lstm', 'transformer', 'gbdt', 'pinn', 'pinn_lai', 'final']
REFS = ['mlp', 'gbdt', 'pinn', 'pinn_lai', 'final']


def _align(q, cell, cyc):
    """Dự đoán của một lượt khác theo đúng thứ tự (cell, chu kỳ) của bản ghi E20."""
    if np.array_equal(q['cell'], cell) and np.array_equal(q['cyc'], cyc):
        return q['p']
    s = pd.Series(q['p'], index=pd.MultiIndex.from_arrays([q['cell'], q['cyc']]))
    out = s.reindex(pd.MultiIndex.from_arrays([cell, cyc])).values
    assert np.isfinite(out).all()
    return out


def build():
    import analyze_e20 as A
    from analyze_e29 import collect
    P, _ = A.build()
    T = pd.read_csv(f'{OUT}/e20_tuned.csv').set_index(['ds', 'f'])
    S = pd.read_csv(f'{OUT}/e29_select.csv')
    sel = S[S.kind == 'pinn_lai_v2'].set_index(['dataset', 'f'])
    G = collect()
    hp = sorted({p for g in G.values() for v in g.values() for p in (v if isinstance(v, list) else [v])
                 if p and os.path.exists(p)})
    hist = dict(zip(hp, cached_map(hp, A._both, 'e29_preds_v1')))
    sp = [cv_path(m, E30_L[m], ds, f, r, k) for m in E30_MODELS for (ds, f, r, k) in P]
    sp = [p for p in sp if os.path.exists(p)]
    seq = dict(zip(sp, cached_map(sp, A._both, 'e30_preds_v1')))
    recs = {}
    for key, rec in P.items():
        ds, f, r, k = key
        t = rec['test']; y, cell, cyc = t['y'], t['cell'], t['cyc']; q = t['pred']
        pr = dict(mlp=q['mlp'], gbdt=q['gbdt'], pinn=q['pinn'])
        om, w = float(T.loc[(ds, f), 'omega']), int(T.loc[(ds, f), 'w'])
        pr['pinn_lai'] = by_cell(causal_trend, om * q['pinn3'] + (1 - om) * q['gbdt'], cell, cyc, w=w)
        s = sel.loc[(ds, f)]
        if s.src == 'hist':
            g = G[key]
            p3h = np.mean([_align(hist[p]['test'], cell, cyc) for p in g['H']], axis=0)
            gs = _align(hist[g['G']]['test'], cell, cyc) if 'G' in g else q['gbdt']
            pr['final'] = by_cell(causal_trend, s.om * p3h + (1 - s.om) * gs, cell, cyc, w=int(s.w))
        else:
            pr['final'] = by_cell(causal_trend, s.om * q['pinn3'] + (1 - s.om) * q['gbdt'], cell, cyc, w=int(s.w))
        for m in E30_MODELS:
            p = cv_path(m, E30_L[m], ds, f, r, k)
            if p in seq:
                pr[m] = _align(seq[p]['test'], cell, cyc)
        recs[key] = dict(y=y, cell=cell, cyc=cyc, pred=pr)
    return recs


def metrics(recs):
    cell_rows, acc = [], {}
    for (ds, f, r, k), t in recs.items():
        y, cell = t['y'], t['cell']; keep = t['cyc'] >= 10
        for m, p in t['pred'].items():
            e = p - y
            for c in np.unique(cell):
                i = cell == c
                cell_rows.append(dict(dataset=ds, f=f, r=r, model=m, cell=c, mae=float(np.abs(e[i]).mean()),
                                      rmse=float(np.sqrt((e[i] ** 2).mean()))))
            a = acc.setdefault((ds, f, r, m), dict(folds=0, n=0, ape=0.0, n10=0, sse=0.0, sy=0.0, syy=0.0))
            a['folds'] += 1; a['n'] += len(e); a['ape'] += float(np.sum(np.abs(e) / y))
            a['n10'] += int(keep.sum()); a['sse'] += float((e[keep] ** 2).sum())
            a['sy'] += float(y[keep].sum()); a['syy'] += float((y[keep] ** 2).sum())
    C = pd.DataFrame(cell_rows)
    pc = C.groupby(['dataset', 'f', 'model', 'cell'])[['mae', 'rmse']].mean().reset_index()
    M = pc.groupby(['dataset', 'f', 'model']).agg(MAE_cell=('mae', 'mean'), RMSE_cell=('rmse', 'mean'),
                                                   n_cells=('cell', 'size')).reset_index()
    R = pd.DataFrame([dict(dataset=ds, f=f, r=r, model=m, MAPE=a['ape'] / a['n'],
                           R2=1 - a['sse'] / (a['syy'] - a['sy'] ** 2 / a['n10']))
                      for (ds, f, r, m), a in acc.items() if a['folds'] == 5])
    R = R.groupby(['dataset', 'f', 'model'])[['MAPE', 'R2']].mean().reset_index()
    return M.merge(R, on=['dataset', 'f', 'model'], how='left'), pc


def tests(pc):
    out = []
    for ds in DS:
        for f in sorted(pc.f.unique()):
            g = pc[(pc.dataset == ds) & (pc.f == f)].pivot(index='cell', columns='model', values='mae')
            for m in [x for x in E30_MODELS if x in g] + ['final']:
                for ref in REFS:
                    if ref == m or ref not in g or m not in g:
                        continue
                    d = (g[m] - g[ref]).dropna()
                    out.append(dict(dataset=ds, f=f, model=m, ref=ref, n_cells=len(d), ratio=g[m].mean() / g[ref].mean(),
                                    better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    T = pd.DataFrame(out)
    if len(T):                         # Holm qua bốn bộ cho mỗi (mô hình, đối chứng, mức nhãn), như Mục thống kê
        from analyze_e20 import holm
        T['p_holm'] = T.groupby(['model', 'ref', 'f']).p.transform(lambda p: holm(p.values))
    return T


def main():
    recs = build()
    M, pc = metrics(recs)
    os.makedirs(OUT, exist_ok=True)
    M.to_csv(f'{OUT}/e30_metrics.csv', index=False)
    Tt = tests(pc)
    Tt.to_csv(f'{OUT}/e30_tests.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 400):
        V = M.assign(MAE=1e3 * M.MAE_cell).pivot_table(index=['dataset', 'f'], columns='model', values='MAE')
        print(V.reindex(columns=[m for m in MODELS if m in V.columns]).round(2))
        V = M.pivot_table(index=['dataset', 'f'], columns='model', values='R2')
        print(V.reindex(columns=[m for m in MODELS if m in V.columns]).round(3))
        if len(Tt):
            print(Tt.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
