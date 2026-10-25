"""
E29: mô hình dùng lịch sử của cell — đánh giá cuối (tai-lieu/de-cuong-E29.md).

    python analysis/analyze_e29.py     # -> results/e29_summary.csv, e29_tests.csv, e29_tuned.csv, e29_select.csv,
                                       #    e29_val.csv, fig_e29_hist

Cùng fold, hạt giống và mức nhãn với E20 (3 lần lặp × 5 fold × 4 mức nhãn). Lượt mới chỉ ở các bộ có dạng lịch sử được
chọn (XJTU, TJU, HUST); MIT không có dạng nào được chọn nên chỉ có các mô hình của E20. Mô hình (hậu tố _h: có lịch sử):
    pinn, pinn_h          PINN-semi, thành viên 0
    mlp, mlp_h            MLP, thành viên 0 (mlp_h cùng dạng lịch sử với pinn_h)
    gbdt, gbdt_h          GBDT; gbdt_h chỉ ở XJTU và HUST — ở TJU, GBDT của bộ đã chọn là gbdt
    pinn_lai              E20: ½ trung bình 3 PINN-semi + ½ GBDT, bộ lọc xu thế w = 10
    pinn_lai_h            cùng công thức với 3 PINN-semi có lịch sử và GBDT đã chọn cho bộ
    pinn_lai_v2, _h_v2    bản tinh chỉnh: (ω, w) chọn trên validation như analyze_tuning.py, riêng cho từng (bộ, mức nhãn)
    pinn_lai_sel, _v2     phân tích thêm sau khi xem test: với mỗi (bộ, mức nhãn), chọn có hay không lịch sử bằng MAE validation
                          của chính E29 (3 lần lặp × 5 fold); gbdt_best chọn GBDT có / không lịch sử theo cùng cách
    gbdt_best_f           đối chứng chặt hơn: GBDT có / không lịch sử, không lọc hoặc qua bộ lọc xu thế w ∈ {5, 10, 20, 40},
                          chọn trên validation
Sai số: MAE theo cell trên test, trung bình 3 lần lặp; Wilcoxon ghép cặp theo cell.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import collections, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.experiments import E29, E29_SPEC_PINN, E29_SPEC_GBDT
from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.runcache import cached_map
from analyze_tuning import OMEGAS, WINDOWS

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
W = 10


def collect():
    """{(bộ, mức nhãn, lần lặp, fold): {'H': [3 đường dẫn], 'HM': đường dẫn, 'G': đường dẫn | None}}"""
    G = collections.defaultdict(dict)
    for j in E29():
        a = j.args
        if j.kind == 'train':
            key = (a['dataset'], a['label_frac'], a['repeat'], a['fold'])
            if a['model'] == 'pinn_semi':
                G[key].setdefault('H', [None] * 3)[a['member']] = j.out
            else:
                G[key]['HM'] = j.out
        else:
            G[(a['ds'], a['frac'], a['r'], a['k'])]['G'] = j.out
    return G


def cell_errors(y, p, cell):
    return {c: float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)}


def build():
    import analyze_e20 as A
    P, _ = A.build()
    G = collect()
    paths = sorted({p for g in G.values() for v in g.values() for p in (v if isinstance(v, list) else [v])
                    if p and os.path.exists(p)})
    data = dict(zip(paths, cached_map(paths, A._both, 'e29_preds_v1')))
    rows, grid, vrows, frows = [], [], [], []
    for key in sorted(P):
        ds, f, r, k = key
        g = G.get(key)
        hist = ds in E29_SPEC_PINN
        if hist and (g is None or not all(p in data for p in g['H'] + [g['HM']] + ([g['G']] if 'G' in g else []))):
            continue
        e20 = P[key]
        for part in ('test', 'val'):
            base = e20[part]
            y, cell, cyc = base['y'], base['cell'], base['cyc']
            pred = dict(pinn=base['pred']['pinn'], pinn3=base['pred']['pinn3'], gbdt=base['pred']['gbdt'],
                        pinn_lai=base['pred']['pinn_lai'])
            src = dict(e20=(base['pred']['pinn3'], base['pred']['gbdt']))
            gsrc = dict(gbdt=base['pred']['gbdt'])
            if hist:
                H = [data[p][part] for p in g['H']]
                for o in H:
                    assert np.array_equal(o['cell'], cell) and np.array_equal(o['cyc'], cyc), key
                pinn3_h = np.mean([o['p'] for o in H], axis=0)
                gsel = data[g['G']][part]['p'] if 'G' in g else base['pred']['gbdt']
                pred.update(pinn_h=H[0]['p'], pinn3_h=pinn3_h, gbdt_sel=gsel,
                            pinn_lai_h=by_cell(causal_trend, 0.5 * pinn3_h + 0.5 * gsel, cell, cyc, w=W))
                src['hist'] = (pinn3_h, gsel)
                if 'G' in g:
                    gsrc['gbdt_sel'] = gsel
            if part == 'test':
                pred['mlp'] = base['pred']['mlp']
                if hist:
                    hm = data[g['HM']]['test']
                    assert np.array_equal(hm['cell'], cell) and np.array_equal(hm['cyc'], cyc), key
                    pred['mlp_h'] = hm['p']
                for m, p in pred.items():
                    for c, e in cell_errors(y, p, cell).items():
                        rows.append(dict(dataset=ds, f=f, r=r, k=k, model=m, cell=c, mae=e))
            else:
                for m, p in gsrc.items():
                    for c, e in cell_errors(y, p, cell).items():
                        vrows.append(dict(dataset=ds, f=f, r=r, k=k, model=m, cell=c, mae=e))
            for s_, (p3, gb) in src.items():
                for om in OMEGAS:
                    for w in WINDOWS:
                        p = by_cell(causal_trend, om * p3 + (1 - om) * gb, cell, cyc, w=w)
                        for c, e in cell_errors(y, p, cell).items():
                            grid.append(dict(dataset=ds, f=f, r=r, k=k, part=part, src=s_, om=om, w=w, cell=c, mae=e))
            for s_, gb in gsrc.items():                  # GBDT có / không lịch sử, không lọc (w = 0) hoặc qua bộ lọc xu thế
                for w in (0,) + tuple(WINDOWS):
                    p = gb if w == 0 else by_cell(causal_trend, gb, cell, cyc, w=w)
                    for c, e in cell_errors(y, p, cell).items():
                        frows.append(dict(dataset=ds, f=f, r=r, k=k, part=part, src=s_, w=w, cell=c, mae=e))
    C, Gd, Fd = pd.DataFrame(rows), pd.DataFrame(grid), pd.DataFrame(frows)
    V = Gd[Gd.part == 'val'].groupby(['dataset', 'f', 'src', 'om', 'w']).mae.mean().reset_index()
    best = V.loc[V.groupby(['dataset', 'f', 'src']).mae.idxmin()]
    Tb = Gd[Gd.part == 'test'].merge(best[['dataset', 'f', 'src', 'om', 'w']], on=['dataset', 'f', 'src', 'om', 'w'])
    Tb['model'] = Tb.src.map(dict(e20='pinn_lai_v2', hist='pinn_lai_h_v2'))
    cols = ['dataset', 'f', 'r', 'k', 'model', 'cell', 'mae']
    # có hay không lịch sử, chọn trên validation cho từng (bộ, mức nhãn)
    fixed = V[(V.om == 0.5) & (V.w == W)]
    ch1 = fixed.loc[fixed.groupby(['dataset', 'f']).mae.idxmin()][['dataset', 'f', 'src', 'om', 'w']]
    ch2 = best.loc[best.groupby(['dataset', 'f']).mae.idxmin()][['dataset', 'f', 'src', 'om', 'w']]
    T1 = Gd[Gd.part == 'test'].merge(ch1, on=['dataset', 'f', 'src', 'om', 'w']).assign(model='pinn_lai_sel')
    T2 = Gd[Gd.part == 'test'].merge(ch2, on=['dataset', 'f', 'src', 'om', 'w']).assign(model='pinn_lai_sel_v2')
    Vg = pd.DataFrame(vrows).groupby(['dataset', 'f', 'model']).mae.mean().reset_index()
    chg = Vg.loc[Vg.groupby(['dataset', 'f']).mae.idxmin()][['dataset', 'f', 'model']]
    Tg = C.merge(chg, on=['dataset', 'f', 'model']).assign(model='gbdt_best')
    Vf = Fd[Fd.part == 'val'].groupby(['dataset', 'f', 'src', 'w']).mae.mean().reset_index()
    chf = Vf.loc[Vf.groupby(['dataset', 'f']).mae.idxmin()][['dataset', 'f', 'src', 'w']]
    Tf = Fd[Fd.part == 'test'].merge(chf, on=['dataset', 'f', 'src', 'w']).assign(model='gbdt_best_f')
    C = pd.concat([C, Tb[cols], T1[cols], T2[cols], Tg[cols], Tf[cols]], ignore_index=True)
    sel = pd.concat([ch1.assign(kind='pinn_lai'), ch2.assign(kind='pinn_lai_v2'),
                     chg.rename(columns={'model': 'src'}).assign(kind='gbdt'), chf.assign(kind='gbdt_f')],
                    ignore_index=True)
    return C, best, sel


def val_ratios():
    """MAE validation (theo cell) của PINN-semi có lịch sử / không lịch sử, thành viên 0, trung bình mọi lần lặp và riêng
    từng lần lặp — để so với lựa chọn ở E29dev (chỉ lần lặp 0, 30 và 70 % nhãn)."""
    import json
    from pinnsoh.train import Config, run_name
    rows = []
    for j in E29():
        a = j.args
        if j.kind != 'train' or a['model'] != 'pinn_semi' or a.get('member', 0) != 0 or not os.path.exists(j.out):
            continue
        b = os.path.join('runs/E20', run_name(Config(**{**a, 'norm': 'global', 'tag': 'P', 'out_dir': 'runs/E20'})) + '.json')
        r, q = json.load(open(j.out)), json.load(open(b))
        rows.append(dict(dataset=a['dataset'], f=a['label_frac'], r=a['repeat'],
                         val=r['val']['MAE_cell'] / q['val']['MAE_cell'], test=r['test']['MAE_cell'] / q['test']['MAE_cell']))
    D = pd.DataFrame(rows)
    out = D.groupby(['dataset', 'f']).agg(val=('val', 'mean'), test=('test', 'mean')).reset_index()
    per = D.groupby(['dataset', 'f', 'r']).val.mean().unstack('r')
    per.columns = [f'val_r{r}' for r in per.columns]
    return out.merge(per.reset_index(), on=['dataset', 'f'])


def per_cell_mean(C):
    return C.groupby(['dataset', 'f', 'model', 'cell']).mae.mean().reset_index()


PAIRS = [('pinn_h', 'pinn'), ('pinn_lai_h', 'pinn_lai'), ('pinn_lai_h_v2', 'pinn_lai_v2'), ('pinn_lai_h', 'gbdt_sel'),
         ('pinn_lai_h_v2', 'gbdt_sel'), ('pinn_h', 'mlp_h'), ('pinn', 'mlp'), ('mlp_h', 'mlp'), ('gbdt_sel', 'gbdt'),
         ('pinn_lai_sel', 'pinn_lai'), ('pinn_lai_sel_v2', 'pinn_lai_v2'), ('pinn_lai_sel', 'gbdt_best'),
         ('pinn_lai_sel_v2', 'gbdt_best'), ('pinn_lai', 'gbdt'), ('pinn_lai_v2', 'gbdt'), ('pinn_lai_sel_v2', 'gbdt_best_f'),
         ('gbdt_best_f', 'gbdt_best')]


def tests(C):
    a = per_cell_mean(C)
    out = []
    for ds in DS:
        for f in sorted(a.f.unique()):
            for m, ref in PAIRS:
                if m == 'gbdt_sel' and ref == 'gbdt' and ds not in E29_SPEC_GBDT:
                    continue
                x = a[(a.dataset == ds) & (a.f == f) & (a.model == m)].set_index('cell').mae
                z = a[(a.dataset == ds) & (a.f == f) & (a.model == ref)].set_index('cell').mae
                cells = sorted(set(x.index) & set(z.index))
                if not cells:
                    continue
                d = x.reindex(cells).values - z.reindex(cells).values
                out.append(dict(dataset=ds, f=f, model=m, ref=ref, n_cells=len(cells), mae=x.reindex(cells).mean(),
                                mae_ref=z.reindex(cells).mean(), ratio=x.reindex(cells).mean() / z.reindex(cells).mean(),
                                better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pd.DataFrame(out)


def summary(C):
    S = per_cell_mean(C).groupby(['dataset', 'f', 'model']).mae.mean().reset_index()
    S['spec_pinn'] = S.dataset.map(lambda d: E29_SPEC_PINN.get(d, 'global'))
    S['spec_gbdt'] = S.dataset.map(lambda d: E29_SPEC_GBDT.get(d, 'global'))
    return S


def figure(S):
    import figures_report
    figures_report.e29_hist()


def main():
    C, best, sel = build()
    if C.empty:
        print('chưa có kết quả E29')
        return
    os.makedirs(OUT, exist_ok=True)
    S = summary(C); S.to_csv(f'{OUT}/e29_summary.csv', index=False)
    T = tests(C); T.to_csv(f'{OUT}/e29_tests.csv', index=False)
    best.to_csv(f'{OUT}/e29_tuned.csv', index=False); sel.to_csv(f'{OUT}/e29_select.csv', index=False)
    val_ratios().to_csv(f'{OUT}/e29_val.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 400):
        print((S.assign(mae=1e3 * S.mae).pivot_table(index=['dataset', 'f'], columns='model', values='mae')).round(2))
        print(T.round(4).to_string(index=False))
    figure(S)


if __name__ == '__main__':
    main()
