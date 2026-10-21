"""
E24: bộ NASA PCoE — ước lượng SOH (có và không có thời gian nghỉ), cách chia của SambaMixer.

    python analysis/analyze_e24.py      # -> results/e24_*.csv, results/fig_e24_*.png|pdf

Mô hình như E20: pinn (PINN-semi thành viên 0), pinn3 (trung bình 3), gbdt, mlp, pinn_lai (½ pinn3 + ½ gbdt, lọc
quỹ đạo w = 10). Hai biến thể dữ liệu: NASA (16 đặc trưng sạc) và NASAdt (thêm log(1 + giờ nghỉ)).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.data import load_dataset

OUT, ROOT = 'results', 'runs/E24'
SETS = ['NASA', 'NASAdt']
FRACS = [0.3, 0.7]
MODELS = ['mlp', 'gbdt', 'pinn', 'pinn3', 'pinn_lai']
LABEL = {'mlp': 'MLP', 'gbdt': 'GBDT', 'pinn': 'PINN-semi', 'pinn3': '3 PINN-semi', 'pinn_lai': 'PINN-lai'}
# SambaMixer (Olalde-Verano và cộng sự, 2024), Bảng 4: MAE, RMSE (% SOH) trên ba cell đánh giá, đầu vào là đường xả
SAMBA_REF = {'B0006': dict(samba=(1.173, 2.068), cnn_bigru=(2.448, 3.177)),
             'B0007': dict(samba=(1.197, 1.285), cnn_bigru=(1.861, 2.252)),
             'B0047': dict(samba=(0.512, 0.645), cnn_bigru=(2.549, 3.094))}


def _pred(path, part='test_predictions'):
    d = json.load(open(path))[part]
    ids = d['cell_ids']
    return dict(y=np.asarray(d['y'], float), p=np.asarray(d['p'], float), cell=np.asarray([ids[i] for i in d['cell']]),
                cyc=np.asarray(d['cycle'], float))


def combine(P, G, M=None):
    base = P[0]
    for o in P[1:] + [G] + ([M] if M else []):
        assert np.array_equal(o['cell'], base['cell']) and np.array_equal(o['cyc'], base['cyc'])
    pinn3 = np.mean([x['p'] for x in P], axis=0)
    pr = dict(pinn=P[0]['p'], pinn3=pinn3, gbdt=G['p'],
              pinn_lai=by_cell(causal_trend, 0.5 * pinn3 + 0.5 * G['p'], base['cell'], base['cyc'], w=10))
    if M:
        pr['mlp'] = M['p']
    return base, pr


def cell_rows(base, pr, **key):
    rows = []
    for c in np.unique(base['cell']):
        i = base['cell'] == c
        for m, p in pr.items():
            e = p[i] - base['y'][i]
            rows.append(dict(key, cell=c, model=m, mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                             mape=np.mean(np.abs(e) / base['y'][i])))
    return rows


def rest_of():
    """Giờ nghỉ trước mỗi lần xả, theo (cell, chu kỳ) — để tách các chu kỳ ngay sau khoảng nghỉ dài."""
    out = {}
    for f in glob.glob('PINN4SOH/data/NASA data/*/*.csv'):
        d = pd.read_csv(f)
        out[f'NASA/{os.path.basename(f)[:-4]}'] = d['rest_h'].values
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rest = rest_of()
    rows, long_rows = [], []
    for ds, f, r, k in [(ds, f, r, k) for ds in SETS for f in FRACS for r in range(3) for k in range(5)]:
        s = 100 * r + k
        Pp = [f'{ROOT}/P_{ds}_pinn_semi_f{f}_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}') + '.json'
              for m in range(3)]
        Gp = f'{ROOT}/gbdt_{ds}_f{f}_cv{r}-{k}of5.json'
        Mp = f'{ROOT}/M_{ds}_mlp_f{f}_s{s}_global_global_cv{r}-{k}of5.json'
        if not all(os.path.exists(p) for p in Pp + [Gp, Mp]):
            continue
        base, pr = combine([_pred(p) for p in Pp], _pred(Gp), _pred(Mp))
        rows += cell_rows(base, pr, ds=ds, frac=f, r=r, k=k)
        # sai số ở các chu kỳ ngay sau khoảng nghỉ dài (> 2 lần trung vị của cell) — nơi dung lượng hồi phục
        for c in np.unique(base['cell']):
            i = np.flatnonzero(base['cell'] == c)
            rh = rest[c][base['cyc'][i].astype(int)]
            lng = rh > 2 * np.nanmedian(rest[c])
            for m, p in pr.items():
                e = np.abs(p[i] - base['y'][i])
                long_rows.append(dict(ds=ds, frac=f, r=r, cell=c, model=m, mae_long=e[lng].mean() if lng.any() else np.nan,
                                      mae_norm=e[~lng].mean(), n_long=int(lng.sum())))
    C = pd.DataFrame(rows)
    pc = C.groupby(['ds', 'frac', 'model', 'cell'])[['mae', 'rmse', 'mape']].mean().reset_index()
    summ = pc.groupby(['ds', 'frac', 'model'])[['mae', 'rmse', 'mape']].mean().reset_index()
    summ['n_cell'] = pc.groupby(['ds', 'frac', 'model']).size().values
    summ.to_csv(f'{OUT}/e24_cv.csv', index=False)
    # kiểm định: NASAdt so với NASA (cùng mô hình, ghép cặp theo cell); PINN-lai so với từng mô hình trong mỗi bộ
    tests = []
    for f in FRACS:
        for m in MODELS:
            a = pc[(pc.ds == 'NASA') & (pc.frac == f) & (pc.model == m)].set_index('cell').mae
            b = pc[(pc.ds == 'NASAdt') & (pc.frac == f) & (pc.model == m)].set_index('cell').mae
            j = a.index.intersection(b.index)
            if len(j) > 3:
                tests.append(dict(kind='dt_vs_nodt', frac=f, model=m, ratio=float(np.exp(np.mean(np.log(b[j] / a[j])))),
                                  better=int((b[j] < a[j]).sum()), n=len(j), p=stats.wilcoxon(b[j] - a[j]).pvalue))
        for ds in SETS:
            g = pc[(pc.ds == ds) & (pc.frac == f)].pivot(index='cell', columns='model', values='mae')
            for m in ['mlp', 'gbdt', 'pinn', 'pinn3']:
                tests.append(dict(kind=f'lai_vs_{m}', ds=ds, frac=f, model=m,
                                  ratio=float(np.exp(np.mean(np.log(g.pinn_lai / g[m])))),
                                  better=int((g.pinn_lai < g[m]).sum()), n=len(g), p=stats.wilcoxon(g.pinn_lai - g[m]).pvalue))
    pd.DataFrame(tests).to_csv(f'{OUT}/e24_tests.csv', index=False)
    L = pd.DataFrame(long_rows)
    lr = L.groupby(['ds', 'frac', 'model', 'cell'])[['mae_long', 'mae_norm']].mean().groupby(['ds', 'frac', 'model']).mean()
    lr.reset_index().to_csv(f'{OUT}/e24_rest.csv', index=False)
    # cách chia SambaMixer
    srows = []
    for ds, s in [(ds, s) for ds in SETS for s in range(3)]:
        Pp = [f'{ROOT}/S_{ds}_pinn_semi_f0.7_s{s}_global_global' + ('' if m == 0 else f'_m{m}') + '.json' for m in range(3)]
        Gp = f'{ROOT}/gbdt_{ds}_samba_s{s}.json'
        Mp = f'{ROOT}/S_{ds}_mlp_f0.7_s{s}_global_global.json'
        if not all(os.path.exists(p) for p in Pp + [Gp, Mp]):
            continue
        base, pr = combine([_pred(p) for p in Pp], _pred(Gp), _pred(Mp))
        srows += cell_rows(base, pr, ds=ds, seed=s)
    S = pd.DataFrame(srows)
    if len(S):
        ss = S.groupby(['ds', 'model', 'cell'])[['mae', 'rmse', 'mape']].mean().reset_index()
        ss.to_csv(f'{OUT}/e24_samba.csv', index=False)
    pd.set_option('display.width', 200)
    print(summ.assign(mae=lambda x: (100 * x.mae).round(2), rmse=lambda x: (100 * x.rmse).round(2)).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print((100 * lr).round(2).to_string())
    if len(S):
        print(ss.assign(mae=lambda x: (100 * x.mae).round(2), rmse=lambda x: (100 * x.rmse).round(2),
                        mape=lambda x: (100 * x.mape).round(2)).to_string(index=False))
    figure()


def figure():
    """Quỹ đạo SOH của ba cell đánh giá theo cách chia SambaMixer (hạt giống 0): thực đo, PINN-lai không và có giờ nghỉ."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save, MODEL_COLOR
    apply_theme()
    res = {}
    for ds in SETS:
        Pp = [f'{ROOT}/S_{ds}_pinn_semi_f0.7_s0_global_global' + ('' if m == 0 else f'_m{m}') + '.json' for m in range(3)]
        Gp = f'{ROOT}/gbdt_{ds}_samba_s0.json'
        if not all(os.path.exists(p) for p in Pp + [Gp]):
            return
        res[ds] = combine([_pred(p) for p in Pp], _pred(Gp))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, c in zip(axes, ['NASA/B0006', 'NASA/B0007', 'NASA/B0047']):
        for ds, col, lw in (('NASA', '#9aa5ad', 1.1), ('NASAdt', '#0b4f3c', 1.4)):
            base, pr = res[ds]
            i = base['cell'] == c; o = np.argsort(base['cyc'][i])
            if ds == 'NASA':
                ax.plot(base['cyc'][i][o], 100 * base['y'][i][o], color=MODEL_COLOR['truth'], lw=1.5, zorder=4)
            ax.plot(base['cyc'][i][o], 100 * pr['pinn_lai'][i][o], color=col, lw=lw, zorder=3)
        ax.axhline(70, color='#c0392b', lw=0.7, ls=(0, (4, 2)))
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        ax.locator_params(axis='x', nbins=4)
        panel_title(ax, c.split('/')[1], pad=3)
    axes[0].set_ylabel('SOH (%)')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.5, label='thực đo'),
                        plt.Line2D([], [], color='#9aa5ad', lw=1.3, label='PINN-lai, 16 đặc trưng'),
                        plt.Line2D([], [], color='#0b4f3c', lw=1.4, label='PINN-lai, thêm giờ nghỉ'),
                        plt.Line2D([], [], color='#c0392b', lw=0.8, ls=(0, (4, 2)), label='ngưỡng 70 %')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.text(0.5, 0.1, 'chu kỳ xả', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e24_samba')


if __name__ == '__main__':
    main()
