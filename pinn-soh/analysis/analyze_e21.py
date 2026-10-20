"""
E21: huấn luyện trên một bộ, kiểm tra trên các bộ còn lại; thích nghi không nhãn và tự huấn luyện nhiều vòng.

    python analysis/analyze_e21.py        # -> results/e21_*.csv, results/fig_e21_*.png|pdf

Mô hình được đánh giá (mọi dự đoán giới hạn trong [0,40; 1,15], miền của L_range):
  mlp, gbdt, pinn (thành viên 0), pinn3 (trung bình 3 thành viên), ket_hop = ½ pinn3 + ½ gbdt qua bộ lọc quỹ đạo.
Giao thức A (70/30): test trên mọi cell của 3 bộ đích. Giao thức B (70/15/15): thêm test trong bộ (đường chéo).
Chuỗi thích nghi (giao thức A): sai số trên nửa bộ đích không dùng để thích nghi, theo vòng 0, 1, 2; so ghép cặp
với mô hình không thích nghi trên đúng các cell đó.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.runcache import cached_map

OUT, ROOT = 'results', 'runs/E21'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
LO, HI, W = 0.40, 1.15, 10
LABEL = {'mlp': 'MLP', 'gbdt': 'GBDT', 'pinn': 'PINN', 'pinn3': '3 PINN', 'ket_hop': 'Kết hợp'}


def _load(path):
    d = json.load(open(path))
    out = {}
    for k, pr in d['preds'].items():               # nhãn cell lưu dạng chỉ số int + danh sách tên (gọn bộ nhớ đệm)
        out[k] = dict(y=np.asarray(pr['y'], np.float32), p=np.clip(np.asarray(pr['p'], np.float32), LO, HI),
                      cell=np.asarray(pr['cell'], np.int16), cyc=np.asarray(pr['cycle'], np.float32),
                      ids=list(pr['cell_ids']))
    meta = {k: d.get(k) for k in ('kind', 'proto', 'src', 'tgt', 'seed', 'half', 'round', 'model', 'member', 'val_MAE')}
    return dict(meta=meta, preds=out)


def per_cell(y, p, cell, ids):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows = {}
    for c in np.unique(cell):
        m = cell == c
        if np.isfinite(y[m]).sum() == 0:
            continue
        k = m & np.isfinite(y)
        e = p[k] - y[k]
        rows[ids[c]] = (np.abs(e).mean(), np.sqrt((e ** 2).mean()), e.mean(), int(k.sum()), float((e ** 2).sum()),
                   float(y[k].sum()), float((y[k] ** 2).sum()))
    return rows


def combine(preds_list, gb, key):
    """Trung bình các thành viên PINN, ghép GBDT, lọc quỹ đạo; trả dict tên mô hình -> dự đoán."""
    base = preds_list[0][key]
    for o in preds_list[1:] + [gb]:
        assert np.array_equal(o[key]['cell'], base['cell']) and np.array_equal(o[key]['cyc'], base['cyc'])
    p3 = np.mean([np.asarray(q[key]['p'], float) for q in preds_list], axis=0)
    g = np.asarray(gb[key]['p'], float)
    kh = np.clip(by_cell(causal_trend, 0.5 * p3 + 0.5 * g, base['cell'], base['cyc'], w=W), LO, HI)
    return base, dict(pinn=preds_list[0][key]['p'], pinn3=p3, gbdt=g, ket_hop=kh)


def base_table():
    files = sorted(glob.glob(f'{ROOT}/base_*.json'))
    D = dict(zip(files, cached_map(files, _load, 'e21_base_v1')))
    rows, cells = [], []
    for proto in ('a7030', 'b701515'):
        for src in DS:
            for s in (0, 1, 2):
                f = lambda m, mem=0: f'{ROOT}/base_{proto}_{src}_s{s}_{m}' + (f'_m{mem}' if mem else '') + '.json'
                need = [f('mlp'), f('gbdt')] + [f('pinn_semi', m) for m in range(3)]
                if not all(x in D for x in need):
                    continue
                mlp, gb = D[f('mlp')]['preds'], D[f('gbdt')]['preds']
                pins = [D[f('pinn_semi', m)]['preds'] for m in range(3)]
                for key in mlp:
                    if key == 'adapt':
                        continue
                    base, pr = combine(pins, gb, key)
                    pr['mlp'] = mlp[key]['p']
                    tgt = src if key == 'src_test' else key
                    for m, p in pr.items():
                        pcs = per_cell(base['y'], p, base['cell'], base['ids'])
                        for c, v in pcs.items():
                            cells.append(dict(proto=proto, src=src, tgt=tgt, seed=s, model=m, cell=c, mae=v[0],
                                              rmse=v[1], bias=v[2], n=v[3], sse=v[4], sy=v[5], syy=v[6]))
    cdf = pd.DataFrame(cells)
    g = cdf.groupby(['proto', 'src', 'tgt', 'seed', 'model'])
    agg = g.agg(MAE_cell=('mae', 'mean'), RMSE_cell=('rmse', 'mean'), bias=('bias', 'mean'), n=('n', 'sum'),
                sse=('sse', 'sum'), sy=('sy', 'sum'), syy=('syy', 'sum')).reset_index()
    agg['R2'] = 1 - agg.sse / (agg.syy - agg.sy ** 2 / agg.n)
    summ = agg.groupby(['proto', 'src', 'tgt', 'model'])[['MAE_cell', 'RMSE_cell', 'bias', 'R2']].mean().reset_index()
    return cdf, summ


def chain_table():
    files = sorted(glob.glob(f'{ROOT}/chain_*.json'))
    D = dict(zip(files, cached_map(files, _load, 'e21_chain_v1')))
    bfiles = sorted(glob.glob(f'{ROOT}/base_a7030_*.json'))
    B = dict(zip(bfiles, cached_map(bfiles, _load, 'e21_basea_v1')))
    rows = []
    for src in DS:
        for tgt in DS:
            if src == tgt:
                continue
            for s in (0, 1):
                bf = lambda m, mem=0: f'{ROOT}/base_a7030_{src}_s{s}_{m}' + (f'_m{mem}' if mem else '') + '.json'
                if not all(bf(m, mm) in B for m, mm in [('gbdt', 0)] + [('pinn_semi', i) for i in range(3)]):
                    continue
                bb, bpr = combine([B[bf('pinn_semi', i)]['preds'] for i in range(3)], B[bf('gbdt')]['preds'], tgt)
                for h in (0, 1):
                    for rnd in range(3):
                        cf = lambda m, mem=0: (f'{ROOT}/chain_{src}_to_{tgt}_s{s}_h{h}_r{rnd}_{m}'
                                               + (f'_m{mem}' if mem else '') + '.json')
                        if not all(cf(m, mm) in D for m, mm in [('gbdt', 0)] + [('pinn', i) for i in range(3)]):
                            continue
                        base, pr = combine([D[cf('pinn', i)]['preds'] for i in range(3)], D[cf('gbdt')]['preds'],
                                           'test')
                        test_names = {base['ids'][i] for i in np.unique(base['cell'])}
                        keep = np.isin(bb['cell'], [i for i, n in enumerate(bb['ids']) if n in test_names])
                        for m, p in pr.items():
                            for c, v in per_cell(base['y'], p, base['cell'], base['ids']).items():
                                rows.append(dict(src=src, tgt=tgt, seed=s, half=h, round=rnd, model=m, cell=c,
                                                 mae=v[0], bias=v[2]))
                        if rnd == 0:                       # mô hình không thích nghi trên đúng các cell test này
                            for m, p in bpr.items():
                                for c, v in per_cell(bb['y'][keep], p[keep], bb['cell'][keep], bb['ids']).items():
                                    rows.append(dict(src=src, tgt=tgt, seed=s, half=h, round=-1, model=m, cell=c,
                                                     mae=v[0], bias=v[2]))
    return pd.DataFrame(rows)


def main():
    cdf, summ = base_table()
    summ.to_csv(f'{OUT}/e21_base.csv', index=False)
    ch = chain_table()
    ch.to_csv(f'{OUT}/e21_chain_cells.csv.gz', index=False)
    pd.set_option('display.width', 220, 'display.max_rows', 400)
    for proto in ('a7030', 'b701515'):
        t = summ[summ.proto == proto].pivot_table(index=['src', 'model'], columns='tgt', values='MAE_cell')
        print(proto); print((1e3 * t).round(1).to_string())
    if len(ch):
        cs = ch.groupby(['src', 'tgt', 'round', 'model']).mae.mean().unstack('round')
        cs.to_csv(f'{OUT}/e21_chain.csv')
        print((1e3 * cs).round(1).to_string())
        tests = chain_tests(ch)
        tests.to_csv(f'{OUT}/e21_chain_tests.csv', index=False)
        print(tests.round(4).to_string(index=False))
    figures(summ, ch)


def chain_tests(ch):
    """Kết hợp ở vòng r so với kết hợp không thích nghi (vòng -1), ghép cặp theo cell (trung bình qua hạt giống)."""
    rows = []
    g = ch[ch.model == 'ket_hop'].groupby(['src', 'tgt', 'round', 'cell']).mae.mean().reset_index()
    for (src, tgt), h in g.groupby(['src', 'tgt']):
        base = h[h['round'] == -1].set_index('cell').mae
        for rnd in (0, 1, 2):
            x = h[h['round'] == rnd].set_index('cell').mae
            common = base.index.intersection(x.index)
            if len(common) < 6:
                continue
            a, b = base.loc[common].values, x.loc[common].values
            rows.append(dict(src=src, tgt=tgt, round=rnd, n_cell=len(common), ti_so=b.mean() / a.mean(),
                             cell_tot_hon=float((b < a).mean()), p=stats.wilcoxon(b, a).pvalue))
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────── hình
def figures(summ, ch):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from pinnsoh.plotstyle import apply_theme, GRID, CHEM, panel_title, save, SEQ_CMAP
    apply_theme()

    # 1. ma trận MAE nguồn -> đích (giao thức B: đường chéo là test trong bộ)
    mods = ['mlp', 'gbdt', 'pinn3', 'ket_hop']
    sb = summ[summ.proto == 'b701515']
    if len(sb):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
        vals = 1e3 * sb[sb.model.isin(mods)].MAE_cell
        norm = LogNorm(vmin=max(1, vals.min()), vmax=vals.max())
        for ax, m in zip(axes, mods):
            M = np.full((4, 4), np.nan)
            for i, s in enumerate(DS):
                for j, t in enumerate(DS):
                    r = sb[(sb.src == s) & (sb.tgt == t) & (sb.model == m)]
                    if len(r):
                        M[i, j] = 1e3 * r.MAE_cell.iloc[0]
            im = ax.imshow(M, cmap=SEQ_CMAP, norm=norm)
            for i in range(4):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec='#1b2330', lw=1.0))
            ax.set_xticks(range(4)); ax.set_xticklabels(DS, rotation=90, fontsize=6.5)
            ax.set_yticks(range(4)); ax.set_yticklabels(DS if m == mods[0] else [], fontsize=6.5)
            ax.grid(False); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            panel_title(ax, LABEL[m], pad=4)
        axes[0].set_ylabel('bộ huấn luyện')
        fig.text(0.47, 0.01, 'bộ kiểm tra', ha='center', fontsize=8.5)
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label('MAE theo cell (×10$^{-3}$)', fontsize=7); cb.ax.tick_params(labelsize=6.5)
        save(fig, f'{OUT}/fig_e21_matrix')

    # 2. sai số qua các vòng học lặp, mỗi panel một bộ đích
    if len(ch):
        k = ch[ch.model == 'ket_hop'].groupby(['src', 'tgt', 'round']).mae.mean().reset_index()
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
        cols = {'XJTU': '#2a78d6', 'TJU': '#eb6834', 'MIT': '#1baf7a', 'HUST': '#7b5cd6'}
        for ax, tgt in zip(axes, DS):
            for src in DS:
                h = k[(k.src == src) & (k.tgt == tgt)].sort_values('round')
                if len(h) < 2:
                    continue
                ax.plot(h['round'] + 1, 1e3 * h.mae, marker='o', ms=3.4, lw=1.4, color=cols[src], mec='white',
                        mew=0.4, label=src)
            ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(['không', '0', '1', '2'], fontsize=6.6)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
            panel_title(ax, f'→ {tgt}', CHEM[tgt], pad=4)
        axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
        handles = [plt.Line2D([], [], color=cols[s], marker='o', ms=3.4, lw=1.4, label=f'nguồn {s}') for s in DS]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.14, 1, 1])
        fig.text(0.5, 0.1, 'vòng thích nghi (không: chưa thích nghi; 0: chuẩn hoá theo bộ + vật lý; 1–2: tự huấn luyện)',
                 ha='center', fontsize=7.5)
        save(fig, f'{OUT}/fig_e21_rounds')


def fig_traj(pairs=(('MIT', 'HUST'), ('XJTU', 'TJU'), ('HUST', 'MIT'), ('TJU', 'XJTU'))):
    """Quỹ đạo SOH của một cell bộ đích (nửa test, hạt giống 0) với mô hình kết hợp chưa thích nghi, vòng 0 và vòng 2.
    Cell chọn là cell có sai số ở trung vị dưới mô hình chưa thích nghi."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, CHEM, panel_title, save, MODEL_COLOR
    apply_theme()
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2, 2.35))
    cols = {'base': '#9aa5ad', 0: '#2a78d6', 2: '#0b4f3c'}
    for ax, (src, tgt) in zip(axes, pairs):
        bf = lambda m, mem=0: f'{ROOT}/base_a7030_{src}_s0_{m}' + (f'_m{mem}' if mem else '') + '.json'
        cf = lambda r, m, mem=0: f'{ROOT}/chain_{src}_to_{tgt}_s0_h0_r{r}_{m}' + (f'_m{mem}' if mem else '') + '.json'
        need = [bf('gbdt')] + [bf('pinn_semi', i) for i in range(3)] + [cf(r, 'gbdt') for r in (0, 2)] \
            + [cf(r, 'pinn', i) for r in (0, 2) for i in range(3)]
        if not all(os.path.exists(x) for x in need):
            ax.axis('off'); continue
        bb, bpr = combine([_load(bf('pinn_semi', i))['preds'] for i in range(3)], _load(bf('gbdt'))['preds'], tgt)
        res = {r: combine([_load(cf(r, 'pinn', i))['preds'] for i in range(3)], _load(cf(r, 'gbdt'))['preds'], 'test')
               for r in (0, 2)}
        base, _ = res[0]
        names = [base['ids'][i] for i in np.unique(base['cell'])]
        err = []
        for nm in names:
            k = bb['ids'].index(nm); m = bb['cell'] == k
            err.append((np.nanmean(np.abs(bpr['ket_hop'][m] - bb['y'][m])), nm))
        err.sort(); nm = err[(len(err) - 1) // 2][1]
        k = bb['ids'].index(nm); m = bb['cell'] == k; o = np.argsort(bb['cyc'][m])
        ax.plot(bb['cyc'][m][o], bb['y'][m][o], color=MODEL_COLOR['truth'], lw=1.6, zorder=4)
        ax.plot(bb['cyc'][m][o], bpr['ket_hop'][m][o], color=cols['base'], lw=1.0, zorder=2)
        for r in (0, 2):
            b, pr = res[r]; kk = b['ids'].index(nm); mm = b['cell'] == kk; oo = np.argsort(b['cyc'][mm])
            ax.plot(b['cyc'][mm][oo], pr['ket_hop'][mm][oo], color=cols[r], lw=1.1, zorder=3)
        ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        panel_title(ax, f'{src} → {tgt}', CHEM[tgt], pad=3)
    axes[0].set_ylabel('SOH')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.5, label='thực đo'),
                        plt.Line2D([], [], color=cols['base'], lw=1.3, label='chưa thích nghi'),
                        plt.Line2D([], [], color=cols[0], lw=1.3, label='vòng 0'),
                        plt.Line2D([], [], color=cols[2], lw=1.3, label='vòng 2')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.text(0.5, 0.11, 'chu kỳ', ha='center', fontsize=8.5)
    save(fig, f'{OUT}/fig_e21_traj')


if __name__ == '__main__':
    main()
    fig_traj()
