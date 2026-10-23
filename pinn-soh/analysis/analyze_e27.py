"""
E27: so với PINN4SOH (Wang và cộng sự, 2024) trên đúng cách chia của bài, ở hai cách tiền xử lý.

    python analysis/analyze_e27.py      # -> results/e27_summary.csv, e27_cells.csv, e27_tests.csv, fig_e27_*

Mô hình của đồ án trên mỗi nhóm (batch của XJTU/TJU, cả bộ với MIT/HUST) và mỗi hạt giống:
    pinn3      trung bình 3 PINN-semi
    gbdt       GBDT
    pinn_lai   ½ pinn3 + ½ gbdt qua bộ lọc quỹ đạo w = 10 (đúng công thức E20)
    pinn_bb    PINN động học hộp đen kiểu PINN4SOH (chỉ các nhóm đầy đủ)
Hai cách tiền xử lý: 'strict' (của đồ án, không rò rỉ) và 'paper' (bản sao tiền xử lý của bài). Sai số: MAE theo cell
trên cell test, trung bình qua hạt giống, rồi trung bình qua các nhóm như bài làm. Số của bài lấy từ các tệp kết quả công bố
(pinnsoh/paper.py: PAPER_MAE; giá trị từng cell: pinnsoh/pinn4soh_reported.csv). Kiểm định Wilcoxon ghép cặp theo cell
test giữa PINN-lai và PINN4SOH ở các nhóm đầy đủ.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import collections, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.experiments import E27
from pinnsoh.paper import PAPER_MAE, gbdt_path
from pinnsoh.postproc import by_cell, causal_trend

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
W = 10
REPORTED = os.path.join('pinnsoh', 'pinn4soh_reported.csv')


def _pred(path):
    tp = json.load(open(path))['test_predictions']
    ids = tp['cell_ids']
    return dict(y=np.asarray(tp['y'], float), p=np.asarray(tp['p'], float),
                cell=np.asarray([ids[i] for i in tp['cell']]), cyc=np.asarray(tp['cycle'], float))


def collect():
    """{(key, prep, seed): {'pinn': [3 đường dẫn], 'bb': đường dẫn | None, 'gbdt': đường dẫn}}"""
    G = collections.defaultdict(dict)
    for j in E27():
        if j.kind == 'train_cfg':
            a = j.args
            prep = a['tag'][3:]
            g = G[(a['paper_key'], prep, a['seed'])]
            if a['model'] == 'pinn_semi':
                g.setdefault('pinn', [None] * 3)[a['member']] = j.out
            else:
                g['bb'] = j.out
        else:
            a = j.args
            G[(a['key'], a['prep'], a['seed'])]['gbdt'] = j.out
    return G


def cell_errors(y, p, cell):
    return {c: float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)}


def build():
    rows = []
    for (key, prep, seed), g in sorted(collect().items()):
        need = g['pinn'] + [g['gbdt']] + ([g['bb']] if g.get('bb') else [])
        if not all(p and os.path.exists(p) for p in need):
            continue
        P = [_pred(p) for p in g['pinn']]; Gb = _pred(g['gbdt'])
        y, cell, cyc = P[0]['y'], P[0]['cell'], P[0]['cyc']
        for o in P[1:] + [Gb]:
            assert np.array_equal(o['cell'], cell) and np.array_equal(o['cyc'], cyc), (key, prep, seed)
        pinn3 = np.mean([m['p'] for m in P], axis=0)
        pred = dict(pinn=P[0]['p'], pinn3=pinn3, gbdt=Gb['p'],
                    pinn_lai=by_cell(causal_trend, 0.5 * pinn3 + 0.5 * Gb['p'], cell, cyc, w=W))
        if g.get('bb'):
            B = _pred(g['bb'])
            assert np.array_equal(B['cell'], cell)
            pred['pinn_bb'] = B['p']
        ds, mode, grp = key.split('|')
        for m, p in pred.items():
            for c, e in cell_errors(y, p, cell).items():
                rows.append(dict(dataset=ds, mode=mode, group=grp, prep=prep, seed=seed, model=m, cell=c, mae=e))
    return pd.DataFrame(rows)


def summarize(C):
    # MAE theo cell -> trung bình qua cell (từng nhóm, hạt giống) -> qua hạt giống -> qua nhóm (như bài)
    g = C.groupby(['dataset', 'mode', 'prep', 'model', 'group', 'seed']).mae.mean()
    g = g.groupby(['dataset', 'mode', 'prep', 'model', 'group']).mean()
    S = g.groupby(['dataset', 'mode', 'prep', 'model']).mean().rename('mae').reset_index()
    S['n_groups'] = g.groupby(['dataset', 'mode', 'prep', 'model']).size().values
    for ref in ('PINN4SOH', 'MLP', 'CNN'):
        S[f'paper_{ref}'] = [PAPER_MAE.get((r.dataset, r.mode), {}).get(ref, np.nan) for r in S.itertuples()]
    S['ratio_vs_pinn4soh'] = S.mae / S.paper_PINN4SOH
    return S


def tests(C):
    """Wilcoxon theo cell test (nhóm đầy đủ): PINN-lai (trung bình 3 hạt giống) so với giá trị từng cell của PINN4SOH."""
    R = pd.read_csv(REPORTED)
    out = []
    for prep in ('strict', 'paper'):
        for ds in DS:
            ours = C[(C.dataset == ds) & (C['mode'] == 'full') & (C.prep == prep) & (C.model == 'pinn_lai')]
            ours = ours.groupby('cell').mae.mean()
            for ref in ('PINN4SOH',):
                theirs = R[(R.dataset == ds) & (R.model == ref)].set_index('cell').MAE
                cells = sorted(set(ours.index) & set(theirs.index))
                if len(cells) < 3:
                    continue
                d = ours.reindex(cells).values - theirs.reindex(cells).values
                out.append(dict(dataset=ds, prep=prep, ref=ref, n_cells=len(cells), ours=ours.reindex(cells).mean(),
                                theirs=theirs.reindex(cells).mean(), better=int((d < 0).sum()),
                                p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pd.DataFrame(out)


def figure(S):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save, SERIES
    apply_theme()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.65, wspace=0.42)
    # (trên) chế độ đầy đủ: bốn cột cho mỗi bộ
    bars = [('PINN4SOH (công bố)', None, '#9aa5ad'), ('hộp đen, tiền xử lý của bài', ('paper', 'pinn_bb'), '#c9b27c'),
            ('PINN-lai, tiền xử lý của bài', ('paper', 'pinn_lai'), '#86b6ef'),
            ('hộp đen, không rò rỉ', ('strict', 'pinn_bb'), '#eda100'),
            ('PINN-lai, không rò rỉ', ('strict', 'pinn_lai'), '#0b3d73')]
    for i, ds in enumerate(DS):
        ax = fig.add_subplot(gs[0, i])
        vals = []
        for lab, sel, col in bars:
            if sel is None:
                v = PAPER_MAE[(ds, 'full')]['PINN4SOH']
            else:
                r = S[(S.dataset == ds) & (S['mode'] == 'full') & (S.prep == sel[0]) & (S.model == sel[1])]
                v = float(r.mae.iloc[0]) if len(r) else np.nan
            vals.append((v, col, lab))
        ax.bar(range(len(vals)), [1e3 * v for v, _, _ in vals], color=[c for _, c, _ in vals], width=0.75)
        ax.set_xticks([]); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.tick_params(labelsize=6.5)
        panel_title(ax, ds, pad=4)
        if i == 0:
            ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
    hd = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, _, c in bars]
    fig.legend(handles=hd, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.535), fontsize=6.5)
    # (dưới) chế độ ít mẫu: MAE theo số cell có nhãn
    for i, ds in enumerate(('XJTU', 'HUST')):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        n = [1, 2, 3, 4]
        lines = [('PINN4SOH (công bố)', lambda k: PAPER_MAE[(ds, f'n{k}')]['PINN4SOH'], '#9aa5ad', '--'),
                 ('MLP (công bố)', lambda k: PAPER_MAE[(ds, f'n{k}')]['MLP'], '#c3cbd1', ':'),
                 ('PINN-lai, tiền xử lý của bài', lambda k: _get(S, ds, f'n{k}', 'paper', 'pinn_lai'), '#86b6ef', '-'),
                 ('PINN-lai, không rò rỉ', lambda k: _get(S, ds, f'n{k}', 'strict', 'pinn_lai'), '#0b3d73', '-'),
                 ('GBDT, không rò rỉ', lambda k: _get(S, ds, f'n{k}', 'strict', 'gbdt'), '#1baf7a', '-')]
        for lab, fn, col, ls in lines:
            ax.plot(n, [1e3 * fn(k) for k in n], color=col, ls=ls, marker='o', ms=3, lw=1.3, label=lab)
        ax.set_yscale('log'); ax.set_xticks(n)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in (3, 5, 8, 10, 15, 20, 30, 50, 80, 120) if lo <= t <= hi])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        ax.set_xlabel('số cell có nhãn' + (' mỗi batch' if ds == 'XJTU' else ''), fontsize=7)
        panel_title(ax, f'{ds}, ít mẫu', pad=4)
        if i == 0:
            ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    save(fig, f'{OUT}/fig_e27_compare')


def _get(S, ds, mode, prep, model):
    r = S[(S.dataset == ds) & (S['mode'] == mode) & (S.prep == prep) & (S.model == model)]
    return float(r.mae.iloc[0]) if len(r) else np.nan


def main():
    C = build()
    if C.empty:
        print('chưa có nhóm nào đủ kết quả')
        return
    os.makedirs(OUT, exist_ok=True)
    C.to_csv(f'{OUT}/e27_cells.csv', index=False)
    S = summarize(C); S.to_csv(f'{OUT}/e27_summary.csv', index=False)
    T = tests(C); T.to_csv(f'{OUT}/e27_tests.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print((S.assign(mae=1e3 * S.mae, paper=1e3 * S.paper_PINN4SOH)
               [['dataset', 'mode', 'prep', 'model', 'mae', 'paper', 'ratio_vs_pinn4soh', 'n_groups']]).round(3).to_string(index=False))
        print(T.round(4).to_string(index=False))
    figure(S)


if __name__ == '__main__':
    main()
