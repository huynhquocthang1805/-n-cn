"""
R², RMSE và các chỉ số tính hợp lý vật lý của dự đoán, cho mọi mô hình và mọi mức nhãn.

    python analysis/analyze_metrics.py    # -> results/metrics_*.csv, results/fig_metrics_*.png|pdf

Đọc dự đoán test đã lưu của E13, E16, E17 (cùng cách chia, cùng tập nhãn) và kết quả chuyển miền E14/E18/E19.
Không huấn luyện lại.

  RMSE_cell   căn trung bình bình phương sai số của từng cell, trung bình theo cell (mỗi cell một phiếu)
  RMSE, R²    gộp mọi chu kỳ của mọi cell: trong một lần lặp, 5 fold phủ mỗi cell đúng một lần, nên ghép 5
              fold thành dự đoán cho TOÀN BỘ bộ dữ liệu rồi tính; báo trung bình ± độ lệch chuẩn qua 5 lần lặp
  R²_10       như R² nhưng bỏ 10 chu kỳ đầu của mỗi cell (vài chu kỳ đầu của XJTU có đặc trưng bất thường)
  vi phạm đơn điệu   tỉ lệ cặp chu kỳ (n, n') cách nhau ≥ 10 chu kỳ của cùng cell mà SOH dự đoán TĂNG quá
              0,002 — cùng ngưỡng với hàm mất mát đơn điệu; tính cả trên nhãn thật để có mức tham chiếu
  ngoài miền  tỉ lệ chu kỳ có SOH dự đoán ngoài [0,40; 1,15]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = 'results'
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
MODELS = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']
LABEL = {'mlp': 'MLP', 'ridge': 'Ridge', 'gbdt': 'GBDT', 'gpr': 'GPR', 'pinn_sup': 'PINN-sup',
         'pinn_bb': 'PINN hộp đen', 'pinn_semi': 'PINN-semi', 'mono_only': 'chỉ đơn điệu', 'euler': 'PINN-semi (Euler)'}
COLOR = {'mlp': '#2a78d6', 'pinn_sup': '#eb6834', 'pinn_semi': '#1baf7a', 'pinn_bb': '#eda100',
         'ridge': '#e87ba4', 'gbdt': '#7b5cd6', 'gpr': '#8a5a00'}
H_MIN, EPS, LO, HI = 10, 0.002, 0.40, 1.15
RNG = np.random.default_rng(2026)
B = 10_000


def _parse(path):
    """Một lượt -> tổng đủ để ghép fold (n, Σe², Σy, Σy²...) và chỉ số theo cell."""
    r = json.load(open(path)); c = r['config']; tp = r.get('test_predictions')
    if tp is None:
        return None
    model = {'D': 'mono_only', 'E': 'euler'}.get(c.get('tag', ''), c['model'])
    y = np.asarray(tp['y'], float); p = np.asarray(tp['p'], float)
    cell = np.asarray(tp['cell']); cyc = np.asarray(tp['cycle']); ids = tp['cell_ids']
    e = p - y; k10 = cyc >= 10
    agg = dict(n=len(y), sse=float((e ** 2).sum()), sy=float(y.sum()), syy=float((y ** 2).sum()),
               n10=int(k10.sum()), sse10=float((e[k10] ** 2).sum()), sy10=float(y[k10].sum()),
               syy10=float((y[k10] ** 2).sum()), out=int(((p < LO) | (p > HI)).sum()))
    cells = []
    for k in np.unique(cell):
        m = cell == k
        yy, pp, cc = y[m], p[m], cyc[m]
        j = np.searchsorted(cc, cc + H_MIN)                 # chu kỳ đầu tiên cách ít nhất H_MIN chu kỳ
        ok = j < len(cc)
        i, j = np.nonzero(ok)[0], j[ok]
        cells.append((ids[k], int(m.sum()), float(((pp - yy) ** 2).sum()), int(len(i)),
                      int((pp[j] > pp[i] + EPS).sum()), int((yy[j] > yy[i] + EPS).sum())))
    return dict(model=model, frac=c['label_frac'], dataset=c['dataset'], repeat=c['repeat'], fold=c['fold'],
                agg=agg, cells=cells)


def load():
    from pinnsoh.runcache import cached_map
    files = sorted(glob.glob('runs/E13/*.json') + glob.glob('runs/E16/*.json') + glob.glob('runs/E17/*.json'))
    recs = [x for x in cached_map(files, _parse, 'metrics_records_v1') if x is not None]
    # ở 70 % nhãn mọi cell train đều có nhãn: PINN-sup trùng từng bit với PINN-semi (không chạy lại)
    recs += [dict(x, model='pinn_sup') for x in recs if x['model'] == 'pinn_semi' and x['frac'] == 0.7]
    runs = pd.DataFrame([dict({k: x[k] for k in ('model', 'frac', 'dataset', 'repeat', 'fold')}, **x['agg'])
                         for x in recs])
    cells = pd.DataFrame([dict(model=x['model'], frac=x['frac'], dataset=x['dataset'], repeat=x['repeat'],
                               cell=cid, n=n, sse=sse, npair=npair, viol=v, viol_true=vt)
                          for x in recs for cid, n, sse, npair, v, vt in x['cells']])
    return runs, cells


def r2(sse, sy, syy, n):
    sst = syy - sy ** 2 / n
    return 1 - sse / sst


def summary(runs, cells):
    """Một hàng cho mỗi (bộ, mô hình, mức nhãn)."""
    rep = runs.groupby(['dataset', 'model', 'frac', 'repeat']).sum(numeric_only=True).reset_index()
    rep['R2'] = r2(rep.sse, rep.sy, rep.syy, rep.n)
    rep['R2_10'] = r2(rep.sse10, rep.sy10, rep.syy10, rep.n10)
    rep['RMSE'] = np.sqrt(rep.sse / rep.n)
    g = rep.groupby(['dataset', 'model', 'frac'])
    out = g.agg(R2=('R2', 'mean'), R2_sd=('R2', 'std'), R2_10=('R2_10', 'mean'), RMSE=('RMSE', 'mean'),
                RMSE_sd=('RMSE', 'std'), n_lap=('repeat', 'nunique'), out=('out', 'sum'), n=('n', 'sum')).reset_index()
    out['ngoai_mien_pct'] = 100 * out.out / out.n
    cells = cells.assign(rmse=np.sqrt(cells.sse / cells.n))
    pc = cells.groupby(['dataset', 'model', 'frac', 'cell']).agg(rmse=('rmse', 'mean'), viol=('viol', 'sum'),
                                                                 viol_true=('viol_true', 'sum'),
                                                                 npair=('npair', 'sum')).reset_index()
    cm = pc.groupby(['dataset', 'model', 'frac']).agg(RMSE_cell=('rmse', 'mean'), viol=('viol', 'sum'),
                                                       viol_true=('viol_true', 'sum'), npair=('npair', 'sum')).reset_index()
    cm['vi_pham_don_dieu_pct'] = 100 * cm.viol / cm.npair
    cm['vi_pham_nhan_that_pct'] = 100 * cm.viol_true / cm.npair
    out = out.merge(cm[['dataset', 'model', 'frac', 'RMSE_cell', 'vi_pham_don_dieu_pct', 'vi_pham_nhan_that_pct']],
                    on=['dataset', 'model', 'frac'])
    return out.drop(columns=['out', 'n']), pc


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (m - rank) * p[i]); adj[i] = min(1.0, prev)
    return adj


def tests(pc):
    """PINN-semi so với từng mô hình: tỉ số RMSE theo cell, KTC 95 % bootstrap ghép cặp, Wilcoxon, Holm trên 4 bộ."""
    rows = []
    for other in [m for m in MODELS if m != 'pinn_semi'] + ['mono_only', 'euler']:
        for f in FRACS:
            block = []
            for ds in DATASETS:
                a = pc[(pc.model == other) & (pc.frac == f) & (pc.dataset == ds)].set_index('cell').rmse
                b = pc[(pc.model == 'pinn_semi') & (pc.frac == f) & (pc.dataset == ds)].set_index('cell').rmse
                common = a.index.intersection(b.index)
                if len(common) < 6 or (other == 'pinn_sup' and f == 0.7):
                    continue
                x, y = a.loc[common].values, b.loc[common].values
                idx = RNG.integers(0, len(x), size=(B, len(x))); rr = y[idx].mean(1) / x[idx].mean(1)
                block.append(dict(doi_chung=other, frac=f, bo=ds, ti_so=y.mean() / x.mean(), lo=np.percentile(rr, 2.5),
                                  hi=np.percentile(rr, 97.5), cell_pinn_tot_hon=float((y < x).mean()),
                                  p=stats.wilcoxon(y, x).pvalue))
            for bl, ph in zip(block, holm([x['p'] for x in block]) if block else []):
                bl['p_holm'] = ph
            rows += block
    return pd.DataFrame(rows)


def transfer():
    rows = []
    for f in glob.glob('runs/E14/samenorm_*.json'):
        r = json.load(open(f)); c = r['config']; t = r['target_test']
        rows.append(dict(nguon=c['dataset'], dich=c['target'], k=c['k_target'], model=c['model'], seed=c['seed'],
                         R2=t['R2'], RMSE=t['RMSE'], MAE_cell=t['MAE_cell']))
    d = pd.DataFrame(rows)
    return d.groupby(['nguon', 'dich', 'k', 'model'])[['R2', 'RMSE', 'MAE_cell']].mean().reset_index()


def figures(sm):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save, log_mae_axis
    apply_theme()
    CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}
    x = [10, 30, 50, 70]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6))
    for j, ds in enumerate(DATASETS):
        for i, (col, lab) in enumerate([('RMSE_cell', 'RMSE theo cell (×10$^{-3}$)'), ('R2_10', '$R^2$')]):
            ax = axes[i, j]
            for m in MODELS:
                g = sm[(sm.dataset == ds) & (sm.model == m)].sort_values('frac')
                if g.empty:
                    continue
                v = 1e3 * g[col] if col == 'RMSE_cell' else g[col]
                ax.plot(100 * g.frac, v, marker='o' if m.startswith('pinn') else 's', ms=3.2, lw=1.4,
                        color=COLOR[m], label=LABEL[m])
            ax.set_xticks(x); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
            if col == 'RMSE_cell':
                vals = 1e3 * sm[(sm.dataset == ds) & (sm.model.isin(MODELS))].RMSE_cell
                lo, hi = vals.min() / 1.15, vals.max() * 1.15
                ticks = [t for t in [1, 2, 3, 5, 10, 15, 20, 30, 50] if lo <= t <= hi]
                log_mae_axis(ax, ticks=ticks, lo=lo, hi=hi)
                panel_title(ax, ds, CHEM[ds], pad=4)
            if j == 0:
                ax.set_ylabel(lab)
    fig.supxlabel('cell có nhãn (% tổng số cell)', fontsize=8.5, y=0.055)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, f'{OUT}/fig_metrics_rmse_r2')

    # tỉ lệ vi phạm đơn điệu ở 30 % nhãn, so với chính nhãn thật
    order = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'mono_only', 'pinn_semi']
    cols = dict(COLOR, mono_only='#5fc9a0')
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    w = 0.1
    for i, m in enumerate(order):
        v = [sm[(sm.dataset == ds) & (sm.model == m) & (sm.frac == 0.3)].vi_pham_don_dieu_pct.iloc[0] for ds in DATASETS]
        ax.bar(np.arange(4) + (i - 3.5) * w, v, width=w * 0.9, color=cols[m], label=LABEL[m])
    true = [sm[(sm.dataset == ds) & (sm.model == 'mlp') & (sm.frac == 0.3)].vi_pham_nhan_that_pct.iloc[0] for ds in DATASETS]
    for k, t in enumerate(true):
        ax.hlines(t, k - 0.45, k + 0.45, color='#1b2330', lw=1.2, label='nhãn thật' if k == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(DATASETS); ax.set_ylabel('cặp tăng ngược (%)')
    ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.legend(ncol=5, fontsize=6.5, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.48), handlelength=1.2)
    fig.tight_layout()
    save(fig, f'{OUT}/fig_metrics_mono')


def main():
    runs, cells = load()
    sm, pc = summary(runs, cells)
    ts = tests(pc)
    tr = transfer()
    sm.to_csv(f'{OUT}/metrics_summary.csv', index=False)
    ts.to_csv(f'{OUT}/metrics_tests.csv', index=False)
    tr.to_csv(f'{OUT}/metrics_transfer.csv', index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 400)
    print(sm.round(4).to_string(index=False))
    print(ts.round(4).to_string(index=False))
    figures(sm)
    print('xong')


if __name__ == '__main__':
    main()
