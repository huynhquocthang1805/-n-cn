"""
E20: đánh giá mô hình cải tiến PINN-lai trên test, phân rã lợi ích và khoảng dự đoán phù hợp.

    python analysis/analyze_e20.py    # -> results/e20_*.csv, results/fig_e20_*.png|pdf

Cấu hình chốt trong docs/de-cuong-E20.md. Mọi mô hình so trên đúng 3 lần lặp đầu (0, 1, 2):
  pinn       PINN-semi đơn lẻ (thành viên 0, trùng E13)
  pinn3      trung bình 3 PINN-semi
  gbdt       GBDT (trùng E16)
  mlp        MLP đơn lẻ (E13)
  lai_thô    ½ pinn3 + ½ gbdt
  pinn_lai   lai_thô qua bộ lọc quỹ đạo w = 10            ← mô hình cải tiến
  mlp_lai    ½ (trung bình 3 MLP) + ½ gbdt, qua bộ lọc     ← cùng công thức, không có vật lý
  gbdt_loc   gbdt qua bộ lọc
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
from pinnsoh.runcache import cached_map

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
REPEATS = (0, 1, 2)
W, ALPHA = 10, 0.10
E13_ARM = {('mlp', 0.1): 'L', ('mlp', 0.3): 'A', ('mlp', 0.5): 'L', ('mlp', 0.7): 'F'}
LABEL = {'mlp': 'MLP', 'gbdt': 'GBDT', 'pinn': 'PINN-semi', 'pinn3': '3 PINN-semi', 'lai_tho': '3 PINN-semi + GBDT',
         'pinn_lai': 'PINN-lai', 'mlp_lai': 'MLP-lai', 'gbdt_loc': 'GBDT + lọc'}
RNG = np.random.default_rng(2026)
B = 10_000


def _pred(path, val=False):
    d = json.load(open(path))
    tp = d['val_predictions' if val else 'test_predictions']
    ids = tp['cell_ids']
    return dict(y=np.asarray(tp['y'], float), p=np.asarray(tp['p'], float), cell=np.asarray([ids[k] for k in tp['cell']]),
                cyc=np.asarray(tp['cycle'], float))


def _both(path):
    d = json.load(open(path))
    out = {'test': _pred(path)}
    if 'val_predictions' in d:
        out['val'] = _pred(path, val=True)
    return out


def files():
    F = {}
    for f, r, ds, k in [(f, r, ds, k) for f in FRACS for r in REPEATS for ds in DS for k in range(5)]:
        s = 100 * r + k
        key = (ds, f, r, k)
        F[key] = dict(
            P=[f'runs/E20/P_{ds}_pinn_semi_f{f}_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}') + '.json'
               for m in range(3)],
            M=[f'runs/E13/{E13_ARM[("mlp", f)]}_{ds}_mlp_f{f}_s{s}_global_global_cv{r}-{k}of5.json']
              + [f'runs/E20/M_{ds}_mlp_f{f}_s{s}_global_global_cv{r}-{k}of5_m{m}.json' for m in (1, 2)],
            G=[f'runs/E20/gbdt_{ds}_f{f}_cv{r}-{k}of5.json'],
            E16=[f'runs/E16/gbdt_{ds}_f{f}_cv{r}-{k}of5.json'],
            E13=[f'runs/E13/{"L" if f in (0.1, 0.5) else ("C" if f == 0.3 else "G")}_{ds}_pinn_semi_f{f}_s{s}'
                 f'_global_global_cv{r}-{k}of5.json'])
        if os.environ.get('E20_SMOKE'):            # chạy thử khi nhánh MLP chưa xong: dùng tạm PINN thay MLP
            F[key]['M'] = F[key]['P']
    return F


def build():
    """Dự đoán của mọi mô hình trên test (và validation khi có) cho từng (bộ, mức nhãn, lần lặp, fold)."""
    F = files()
    need = sorted({p for v in F.values() for ps in v.values() for p in ps if os.path.exists(p)})
    missing = sum(1 for v in F.values() for ps in v.values() for p in ps if not os.path.exists(p))
    if missing:
        print(f'cảnh báo: thiếu {missing} file — chỉ dùng các fold đủ dữ liệu')
    data = dict(zip(need, cached_map(need, _both, 'e20_preds_v1')))
    out, check = {}, []
    for key, v in F.items():
        if not all(os.path.exists(p) for ps in v.values() for p in ps):
            continue
        P = [data[p] for p in v['P']]; M = [data[p] for p in v['M']]; G = data[v['G'][0]]
        # thành viên 0 phải trùng E13, GBDT phải trùng E16 (cùng cấu hình, cùng hạt giống)
        check.append(dict(ds=key[0], frac=key[1], r=key[2], k=key[3],
                          pinn_same=bool(np.array_equal(P[0]['test']['p'], data[v['E13'][0]]['test']['p'])),
                          gbdt_same=bool(np.array_equal(G['test']['p'], data[v['E16'][0]]['test']['p']))))
        rec = {}
        for part in ('test', 'val'):
            base = P[0][part]
            y, cell, cyc = base['y'], base['cell'], base['cyc']
            for o in P[1:] + [G]:
                assert np.array_equal(o[part]['cell'], cell) and np.array_equal(o[part]['cyc'], cyc), key
            pinn3 = np.mean([m[part]['p'] for m in P], axis=0)
            g = G[part]['p']
            lai = 0.5 * pinn3 + 0.5 * g
            pr = dict(pinn=P[0][part]['p'], pinn3=pinn3, gbdt=g, lai_tho=lai,
                      pinn_lai=by_cell(causal_trend, lai, cell, cyc, w=W),
                      gbdt_loc=by_cell(causal_trend, g, cell, cyc, w=W),
                      sigma=np.abs(pinn3 - g))
            if part == 'test':
                for o in M:
                    assert np.array_equal(o['test']['cell'], cell) and np.array_equal(o['test']['cyc'], cyc), key
                mlp3 = np.mean([m['test']['p'] for m in M], axis=0)
                pr.update(mlp=M[0]['test']['p'],
                          mlp_lai=by_cell(causal_trend, 0.5 * mlp3 + 0.5 * g, cell, cyc, w=W))
            rec[part] = dict(y=y, cell=cell, cyc=cyc, pred=pr)
        out[key] = rec
    return out, pd.DataFrame(check)


MODELS = ['mlp', 'gbdt', 'pinn', 'pinn3', 'lai_tho', 'pinn_lai', 'mlp_lai', 'gbdt_loc']


def per_cell(P):
    rows = []
    for (ds, f, r, k), rec in P.items():
        t = rec['test']
        for m in MODELS:
            p = t['pred'][m]
            for c in np.unique(t['cell']):
                i = t['cell'] == c
                o = np.argsort(t['cyc'][i]); pp, yy = p[i][o], t['y'][i][o]
                e = pp - yy
                rows.append(dict(dataset=ds, frac=f, repeat=r, model=m, cell=c, mae=np.abs(e).mean(),
                                 rmse=np.sqrt((e ** 2).mean()), bias=e.mean()))
    return pd.DataFrame(rows)


def pooled(P):
    """R² gộp 5 fold trong một lần lặp (bỏ 10 chu kỳ đầu), rồi trung bình qua lần lặp; tỉ lệ cặp tăng ngược."""
    acc = {}
    for (ds, f, r, k), rec in P.items():
        t = rec['test']; keep = t['cyc'] >= 10
        for m in MODELS:
            p = t['pred'][m]; e = (p - t['y'])[keep]; y = t['y'][keep]
            a = acc.setdefault((ds, f, m, r), [0, 0.0, 0.0, 0.0, 0, 0])
            a[0] += len(y); a[1] += float((e ** 2).sum()); a[2] += float(y.sum()); a[3] += float((y ** 2).sum())
            for c in np.unique(t['cell']):
                i = np.flatnonzero(t['cell'] == c); i = i[np.argsort(t['cyc'][i])]
                cc, pp = t['cyc'][i], p[i]
                j = np.searchsorted(cc, cc + 10); ok = j < len(cc)
                a[4] += int(ok.sum()); a[5] += int((pp[j[ok]] > pp[np.flatnonzero(ok)] + 0.002).sum())
    rows = []
    for (ds, f, m, r), (n, sse, sy, syy, npair, viol) in acc.items():
        rows.append(dict(dataset=ds, frac=f, model=m, repeat=r, R2_10=1 - sse / (syy - sy * sy / n),
                         tang_nguoc=100 * viol / max(1, npair)))
    return pd.DataFrame(rows).groupby(['dataset', 'frac', 'model'])[['R2_10', 'tang_nguoc']].mean().reset_index()


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); out[j] = run
    return out


def bias_corr(pc):
    """Tương quan (Pearson) giữa lệch mức theo cell của trung bình 3 PINN-semi và của GBDT."""
    rows = []
    for (ds, f), h in pc[pc.model.isin(['pinn3', 'gbdt'])].groupby(['dataset', 'frac']):
        w = h.pivot_table(index=['repeat', 'cell'], columns='model', values='bias')
        rows.append(dict(dataset=ds, frac=f, rho=float(np.corrcoef(w.pinn3, w.gbdt)[0, 1]),
                         ti_le_ngc_dau=float((np.sign(w.pinn3) != np.sign(w.gbdt)).mean())))
    return pd.DataFrame(rows)


def summary(pc, pool):
    g = pc.groupby(['dataset', 'frac', 'model', 'cell'])[['mae', 'rmse']].mean().reset_index()
    s = g.groupby(['dataset', 'frac', 'model'])[['mae', 'rmse']].mean().reset_index()
    s = s.merge(pool, on=['dataset', 'frac', 'model'])
    s = s.rename(columns={'mae': 'MAE_cell', 'rmse': 'RMSE_cell'})
    return s, g


def tests(g):
    rows = []
    for other in ['pinn', 'gbdt', 'mlp', 'pinn3', 'lai_tho', 'mlp_lai', 'gbdt_loc']:
        for f in FRACS:
            block = []
            for ds in DS:
                a = g[(g.model == other) & (g.frac == f) & (g.dataset == ds)].set_index('cell').mae
                b = g[(g.model == 'pinn_lai') & (g.frac == f) & (g.dataset == ds)].set_index('cell').mae
                common = a.index.intersection(b.index)
                x, y = a.loc[common].values, b.loc[common].values
                if len(x) < 6:
                    continue
                idx = RNG.integers(0, len(x), size=(B, len(x))); rr = y[idx].mean(1) / x[idx].mean(1)
                block.append(dict(doi_chung=other, frac=f, bo=ds, n_cell=len(x), ti_so=y.mean() / x.mean(),
                                  lo=np.percentile(rr, 2.5), hi=np.percentile(rr, 97.5),
                                  cell_tot_hon=float((y < x).mean()), p=stats.wilcoxon(y, x).pvalue))
            for bl, ph in zip(block, holm(np.array([b['p'] for b in block])) if block else []):
                bl['p_holm'] = ph
            rows += block
    return pd.DataFrame(rows)


def conformal(P):
    """Khoảng dự đoán 90 %: phân vị phần dư trên validation của chính fold, áp lên test của fold đó."""
    rows = []
    for (ds, f, r, k), rec in P.items():
        v, t = rec['val'], rec['test']
        rv = np.abs(v['pred']['pinn_lai'] - v['y']); rt = np.abs(t['pred']['pinn_lai'] - t['y'])
        n = len(rv); q = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        qa = np.quantile(rv, q, method='higher')
        # dạng thích nghi: chia cho độ lệch giữa trung bình PINN và GBDT (cộng trung vị để không chia cho ~0)
        sv, st = v['pred']['sigma'], t['pred']['sigma']
        d0 = np.median(sv) + 1e-6
        qs = np.quantile(rv / (sv + d0), q, method='higher')
        for kind, cover, width in (('co_dinh', rt <= qa, np.full(len(rt), 2 * qa)),
                                   ('thich_nghi', rt <= qs * (st + d0), 2 * qs * (st + d0))):
            cells = pd.Series(cover).groupby(t['cell']).mean()
            rows.append(dict(dataset=ds, frac=f, repeat=r, fold=k, kieu=kind, n=len(rt), phu=float(cover.mean()),
                             rong=float(width.mean()), phu_cell_min=float(cells.min()),
                             phu_cell_duoi80=float((cells < 0.8).mean())))
    d = pd.DataFrame(rows)
    w = lambda h: pd.Series(dict(phu=np.average(h.phu, weights=h.n), rong=np.average(h.rong, weights=h.n),
                                 phu_cell_duoi80=h.phu_cell_duoi80.mean()))
    return d.groupby(['dataset', 'frac', 'kieu']).apply(w, include_groups=False).reset_index()


def main():
    P, check = build()
    print(f'{len(P)} tổ hợp đủ dữ liệu; thành viên 0 trùng E13: {check.pinn_same.sum()}/{len(check)}; '
          f'GBDT trùng E16: {check.gbdt_same.sum()}/{len(check)}')
    check.to_csv(f'{OUT}/e20_check.csv', index=False)
    pc = per_cell(P)
    pool = pooled(P)
    s, g = summary(pc, pool)
    ts = tests(g)
    cf = conformal(P)
    bc = bias_corr(pc)
    bc.to_csv(f'{OUT}/e20_biascorr.csv', index=False)
    print(bc.round(3).to_string(index=False))
    s.to_csv(f'{OUT}/e20_summary.csv', index=False)
    ts.to_csv(f'{OUT}/e20_tests.csv', index=False)
    cf.to_csv(f'{OUT}/e20_conformal.csv', index=False)
    g.to_csv(f'{OUT}/e20_percell.csv.gz', index=False)
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    t = s.pivot_table(index=['dataset', 'model'], columns='frac', values='MAE_cell') * 1e3
    print(t.round(2).to_string())
    print(ts[ts.doi_chung.isin(['pinn', 'gbdt', 'mlp', 'mlp_lai'])].round(4).to_string(index=False))
    print(cf.round(4).to_string(index=False))
    figures(s, g, cf, P)


# ───────────────────────────────────────────────────────────── hình
def figures(s, g, cf, P):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, CHEM, panel_title, save, log_mae_axis, MODEL_COLOR
    apply_theme()
    COL = {'mlp': MODEL_COLOR['mlp'], 'gbdt': '#7b5cd6', 'pinn': MODEL_COLOR['pinn_semi'], 'pinn_lai': '#0b4f3c',
           'mlp_lai': '#8fb8e8', 'pinn3': '#6fcfa8', 'lai_tho': '#2f8f6c', 'gbdt_loc': '#b6a3ea'}
    x = [10, 30, 50, 70]

    # 1. MAE theo cell theo lượng nhãn: MLP, GBDT, PINN-semi, PINN-lai
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    for ax, ds in zip(axes, DS):
        vals = []
        for m in ['mlp', 'gbdt', 'pinn', 'pinn_lai']:
            h = s[(s.dataset == ds) & (s.model == m)].sort_values('frac')
            v = 1e3 * h.MAE_cell.values; vals += list(v)
            ax.plot(x, v, color=COL[m], lw=2.0 if m == 'pinn_lai' else 1.3, marker='o', ms=3.6 if m == 'pinn_lai' else 3,
                    mec='white', mew=0.5, label=LABEL[m], zorder=3 if m == 'pinn_lai' else 2)
        lo, hi = min(vals) / 1.15, max(vals) * 1.15
        log_mae_axis(ax, ticks=[t for t in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40] if lo <= t <= hi], lo=lo, hi=hi)
        ax.set_xticks(x); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.text(0.5, 0.105, 'cell có nhãn (% tổng số cell)', ha='center', fontsize=8.5)
    save(fig, f'{OUT}/fig_e20_curve')

    # 2. phân rã: tỉ số MAE so với PINN-semi qua từng bước, trung bình nhân trên 4 bộ
    steps = ['pinn', 'pinn3', 'lai_tho', 'pinn_lai']
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    for ax, ds in zip(axes, DS):
        for j, f in enumerate(FRACS):
            base = s[(s.dataset == ds) & (s.frac == f) & (s.model == 'pinn')].MAE_cell.iloc[0]
            v = [s[(s.dataset == ds) & (s.frac == f) & (s.model == m)].MAE_cell.iloc[0] / base for m in steps]
            ax.plot(range(4), v, color=plt.cm.Greens(0.45 + 0.17 * j), lw=1.4, marker='o', ms=3, mec='white', mew=0.4,
                    label=f'{int(100 * f)} %')
            gb = s[(s.dataset == ds) & (s.frac == f) & (s.model == 'gbdt')].MAE_cell.iloc[0] / base
            ax.plot([4.1 + 0.12 * (j - 1.5)], [gb], marker='D', ms=3.2, color=plt.cm.Purples(0.45 + 0.17 * j), ls='none',
                    mec='white', mew=0.4)
        ax.axhline(1, color='#9aa5ad', lw=0.7, ls=(0, (2, 2)))
        ax.axvline(3.55, color='#e3e8e6', lw=0.8)
        ax.set_xticks([0, 1, 2, 3, 4.1]); ax.set_xticklabels(['PINN', '×3', '+GBDT', '+lọc', 'GBDT'], fontsize=6.6)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.set_xlim(-0.3, 4.5)
        panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('MAE / MAE của PINN-semi')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), title=None)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, f'{OUT}/fig_e20_steps')

    # 3. khoảng dự đoán: hàng trên độ phủ, hàng dưới độ rộng; mỗi cột một bộ
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), sharex=True)
    for j, ds in enumerate(DS):
        for kind, mk, col, lab in (('co_dinh', 's', '#9aa5ad', 'độ rộng cố định'),
                                   ('thich_nghi', 'o', '#0b4f3c', 'độ rộng thích nghi')):
            hh = cf[(cf.kieu == kind) & (cf.dataset == ds)].sort_values('frac')
            axes[0, j].plot(x, 100 * hh.phu, marker=mk, ms=3.6, lw=1.2, color=col, mec='white', mew=0.4,
                            label=lab if j == 0 else None)
            axes[1, j].plot(x, 100 * hh.rong, marker=mk, ms=3.6, lw=1.2, color=col, mec='white', mew=0.4)
        axes[0, j].axhline(90, color='#1b2330', lw=0.8, ls=(0, (3, 2)))
        axes[0, j].set_ylim(80, 95)
        axes[1, j].set_ylim(bottom=0)
        panel_title(axes[0, j], ds, CHEM[ds], pad=4)
        for ax in axes[:, j]:
            ax.set_xticks(x); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.locator_params(axis='y', nbins=4)
    axes[0, 0].set_ylabel('độ phủ (%)'); axes[1, 0].set_ylabel('độ rộng (% SOH)')
    fig.legend(loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.text(0.5, 0.075, 'cell có nhãn (% tổng số cell)', ha='center', fontsize=8.5)
    save(fig, f'{OUT}/fig_e20_conformal')

    # 4. quỹ đạo của cùng một cell ở 10, 30, 70 % nhãn (lần lặp 0), kèm khoảng 90 % độ rộng cố định. Cell chọn như
    #    hình quỹ đạo của Chương 6: MAE của MLP 30 % ở trung vị, bỏ cell có nhãn nhảy quá 0,02.
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.6), sharey='row')
    for i, ds in enumerate(DS):
        errs = []
        for k in range(5):
            t = P[(ds, 0.3, 0, k)]['test']
            for c in np.unique(t['cell']):
                sel = t['cell'] == c; o = np.argsort(t['cyc'][sel]); y = t['y'][sel][o]
                if np.abs(np.diff(y)).max() <= 0.02:
                    errs.append((np.abs(t['pred']['mlp'][sel] - t['y'][sel]).mean(), k, c))
        errs.sort()
        _, k, c = errs[(len(errs) - 1) // 2]
        for j, fr in enumerate((0.1, 0.3, 0.7)):
            ax = axes[i, j]; rec = P[(ds, fr, 0, k)]; t, v = rec['test'], rec['val']
            sel = t['cell'] == c; o = np.argsort(t['cyc'][sel])
            cyc = t['cyc'][sel][o]; y = t['y'][sel][o]
            pl = t['pred']['pinn_lai'][sel][o]; pp = t['pred']['pinn'][sel][o]
            rv = np.abs(v['pred']['pinn_lai'] - v['y'])
            n = len(rv); half = np.quantile(rv, min(1, np.ceil((n + 1) * 0.9) / n), method='higher')
            ax.fill_between(cyc, pl - half, pl + half, color='#0b4f3c', alpha=0.12, lw=0)
            ax.plot(cyc, y, color=MODEL_COLOR['truth'], lw=1.6, zorder=3)
            ax.plot(cyc, pp, color=COL['pinn'], lw=0.75, alpha=0.85, zorder=2)
            ax.plot(cyc, pl, color=COL['pinn_lai'], lw=1.1, zorder=4)
            lo, hi = y.min(), y.max(); pad = max(0.03, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
            panel_title(ax, ds, f'{int(round(100 * fr))} % nhãn', pad=3)
            if j == 0:
                ax.set_ylabel('SOH')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.5, label='thực đo'),
                        plt.Line2D([], [], color=COL['pinn'], lw=1.0, label='PINN-semi'),
                        plt.Line2D([], [], color=COL['pinn_lai'], lw=1.3, label='PINN-lai'),
                        plt.Rectangle((0, 0), 1, 1, color='#0b4f3c', alpha=0.13, label='khoảng 90 %')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.035, 'chu kỳ', ha='center', fontsize=8.5)
    save(fig, f'{OUT}/fig_e20_traj')


def figs_only():
    """Vẽ lại hình từ results/e20_*.csv và dự đoán đã nhớ đệm, không in lại bảng."""
    P, _ = build()
    s = pd.read_csv(f'{OUT}/e20_summary.csv'); cf = pd.read_csv(f'{OUT}/e20_conformal.csv')
    figures(s, None, cf, P)


if __name__ == '__main__':
    figs_only() if '--fig' in _sys.argv else main()
