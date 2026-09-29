"""
Phân tích E13 (và E14) đúng theo đề cương ấn định trước — tai-lieu/de-cuong-E13.md.

    python analyze_e13.py            # runs/E13/*.json, runs/E14/*.json -> results/E13_*, E14_*, fig_E13_*

Mọi chỉ số được TÍNH LẠI từ test_predictions đã lưu (y, p, chu kỳ gốc, id cell), không đọc số tóm tắt.
Bảng trung gian results/E13_percell.csv (một hàng = một cell × một lượt huấn luyện) đủ để kiểm lại
mọi con số mà không cần 1 100 file JSON.

Đơn vị phân tích chính là CELL: e_a(c) = trung bình qua R lần lặp của MAE cell c dưới nhánh a.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinnsoh.metrics import EOL_THRESHOLD, LATE_SOH

RUNS, OUT = 'runs/E13', 'results'
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
B = 20_000
RNG = np.random.default_rng(20260929)
MARGIN = 1.10                     # biên không kém hơn, ấn định trước

ARM = {  # (tag, model, frac) -> mã nhánh
    ('A', 'mlp', 0.3): 'A', ('B', 'pinn_sup', 0.3): 'B', ('C', 'pinn_semi', 0.3): 'C',
    ('D', 'pinn_semi', 0.3): 'D', ('E', 'pinn_semi', 0.3): 'E', ('F', 'mlp', 0.7): 'F',
    ('G', 'pinn_semi', 0.7): 'G', ('L', 'mlp', 0.1): 'M10', ('L', 'pinn_semi', 0.1): 'P10',
    ('L', 'mlp', 0.5): 'M50', ('L', 'pinn_semi', 0.5): 'P50',
}
ARM_LABEL = {'A': 'MLP · 30 %', 'B': 'PINN-sup · 30 %', 'C': 'PINN-semi · 30 %', 'D': 'chỉ L_mono · 30 %',
             'E': 'PINN-semi, phần dư Euler · 30 %', 'F': 'MLP · 70 %', 'G': 'PINN-semi · 70 %',
             'M10': 'MLP · 10 %', 'P10': 'PINN-semi · 10 %', 'M50': 'MLP · 50 %', 'P50': 'PINN-semi · 50 %'}
CURVE = {'mlp': {0.1: 'M10', 0.3: 'A', 0.5: 'M50', 0.7: 'F'},
         'pinn_semi': {0.1: 'P10', 0.3: 'C', 0.5: 'P50', 0.7: 'G'}}


# ───────────────────────────────────────────────────────────── 1. bảng theo cell
def _first(v, thr):
    i = np.nonzero(v <= thr)[0]
    return int(i[0]) if len(i) else None


def build_percell() -> pd.DataFrame:
    rows, folds = [], []
    files = sorted(glob.glob(f'{RUNS}/*.json'))
    for f in files:
        r = json.load(open(f))
        c = r['config']
        arm = ARM.get((c['tag'], c['model'], c['label_frac']))
        if arm is None:
            continue
        tp = r['test_predictions']
        y = np.asarray(tp['y'], float); p = np.asarray(tp['p'], float)
        cell = np.asarray(tp['cell']); cyc = np.asarray(tp['cycle'])
        ids = tp['cell_ids']
        base = dict(arm=arm, dataset=c['dataset'], repeat=c['repeat'], fold=c['fold'])
        folds.append(dict(base, n_train=r['n_cells']['train_labelled'] + r['n_cells']['train_unlabelled'],
                          n_lab=r['n_cells']['train_labelled'], n_test=r['n_cells']['test'],
                          steps=r['steps'], mae_row=float(np.abs(p - y).mean()),
                          lam=r.get('physics_params', {}).get('lam', np.nan),
                          Ea=r.get('physics_params', {}).get('Ea_kJ_mol', np.nan)))
        for k in np.unique(cell):
            m = cell == k
            yy, pp, cc = y[m], p[m], cyc[m]
            e = pp - yy; d = np.diff(pp)
            late = yy <= LATE_SOH
            a, b = _first(yy, EOL_THRESHOLD), _first(pp, EOL_THRESHOLD)
            rows.append(dict(base, cell=ids[k], n=int(m.sum()),
                             mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                             bias=float(e.mean()), sse=float((e ** 2).sum()), sae=float(np.abs(e).sum()),
                             sae_late=float(np.abs(e[late]).sum()), n_late=int(late.sum()),
                             mono100=float(d[d > 0].sum()) / (max(1.0, float(cc[-1] - cc[0])) / 100.0),
                             jitter=float(np.abs(d).mean()),
                             eol_true=float(cc[a]) if a is not None else np.nan,
                             eol_pred=float(cc[b]) if b is not None else np.nan,
                             last_cycle=float(cc[-1])))
    print(f'đọc {len(files)} lượt huấn luyện')
    return pd.DataFrame(rows), pd.DataFrame(folds)


# ───────────────────────────────────────────────────────────── 2. công cụ thống kê
def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (m - rank) * p[i]); adj[i] = min(1.0, prev)
    return adj


def boot_ratio(a, b, alpha=0.05):
    """KTC percentile cho mean(b)/mean(a), lấy lại mẫu GHÉP CẶP theo cell."""
    n = len(a); idx = RNG.integers(0, n, size=(B, n))
    r = b[idx].mean(1) / a[idx].mean(1)
    return float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2)))


def nb_ttest(d, n_test, n_train, alternative='two-sided'):
    """Phép t hiệu chỉnh cho kiểm định chéo lặp lại (Nadeau & Bengio 2003; Bouckaert & Frank 2004)."""
    d = np.asarray(d, float); J = len(d)
    se = np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
    t = d.mean() / se
    if alternative == 'two-sided':
        p = 2 * stats.t.sf(abs(t), J - 1)
    else:                                   # 'less'
        p = stats.t.cdf(t, J - 1)
    return float(t), float(p)


def cell_table(pc: pd.DataFrame, arm: str) -> pd.Series:
    """e_arm(c): trung bình qua các lần lặp của MAE cell c (index = (dataset, cell))."""
    g = pc[pc.arm == arm].groupby(['dataset', 'cell'])
    s = g.mae.mean()
    return s


def compare(pc, fl, a, b, family, kind='sup'):
    """So nhánh b với nhánh a trên từng bộ. kind='sup': hai phía, H0 bằng nhau.
    kind='ni': một phía, H0: mean(b) >= MARGIN * mean(a)."""
    ea, eb = cell_table(pc, a), cell_table(pc, b)
    out = []
    for ds in DATASETS:
        if ds not in ea.index.get_level_values(0) or ds not in eb.index.get_level_values(0):
            continue
        x, y = ea.loc[ds], eb.loc[ds]
        common = x.index.intersection(y.index)
        x, y = x.loc[common].values, y.loc[common].values
        if len(common) < 6:
            continue
        # phép kiểm chính: Wilcoxon trên cell
        if kind == 'sup':
            p = stats.wilcoxon(y, x, alternative='two-sided', zero_method='wilcox').pvalue
            lo, hi = boot_ratio(x, y, 0.05)
        else:
            p = stats.wilcoxon(y - MARGIN * x, alternative='less', zero_method='wilcox').pvalue
            lo, hi = boot_ratio(x, y, 0.10)          # KTC 90 % hai phía = KTC 95 % một phía
        # độ nhạy: phép t hiệu chỉnh ở mức fold
        fa = fl[(fl.arm == a) & (fl.dataset == ds)].set_index(['repeat', 'fold'])
        fb = fl[(fl.arm == b) & (fl.dataset == ds)].set_index(['repeat', 'fold'])
        ka = pc[(pc.arm == a) & (pc.dataset == ds)].groupby(['repeat', 'fold']).mae.mean()
        kb = pc[(pc.arm == b) & (pc.dataset == ds)].groupby(['repeat', 'fold']).mae.mean()
        kk = ka.index.intersection(kb.index)
        dd = (kb.loc[kk] - (MARGIN if kind == 'ni' else 1.0) * ka.loc[kk]).values
        t, p_nb = nb_ttest(dd, fa.n_test.mean(), fa.n_train.mean(),
                           'two-sided' if kind == 'sup' else 'less') if len(kk) >= 3 else (np.nan, np.nan)
        out.append(dict(ho=family, so_sanh=f'{b} vs {a}', bo=ds, n_cell=len(common), n_fold=len(kk),
                        mae_a=x.mean(), mae_b=y.mean(), ti_so=y.mean() / x.mean(), ci_lo=lo, ci_hi=hi,
                        trung_vi_hieu=float(np.median(y - x)), cell_b_tot_hon=float((y < x).mean()),
                        p=p, t_nb=t, p_nb=p_nb))
    df = pd.DataFrame(out)
    if len(df):
        df['p_holm'] = holm(df.p.values)
        df['p_nb_holm'] = holm(df.p_nb.values)
        if kind == 'sup':
            df['ket_luan'] = np.where(df.p_holm >= 0.05, 'chưa đủ bằng chứng',
                                      np.where(df.trung_vi_hieu < 0, f'{b} tốt hơn', f'{a} tốt hơn'))
            df['ket_luan_nb'] = np.where(df.p_nb_holm >= 0.05, 'chưa đủ bằng chứng',
                                         np.where(df.ti_so < 1, f'{b} tốt hơn', f'{a} tốt hơn'))
        else:
            df['ket_luan'] = np.where((df.p_holm < 0.05) & (df.ci_hi < MARGIN), 'không kém hơn (δ = 10 %)',
                                      'chưa chứng minh được')
            df['ket_luan_nb'] = np.where(df.p_nb_holm < 0.05, 'không kém hơn (δ = 10 %)', 'chưa chứng minh được')
    return df


# ───────────────────────────────────────────────────────────── 3. các bảng
def secondary(pc: pd.DataFrame) -> pd.DataFrame:
    """Chỉ số phụ theo nhánh × bộ. Gộp qua mọi lượt (cell × lần lặp)."""
    out = []
    for (arm, ds), g in pc.groupby(['arm', 'dataset']):
        cross = g.dropna(subset=['eol_true'])
        both = cross.dropna(subset=['eol_pred'])
        err = (both.eol_pred - both.eol_true).values
        false = g[g.eol_true.isna() & g.eol_pred.notna()]
        out.append(dict(nhanh=arm, bo=ds, MAE_cell=g.groupby('cell').mae.mean().mean(),
                        MAE_hang=g.sae.sum() / g.n.sum(),
                        MAE_late=g.sae_late.sum() / max(1, g.n_late.sum()),
                        EOL_MAE=float(np.abs(err).mean()) if len(err) else np.nan,
                        EOL_bias=float(err.mean()) if len(err) else np.nan,
                        EOL_n=len(err), EOL_bo_sot=len(cross) - len(both), EOL_bao_gia=len(false),
                        MonoViol100=g.mono100.mean(), Jitter=g.jitter.mean()))
    return pd.DataFrame(out)


def curve(pc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, fr in CURVE.items():
        for f, arm in fr.items():
            e = cell_table(pc, arm)
            for ds in DATASETS:
                if ds not in e.index.get_level_values(0):
                    continue
                v = e.loc[ds].values
                idx = RNG.integers(0, len(v), size=(B, len(v)))
                m = v[idx].mean(1)
                out.append(dict(model=model, frac=f, bo=ds, MAE_cell=v.mean(),
                                lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5), n_cell=len(v)))
    return pd.DataFrame(out)


def e14() -> pd.DataFrame:
    rows = []
    for f in glob.glob('runs/E14/*.json'):
        r = json.load(open(f)); c = r['config']
        rows.append(dict(cap=f'{c["dataset"]} → {c["target"]}', k=c['k_target'], model=c['model'],
                         seed=c['seed'], MAE_cell=r['target_test']['MAE_cell'], MAE=r['target_test']['MAE']))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    t = d.groupby(['cap', 'k', 'model']).agg(MAE_cell=('MAE_cell', 'mean'), sd=('MAE_cell', 'std'),
                                             MAE=('MAE', 'mean'), n=('seed', 'size')).reset_index()
    w = t.pivot_table(index=['cap', 'k'], columns='model', values=['MAE_cell', 'sd']).reset_index()
    w.columns = ['cap', 'k', 'mlp', 'pinn_semi', 'sd_mlp', 'sd_pinn']
    w['ti_so'] = w.pinn_semi / w.mlp
    # thắng theo seed (ghép cặp: cùng seed, cùng cặp, cùng k)
    p = d.pivot_table(index=['cap', 'k', 'seed'], columns='model', values='MAE_cell').reset_index()
    wins = p.assign(w=p.pinn_semi < p.mlp).groupby(['cap', 'k']).w.agg(['sum', 'size']).reset_index()
    w = w.merge(wins, on=['cap', 'k']).rename(columns={'sum': 'seed_pinn_thang', 'size': 'n_seed'})
    return w


def to_md(df: pd.DataFrame, path: str, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(fmt.get(c, '{:.4g}').format(v) if np.isfinite(v) else '—')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    open(path, 'w').write('\n'.join(lines) + '\n')


# ───────────────────────────────────────────────────────────── 4. hình
def figures(pc, tests, cur):
    import matplotlib, matplotlib.pyplot as plt
    from pinnsoh.plotstyle import (apply_theme, MODEL_COLOR, INK, INK2, MUTED, GRID, CHEM, panel_title,
                                   direct_label, figure_note, save, log_mae_axis)
    apply_theme()
    GOOD, BAD, NEU = '#0d6e56', '#b8531f', '#78848d'

    # 4a. forest plot các tỉ số, mỗi bộ một cột
    order = [('C vs A', 'H1  PINN-semi / MLP, cùng 30 %'), ('C vs F', 'H2  PINN-semi 30 % / MLP 70 %'),
             ('B vs A', 'PINN-sup / MLP, 30 %'), ('C vs B', 'PINN-semi / PINN-sup'),
             ('D vs C', 'chỉ L_mono / PINN-semi'), ('E vs C', 'phần dư Euler / autograd'),
             ('G vs F', 'PINN-semi / MLP, 70 %')]
    order = [o for o in order if o[0] in set(tests.so_sanh)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 0.36 * len(order) + 0.9), sharey=True)
    for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
        for i, (key, lab) in enumerate(order):
            r = tests[(tests.so_sanh == key) & (tests.bo == ds)]
            if r.empty:
                continue
            r = r.iloc[0]
            ni = key == 'C vs F'
            if ni:
                col = GOOD if r.ket_luan.startswith('không kém') else NEU
            else:
                col = GOOD if (r.p_holm < 0.05 and r.ti_so < 1) else (BAD if r.p_holm < 0.05 else NEU)
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.6, solid_capstyle='round')
            ax.plot(r.ti_so, i, 'o', ms=4.6, color=col, mec='white', mew=0.8, zorder=3)
            ax.annotate(f'{r.ti_so:.2f}', (r.ci_hi, i), xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=6.3, color=INK2)
        ax.axvline(1.0, color='#c4ccc8', lw=0.8, zorder=0)
        ax.axvline(MARGIN, color='#c4ccc8', lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.set_xlim(0.45, 1.75)
        ax.set_xticks([0.6, 1.0, 1.4])
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=0.5)
        panel_title(ax, ds, CHEM[ds], pad=4)
        if j == 0:
            ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order])
        ax.tick_params(axis='y', length=0)
    axes[1].set_xlabel('tỉ số MAE theo cell (< 1: nhánh thứ nhất sai số thấp hơn)', x=1.05)
    figure_note(fig, 'Chấm = tỉ số trung bình trên các cell; vạch = KTC 95 % bootstrap theo cell (H2: KTC 90 %). '
                     'Xanh = có ý nghĩa sau Holm theo hướng tốt hơn; cam = tệ hơn; xám = chưa đủ bằng chứng. '
                     'Vạch đứt tại 1,10 = biên không kém hơn.', y=-0.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, f'{OUT}/fig_E13_forest')

    # 4b. đường cong theo lượng nhãn
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
        for model in ['mlp', 'pinn_semi']:
            g = cur[(cur.bo == ds) & (cur.model == model)].sort_values('frac')
            if g.empty:
                continue
            x = 100 * g.frac.values
            ax.fill_between(x, 1e3 * g.lo, 1e3 * g.hi, color=MODEL_COLOR[model], alpha=0.14, lw=0)
            ax.plot(x, 1e3 * g.MAE_cell, color=MODEL_COLOR[model], lw=1.8, marker='o', ms=3.4,
                    mec='white', mew=0.6)
            if j == 3:
                direct_label(ax, x[-1], 1e3 * g.MAE_cell.values[-1], 'MLP' if model == 'mlp' else 'PINN-semi',
                             MODEL_COLOR[model], dx=4, dy=5 if model == 'mlp' else -5)
        f70 = cur[(cur.bo == ds) & (cur.model == 'mlp') & (cur.frac == 0.7)]
        if len(f70):
            ax.axhline(1e3 * f70.MAE_cell.iloc[0], color=MODEL_COLOR['mlp'], lw=0.7, ls=(0, (2, 2)), alpha=0.7)
        ax.set_xticks([10, 30, 50, 70]); ax.set_xlim(5, 75)
        log_mae_axis(ax, ticks=(3, 5, 10, 20, 40))
        panel_title(ax, ds, CHEM[ds], pad=4)
        if j == 0:
            ax.set_ylabel(r'MAE theo cell (×10$^{-3}$ SOH)')
    axes[1].set_xlabel('cell có nhãn (% tổng số cell)', x=1.05)
    figure_note(fig, 'Dải = KTC 95 % bootstrap theo cell. Vạch đứt = MLP dùng 70 % nhãn. '
                     'Kiểm định chéo 5 fold × 5 lần lặp, pipeline v2.', y=-0.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, f'{OUT}/fig_E13_curve')

    # 4c. từng cell: MLP 30 % so với PINN-semi 30 %
    ea, ec = cell_table(pc, 'A'), cell_table(pc, 'C')
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))
    for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
        if ds not in ea.index.get_level_values(0):
            continue
        x, y = 1e3 * ea.loc[ds], 1e3 * ec.loc[ds].reindex(ea.loc[ds].index)
        lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color='#c4ccc8', lw=0.8, zorder=0)
        better = y < x
        ax.scatter(x[better], y[better], s=11, color=MODEL_COLOR['pinn_semi'], lw=0.4, edgecolor='white', zorder=3)
        ax.scatter(x[~better], y[~better], s=11, color=MODEL_COLOR['mlp'], lw=0.4, edgecolor='white', zorder=3)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        tk = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.FixedLocator(tk))
            axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in tk]))
            axis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_aspect('equal'); ax.grid(False)
        ax.text(0.04, 0.96, f'{better.mean() * 100:.0f} % cell\nPINN thấp hơn', transform=ax.transAxes,
                va='top', fontsize=6.4, color=MODEL_COLOR['pinn_semi'])
        panel_title(ax, ds, f'{len(x)} cell', pad=4)
        ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.set_ylabel(r'PINN-semi 30 % (×10$^{-3}$)')
    axes[1].set_xlabel(r'MLP 30 % — MAE cell (×10$^{-3}$, trung bình 5 lần lặp)', x=1.05)
    fig.tight_layout()
    save(fig, f'{OUT}/fig_E13_cells')


def fig_traj(pc):
    """Một cell mỗi bộ: cell có MAE của MLP 30 % ĐÚNG TRUNG VỊ trong bộ (quy tắc chọn độc lập với PINN)."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, MODEL_COLOR, MUTED, panel_title, save, direct_label, CHEM
    apply_theme()
    ea = cell_table(pc, 'A')
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))
    for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
        if ds not in ea.index.get_level_values(0):
            continue
        s = ea.loc[ds].sort_values()
        cid = s.index[len(s) // 2]
        sub = pc[(pc.dataset == ds) & (pc.cell == cid) & (pc.repeat == 0)]
        fold = int(sub.fold.iloc[0])
        series = {}
        for arm in ['A', 'C', 'F']:
            m = {'A': 'mlp', 'C': 'pinn_semi', 'F': 'mlp'}[arm]; f = {'A': 0.3, 'C': 0.3, 'F': 0.7}[arm]
            path = glob.glob(f'{RUNS}/{arm}_{ds}_{m}_f{f}_s{fold}_*_cv0-{fold}of5.json')
            if not path:
                continue
            tp = json.load(open(path[0]))['test_predictions']
            k = tp['cell_ids'].index(cid)
            msk = np.asarray(tp['cell']) == k
            series[arm] = (np.asarray(tp['cycle'])[msk], np.asarray(tp['y'])[msk], np.asarray(tp['p'])[msk])
        if 'A' not in series:
            continue
        cyc, y, _ = series['A']
        ax.plot(cyc, y, color=MODEL_COLOR['truth'], lw=1.7, zorder=2)
        style = {'A': (MODEL_COLOR['mlp'], '-'), 'C': (MODEL_COLOR['pinn_semi'], '-'), 'F': ('#8fb6e8', (0, (2, 1.5)))}
        for arm, (c_, y_, p_) in series.items():
            col, ls = style[arm]
            ax.plot(c_, p_, color=col, lw=1.0, ls=ls, zorder=3)
        txt = '  '.join(f'{a}: {1e3 * np.abs(series[a][2] - series[a][1]).mean():.1f}' for a in ['A', 'C', 'F'] if a in series)
        ax.text(0.03, 0.04, r'MAE ×10$^{-3}$  ' + txt, transform=ax.transAxes, fontsize=5.6, color=MUTED)
        panel_title(ax, ds, cid.split('/')[-1], pad=4)
        ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
        ax.grid(False); ax.yaxis.grid(True, color='#eef1ef', lw=.55)
        if j == 0:
            ax.set_ylabel('SOH')
    axes[1].set_xlabel('chu kỳ gốc', x=1.05)
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.7, label='thực đo'),
                        plt.Line2D([], [], color=MODEL_COLOR['mlp'], lw=1.4, label='A: MLP 30 %'),
                        plt.Line2D([], [], color=MODEL_COLOR['pinn_semi'], lw=1.4, label='C: PINN-semi 30 %'),
                        plt.Line2D([], [], color='#8fb6e8', lw=1.4, ls=(0, (2, 1.5)), label='F: MLP 70 %')],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, f'{OUT}/fig_E13_traj')


# ───────────────────────────────────────────────────────────── main
def main():
    os.makedirs(OUT, exist_ok=True)
    pc, fl = build_percell()
    pc.to_csv(f'{OUT}/E13_percell.csv', index=False)
    fl.to_csv(f'{OUT}/E13_perfold.csv', index=False)
    print('số lượt theo nhánh:', fl.groupby('arm').size().to_dict())

    fams = [('H1', 'A', 'C', 'sup'), ('H2', 'F', 'C', 'ni'),
            ('phụ', 'A', 'B', 'sup'), ('phụ', 'B', 'C', 'sup'), ('phụ', 'C', 'D', 'sup'),
            ('phụ', 'C', 'E', 'sup'), ('phụ', 'F', 'G', 'sup'),
            ('phụ', 'M10', 'P10', 'sup'), ('phụ', 'M50', 'P50', 'sup')]
    tests = pd.concat([compare(pc, fl, a, b, fam, kind) for fam, a, b, kind in fams], ignore_index=True)
    tests.to_csv(f'{OUT}/E13_tests.csv', index=False)
    show = tests[['ho', 'so_sanh', 'bo', 'n_cell', 'mae_a', 'mae_b', 'ti_so', 'ci_lo', 'ci_hi',
                  'cell_b_tot_hon', 'p_holm', 'p_nb_holm', 'ket_luan', 'ket_luan_nb']]
    to_md(show, f'{OUT}/E13_tests.md', dict(mae_a='{:.5f}', mae_b='{:.5f}', ti_so='{:.3f}', ci_lo='{:.3f}',
                                             ci_hi='{:.3f}', cell_b_tot_hon='{:.2f}', p_holm='{:.4f}',
                                             p_nb_holm='{:.4f}'))
    sec = secondary(pc)
    sec.to_csv(f'{OUT}/E13_secondary.csv', index=False)
    to_md(sec, f'{OUT}/E13_secondary.md', dict(MAE_cell='{:.5f}', MAE_hang='{:.5f}', MAE_late='{:.5f}',
                                               EOL_MAE='{:.1f}', EOL_bias='{:+.1f}', MonoViol100='{:.4f}',
                                               Jitter='{:.5f}'))
    cur = curve(pc)
    cur.to_csv(f'{OUT}/E13_curve.csv', index=False)
    to_md(cur, f'{OUT}/E13_curve.md', dict(MAE_cell='{:.5f}', lo='{:.5f}', hi='{:.5f}'))
    phys = fl[fl.arm.isin(['B', 'C', 'G'])].groupby(['arm', 'dataset'])[['lam', 'Ea']].agg(['mean', 'std']).round(3)
    phys.to_csv(f'{OUT}/E13_physics_params.csv')
    w = e14()
    if len(w):
        w.to_csv(f'{OUT}/E14_table.csv', index=False)
        to_md(w, f'{OUT}/E14_table.md', dict(mlp='{:.4f}', pinn_semi='{:.4f}', sd_mlp='{:.4f}',
                                             sd_pinn='{:.4f}', ti_so='{:.3f}'))

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('\n=== Kiểm định (MAE theo cell, đơn vị = cell; p_nb = phép t hiệu chỉnh mức fold) ===')
    print(show.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n=== Đường cong theo lượng nhãn ===')
    print(cur.pivot_table(index='bo', columns=['model', 'frac'], values='MAE_cell').round(5).to_string())
    print('\n=== Chỉ số phụ ===')
    print(sec.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if len(w):
        print('\n=== E14 — chuyển miền, cùng normaliser ===')
        print(w.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    figures(pc, tests, cur)
    fig_traj(pc)
    print('\nđã ghi results/E13_*.csv|md, E14_table.*, fig_E13_{forest,curve,cells,traj}.{png,pdf}')


if __name__ == '__main__':
    main()
