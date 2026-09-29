"""Aggregate run JSONs -> tables (CSV/Markdown) + figures.   usage: python analyze.py"""
import json, glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results'; os.makedirs(OUT, exist_ok=True)
# validated categorical palette (light surface) - fixed slot order, never cycled
C = {'mlp': '#2a78d6', 'pinn_sup': '#eb6834', 'pinn_semi': '#1baf7a', 'pinn_bb': '#eda100', 'truth': '#4b4b4b'}
LABEL = {'mlp': 'MLP (chỉ dữ liệu)', 'pinn_sup': 'PINN-sup (vật lý trên cell có nhãn)',
         'pinn_semi': 'PINN-semi (vật lý trên mọi cell)', 'pinn_bb': 'PINN black-box (kiểu PINN4SOH)'}
INK, INK2, GRID = '#1f1f1f', '#5f5f5f', '#e6e6e6'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 170, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})


def load(pattern):
    rows = []
    for p in glob.glob(pattern):
        with open(p) as f:
            r = json.load(f)
        c = r['config']
        row = dict(run=os.path.basename(p)[:-5], dataset=c['dataset'], model=c['model'], frac=c['label_frac'],
                   seed=c['seed'], norm=c['norm'], cyc=c['cyc'], tag=c['tag'], use_cycle=c['use_cycle'],
                   steps=r['steps'], n_lab=r['n_cells']['train_labelled'], n_unl=r['n_cells']['train_unlabelled'],
                   n_test=r['n_cells']['test'], **{f'test_{k}': v for k, v in r['test'].items()},
                   **{f'val_{k}': v for k, v in r['val'].items()})
        for k, v in r.items():
            if k.startswith('transfer_'):
                row.update({f'{k}_{m}': vv for m, vv in v.items()})
        if 'physics_params' in r:
            row.update(lam=r['physics_params']['lam'], Ea=r['physics_params']['Ea_kJ_mol'])
        row['n_params'] = r.get('n_params', np.nan)
        row['act'] = c.get('act', 'silu'); row['arch'] = c.get('arch', 'mlp')
        rows.append(row)
    return pd.DataFrame(rows)


def ms(x):
    return f'{x.mean():.4f} ± {x.std(ddof=0):.4f}' if len(x) > 1 else f'{x.mean():.4f}'


# ------------------------------------------------------------------ E1 label efficiency
def E1():
    df = load('runs/E1/*.json')
    if df.empty:
        return
    g = df.groupby(['dataset', 'model', 'frac'])
    tab = g.agg(MAE=('test_MAE', 'mean'), MAE_sd=('test_MAE', 'std'), RMSE=('test_RMSE', 'mean'),
                MAPE=('test_MAPE', 'mean'), MAPE_sd=('test_MAPE', 'std'), n=('seed', 'count'),
                n_lab=('n_lab', 'first'), n_unl=('n_unl', 'first')).reset_index()
    tab.to_csv(f'{OUT}/E1_table.csv', index=False)
    # markdown: MAE (mean ± sd) per dataset x frac, columns models
    lines = []
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        sub = df[df.dataset == ds]
        if sub.empty: continue
        lines.append(f'\n**{ds}** — test MAE (SOH units), mean ± sd over seeds\n')
        lines.append('| labelled cells (% of all) | ' + ' | '.join(LABEL[m] for m in ['mlp', 'pinn_sup', 'pinn_semi']) + ' |')
        lines.append('|---|---|---|---|')
        for fr in sorted(sub.frac.unique()):
            cells = []
            for m in ['mlp', 'pinn_sup', 'pinn_semi']:
                x = sub[(sub.frac == fr) & (sub.model == m)].test_MAE
                cells.append(ms(x) if len(x) else '—')
            nl = sub[sub.frac == fr].n_lab.iloc[0]
            lines.append(f'| {int(fr*100)} % ({nl} cells) | ' + ' | '.join(cells) + ' |')
    # hypothesis table: PINN-semi @30% vs MLP @70%
    lines.append('\n**Giả thuyết chính: PINN-semi với 30 % nhãn so với MLP với 70 % nhãn (protocol 70/15/15 đầy đủ)**\n')
    lines.append('| dataset | MLP @70 % | MLP @30 % | PINN-semi @30 % | PINN-semi@30 / MLP@70 | MLP@30 / MLP@70 |')
    lines.append('|---|---|---|---|---|---|')
    hyp = []
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        sub = df[df.dataset == ds]
        a = sub[(sub.model == 'mlp') & (sub.frac == 0.7)].test_MAE
        b = sub[(sub.model == 'mlp') & (sub.frac == 0.3)].test_MAE
        c = sub[(sub.model == 'pinn_semi') & (sub.frac == 0.3)].test_MAE
        if len(a) and len(b) and len(c):
            lines.append(f'| {ds} | {ms(a)} | {ms(b)} | {ms(c)} | {c.mean()/a.mean():.2f}× | {b.mean()/a.mean():.2f}× |')
            hyp.append(dict(dataset=ds, mlp70=a.mean(), mlp30=b.mean(), pinn30=c.mean(),
                            ratio_pinn30_mlp70=c.mean() / a.mean(), ratio_mlp30_mlp70=b.mean() / a.mean()))
    pd.DataFrame(hyp).to_csv(f'{OUT}/E1_hypothesis.csv', index=False)
    open(f'{OUT}/E1_tables.md', 'w').write('\n'.join(lines))

    # figure: 4 panels, MAE vs labelled fraction
    dss = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        sub = df[df.dataset == ds]
        for m in ['mlp', 'pinn_sup', 'pinn_semi']:
            s = sub[sub.model == m].groupby('frac').test_MAE
            if s.count().sum() == 0: continue
            mu, sd = s.mean(), s.std().fillna(0)
            ax.plot(mu.index * 100, mu.values, '-', lw=2, color=C[m], marker='o', ms=6, label=LABEL[m])
            ax.fill_between(mu.index * 100, (mu - sd).values, (mu + sd).values, color=C[m], alpha=0.15, lw=0)
        ax.set_title(ds, loc='left', fontweight='bold'); ax.set_xlabel('% cell có nhãn (trên tổng số cell)')
        ax.set_xticks([10, 30, 50, 70]); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('MAE trên tập test (đơn vị SOH)')
    axes[-1].legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('Hiệu quả theo lượng nhãn — chia theo cell 70/15/15, val/test cố định, 3 seed', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E1_label_efficiency.png'); plt.close(fig)

    # figure: example trajectories, XJTU 30 % seed 0
    for ds in dss:
        preds = {}
        for m in ['mlp', 'pinn_semi']:
            p = f'runs/E1/{ds}_{m}_f0.3_s0_global_global.json'
            if os.path.exists(p):
                preds[m] = json.load(open(p))['test_predictions']
        if len(preds) < 2: continue
        y = np.array(preds['mlp']['y']); cell = np.array(preds['mlp']['cell'])
        cells = np.unique(cell)[:4]
        fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.0), sharey=True)
        for ax, k in zip(np.atleast_1d(axes), cells):
            idx = cell == k; n = idx.sum()
            ax.plot(np.arange(n), y[idx], color=C['truth'], lw=2, label='thực đo')
            for m in preds:
                ax.plot(np.arange(n), np.array(preds[m]['p'])[idx], color=C[m], lw=1.6, alpha=0.9, label=LABEL[m].split(' (')[0])
            ax.set_title(f'test cell #{k}', loc='left', fontsize=9); ax.set_xlabel('chu kỳ (sau lọc)')
        np.atleast_1d(axes)[0].set_ylabel('SOH'); np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{ds}: dự đoán trên cell test khi chỉ 30 % cell có nhãn (seed 0)', x=0.01, ha='left', fontsize=10)
        fig.tight_layout(); fig.savefig(f'{OUT}/fig_traj_{ds}.png'); plt.close(fig)
    return tab


# ------------------------------------------------------------------ E2 cross-dataset
def E2():
    df = load('runs/E2/*.json')
    if df.empty: return
    rows = []
    for _, r in df.iterrows():
        for k in r.index:
            if k.startswith('transfer_') and k.endswith('_MAE') and not pd.isna(r[k]) and k[9:-4] != r.dataset:
                rows.append(dict(source=r.dataset, target=k[9:-4], model=r.model, norm=r.norm, seed=r.seed,
                                 in_domain_MAE=r.test_MAE, transfer_MAE=r[k], transfer_MAPE=r[k[:-4] + '_MAPE']))
    t = pd.DataFrame(rows)
    t.to_csv(f'{OUT}/E2_raw.csv', index=False)
    agg = t.groupby(['source', 'target', 'norm', 'model']).agg(in_domain=('in_domain_MAE', 'mean'),
        transfer=('transfer_MAE', 'mean'), transfer_sd=('transfer_MAE', 'std'), transfer_MAPE=('transfer_MAPE', 'mean'),
        n=('seed', 'count')).reset_index()
    agg.to_csv(f'{OUT}/E2_table.csv', index=False)
    lines = ['| nguồn → đích | chuẩn hoá | mô hình | MAE trong miền (test nguồn) | MAE zero-shot trên toàn bộ đích | MAPE đích |', '|---|---|---|---|---|---|']
    for _, r in agg.iterrows():
        lines.append(f'| {r.source} → {r.target} | {r.norm} | {LABEL[r.model].split(" (")[0]} | {r.in_domain:.4f} | {r.transfer:.4f} ± {0 if np.isnan(r.transfer_sd) else r.transfer_sd:.4f} | {r.transfer_MAPE:.1f} % |')
    open(f'{OUT}/E2_table.md', 'w').write('\n'.join(lines))
    # figure
    pairs = sorted(set(zip(agg.source, agg.target)))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(pairs)); w = 0.2
    combos = [('global', 'mlp'), ('global', 'pinn_semi'), ('first_cycle', 'mlp'), ('first_cycle', 'pinn_semi')]
    for i, (nm, m) in enumerate(combos):
        vals = [agg[(agg.source == s) & (agg.target == tg) & (agg.norm == nm) & (agg.model == m)].transfer.mean() for s, tg in pairs]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=C[m], alpha=1.0 if nm == 'first_cycle' else 0.45,
               label=f'{LABEL[m].split(" (")[0]} · {"first-cycle" if nm=="first_cycle" else "global"} norm')
    ax.set_xticks(x); ax.set_xticklabels([f'{s}→{t}' for s, t in pairs]); ax.set_ylabel('MAE zero-shot trên miền đích')
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.set_title('Chuyển miền zero-shot (không fine-tune) — 70 % nhãn nguồn', loc='left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E2_transfer.png'); plt.close(fig)


# ------------------------------------------------------------------ E3 ablation
def E3():
    df = load('runs/E3/*.json')
    if df.empty: return
    df['variant'] = np.where(df.model == 'pinn_bb', 'blackbox', df.tag)
    agg = df.groupby(['dataset', 'variant']).agg(MAE=('test_MAE', 'mean'), sd=('test_MAE', 'std'),
                                                 MAPE=('test_MAPE', 'mean'), n=('seed', 'count'),
                                                 lam=('lam', 'mean'), Ea=('Ea', 'mean')).reset_index()
    agg.to_csv(f'{OUT}/E3_table.csv', index=False)
    order = ['full', 'no_ode', 'no_mono', 'no_range', 'only_mono', 'no_knee', 'no_arrh', 'res_fd', 'res_autograd', 'blackbox']
    lines = ['| biến thể (30 % nhãn) | ' + ' | '.join(sorted(agg.dataset.unique())) + ' |', '|---|' + '---|' * agg.dataset.nunique()]
    for v in order:
        cells = []
        for ds in sorted(agg.dataset.unique()):
            r = agg[(agg.dataset == ds) & (agg.variant == v)]
            cells.append('—' if r.empty else f'{r.MAE.iloc[0]:.4f} ± {0 if np.isnan(r.sd.iloc[0]) else r.sd.iloc[0]:.4f}')
        lines.append(f'| {v} | ' + ' | '.join(cells) + ' |')
    open(f'{OUT}/E3_table.md', 'w').write('\n'.join(lines))


# ------------------------------------------------------------------ E4 leakage audit
def E4():
    df = load('runs/E4/*.json')
    if df.empty: return
    agg = df.groupby(['dataset', 'tag']).agg(MAE=('test_MAE', 'mean'), sd=('test_MAE', 'std'), MAPE=('test_MAPE', 'mean'),
                                             n=('seed', 'count')).reset_index()
    agg.to_csv(f'{OUT}/E4_table.csv', index=False)
    order = ['causal', 'no_cycle_input', 'leak_cycle', 'leak_feat', 'leak_both']
    desc = {'causal': 'chuẩn hoá nhân quả (đề xuất): z-score theo train, cycle/1000',
            'no_cycle_input': 'như trên nhưng bỏ hẳn đầu vào chỉ số chu kỳ',
            'leak_cycle': 'chỉ số chu kỳ min-max theo từng cell (rò rỉ tuổi thọ)',
            'leak_feat': 'đặc trưng min-max theo từng cell (dùng cả tương lai)',
            'leak_both': 'cả hai — đúng pipeline PINN4SOH'}
    lines = ['| cách chuẩn hoá (MLP, 70 % nhãn) | ' + ' | '.join(sorted(agg.dataset.unique())) + ' |', '|---|' + '---|' * agg.dataset.nunique()]
    for v in order:
        cells = []
        for ds in sorted(agg.dataset.unique()):
            r = agg[(agg.dataset == ds) & (agg.tag == v)]
            cells.append('—' if r.empty else f'{r.MAE.iloc[0]:.4f} ± {0 if np.isnan(r.sd.iloc[0]) else r.sd.iloc[0]:.4f}')
        lines.append(f'| {desc[v]} | ' + ' | '.join(cells) + ' |')
    open(f'{OUT}/E4_table.md', 'w').write('\n'.join(lines))


# ------------------------------------------------------------------ E5 adaptation
def E5():
    rows = []
    for p in glob.glob('runs/E5/*.json'):
        r = json.load(open(p)); c = r['config']
        rows.append(dict(source=c['dataset'], target=c['target'], model=c['model'], k=c['k_target'], seed=c['seed'],
                         MAE=r['target_test']['MAE'], MAPE=r['target_test']['MAPE'], MAE_cell=r['target_test']['MAE_cell_mean']))
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/E5_raw.csv', index=False)
    agg = df.groupby(['source', 'target', 'k', 'model']).agg(MAE=('MAE', 'mean'), sd=('MAE', 'std'), MAPE=('MAPE', 'mean'),
                                                             n=('seed', 'count')).reset_index()
    agg.to_csv(f'{OUT}/E5_table.csv', index=False)
    name = {(0, 'mlp'): 'MLP zero-shot (k=0)', (0, 'pinn_semi'): 'PINN-semi, thích nghi KHÔNG nhãn đích (k=0)',
            (3, 'mlp'): 'MLP, 3 cell đích có nhãn', (3, 'pinn_semi'): 'PINN-semi, 3 cell đích có nhãn + vật lý trên cell không nhãn'}
    pairs = sorted(set(zip(agg.source, agg.target)))
    lines = ['| phương pháp | ' + ' | '.join(f'{s}→{t}' for s, t in pairs) + ' |', '|---|' + '---|' * len(pairs)]
    for key in [(0, 'mlp'), (0, 'pinn_semi'), (3, 'mlp'), (3, 'pinn_semi')]:
        cells = []
        for s_, t_ in pairs:
            r = agg[(agg.source == s_) & (agg.target == t_) & (agg.k == key[0]) & (agg.model == key[1])]
            cells.append('—' if r.empty else f'{r.MAE.iloc[0]:.4f} ± {0 if np.isnan(r.sd.iloc[0]) else r.sd.iloc[0]:.4f}')
        lines.append(f'| {name[key]} | ' + ' | '.join(cells) + ' |')
    # in-domain reference (MLP 70 % on the target, from E1)
    e1 = load('runs/E1/*.json')
    if not e1.empty:
        cells = []
        for s_, t_ in pairs:
            r = e1[(e1.dataset == t_) & (e1.model == 'mlp') & (e1.frac == 0.7)].test_MAE
            cells.append('—' if r.empty else f'{r.mean():.4f}')
        lines.append('| *tham chiếu: MLP huấn luyện trong miền đích, 70 % nhãn* | ' + ' | '.join(cells) + ' |')
    open(f'{OUT}/E5_table.md', 'w').write('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(pairs)); w = 0.2
    for i, key in enumerate([(0, 'mlp'), (0, 'pinn_semi'), (3, 'mlp'), (3, 'pinn_semi')]):
        vals = [agg[(agg.source == s_) & (agg.target == t_) & (agg.k == key[0]) & (agg.model == key[1])].MAE.mean() for s_, t_ in pairs]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=C[key[1]], alpha=0.45 if key[0] == 0 else 1.0, label=name[key])
    ax.set_xticks(x); ax.set_xticklabels([f'{s}→{t}' for s, t in pairs]); ax.set_ylabel('MAE trên cell test của miền đích')
    ax.set_yscale('log'); ax.legend(frameon=False, fontsize=7.5, ncol=1, loc='upper right')
    ax.set_title('Thích nghi sang bộ dữ liệu khác — trục log', loc='left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E5_adapt.png'); plt.close(fig)


# ------------------------------------------------------------------ E6 validation-selected physics weight
def E6():
    df = load('runs/E6/*.json')
    if df.empty: return
    e1 = load('runs/E1/*.json')
    base = e1[(e1.frac == 0.3) & (e1.model == 'pinn_semi')].copy(); base['tag'] = 'b5 (mặc định)'
    mlp = e1[(e1.frac == 0.3) & (e1.model == 'mlp')].copy(); mlp['tag'] = 'MLP (tham chiếu)'
    mlp70 = e1[(e1.frac == 0.7) & (e1.model == 'mlp')].copy(); mlp70['tag'] = 'MLP 70 % (tham chiếu)'
    allr = pd.concat([df, base, mlp, mlp70])
    agg = allr.groupby(['dataset', 'tag']).agg(val=('val_MAE', 'mean'), test=('test_MAE', 'mean'), sd=('test_MAE', 'std'),
                                               n=('seed', 'count')).reset_index()
    agg.to_csv(f'{OUT}/E6_table.csv', index=False)
    sel = []
    lines = ['| dataset | biến thể (30 % nhãn) | MAE val | MAE test |', '|---|---|---|---|']
    for ds in [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if (df.dataset == d).any()]:
        sub = agg[agg.dataset == ds]
        cand = sub[~sub.tag.str.startswith('MLP')]
        best = cand.loc[cand.val.idxmin()]
        sel.append(dict(dataset=ds, selected=best.tag, val=best.val, test=best.test,
                        mlp30=float(sub[sub.tag == 'MLP (tham chiếu)'].test.iloc[0]),
                        mlp70=float(sub[sub.tag == 'MLP 70 % (tham chiếu)'].test.iloc[0])))
        for _, r in sub.sort_values('val').iterrows():
            mark = ' ← chọn theo val' if r.tag == best.tag else ''
            lines.append(f'| {ds} | {r.tag}{mark} | {r.val:.4f} | {r.test:.4f} ± {0 if np.isnan(r.sd) else r.sd:.4f} |')
    pd.DataFrame(sel).to_csv(f'{OUT}/E6_selected.csv', index=False)
    open(f'{OUT}/E6_table.md', 'w').write('\n'.join(lines))


# ------------------------------------------------------------------ E7 val-selected weight across fractions
def E7():
    df = load('runs/E7/*.json')
    if df.empty: return
    e1 = load('runs/E1/*.json')
    mlp = e1[e1.model == 'mlp']
    # per (dataset, frac): pick the variant with the lowest VALIDATION MAE, then read its test MAE
    rows = []
    for ds in sorted(df.dataset.unique()):
        for fr in sorted(df.frac.unique()):
            sub = df[(df.dataset == ds) & (df.frac == fr)]
            if sub.empty: continue
            g = sub.groupby('tag').agg(val=('val_MAE', 'mean'), test=('test_MAE', 'mean'),
                                       sd=('test_MAE', 'std'), n=('seed', 'count')).reset_index()
            best = g.loc[g.val.idxmin()]
            m = mlp[(mlp.dataset == ds) & (mlp.frac == fr)]
            m70 = mlp[(mlp.dataset == ds) & (mlp.frac == 0.7)]
            rows.append(dict(dataset=ds, frac=fr, selected=best.tag, val=best.val, pinn=best.test,
                             pinn_sd=0 if np.isnan(best.sd) else best.sd, n=int(best.n),
                             mlp=m.test_MAE.mean() if len(m) else np.nan,
                             mlp70=m70.test_MAE.mean() if len(m70) else np.nan,
                             adapt=g[g.tag == 'adapt'].test.iloc[0] if (g.tag == 'adapt').any() else np.nan,
                             b5=g[g.tag == 'b5'].test.iloc[0] if (g.tag == 'b5').any() else np.nan))
    t = pd.DataFrame(rows)
    t['ratio_vs_mlp'] = t.pinn / t.mlp
    t['ratio_vs_mlp70'] = t.pinn / t.mlp70
    t.to_csv(f'{OUT}/E7_table.csv', index=False)
    lines = ['| dataset | % nhãn | MLP cùng mức | PINN β mặc định (5) | PINN chọn theo val | biến thể chọn | PINN tự thích ứng | PINN(val)/MLP |',
             '|---|---|---|---|---|---|---|---|']
    for _, r in t.iterrows():
        win = '**' if r.ratio_vs_mlp < 1 else ''
        lines.append(f'| {r.dataset} | {int(r.frac*100)} % | {r.mlp:.4f} | {r.b5:.4f} | {win}{r.pinn:.4f} ± {r.pinn_sd:.4f}{win} | `{r.selected}` | {r.adapt:.4f} | {win}{r.ratio_vs_mlp:.2f}×{win} |')
    open(f'{OUT}/E7_table.md', 'w').write('\n'.join(lines))

    # figure: MAE vs fraction, MLP vs default-beta PINN vs val-selected PINN vs adaptive
    dss = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if d in t.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.5))
    axes = np.atleast_1d(axes)
    series = [('mlp', '#2a78d6', 'MLP (chỉ dữ liệu)'), ('b5', '#eda100', 'PINN β=5 (mặc định cũ)'),
              ('adapt', '#e87ba4', 'PINN trọng số tự thích ứng'), ('pinn', '#1baf7a', 'PINN chọn β theo val')]
    for ax, ds in zip(axes, dss):
        sub = t[t.dataset == ds].sort_values('frac')
        for col, col_c, lab in series:
            ax.plot(sub.frac * 100, sub[col], '-o', lw=2, ms=6, color=col_c, label=lab)
        ax.set_title(ds, loc='left', fontweight='bold'); ax.set_xlabel('% cell có nhãn')
        ax.set_xticks([10, 30, 50, 70]); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('MAE trên tập test (đơn vị SOH)')
    axes[-1].legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.suptitle('Trọng số vật lý quyết định, không phải lượng nhãn — chọn β theo validation ở mọi mức nhãn',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E7_tuned.png'); plt.close(fig)
    return t


# ------------------------------------------------------------------ E9 network / activation
E9_ORDER = ['silu-mlp', 'tanh-mlp', 'sin-mlp', 'snake-mlp', 'silu-res', 'silu-fourier', 'silu-mono', 'silu-mono+L']
E9_DESC = {'silu-mlp': 'MLP + SiLU (cơ sở)', 'tanh-mlp': 'MLP + tanh (PINN kinh điển)',
           'sin-mlp': 'MLP + sin (PINN4SOH)', 'snake-mlp': 'MLP + Snake (xu thế+dao động)',
           'silu-res': 'Residual MLP + SiLU', 'silu-fourier': 'Fourier(t) + MLP + SiLU',
           'silu-mono': 'Đơn điệu theo cấu trúc (tắt L_mono)',
           'silu-mono+L': 'Đơn điệu theo cấu trúc + giữ L_mono'}


def E9():
    df = load('runs/E9/*.json')
    if df.empty: return
    keep = ['test_MAE', 'test_MAE_cell', 'test_MAE_cell_max', 'test_MAE_late', 'test_MonoViol',
            'test_Jitter', 'test_EOL_MAE', 'test_RMSE']
    keep = [k for k in keep if k in df.columns]
    agg = df.groupby(['dataset', 'model', 'tag']).agg(
        **{k[5:]: (k, 'mean') for k in keep}, sd=('test_MAE', 'std'), val=('val_MAE', 'mean'),
        params=('n_params', 'first'), n=('seed', 'count')).reset_index()
    agg.to_csv(f'{OUT}/E9_table.csv', index=False)
    dss = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if (agg.dataset == d).any()]
    # main table: test MAE per architecture x dataset, for both models
    lines = []
    for m, title in [('mlp', 'Baseline chỉ dữ liệu'), ('pinn_semi', 'PINN-semi (có loss vật lý)')]:
        sub = agg[agg.model == m]
        if sub.empty: continue
        lines.append(f'\n**{title}** — MAE test ở 30 % nhãn, trung bình 3 seed\n')
        lines.append('| kiến trúc | tham số | ' + ' | '.join(dss) + ' | trung bình chuẩn hoá |')
        lines.append('|---|---|' + '---|' * (len(dss) + 1))
        full = [d for d in dss if set(sub[sub.dataset == d].tag) >= set(t for t in E9_ORDER
                if ((sub.tag == t).any()))]
        base = {d: float(sub[(sub.dataset == d) & (sub.tag == 'silu-mlp')].MAE.iloc[0]) for d in full
                if ((sub.dataset == d) & (sub.tag == 'silu-mlp')).any()}
        for tg in E9_ORDER:
            r = sub[sub.tag == tg]
            if r.empty: continue
            cells, rel = [], []
            for d in dss:
                q = r[r.dataset == d]
                if q.empty: cells.append('—'); continue
                v = float(q.MAE.iloc[0]); cells.append(f'{v:.4f}')
                if d in base: rel.append(v / base[d])
            pr = int(r.params.iloc[0])
            lines.append(f'| {E9_DESC.get(tg, tg)} | {pr:,} | ' + ' | '.join(cells) +
                         f' | {np.mean(rel):.3f}× |' if rel else ' | — |')
    # metric table for the two most interesting architectures
    ps = agg[agg.model == 'pinn_semi']
    common = [d for d in dss if all(((ps.tag == t) & (ps.dataset == d)).any()
                                    for t in E9_ORDER if (ps.tag == t).any())]
    lines.append(f'\n**Bộ chỉ số mở rộng** — PINN-semi, 30 % nhãn, trung bình trên {len(common)} bộ '
                 f'({", ".join(common)}); mọi hàng dùng CÙNG tập bộ dữ liệu\n')
    cols = [('MAE', 'MAE'), ('MAE_cell', 'MAE theo cell'), ('MAE_cell_max', 'cell tệ nhất'),
            ('MAE_late', 'MAE vùng SOH≤0.90'), ('MonoViol', 'vi phạm đơn điệu'),
            ('Jitter', 'nhiễu quỹ đạo'), ('EOL_MAE', 'sai số EOL (chu kỳ)')]
    cols = [(c, l) for c, l in cols if c in agg.columns]
    lines.append('| kiến trúc | ' + ' | '.join(l for _, l in cols) + ' |')
    lines.append('|---|' + '---|' * len(cols))
    for tg in E9_ORDER:
        r = agg[(agg.model == 'pinn_semi') & (agg.tag == tg) & (agg.dataset.isin(common))]
        if r.empty: continue
        vals = []
        for c, _ in cols:
            v = r[c].mean()
            vals.append('—' if np.isnan(v) else (f'{v:.1f}' if c == 'EOL_MAE' else
                        (f'{v:.4f}' if v >= 1e-3 else f'{v:.5f}')))
        lines.append(f'| {E9_DESC.get(tg, tg)} | ' + ' | '.join(vals) + ' |')
    open(f'{OUT}/E9_table.md', 'w').write('\n'.join(lines))

    # figure: normalised MAE per architecture, grouped by model
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(E9_ORDER)); w = 0.38
    for i, (m, col, lab) in enumerate([('mlp', C['mlp'], 'chỉ dữ liệu'), ('pinn_semi', C['pinn_semi'], 'PINN-semi')]):
        vals = []
        for tg in E9_ORDER:
            rel = []
            for d in dss:
                r = agg[(agg.model == m) & (agg.tag == tg) & (agg.dataset == d)]
                b = agg[(agg.model == m) & (agg.tag == 'silu-mlp') & (agg.dataset == d)]
                if len(r) and len(b): rel.append(float(r.MAE.iloc[0]) / float(b.MAE.iloc[0]))
            vals.append(np.mean(rel) if rel else np.nan)
        ax.bar(x + (i - 0.5) * w, vals, w * 0.9, color=col, label=lab)
    ax.axhline(1.0, color=INK2, lw=1, ls='--')
    ax.set_xticks(x); ax.set_xticklabels([E9_DESC[t].replace(' + ', '\n+ ') for t in E9_ORDER], fontsize=8)
    ax.set_ylabel('MAE chuẩn hoá theo MLP+SiLU'); ax.legend(frameon=False, fontsize=9)
    ax.set_title('Kiến trúc & hàm kích hoạt — 30 % nhãn, trung bình 4 bộ (thấp hơn = tốt hơn)', loc='left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E9_arch.png'); plt.close(fig)
    return agg


# ------------------------------------------------------------------ E10 full transfer matrix
def E10():
    df = load('runs/E10/*.json')
    if df.empty: return
    rows = []
    for _, r in df.iterrows():
        for k in r.index:
            if k.startswith('transfer_') and k.endswith('_MAE') and not pd.isna(r[k]):
                rows.append(dict(source=r.dataset, target=k[9:-4], model=r.tag, norm=r.norm,
                                 seed=r.seed, MAE=r[k]))
    t = pd.DataFrame(rows)
    t.to_csv(f'{OUT}/E10_raw.csv', index=False)
    agg = t.groupby(['model', 'norm', 'source', 'target']).MAE.mean().reset_index()
    agg.to_csv(f'{OUT}/E10_table.csv', index=False)
    order = ['XJTU', 'TJU', 'MIT', 'HUST']
    CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}
    lines = []
    for norm in ['global', 'first_cycle']:
        for m, lab in [('mlp', 'Baseline chỉ dữ liệu'), ('pinn', 'PINN-semi')]:
            sub = agg[(agg.model == m) & (agg.norm == norm)]
            if sub.empty: continue
            lines.append(f'\n**{lab} · chuẩn hoá `{norm}`** — MAE trên tập test của bộ ĐÍCH '
                         f'(hàng = huấn luyện trên, cột = đánh giá trên; đường chéo = trong miền)\n')
            lines.append('| huấn luyện ↓ / đánh giá → | ' + ' | '.join(f'{d}<br><small>{CHEM[d]}</small>' for d in order) + ' |')
            lines.append('|---|' + '---|' * len(order))
            for src in order:
                cells = []
                for tgt in order:
                    q = sub[(sub.source == src) & (sub.target == tgt)]
                    if q.empty: cells.append('—'); continue
                    v = float(q.MAE.iloc[0])
                    cells.append(f'**{v:.4f}**' if src == tgt else (f'{v:.3f}' if v < 10 else f'{v:.0f}'))
                lines.append(f'| **{src}** | ' + ' | '.join(cells) + ' |')
    open(f'{OUT}/E10_table.md', 'w').write('\n'.join(lines))

    # figure: log-scale heatmap, one panel per (model, norm)
    combos = [(m, n) for m in ['mlp', 'pinn'] for n in ['global', 'first_cycle'] if
              not agg[(agg.model == m) & (agg.norm == n)].empty]
    if not combos: return agg
    fig, axes = plt.subplots(1, len(combos), figsize=(3.5 * len(combos), 3.4))
    axes = np.atleast_1d(axes)
    M = {}
    for (m, n) in combos:
        sub = agg[(agg.model == m) & (agg.norm == n)]
        M[(m, n)] = np.array([[sub[(sub.source == a) & (sub.target == b)].MAE.mean() for b in order] for a in order])
    vmin = max(1e-3, min(np.nanmin(v) for v in M.values())); vmax = max(np.nanmax(v) for v in M.values())
    from matplotlib.colors import LogNorm
    for ax, (m, n) in zip(axes, combos):
        im = ax.imshow(M[(m, n)], cmap='Blues', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xticks(range(4)); ax.set_xticklabels(order, fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(4)); ax.set_yticklabels(order, fontsize=8)
        ax.set_title(f'{"MLP" if m == "mlp" else "PINN"} · {n}', loc='left', fontsize=10)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                v = M[(m, n)][i, j]
                txt = f'{v:.3f}' if v < 1 else (f'{v:.1f}' if v < 100 else f'{v:.0f}')
                ax.text(j, i, txt, ha='center', va='center', fontsize=7.5,
                        color='white' if v > vmax ** 0.5 else INK)
    axes[0].set_ylabel('huấn luyện trên')
    fig.suptitle('Ma trận chuyển miền 4×4 — MAE trên tập test của bộ đích (thang log)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E10_matrix.png'); plt.close(fig)
    return agg


# ------------------------------------------------------------------ E11 metric suite on saved E1 predictions
def E11():
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pinnsoh.metrics import evaluate
    rows = []
    for p in glob.glob('runs/E1/*.json'):
        r = json.load(open(p))
        if 'test_predictions' not in r: continue
        c = r['config']; tp = r['test_predictions']
        m = evaluate(np.asarray(tp['y']), np.asarray(tp['p']), np.asarray(tp['cell']))
        rows.append(dict(dataset=c['dataset'], model=c['model'], frac=c['label_frac'], seed=c['seed'], **m))
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/E11_raw.csv', index=False)
    sub = df[df.frac == 0.3]
    cols = [('MAE', 'MAE'), ('MAE_cell', 'MAE theo cell'), ('MAE_cell_max', 'cell tệ nhất'),
            ('MAE_late', 'MAE SOH≤0.90'), ('MonoViol', 'vi phạm đơn điệu'), ('Jitter', 'nhiễu quỹ đạo'),
            ('EOL_MAE', 'sai số EOL (chu kỳ)'), ('EOL_cov', 'phủ EOL')]
    agg = sub.groupby(['dataset', 'model'])[[c for c, _ in cols]].mean().reset_index()
    agg.to_csv(f'{OUT}/E11_table.csv', index=False)
    lines = ['| bộ | mô hình | ' + ' | '.join(l for _, l in cols) + ' |', '|---|---|' + '---|' * len(cols)]
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        for m in ['mlp', 'pinn_semi']:
            r = agg[(agg.dataset == ds) & (agg.model == m)]
            if r.empty: continue
            vals = []
            for c, _ in cols:
                v = float(r[c].iloc[0])
                vals.append('—' if np.isnan(v) else (f'{v:.1f}' if c == 'EOL_MAE' else
                            (f'{v:.2f}' if c == 'EOL_cov' else (f'{v:.4f}' if v >= 1e-3 else f'{v:.5f}'))))
            lines.append(f'| {ds} | {LABEL[m].split(" (")[0]} | ' + ' | '.join(vals) + ' |')
    open(f'{OUT}/E11_table.md', 'w').write('\n'.join(lines))
    return agg


# ------------------------------------------------------------------ E12 fair tuning budget
def _pick(df, keys=('dataset', 'frac', 'model')):
    """Chọn theo MAE VALIDATION (không chạm test), trả về test MAE của biến thể được chọn."""
    out = []
    g = df.groupby(list(keys) + ['tag']).agg(val=('val_MAE', 'mean'), test=('test_MAE', 'mean'),
                                             sd=('test_MAE', 'std'), n=('seed', 'count')).reset_index()
    for k, sub in g.groupby(list(keys)):
        b = sub.loc[sub.val.idxmin()]
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                        tag=b.tag, val=b.val, test=b.test, sd=0 if np.isnan(b.sd) else b.sd,
                        n_variants=len(sub)))
    return pd.DataFrame(out)


def E12():
    d12 = load('runs/E12/*.json')
    if d12.empty: return
    e1 = load('runs/E1/*.json')
    sel = _pick(d12)
    sel.to_csv(f'{OUT}/E12_selected.csv', index=False)

    # bảng A — lưới GIỐNG HỆT nhau cho hai mô hình (so sánh cân bằng)
    rows = []
    for (ds, fr), sub in sel.groupby(['dataset', 'frac']):
        m = sub[sub.model == 'mlp']; p = sub[sub.model == 'pinn_semi']
        if m.empty or p.empty: continue
        d = e1[(e1.dataset == ds) & (e1.frac == fr) & (e1.model == 'mlp')].test_MAE
        rows.append(dict(dataset=ds, frac=fr, mlp_default=d.mean() if len(d) else np.nan,
                         mlp=float(m.test.iloc[0]), mlp_tag=m.tag.iloc[0], mlp_sd=float(m.sd.iloc[0]),
                         pinn=float(p.test.iloc[0]), pinn_tag=p.tag.iloc[0], pinn_sd=float(p.sd.iloc[0]),
                         n_variants=int(m.n_variants.iloc[0])))
    t = pd.DataFrame(rows)
    if t.empty: return
    t['ratio'] = t.pinn / t.mlp
    t['mlp_gain'] = 1 - t.mlp / t.mlp_default
    t.to_csv(f'{OUT}/E12_table.csv', index=False)

    lines = ['| bộ | % nhãn | MLP cấu hình mặc định | MLP sau tinh chỉnh | biến thể | PINN sau tinh chỉnh | biến thể | PINN/MLP |',
             '|---|---|---|---|---|---|---|---|']
    for _, r in t.sort_values(['dataset', 'frac']).iterrows():
        w = '**' if r.ratio < 1 else ''
        lines.append(f'| {r.dataset} | {int(r.frac*100)} % | {r.mlp_default:.4f} | {r.mlp:.4f} ± {r.mlp_sd:.4f} | '
                     f'`{r.mlp_tag}` | {w}{r.pinn:.4f} ± {r.pinn_sd:.4f}{w} | `{r.pinn_tag}` | {w}{r.ratio:.2f}×{w} |')
    open(f'{OUT}/E12_table.md', 'w').write('\n'.join(lines))

    # bảng B — mỗi mô hình dùng TOÀN BỘ kho biến thể sẵn có của nó
    pools = {}
    d9 = load('runs/E9/*.json'); d7 = load('runs/E7/*.json')
    mlp_pool = pd.concat([x for x in [d12[d12.model == 'mlp'], d9[d9.model == 'mlp'],
                                      e1[e1.model == 'mlp'].assign(tag='e1_default')] if not x.empty])
    pinn_pool = pd.concat([x for x in [d12[d12.model == 'pinn_semi'], d9[d9.model == 'pinn_semi'], d7] if not x.empty])
    b = []
    for name, pool in [('mlp', mlp_pool), ('pinn_semi', pinn_pool)]:
        q = _pick(pool.assign(model=name))
        q['which'] = name; b.append(q)
    bb = pd.concat(b)
    bb.to_csv(f'{OUT}/E12_poolB.csv', index=False)
    lines2 = ['| bộ | % nhãn | MLP (kho đầy đủ) | #biến thể | PINN (kho đầy đủ) | #biến thể | PINN/MLP |',
              '|---|---|---|---|---|---|---|']
    for (ds, fr), sub in bb.groupby(['dataset', 'frac']):
        m = sub[sub.which == 'mlp']; p = sub[sub.which == 'pinn_semi']
        if m.empty or p.empty: continue
        rt = float(p.test.iloc[0]) / float(m.test.iloc[0]); w = '**' if rt < 1 else ''
        lines2.append(f'| {ds} | {int(fr*100)} % | {float(m.test.iloc[0]):.4f} | {int(m.n_variants.iloc[0])} | '
                      f'{w}{float(p.test.iloc[0]):.4f}{w} | {int(p.n_variants.iloc[0])} | {w}{rt:.2f}×{w} |')
    open(f'{OUT}/E12_poolB.md', 'w').write('\n'.join(lines2))

    # hình: MLP mặc định -> MLP tinh chỉnh -> PINN tinh chỉnh, theo bộ và mức nhãn
    dss = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if (t.dataset == d).any()]
    fracs = sorted(t.frac.unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(4.4 * len(fracs), 3.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, fr in zip(axes, fracs):
        sub = t[t.frac == fr].set_index('dataset').reindex(dss)
        x = np.arange(len(dss)); w = 0.26
        ax.bar(x - w, sub.mlp_default, w * 0.9, color='#9ec5f4', label='MLP cấu hình mặc định')
        ax.bar(x, sub.mlp, w * 0.9, color=C['mlp'], label='MLP sau tinh chỉnh')
        ax.bar(x + w, sub.pinn, w * 0.9, color=C['pinn_semi'], label='PINN sau tinh chỉnh')
        ax.set_xticks(x); ax.set_xticklabels(dss); ax.set_title(f'{int(fr*100)} % nhãn', loc='left', fontweight='bold')
    axes[0].set_ylabel('MAE trên tập test'); axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('Cùng một ngân sách tinh chỉnh cho hai mô hình (8 biến thể, chọn theo validation)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_E12_fair.png'); plt.close(fig)
    return t


if __name__ == '__main__':
    E1(); E2(); E3(); E4(); E5(); E6(); E7(); E9(); E10(); E11(); E12()
    # Hình do figures.py vẽ (khuôn báo khoa học) — ghi đè các hình nháp ở trên.
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, 'figures.py'], check=False)
    for f in sorted(glob.glob(f'{OUT}/*.md')):
        print(f'\n===== {f}\n' + open(f).read())
