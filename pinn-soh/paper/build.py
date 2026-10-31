"""
Dựng bản thảo bài báo: bảng (paper/generated/table_*.tex), con số trong lời văn (generated/numbers.tex), hình
(paper/figures/*.pdf), rồi biên dịch paper/main.tex -> paper/manuscript.pdf (latexmk + pdflatex). Thư mục paper/ tự đủ:
các tệp sinh ra được đưa lên git, nên main.tex biên dịch được ngay (kể cả trên Overleaf) mà không cần chạy lại gì.

    python paper/build.py            # hoặc: python run.py paper
    python paper/build.py --no-pdf   # chỉ bảng, con số và hình

Nguồn số liệu, không số nào gõ tay:
  * bảng của báo cáo đồ án (archive/thesis/do-an/main.tex, đọc bằng paper/thesis_tables.py): dữ liệu, sai số theo
    mức nhãn, tách nguồn lợi ích, mô hình chính, ngân sách đo — sinh từ results/ của lần chạy công bố (commit 2d66f75);
  * benchmark/results/summary.csv, tests.csv, logged_constant.json (benchmark/compare.py): so với PINN4SOH;
  * results/thesis_leakage.csv (analysis/analyze_thesis.py; cần runs/E13 nhánh F và runs/E15): kiểm toán rò rỉ; khi
    chưa có, bảng dùng các tỉ số mà báo cáo đồ án nêu trong lời văn.
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'paper')]
_os.chdir(_ROOT)
import argparse, json, os, shutil, subprocess
import numpy as np
import pandas as pd

from thesis_tables import table, ratio

PAPER = 'paper'
GEN = os.path.join(PAPER, 'generated')
FIG = os.path.join(PAPER, 'figures')
BENCH = 'benchmark/results'                 # benchmark/compare.py
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = ['10', '30', '50', '70']


def fmt(v, n=2):
    return '--' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{n}f}'


def num(s):
    """'13,44 (0,84)' -> (13.44, 0.84); 13.44 -> (13.44, None)."""
    if isinstance(s, float):
        return s, None
    a, b = s.replace(')', '').split('(')
    return float(a.strip().replace(',', '.')), float(b.strip().replace(',', '.'))


def write(name, text):
    os.makedirs(GEN, exist_ok=True)
    open(os.path.join(GEN, name), 'w').write(text)


# --------------------------------------------------------------------------- bảng từ báo cáo đồ án
def table_data():
    chem = {'NCM': 'NCM', 'NCA/NCM': 'NCA, NCM, NCM+NCA', 'LFP': 'LFP'}
    out = [r'\begin{table}[t]\centering\small',
           r'\caption{Datasets after preprocessing. Protocols: distinct test conditions used for stratification. Life: '
           r'number of recorded cycles (min / median / max).}\label{tab:data}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{llrrrcccc}\toprule',
           r'Dataset & Chemistry & Cells & Cycles & Protocols & Life & Initial SOH & Final SOH & $Q_\text{nom}$ (Ah) \\\midrule']
    for r in table('tab:datastats'):
        if r[0] == 'Tổng':
            out.append(r'\midrule Total & & %d & %s & & & & & \\' % (r[2], f'{int(r[3]):,}'))
            continue
        life = r[5].replace('.', ',')
        dash = lambda t: t.replace(',', '.').replace('–', '--')
        out.append(f'{r[0]} & {chem[r[1]]} & {int(r[2])} & {int(r[3]):,} & {int(r[4])} & {life} & {dash(r[6])} & '
                   f'{dash(r[7])} & {str(r[8]).replace(" / ", "/").replace(",", ".")} \\\\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_data.tex', '\n'.join(out))


LABEL_MODELS = [('Ridge', 'Ridge'), ('GPR', 'GPR'), ('GBDT', 'GBDT'), ('MLP', 'MLP'), ('PINN-sup', 'PINN-sup'),
                ('PINN hộp đen', 'Black-box PINN'), ('PINN-semi', 'PINN-semi')]


def labels_table():
    rows = table('tab:big')
    V = {(r[0], r[1]): r[2:] for r in rows}
    out = [r'\begin{table}[t]\centering\small',
           r'\caption{Per-cell MAE ($\times10^{-3}$) on test cells at four fractions of labelled cells; five repetitions '
           r'$\times$ five folds, every cell tested five times. Bold: lowest per dataset and column.}\label{tab:labels}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{llrrrr}\toprule',
           r'Dataset & Model & 10\,\% & 30\,\% & 50\,\% & 70\,\% \\\midrule']
    for i, ds in enumerate(DS):
        best = [min(V[(ds, m)][j] for m, _ in LABEL_MODELS) for j in range(4)]
        for k, (m, lab) in enumerate(LABEL_MODELS):
            cells = [(r'\textbf{%s}' if V[(ds, m)][j] == best[j] else '%s') % fmt(V[(ds, m)][j]) for j in range(4)]
            out.append(f'{ds if k == 0 else ""} & {lab} & ' + ' & '.join(cells) + r' \\')
        if i < 3:
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_labels.tex', '\n'.join(out))
    return V


ABL = [('MLP', 'MLP'), ('PINN-sup', 'PINN-sup (physics on labelled cells only)'),
       ('chỉ đơn điệu', r'PINN-semi, monotonicity only ($\alpha = \gamma = 0$)'),
       ('PINN-semi (Euler)', 'PINN-semi, Euler residual along the trajectory'),
       ('PINN hộp đen', 'Black-box PINN (PINN4SOH-type dynamics)'), ('PINN-semi', 'PINN-semi')]


def ablation_table():
    V = {r[0]: r[1:] for r in table('tab:source')}
    out = [r'\begin{table}[t]\centering\small',
           r'\caption{Source of the gain at 30\,\% labelled cells: per-cell MAE ($\times10^{-3}$) and, in parentheses, '
           r'the ratio to the MLP.}\label{tab:ablation}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{lrrrr}\toprule',
           r'Model (30\,\% labels) & XJTU & TJU & MIT & HUST \\\midrule']
    for key, lab in ABL:
        cells = []
        for c in V[key]:
            v, rr = num(c)
            cells.append(fmt(v) + ('' if rr is None else f' ({fmt(rr)})'))
        out.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_ablation.tex', '\n'.join(out))


MAIN_COLS = ['MLP', 'CNN', 'LSTM', 'Transformer', 'GBDT', 'PINN-semi', 'PINN-hybrid', r'+ history$^\dagger$']


def main_table():
    rows = table('tab:allmae')
    out = [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4.5pt}',
           r'\caption{Per-cell MAE ($\times10^{-3}$) of the main models on the same folds and labelled cells (three '
           r'repetitions $\times$ five folds). PINN-hybrid: weights and filter window tuned on validation. '
           r'$^\dagger$Hybrid with or without causal feature history, selected on validation per dataset and label '
           r'fraction (post hoc). Bold: lowest per row.}\label{tab:main}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{llrrrrrrrr}\toprule',
           r'Dataset & Labels & ' + ' & '.join(MAIN_COLS) + r' \\\midrule']
    for i, r in enumerate(rows):
        vals = r[2:]
        best = min(vals)
        cells = [(r'\textbf{%s}' if v == best else '%s') % fmt(v) for v in vals]
        pct = r[1].replace('%', r'\,\%')
        out.append(f'{r[0] if r[1] == "10%" else ""} & {pct} & ' + ' & '.join(cells) + r' \\')
        if r[1] == '70%' and i < len(rows) - 1:
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_main.tex', '\n'.join(out))
    return {(r[0], r[1]): dict(zip(MAIN_COLS, r[2:])) for r in rows}


def budget_table():
    rows = table('tab:e28')
    out = [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption{Measurement budget: per-cell MAE ($\times10^{-3}$) on test cells, three repetitions; in parentheses, '
           r'the percentage of training cycles with a capacity measurement (70\,\% of the cells labelled is all of '
           r'them). Sparse: every training cell measured at its first cycle, every $qL$ cycles and its last cycle. Last '
           r'column: the cell in service is also measured every $0.1L$ cycles and the estimate is updated with the '
           r'Kalman filter; error on unmeasured cycles.}\label{tab:budget}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{lrrrrrr}\toprule',
           r' & \multicolumn{3}{c}{labels per cell} & \multicolumn{2}{c}{sparse, all cells} & sparse + in service \\'
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}',
           r'Dataset & Hybrid, 30\,\% & Hybrid, 70\,\% & GBDT, 70\,\% & Hybrid, $0.1L$ & Hybrid, $0.05L$ & '
           r'Hybrid, $0.1L$ \\\midrule']
    for r in rows:
        cells = []
        for c in r[1:]:
            v, p = num(c)
            cells.append(fmt(v) + ('' if p is None else f' ({p:.1f})'))
        out.append(f'{r[0]} & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_budget.tex', '\n'.join(out))


# --------------------------------------------------------------------------- kiểm toán rò rỉ
LEAK_SETTINGS = [('row', 'row-wise split'), ('minmax', 'per-cell min--max'), ('life3s', r'whole-life $3\sigma$ filter'),
                 ('normtest', 'normaliser incl.\ test')]


def leak_part():
    path, kept = 'results/thesis_leakage.csv', os.path.join(GEN, 'leakage.csv')
    if os.path.exists(path):                     # giữ một bản trong paper/ (results/ không lên git)
        os.makedirs(GEN, exist_ok=True)
        shutil.copy(path, kept)
    elif os.path.exists(kept):
        path = kept
    if os.path.exists(path):
        T = pd.read_csv(path).set_index('bo').reindex(DS)
        out = [r'\begin{table}[t]\centering\small',
               r'\caption{Leakage audit: per-cell MAE of the same MLP (70\,\% labels, two repetitions $\times$ five folds) '
               r'under the leakage-free protocol, and the ratio of the MAE under each faulty protocol to it (below 1: the '
               r'error looks smaller than it is). Last column: copying the label of the nearest training cycle of the '
               r'same cell under the row-wise split, without any feature.}\label{tab:leak}',
               r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{lrrrrrr}\toprule',
               r'Dataset & MAE ($\times10^{-3}$) & ' + ' & '.join(l for _, l in LEAK_SETTINGS) + r' & copy nearest \\\midrule']
        for ds, r in T.iterrows():
            out.append(f'{ds} & {fmt(1e3 * r.chuan)} & ' + ' & '.join(fmt(r[f"ti_so_{s}"]) for s, _ in LEAK_SETTINGS)
                       + f' & {fmt(r.ti_so_noi_suy_hang)} \\\\')
        out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
        lo = lambda s: T[f'ti_so_{s}'].min(); hi = lambda s: T[f'ti_so_{s}'].max()
        macros = dict(LeakRowAbstract=f'{1 / hi("row"):.1f}--{1 / lo("row"):.1f}',
                      LeakMinmaxAbstract=f'{lo("minmax"):.2f}--{hi("minmax"):.2f}',
                      LeakMaxFactor=f'{max(1 / lo("minmax"), hi("minmax")):.1f}')
        text = (f'A row-wise split lowers the error by a factor of {1 / hi("row"):.1f}--{1 / lo("row"):.1f}, and copying '
                f'the label of the nearest training cycle, without using any feature, gives '
                f'{lo("noi_suy_hang"):.2f}--{hi("noi_suy_hang"):.2f} times the error of the model evaluated correctly, '
                r'as the bound of Section~\ref{sec:data} predicts. ')
    else:
        # các tỉ số báo cáo đồ án nêu trong lời văn (Mục Kiểm toán rò rỉ); bảng đầy đủ cần chạy lại E13 nhánh F và E15
        mm = {'XJTU': 0.80, 'TJU': 1.51, 'MIT': 2.76, 'HUST': 0.46}
        out = [r'\begin{table}[t]\centering\small',
               r'\caption{Leakage audit: ratio of the per-cell MAE of the same MLP (70\,\% labels) under each faulty '
               r'protocol to its MAE under the leakage-free protocol (below 1: the error looks smaller than it '
               r'is).}\label{tab:leak}',
               r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{lcccc}\toprule',
               r'Dataset & ' + ' & '.join(l for _, l in LEAK_SETTINGS) + r' \\\midrule']
        for i, ds in enumerate(DS):
            ranges = [r'\multirow{4}{*}{0.56--0.83}', fmt(mm[ds]), r'\multirow{4}{*}{0.87--1.01}',
                      r'\multirow{4}{*}{0.98--1.03}']
            out.append(f'{ds} & ' + ' & '.join(c if (i == 0 or j == 1) else '' for j, c in enumerate(ranges)) + r' \\')
        out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
        text = ('A row-wise split lowers the error by a factor of 1.2--1.8, and copying the label of the nearest '
                'training cycle, without using any feature, is better than every real model evaluated correctly, as '
                r'the bound of Section~\ref{sec:data} predicts. ')
        T = pd.DataFrame({'ti_so_minmax': mm}).reindex(DS)
        macros = dict(LeakRowAbstract='1.2--1.8', LeakMinmaxAbstract='0.46--2.76', LeakMaxFactor='2.8')
    d = T['ti_so_minmax']
    text += (f'Per-cell min--max normalisation over the whole life, as in the PINN4SOH code, lowers the error on XJTU '
             f'({fmt(d["XJTU"])}) and HUST ({fmt(d["HUST"])}), where all cells end at almost the same SOH and the '
             f'normalised cycle index, the fraction of life already elapsed, nearly determines SOH; it raises the '
             f'error on TJU ({fmt(d["TJU"])}) and MIT ({fmt(d["MIT"])}), where end-of-life SOH varies and the '
             'normalisation erases the absolute level of the features. Either way the number does not describe '
             'deployment, where the life of a cell is unknown. A whole-life outlier filter and a normaliser fitted '
             'with test cells have smaller effects, but are leaks as well. All further results use the leakage-free '
             'protocol.')
    write('table_leak.tex', '\n'.join(out))
    return text, macros


# --------------------------------------------------------------------------- so với PINN4SOH
BENCH_ROWS = [(ds, 'full') for ds in DS] + [(ds, f'n{k}') for ds in ('XJTU', 'HUST') for k in (1, 2, 3, 4)]
BENCH_COLS = [('published_paper', 'Published'), ('rerun_published', 'Re-run, as published'),
              ('rerun_current', 'Re-run, current code'), ('paper_pinn_lai', 'Hybrid, original prep.'),
              ('paper_gbdt', 'GBDT, original prep.'), ('strict_pinn_lai', 'Hybrid, leakage-free')]


def bench_load():
    if not os.path.exists(f'{BENCH}/summary.csv'):
        return None, None
    S = pd.read_csv(f'{BENCH}/summary.csv').set_index(['dataset', 'mode'])
    T = pd.read_csv(f'{BENCH}/tests.csv') if os.path.exists(f'{BENCH}/tests.csv') else pd.DataFrame()
    return S, T


def _complete(S, key, col):
    c = f'{col}_complete'
    return col in S.columns and pd.notna(S.loc[key, col]) and (c not in S.columns or bool(S.loc[key, c]))


def bench_table(S, T):
    mark = {}
    if T is not None and len(T):
        for r in T[(T.a == 'paper_pinn_lai') & (T.b == 'rerun_published')].itertuples():
            if r.p_holm < 0.05:
                mark[(r.dataset, r.mode)] = '*' if r.ratio < 1 else r'\dagger'
    out = [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption{Per-cell MAE ($\times10^{-3}$) on the PINN4SOH splits, averaged over test cells, runs and groups '
           r'as in the original paper. Re-runs: the released code as published (label-only monotonicity term, logged '
           r'hyperparameters; ten runs per group, seeds 0--9) and as currently released (corrected term, new defaults; '
           r'three runs per group, seeds 0--2); $\pm$: standard deviation over runs. Ours: three seeds. $K$: labelled training cells in the '
           r'few-shot experiments. $^{*}$/$^{\dagger}$: hybrid with the original preprocessing significantly lower/higher '
           r'than the re-run as published (Wilcoxon over test cells, Holm across the 12 rows).}\label{tab:bench}',
           r'\begin{adjustbox}{max width=\textwidth}'
           r'\begin{tabular}{llrrrrrr}\toprule',
           r' & & \multicolumn{3}{c}{PINN4SOH} & \multicolumn{3}{c}{ours} \\\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
           r'Dataset & Setting & published & re-run, as published & re-run, current & hybrid, orig.\ prep. & '
           r'GBDT, orig.\ prep. & hybrid, leakage-free \\\midrule']
    for i, key in enumerate(BENCH_ROWS):
        cells = []
        for col, _ in BENCH_COLS:
            if not _complete(S, key, col):
                cells.append('--'); continue
            v = 1e3 * S.loc[key, col]
            sd = S.loc[key, f'{col}_sd'] if f'{col}_sd' in S.columns else np.nan
            s = fmt(v) + (f' $\\pm$ {fmt(1e3 * sd)}' if col.startswith('rerun') and pd.notna(sd) else '')
            if col == 'paper_pinn_lai' and key in mark:
                s += f'$^{{{mark[key]}}}$'
            cells.append(s)
        setting = 'all cells' if key[1] == 'full' else f'$K = {key[1][1]}$'
        out.append(f'{key[0] if (key[1] in ("full", "n1")) else ""} & {setting} & ' + ' & '.join(cells) + r' \\')
        if key == ('HUST', 'full') or key == ('XJTU', 'n4'):
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}', '']
    write('table_bench.tex', '\n'.join(out))


def _rng(vals, n=2):
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return '--'
    lo, hi = min(vals), max(vals)
    return f'{lo:.{n}f}' if f'{lo:.{n}f}' == f'{hi:.{n}f}' else f'{lo:.{n}f}--{hi:.{n}f}'


def bench_text(S, T):
    """Lời văn của mục so với PINN4SOH, mọi con số tính từ benchmark/results/."""
    full = [k for k in BENCH_ROWS if k[1] == 'full']
    few = [k for k in BENCH_ROWS if k[1] != 'full']
    have = lambda k, c: _complete(S, k, c)
    r = lambda a, b, ks: [S.loc[k, a] / S.loc[k, b] for k in ks if have(k, a) and have(k, b)]
    def wins(a, b, ks):
        if T is None or not len(T):
            return 0, 0
        g = T[(T.a == a) & (T.b == b) & T.apply(lambda x: (x.dataset, x['mode']) in ks, axis=1)]
        return int(((g.p_holm < 0.05) & (g.ratio < 1)).sum()), int(((g.p_holm < 0.05) & (g.ratio > 1)).sum())
    rep_full = r('rerun_published', 'published_paper', full)
    rep_few = r('rerun_published', 'published_paper', few)
    cur = r('rerun_current', 'rerun_published', full + few)
    hyb_full = r('paper_pinn_lai', 'rerun_published', full)
    hyb_few = r('paper_pinn_lai', 'rerun_published', few)
    hyb_pub_full = r('paper_pinn_lai', 'published_paper', full)
    hyb_pub_few = r('paper_pinn_lai', 'published_paper', few)
    strict_full = {k[0]: S.loc[k, 'strict_pinn_lai'] / S.loc[k, 'published_paper'] for k in full
                   if have(k, 'strict_pinn_lai')}
    w_full, l_full = wins('paper_pinn_lai', 'rerun_published', full)
    w_few, l_few = wins('paper_pinn_lai', 'rerun_published', few)
    def nruns(col):
        v = [S.loc[k, f'{col}_runs'] for k in BENCH_ROWS if have(k, col)]
        return int(np.nanmin(v)) if v else 0
    runs, runs_cur = nruns('rerun_published'), nruns('rerun_current')
    s = []
    s.append(r'PINN4SOH~\citep{wang2024} is the reference result on these four datasets. We reconstructed its 39 '
             r'evaluation groups (one model per batch for XJTU and TJU, one per dataset for MIT and HUST with 23 and 20 '
             r'fixed test cells, and few-shot runs with $K = 1, \dots, 4$ labelled cells for XJTU and HUST) from the '
             r'\texttt{data path} lines of the training logs released with the code, and evaluate everything as the '
             r'paper does: MAE per test cell, averaged over cells, runs and groups.')
    L = json.load(open(f'{BENCH}/logged_constant.json')) if os.path.exists(f'{BENCH}/logged_constant.json') else []
    n_const, n_eq = sum(x['constant'] for x in L), sum(x['equal'] for x in L)
    ex = next((x for x in L if x['key'] == 'XJTU|full|0-0'), None)
    s.append(r'\paragraph{The released configuration.} The training logs record, for every epoch, the mean ``physics '
             r"loss'' of the monotonicity term. In the first released version of the model this term is "
             r'\path{relu(y2-y1).sum()}, computed from the labels of consecutive cycles only. ' +
             ((f'In all {len(L)} groups' if n_const == len(L) else f'In {n_const} of the {len(L)} groups') +
              ' the logged value is one constant across all epochs of all ten runs, and ' +
              ('in all of them' if n_eq == len(L) else f'in {n_eq} of {len(L)}') +
              ' it equals this expression evaluated on the training rows that the original data loader builds from '
              'our copy of the data'
              + (f' (e.g.\ {ex["logged"][0]:.6f} for the XJTU 2C batch)' if ex else '') + '. '
              if L else '') +
             r'The term therefore adds a constant and no gradient: the published numbers come from a model trained '
             r'with the data and PDE terms only. The match also shows that our reconstructed splits and data reproduce '
             r'the training rows of every published group. Two later revisions replaced the term, first by '
             r'\path{relu(u2-u1).sum()} and then by \path{relu((u2-u1)*(y1-y2)).sum()}, and the latest commit '
             r'notes that the loss weights $\alpha$ and $\beta$ changed after a bug fix; the default weights in the '
             r'released scripts no longer match those in the logs.')
    rep_sig = wins('rerun_published', 'published', full)
    xj = [k for k in few if k[0] == 'XJTU' and k[1] != 'n1']
    cur_main = r('rerun_current', 'rerun_published', [k for k in full + few if k not in xj and k != ('XJTU', 'n1')])
    cur_xjtu = r('rerun_current', 'rerun_published', xj)
    cur_sig_main = sum(wins('rerun_current', 'rerun_published',
                            [k for k in full + few if k not in xj and k != ('XJTU', 'n1')]))
    cur_sig_xjtu = wins('rerun_current', 'rerun_published', xj)
    gb_full = r('paper_gbdt', 'rerun_published', full)
    gb_few = r('paper_gbdt', 'rerun_published', few)
    gb_vs_hyb = r('paper_gbdt', 'paper_pinn_lai', full + few)
    gw, gl = wins('paper_gbdt', 'paper_pinn_lai', full + few)
    sd_rel = [S.loc[k, 'rerun_published_sd'] / S.loc[k, 'rerun_published'] for k in BENCH_ROWS
              if have(k, 'rerun_published') and pd.notna(S.loc[k, 'rerun_published_sd'])]
    if rep_full:
        s.append(r'\paragraph{Re-running the original code.} We ran the released model and data-loading classes '
                 r'unchanged on the reconstructed splits, for two versions of the model: the version that produced the '
                 f'published numbers, with the logged hyperparameters ({runs} seeded runs per group, as in the paper), '
                 f'and the current version with its default weights ({runs_cur} runs per group); '
                 r'Table~\ref{tab:bench} and Figure~\ref{fig:bench} give the results. The original code needs '
                 r'pandas~1.x semantics; we '
                 r'used pandas~2.3, '
                 r'PyTorch~2 and CPUs, whereas the original runs used GPUs and older libraries. '
                 f'The re-run of the published version gives {_rng(rep_full)} times the published error on the four '
                 f'full datasets' + (f' and {_rng(rep_few)} times in the few-shot settings' if rep_few else '') +
                 f'. Per test cell, the re-run is significantly worse on {rep_sig[1]} of the 4 full datasets. The '
                 rf'standard deviation over runs is {_rng([100 * v for v in sd_rel], 0)}\,\% of the mean, so single runs '
                 'differ by as much as these gaps. Since the training rows are identical, the remaining differences are '
                 'hardware, library versions and random seeds; we could not reproduce the original environment and do '
                 'not know which of them causes the systematic part. We therefore report both the '
                 'published numbers and the re-run, and use the re-run for paired tests because it gives per-cell '
                 'predictions under known conditions. ' +
                 (f'The corrected current version gives {_rng(cur_main)} times the error of the published version on '
                  f'the full datasets and the HUST few-shot settings (significantly different in {cur_sig_main} of '
                  f'{len(cur_main)}), but {_rng(cur_xjtu)} times on the XJTU few-shot settings with $K \\ge 2$ '
                  f'(significantly higher in {cur_sig_xjtu[1]} of {len(cur_xjtu)}), where the released script keeps '
                  r'$\beta = 10$ for the monotonicity term that is now active.' if cur_main and cur_xjtu else ''))
    if hyb_full:
        s.append(r'\paragraph{Same splits, same preprocessing.} With the original preprocessing (whole-life $3\sigma$ '
                 r'filter on all columns including capacity, per-cell min--max normalisation of features and cycle '
                 r'index), PINN-hybrid in its fixed configuration ($\omega = \tfrac12$, $w = 10$, three seeds, no tuning '
                 f'on these splits) has {_rng(hyb_full)} times the error of the re-run original code on the full '
                 f'datasets (significantly lower in {w_full} and higher in {l_full} of 4) and {_rng(hyb_few)} times in '
                 f'the few-shot settings (significantly lower in {w_few} and higher in {l_few} of 8). Against the published numbers the '
                 f'ratios are {_rng(hyb_pub_full)} and {_rng(hyb_pub_few)}. In the few-shot settings the unlabelled '
                 'training cells enter the physics losses of our model, which is where the semi-supervised design '
                 'pays off. A GBDT trained with the same preprocessing does about as well: it has '
                 f'{_rng(gb_full)} times the re-run error on the full datasets and {_rng(gb_few)} times in the few-shot '
                 f'settings, and compared with PINN-hybrid it is significantly lower in {gw} and higher in {gl} of the '
                 f'12 settings (ratio {_rng(gb_vs_hyb)}). With per-cell min--max normalisation the normalised cycle '
                 'index is the fraction of life already elapsed, which a tree exploits directly, so under these '
                 'conditions the advantage over the original code cannot be credited to the physics.')
    if strict_full:
        better = [d for d, v in strict_full.items() if v < 0.95]
        worse = [d for d, v in strict_full.items() if v > 1.05]
        sw, sl = wins('strict_pinn_lai', 'published', full)
        s.append(r'\paragraph{Without the leaks.} With our leakage-free preprocessing the same model has '
                 + ', '.join(f'{v:.2f} ({d})' for d, v in strict_full.items()) +
                 f' times the published error on the full datasets (per test cell significantly lower on {sw} and '
                 f'higher on {sl} of 4). The direction follows the leakage audit '
                 r'(Section~\ref{sec:leak}): the original preprocessing helps on datasets where the elapsed fraction of '
                 r'life determines SOH' + (f' ({", ".join(worse)})' if worse else '') +
                 r' and hurts where the absolute feature level matters' + (f' ({", ".join(better)})' if better else '')
                 + '. A comparison that changes the preprocessing therefore cannot separate the model from the '
                 'protocol, and the published numbers on the few-shot settings, where knowing the elapsed fraction of '
                 'life is almost knowing SOH, mostly reflect the normalisation.')
    if not hyb_full and not rep_full:
        s.append('(Benchmark results are not available yet: run \\texttt{python run.py benchmark}.)')
    abstract = ('has ' + _rng(hyb_full) + ' times the error of the re-run original code on the full datasets and '
                + _rng(hyb_few) + ' times in its few-shot settings, about as low as a GBDT with the same '
                'preprocessing') if hyb_full and hyb_few else \
        'is compared with the re-run original code (results pending)'
    disc = (r'Three practical points follow. Reported numbers should come with per-cell errors and the exact splits, '
            r'which made the comparison possible here; the logs released with PINN4SOH allowed us to reconstruct both. '
            r'Released code should be the code that produced the numbers, or the differences should be listed: a '
            r'constant loss term and changed weights alter the method that is being compared. And preprocessing that '
            r'uses the whole life of a cell should be avoided, because its effect on the error depends on the dataset '
            r'and can be large in either direction.')
    return '\n\n'.join(s), abstract, disc


# --------------------------------------------------------------------------- hình
COL = {'mlp': '#2a78d6', 'pinn_sup': '#eb6834', 'pinn_semi': '#1baf7a', 'pinn_bb': '#eda100', 'gbdt': '#3f4a52'}


def _theme():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#78848d', 'axes.linewidth': 0.6,
                         'xtick.color': '#4a5563', 'ytick.color': '#4a5563', 'axes.labelcolor': '#1b2330',
                         'pdf.fonttype': 42, 'svg.hashsalt': 'pinnsoh', 'figure.dpi': 150})
    return plt


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight', dpi=200)


def fig_labels(V):
    plt = _theme()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3))
    x = [10, 30, 50, 70]
    series = [('MLP', 'mlp', 'MLP'), ('PINN-sup', 'pinn_sup', 'PINN-sup'), ('PINN hộp đen', 'pinn_bb', 'Black-box PINN'),
              ('PINN-semi', 'pinn_semi', 'PINN-semi'), ('GBDT', 'gbdt', 'GBDT')]
    for ax, ds in zip(axes, DS):
        for key, c, lab in series:
            ax.plot(x, V[(ds, key)], color=COL[c], lw=1.6 if c in ('pinn_semi', 'gbdt') else 1.1,
                    ls='--' if c == 'gbdt' else '-', marker='o', ms=3.2, label=lab)
        ax.set_yscale('log')
        ax.set_xticks(x); ax.set_xticklabels([f'{v}%' for v in x])
        lo, hi = ax.get_ylim()
        ticks = [t for t in (1.5, 2, 3, 5, 8, 10, 15, 20, 30, 40) if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
        ax.grid(axis='y', color='#e7ebe9', lw=0.6)
        ax.set_title(ds, loc='left', fontweight='bold')
        ax.set_xlabel('labelled cells')
    axes[0].set_ylabel(r'per-cell MAE ($\times10^{-3}$)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.0)
    _save(fig, 'fig_labels')
    plt.close(fig)


def fig_benchmark(S):
    plt = _theme()
    fig = plt.figure(figsize=(7.0, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.38)
    cols = [('published_paper', 'PINN4SOH, published', '#9aa5ad', '--', 'o'),
            ('rerun_published', 'PINN4SOH re-run, as published', '#eb6834', '-', 's'),
            ('rerun_current', 'PINN4SOH re-run, current code', '#eda100', '-', 'D'),
            ('paper_pinn_lai', 'PINN-hybrid, original preprocessing', '#2a78d6', '-', 'o'),
            ('strict_pinn_lai', 'PINN-hybrid, leakage-free', '#0b3d73', '-', '^')]
    for i, ds in enumerate(DS):
        ax = fig.add_subplot(gs[0, i])
        key = (ds, 'full')
        for j, (c, lab, color, _, mk) in enumerate(cols):
            if not _complete(S, key, c):
                continue
            v = 1e3 * S.loc[key, c]
            sd = S.loc[key, f'{c}_sd'] if f'{c}_sd' in S.columns else np.nan
            ax.bar(j, v, color=color, width=0.72, label=lab)
            if c.startswith('rerun') and pd.notna(sd):
                ax.errorbar(j, v, yerr=1e3 * sd, color='#1b2330', lw=0.8, capsize=2)
        ax.set_xticks([]); ax.grid(axis='y', color='#e7ebe9', lw=0.6); ax.set_axisbelow(True)
        ax.set_title(ds, loc='left', fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'per-cell MAE ($\times10^{-3}$)')
    for i, ds in enumerate(('XJTU', 'HUST')):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        K = [1, 2, 3, 4]
        for c, lab, color, ls, mk in cols:
            keys = [(ds, f'n{k}') for k in K]
            if not all(_complete(S, k, c) for k in keys):
                continue
            ax.plot(K, [1e3 * S.loc[k, c] for k in keys], color=color, ls=ls, marker=mk, ms=3.4, lw=1.3, label=lab)
        ax.set_yscale('log'); ax.set_xticks(K)
        lo, hi = ax.get_ylim()
        ticks = [t for t in (3, 5, 8, 10, 15, 20, 30, 50, 80) if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
        ax.grid(axis='y', color='#e7ebe9', lw=0.6)
        ax.set_xlabel('labelled training cells $K$' + (' per batch' if ds == 'XJTU' else ''))
        ax.set_title(f'{ds}, few-shot', loc='left', fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'per-cell MAE ($\times10^{-3}$)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[2]) for c in cols]
    fig.legend(handles, [c[1] for c in cols], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.17)
    _save(fig, 'fig_benchmark')
    plt.close(fig)


# --------------------------------------------------------------------------- dựng
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pdf', action='store_true')
    a = ap.parse_args()
    table_data()
    V = labels_table()
    ablation_table()
    main_table()
    budget_table()
    leak, leak_macros = leak_part()
    S, T = bench_load()
    if S is not None:
        bench_table(S, T)
        text, abstract, disc = bench_text(S, T)
        fig_benchmark(S)
    else:
        write('table_bench.tex', '')
        text, abstract, disc = ('(Benchmark results are not available yet: run \\texttt{python run.py benchmark}.)',
                                'is compared with the re-run original code (results pending)', '')
    fig_labels(V)
    _sys.path.insert(0, os.path.join(_ROOT, 'analysis'))
    import diagrams
    os.makedirs(FIG, exist_ok=True)
    diagrams.arch('en', os.path.join(FIG, 'fig_arch'))
    write('numbers.tex', '\n'.join([r'\newcommand{\LeakText}{' + leak + '}',
                                    *[f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in leak_macros.items()],
                                    r'\newcommand{\BenchText}{' + text + '}',
                                    r'\newcommand{\BenchAbstract}{' + abstract + '}',
                                    r'\newcommand{\BenchDiscussion}{' + disc + '}', '']))
    if a.no_pdf:
        return
    if not shutil.which('latexmk'):
        raise SystemExit('cần latexmk + pdflatex (TeX Live) để dựng paper/main.pdf')
    r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=PAPER,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:])
        raise SystemExit('latexmk lỗi')
    shutil.copy(os.path.join(PAPER, 'main.pdf'), os.path.join(PAPER, 'manuscript.pdf'))
    print('đã dựng', os.path.join(PAPER, 'manuscript.pdf'))


if __name__ == '__main__':
    main()
