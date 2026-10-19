"""
Phân tích và hình cho báo cáo dạng đồ án.

    python analyze_thesis.py          # -> results/thesis_*.csv, results/fig_thesis_*.png|pdf

Gom các thí nghiệm theo giao thức kiểm định chéo theo cell (E13, E16, E17), kiểm toán rò rỉ (E15) và
chuyển miền (E14/E18). Mọi chỉ số tính lại từ dự đoán test đã lưu. Hình không mang chữ chú thích.
"""
from __future__ import annotations
import glob, itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = 'results'
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
RNG = np.random.default_rng(20261019)
B = 10_000

MODELS = ['mlp', 'ridge', 'gbdt', 'gpr', 'pinn_sup', 'pinn_bb', 'pinn_semi']
LABEL = {'mlp': 'MLP', 'ridge': 'Ridge', 'gbdt': 'GBDT', 'gpr': 'GPR', 'pinn_sup': 'PINN-sup',
         'pinn_bb': 'PINN hộp đen', 'pinn_semi': 'PINN-semi'}
COLOR = {'mlp': '#2a78d6', 'pinn_sup': '#eb6834', 'pinn_semi': '#1baf7a', 'pinn_bb': '#eda100',
         'ridge': '#e87ba4', 'gbdt': '#7b5cd6', 'gpr': '#8a5a00'}
MARK = {'mlp': 's', 'ridge': 's', 'gbdt': 's', 'gpr': 's', 'pinn_sup': 'o', 'pinn_bb': 'o', 'pinn_semi': 'o'}
SUPERVISED = ['mlp', 'ridge', 'gbdt', 'gpr']


# ════════════════════════════════════════════════════════════ 1. bảng theo cell
def _records(path):
    r = json.load(open(path)); c = r['config']; tp = r.get('test_predictions')
    if tp is None:
        return []
    tag, model, frac = c.get('tag', ''), c['model'], c['label_frac']
    if tag == 'D':
        model = 'mono_only'
    elif tag == 'E':
        model = 'euler'
    y = np.asarray(tp['y'], float); p = np.asarray(tp['p'], float); cell = np.asarray(tp['cell'])
    ids = tp['cell_ids']; nt = r['n_cells']['train_labelled'] + r['n_cells']['train_unlabelled']
    out = []
    for k in np.unique(cell):
        m = cell == k
        out.append((model, frac, c['dataset'], c['repeat'], c['fold'], ids[k], float(np.abs(p[m] - y[m]).mean()),
                    int(m.sum()), nt, r['n_cells']['test']))
    return out


def percell(refresh=False):
    path = f'{OUT}/thesis_percell.csv.gz'
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    rows = []
    for f in sorted(glob.glob('runs/E13/*.json') + glob.glob('runs/E16/*.json') + glob.glob('runs/E17/*.json')):
        rows += _records(f)
    df = pd.DataFrame(rows, columns=['model', 'frac', 'dataset', 'repeat', 'fold', 'cell', 'mae', 'n', 'n_train', 'n_test'])
    # ở 70 % nhãn mọi cell train đều có nhãn -> PINN-sup trùng từng bit với PINN-semi (không chạy lại)
    g = df[(df.model == 'pinn_semi') & (df.frac == 0.7)].copy(); g['model'] = 'pinn_sup'
    df = pd.concat([df, g], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def cell_mean(df, model, frac, ds):
    g = df[(df.model == model) & (df.frac == frac) & (df.dataset == ds)]
    return g.groupby('cell').mae.mean()


# ════════════════════════════════════════════════════════════ 2. thiết kế cách chia
def split_design():
    """Bảng khả thi cho số fold K, với validation v = 10 %, mức nhãn lớn nhất 70 %."""
    from pinnsoh.data import load_dataset, cv_split
    rows = []
    for ds in DATASETS:
        cells = load_dataset('PINN4SOH/data', ds, clean='causal'); N = len(cells)
        for K in [3, 4, 5, 10]:
            tr, va, te = [], [], []
            for k in range(K):
                sp = cv_split(cells, k, 0, K, 0.10)
                tr.append(len(sp['train'])); va.append(len(sp['val'])); te.append(len(sp['test']))
            rows.append(dict(bo=ds, N=N, K=K, train_min=min(tr), val=va[0], test_min=min(te), test_max=max(te),
                             frac_train=min(tr) / N, nhan_70=int(round(0.7 * N)), nhan_10=int(round(0.1 * N)),
                             kha_thi_70=int(round(0.7 * N)) <= min(tr) + 1, luot=K * 5))
    return pd.DataFrame(rows)


def stratification(repeats=5):
    """Ma trận batch × fold (số cell test), độ lệch lớn nhất giữa các fold trong từng batch."""
    from pinnsoh.data import load_dataset, cv_split
    mats, rows = {}, []
    for ds in DATASETS:
        cells = load_dataset('PINN4SOH/data', ds, clean='causal')
        batch = {c.cid: c.batch for c in cells}
        bs = sorted(set(batch.values()))
        worst, tested = 0, []
        for r in range(repeats):
            M = np.zeros((len(bs), 5), int); seen = []
            for k in range(5):
                for cid in cv_split(cells, k, r, 5, 0.10)['test']:
                    M[bs.index(batch[cid]), k] += 1; seen.append(cid)
            worst = max(worst, int((M.max(1) - M.min(1)).max()))
            tested.append(len(seen) == len(cells) == len(set(seen)))
            if r == 0:
                mats[ds] = (bs, M)
        rows.append(dict(bo=ds, so_batch=len(bs), lech_lon_nhat=worst, moi_cell_test_dung_1_lan=all(tested)))
    return pd.DataFrame(rows), mats


def repeat_convergence(df):
    """Mô phỏng lấy lại: dùng R' ∈ {1..5} lần lặp (mọi tổ hợp), ước lượng tỉ số PINN-semi/MLP ở 30 % nhãn."""
    rows = []
    for ds in DATASETS:
        a = df[(df.model == 'mlp') & (df.frac == 0.3) & (df.dataset == ds)]
        c = df[(df.model == 'pinn_semi') & (df.frac == 0.3) & (df.dataset == ds)]
        for R in range(1, 6):
            ests, widths, ps = [], [], []
            for sub in itertools.combinations(range(5), R):
                ea = a[a.repeat.isin(sub)].groupby('cell').mae.mean()
                ec = c[c.repeat.isin(sub)].groupby('cell').mae.mean().reindex(ea.index)
                x, y = ea.values, ec.values
                ests.append(y.mean() / x.mean())
                idx = RNG.integers(0, len(x), size=(2000, len(x)))
                rr = y[idx].mean(1) / x[idx].mean(1)
                widths.append(np.percentile(rr, 97.5) - np.percentile(rr, 2.5))
                ps.append(stats.wilcoxon(y, x).pvalue)
            rows.append(dict(bo=ds, R=R, n_to_hop=len(ests), ti_so_tb=np.mean(ests), ti_so_sd=np.std(ests),
                             ti_so_min=np.min(ests), ti_so_max=np.max(ests), rong_KTC=np.mean(widths),
                             p_trung_vi=np.median(ps)))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════ 3. so sánh theo lượng nhãn
def curve(df):
    rows = []
    for m, f, ds in itertools.product(MODELS + ['mono_only', 'euler'], FRACS, DATASETS):
        e = cell_mean(df, m, f, ds)
        if e.empty:
            continue
        v = e.values; idx = RNG.integers(0, len(v), size=(B, len(v))); bm = v[idx].mean(1)
        rows.append(dict(model=m, frac=f, bo=ds, MAE_cell=v.mean(), lo=np.percentile(bm, 2.5),
                         hi=np.percentile(bm, 97.5), n_cell=len(v),
                         n_lap=int(df[(df.model == m) & (df.frac == f) & (df.dataset == ds)].repeat.nunique())))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); prev = 0
    for rank, i in enumerate(o):
        prev = max(prev, (m - rank) * p[i]); adj[i] = min(1, prev)
    return adj


def nb_t(d, n_test, n_train):
    d = np.asarray(d, float); J = len(d)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    return 2 * stats.t.sf(abs(t), J - 1)


def compare_all(df, target='pinn_semi'):
    """PINN-semi so với từng mô hình khác, ở từng mức nhãn. Họ Holm = 4 bộ dữ liệu của một phép so."""
    out = []
    for other, f in itertools.product([m for m in MODELS if m != target], FRACS):
        if other == 'pinn_sup' and f == 0.7:
            continue                                   # trùng từng bit với PINN-semi ở 70 % nhãn
        block = []
        for ds in DATASETS:
            ea, eb = cell_mean(df, other, f, ds), cell_mean(df, target, f, ds)
            common = ea.index.intersection(eb.index)
            if len(common) < 6:
                continue
            x, y = ea.loc[common].values, eb.loc[common].values
            idx = RNG.integers(0, len(x), size=(B, len(x))); rr = y[idx].mean(1) / x[idx].mean(1)
            fa = df[(df.model == other) & (df.frac == f) & (df.dataset == ds)].groupby(['repeat', 'fold'])
            fb = df[(df.model == target) & (df.frac == f) & (df.dataset == ds)].groupby(['repeat', 'fold'])
            ka, kb = fa.mae.mean(), fb.mae.mean(); kk = ka.index.intersection(kb.index)
            ntest, ntrain = fa.n_test.first().mean(), fa.n_train.first().mean()
            block.append(dict(doi_chung=other, frac=f, bo=ds, n_cell=len(common), mae_doi_chung=x.mean(),
                              mae_pinn=y.mean(), ti_so=y.mean() / x.mean(), lo=np.percentile(rr, 2.5),
                              hi=np.percentile(rr, 97.5), cell_pinn_tot_hon=float((y < x).mean()),
                              p=stats.wilcoxon(y, x).pvalue,
                              p_nb=nb_t(kb.loc[kk].values - ka.loc[kk].values, ntest, ntrain) if len(kk) > 2 else np.nan))
        if block:
            ph = holm([b['p'] for b in block]); pn = holm([b['p_nb'] for b in block])
            for b, a1, a2 in zip(block, ph, pn):
                b['p_holm'] = a1; b['p_nb_holm'] = a2
            out += block
    return pd.DataFrame(out)


def equivalence(cur):
    """Lượng nhãn tương đương: MLP (và đường bao tốt nhất của các mô hình có nhãn) cần bao nhiêu % nhãn để
    đạt MAE của PINN-semi ở mức f. Nội suy tuyến tính log(MAE) theo f giữa các mốc 10/30/50/70 %.
    -1 nghĩa là ≤ 10 % (đối chứng đã tốt bằng ngay ở 10 %), inf nghĩa là > 70 % (chưa đạt trong dải đo)."""
    rows = []
    for ds in DATASETS:
        g = cur[cur.bo == ds]
        sup = {m: g[g.model == m].set_index('frac').MAE_cell.sort_index() for m in SUPERVISED if (g.model == m).any()}
        if not sup:
            continue
        env = pd.concat(sup.values(), axis=1).min(axis=1)
        for f in [0.1, 0.3, 0.5]:
            tt = g[(g.model == 'pinn_semi') & (g.frac == f)].MAE_cell
            if tt.empty:
                continue
            t = float(tt.iloc[0])
            for name, c in [('mlp', sup.get('mlp')), ('tot_nhat_co_nhan', env)]:
                if c is None or len(c) < 2:
                    continue
                xs, ys = np.array(c.index, float), np.log(c.values); lt = np.log(t)
                if lt >= ys[0]:
                    feq = -1.0
                elif lt < ys.min():
                    feq = np.inf
                else:
                    feq = np.nan
                    for a in range(len(xs) - 1):
                        if ys[a] >= lt >= ys[a + 1]:
                            feq = xs[a] + (lt - ys[a]) * (xs[a + 1] - xs[a]) / (ys[a + 1] - ys[a]) if ys[a + 1] != ys[a] else xs[a]
                            break
                rows.append(dict(bo=ds, frac_pinn=f, doi_chung=name, mae_pinn=t, frac_tuong_duong=feq))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════ 4. rò rỉ
def leakage():
    rows = []
    for ds in DATASETS:
        for r in [0, 1]:
            base = []
            for k in range(5):
                p = glob.glob(f'runs/E13/F_{ds}_mlp_f0.7_s{100 * r + k}_*_cv{r}-{k}of5.json')
                if p:
                    base.append(json.load(open(p[0]))['test']['MAE_cell'])
            if len(base) == 5:
                rows.append(dict(bo=ds, thiet_lap='chuan', repeat=r, MAE_cell=np.mean(base)))
            for s in ['row', 'minmax', 'life3s', 'normtest']:
                v = [json.load(open(p))['test']['MAE_cell'] for p in sorted(glob.glob(f'runs/E15/{s}_{ds}_cv{r}-*.json'))]
                if len(v) == 5:
                    rows.append(dict(bo=ds, thiet_lap=s, repeat=r, MAE_cell=np.mean(v)))
            rows.append(dict(bo=ds, thiet_lap='noi_suy_hang', repeat=r, MAE_cell=naive_row_interp(ds, r)))
    d = pd.DataFrame(rows)
    t = d.groupby(['bo', 'thiet_lap']).MAE_cell.mean().unstack()
    for c in t.columns:
        if c != 'chuan':
            t[f'ti_so_{c}'] = t[c] / t['chuan']
    return d, t.reset_index()


def naive_row_interp(ds, r, K=5):
    """Dự đoán 'ngây thơ' KHÔNG dùng đặc trưng: SOH của chu kỳ train gần nhất trong CÙNG cell, dưới
    cách chia theo hàng (cùng bộ sinh ngẫu nhiên với thiết lập 'row' của E15)."""
    from pinnsoh.data import load_dataset
    cells = load_dataset('PINN4SOH/data', ds, clean='causal')
    per_fold = []
    for k in range(K):
        rng = np.random.RandomState(50_000 + 100 * r + k)
        maes = []
        for c in cells:
            n = len(c.soh)
            fold = rng.randint(0, K, size=n); is_te = fold == k
            is_va = (~is_te) & (rng.rand(n) < 0.125); tr = ~is_te & ~is_va
            if is_te.sum() < 2 or tr.sum() < 1:
                continue
            ct, yt = c.cycle[tr], c.soh[tr]
            pos = np.clip(np.searchsorted(ct, c.cycle[is_te]), 1, len(ct) - 1) if len(ct) > 1 else np.zeros(is_te.sum(), int)
            if len(ct) > 1:
                left = np.abs(c.cycle[is_te] - ct[pos - 1]) <= np.abs(ct[pos] - c.cycle[is_te])
                pred = np.where(left, yt[pos - 1], yt[pos])
            else:
                pred = np.full(is_te.sum(), yt[0])
            maes.append(np.abs(pred - c.soh[is_te]).mean())
        per_fold.append(np.mean(maes))
    return float(np.mean(per_fold))


def row_bound():
    """Cận của dự đoán nội suy khi chia theo chu kỳ (Mệnh đề trong Chương 5): sai số ≈ E|y_n − y_{n+d}|,
    d = khoảng cách tới chu kỳ train gần nhất. Với xác suất một chu kỳ thuộc train q = 0,7, E[d] = 1/(1 − (1 − q)^2)."""
    from pinnsoh.data import load_dataset
    q = 0.7; Ed = 1 / (1 - (1 - q) ** 2)
    rows = []
    for ds in DATASETS:
        cells = load_dataset('PINN4SOH/data', ds, clean='causal')
        dy = np.array([np.abs(np.diff(c.soh)).mean() for c in cells])        # trung bình theo cell, như MAE_cell
        slope = np.mean([(c.soh[0] - c.soh[-1]) / max(1, c.cycle[-1] - c.cycle[0]) for c in cells])
        rows.append(dict(bo=ds, E_d=Ed, dy_tb=float(dy.mean()), doc_tb=float(slope), can_tren=float(dy.mean() * Ed)))
    return pd.DataFrame(rows)


def row_sim(qs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), reps=3):
    """Mô phỏng Mệnh đề chia theo chu kỳ: (a) E[d] theo q bằng Monte Carlo trên chuỗi dài; (b) chép nhãn của chu kỳ
    train gần nhất trên quỹ đạo SOH thật, so với cận E[d]·mean|Δy|."""
    from pinnsoh.data import load_dataset
    rng = np.random.default_rng(2026)
    rows = []
    for q in qs:
        tr = rng.random(1_000_000) < q
        pos = np.flatnonzero(tr); te = np.flatnonzero(~tr)
        te = te[(te > pos[0]) & (te < pos[-1])]
        j = np.searchsorted(pos, te)
        d = np.minimum(te - pos[j - 1], pos[j] - te)
        rows.append(dict(bo='mo_phong', q=q, E_d_mc=float(d.mean()), E_d=1 / (1 - (1 - q) ** 2)))
    for ds in DATASETS:
        cells = load_dataset('PINN4SOH/data', ds, clean='causal')
        dy = float(np.mean([np.abs(np.diff(c.soh)).mean() for c in cells]))
        for q in qs:
            maes = []
            for _ in range(reps):
                per = []
                for c in cells:
                    y = c.soh; tr = rng.random(len(y)) < q
                    if tr.sum() < 1 or (~tr).sum() < 1:
                        continue
                    pos = np.flatnonzero(tr); te = np.flatnonzero(~tr)
                    j = np.clip(np.searchsorted(pos, te), 1, len(pos))
                    left = pos[np.clip(j - 1, 0, len(pos) - 1)]; right = pos[np.clip(j, 0, len(pos) - 1)]
                    near = np.where(np.abs(te - left) <= np.abs(right - te), left, right)
                    per.append(np.abs(y[te] - y[near]).mean())
                maes.append(np.mean(per))
            rows.append(dict(bo=ds, q=q, mae_chep=float(np.mean(maes)), can=dy / (1 - (1 - q) ** 2)))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════ 5. chuyển miền
def transfer():
    rows = []
    for f in glob.glob('runs/E14/samenorm_*.json'):
        r = json.load(open(f)); c = r['config']
        rows.append(dict(nguon=c['dataset'], dich=c['target'], k=c['k_target'], model=c['model'], seed=c['seed'],
                         MAE_cell=r['target_test']['MAE_cell']))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return d.groupby(['nguon', 'dich', 'k', 'model']).MAE_cell.agg(['mean', 'std', 'count']).reset_index()


# ════════════════════════════════════════════════════════════ 6. hình
def figures(df, cur, cmp_, conv, lk, mats, tr, rs=None):
    import matplotlib, matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap
    from pinnsoh.plotstyle import apply_theme, GRID, CHEM, panel_title, save, log_mae_axis, SEQ_CMAP
    from pinnsoh.data import load_dataset, cv_split, choose_labelled
    apply_theme()

    # 6.1 sơ đồ chia dữ liệu thật của XJTU, lần lặp 0, 30 % nhãn
    cells = load_dataset('PINN4SOH/data', 'XJTU', clean='causal'); by = {c.cid: c for c in cells}
    order = sorted(cells, key=lambda c: (c.batch, c.cid)); ids = [c.cid for c in order]
    R = np.zeros((5, len(ids)), int)
    for k in range(5):
        sp = cv_split(cells, k, 0, 5, 0.10)
        lab, unl = choose_labelled(sp['train'], 0.3, len(cells), k, by)
        for cid in sp['test']: R[k, ids.index(cid)] = 0
        for cid in sp['val']: R[k, ids.index(cid)] = 1
        for cid in lab: R[k, ids.index(cid)] = 2
        for cid in unl: R[k, ids.index(cid)] = 3
    roles = ['test', 'validation', 'train có nhãn', 'train không nhãn']
    cmap = ListedColormap(['#eb6834', '#9aa5ad', '#2a78d6', '#b9d3f2'])
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.imshow(R, aspect='auto', cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
    ax.set_yticks(range(5)); ax.set_yticklabels([f'fold {k + 1}' for k in range(5)])
    batches = [c.batch for c in order]; bnames = sorted(set(batches))
    edges = [i for i in range(1, len(batches)) if batches[i] != batches[i - 1]]
    for e in edges:
        ax.axvline(e - 0.5, color='white', lw=2.5)
    centers = [np.mean([i for i, b in enumerate(batches) if b == n]) for n in bnames]
    ax.set_xticks(centers); ax.set_xticklabels([n.replace('Sim_satellite', 'vệ tinh') for n in bnames])
    ax.vlines(np.arange(0.5, len(ids) - 1, 1), -0.5, 4.5, color='white', lw=0.6)          # ranh giới từng cell
    for k in range(6):
        ax.axhline(k - 0.5, color='white', lw=1.5)
    ax.grid(False); ax.set_xlabel('55 cell XJTU, xếp theo nhóm chế độ thí nghiệm')
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(4)], labels=roles,
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, f'{OUT}/fig_thesis_split')

    # 6.2 ma trận sử dụng dữ liệu: trong cùng bộ | chuyển miền
    parts_in = ['train có nhãn', 'train không nhãn', 'validation', 'test']
    parts_tr = ['nguồn: train (đều có nhãn)', 'nguồn: validation, test', 'đích: k cell có nhãn',
                'đích: train không nhãn', 'đích: validation', 'đích: test']
    uses = ['chuẩn hoá', '$\\mathcal{L}_{\\mathrm{data}}$', 'loss vật lý', 'dừng sớm', 'đánh giá']
    U_in = np.array([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    U_tr = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1]])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[4, 5]))
    for ax, U, parts, title in [(axes[0], U_in, parts_in, 'trong cùng bộ'), (axes[1], U_tr, parts_tr, 'chuyển miền')]:
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                           color='#1baf7a' if U[i, j] else '#eef1ef', lw=0))
        ax.set_xlim(-0.5, U.shape[1] - 0.5); ax.set_ylim(U.shape[0] - 0.5, -0.5)
        ax.set_xticks(range(len(uses))); ax.set_xticklabels(uses, rotation=35, ha='right')
        ax.set_yticks(range(len(parts))); ax.set_yticklabels(parts)
        ax.tick_params(length=0); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_title(ax, title, pad=4)
    fig.tight_layout()
    save(fig, f'{OUT}/fig_thesis_usage')

    # 6.3 phân tầng: batch × fold của TJU (9 nhóm) và XJTU (6 nhóm), lần lặp 0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, ds in zip(axes, ['XJTU', 'TJU']):
        bs, M = mats[ds]
        from matplotlib.colors import BoundaryNorm
        cm = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.06, 0.92, M.max() + 1)))
        bnorm = BoundaryNorm(np.arange(-0.5, M.max() + 1.5, 1), cm.N)
        im = ax.imshow(M, cmap=cm, norm=bnorm, aspect='auto')
        ax.set_xticks(range(5)); ax.set_xticklabels([f'{k + 1}' for k in range(5)])
        lab = [b.replace('Sim_satellite', 'vệ tinh') for b in bs]
        if ds == 'TJU':
            chem = lambda x: x.replace('Dataset_', '').replace('_battery', '').split('_', 1)[1].replace('_', '+')
            lab = [chem(b.split('|')[0]) + ' · ' + b.split('|')[1] for b in bs]
        ax.set_yticks(range(len(bs))); ax.set_yticklabels(lab, fontsize=6.5)
        ax.set_xlabel('fold'); ax.grid(False); panel_title(ax, ds, CHEM[ds], pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=range(0, M.max() + 1)); cb.set_label('số cell test', fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, f'{OUT}/fig_thesis_strat')

    # 6.4 hội tụ theo số lần lặp
    dsc = {'XJTU': '#2a78d6', 'TJU': '#eb6834', 'MIT': '#1baf7a', 'HUST': '#e87ba4'}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ds in DATASETS:
        g = conv[conv.bo == ds]
        axes[0].fill_between(g.R, g.ti_so_min, g.ti_so_max, color=dsc[ds], alpha=0.15, lw=0)
        axes[0].plot(g.R, g.ti_so_tb, marker='o', ms=3.5, color=dsc[ds], lw=1.6)
        axes[1].plot(g.R, g.rong_KTC, marker='o', ms=3.5, color=dsc[ds], lw=1.6)
    axes[0].axhline(1, color='#9aa5ad', lw=0.8)
    axes[0].set_ylabel('tỉ số MAE PINN-semi / MLP'); axes[1].set_ylabel('độ rộng KTC 95 %')
    for ax in axes:
        ax.set_xlabel('số lần lặp R'); ax.set_xticks(range(1, 6)); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
    axes[1].set_ylim(bottom=0)
    fig.legend(handles=[plt.Line2D([], [], color=dsc[d], marker='o', ms=3.5, lw=1.6, label=d) for d in DATASETS],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, f'{OUT}/fig_thesis_repeats')

    # 6.5 đường cong theo lượng nhãn: (a) các biến thể vật lý, (b) các mô hình học có nhãn
    for name, models in [('physics', ['mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']),
                         ('baselines', ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_semi'])]:
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
        for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
            for m in models:
                g = cur[(cur.bo == ds) & (cur.model == m)].sort_values('frac')
                if g.empty:
                    continue
                x = 100 * g.frac.values
                ax.fill_between(x, 1e3 * g.lo, 1e3 * g.hi, color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(x, 1e3 * g.MAE_cell, color=COLOR[m], lw=1.6, marker=MARK[m], ms=3.2, mec='white', mew=0.5)
            ax.set_xticks([10, 30, 50, 70]); ax.set_xlim(5, 75)
            lo = 1e3 * cur[(cur.bo == ds) & cur.model.isin(models)].lo.min() * 0.85
            hi = 1e3 * cur[(cur.bo == ds) & cur.model.isin(models)].hi.max() * 1.15
            ax.set_yscale('log'); ax.set_ylim(lo, hi)
            tk = [t for t in (2, 3, 5, 10, 15, 20, 30, 50) if lo <= t <= hi]
            ax.set_yticks(tk); ax.set_yticklabels([f'{t:g}' for t in tk])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
            panel_title(ax, ds, CHEM[ds], pad=4)
            if j == 0:
                ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
        axes[1].set_xlabel('cell có nhãn (% tổng số cell)', x=1.05)
        fig.legend(handles=[plt.Line2D([], [], color=COLOR[m], marker=MARK[m], ms=3.5, lw=1.6, label=LABEL[m])
                            for m in models], loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        save(fig, f'{OUT}/fig_thesis_curve_{name}')

    # 6.6 tỉ số PINN-semi / đối chứng theo lượng nhãn
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
    for j, (ax, ds) in enumerate(zip(axes, DATASETS)):
        for m in ['mlp', 'gbdt', 'pinn_sup']:
            g = cmp_[(cmp_.bo == ds) & (cmp_.doi_chung == m)].sort_values('frac')
            if g.empty:
                continue
            x = 100 * g.frac.values + {'mlp': -1.5, 'gbdt': 0, 'pinn_sup': 1.5}[m]
            ax.vlines(x, g.lo, g.hi, color=COLOR[m], lw=1.2)
            ax.plot(x, g.ti_so, color=COLOR[m], marker=MARK[m], ms=3.5, lw=1.2)
        ax.axhline(1, color='#9aa5ad', lw=0.8)
        ax.set_xticks([10, 30, 50, 70]); ax.set_xlim(3, 77); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, ds, CHEM[ds], pad=4); ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.set_ylabel('tỉ số MAE (PINN-semi / đối chứng)')
    axes[1].set_xlabel('cell có nhãn (% tổng số cell)', x=1.05)
    fig.legend(handles=[plt.Line2D([], [], color=COLOR[m], marker=MARK[m], ms=3.5, lw=1.2, label=f'so với {LABEL[m]}')
                        for m in ['mlp', 'gbdt', 'pinn_sup']], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, f'{OUT}/fig_thesis_ratio')

    # 6.6b mô phỏng chia theo chu kỳ
    if rs is not None:
        dsc = {'XJTU': '#2a78d6', 'TJU': '#eb6834', 'MIT': '#1baf7a', 'HUST': '#e87ba4'}
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
        g = rs[rs.bo == 'mo_phong']
        qq = np.linspace(0.05, 0.95, 100)
        axes[0].plot(qq, 1 / (1 - (1 - qq) ** 2), color='#4a5563', lw=1.4, label='công thức')
        axes[0].plot(g.q, g.E_d_mc, 'o', color='#eb6834', ms=4.5, label='Monte Carlo')
        axes[0].set_xlabel('xác suất một chu kỳ thuộc train $q$'); axes[0].set_ylabel('$\\mathbb{E}[d]$ (chu kỳ)')
        axes[0].legend(fontsize=7, frameon=False)
        for ds in DATASETS:
            h = rs[rs.bo == ds].sort_values('q')
            axes[1].plot(h.q, 1e3 * h.can, color=dsc[ds], lw=1.3, ls='--')
            axes[1].plot(h.q, 1e3 * h.mae_chep, 'o', color=dsc[ds], ms=4, label=ds)
        log_mae_axis(axes[1], ticks=[0.2, 0.5, 1, 2, 5, 10]); axes[1].set_xlabel('xác suất một chu kỳ thuộc train $q$')
        axes[1].set_ylabel('MAE chép nhãn (×10$^{-3}$)')
        axes[1].plot([], [], color='#4a5563', ls='--', lw=1.3, label='cận')
        axes[1].legend(fontsize=6.5, frameon=False, ncol=1, loc='upper right')
        for ax in axes:
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        fig.tight_layout()
        save(fig, f'{OUT}/fig_thesis_rowsim')

    # 6.7 rò rỉ
    if lk is not None and not lk.empty:
        sets = [('chuan', 'chia theo cell (chuẩn)', '#1baf7a'), ('normtest', 'chuẩn hoá gồm cả test', '#9aa5ad'),
                ('life3s', 'lọc dùng cả vòng đời', '#eda100'), ('minmax', 'min–max theo cell', '#e87ba4'),
                ('row', 'chia theo chu kỳ', '#eb6834'), ('noi_suy_hang', 'chia theo chu kỳ, chép nhãn', '#7b5cd6')]
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        w = 0.13
        for i, (key, lab, col) in enumerate(sets):
            if key not in lk.columns:
                continue
            v = [1e3 * lk.set_index('bo').loc[ds, key] for ds in DATASETS]
            ax.bar(np.arange(4) + (i - 2.5) * w, v, width=w * 0.9, color=col, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels(DATASETS); ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=6.5, handlelength=1.0)
        fig.tight_layout()
        save(fig, f'{OUT}/fig_thesis_leakage')

    # 6.8 ma trận chuyển miền
    if tr is not None and not tr.empty:
        tmods = [m for m in ['mlp', 'gbdt', 'gpr', 'pinn_semi'] if m in set(tr.model)]
        fig, axes = plt.subplots(2, len(tmods), figsize=(2.1 * len(tmods) + 0.9, 4.9), squeeze=False)
        vmin, vmax = tr['mean'].min(), tr['mean'].max()
        for i, k in enumerate([0, 3]):
            for j, m in enumerate(tmods):
                ax = axes[i, j]
                M = np.full((4, 4), np.nan)
                g = tr[(tr.k == k) & (tr.model == m)]
                for _, r in g.iterrows():
                    M[DATASETS.index(r.nguon), DATASETS.index(r.dich)] = r['mean']
                cmb = SEQ_CMAP.copy(); cmb.set_bad('white')
                im = ax.imshow(np.ma.masked_invalid(M), cmap=cmb, norm=LogNorm(vmin=vmin / 2, vmax=vmax))
                for q in range(4):                                     # ô chéo: không chuyển miền
                    ax.add_patch(plt.Rectangle((q - 0.5, q - 0.5), 1, 1, fill=False, hatch='////', ec='#d5dbe1', lw=0))
                ax.set_xticks(range(4)); ax.set_xticklabels(DATASETS); ax.set_yticks(range(4)); ax.set_yticklabels(DATASETS)
                ax.grid(False); ax.tick_params(length=0)
                panel_title(ax, LABEL[m], f'k = {k}', pad=4)
                if j == 0:
                    ax.set_ylabel('nguồn')
                if i == 1:
                    ax.set_xlabel('đích')
        cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03); cb.set_label('MAE theo cell trên test đích (thang log)')
        save(fig, f'{OUT}/fig_thesis_transfer')


def to_md(df, path, fmt=None):
    fmt = fmt or {}
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(fmt.get(c, '{:.4g}').format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for c, v in r.items()) + ' |')
    open(path, 'w').write('\n'.join(lines) + '\n')


def main():
    df = percell(refresh=True)
    print('số lượt theo mô hình × mức nhãn:')
    print(df.groupby(['model', 'frac']).apply(lambda g: g.groupby(['dataset', 'repeat', 'fold']).ngroups,
                                              include_groups=False).unstack().fillna(0).astype(int))
    sd = split_design(); sd.to_csv(f'{OUT}/thesis_split_design.csv', index=False)
    st, mats = stratification(); st.to_csv(f'{OUT}/thesis_strat.csv', index=False)
    conv = repeat_convergence(df); conv.to_csv(f'{OUT}/thesis_repeats.csv', index=False)
    cur = curve(df); cur.to_csv(f'{OUT}/thesis_curve.csv', index=False)
    cmp_ = compare_all(df); cmp_.to_csv(f'{OUT}/thesis_compare.csv', index=False)
    eq = equivalence(cur); eq.to_csv(f'{OUT}/thesis_equivalence.csv', index=False)
    lkd, lk = leakage(); lkd.to_csv(f'{OUT}/thesis_leakage_raw.csv', index=False); lk.to_csv(f'{OUT}/thesis_leakage.csv', index=False)
    tr = transfer(); tr.to_csv(f'{OUT}/thesis_transfer.csv', index=False)
    rb = row_bound(); rb.to_csv(f'{OUT}/thesis_rowbound.csv', index=False); print(rb.to_string(index=False))
    rs = row_sim(); rs.to_csv(f'{OUT}/thesis_rowsim.csv', index=False); print(rs.round(5).to_string(index=False))
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 300)
    print(sd.to_string(index=False)); print(st.to_string(index=False)); print(conv.round(4).to_string(index=False))
    print(cur.pivot_table(index=['bo', 'model'], columns='frac', values='MAE_cell').round(5).to_string())
    print(cmp_.round(4).to_string(index=False)); print(eq.to_string(index=False)); print(lk.round(4).to_string(index=False))
    print(tr.round(4).to_string(index=False))
    figures(df, cur, cmp_, conv, lk, mats, tr, rs)
    print('xong')


if __name__ == '__main__':
    main()
