"""
Phần vật lý tác động vào mô hình ở đâu: đo trực tiếp trên mô hình đã huấn luyện.

    python analysis/analyze_physics.py [--workers 4]    # -> results/physics_*.csv, results/fig_physics_*.png|pdf

Huấn luyện lại đúng các lượt của E13/E17 ở 30 % nhãn, lần lặp 0, cả 5 fold (huấn luyện tất định nên mô hình
giống từng bit lượt đã công bố — script kiểm lại MAE từng lượt), rồi trên các cell TEST đo:

  ∂F/∂t        đạo hàm riêng của SOH dự đoán theo thời gian khi giữ nguyên đặc trưng sạc (vi phân tự động) —
               đúng đại lượng mà L_ode ràng buộc; vật lý đòi ∂F/∂t ≤ 0
  r            tốc độ suy giảm học được của động học xám (PINN-sup, PINN-semi, PINN-semi Euler)
  phần dư ODE  |∂F/∂t + r|, so với |∂F/∂t|
  tốc độ thật  −độ dốc của SOH thực đo theo chu kỳ (trên 1000 chu kỳ), từng cell và từng khoảng SOH
  tách biến thiên   tổng giảm SOH dự đoán trên vòng đời cell = phần do thời gian Σ[F(x_i, t_{i+1}) − F(x_i, t_i)]
               + phần do đặc trưng Σ[F(x_{i+1}, t_{i+1}) − F(x_i, t_{i+1})]  (cộng lại đúng bằng tổng)
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import argparse, json, os, pickle
import numpy as np
import pandas as pd

OUT = 'results'
CACHE = os.path.join('.cache', 'physics_v1')
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
VARIANTS = [   # (nhánh, mô hình, thư mục kết quả gốc, khoá cấu hình vật lý)
    ('A', 'mlp', 'runs/E13', 'none'), ('B', 'pinn_sup', 'runs/E13', 'phys'), ('C', 'pinn_semi', 'runs/E13', 'phys'),
    ('D', 'pinn_semi', 'runs/E13', 'mono'), ('E', 'pinn_semi', 'runs/E13', 'euler'), ('H', 'pinn_bb', 'runs/E17', 'phys')]
NAME = {'A': 'MLP', 'B': 'PINN-sup', 'C': 'PINN-semi', 'D': 'chỉ đơn điệu', 'E': 'PINN-semi (Euler)', 'H': 'PINN hộp đen'}
SOH_BINS = np.round(np.arange(0.70, 1.151, 0.05), 2)


def _phys(key):
    from pinnsoh.experiments import E13_PHYS
    return {'none': {}, 'phys': E13_PHYS, 'mono': dict(E13_PHYS, alpha_ode=0.0, gamma_range=0.0),
            'euler': dict(beta_mono=0.5, residual='euler', alpha_ode=2.0)}[key]


def _slope(c, v):
    """−độ dốc của v theo chu kỳ, đơn vị trên 1000 chu kỳ (cùng đơn vị với r)."""
    return -np.polyfit(c / 1000.0, v, 1)[0] if len(c) >= 5 and np.ptp(c) > 0 else np.nan


def work(job):
    tag, model, out_dir, key, ds, fold = job
    path = os.path.join(CACHE, f'{tag}_{ds}_{fold}.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    import torch
    from torch.autograd import grad
    from pinnsoh.train import Config, Trainer, run_name
    from pinnsoh.experiments import E13_NET
    torch.set_num_threads(1)
    kw = dict(dataset=ds, label_frac=0.3, model=model, seed=fold, split='cv', fold=fold, repeat=0, n_folds=5, tag=tag,
              out_dir=out_dir, **E13_NET, **_phys(key))
    cfg = Config(**kw).v2(); cfg.save_predictions = False
    ref_path = os.path.join(out_dir, run_name(cfg) + '.json')
    ref = json.load(open(ref_path))['test']['MAE_cell'] if os.path.exists(ref_path) else None
    tr = Trainer(cfg).prepare()
    res = tr.fit(verbose=False)
    m = tr.model.eval()
    S = tr.test
    X, T, invT = torch.from_numpy(S.X), torch.from_numpy(S.T).clone().requires_grad_(True), torch.from_numpy(S.invT)
    u = m.u(X, T)
    dudt = grad(u.sum(), T)[0].detach().numpy().ravel()
    u = u.detach()
    with torch.no_grad():
        r = m.dyn(X, u, invT).numpy().ravel() if (m.dyn_kind == 'greybox' and key != 'mono') else None
        u = u.numpy().ravel()
    y, cell, cyc = S.Y.ravel().astype(float), S.cell, S.cyc
    cells, bins = [], []
    for k in np.unique(cell):
        idx = np.nonzero(cell == k)[0]
        c, yy, uu, dd = cyc[idx].astype(float), y[idx], u[idx].astype(float), dudt[idx].astype(float)
        with torch.no_grad():                                  # tách biến thiên dọc quỹ đạo: thời gian | đặc trưng
            xa, ta, tb = X[idx[:-1]], T[idx[:-1]].detach(), T[idx[1:]].detach()
            d_t = float((m.u(xa, tb) - m.u(xa, ta)).sum())
        d_tot = float(uu[-1] - uu[0])
        rr = r[idx].astype(float) if r is not None else None
        cells.append(dict(cell=k, n=len(idx), rate_true=_slope(c, yy), rate_pred=_slope(c, uu), dudt=float(dd.mean()),
                          dudt_pos=float((dd > 0).mean()), r=float(rr.mean()) if rr is not None else np.nan,
                          resid=float(np.abs(dd + rr).mean()) if rr is not None else np.nan,
                          absdudt=float(np.abs(dd).mean()), d_tot=d_tot, d_t=d_t))
        b = np.digitize(yy, SOH_BINS)
        for v in np.unique(b):
            s = b == v
            if s.sum() >= 15 and 0 < v < len(SOH_BINS):
                bins.append(dict(bin=float(SOH_BINS[v - 1] + 0.025), n=int(s.sum()), rate_true=_slope(c[s], yy[s]),
                                 rate_pred=_slope(c[s], uu[s]), dudt=float(dd[s].mean()),
                                 r=float(rr[s].mean()) if rr is not None else np.nan))
    out = dict(tag=tag, ds=ds, fold=fold, same=res['test']['MAE_cell'] == ref, mae=res['test']['MAE_cell'], ref=ref,
               cells=cells, bins=bins, lam=float(m.dyn.lam) if r is not None else np.nan,
               Ea=float(m.dyn.Ea) / 1e3 if r is not None else np.nan)
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(out, open(path, 'wb'))
    return out


def collect(workers):
    jobs = [(t, m, o, k, ds, f) for t, m, o, k in VARIANTS for ds in DATASETS for f in range(5)]
    jobs.sort(key=lambda j: {'mlp': 0}.get(j[1], 1), reverse=True)
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        os.environ['OMP_NUM_THREADS'] = '1'
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
            res = list(ex.map(work, jobs))
    else:
        res = [work(j) for j in jobs]
    have = [r for r in res if r['ref'] is not None]
    bad = [(r['tag'], r['ds'], r['fold']) for r in have if not r['same']]
    print(f'{len(res)} lượt huấn luyện lại; so được với lượt công bố {len(have)}, khớp từng bit '
          f'{len(have) - len(bad)}/{len(have)}', bad or '')
    cells = pd.DataFrame([dict(c, tag=r['tag'], ds=r['ds'], fold=r['fold']) for r in res for c in r['cells']])
    bins = pd.DataFrame([dict(b, tag=r['tag'], ds=r['ds']) for r in res for b in r['bins']])
    params = pd.DataFrame([dict(tag=r['tag'], ds=r['ds'], fold=r['fold'], lam=r['lam'], Ea=r['Ea']) for r in res])
    pd.DataFrame([dict(n=len(res), n_ref=len(have), n_same=len(have) - len(bad))]).to_csv(f'{OUT}/physics_check.csv',
                                                                                         index=False)
    return cells, bins, params


def tables(cells, bins):
    from scipy.stats import spearmanr
    rows = []
    for ds in DATASETS:
        for tag in [v[0] for v in VARIANTS]:
            g = cells[(cells.ds == ds) & (cells.tag == tag)]
            w = g.n / g.n.sum()
            share = (g.d_t / g.d_tot).where(g.d_tot.abs() > 1e-3)
            row = dict(bo=ds, nhanh=tag, mo_hinh=NAME[tag], n_cell=len(g),
                       dudt_duong_pct=100 * float((g.dudt_pos * w).sum()),
                       toc_do_rieng=float(-(g.dudt * w).sum()),
                       toc_do_r=float((g.r * w).sum()) if g.r.notna().all() else np.nan,
                       phan_du_tuong_doi=float((g.resid * w).sum() / (g.absdudt * w).sum()) if g.resid.notna().all() else np.nan,
                       phan_thoi_gian_trung_vi=float(share.median()),
                       rho_tong=spearmanr(g.rate_true, g.rate_pred).statistic,
                       rho_rieng=spearmanr(g.rate_true, -g.dudt).statistic,
                       rho_r=spearmanr(g.rate_true, g.r).statistic if g.r.notna().all() else np.nan,
                       toc_do_that=float((g.rate_true * w).sum()))
            rows.append(row)
    t = pd.DataFrame(rows)
    b = bins.groupby(['ds', 'tag', 'bin']).apply(
        lambda g: pd.Series(dict(n=g.n.sum(), rate_true=np.average(g.rate_true, weights=g.n),
                                 rate_pred=np.average(g.rate_pred, weights=g.n), dudt=np.average(g.dudt, weights=g.n),
                                 r=np.average(g.r, weights=g.n) if g.r.notna().all() else np.nan)),
        include_groups=False).reset_index()
    return t, b


def figures(cells, bins):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme()
    CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}
    GREEN, BLUE, DARK, ORANGE = '#1baf7a', '#2a78d6', '#1b2330', '#eb6834'

    # tốc độ suy giảm theo cell: thực đo (trục ngang) so với mô hình (trục dọc)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    for ax, ds in zip(axes, DATASETS):
        series = [('A', 'rate_pred', BLUE, 's', 'MLP: độ dốc dự đoán'),
                  ('C', 'rate_pred', GREEN, 'o', 'PINN-semi: độ dốc dự đoán'),
                  ('C', 'r', ORANGE, '^', 'PINN-semi: tốc độ r học được'),
                  ('E', 'r', DARK, 'v', 'PINN-semi Euler: tốc độ r học được')]
        allv = []
        for tag, col, color, mk, lab in series:
            g = cells[(cells.ds == ds) & (cells.tag == tag)]
            ax.scatter(g.rate_true, g[col], s=7, color=color, marker=mk, alpha=0.75, lw=0, label=lab)
            allv += list(g.rate_true) + list(g[col])
        lo, hi = np.nanpercentile(allv, 1), np.nanpercentile(allv, 99)
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#9aa5ad', lw=0.8)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.grid(False); panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('tốc độ của mô hình')
    fig.supxlabel('tốc độ suy giảm thực đo của cell (SOH / 1000 chu kỳ)', fontsize=8.5, y=0.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.6, markerscale=1.6)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save(fig, f'{OUT}/fig_physics_rate_cell')

    # tốc độ suy giảm theo SOH (hiện tượng đầu gối): thực đo, dự đoán và r
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    for ax, ds in zip(axes, DATASETS):
        for tag, col, color, ls, lab in [('A', 'rate_true', DARK, '-', 'thực đo'), ('A', 'rate_pred', BLUE, '-', 'MLP'),
                                         ('C', 'rate_pred', GREEN, '-', 'PINN-semi'),
                                         ('C', 'r', ORANGE, '--', 'PINN-semi: r'), ('E', 'r', '#7b5cd6', '--', 'Euler: r')]:
            g = bins[(bins.ds == ds) & (bins.tag == tag) & (bins.n >= 200)].sort_values('bin')
            ax.plot(g.bin, g[col], color=color, ls=ls, lw=1.4, marker='o', ms=2.6, label=lab)
        ax.invert_xaxis(); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.axhline(0, color='#9aa5ad', lw=0.6)
        panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('SOH / 1000 chu kỳ')
    fig.supxlabel('SOH thực đo', fontsize=8.5, y=0.085)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    save(fig, f'{OUT}/fig_physics_rate_soh')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    cells, bins, params = collect(a.workers)
    t, b = tables(cells, bins)
    t.to_csv(f'{OUT}/physics_summary.csv', index=False)
    b.to_csv(f'{OUT}/physics_rate_soh.csv', index=False)
    cells.to_csv(f'{OUT}/physics_cells.csv', index=False)
    params.to_csv(f'{OUT}/physics_params.csv', index=False)
    pd.set_option('display.width', 240)
    print(t.round(3).to_string(index=False))
    figures(cells, b)
    print('xong')


if __name__ == '__main__':
    main()
