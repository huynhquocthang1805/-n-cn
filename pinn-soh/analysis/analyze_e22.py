"""
E22: dự báo SOH tương lai, dung lượng còn lại (Ah) và RUL — bảng, kiểm định và hình.

    python analysis/analyze_e22.py      # -> results/e22_*.csv, results/fig_e22_*.png|pdf

Đơn vị thống kê là cell: sai số của một cell lấy trung bình qua lần lặp (0–2) và các gốc dự báo, rồi mới lấy
trung bình hay kiểm định qua cell. Kịch bản 'uoc' (lịch sử SOH do PINN-lai ước lượng, không đo dung lượng cell test)
là kịch bản chính; 'do' (lịch sử đo thật) cho biết phần sai số do ước lượng.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.forecast import MODELS, SCEN, H_REL, ROOT, TAU, RUL_ORIGINS, e20_estimates
from pinnsoh.data import load_dataset

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
LABEL = {'giu': 'Giữ nguyên', 'tuyen': 'Tuyến tính', 'vatly': 'Vật lý theo cell', 'gbdt': 'GBDT',
         'pinn': 'PINN-dự báo', 'ket_hop': 'Kết hợp'}
TCOL = ['scen', 'cell', 'oq', 'k', 'h', 'y'] + MODELS + ['lo', 'hi']
RCOL = ['scen', 'cell', 'q', 'k', 'true'] + MODELS + ['lo', 'hi', 'a', 'lam']


def load():
    T, R, P, info = [], [], [], {}
    for f in sorted(glob.glob(f'{ROOT}/fc_*.npz')):
        d = np.load(f)
        inf = json.loads(str(d['info'])); ds, r, k = inf['ds'], inf['repeat'], inf['fold']
        info[(ds, r, k)] = inf
        cells = list(d['cells']); nom = np.asarray(inf['nominal_ah'])
        t = pd.DataFrame(d['traj'].astype(np.float64), columns=TCOL)
        t['ds'], t['r'], t['fold'] = ds, r, k
        t['cid'] = np.asarray(cells)[t.cell.astype(int)]; t['nom'] = nom[t.cell.astype(int)]
        t['L'] = inf['L']
        T.append(t)
        if len(d['rul']):
            u = pd.DataFrame(d['rul'], columns=RCOL); u['ds'], u['r'] = ds, r
            u['cid'] = np.asarray(cells)[u.cell.astype(int)]; u['L'] = inf['L']
            R.append(u)
        p = pd.DataFrame(d['par'], columns=['scen', 'cell', 'oq', 'k', 'a', 'lam']); p['ds'] = ds
        P.append(p)
    T = pd.concat(T, ignore_index=True); R = pd.concat(R, ignore_index=True); P = pd.concat(P, ignore_index=True)
    for X in (T, R, P):
        X['scen'] = np.asarray(SCEN)[X.scen.astype(int)]
    return T, R, P, info


def at_horizons(T):
    """Với mỗi (lượt, cell, gốc) và mỗi tầm h* = h_rel·L: hàng đầu tiên có h >= h*, nếu cell còn đủ dài."""
    rows = []
    for (ds, r, sc, cid, oq), g in T.groupby(['ds', 'r', 'scen', 'cid', 'oq'], sort=False):
        L = g.L.iloc[0]; h = g.h.values
        for hr in H_REL:
            j = np.searchsorted(h, hr * L, side='left')
            if j >= len(h) or h[j] > hr * L * 1.1 + 2:
                continue
            row = g.iloc[j]
            rows.append(dict(ds=ds, r=r, scen=sc, cid=cid, oq=oq, h_rel=hr, y=row.y, nom=row.nom,
                             **{m: row[m] for m in MODELS}, lo=row.lo, hi=row.hi))
    return pd.DataFrame(rows)


def cell_mean(df, keys, cols):
    """Trung bình theo cell (qua lần lặp, gốc), giữ `keys`."""
    return df.groupby(keys + ['cid'])[cols].mean().reset_index()


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1.0, run)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    T, R, P, info = load()
    print(f'{len(info)} lượt, {T.cid.nunique()} cell, {len(T)} điểm dự báo, {len(R)} gốc RUL')
    # ---- 1. sai số SOH tại các tầm dự báo (×1e-3) và dung lượng (mAh)
    H = at_horizons(T)
    for m in MODELS:
        H[f'e_{m}'] = (H[m] - H.y).abs()
    H['mah'] = H.e_ket_hop * H.nom * 1000
    H['cov'] = ((H.y >= H.lo) & (H.y <= H.hi)).astype(float)
    H['wid'] = H.hi - H.lo
    ecols = [f'e_{m}' for m in MODELS] + ['mah', 'cov', 'wid']
    hc = cell_mean(H, ['ds', 'scen', 'h_rel'], ecols)
    hs = hc.groupby(['ds', 'scen', 'h_rel'])[ecols].mean().reset_index()
    hs['n_cell'] = hc.groupby(['ds', 'scen', 'h_rel']).size().values
    hs.to_csv(f'{OUT}/e22_horizon.csv', index=False)
    # ---- 2. sai số trên toàn bộ phần đời còn lại từ mỗi gốc
    for m in MODELS:
        T[f'e_{m}'] = (T[m] - T.y).abs()
    T['cov'] = ((T.y >= T.lo) & (T.y <= T.hi)).astype(float); T['wid'] = T.hi - T.lo
    T['mah'] = T.e_ket_hop * T.nom * 1000
    ecols2 = [f'e_{m}' for m in MODELS] + ['cov', 'wid', 'mah']
    tg = T.groupby(['ds', 'r', 'scen', 'cid', 'oq'])[ecols2].mean().reset_index()
    tc = cell_mean(tg, ['ds', 'scen'], ecols2)
    ts = tc.groupby(['ds', 'scen'])[ecols2].mean().reset_index()
    ts['n_cell'] = tc.groupby(['ds', 'scen']).size().values
    ts.to_csv(f'{OUT}/e22_traj.csv', index=False)
    # kiểm định: kết hợp so với từng mô hình, ghép cặp theo cell, Holm trong mỗi (bộ, kịch bản)
    rows = []
    for (ds, sc), g in tc.groupby(['ds', 'scen']):
        ps, tmp = [], []
        for m in MODELS:
            if m == 'ket_hop':
                continue
            d = g.e_ket_hop.values - g[f'e_{m}'].values
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ratio = float(np.exp(np.mean(np.log(g.e_ket_hop.values / g[f'e_{m}'].values))))
            tmp.append(dict(ds=ds, scen=sc, vs=m, ratio=ratio, better=int((d < 0).sum()), n=len(d), p=p)); ps.append(p)
        for t_, ph in zip(tmp, holm(ps)):
            t_['p_holm'] = ph; rows.append(t_)
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_tests.csv', index=False)
    # ---- 3. RUL (chu kỳ)
    for m in MODELS + ['lo', 'hi']:
        R[f'd_{m}'] = R[m] - R.true
    for m in MODELS:
        R[f'e_{m}'] = R[f'd_{m}'].abs(); R[f'rel_{m}'] = R[f'e_{m}'] / R.true.clip(lower=1)
        R[f'in20_{m}'] = (R[f'rel_{m}'] <= 0.2).astype(float)
    R['cov'] = ((R.true >= R.lo) & (R.true <= R.hi)).astype(float)
    rc_cols = [f'e_{m}' for m in MODELS] + [f'in20_{m}' for m in MODELS] + ['cov']
    rcell = cell_mean(R, ['ds', 'scen'], rc_cols + ['true'])
    rs = rcell.groupby(['ds', 'scen'])[rc_cols + ['true']].mean().reset_index()
    rs['n_cell'] = rcell.groupby(['ds', 'scen']).size().values
    med = R.groupby(['ds', 'scen'])[[f'rel_{m}' for m in MODELS]].median().reset_index()
    rs = rs.merge(med, on=['ds', 'scen'])
    rs['L'] = rs.ds.map({d: float(R[R.ds == d].L.iloc[0]) for d in R.ds.unique()})
    rs.to_csv(f'{OUT}/e22_rul.csv', index=False)
    rq = R.groupby(['ds', 'scen', 'q'])[['e_ket_hop', 'e_gbdt', 'e_pinn', 'true']].mean().reset_index()
    rq.to_csv(f'{OUT}/e22_rul_origin.csv', index=False)
    rows = []
    for (ds, sc), g in rcell.groupby(['ds', 'scen']):
        ps, tmp = [], []
        for m in ['tuyen', 'vatly', 'gbdt', 'pinn']:
            d = g.e_ket_hop.values - g[f'e_{m}'].values
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            tmp.append(dict(ds=ds, scen=sc, vs=m, better=int((d < 0).sum()), n=len(d), p=p)); ps.append(p)
        for t_, ph in zip(tmp, holm(ps)):
            t_['p_holm'] = ph; rows.append(t_)
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_rul_tests.csv', index=False)
    # ---- 4. hệ số gãy lam của PINN-dự báo theo bộ (lịch sử đo)
    pp = P[P.scen == 'do']
    lam = pp.groupby('ds').lam.agg(['median', lambda x: float((x > 0).mean()), 'count']).reset_index()
    lam.columns = ['ds', 'lam_median', 'frac_pos', 'n']
    lam['a_median'] = pp.groupby('ds').a.median().values
    lam.to_csv(f'{OUT}/e22_lambda.csv', index=False)
    # ---- 5. tham số giao thức theo bộ: L, ngưỡng, tỉ lệ cell chạm ngưỡng, dung lượng danh định
    from pinnsoh.forecast import eol_cycle, life_scale, nominal_ah
    rows = []
    for ds in DS:
        cs = load_dataset('PINN4SOH/data', ds, clean='causal')
        L = life_scale(cs); nom = sorted({nominal_ah(c) for c in cs})
        rows.append(dict(ds=ds, n_cell=len(cs), L=L, tau=TAU[ds], reach=float(np.mean([eol_cycle(c, TAU[ds]) is not None
                                                                                      for c in cs])),
                         nom_min=nom[0], nom_max=nom[-1], h1=int(round(H_REL[0] * L)), h2=int(round(H_REL[1] * L)),
                         h3=int(round(H_REL[2] * L))))
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_setting.csv', index=False)
    # ---- 6. thời gian, số mẫu
    meta = pd.DataFrame([dict(ds=k[0], r=k[1], fold=k[2], L=v['L'], tau=v['tau'], time_s=v['time_s'],
                              n_train=v['n_train_samples'], n_trees=v['gbdt']['n_trees']) for k, v in info.items()])
    meta.to_csv(f'{OUT}/e22_meta.csv', index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    show = lambda df, cols, f=1e3: print(df.assign(**{c: (df[c] * f).round(1) for c in cols}).to_string(index=False))
    print('\nSOH tại tầm dự báo (×1e-3):'); show(hs, [f'e_{m}' for m in MODELS] + ['wid'])
    print('\ntoàn phần đời còn lại (×1e-3):'); show(ts, [f'e_{m}' for m in MODELS] + ['wid'])
    print('\nRUL (chu kỳ):'); print(rs.round(2).to_string(index=False))
    print(pd.read_csv(f'{OUT}/e22_tests.csv').round(4).to_string(index=False))
    print(pd.read_csv(f'{OUT}/e22_rul_tests.csv').round(4).to_string(index=False))
    print(lam.round(3).to_string(index=False))
    figures(T, H, R)


# --------------------------------------------------------------------------- hình
def figures(T, H, R):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, CHEM, panel_title, save, MODEL_COLOR, MUTED
    apply_theme()
    col = {'giu': '#b9c1c7', 'tuyen': '#8a96a0', 'vatly': '#eda100', 'gbdt': '#2a78d6', 'pinn': '#1baf7a',
           'ket_hop': '#0b4f3c'}
    # 1. sai số theo tầm dự báo (kịch bản ước lượng)
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45))
    for ax, ds in zip(axes, DS):
        t = T[(T.ds == ds) & (T.scen == 'uoc')].copy()
        t['b'] = np.digitize(t.h / t.L, edges) - 1
        t = t[(t.b >= 0) & (t.b < len(edges) - 1)]
        g = t.groupby(['b', 'cid'])[[f'e_{m}' for m in MODELS]].mean().groupby('b').mean()
        x = (edges[:-1] + edges[1:])[g.index.values] / 2
        for m in MODELS:
            ax.plot(x, 1e3 * g[f'e_{m}'].values, color=col[m], lw=1.9 if m == 'ket_hop' else 1.1,
                    ls='-' if m in ('gbdt', 'pinn', 'ket_hop') else (0, (3, 1.5)), zorder=4 if m == 'ket_hop' else 3)
        ax.set_yscale('log')
        lo_, hi_ = ax.get_ylim()
        ax.set_yticks([v for v in (3, 5, 10, 20, 50, 100, 200) if lo_ <= v <= hi_])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.6)
        panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    hd = [plt.Line2D([], [], color=col[m], lw=1.9 if m == 'ket_hop' else 1.2,
                     ls='-' if m in ('gbdt', 'pinn', 'ket_hop') else (0, (3, 1.5)), label=LABEL[m]) for m in MODELS]
    fig.legend(handles=hd, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.text(0.5, 0.095, 'tầm dự báo h / L (L: trung vị số chu kỳ của bộ)', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e22_horizon')

    # 2. RUL dự báo so với thực, mô hình kết hợp, kịch bản ước lượng
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    qcol = {0.25: '#9cc3e6', 0.5: '#2a78d6', 0.75: '#0b3d73'}
    for ax, ds in zip(axes, DS):
        u = R[(R.ds == ds) & (R.scen == 'uoc')].groupby(['cid', 'q'])[['true', 'ket_hop']].mean().reset_index()
        mx = float(max(u.true.max(), u.ket_hop.max())) * 1.05
        xx = np.array([0, mx])
        ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color='#e7ebe9', lw=0, zorder=0)
        ax.plot(xx, xx, color=MUTED, lw=0.7, zorder=1)
        for q in RUL_ORIGINS:
            v = u[u.q == q]
            ax.scatter(v.true, v.ket_hop, s=7, color=qcol[q], lw=0.2, ec='white', zorder=3)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect('equal')
        ax.locator_params(nbins=3); ax.tick_params(labelsize=6.5); ax.grid(False)
        panel_title(ax, ds, f'τ = {TAU[ds]:.2f}', pad=4)
    axes[0].set_ylabel('RUL dự báo (chu kỳ)')
    hd = [plt.Line2D([], [], color=qcol[q], marker='o', ls='', ms=4, label=f'gốc ở {int(q * 100)} % đời') for q in RUL_ORIGINS]
    hd.append(plt.Rectangle((0, 0), 1, 1, color='#e7ebe9', label='sai lệch ±20 %'))
    fig.legend(handles=hd, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.text(0.5, 0.1, 'RUL thực (chu kỳ)', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e22_rul')

    # 3. quỹ đạo dự báo của một cell mỗi bộ (lần lặp 0, gốc giữa, cell có sai số ở trung vị)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    cells = {}
    for ax, ds in zip(axes, DS):
        t = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.oq == 1)]
        e = t.groupby('cid').e_ket_hop.mean().sort_values()
        cid = e.index[(len(e) - 1) // 2]
        g = t[t.cid == cid].sort_values('h')
        fold = int(g.fold.iloc[0]); k0 = float(g.k.iloc[0]); nom = float(g.nom.iloc[0])
        cell = [c for c in load_dataset('PINN4SOH/data', ds, clean='causal') if c.cid == cid][0]
        est = e20_estimates(ds, 0, fold, 'test')[cid]
        cells[ds] = (cid, k0)
        past = cell.cycle <= k0
        ax.plot(cell.cycle, cell.soh, color=MODEL_COLOR['truth'], lw=1.0, zorder=2)
        ax.plot(est[0][est[0] <= k0], est[1][est[0] <= k0], color='#7b5cd6', lw=1.3, zorder=3)
        x = k0 + g.h.values
        ax.fill_between(x, g.lo, g.hi, color='#1baf7a', alpha=0.18, lw=0, zorder=1)
        ax.plot(x, g.gbdt, color=col['gbdt'], lw=0.9, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(x, g.pinn, color=col['pinn'], lw=0.9, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(x, g.ket_hop, color=col['ket_hop'], lw=1.7, zorder=4)
        ax.axhline(TAU[ds], color='#c0392b', lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.axvline(k0, color='#9aa5ad', lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        sec = ax.secondary_yaxis('right', functions=(lambda s, n=nom: s * n, lambda q, n=nom: q / n))
        sec.tick_params(labelsize=6.2, length=2); sec.locator_params(nbins=4)
        panel_title(ax, ds, CHEM[ds], pad=4)
    axes[0].set_ylabel('SOH')
    fig.text(0.995, 0.56, 'dung lượng (Ah)', rotation=90, ha='right', va='center', fontsize=7.5)
    hd = [plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.2, label='thực đo'),
          plt.Line2D([], [], color='#7b5cd6', lw=1.3, label='lịch sử ước lượng'),
          plt.Line2D([], [], color=col['ket_hop'], lw=1.7, label='dự báo kết hợp'),
          plt.Rectangle((0, 0), 1, 1, color='#1baf7a', alpha=0.25, label='khoảng 90 %'),
          plt.Line2D([], [], color=col['gbdt'], lw=1.0, ls=(0, (3, 1.5)), label='GBDT'),
          plt.Line2D([], [], color=col['pinn'], lw=1.0, ls=(0, (3, 1.5)), label='PINN-dự báo'),
          plt.Line2D([], [], color='#c0392b', lw=0.8, ls=(0, (4, 2)), label='ngưỡng cuối đời')]
    fig.legend(handles=hd, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6.6)
    fig.tight_layout(rect=[0, 0.12, 0.985, 1])
    fig.text(0.5, 0.09, 'chu kỳ', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e22_traj')
    json.dump({k: [v[0], v[1]] for k, v in cells.items()}, open(f'{OUT}/e22_traj_cells.json', 'w'))


if __name__ == '__main__':
    main()
