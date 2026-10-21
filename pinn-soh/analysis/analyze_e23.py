"""
E23: dự báo có xét hồi phục dung lượng — so ghép cặp với mô hình E22 trên cùng fold, cùng gốc.

    python analysis/analyze_e23.py      # -> results/e23_*.csv, results/fig_e23_*.png|pdf

Hai loại gốc: gốc thường (20/40/60 % số chu kỳ của cell, như E22) và gốc ngay sau một lần hồi phục (SOH đo vượt
trung vị 5 chu kỳ trước quá 0,004, sau giai đoạn kích hoạt đầu đời). Đơn vị thống kê là cell.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.recovery import MODELS_R, ROOT
from pinnsoh.forecast import SCEN, H_REL, TAU

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST', 'NASA']
TCOL = ['scen', 'cell', 'otype', 'oq', 'k', 'h', 'y'] + MODELS_R + ['lo', 'hi', 'lo_r', 'hi_r']
RCOL = ['scen', 'cell', 'q', 'k', 'true'] + MODELS_R + ['lo_r', 'hi_r']
DCOL = ['scen', 'cell', 'otype', 'oq', 'k', 'level', 'e', 'P', 'amp', 'r2', 'slope']
PAIRS = [('ket_hop', 'ket_hop_r'), ('gbdt', 'gbdt_r'), ('pinn', 'pinn_r')]


def load(root=ROOT):
    T, R, D, info = [], [], [], {}
    for f in sorted(glob.glob(f'{root}/rc_*.npz')):
        d = np.load(f)
        inf = json.loads(str(d['info'])); ds, r, k = inf['ds'], inf['repeat'], inf['fold']
        info[(ds, r, k)] = inf
        cells = np.asarray(d['cells'])
        t = pd.DataFrame(d['traj'].astype(np.float64), columns=TCOL); t['ds'], t['r'], t['fold'], t['L'] = ds, r, k, inf['L']
        t['cid'] = cells[t.cell.astype(int)]; T.append(t)
        if len(d['rul']):
            u = pd.DataFrame(d['rul'], columns=RCOL); u['ds'], u['r'] = ds, r; u['cid'] = cells[u.cell.astype(int)]
            R.append(u)
        x = pd.DataFrame(d['dec'], columns=DCOL); x['ds'] = ds; x['cid'] = cells[x.cell.astype(int)]; D.append(x)
    T, R, D = (pd.concat(x, ignore_index=True) for x in (T, R, D))
    for X in (T, R, D):
        X['scen'] = np.asarray(SCEN)[X.scen.astype(int)]
    return T, R, D, info


def paired(df, a, b, keys):
    """Tỉ số trung bình nhân b/a theo cell, số cell b tốt hơn, Wilcoxon."""
    rows = []
    for key, g in df.groupby(keys):
        g = g.dropna(subset=[a, b])
        if len(g) < 3:
            continue
        d = g[b].values - g[a].values
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), old=a, new=b, n=len(g),
                         mae_old=g[a].mean(), mae_new=g[b].mean(),
                         ratio=float(np.exp(np.mean(np.log(np.maximum(g[b].values, 1e-9) / np.maximum(g[a].values, 1e-9))))),
                         better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pd.DataFrame(rows)


def cell_errors(T):
    """Sai số theo cell (trung bình qua lần lặp và gốc) trên toàn bộ phần đời còn lại và trong tầm 0,1L."""
    ms = sorted({m for pr in PAIRS for m in pr})
    for m in ms:
        T[f'e_{m}'] = (T[m] - T.y).abs()
    T['short'] = T.h <= 0.1 * T.L
    T['cov_r'] = ((T.y >= T.lo_r) & (T.y <= T.hi_r)).astype(float)
    T['cov'] = ((T.y >= T.lo) & (T.y <= T.hi)).astype(float)
    ecols = [f'e_{m}' for m in ms]
    rows = []
    for part, mask in (('toan_bo', np.ones(len(T), bool)), ('ngan', T.short.values)):
        g = T[mask].groupby(['ds', 'scen', 'otype', 'r', 'cid', 'oq'])[ecols + ['cov', 'cov_r']].mean()
        g = g.groupby(['ds', 'scen', 'otype', 'cid']).mean().reset_index(); g['part'] = part
        rows.append(g)
    C = pd.concat(rows, ignore_index=True)
    tests = pd.concat([paired(C, f'e_{a}', f'e_{b}', ['ds', 'scen', 'otype', 'part']) for a, b in PAIRS], ignore_index=True)
    return C, tests


def main():
    os.makedirs(OUT, exist_ok=True)
    T, R, D, info = load()
    print(f'{len(info)} lượt, {T.cid.nunique()} cell')
    ms = sorted({m for pr in PAIRS for m in pr})
    C, tests = cell_errors(T)
    tests.to_csv(f'{OUT}/e23_tests.csv', index=False)
    # bản đầu của E23 (tau học bằng gradient, e là trung vị 3 hàng cuối), giữ ở runs/E23_v1 để so trong báo cáo
    if glob.glob('runs/E23_v1/rc_*.npz'):
        cell_errors(load('runs/E23_v1')[0])[1].to_csv(f'{OUT}/e23_tests_v1.csv', index=False)
    cov = C[C.part == 'toan_bo'].groupby(['ds', 'scen', 'otype'])[['cov', 'cov_r']].mean().reset_index()
    cov.to_csv(f'{OUT}/e23_cover.csv', index=False)
    # ở ba tầm như E22 (gốc thường)
    hrows = []
    for (ds, r, sc, cid, oq), g in T[T.otype == 0].groupby(['ds', 'r', 'scen', 'cid', 'oq'], sort=False):
        L = g.L.iloc[0]; h = g.h.values
        for hr in H_REL:
            j = np.searchsorted(h, hr * L)
            if j < len(h) and h[j] <= hr * L * 1.1 + 2:
                hrows.append(dict(ds=ds, scen=sc, cid=cid, h_rel=hr, **{f'e_{m}': abs(g[m].iloc[j] - g.y.iloc[j]) for m in ms}))
    H = pd.DataFrame(hrows).groupby(['ds', 'scen', 'h_rel', 'cid']).mean().groupby(['ds', 'scen', 'h_rel']).mean()
    H.reset_index().to_csv(f'{OUT}/e23_horizon.csv', index=False)
    # RUL
    for m in ms:
        R[f'e_{m}'] = (R[m] - R.true).abs(); R[f'in20_{m}'] = (R[f'e_{m}'] <= 0.2 * R.true.clip(lower=1)).astype(float)
    R['cov_r'] = ((R.true >= R.lo_r) & (R.true <= R.hi_r)).astype(float)
    rc = R.groupby(['ds', 'scen', 'cid'])[[f'e_{m}' for m in ms] + [f'in20_{m}' for m in ms] + ['cov_r', 'true']].mean().reset_index()
    rc.groupby(['ds', 'scen']).mean(numeric_only=True).reset_index().to_csv(f'{OUT}/e23_rul.csv', index=False)
    rc['part'] = 'rul'; rc['otype'] = 0
    paired(rc, 'e_ket_hop', 'e_ket_hop_r', ['ds', 'scen']).to_csv(f'{OUT}/e23_rul_tests.csv', index=False)
    # thành phần tách được: tỉ lệ gốc có dao động, biên độ, |e|, (rho, tau) khớp trên cell train
    dd = D[D.scen == 'do'].groupby('ds').agg(per=('P', lambda x: float((x > 0).mean())), amp=('amp', 'mean'),
                                              e_abs=('e', lambda x: float(np.abs(x).mean()))).reset_index()
    dd['tau'] = dd.ds.map(lambda ds: float(np.median([v['new']['tau'] for k, v in info.items() if k[0] == ds])))
    dd['rho'] = dd.ds.map(lambda ds: float(np.median([v['new'].get('rho', 0.0) for k, v in info.items() if k[0] == ds])))
    dd['rho_lo'] = dd.ds.map(lambda ds: float(np.min([v['new'].get('rho', 0.0) for k, v in info.items() if k[0] == ds])))
    dd['rho_hi'] = dd.ds.map(lambda ds: float(np.max([v['new'].get('rho', 0.0) for k, v in info.items() if k[0] == ds])))
    dd['n_ev_train'] = dd.ds.map(lambda ds: float(np.median([v['new'].get('n_events', 0) for k, v in info.items()
                                                              if k[0] == ds])))
    dd['n_ev'] = dd.ds.map(lambda ds: int(T[(T.ds == ds) & (T.otype == 1) & (T.scen == 'do')].groupby(['r', 'cid', 'oq']).ngroups))
    dd.to_csv(f'{OUT}/e23_decomp.csv', index=False)
    per_batch = D[(D.scen == 'do') & (D.ds == 'XJTU')].assign(batch=lambda x: x.cid.str.split('/').str[1].str.split('_').str[0])
    per_batch.groupby('batch').P.apply(lambda x: float((x > 0).mean())).reset_index().to_csv(f'{OUT}/e23_periodic_xjtu.csv', index=False)
    pd.set_option('display.width', 220)
    t = tests[tests.old == 'e_ket_hop'].copy()
    for c in ('mae_old', 'mae_new'):
        t[c] = (1e3 * t[c]).round(2)
    print(t.round(4).to_string(index=False))
    print((1e3 * H[['e_ket_hop', 'e_ket_hop_r', 'e_gbdt', 'e_gbdt_r', 'e_pinn', 'e_pinn_r']]).round(1).to_string())
    print(pd.read_csv(f'{OUT}/e23_rul.csv')[['ds', 'scen', 'e_ket_hop', 'e_ket_hop_r', 'in20_ket_hop', 'in20_ket_hop_r', 'cov_r']].round(2).to_string(index=False))
    print(pd.read_csv(f'{OUT}/e23_rul_tests.csv').round(4).to_string(index=False))
    print(cov.round(3).to_string(index=False)); print(dd.round(4).to_string(index=False))
    figure(T)


def figure(T):
    """Ba ví dụ, lịch sử đo: dao động đều (XJTU R2.5), gốc ngay sau hồi phục (XJTU) và (NASA)."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save, MODEL_COLOR
    from pinnsoh.data import load_dataset
    apply_theme()
    t = T[(T.scen == 'do') & (T.r == 0)]
    picks = []
    a = t[(t.ds == 'XJTU') & (t.otype == 0) & (t.oq == 1) & t.cid.str.contains('R2.5')]
    if len(a):
        picks.append(('XJTU', a.cid.iloc[0], 0, 1, 'dao động đều (R2.5)'))
    for ds, lab in (('XJTU', 'sau hồi phục'), ('NASA', 'sau khoảng nghỉ dài')):
        b = t[(t.ds == ds) & (t.otype == 1)]
        if ds == 'XJTU':
            b = b[~b.cid.str.contains('R2.5')]
        if len(b):
            g = b.groupby(['cid', 'oq']).apply(lambda x: (x.e_ket_hop.mean() if 'e_ket_hop' in x else 0)).sort_values()
            cid, oq = g.index[len(g) // 2]
            picks.append((ds, cid, 1, oq, lab))
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(7.2, 2.4))
    axes = np.atleast_1d(axes)
    cache = {}
    for ax, (ds, cid, ot, oq, lab) in zip(axes, picks):
        if ds not in cache:
            cache[ds] = {c.cid: c for c in load_dataset('PINN4SOH/data', ds, clean='causal')}
        c = cache[ds][cid]
        g = t[(t.cid == cid) & (t.otype == ot) & (t.oq == oq)].sort_values('h')
        k0 = g.k.iloc[0]; span = max(40.0, 0.25 * g.L.iloc[0])
        g = g[g.h <= span]
        m = (c.cycle >= k0 - span) & (c.cycle <= k0 + span)
        ax.plot(c.cycle[m], c.soh[m], color=MODEL_COLOR['truth'], lw=1.0, zorder=2)
        ax.plot(k0 + g.h, g.ket_hop, color='#9aa5ad', lw=1.3, zorder=3)
        ax.fill_between(k0 + g.h, g.lo_r, g.hi_r, color='#1baf7a', alpha=0.15, lw=0)
        ax.plot(k0 + g.h, g.ket_hop_r, color='#0b4f3c', lw=1.5, zorder=4)
        ax.axvline(k0, color='#9aa5ad', lw=0.7, ls=(0, (2, 2)))
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
        panel_title(ax, f'{ds}', lab, pad=3)
    axes[0].set_ylabel('SOH')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.2, label='thực đo'),
                        plt.Line2D([], [], color='#9aa5ad', lw=1.3, label='dự báo E22'),
                        plt.Line2D([], [], color='#0b4f3c', lw=1.5, label='dự báo có xét hồi phục'),
                        plt.Rectangle((0, 0), 1, 1, color='#1baf7a', alpha=0.25, label='khoảng 90 %')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.text(0.5, 0.1, 'chu kỳ', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e23_examples')


if __name__ == '__main__':
    main()
