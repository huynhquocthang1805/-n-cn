"""
Vẽ lại các hình kết quả của báo cáo cho dễ đọc khi in: đúng bề rộng vùng chữ (16 cm), chữ trục 8,5 pt, bốn bộ dữ liệu
xếp lưới 2 × 2 thay vì một hàng 4 panel. Chỉ đọc các bảng đã lưu trong results/, không tính lại thống kê nào; chạy sau
các script phân tích (run.py analyze gọi nó ở cuối).

    python analysis/figures_report.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import os
import numpy as np
import pandas as pd

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
TW = 6.3                                      # bề rộng vùng chữ của báo cáo, inch


def setup():
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme
    apply_theme(base=9)
    plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
                         'axes.titlesize': 9.5, 'axes.labelsize': 9})
    return plt


def _grid(plt, h):
    fig, axes = plt.subplots(2, 2, figsize=(TW, h))
    return fig, axes.ravel()


def _log_ticks(ax, lo, hi, cands=(1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)):
    import matplotlib.pyplot as plt
    ax.set_yscale('log'); ax.set_ylim(lo, hi)
    tk = [t for t in cands if lo <= t <= hi]
    if len(tk) > 6:
        tk = tk[::2]
    ax.set_yticks(tk); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(plt.NullLocator())


def _finish(fig, axes, xlabel, ylabel, legend_rows, ncol):
    from pinnsoh.plotstyle import GRID
    for i, ax in enumerate(axes):
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        if i % 2 == 0 and ylabel:
            ax.set_ylabel(ylabel)
        if i >= len(axes) - 2 and xlabel:
            ax.set_xlabel(xlabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.045 * legend_rows + 0.01, 1, 1], h_pad=1.2, w_pad=1.0)


# ─────────────────────────────────────────────────────────── MAE theo lượng nhãn: MLP và các biến thể có vật lý
def curve_physics():
    plt = setup()
    from analyze_thesis import COLOR, MARK, LABEL
    from pinnsoh.plotstyle import panel_title, save, CHEM
    cur = pd.read_csv(f'{OUT}/thesis_curve.csv')
    models = ['mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']
    fig, axes = _grid(plt, 4.3)
    for ax, ds in zip(axes, DS):
        for m in models:
            g = cur[(cur.bo == ds) & (cur.model == m)].sort_values('frac')
            x = 100 * g.frac.values
            ax.fill_between(x, 1e3 * g.lo, 1e3 * g.hi, color=COLOR[m], alpha=0.12, lw=0)
            ax.plot(x, 1e3 * g.MAE_cell, color=COLOR[m], lw=1.8, marker=MARK[m], ms=4, mec='white', mew=0.5,
                    label=LABEL[m])
        sub = cur[(cur.bo == ds) & cur.model.isin(models)]
        _log_ticks(ax, 1e3 * sub.lo.min() * 0.85, 1e3 * sub.hi.max() * 1.15)
        ax.set_xticks([10, 30, 50, 70]); ax.set_xlim(5, 75)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    _finish(fig, axes, '% số cell có nhãn', 'MAE theo cell (×10$^{-3}$)', 1, 4)
    save(fig, f'{OUT}/fig_thesis_curve_physics')


# ─────────────────────────────────────────────────────────── kiểm toán rò rỉ
def leakage():
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    L = pd.read_csv(f'{OUT}/thesis_leakage.csv').set_index('bo')
    sets = [('chuan', 'chia theo cell (đúng)', '#1baf7a'), ('normtest', 'chuẩn hoá gồm cả test', '#9aa5ad'),
            ('life3s', 'lọc dùng cả vòng đời', '#eda100'), ('minmax', 'min–max theo cell', '#e87ba4'),
            ('row', 'chia theo chu kỳ', '#eb6834'), ('noi_suy_hang', 'chia theo chu kỳ, chép nhãn', '#7b5cd6')]
    sets = [x for x in sets if x[0] in L.columns]
    fig, axes = _grid(plt, 3.9)
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        ref = 1e3 * L.loc[ds, 'chuan']
        ax.axvline(ref, color='#1b2330', lw=0.9, ls=(0, (2.5, 1.5)), zorder=1)
        for j, (key, lab, col) in enumerate(sets):
            v = 1e3 * L.loc[ds, key]
            ax.plot([ref, v], [j, j], color=col, lw=1.2, alpha=0.55, zorder=2)
            ax.plot(v, j, 'o', ms=6, color=col, mec='white', mew=0.6, zorder=3)
        ax.set_xlim(0, 1.12 * max(1e3 * L.loc[ds, k] for k, _, _ in sets))
        ax.locator_params(axis='x', nbins=4)
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([lab for _, lab, _ in sets] if i % 2 == 0 else [])
        ax.tick_params(axis='y', length=0); ax.invert_yaxis()
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
        if i >= 2:
            ax.set_xlabel('MAE theo cell (×10$^{-3}$)')
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, f'{OUT}/fig_thesis_leakage')


# ─────────────────────────────────────────────────────────── PINN-lai qua từng bước
def e20_steps():
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM
    s = pd.read_csv(f'{OUT}/e20_summary.csv')
    steps = ['pinn', 'pinn3', 'lai_tho', 'pinn_lai']
    fracs = sorted(s.frac.unique())
    fig, axes = _grid(plt, 4.4)
    for ax, ds in zip(axes, DS):
        for j, f in enumerate(fracs):
            base = s[(s.dataset == ds) & (s.frac == f) & (s.model == 'pinn')].MAE_cell.iloc[0]
            v = [s[(s.dataset == ds) & (s.frac == f) & (s.model == m)].MAE_cell.iloc[0] / base for m in steps]
            ax.plot(range(4), v, color=plt.cm.Greens(0.4 + 0.18 * j), lw=1.7, marker='o', ms=4, mec='white', mew=0.5,
                    label=f'{int(round(100 * f))} % nhãn')
            gb = s[(s.dataset == ds) & (s.frac == f) & (s.model == 'gbdt')].MAE_cell.iloc[0] / base
            ax.plot([4.2 + 0.12 * (j - 1.5)], [gb], marker='D', ms=4, color=plt.cm.Purples(0.4 + 0.18 * j), ls='none',
                    mec='white', mew=0.5)
        ax.axhline(1, color='#9aa5ad', lw=0.8, ls=(0, (2, 2)))
        ax.axvline(3.6, color='#e3e8e6', lw=0.9)
        ax.set_xticks([0, 1, 2, 3, 4.2]); ax.set_xticklabels(['PINN', '×3', '+GBDT', '+lọc', 'GBDT'])
        ax.set_xlim(-0.3, 4.6)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    _finish(fig, axes, None, 'MAE / MAE của PINN-semi', 1, 4)
    save(fig, f'{OUT}/fig_e20_steps')


# ─────────────────────────────────────────────────────────── so với PINN4SOH
def e27_compare():
    plt = setup()
    from pinnsoh.paper import PAPER_MAE
    from pinnsoh.plotstyle import panel_title, save, GRID
    S = pd.read_csv(f'{OUT}/e27_summary.csv')
    get = lambda ds, mode, prep, model: (lambda r: float(r.mae.iloc[0]) if len(r) else np.nan)(
        S[(S.dataset == ds) & (S['mode'] == mode) & (S.prep == prep) & (S.model == model)])
    fig = plt.figure(figsize=(TW, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    bars = [('PINN4SOH (số công bố)', None, '#9aa5ad'),
            ('PINN hộp đen, tiền xử lý của bài', ('paper', 'pinn_bb'), '#c9b27c'),
            ('PINN-lai, tiền xử lý của bài', ('paper', 'pinn_lai'), '#86b6ef'),
            ('PINN hộp đen, không rò rỉ', ('strict', 'pinn_bb'), '#eda100'),
            ('PINN-lai, không rò rỉ', ('strict', 'pinn_lai'), '#0b3d73')]
    wbar = 0.16
    for i, (lab, sel, col) in enumerate(bars):
        v = [PAPER_MAE[(ds, 'full')]['PINN4SOH'] if sel is None else get(ds, 'full', *sel) for ds in DS]
        ax.bar(np.arange(4) + (i - 2) * wbar, [1e3 * x for x in v], width=wbar, color=col, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(DS)
    ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
    panel_title(ax, 'Chế độ đầy đủ', pad=4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    for i, ds in enumerate(('XJTU', 'HUST')):
        ax = fig.add_subplot(gs[1, i])
        n = [1, 2, 3, 4]
        lines = [('PINN4SOH (số công bố)', lambda k: PAPER_MAE[(ds, f'n{k}')]['PINN4SOH'], '#9aa5ad', '--'),
                 ('PINN-lai, tiền xử lý của bài', lambda k: get(ds, f'n{k}', 'paper', 'pinn_lai'), '#86b6ef', '-'),
                 ('PINN-lai, không rò rỉ', lambda k: get(ds, f'n{k}', 'strict', 'pinn_lai'), '#0b3d73', '-'),
                 ('GBDT, không rò rỉ', lambda k: get(ds, f'n{k}', 'strict', 'gbdt'), '#1baf7a', '-')]
        for lab, fn, col, ls in lines:
            ax.plot(n, [1e3 * fn(k) for k in n], color=col, ls=ls, marker='o', ms=4, lw=1.7, label=lab)
        vals = [1e3 * fn(k) for _, fn, _, _ in lines for k in n]
        _log_ticks(ax, min(vals) / 1.15, max(vals) * 1.15, (3, 5, 8, 10, 15, 20, 30, 50, 80))
        ax.set_xticks(n); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_xlabel('số cell có nhãn' + (' mỗi batch' if ds == 'XJTU' else ''))
        panel_title(ax, f'{ds}, ít mẫu', pad=4)
        if i == 0:
            ax.set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17)
    save(fig, f'{OUT}/fig_e27_compare')


# ─────────────────────────────────────────────────────────── ngân sách đo
def e28_budget():
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM
    S = pd.read_csv(f'{OUT}/e28_summary.csv')
    spec = [('theo_cell', 'pinn_lai', '#9cc3e6', '--', 's', 'PINN-lai, nhãn theo cell'),
            ('theo_cell', 'gbdt', '#f2c6a0', '--', 's', 'GBDT, nhãn theo cell'),
            ('do_thua', 'mlp', '#b9c2c9', '-', 'o', 'MLP, đo thưa'),
            ('do_thua', 'gbdt_interp', '#eb6834', '-', 'o', 'GBDT, đo thưa + nội suy'),
            ('do_thua', 'pinn_lai', '#0b3d73', '-', 'o', 'PINN-lai, đo thưa')]
    fig, axes = _grid(plt, 4.5)
    for ax, ds in zip(axes, DS):
        vals = []
        for kind, m, col, ls, mk, lab in spec:
            g = S[(S.dataset == ds) & (S.kind == kind) & (S.model == m)].sort_values('frac_rows')
            ax.plot(100 * g.frac_rows, 1e3 * g.mae, color=col, ls=ls, marker=mk, ms=4, lw=1.7, label=lab)
            vals += list(1e3 * g.mae)
        ax.set_xscale('log')
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in (0.1, 0.3, 1, 3, 10, 30, 100) if lo <= t <= hi])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        _log_ticks(ax, min(vals) / 1.1, max(vals) * 1.1)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    _finish(fig, axes, '% số chu kỳ của cell train có đo dung lượng', 'MAE theo cell (×10$^{-3}$)', 2, 3)
    save(fig, f'{OUT}/fig_e28_budget')


# ─────────────────────────────────────────────────────────── tổng quan 30 % / 70 % / đo thưa
def overview():
    plt = setup()
    from analyze_overview import MODELS
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    D = pd.read_csv(f'{OUT}/overview.csv')
    col = {'mlp': '#9aa5ad', 'pinn': '#1baf7a', 'gbdt': '#eb6834', 'pinn_lai': '#2a78d6', 'pinn_lai_v2': '#0b3d73'}
    fig, axes = _grid(plt, 4.5)
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        vals = []
        for j, (m, _) in enumerate(MODELS):
            g = D[(D.dataset == ds) & (D.model == m)].set_index('regime').mae * 1e3
            if 'f0.3' in g and 'f0.7' in g:
                ax.plot([g['f0.3'], g['f0.7']], [j, j], color=col[m], lw=1.5, zorder=1)
                ax.scatter([g['f0.3']], [j], s=34, facecolor='white', edgecolor=col[m], lw=1.5, zorder=2)
                ax.scatter([g['f0.7']], [j], s=34, color=col[m], zorder=2)
                vals += [g['f0.3'], g['f0.7']]
            if 'sparse' in g:
                ax.scatter([g['sparse']], [j + 0.32], s=30, marker='D', color=col[m], edgecolor='white', lw=0.5, zorder=3)
                vals.append(g['sparse'])
        ref = D[(D.dataset == ds) & (D.model == 'pinn_lai_v2') & (D.regime == 'f0.3')].mae
        if len(ref):
            ax.axvline(1e3 * float(ref.iloc[0]), color='#c9d3dc', lw=0.9, ls=(0, (2, 2)), zorder=0)
        ax.set_xscale('log')
        lo, hi = min(vals) / 1.08, max(vals) * 1.08
        ax.set_xlim(lo, hi)
        tk = [t for t in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) if lo <= t <= hi]
        ax.set_xticks(tk if len(tk) <= 5 else tk[::2])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(-0.6, len(MODELS) - 0.25)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([l for _, l in MODELS] if i % 2 == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
        if i >= 2:
            ax.set_xlabel('MAE theo cell (×10$^{-3}$, thang log)')
    hd = [plt.Line2D([], [], ls='', marker='o', mfc='white', mec='#4a5563', ms=6, label='30 % số cell có nhãn'),
          plt.Line2D([], [], ls='', marker='o', color='#4a5563', ms=6, label='70 % số cell có nhãn'),
          plt.Line2D([], [], ls='', marker='D', color='#4a5563', ms=5.5, label='đo thưa mọi cell')]
    fig.legend(handles=hd, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.055, 1, 1], h_pad=1.2, w_pad=1.0)
    save(fig, f'{OUT}/fig_overview')


# ─────────────────────────────────────────────────────────── quỹ đạo SOH của một cell test ở 10, 30, 70 % nhãn
def pred_traj():
    """Cùng cách chọn cell như analysis/analyze_predictions.py (MLP 30 % ở trung vị, lần lặp 0), chỉ đổi bố cục."""
    plt = setup()
    import analyze_predictions as AP
    from pinnsoh.plotstyle import panel_title, save, GRID, MODEL_COLOR
    df = AP.load(); r0 = df[df.repeat == 0]
    lv = AP.load_levels()
    fig, axes = plt.subplots(4, 3, figsize=(TW, 6.6), sharey='row')
    for i, ds in enumerate(DS):
        cid = AP.pick_cell(r0, ds)
        for j, frac in enumerate((0.1, 0.3, 0.7)):
            ax = axes[i, j]
            h = lv[(lv.dataset == ds) & (lv.frac == frac) & (lv.cell == cid)]
            t = h[h.model == 'mlp'].sort_values('cycle')
            ax.plot(t.cycle, t.y, color=MODEL_COLOR['truth'], lw=2.0, zorder=2)
            for m in ('mlp', 'pinn_semi'):
                q = h[h.model == m].sort_values('cycle')
                ax.plot(q.cycle, q.p, color=MODEL_COLOR[m], lw=1.0, alpha=0.8 if m == 'mlp' else 0.95, zorder=3)
            panel_title(ax, f'{ds}, {int(round(100 * frac))} % nhãn', pad=3)
            lo, hi = t.y.min(), t.y.max(); pad = max(0.03, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
            if j == 0:
                ax.set_ylabel('SOH')
            if i == len(DS) - 1:
                ax.set_xlabel('chu kỳ')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=2, label='SOH đo được'),
                        plt.Line2D([], [], color=MODEL_COLOR['mlp'], lw=1.6, label='MLP'),
                        plt.Line2D([], [], color=MODEL_COLOR['pinn_semi'], lw=1.6, label='PINN-semi')],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=0.9, w_pad=0.8)
    save(fig, f'{OUT}/fig_pred_traj')


# ─────────────────────────────────────────────────────────── dự báo SOH cho một cell test mỗi bộ
def e22_traj():
    """Cùng cách chọn cell như analysis/analyze_e22.py (lần lặp 0, gốc giữa, sai số kết hợp ở trung vị)."""
    plt = setup()
    import analyze_e22 as A22
    from pinnsoh.forecast import TAU, e20_estimates
    from pinnsoh.data import load_dataset
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID, MODEL_COLOR
    T, _, _, _ = A22.load()
    col = {'gbdt': '#2a78d6', 'pinn': '#1baf7a', 'ket_hop': '#0b4f3c'}
    fig, axes = _grid(plt, 4.6)
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        t = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.oq == 1)]
        e = t.groupby('cid').e_ket_hop.mean().sort_values() if 'e_ket_hop' in t else \
            t.assign(e=np.abs(t.ket_hop - t.y)).groupby('cid').e.mean().sort_values()
        cid = e.index[(len(e) - 1) // 2]
        g = t[t.cid == cid].sort_values('h')
        fold = int(g.fold.iloc[0]); k0 = float(g.k.iloc[0]); nom = float(g.nom.iloc[0])
        cell = [c for c in load_dataset('PINN4SOH/data', ds, clean='causal') if c.cid == cid][0]
        est = e20_estimates(ds, 0, fold, 'test')[cid]
        ax.plot(cell.cycle, cell.soh, color=MODEL_COLOR['truth'], lw=1.4, zorder=2)
        ax.plot(est[0][est[0] <= k0], est[1][est[0] <= k0], color='#7b5cd6', lw=1.6, zorder=3)
        x = k0 + g.h.values
        ax.fill_between(x, g.lo, g.hi, color='#1baf7a', alpha=0.18, lw=0, zorder=1)
        ax.plot(x, g.gbdt, color=col['gbdt'], lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(x, g.pinn, color=col['pinn'], lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(x, g.ket_hop, color=col['ket_hop'], lw=2.0, zorder=4)
        ax.axhline(TAU[ds], color='#c0392b', lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.axvline(k0, color='#9aa5ad', lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
        sec = ax.secondary_yaxis('right', functions=(lambda s_, n=nom: s_ * n, lambda q, n=nom: q / n))
        if i % 2 == 1:
            sec.set_ylabel('dung lượng (Ah)')
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    hd = [plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.6, label='SOH đo được'),
          plt.Line2D([], [], color='#7b5cd6', lw=1.6, label='SOH ước lượng tới gốc'),
          plt.Line2D([], [], color=col['ket_hop'], lw=2, label='dự báo kết hợp'),
          plt.Rectangle((0, 0), 1, 1, color='#1baf7a', alpha=0.25, label='khoảng 90 %'),
          plt.Line2D([], [], color=col['gbdt'], lw=1.2, ls=(0, (3, 1.5)), label='GBDT'),
          plt.Line2D([], [], color=col['pinn'], lw=1.2, ls=(0, (3, 1.5)), label='PINN-dự báo'),
          plt.Line2D([], [], color='#c0392b', lw=1, ls=(0, (4, 2)), label='ngưỡng cuối đời')]
    for j, ax in enumerate(axes):
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        if j % 2 == 0:
            ax.set_ylabel('SOH')
        if j >= 2:
            ax.set_xlabel('chu kỳ')
    fig.legend(handles=hd, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e22_traj')


FIGS = dict(curve_physics=curve_physics, leakage=leakage, e20_steps=e20_steps, e27_compare=e27_compare,
            e28_budget=e28_budget, overview=overview, pred_traj=pred_traj, e22_traj=e22_traj)
NEEDS = dict(curve_physics='thesis_curve', leakage='thesis_leakage', e20_steps='e20_summary', e27_compare='e27_summary',
             e28_budget='e28_summary', overview='overview', pred_traj='pred_metrics', e22_traj='e22_horizon')


def main(names=None):
    for name in names or FIGS:
        if os.path.exists(f'{OUT}/{NEEDS[name]}.csv'):
            FIGS[name]()
            print('đã vẽ', name)
        else:
            print('bỏ qua', name, '(chưa có', NEEDS[name] + '.csv)')


if __name__ == '__main__':
    main(_sys.argv[1:] or None)
