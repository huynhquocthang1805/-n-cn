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


def _log_ticks(ax, lo, hi, cands=(1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100)):
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
    _finish(fig, axes, '% số chu kỳ của cell huấn luyện có đo dung lượng', 'MAE theo cell (×10$^{-3}$)', 2, 3)
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
    lv = AP.load_levels(lai=True)
    # PINN-lai: hổ phách, đạt ngưỡng phân biệt (ΔE OKLab ×100 >= 9,1 với mù màu, >= 22,9 với mắt thường) so với ba đường
    # còn lại; xanh đậm của PINN-lai ở các hình khác quá gần đường SOH đo xám đậm
    color = dict(MODEL_COLOR, pinn_lai='#eda100')
    fig, axes = plt.subplots(4, 3, figsize=(TW, 6.6), sharey='row')
    rows = []
    for i, ds in enumerate(DS):
        cid = AP.pick_cell(r0, ds)
        for j, frac in enumerate((0.1, 0.3, 0.7)):
            ax = axes[i, j]
            h = lv[(lv.dataset == ds) & (lv.frac == frac) & (lv.cell == cid)]
            t = h[h.model == 'mlp'].sort_values('cycle')
            ax.plot(t.cycle, t.y, color=MODEL_COLOR['truth'], lw=2.0, zorder=2)
            for m in ('mlp', 'pinn_semi', 'pinn_lai'):
                q = h[h.model == m].sort_values('cycle')
                ax.plot(q.cycle, q.p, color=color[m], lw=1.25 if m == 'pinn_lai' else 1.0,
                        alpha=0.8 if m == 'mlp' else 0.95, zorder=4 if m == 'pinn_lai' else 3)
                rows.append(dict(ds=ds, cell=cid, frac=frac, model=m, n=len(q), mae=float(np.abs(q.p - q.y).mean()),
                                 tv=float(np.abs(np.diff(q.p.values)).mean())))    # dao động giữa hai chu kỳ liền nhau
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
                        plt.Line2D([], [], color=MODEL_COLOR['pinn_semi'], lw=1.6, label='PINN-semi'),
                        plt.Line2D([], [], color=color['pinn_lai'], lw=1.6, label='PINN-lai')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=0.9, w_pad=0.8)
    save(fig, f'{OUT}/fig_pred_traj')
    pd.DataFrame(rows).to_csv(f'{OUT}/pred_traj_cells.csv', index=False)      # MAE của từng ô, cho lời văn


# ─────────────────────────────────────────────────────────── E22: dự báo SOH so với SOH đo được
# Mọi hình E22 dùng kịch bản 'uoc': cell test không được đo dung lượng lần nào, lịch sử SOH tới gốc là ước lượng
# PINN-lai; SOH đo được chỉ dùng để chấm. SOH đo vẽ bằng chấm, dự báo bằng đường.
E22C = {'gbdt': '#2a78d6', 'pinn': '#1baf7a', 'ket_hop': '#d9480f', 'est': '#7b5cd6', 'tau': '#4a5563',
        'org': '#9aa5ad', 'band': '#fd7e14', 'meas': '#1b2330'}      # PINN-GBDT màu cam để tách khỏi chấm số đo
E22_ORIGIN = {0: '20 %', 1: '40 %', 2: '60 %'}          # gốc dự báo: tỉ lệ số hàng dữ liệu đã qua của cell
_E22 = {}


def _e22():
    """Kết quả E22 (analyze_e22.load) kèm sai số tuyệt đối của mọi mô hình; nạp một lần."""
    if not _E22:
        import analyze_e22 as A22
        from pinnsoh.forecast import MODELS
        T, R, P, info = A22.load()
        for m in MODELS:
            T[f'e_{m}'] = (T[m] - T.y).abs()
        _E22.update(A=A22, T=T, R=R)
    return _E22['A'], _E22['T'], _E22['R']


def _e22_cells(ds):
    from pinnsoh.data import load_dataset
    key = ('cells', ds)
    if key not in _E22:
        _E22[key] = {c.cid: c for c in load_dataset('PINN4SOH/data', ds, clean='causal')}
    return _E22[key]


def _e22_rank(T, ds):
    """Sai số của PINN-GBDT ở mỗi cell test, trung bình trên phần đời còn lại (lần lặp 0, gốc 40 %), xếp tăng dần."""
    t = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.oq == 1)]
    return t.groupby('cid').e_ket_hop.mean().sort_values()


def _e22_pick(T, ds, q=50):
    """Cell ở phân vị q của _e22_rank (q = 50: cùng cách chọn như analyze_e22.py)."""
    e = _e22_rank(T, ds)
    return e.index[(len(e) - 1) // 2] if q == 50 else e.index[int(round(q / 100 * (len(e) - 1)))]


def _e22_measured(ax, cell, s=1.5):
    ax.scatter(cell.cycle, cell.soh, s=s, color=E22C['meas'], lw=0, alpha=0.55, zorder=2, rasterized=True)


def _e22_handles(plt, models=False):
    hd = [plt.Line2D([], [], color=E22C['meas'], marker='o', ls='', ms=3, alpha=0.7, label='SOH đo được'),
          plt.Line2D([], [], color=E22C['est'], lw=1.6, label='SOH ước lượng tới gốc'),
          plt.Line2D([], [], color=E22C['ket_hop'], lw=2, label='PINN-GBDT'),
          plt.Rectangle((0, 0), 1, 1, color=E22C['band'], alpha=0.25, label='khoảng 90 %')]
    if models:
        hd += [plt.Line2D([], [], color=E22C['gbdt'], lw=1.2, ls=(0, (3, 1.5)), label='GBDT'),
               plt.Line2D([], [], color=E22C['pinn'], lw=1.2, ls=(0, (3, 1.5)), label='PINN-dự báo')]
    return hd + [plt.Line2D([], [], color=E22C['tau'], lw=1, ls=(0, (4, 2)), label='ngưỡng cuối đời'),
                 plt.Line2D([], [], color=E22C['org'], lw=1, ls=(0, (2, 2)), label='gốc dự báo')]


def _e22_panel(ax, T, ds, cid, models=False, lw=2.0):
    """Một cell: SOH đo (chấm), SOH ước lượng tới gốc, dự báo từ gốc 40 % và khoảng 90 %; trả (k0, dung lượng danh định)."""
    from pinnsoh.forecast import TAU, e20_estimates
    g = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.oq == 1) & (T.cid == cid)].sort_values('h')
    fold, k0, nom = int(g.fold.iloc[0]), float(g.k.iloc[0]), float(g.nom.iloc[0])
    if ('est', ds, fold) not in _E22:                    # lịch sử ước lượng PINN-lai của fold (E20), nạp một lần
        _E22[('est', ds, fold)] = e20_estimates(ds, 0, fold, 'test')
    cyc, est = _E22[('est', ds, fold)][cid]
    _e22_measured(ax, _e22_cells(ds)[cid])
    ax.plot(cyc[cyc <= k0], est[cyc <= k0], color=E22C['est'], lw=1.5, zorder=3)
    x = k0 + g.h.values
    ax.fill_between(x, g.lo, g.hi, color=E22C['band'], alpha=0.18, lw=0, zorder=1)
    if models:
        ax.plot(x, g.gbdt, color=E22C['gbdt'], lw=1.1, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(x, g.pinn, color=E22C['pinn'], lw=1.1, ls=(0, (3, 1.5)), zorder=3)
    ax.plot(x, g.ket_hop, color=E22C['ket_hop'], lw=lw, zorder=4)
    ax.axhline(TAU[ds], color=E22C['tau'], lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axvline(k0, color=E22C['org'], lw=0.9, ls=(0, (2, 2)), zorder=1)
    cell = _e22_cells(ds)[cid]
    meas0 = float(np.median(cell.soh[cell.cycle <= k0][-10:]))      # SOH đo quanh gốc (trung vị 10 hàng cuối)
    return k0, nom, float(est[cyc <= k0][-1]) - meas0                # độ lệch của mức ước lượng tại gốc


def e22_traj():
    """Một cell test mỗi bộ (lần lặp 0, gốc 40 %, sai số PINN-GBDT ở trung vị), đủ ba mô hình và trục dung lượng."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    _, T, _ = _e22()
    fig, axes = _grid(plt, 4.8)
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        _, nom, _ = _e22_panel(ax, T, ds, _e22_pick(T, ds), models=True)
        ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
        sec = ax.secondary_yaxis('right', functions=(lambda s_, n=nom: s_ * n, lambda q, n=nom: q / n))
        if i % 2 == 1:
            sec.set_ylabel('dung lượng (Ah)')
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    for j, ax in enumerate(axes):
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        if j % 2 == 0:
            ax.set_ylabel('SOH')
        if j >= 2:
            ax.set_xlabel('chu kỳ')
    fig.legend(handles=_e22_handles(plt, models=True), loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e22_traj')


def e22_cells():
    """Ba cell test mỗi bộ, chọn theo sai số PINN-GBDT ở phân vị 25, 50 và 75 % của bộ (lần lặp 0, gốc 40 %)."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, GRID
    _, T, _ = _e22()
    Q = (25, 50, 75)
    fig, axes = plt.subplots(4, 3, figsize=(TW, 8.6))
    bias = {}
    for i, ds in enumerate(DS):
        for j, q in enumerate(Q):
            ax = axes[i, j]
            _, _, bias[(ds, q)] = _e22_panel(ax, T, ds, _e22_pick(T, ds, q), lw=1.7)
            ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
            panel_title(ax, ds, f'phân vị {q} %', pad=3)
            if j == 0:
                ax.set_ylabel('SOH')
            if i == len(DS) - 1:
                ax.set_xlabel('chu kỳ')
    fig.legend(handles=_e22_handles(plt), loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1], h_pad=1.0, w_pad=0.9)
    save(fig, f'{OUT}/fig_e22_cells')
    rows = []
    for ds in DS:
        e = _e22_rank(T, ds)
        for q in Q:
            cid = _e22_pick(T, ds, q)
            g = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.oq == 1) & (T.cid == cid)]
            rows.append(dict(ds=ds, q=q, cid=cid, mae=float(e[cid]), n_cell=len(e), n_future=len(g),
                             cov=float(((g.y >= g.lo) & (g.y <= g.hi)).mean()), bias0=bias[(ds, q)]))
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_cells_pick.csv', index=False)


def e22_origins():
    """Cùng cell của e22_traj, dự báo PINN-GBDT từ ba gốc 20, 40 và 60 % so với SOH đo được."""
    plt = setup()
    from pinnsoh.forecast import TAU
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    _, T, _ = _e22()
    shade = {0: '#ffa94d', 1: '#e8590c', 2: '#862e0b'}
    fig, axes = _grid(plt, 4.8)
    rows = []
    for ax, ds in zip(axes, DS):
        cid = _e22_pick(T, ds)
        cell = _e22_cells(ds)[cid]
        _e22_measured(ax, cell)
        lo_, hi_ = min(float(cell.soh.min()), TAU[ds]), float(cell.soh.max())
        ax.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.06 * (hi_ - lo_))   # dự báo rơi quá khung thì đi ra mép dưới
        ax.set_autoscaley_on(False)
        for oq in (0, 1, 2):
            g = T[(T.ds == ds) & (T.r == 0) & (T.scen == 'uoc') & (T.cid == cid) & (T.oq == oq)].sort_values('h')
            if g.empty:
                continue
            k0 = float(g.k.iloc[0])
            ax.plot(k0 + g.h.values, g.ket_hop.values, color=shade[oq], lw=1.8, zorder=4 + oq,
                    label=f'dự báo từ gốc {E22_ORIGIN[oq]}')
            rows.append(dict(ds=ds, cid=cid, oq=oq, k0=k0, mae=float(g.e_ket_hop.mean()),
                             below=bool(g.ket_hop.min() < ax.get_ylim()[0]), floor=bool(g.pinn.min() <= 0.4001)))
            ax.axvline(k0, color=shade[oq], lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.axhline(TAU[ds], color=E22C['tau'], lw=0.9, ls=(0, (4, 2)), zorder=1, label='ngưỡng cuối đời')
        ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    axes[0].scatter([], [], s=9, color=E22C['meas'], alpha=0.7, label='SOH đo được')
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index('SOH đo được')] + [i for i, x in enumerate(l) if x != 'SOH đo được']
    for i, ax in enumerate(axes):
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        if i % 2 == 0:
            ax.set_ylabel('SOH')
        if i >= 2:
            ax.set_xlabel('chu kỳ')
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e22_origins')
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_origins_pick.csv', index=False)


def e22_parity():
    """SOH dự báo (PINN-GBDT) so với SOH đo được ở ba tầm h = 0,1L; 0,25L; 0,5L (lần lặp 0, mọi cell và gốc)."""
    plt = setup()
    from pinnsoh.forecast import H_REL
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID, MUTED
    A22, T, _ = _e22()
    H = A22.at_horizons(T[(T.r == 0) & (T.scen == 'uoc')])
    hcol = dict(zip(H_REL, ('#9cc3e6', '#2a78d6', '#0b3d73')))
    fig, axes = _grid(plt, 6.0)
    rows = []
    for ax, ds in zip(axes, DS):
        h = H[H.ds == ds]
        lo, hi = float(h.y.min()), float(max(h.y.max(), h.ket_hop.max()))     # khung theo SOH đo được
        pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
        for hr in H_REL:
            v = h[h.h_rel == hr]; inside = (v.ket_hop >= lo).values
            ax.scatter(v.y[inside], v.ket_hop[inside], s=9, color=hcol[hr], lw=0.25, ec='white', zorder=3,
                       label=f'tầm {hr:g}'.replace('.', ',') + ' L')
            ax.scatter(v.y[~inside], np.full(int((~inside).sum()), lo), s=22, marker='v', color=hcol[hr], lw=0.3,
                       ec='white', zorder=4, clip_on=False)               # dự báo thấp hơn khung: vẽ ở mép dưới
            rows.append(dict(ds=ds, h_rel=hr, n=len(v), below=int((~inside).sum())))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.locator_params(nbins=4)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.xaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel('SOH dự báo')
        if i >= 2:
            ax.set_xlabel('SOH đo được')
    h_, l_ = axes[0].get_legend_handles_labels()
    tri, one = plt.Line2D([], [], color=MUTED, marker='v', ls='', ms=3.2), plt.Line2D([], [], color=MUTED, lw=0.8)
    h_ = [h_[0], tri, h_[1], one, h_[2]]                 # 3 cột, điền theo cột: hàng 1 là ba tầm, hàng 2 là hai ký hiệu
    l_ = [l_[0], 'dự báo thấp hơn khung', l_[1], 'dự báo đúng bằng số đo', l_[2]]
    fig.legend(h_, l_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), markerscale=1.6)
    fig.tight_layout(rect=[0, 0.09, 1, 1], h_pad=1.0, w_pad=1.0)
    save(fig, f'{OUT}/fig_e22_parity')
    pd.DataFrame(rows).to_csv(f'{OUT}/e22_parity.csv', index=False)


def e22_horizon():
    """MAE theo cell của sáu mô hình theo tầm dự báo h/L (mọi lần lặp, mọi gốc; kịch bản lịch sử ước lượng)."""
    plt = setup()
    from pinnsoh.forecast import MODELS
    from pinnsoh.plotstyle import panel_title, save, CHEM
    A22, T, _ = _e22()
    col = {'giu': '#b9c1c7', 'tuyen': '#8a96a0', 'vatly': '#eda100', 'gbdt': E22C['gbdt'], 'pinn': E22C['pinn'],
           'ket_hop': E22C['ket_hop']}
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    fig, axes = _grid(plt, 5.0)
    for ax, ds in zip(axes, DS):
        t = T[(T.ds == ds) & (T.scen == 'uoc')].copy()
        t['b'] = np.digitize(t.h / t.L, edges) - 1
        t = t[(t.b >= 0) & (t.b < len(edges) - 1)]
        g = t.groupby(['b', 'cid'])[[f'e_{m}' for m in MODELS]].mean().groupby('b').mean()
        x = (edges[:-1] + edges[1:])[g.index.values] / 2
        for m in MODELS:
            ax.plot(x, 1e3 * g[f'e_{m}'].values, color=col[m], lw=2.0 if m == 'ket_hop' else 1.2,
                    ls='-' if m in ('gbdt', 'pinn', 'ket_hop') else (0, (3, 1.5)), zorder=4 if m == 'ket_hop' else 3,
                    label=A22.LABEL[m])
        v = 1e3 * g[[f'e_{m}' for m in MODELS]].values
        _log_ticks(ax, 0.85 * np.nanmin(v), 1.15 * np.nanmax(v), cands=(2, 3, 5, 10, 20, 30, 50, 100, 200, 300))
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f'{v_:g}'.replace('.', ',')))
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    _finish(fig, axes, 'tầm dự báo h / L', 'MAE theo cell (×10$^{-3}$)', 2, 3)
    save(fig, f'{OUT}/fig_e22_horizon')


def e22_rul():
    """RUL dự báo (PINN-GBDT) so với RUL thực ở ba gốc 25, 50, 75 % của chu kỳ cuối đời thật; trung bình qua lần lặp."""
    plt = setup()
    from pinnsoh.forecast import TAU, RUL_ORIGINS
    from pinnsoh.plotstyle import panel_title, save, MUTED
    _, _, R = _e22()
    qcol = {0.25: '#9cc3e6', 0.5: '#2a78d6', 0.75: '#0b3d73'}
    fig, axes = _grid(plt, 6.0)
    for ax, ds in zip(axes, DS):
        u = R[(R.ds == ds) & (R.scen == 'uoc')].groupby(['cid', 'q'])[['true', 'ket_hop']].mean().reset_index()
        mx = float(max(u.true.max(), u.ket_hop.max())) * 1.05
        xx = np.array([0, mx])
        ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color='#e7ebe9', lw=0, zorder=0)
        ax.plot(xx, xx, color=MUTED, lw=0.8, zorder=1)
        for q in RUL_ORIGINS:
            v = u[u.q == q]
            ax.scatter(v.true, v.ket_hop, s=9, color=qcol[q], lw=0.25, ec='white', zorder=3,
                       label=f'gốc ở {int(q * 100)} % đời')
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect('equal'); ax.locator_params(nbins=4)
        panel_title(ax, ds, f'τ = {TAU[ds]:.2f}'.replace('.', ','), pad=4)
    h_, l_ = axes[0].get_legend_handles_labels()
    h_.append(plt.Rectangle((0, 0), 1, 1, color='#e7ebe9')); l_.append('sai lệch trong ±20 %')
    for i, ax in enumerate(axes):
        ax.grid(False)
        if i % 2 == 0:
            ax.set_ylabel('RUL dự báo (chu kỳ)')
        if i >= 2:
            ax.set_xlabel('RUL thực (chu kỳ)')
    fig.legend(h_, l_, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), markerscale=1.6)
    fig.tight_layout(rect=[0, 0.06, 1, 1], h_pad=1.0, w_pad=1.0)
    save(fig, f'{OUT}/fig_e22_rul')



# ─────────────────────────────────────────────────────────── E33: dự báo có đặc trưng lịch sử (HiPINN)
E33_SET = [('lab30', 'do', '30 % nhãn', '#86b6ef', 'o'), ('all', 'do', 'mọi cell huấn luyện', '#0b3d73', 's')]   # lịch sử đo


E33_FIG_MODELS = ['giu', 'tuyen', 'gbdt', 'pinn', 'ket_hop', 'gbdt_ls', 'pinn_ls', 'doi', 'hipinn_w', 'hipinn']


def e33_mae():
    """MAE theo cell (trung bình qua 3 lần lặp) của các mô hình, hai ngân sách nhãn, lịch sử SOH đo; mỗi bộ một panel."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    import analyze_e33 as A
    S = pd.read_csv(f'{OUT}/e33_summary.csv')
    fig, axes = _grid(plt, 5.6)
    order = list(reversed(E33_FIG_MODELS))
    for ax, ds in zip(axes, DS):
        for j, (b, sc, lab, col, mk) in enumerate(E33_SET):
            r = S[(S.budget == b) & (S.scen == sc) & (S.ds == ds)]
            if r.empty:
                continue
            y = np.arange(len(order)) + (j - (len(E33_SET) - 1) / 2) * 0.24
            ax.scatter([1e3 * float(r[m].iloc[0]) for m in order], y, s=22, color=col, marker=mk, zorder=3,
                       lw=0.3, ec='white', label=lab)
        ax.set_xscale('log')
        ax.set_yticks(range(len(order))); ax.set_yticklabels([A.LABEL[m] for m in order])
        lo, hi = ax.get_xlim()
        tk = [t for t in (5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100) if lo <= t <= hi]
        ax.set_xticks(tk[::2] if len(tk) > 5 else tk); ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    for i, ax in enumerate(axes):
        if i >= 2:
            ax.set_xlabel('MAE theo cell (×10$^{-3}$)')
        if i % 2 == 1:
            ax.set_yticklabels([])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(E33_SET), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1], h_pad=1.2, w_pad=1.0)
    save(fig, f'{OUT}/fig_e33_mae')


MIN_H = 0.25                                     # E33: cell minh hoạ phải còn ít nhất 0,25L chu kỳ đo sau gốc
E33_TAIL = [('ket_hop', 'PINN-GBDT', '#9aa5ad', (0, (5, 1.5)), 'v'),
            ('gbdt_ls', 'HiGBDT', '#2a78d6', (0, (3, 1.5)), 's'), ('pinn', 'PINN-dự báo', '#1baf7a', (0, (3, 1.5)), '^'),
            ('doi', 'nhánh theo đời', '#7b5cd6', (0, (1, 1.2)), 'D'), ('hipinn', 'HiPINN', '#d9480f', '-', 'o')]


def e33_tail():
    """MAE (×10⁻³) và độ lệch theo phần đời của chính cell (mọi cell train, lịch sử đo): phần đuôi dự báo."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    import analyze_e33 as A
    T = pd.read_csv(f'{OUT}/e33_tail.csv')
    T = T[(T.budget == 'all') & (T.scen == 'do')]
    fig, axes = _grid(plt, 4.6)
    for ax, ds in zip(axes, DS):
        g = T[T.ds == ds].set_index('pb').reindex(A.PLAB)
        x = np.arange(len(A.PLAB))
        for m, lab, col, ls, mk in E33_TAIL:
            ax.plot(x, 1e3 * g[m].values, color=col, ls=ls, lw=1.6 if m == 'hipinn' else 1.1, marker=mk, ms=3.5,
                    label=lab, zorder=4 if m == 'hipinn' else 3)
        ax.set_xticks(x); ax.set_xticklabels(['<0,5', '0,5–0,7', '0,7–0,8', '0,8–0,9', '0,9–1'])
        ax.set_ylim(bottom=0)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel('MAE (×10$^{-3}$)')
        if i >= 2:
            ax.set_xlabel('phần đời của cell (chu kỳ / chu kỳ cuối)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e33_tail')


def e33_traj():
    """Một cell test mỗi bộ (mọi cell train, lịch sử đo, lần lặp 0, gốc 0,4L; trong các cell còn ít nhất 0,25L chu kỳ
    đo sau gốc, cell có sai số HiPINN ở trung vị): SOH đo, HiGBDT, nhánh theo đời, HiPINN và khoảng 90 %.
    PINN-dự báo (nhánh thứ ba của HiPINN) không vẽ để hình dễ đọc."""
    plt = setup()
    import json as _json, glob as _glob
    from pinnsoh.forecast_combo import TCOL, ROOT as R33
    from pinnsoh.forecast import TAU
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    fig, axes = _grid(plt, 4.8)
    rows = []
    for ax, ds in zip(axes, DS):
        parts = []
        for f in sorted(_glob.glob(f'{R33}/fc_{ds}_all_cv0-*of5.npz')):
            q = np.load(f); info = _json.loads(str(q['info']))
            T = pd.DataFrame(q['traj'], columns=TCOL); T = T[(T.scen == 0) & (T.origin == T.origin)]
            T['cid'] = np.asarray(info['split']['test'])[T.cell.astype(int)]; T['L'] = info['L']
            T['o_idx'] = T.groupby('cid').origin.rank(method='dense').astype(int)
            parts.append(T)
        if not parts:
            continue
        T = pd.concat(parts, ignore_index=True)
        t = T[T.o_idx == 2]
        hmax = t.groupby('cid').h.max(); L = float(t.L.iloc[0])
        t = t[t.cid.isin(hmax.index[hmax >= MIN_H * L])]       # bỏ cell gần hết đời ngay ở gốc: đoạn dự báo quá ngắn
        e = t.assign(e=(t.hipinn - t.y).abs()).groupby('cid').e.mean().sort_values()
        cid = e.index[(len(e) - 1) // 2]
        g = t[t.cid == cid].sort_values('h'); k0 = float(g.origin.iloc[0])
        cell = _e22_cells(ds)[cid]
        _e22_measured(ax, cell)
        x = k0 + g.h.values
        ax.fill_between(x, g.lo_h, g.hi_h, color=E22C['band'], alpha=0.18, lw=0, zorder=1)
        ax.plot(x, g.hipinn, color=E22C['ket_hop'], lw=1.9, zorder=3)
        # hai nhánh vẽ đè lên HiPINN: chỗ HiPINN trùng một nhánh (trung vị) vẫn thấy nét đứt của nhánh đó
        ax.plot(x, g.doi, color=E22C['est'], lw=1.0, ls=(0, (1, 1.2)), zorder=4)
        ax.plot(x, g.gbdt_ls, color='#2a78d6', lw=1.0, ls=(0, (3, 1.5)), zorder=4)
        ax.axhline(TAU[ds], color=E22C['tau'], lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.axvline(k0, color=E22C['org'], lw=0.9, ls=(0, (2, 2)), zorder=1)
        lo_, hi_ = min(float(cell.soh.min()), TAU[ds]), float(cell.soh.max())
        ax.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.06 * (hi_ - lo_))
        ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
        rows.append(dict(ds=ds, cid=cid, k0=k0, mae=float(e[cid]), h_max=float(g.h.max()), L=L,
                         **{f'mae_{m}': float((g[m] - g.y).abs().mean()) for m in ('gbdt_ls', 'pinn', 'doi')},
                         n_cand=len(e), min_h=MIN_H, cov=float(((g.y >= g.lo_h) & (g.y <= g.hi_h)).mean())))
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel('SOH')
        if i >= 2:
            ax.set_xlabel('chu kỳ')
    hd = [plt.Line2D([], [], color=E22C['meas'], marker='o', ls='', ms=3, alpha=0.7, label='SOH đo được'),
          plt.Line2D([], [], color=E22C['ket_hop'], lw=2, label='HiPINN'),
          plt.Rectangle((0, 0), 1, 1, color=E22C['band'], alpha=0.25, label='khoảng 90 %'),
          plt.Line2D([], [], color='#2a78d6', lw=1.2, ls=(0, (3, 1.5)), label='HiGBDT'),
          plt.Line2D([], [], color=E22C['est'], lw=1.2, ls=(0, (1, 1.2)), label='nhánh theo đời'),
          plt.Line2D([], [], color=E22C['tau'], lw=1, ls=(0, (4, 2)), label='ngưỡng cuối đời'),
          plt.Line2D([], [], color=E22C['org'], lw=1, ls=(0, (2, 2)), label='gốc dự báo')]
    fig.legend(handles=hd, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e33_traj')
    pd.DataFrame(rows).to_csv(f'{OUT}/e33_traj_pick.csv', index=False)

# ─────────────────────────────────────────────────────────── đầu vào có lịch sử (E29), đủ bốn bộ
# ─────────────────────────────────────────────────────────── E29: mỗi phép so sánh một hình riêng
E29_DS = ['XJTU', 'TJU', 'HUST']               # MIT không có dạng lịch sử nào được chọn nên không có lượt có lịch sử


def _e29_row(name, spec):
    """Một hàng ba panel (XJTU, TJU, HUST): MAE theo cell trên test theo mức nhãn cho các mô hình trong spec."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    S = pd.read_csv(f'{OUT}/e29_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.55))
    for ax, ds in zip(axes, E29_DS):
        vals = []
        for m, col, ls, lab in spec:
            g = S[(S.dataset == ds) & (S.model == m)].sort_values('f')
            ax.plot(100 * g.f, 1e3 * g.mae, color=col, ls=ls, lw=1.8, marker='o', ms=4, label=lab)
            vals += list(1e3 * g.mae)
        _log_ticks(ax, min(vals) / 1.08, max(vals) * 1.08)
        ax.set_xticks([10, 30, 50, 70]); ax.set_xlim(5, 75)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_xlabel('% số cell có nhãn')
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(spec), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.09, 1, 1], w_pad=1.0)
    save(fig, f'{OUT}/{name}')


def e29_pinn():
    """PINN-semi không / có lịch sử."""
    _e29_row('fig_e29_pinn', [('pinn', '#9fd8c1', (0, (4, 2)), 'PINN-semi'),
                              ('pinn_h', '#1baf7a', '-', 'PINN-semi + lịch sử')])


def e29_lai():
    """PINN-lai tinh chỉnh không / có lịch sử, cạnh GBDT có hoặc không lịch sử chọn trên validation."""
    _e29_row('fig_e29_lai', [('pinn_lai_v2', '#86b6ef', (0, (4, 2)), 'PINN-lai'),
                             ('pinn_lai_h_v2', '#0b3d73', '-', 'PINN-lai + lịch sử'),
                             ('gbdt_best', '#eb6834', '-', 'GBDT')])


def e29_phys():
    """Vật lý khi đã có lịch sử: MLP và PINN-semi cùng đầu vào có lịch sử."""
    _e29_row('fig_e29_phys', [('mlp_h', '#7d8791', '-', 'MLP + lịch sử'),
                              ('pinn_h', '#1baf7a', '-', 'PINN-semi + lịch sử')])


MODEL_STYLE = {   # màu theo mô hình, giữ nguyên giữa các hình (SERIES của plotstyle, thứ tự cố định)
    'mlp': ('MLP', '#7d8791'), 'cnn': ('CNN', '#e87ba4'), 'lstm': ('LSTM', '#6250d6'),
    'transformer': ('Transformer', '#eda100'), 'gbdt': ('GBDT', '#eb6834'), 'pinn': ('PINN-semi', '#1baf7a'),
    'pinn3': ('3 PINN-semi', '#1baf7a'), 'pinn_lai': ('PINN-lai', '#0b3d73')}


def models_compare():
    """MAE theo cell (×10⁻³) của tám mô hình chính trên bốn bộ: 30 % (vòng rỗng) và 70 % nhãn (vòng đặc); dùng cho slide."""
    plt = setup()
    from pinnsoh.plotstyle import panel_title, save, GRID, INK, CHEM
    M = pd.read_csv(f'{OUT}/e30_metrics.csv')
    order = [m for m in ('mlp', 'cnn', 'lstm', 'transformer', 'gbdt', 'pinn', 'pinn_lai', 'final') if m in set(M.model)]
    style = dict(MODEL_STYLE, final=('Mô hình cuối', INK))
    fig, axes = plt.subplots(1, 4, figsize=(TW, 2.75), sharey=True)
    y = np.arange(len(order))[::-1]
    v = lambda ds, fr, m: 1e3 * float(M[(M.dataset == ds) & (M.f == fr) & (M.model == m)].MAE_cell.iloc[0])
    for ax, ds in zip(axes, ['XJTU', 'TJU', 'MIT', 'HUST']):
        for i, m in zip(y, order):
            c = style[m][1]
            ax.plot([v(ds, 0.7, m), v(ds, 0.3, m)], [i, i], color='#c9d0d4', lw=1.4, zorder=1)
            ax.scatter([v(ds, 0.3, m)], [i], s=30, facecolor='white', edgecolor=c, lw=1.5, zorder=3)
            ax.scatter([v(ds, 0.7, m)], [i], s=30, color=c, zorder=3)
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=.5)
        ax.locator_params(axis='x', nbins=4)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    axes[0].set_yticks(y); axes[0].set_yticklabels([style[m][0] for m in order])
    h = [plt.Line2D([], [], ls='', marker='o', mfc='white', mec=INK, mew=1.5, ms=6, label='30 % cell có nhãn'),
         plt.Line2D([], [], ls='', marker='o', color=INK, ms=6, label='70 % cell có nhãn')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.15, 1, 1], w_pad=0.8)
    fig.text(0.5, 0.125, 'MAE theo cell (×10$^{-3}$)', ha='center', va='center', fontsize=plt.rcParams['axes.labelsize'])
    save(fig, f'{OUT}/fig_models_compare')


FIGS = dict(curve_physics=curve_physics, leakage=leakage, e20_steps=e20_steps, e27_compare=e27_compare,
            e28_budget=e28_budget, overview=overview, pred_traj=pred_traj, e22_traj=e22_traj, e22_cells=e22_cells,
            e22_origins=e22_origins, e22_parity=e22_parity, e22_horizon=e22_horizon, e22_rul=e22_rul,
            e33_mae=e33_mae, e33_traj=e33_traj, e33_tail=e33_tail,
            e29_pinn=e29_pinn,
            e29_lai=e29_lai, e29_phys=e29_phys, models_compare=models_compare)
NEEDS = dict(curve_physics='thesis_curve', leakage='thesis_leakage', e20_steps='e20_summary', e27_compare='e27_summary',
             e28_budget='e28_summary', overview='overview', pred_traj='pred_metrics', e22_traj='e22_horizon',
             e22_cells='e22_horizon', e22_origins='e22_horizon', e22_parity='e22_horizon',
             e22_horizon='e22_horizon', e22_rul='e22_horizon', e33_mae='e33_summary', e33_traj='e33_summary',
             e33_tail='e33_tail',
             e29_pinn='e29_summary', e29_lai='e29_summary', e29_phys='e29_summary', models_compare='e30_metrics')


def main(names=None):
    for name in names or FIGS:
        if os.path.exists(f'{OUT}/{NEEDS[name]}.csv'):
            FIGS[name]()
            print('đã vẽ', name)
        else:
            print('bỏ qua', name, '(chưa có', NEEDS[name] + '.csv)')


if __name__ == '__main__':
    main(_sys.argv[1:] or None)
