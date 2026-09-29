"""
Vẽ lại toàn bộ hình kết quả theo khuôn báo khoa học (pinnsoh/plotstyle.py).

    python figures.py            # sinh results/fig_*.png và .pdf

Đọc trực tiếp results/*.csv và runs/**/*.json — không có số nào gõ tay.
"""
from __future__ import annotations
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

from pinnsoh.plotstyle import (apply_theme, banked_height, direct_label, panel_title, tidy, save,
                               spread_labels, log_mae_axis, guide_vline, MAE_TICKS,
                               figure_note, SERIES, MODEL_COLOR, MODEL_LABEL, INK, INK2, MUTED,
                               DATASETS, CHEM, RATIO_CMAP, SEQ_CMAP, ratio_norm, GRID)

R = 'results'
apply_theme()


def _load(pattern):
    rows = []
    for p in glob.glob(pattern):
        r = json.load(open(p)); c = r['config']
        rows.append(dict(dataset=c['dataset'], model=c['model'], frac=c['label_frac'], seed=c['seed'],
                         tag=c['tag'], norm=c['norm'], val_MAE=r['val']['MAE'], test_MAE=r['test']['MAE']))
    return pd.DataFrame(rows)


def _dss(df):
    return [d for d in DATASETS if (df.dataset == d).any()]


# ─────────────────────────────────────────── khuôn chung cho hai hình "MAE theo mức nhãn"
FR_TICKS = [10, 30, 50, 70]
MILLI = 1000.0            # đổi MAE sang đơn vị 10^-3 SOH cho nhãn trục gọn


def _lines_grid(n_panel, right_strip=0.0, head=0.0, foot=0.62,
                pw=1.46, ph=1.92, ygutter=0.66):
    """Lưới small-multiples dùng chung: 4 panel cao, một trục y log duy nhất.

    Panel được đặt CAO HƠN RỘNG (dạng chân dung) để bốn đường không bị bẹt dí vào
    nhau — đây là thay đổi chính so với bản cũ, nơi panel rộng 1,95 in mà chỉ cao
    ~1,2 in nên mọi độ dốc đều gần như nằm ngang."""
    W = ygutter + pw * n_panel + right_strip
    H = head + ph + foot
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(n_panel):
        left = (ygutter + pw * k) / W
        ax = fig.add_axes([left, foot / H, pw / W * 0.985, ph / H])
        axes.append(ax)
    return fig, axes, (W, H)


def _axis_note(fig, W, H, n_panel, note, pw=1.46, ygutter=0.66):
    """Nhãn trục x đặt giữa dải panel + ghi chú chân hình, toạ độ tính bằng inch."""
    fig.text((ygutter + pw * n_panel / 2) / W, 0.30 / H, 'phần trăm cell mang nhãn',
             ha='center', va='bottom')
    fig.text(0.02 / W, 0.10 / H, note, ha='left', va='bottom', color=MUTED,
             fontsize=plt.rcParams['font.size'] - 1.5)


def _panel_common(ax, ds, k, n, show_y, lo, hi, mark30=True):
    panel_title(ax, ds, CHEM[ds], pad=5)
    ax.set_xticks(FR_TICKS)
    ax.set_xlim(2, 78)
    if mark30:
        guide_vline(ax, 30, 'mốc 30 %' if k == 0 else None)
    log_mae_axis(ax, lo=lo, hi=hi)
    ax.grid(True, axis='y', color=GRID, lw=.55, zorder=0)
    if not show_y:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    for s in ('left',):
        ax.spines[s].set_visible(show_y)


# ─────────────────────────────────────────────────────────── 1. hiệu quả theo lượng nhãn
def fig_label_efficiency():
    df = _load('runs/E1/*.json')
    if df.empty: return
    dss = _dss(df)
    models = ['mlp', 'pinn_sup', 'pinn_semi']

    # miền hiển thị lấy từ chính dữ liệu (kể cả dải ±1sd), không gõ tay
    lo, hi = np.inf, -np.inf
    for ds in dss:
        for m in models:
            g = df[(df.dataset == ds) & (df.model == m)].groupby('frac').test_MAE
            if not len(g): continue
            mu, sd = g.mean() * MILLI, g.std().fillna(0) * MILLI
            lo = min(lo, float((mu - sd).min())); hi = max(hi, float((mu + sd).max()))
    lo, hi = lo * 0.92, hi * 1.10

    fig, axes, (W, H) = _lines_grid(len(dss), right_strip=1.30, foot=0.66)
    for k, (ax, ds) in enumerate(zip(axes, dss)):
        sub = df[df.dataset == ds]
        ends = []
        for m in models:
            g = sub[sub.model == m].groupby('frac').test_MAE
            if not len(g): continue
            mu, sd = g.mean() * MILLI, g.std().fillna(0) * MILLI
            x = mu.index.values * 100
            ax.fill_between(x, mu - sd, mu + sd, color=MODEL_COLOR[m], alpha=.17, lw=0, zorder=2)
            ax.plot(x, (mu - sd).values, color=MODEL_COLOR[m], lw=.55, alpha=.5, zorder=2)
            ax.plot(x, (mu + sd).values, color=MODEL_COLOR[m], lw=.55, alpha=.5, zorder=2)
            ax.plot(x, mu.values, '-', lw=2.0, color=MODEL_COLOR[m], zorder=4)
            ax.plot(x, mu.values, 'o', ms=4.0, color=MODEL_COLOR[m],
                    mec='white', mew=0.9, zorder=5)          # viền trắng: điểm nổi trên dải
            if k == len(dss) - 1:
                ends.append((float(mu.values[-1]), MODEL_LABEL[m], MODEL_COLOR[m]))
        _panel_common(ax, ds, k, len(dss), k == 0, lo, hi)
        # nhãn trực tiếp (Flint, quy tắc 4) trong dải chừa sẵn bên phải panel cuối.
        # Có đoạn dẫn nối từ điểm cuối tới chữ vì hai đường PINN trùng nhau ở HUST,
        # không có đoạn dẫn thì người đọc không biết nhãn nào thuộc đường nào.
        if ends:
            orig = {t: y for y, t, _ in ends}      # spread_labels trả về theo thứ tự đã sắp
            for y1, t, c in spread_labels(ax, ends, min_gap_px=11):
                y0 = orig[t]
                ax.annotate('', xy=(78, y1), xytext=(70.6, y0), textcoords='data',
                            annotation_clip=False, zorder=6,
                            arrowprops=dict(arrowstyle='-', color=c, lw=0.7, alpha=.7,
                                            shrinkA=1.5, shrinkB=0,
                                            connectionstyle='arc3,rad=0'))
                direct_label(ax, 78, y1, t, c, dx=3.5,
                             size=plt.rcParams['font.size'] - 1.0, weight=650)

    axes[0].set_ylabel('MAE trên tập test  ($\\times 10^{-3}$ SOH)', labelpad=3)
    _axis_note(fig, W, H, len(dss),
               'Trục dọc thang log dùng chung cho cả bốn bộ: cùng khoảng cách dọc = cùng TỈ SỐ sai số. '
               'Vùng nhạt là ±1 độ lệch chuẩn trên 3 seed.')
    save(fig, f'{R}/fig_E1_label_efficiency')


# ─────────────────────────────────────────────────────────── 2. tỉ số PINN/MLP (phân kỳ quanh 1)
def fig_ratio_matrix():
    """Hình chủ đạo: đại lượng CÓ DẤU quanh 1.0 nên dùng thang phân kỳ, tâm đúng 1.0."""
    e7 = pd.read_csv(f'{R}/E7_table.csv') if os.path.exists(f'{R}/E7_table.csv') else None
    e1 = pd.read_csv(f'{R}/E1_table.csv') if os.path.exists(f'{R}/E1_table.csv') else None
    if e7 is None or e1 is None: return
    fr = [0.1, 0.3, 0.5, 0.7]
    dss = [d for d in DATASETS if (e7.dataset == d).any()]

    def grid(getter):
        return np.array([[getter(d, f) for f in fr] for d in dss], dtype=float)

    def r_default(d, f):
        a = e1[(e1.dataset == d) & (e1.frac == f) & (e1.model == 'pinn_semi')].MAE
        b = e1[(e1.dataset == d) & (e1.frac == f) & (e1.model == 'mlp')].MAE
        return float(a.iloc[0] / b.iloc[0]) if len(a) and len(b) else np.nan

    def r_tuned(d, f):
        q = e7[(e7.dataset == d) & (e7.frac == f)]
        return float(q.ratio_vs_mlp.iloc[0]) if len(q) else np.nan

    M = [grid(r_default), grid(r_tuned)]
    titles = ['β = 5 cố định (cấu hình ban đầu)', 'β chọn theo validation']
    lo, hi = np.nanmin(M), np.nanmax(M)
    norm = ratio_norm(lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    for ax, m, t in zip(axes, M, titles):
        im = ax.imshow(m, cmap=RATIO_CMAP, norm=norm, aspect='auto')
        ax.set_xticks(range(len(fr)), [f'{int(f*100)}' for f in fr])
        ax.set_yticks(range(len(dss)), [f'{d}' for d in dss])
        ax.set_xlabel('% cell mang nhãn'); ax.grid(False)
        panel_title(ax, t)
        for i in range(len(dss)):
            for j in range(len(fr)):
                v = m[i, j]
                if not np.isfinite(v): continue
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.6,
                        color='white' if (v < 0.72 or v > 1.22) else INK,
                        fontweight='bold' if v < 1 else 'normal')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=.028, pad=.02, ticks=[round(lo, 2), 1.0, round(hi, 2)])
    cb.ax.set_yticklabels([f'{lo:.2f}\nPINN tốt hơn', '1.00\nhoà', f'{hi:.2f}\nMLP tốt hơn'], fontsize=7)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0)
    fig.suptitle('Tỉ số MAE của PINN so với MLP ở cùng mức nhãn — thang phân kỳ, tâm tại 1.00',
                 x=0.012, ha='left', fontsize=9.5, fontweight='bold', y=1.13)
    save(fig, f'{R}/fig_ratio_matrix')


# ─────────────────────────────────────────────────────────── 3. trọng số vật lý theo mức nhãn
def fig_weight():
    e7 = _load('runs/E7/*.json')
    if e7.empty: return
    e1 = _load('runs/E1/*.json')
    dss = _dss(e7)
    order = ['b5', 'b0.5', 'b1.5', 'b0.5_ag', 'adapt']
    lab = {'b5': 'β = 5', 'b0.5': 'β = 0.5', 'b1.5': 'β = 1.5',
           'b0.5_ag': 'β = 0.5, phần dư autograd', 'adapt': 'β tự thích ứng'}
    col = {'b5': SERIES[3], 'b0.5': SERIES[2], 'b1.5': '#7fd0bb',
           'b0.5_ag': '#0d6e56', 'adapt': SERIES[4]}
    strong = ('b5', 'b0.5_ag', 'adapt')       # ba biến thể được bàn kỹ trong bài

    lo, hi = np.inf, -np.inf
    for ds in dss:
        for t in order:
            g = e7[(e7.dataset == ds) & (e7.tag == t)].groupby('frac').test_MAE.mean()
            if len(g):
                lo = min(lo, float(g.min()) * MILLI); hi = max(hi, float(g.max()) * MILLI)
        b = e1[(e1.dataset == ds) & (e1.model == 'mlp')].groupby('frac').test_MAE.mean()
        if len(b):
            lo = min(lo, float(b.min()) * MILLI); hi = max(hi, float(b.max()) * MILLI)
    lo, hi = lo * 0.90, hi * 1.12

    HEAD = 0.42                                # dải chú giải một hàng ngay trên panel
    fig, axes, (W, H) = _lines_grid(len(dss), head=HEAD, foot=0.66)
    for k, (ax, ds) in enumerate(zip(axes, dss)):
        sub = e7[e7.dataset == ds]
        b = e1[(e1.dataset == ds) & (e1.model == 'mlp')].groupby('frac').test_MAE.mean()
        if len(b):
            ax.plot(b.index.values * 100, b.values * MILLI, '--', lw=1.5, dashes=(3.2, 2.2),
                    color=MODEL_COLOR['mlp'], zorder=6)
        for t in order:
            g = sub[sub.tag == t].groupby('frac').test_MAE.mean()
            if not len(g): continue
            hot = t in strong
            x, y = g.index.values * 100, g.values * MILLI
            ax.plot(x, y, '-', lw=2.1 if hot else 1.15, color=col[t],
                    alpha=1.0 if hot else .5, zorder=4 if hot else 3)
            ax.plot(x, y, 'o', ms=4.0 if hot else 2.6, color=col[t],
                    alpha=1.0 if hot else .5, mec='white', mew=0.9 if hot else 0.0,
                    zorder=5 if hot else 3)
        _panel_common(ax, ds, k, len(dss), k == 0, lo, hi)

    axes[0].set_ylabel('MAE trên tập test  ($\\times 10^{-3}$ SOH)', labelpad=3)
    handles = [plt.Line2D([], [], color=col[t], lw=2.1 if t in strong else 1.3,
                          alpha=1.0 if t in strong else .55, label=lab[t])
               for t in order]
    handles.append(plt.Line2D([], [], color=MODEL_COLOR['mlp'], lw=1.5, ls='--',
                              dashes=(3.2, 2.2), label='MLP cùng mức nhãn'))
    fig.legend(handles=handles, loc='upper left', ncol=len(handles),
               bbox_to_anchor=(0.66 / W, 1.0), borderaxespad=0.0,
               handletextpad=0.5, columnspacing=1.0)
    _axis_note(fig, W, H, len(dss),
               'Trục dọc thang log dùng chung cho cả bốn bộ. Nét đậm: ba biến thể được phân tích kỹ; '
               'nét mờ: hai biến thể đối chứng.')
    save(fig, f'{R}/fig_E7_tuned')


# ─────────────────────────────────────────────────────────── 4. kiến trúc: thanh phân kỳ quanh 1
def fig_arch():
    if not os.path.exists(f'{R}/E9_table.csv'): return
    e9 = pd.read_csv(f'{R}/E9_table.csv')
    desc = {'silu-mlp': 'MLP + SiLU  (mốc)', 'tanh-mlp': 'MLP + tanh', 'sin-mlp': 'MLP + sin  (PINN4SOH)',
            'snake-mlp': 'MLP + Snake', 'silu-res': 'Residual MLP', 'silu-fourier': 'Fourier(t) + MLP',
            'silu-mono': 'Đầu ra đơn điệu', 'silu-mono+L': 'Đơn điệu + giữ L_mono'}

    def rel(model, tag):
        s = e9[e9.model == model]
        b = s[s.tag == 'silu-mlp'].set_index('dataset').MAE
        r = s[s.tag == tag].set_index('dataset').MAE
        c = b.index.intersection(r.index)
        return float((r[c] / b[c]).mean()) if len(c) else np.nan

    tags = [t for t in desc if (e9.tag == t).any() and t != 'silu-mlp']
    rows = [(t, rel('mlp', t), rel('pinn_semi', t)) for t in tags]
    rows.sort(key=lambda r: np.nanmean([r[1], r[2]]))
    y = np.arange(len(rows)); h = 0.34
    XMAX = 1.16                                        # kẹp trục; cột vượt khung đánh dấu riêng

    fig, ax = plt.subplots(figsize=(5.6, 0.33 * len(rows) + 1.25))
    ax.axvline(1.0, color=INK2, lw=1.0, zorder=4)
    for i, (t, a, b) in enumerate(rows):
        for v, off, c in [(a, +h / 2, MODEL_COLOR['mlp']), (b, -h / 2, MODEL_COLOR['pinn_semi'])]:
            if not np.isfinite(v): continue
            good = v < 1
            vc = min(v, XMAX)
            ax.barh(i + off, vc - 1, left=1, height=h, color=c, alpha=1.0 if good else .42,
                    edgecolor='none', zorder=3)
            if v > XMAX:                               # vượt khung: mũi tên + giá trị thật
                ax.annotate(f'{v:.2f} →', xy=(XMAX, i + off), xytext=(-3, 0),
                            textcoords='offset points', ha='right', va='center',
                            fontsize=7.2, color='white', fontweight='bold', zorder=6)
            else:
                ax.text(v + (-0.004 if good else 0.004), i + off, f'{v:.3f}',
                        ha='right' if good else 'left', va='center', fontsize=7.2,
                        color=INK if good else MUTED, zorder=6)
    ax.set_yticks(y, [desc[t] for t in [r[0] for r in rows]])
    ax.invert_yaxis(); ax.grid(False)
    ax.xaxis.grid(True, color='#eef1ef', lw=.55)
    ax.set_xlabel('MAE chuẩn hoá theo MLP + SiLU   (bên trái mốc 1.00 = tốt hơn)')
    ax.set_xlim(0.86, XMAX)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR['mlp'], label='baseline chỉ dữ liệu'),
                       plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR['pinn_semi'], label='PINN-semi')],
              loc='upper left', bbox_to_anchor=(0.0, -0.16), ncol=2)
    ax.set_title('Kiến trúc và hàm kích hoạt — 30 % nhãn, trung bình 4 bộ dữ liệu',
                 loc='left', fontweight='bold', pad=8)
    fig.tight_layout()
    save(fig, f'{R}/fig_E9_arch')


# ─────────────────────────────────────────────────────────── 5. ngân sách cân bằng: mũi tên
def fig_fair():
    if not os.path.exists(f'{R}/E12_table.csv'): return
    t = pd.read_csv(f'{R}/E12_table.csv')
    fracs = sorted(t.frac.unique())
    dss = [d for d in DATASETS if (t.dataset == d).any()]
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.25 * len(fracs) + .5, 2.55), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, fr in zip(axes, fracs):
        sub = t[t.frac == fr].set_index('dataset').reindex(dss)
        y = np.arange(len(dss))
        for i, d in enumerate(dss):
            r = sub.loc[d]
            ax.plot([r.mlp_default, r.mlp], [i, i], '-', color='#c9d3da', lw=3.2, solid_capstyle='butt', zorder=2)
            ax.plot([r.mlp, r.pinn], [i, i], '-', color='#bfe3d7', lw=3.2, solid_capstyle='butt', zorder=2)
            ax.plot(r.mlp_default, i, 'o', ms=5.5, color='#9ec5f4', mew=0, zorder=3)
            ax.plot(r.mlp, i, 'o', ms=6, color=MODEL_COLOR['mlp'], mew=0, zorder=4)
            ax.plot(r.pinn, i, 'o', ms=6, color=MODEL_COLOR['pinn_semi'], mew=0, zorder=5)
            ax.annotate(f'{r.ratio:.2f}×', xy=(1.0, i), xycoords=('axes fraction', 'data'),
                        xytext=(4, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=7.4, annotation_clip=False,
                        color=MODEL_COLOR['pinn_semi'] if r.ratio < 1 else MUTED,
                        fontweight='bold' if r.ratio < 1 else 'normal')
        ax.set_yticks(y, dss); ax.invert_yaxis()
        ax.grid(False); ax.xaxis.grid(True, color='#eef1ef', lw=.55)
        vmax = float(sub[['mlp_default', 'mlp', 'pinn']].values.max())
        ax.set_xlim(0, vmax * 1.06); ax.locator_params(axis='x', nbins=4)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        panel_title(ax, f'{int(fr*100)} % cell mang nhãn')
        ax.set_xlabel('MAE trên tập test')
    handles = [plt.Line2D([], [], marker='o', ls='', ms=6, color='#9ec5f4', label='MLP cấu hình mặc định'),
               plt.Line2D([], [], marker='o', ls='', ms=6, color=MODEL_COLOR['mlp'], label='MLP sau tinh chỉnh'),
               plt.Line2D([], [], marker='o', ls='', ms=6, color=MODEL_COLOR['pinn_semi'], label='PINN sau tinh chỉnh')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Cùng một lưới 8 biến thể siêu tham số cho hai mô hình, chọn theo validation',
                 x=0.008, ha='left', fontsize=9.5, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0.08, 0.965, 1], w_pad=3.2)
    save(fig, f'{R}/fig_E12_fair')


# ─────────────────────────────────────────────────────────── 6. ma trận chuyển miền
def fig_transfer_matrix():
    if not os.path.exists(f'{R}/E10_table.csv'): return
    t = pd.read_csv(f'{R}/E10_table.csv')
    combos = [('mlp', 'global'), ('pinn', 'global'), ('mlp', 'first_cycle'), ('pinn', 'first_cycle')]
    combos = [c for c in combos if not t[(t.model == c[0]) & (t.norm == c[1])].empty]
    M = {}
    for m, n in combos:
        s = t[(t.model == m) & (t.norm == n)]
        M[(m, n)] = np.array([[s[(s.source == a) & (s.target == b)].MAE.mean() for b in DATASETS]
                              for a in DATASETS])
    vmin = max(1e-3, min(np.nanmin(v) for v in M.values()))
    vmax = max(np.nanmax(v) for v in M.values())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, len(combos), figsize=(2.15 * len(combos) + 1.0, 2.5))
    axes = np.atleast_1d(axes)
    for ax, (m, n) in zip(axes, combos):
        A = M[(m, n)]
        im = ax.imshow(A, cmap=SEQ_CMAP, norm=norm)
        ax.set_xticks(range(4), DATASETS, fontsize=7.2, rotation=32, ha='right')
        ax.set_yticks(range(4), DATASETS, fontsize=7.2)
        ax.grid(False); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(4):
            for j in range(4):
                v = A[i, j]
                if not np.isfinite(v): continue
                txt = f'{v:.3f}' if v < 1 else (f'{v:.1f}' if v < 100 else f'{v:.0f}')
                ax.text(j, i, txt, ha='center', va='center', fontsize=6.8,
                        color='white' if norm(v) > .55 else INK,
                        fontweight='bold' if i == j else 'normal')
                if i == j:                              # đường chéo = kết quả TRONG miền
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor='#b8531f', lw=1.5, zorder=5))
        panel_title(ax, ('MLP' if m == 'mlp' else 'PINN'),
                    'chuẩn hoá ' + ('toàn cục' if n == 'global' else 'theo chu kỳ đầu'))
    axes[0].set_ylabel('huấn luyện trên')
    cb = fig.colorbar(im, ax=axes, fraction=.02, pad=.015)
    cb.set_label('MAE trên tập test của bộ đích (thang log)', fontsize=7.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=7)
    fig.suptitle('Ma trận chuyển miền 4×4 — viền cam là đường chéo, tức kết quả trong miền',
                 x=0.008, ha='left', fontsize=9.5, fontweight='bold', y=1.03)
    save(fig, f'{R}/fig_E10_matrix')


# ─────────────────────────────────────────────────────────── 7. thích nghi miền
def fig_adapt():
    if not os.path.exists(f'{R}/E12_table.csv') or not os.path.exists(f'{R}/E5_table.csv'): return
    e5 = pd.read_csv(f'{R}/E5_table.csv')
    e1 = pd.read_csv(f'{R}/E1_table.csv')
    pairs = [(a, b) for a, b in zip(e5.source, e5.target)]
    pairs = list(dict.fromkeys(pairs))
    keys = [(0, 'mlp'), (0, 'pinn_semi'), (3, 'mlp'), (3, 'pinn_semi')]
    lab = {(0, 'mlp'): 'MLP, không nhãn đích', (0, 'pinn_semi'): 'PINN, không nhãn đích',
           (3, 'mlp'): 'MLP, 3 cell đích có nhãn', (3, 'pinn_semi'): 'PINN, 3 cell đích có nhãn'}
    col = {(0, 'mlp'): '#9ec5f4', (0, 'pinn_semi'): '#8fd6c1',
           (3, 'mlp'): MODEL_COLOR['mlp'], (3, 'pinn_semi'): MODEL_COLOR['pinn_semi']}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(pairs)); w = .19
    for i, k in enumerate(keys):
        v = [float(e5[(e5.source == a) & (e5.target == b) & (e5.k == k[0]) & (e5.model == k[1])].MAE.iloc[0])
             for a, b in pairs]
        ax.bar(x + (i - 1.5) * w, v, w * .88, color=col[k], label=lab[k], zorder=3)
    ref = [float(e1[(e1.dataset == b) & (e1.model == 'mlp') & (e1.frac == .7)].MAE.iloc[0]) for _, b in pairs]
    for j, r in enumerate(ref):
        ax.plot([x[j] - .42, x[j] + .42], [r, r], '-', color='#b8531f', lw=1.4, zorder=5)
    ax.plot([], [], '-', color='#b8531f', lw=1.4, label='mốc: MLP huấn luyện trong miền đích')
    ax.set_yscale('log'); ax.set_xticks(x, [f'{a}→{b}' for a, b in pairs])
    ax.set_ylabel('MAE trên cell test của bộ đích'); ax.grid(False)
    ax.yaxis.grid(True, color='#eef1ef', lw=.55, which='both')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    ax.set_title('Thích nghi sang bộ dữ liệu khác bằng các loss không cần nhãn — trục log',
                 loc='left', fontweight='bold', pad=8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, f'{R}/fig_E5_adapt')


# ─────────────────────────────────────────────────────────── 8. quỹ đạo trên cell test
def fig_trajectories():
    for ds in DATASETS:
        preds = {}
        for m in ['mlp', 'pinn_semi']:
            p = f'runs/E1/{ds}_{m}_f0.3_s0_global_global.json'
            if os.path.exists(p):
                preds[m] = json.load(open(p))['test_predictions']
        if len(preds) < 2: continue
        y = np.array(preds['mlp']['y']); cell = np.array(preds['mlp']['cell'])
        cells = np.unique(cell)[:4]
        fig, axes = plt.subplots(1, len(cells), figsize=(1.85 * len(cells) + .9, 1.85), sharey=True)
        axes = np.atleast_1d(axes)
        for k, (ax, c) in enumerate(zip(axes, cells)):
            idx = cell == c; n = int(idx.sum()); xx = np.arange(n)
            ax.plot(xx, y[idx], color=MODEL_COLOR['truth'], lw=1.9, zorder=2)
            for m in ['mlp', 'pinn_semi']:
                ax.plot(xx, np.array(preds[m]['p'])[idx], color=MODEL_COLOR[m], lw=1.1, alpha=.95, zorder=3)
            panel_title(ax, f'cell test #{c}', pad=4)
            ax.grid(False); ax.yaxis.grid(True, color='#eef1ef', lw=.55)
            ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
            if k == len(cells) // 2:
                ax.set_xlabel('chu kỳ (sau lọc)')
        axes[0].set_ylabel('SOH')
        fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR[m], lw=1.8,
                                       label=MODEL_LABEL[m].split(' (')[0])
                            for m in ['truth', 'mlp', 'pinn_semi']],
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(f'{ds} — dự đoán khi chỉ 30 % cell mang nhãn (seed 0)',
                     x=0.008, ha='left', fontsize=9.5, fontweight='bold', y=1.02)
        fig.tight_layout(rect=[0, 0.16, 1, 0.99])
        save(fig, f'{R}/fig_traj_{ds}')


if __name__ == '__main__':
    fig_label_efficiency(); fig_ratio_matrix(); fig_weight(); fig_arch()
    fig_fair(); fig_transfer_matrix(); fig_adapt(); fig_trajectories()
    print('xong:', ', '.join(sorted(os.path.basename(p) for p in glob.glob(f'{R}/fig_*.pdf'))))
