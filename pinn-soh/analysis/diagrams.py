"""
Sơ đồ kiến trúc mô hình cho báo cáo đồ án.

    python analysis/diagrams.py       # -> results/fig_thesis_arch.png|pdf
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pinnsoh.plotstyle import apply_theme, save

INK, MUTED = '#1b2330', '#4a5563'
C_IN, C_NET, C_PHY, C_LOSS, C_OUT = '#eef3fb', '#dbe8f8', '#e3f4ec', '#fbeee6', '#f2f2f2'
E_IN, E_NET, E_PHY, E_LOSS = '#9fb7d6', '#2a78d6', '#1baf7a', '#eb6834'


def box(ax, x, y, w, h, title, sub=None, fc=C_IN, ec=E_IN, fs=7.6, weight='bold'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec=ec, lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center', fontsize=fs, color=INK, weight='bold')
        ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center', fontsize=fs - 0.8, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs, color=INK, weight=weight)
    return (x, y, w, h)


def arrow(ax, p, q, color='#6b7680', style='-|>', ls='-'):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=0.9, ls=ls,
                                                    shrinkA=0, shrinkB=0, mutation_scale=8))


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


def poly(ax, pts, color, ls='-', head=True):
    """Đường gấp khúc vuông góc, mũi tên ở đoạn cuối."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=0.9, ls=ls, solid_capstyle='butt',
            dash_capstyle='butt') if not head else ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9, ls=ls)
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, ls=ls)


def region(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc='#f7f9fa', ec='#dfe5e8', lw=0.7, zorder=-1))
    ax.text(x + 0.08, y + h - 0.08, label, ha='left', va='top', fontsize=6.6, color='#78848d', weight='bold')


def arch():
    apply_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis('off')
    region(ax, 0.05, 0.85, 2.25, 4.2, 'DỮ LIỆU')
    region(ax, 2.6, 0.85, 4.0, 4.2, 'MÔ HÌNH')
    region(ax, 7.05, 0.85, 2.9, 4.2, 'HÀM MẤT MÁT')
    lab = box(ax, 0.25, 3.3, 1.85, 0.85, 'Cell có nhãn', '$(\\mathbf{x}, t, y)$')
    unl = box(ax, 0.25, 1.2, 1.85, 0.85, 'Mọi cell train', 'chỉ $(\\mathbf{x}, t)$')
    F = box(ax, 2.85, 3.3, 2.15, 0.85, 'Mạng nghiệm $F_\\varphi$', '17 → 128 → 128 → 64 → 1', C_NET, E_NET)
    u = box(ax, 5.3, 3.3, 1.1, 0.85, 'SOH $u$', '$F_\\varphi(\\tilde{\\mathbf{x}}, t)$', C_OUT, '#9aa5ad')
    G = box(ax, 2.85, 1.2, 1.55, 0.85, 'Mạng $G_\\theta$', '16 → 32 → 32 → 1', C_PHY, E_PHY, fs=7.4)
    r = box(ax, 4.65, 1.2, 1.75, 0.85, '$r = k(\\mathbf{x})\\,g(u)\\,A(T)$', 'tốc độ suy giảm ≥ 0', C_PHY, E_PHY, fs=7.4)
    LX, LW = 7.3, 2.45
    ld = box(ax, LX, 3.75, LW, 0.6, '$\\mathcal{L}_{data}$   $(u-y)^2$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    lm = box(ax, LX, 2.95, LW, 0.6, '$\\mathcal{L}_{mono}$   ReLU$(u_2-u_1-\\varepsilon)$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    lr = box(ax, LX, 2.15, LW, 0.6, '$\\mathcal{L}_{range}$   SOH hợp lý', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    lo = box(ax, LX, 1.2, LW, 0.75, '$\\mathcal{L}_{ode}$   $(\\partial_t u + r)^2$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    box(ax, 0.05, 0.05, 9.9, 0.6,
        '$\\mathcal{L} = \\mathcal{L}_{data} + w(s)\\,(\\alpha\\mathcal{L}_{ode} + \\beta\\mathcal{L}_{mono} + '
        '\\gamma\\mathcal{L}_{range})$     cập nhật đồng thời $\\varphi, \\theta, \\lambda, E_a$', None, '#ffffff', '#9aa5ad', fs=7.6, weight='normal')
    GREY = '#6b7680'
    arrow(ax, right(lab), left(F), color=GREY)
    poly(ax, [(2.1, 1.85), (2.45, 1.85), (2.45, 3.5), (2.85, 3.5)], E_NET)          # mọi cell -> F
    arrow(ax, (2.1, 1.45), (2.85, 1.45), color=E_PHY)                                # mọi cell -> G
    arrow(ax, right(F), left(u), color=GREY)
    arrow(ax, right(G), left(r), color=E_PHY)
    # thanh dẫn từ u tới bốn hàm mất mát
    BX = 6.8
    ax.plot([6.4, BX], [3.725, 3.725], color=E_NET, lw=0.9)
    ax.plot([BX, BX], [4.05, 1.75], color=E_NET, lw=0.9)
    for yy in (4.05, 3.25, 2.45, 1.75):
        arrow(ax, (BX, yy), (LX, yy), color=E_NET)
    arrow(ax, (6.4, 1.4), (LX, 1.4), color=E_PHY)                                    # r -> L_ode
    poly(ax, [(1.175, 4.15), (1.175, 4.5), (8.525, 4.5), (8.525, 4.35)], E_LOSS, ls=(0, (3, 2)))   # nhãn y
    save(fig, 'results/fig_thesis_arch')


def protocol():
    """Sơ đồ giao thức E21: chia bộ nguồn (A: 70/30, B: 70/15/15), kiểm tra trên ba bộ đích, và chuỗi thích nghi."""
    import numpy as np
    apply_theme()
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis('off')
    C_TR, C_VA, C_TE, C_AD = '#2a78d6', '#9aa5ad', '#eb6834', '#b9ded3'
    def bar(x, y, w, parts, label):
        x0 = x
        for frac, col, txt in parts:
            ax.add_patch(FancyBboxPatch((x0, y), w * frac - 0.02, 0.42, boxstyle='square,pad=0', fc=col, ec='white',
                                        lw=0.8))
            if txt:
                ax.text(x0 + w * frac / 2, y + 0.21, txt, ha='center', va='center', fontsize=6.6, color='white',
                        weight='bold')
            x0 += w * frac
        ax.text(x - 0.08, y + 0.21, label, ha='right', va='center', fontsize=7.2, color=INK, weight='bold')
    ax.text(0.1, 4.35, 'Bộ nguồn S', fontsize=7.6, color=MUTED, weight='bold')
    bar(1.25, 3.55, 3.2, [(0.7, C_TR, 'train 70 %'), (0.3, C_VA, 'valid 30 %')], 'A')
    bar(1.25, 2.85, 3.2, [(0.7, C_TR, 'train 70 %'), (0.15, C_VA, 'valid'), (0.15, C_TE, 'test')], 'B')
    ax.text(2.85, 2.55, 'nhãn của S: train để học, valid để dừng sớm và chọn mô hình',
            ha='center', fontsize=6.4, color=MUTED)
    # ba bộ đích
    ax.text(5.6, 4.35, 'Ba bộ đích T ≠ S (không dùng nhãn khi học)', fontsize=7.6, color=MUTED, weight='bold')
    for i, lab in enumerate(['$T_1$', '$T_2$', '$T_3$']):
        y = 3.75 - 0.5 * i
        ax.add_patch(FancyBboxPatch((6.6, y), 3.2, 0.36, boxstyle='square,pad=0', fc=C_TE, ec='white', lw=0.8,
                                    alpha=0.85))
        ax.text(8.2, y + 0.18, 'test: mọi cell', ha='center', va='center', fontsize=6.6, color='white', weight='bold')
        ax.text(6.52, y + 0.18, lab, ha='right', va='center', fontsize=7.2, color=INK, weight='bold')
    G = '#6b7680'
    ax.plot([4.5, 5.2], [3.76, 3.76], color=G, lw=0.9); ax.plot([4.5, 5.2], [3.06, 3.06], color=G, lw=0.9)
    ax.plot([5.2, 5.2], [3.06, 3.76], color=G, lw=0.9); ax.plot([5.2, 5.75], [3.41, 3.41], color=G, lw=0.9)
    ax.plot([5.75, 5.75], [2.93, 3.93], color=G, lw=0.9)
    for y in (3.93, 3.43, 2.93):
        arrow(ax, (5.75, y), (6.3, y), color=G)
    ax.text(8.2, 2.55, '$S, T \\in$ {XJTU, TJU, MIT, HUST}: 4 × 3 = 12 cặp', ha='center', fontsize=6.4, color=MUTED)
    # chuỗi thích nghi
    ax.plot([0.1, 9.9], [2.2, 2.2], color='#dfe5e8', lw=0.8)
    ax.text(0.1, 1.95, 'Thích nghi không nhãn và học lặp (giao thức A, với từng cặp S → T)', fontsize=7.6, color=MUTED,
            weight='bold')
    ax.add_patch(FancyBboxPatch((0.25, 1.05), 1.55, 0.42, boxstyle='square,pad=0', fc=C_AD, ec='white'))
    ax.add_patch(FancyBboxPatch((1.8, 1.05), 1.55, 0.42, boxstyle='square,pad=0', fc=C_TE, ec='white'))
    ax.text(1.02, 1.26, 'nửa thích nghi', ha='center', va='center', fontsize=6.4, color=INK)
    ax.text(2.57, 1.26, 'nửa test', ha='center', va='center', fontsize=6.4, color='white', weight='bold')
    ax.text(1.8, 0.82, 'T chia đôi; đổi vai hai nửa', ha='center', fontsize=6.3, color=MUTED)
    steps = [('Vòng 0', 'chuẩn hoá theo bộ\n+ vật lý trên nửa thích nghi'),
             ('Nhãn giả', 'kết hợp PINN + GBDT\nqua bộ lọc quỹ đạo'),
             ('Vòng 1, 2', 'học lại với nhãn giả\n+ vật lý')]
    xs = [4.1, 6.05, 8.0]
    for (t, sub), x in zip(steps, xs):
        box(ax, x, 0.85, 1.75, 0.85, t, sub, C_PHY if t != 'Nhãn giả' else C_LOSS, E_PHY if t != 'Nhãn giả' else E_LOSS,
            fs=7.0)
    arrow(ax, (3.4, 1.26), (4.1, 1.26), color='#6b7680')
    arrow(ax, (5.85, 1.27), (6.05, 1.27), color=E_PHY)
    arrow(ax, (7.8, 1.27), (8.0, 1.27), color=E_LOSS)
    poly(ax, [(8.87, 0.85), (8.87, 0.45), (6.92, 0.45), (6.92, 0.85)], E_PHY, ls=(0, (3, 2)))
    ax.text(7.9, 0.28, 'lặp lại', ha='center', fontsize=6.3, color=MUTED)
    save(fig, 'results/fig_e21_protocol')


def pipeline():
    """Kiến trúc hiện thực: năm khối của hệ thống, từ dữ liệu thô tới báo cáo và công cụ minh hoạ."""
    apply_theme()
    fig = plt.figure(figsize=(6.3, 2.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis('off')
    w, h, y = 1.72, 1.55, 2.15
    xs = [0.12, 2.13, 4.14, 6.15, 8.16]
    spec = [('Dữ liệu thô', '5 bộ công khai\nCSV, MATLAB', C_OUT, '#9aa5ad'),
            ('Khối dữ liệu', '16 đặc trưng sạc\nlọc ngoại lai nhân quả\nchuẩn hoá, bộ đệm', C_IN, E_IN),
            ('Khối thí nghiệm', 'danh sách lượt chạy\nsong song, chạy tiếp được\nmột tệp kết quả mỗi lượt', C_NET, E_NET),
            ('Khối phân tích', 'MAE, RMSE, MAPE, $R^2$\nkiểm định ghép cặp\nhình, bảng', C_PHY, E_PHY),
            ('Báo cáo, minh hoạ', 'báo cáo sinh tự động\ncông cụ dòng lệnh:\nSOH, dự báo, RUL', C_LOSS, E_LOSS)]
    B = [box(ax, x, y, w, h, t, sub, fc=fc, ec=ec, fs=7.4) for x, (t, sub, fc, ec) in zip(xs, spec)]
    for a, b in zip(B[:-1], B[1:]):
        arrow(ax, right(a), left(b))
    m = box(ax, 1.5, 0.2, 7.0, 1.35, 'Khối mô hình',
            'MLP, CNN, LSTM, Transformer, PINN-semi, PINN-dự báo (PyTorch)\nRidge, GPR, GBDT (scikit-learn)\n'
            'bộ lọc quỹ đạo, khoảng phù hợp, bộ lọc Kalman', fc=C_NET, ec=E_NET, fs=7.4)
    arrow(ax, (B[2][0] + B[2][2] / 2, m[1] + m[3]), bottom(B[2]))
    save(fig, 'results/fig_thesis_pipeline')


if __name__ == '__main__':
    arch()
    protocol()
    pipeline()
