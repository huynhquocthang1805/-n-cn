"""
Sơ đồ kiến trúc mô hình cho báo cáo đồ án.

    python report/diagrams.py         # -> results/fig_thesis_arch.png|pdf
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


if __name__ == '__main__':
    arch()
