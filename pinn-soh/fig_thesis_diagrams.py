"""
Sơ đồ kiến trúc mô hình cho báo cáo đồ án.

    python fig_thesis_diagrams.py     # -> results/fig_thesis_arch.png|pdf
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pinnsoh.plotstyle import apply_theme, save

INK, MUTED = '#1b2330', '#4a5563'
C_IN, C_NET, C_PHY, C_LOSS, C_OUT = '#eef3fb', '#dbe8f8', '#e3f4ec', '#fbeee6', '#f2f2f2'
E_IN, E_NET, E_PHY, E_LOSS = '#9fb7d6', '#2a78d6', '#1baf7a', '#eb6834'


def box(ax, x, y, w, h, title, sub=None, fc=C_IN, ec=E_IN, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec=ec, lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center', fontsize=fs, color=INK, weight='bold')
        ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center', fontsize=fs - 0.8, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs, color=INK, weight='bold')
    return (x, y, w, h)


def arrow(ax, p, q, color='#6b7680', style='-|>', ls='-'):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=0.9, ls=ls,
                                                    shrinkA=0, shrinkB=0, mutation_scale=8))


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


def arch():
    apply_theme()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis('off')
    # hàng 1: dữ liệu có nhãn -> mạng nghiệm -> SOH
    lab = box(ax, 0.05, 4.75, 1.85, 0.95, 'Cell có nhãn', '$(\\mathbf{x}, t, y)$')
    sol = box(ax, 2.45, 4.75, 3.0, 0.95, 'Mạng nghiệm $F_\\varphi$', '17 → 128 → 128 → 64 → 1', C_NET, E_NET)
    u = box(ax, 6.05, 4.75, 1.9, 0.95, 'SOH dự đoán', '$u = F_\\varphi(\\tilde{\\mathbf{x}}, t)$', C_OUT, '#9aa5ad')
    unl = box(ax, 0.05, 2.75, 1.85, 0.95, 'Mọi cell train', 'chỉ $(\\mathbf{x}, t)$')
    # hàng 2: bốn hàm mất mát
    W, G0 = 1.78, 0.12
    xs = [2.45 + i * (W + G0) for i in range(4)]
    ld = box(ax, xs[0], 2.75, W, 0.95, '$\\mathcal{L}_{data}$', '$(u-y)^2$, cell có nhãn', C_LOSS, E_LOSS, fs=7.4)
    lo = box(ax, xs[1], 2.75, W, 0.95, '$\\mathcal{L}_{ode}$', '$(\\partial_t u + r)^2$', C_LOSS, E_LOSS, fs=7.4)
    lm = box(ax, xs[2], 2.75, W, 0.95, '$\\mathcal{L}_{mono}$', 'ReLU$(u_2-u_1-\\varepsilon)$', C_LOSS, E_LOSS, fs=7.4)
    lr = box(ax, xs[3], 2.75, W, 0.95, '$\\mathcal{L}_{range}$', 'SOH trong miền hợp lý', C_LOSS, E_LOSS, fs=7.4)
    # hàng 3: động học xám
    gb = box(ax, 2.45, 0.95, 1.55, 0.9, 'Mạng $G_\\theta$', '16 → 32 → 32 → 1', C_PHY, E_PHY, fs=7.4)
    kb = box(ax, 4.2, 0.95, 1.3, 0.9, '$k(\\mathbf{x})$', 'softplus ≥ 0', C_PHY, E_PHY, fs=7.4)
    gu = box(ax, 5.7, 0.95, 1.3, 0.9, '$\\times\\, g(u)$', '$e^{\\lambda(1-u)}$', C_PHY, E_PHY, fs=7.4)
    at = box(ax, 7.2, 0.95, 1.3, 0.9, '$\\times\\, A(T)$', 'Arrhenius', C_PHY, E_PHY, fs=7.4)
    rb = box(ax, 8.7, 0.95, 1.25, 0.9, '$r \\geq 0$', 'tốc độ suy giảm', C_PHY, E_PHY, fs=7.4)
    box(ax, 0.05, 0.05, 9.9, 0.55,
        '$\\mathcal{L} = \\mathcal{L}_{data} + w(s)\\,(\\alpha\\mathcal{L}_{ode} + \\beta\\mathcal{L}_{mono} + '
        '\\gamma\\mathcal{L}_{range})$   —   cập nhật đồng thời $\\varphi, \\theta, \\lambda, E_a$', None, '#ffffff', '#9aa5ad', fs=7.6)
    # mũi tên
    arrow(ax, right(lab), left(sol)); arrow(ax, right(sol), left(u))
    arrow(ax, (1.9, 3.5), (2.45, 4.85), color=E_NET)                      # cell không nhãn -> F
    arrow(ax, (1.0, 2.75), (2.45, 1.4), color=E_PHY)                      # cell không nhãn -> G
    arrow(ax, (1.7, 4.75), (2.75, 3.7), color=E_LOSS)                     # nhãn y -> L_data
    for b, x0 in zip((ld, lo, lm, lr), (6.25, 6.6, 7.15, 7.7)):
        arrow(ax, (x0, 4.75), top(b), color=E_NET)
    for a, b in ((gb, kb), (kb, gu), (gu, at), (at, rb)):
        arrow(ax, right(a), left(b), color=E_PHY)
    arrow(ax, top(rb), bottom(lo), color=E_PHY)                            # r -> L_ode
    save(fig, 'results/fig_thesis_arch')


if __name__ == '__main__':
    arch()
