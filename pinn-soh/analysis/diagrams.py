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


ARCH_TEXT = {
    'vi': dict(data='DỮ LIỆU', model='MÔ HÌNH', loss='HÀM MẤT MÁT', lab='Cell có nhãn', unl='Mọi cell train',
               only='chỉ', F='Mạng nghiệm', G='Mạng', rate='tốc độ suy giảm ≥ 0', range='SOH hợp lý',
               upd='cập nhật đồng thời', fs=dict(F=7.6, G=7.4, r=7.4)),
    'en': dict(data='DATA', model='MODEL', loss='LOSSES', lab='Labelled cells', unl='All training cells', only='only',
               F='Solution network', G='Rate network', rate='degradation rate ≥ 0', range='plausible SOH',
               upd='joint update of', fs=dict(F=6.6, G=6.4, r=6.6)),
}


def arch(lang='vi', out='results/fig_thesis_arch', font='TeX Gyre Heros'):
    T = ARCH_TEXT[lang]
    apply_theme(font=font)
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis('off')
    region(ax, 0.05, 0.85, 2.25, 4.2, T['data'])
    region(ax, 2.6, 0.85, 4.0, 4.2, T['model'])
    region(ax, 7.05, 0.85, 2.9, 4.2, T['loss'])
    lab = box(ax, 0.25, 3.3, 1.85, 0.85, T['lab'], '$(\\mathbf{x}, t, y)$')
    unl = box(ax, 0.25, 1.2, 1.85, 0.85, T['unl'], T['only'] + ' $(\\mathbf{x}, t)$')
    F = box(ax, 2.85, 3.3, 2.15, 0.85, T['F'] + ' $F_\\varphi$', '17 → 128 → 128 → 64 → 1', C_NET, E_NET, fs=T['fs']['F'])
    u = box(ax, 5.3, 3.3, 1.1, 0.85, 'SOH $u$', '$F_\\varphi(\\tilde{\\mathbf{x}}, t)$', C_OUT, '#9aa5ad')
    G = box(ax, 2.85, 1.2, 1.55, 0.85, T['G'] + ' $G_\\theta$', '16 → 32 → 32 → 1', C_PHY, E_PHY, fs=T['fs']['G'])
    r = box(ax, 4.65, 1.2, 1.75, 0.85, '$r = k(\\mathbf{x})\\,g(u)\\,A(T)$', T['rate'], C_PHY, E_PHY, fs=T['fs']['r'])
    LX, LW = 7.3, 2.45
    ld = box(ax, LX, 3.75, LW, 0.6, '$\\mathcal{L}_{data}$   $(u-y)^2$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    lm = box(ax, LX, 2.95, LW, 0.6, '$\\mathcal{L}_{mono}$   ReLU$(u_2-u_1-\\varepsilon)$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    lr = box(ax, LX, 2.15, LW, 0.6, '$\\mathcal{L}_{range}$   ' + T['range'], None, C_LOSS, E_LOSS, fs=7.2,
             weight='normal')
    lo = box(ax, LX, 1.2, LW, 0.75, '$\\mathcal{L}_{ode}$   $(\\partial_t u + r)^2$', None, C_LOSS, E_LOSS, fs=7.2, weight='normal')
    box(ax, 0.05, 0.05, 9.9, 0.6,
        '$\\mathcal{L} = \\mathcal{L}_{data} + w(s)\\,(\\alpha\\mathcal{L}_{ode} + \\beta\\mathcal{L}_{mono} + '
        '\\gamma\\mathcal{L}_{range})$     ' + T['upd'] + ' $\\varphi, \\theta, \\lambda, E_a$', None, '#ffffff', '#9aa5ad',
        fs=7.6, weight='normal')
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
    save(fig, out)


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


def pipeline(out='results/fig_thesis_pipeline'):
    """Kiến trúc hiện thực: năm khối của hệ thống, từ dữ liệu tới báo cáo và công cụ minh hoạ."""
    apply_theme()
    fig = plt.figure(figsize=(6.3, 2.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis('off')
    w, h, y = 1.72, 1.55, 2.15
    xs = [0.12, 2.13, 4.14, 6.15, 8.16]
    spec = [('Dữ liệu', '4 bộ công khai\n387 cell, 16 đặc trưng\nsạc mỗi chu kỳ', C_OUT, '#9aa5ad'),
            ('Khối dữ liệu', '16 đặc trưng sạc\nlọc ngoại lai nhân quả\nchuẩn hoá, bộ đệm', C_IN, E_IN),
            ('Khối thí nghiệm', 'danh sách lượt chạy\nsong song, chạy tiếp được\nmột tệp kết quả mỗi lượt', C_NET, E_NET),
            ('Khối phân tích', 'MAE, RMSE, MAPE, $R^2$\nkiểm định ghép cặp\nhình, bảng', C_PHY, E_PHY),
            ('Báo cáo, minh hoạ', 'báo cáo, bản thảo bài báo\nsinh tự động; công cụ\ndòng lệnh: SOH, RUL', C_LOSS, E_LOSS)]
    B = [box(ax, x, y, w, h, t, sub, fc=fc, ec=ec, fs=7.4) for x, (t, sub, fc, ec) in zip(xs, spec)]
    for a, b in zip(B[:-1], B[1:]):
        arrow(ax, right(a), left(b))
    m = box(ax, 1.5, 0.2, 7.0, 1.35, 'Khối mô hình',
            'MLP, CNN, LSTM, Transformer, PINN-semi, PINN-dự báo (PyTorch); mã gốc PINN4SOH\nRidge, GPR, GBDT (scikit-learn)\n'
            'bộ lọc quỹ đạo, khoảng phù hợp, bộ lọc Kalman', fc=C_NET, ec=E_NET, fs=7.4)
    arrow(ax, (B[2][0] + B[2][2] / 2, m[1] + m[3]), bottom(B[2]))
    save(fig, out)


def forecast_arch(out='results/fig_forecast_arch', font='TeX Gyre Heros'):
    """Mô hình dự báo (E22, pinnsoh/forecast.py): tại gốc k, đặc trưng gốc z_k -> GBDT theo tầm và PINN-dự báo
    (a, λ của nghiệm đóng) -> kết hợp -> SOH, dung lượng, RUL và khoảng 90 %."""
    apply_theme(font=font)
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis('off')
    GREY = '#6b7680'
    region(ax, 0.05, 1.2, 2.0, 4.1, 'ĐẦU VÀO TẠI GỐC k')
    region(ax, 2.2, 1.2, 5.45, 4.1, 'MÔ HÌNH DỰ BÁO')
    region(ax, 7.8, 1.2, 2.15, 4.1, 'ĐẦU RA')
    hs = box(ax, 0.18, 3.55, 1.74, 1.05, 'Lịch sử SOH tới k', 'đo được, hoặc ước\nlượng PINN-lai khi\ncell không được đo',
             fs=7.0)
    ch = box(ax, 0.18, 1.75, 1.74, 1.05, 'Đặc trưng sạc tới k', '16 đặc trưng\nmỗi chu kỳ,\nđã chuẩn hoá', fs=7.0)
    z = box(ax, 2.38, 2.1, 1.55, 2.15, 'Đặc trưng gốc $\\mathbf{z}_k$',
            '23 giá trị: mức $\\ell_k$,\nđộ dốc 30 và 100\nhàng gần nhất,\n$\\ell_0$, $\\ell_0-\\ell_k$, $k$, $1/T$,\n'
            'đặc trưng sạc\n10 hàng gần nhất', C_OUT, '#9aa5ad', fs=7.0)
    gb = box(ax, 4.2, 4.05, 2.15, 0.95, 'GBDT theo tầm', '$(\\mathbf{z}_k, h) \\to \\ell_k - \\mathrm{SOH}(k+h)$\n'
             'đơn điệu tăng theo $h$', C_NET, E_NET, fs=7.0)
    pn = box(ax, 4.2, 2.75, 2.15, 0.95, 'PINN-dự báo (3 mạng)', '$\\mathbf{z}_k \\to 64 \\to 64 \\to (a \\geq 0,\\ \\lambda)$',
             C_PHY, E_PHY, fs=7.0)
    cf = box(ax, 4.2, 1.42, 2.15, 1.0, 'Nghiệm đóng động học xám',
             '$\\hat y(k+h) = \\ell_k + \\ln(1-\\lambda a h)/\\lambda$\nkhông tăng theo $h$', C_PHY, E_PHY, fs=7.0)
    kh = box(ax, 6.6, 2.6, 0.92, 1.25, 'Kết hợp', '½ GBDT +\n½ PINN-dự báo', C_LOSS, E_LOSS, fs=7.0)
    o1 = box(ax, 7.95, 3.92, 1.85, 1.12, 'SOH và dung lượng', '$\\hat y(k+h)$ với mọi $h$\n$\\hat Q = \\hat y$ × dung lượng\ndanh định',
             C_OUT, '#9aa5ad', fs=7.0)
    o2 = box(ax, 7.95, 2.75, 1.85, 0.95, 'RUL', '$h$ đầu tiên có $\\hat y \\leq \\tau$', C_OUT, '#9aa5ad', fs=7.0)
    o3 = box(ax, 7.95, 1.42, 1.85, 1.0, 'Khoảng 90 %', 'phân vị 5 % và 95 % của\nsai số trên cell validation,\ntheo $h/L$',
             C_OUT, '#9aa5ad', fs=7.0)
    # đầu vào -> z
    poly(ax, [(1.92, 4.075), (2.15, 4.075), (2.15, 3.55), (2.38, 3.55)], GREY)
    poly(ax, [(1.92, 2.275), (2.15, 2.275), (2.15, 2.8), (2.38, 2.8)], GREY)
    # z -> hai nhánh
    poly(ax, [(3.93, 3.75), (4.06, 3.75), (4.06, 4.52), (4.2, 4.52)], E_NET)
    arrow(ax, (3.93, 3.22), (4.2, 3.22), color=E_PHY)
    arrow(ax, bottom(pn), top(cf), color=E_PHY)
    # hai nhánh -> kết hợp
    poly(ax, [(6.35, 4.52), (6.47, 4.52), (6.47, 3.55), (6.6, 3.55)], E_NET)
    poly(ax, [(6.35, 1.92), (6.47, 1.92), (6.47, 2.9), (6.6, 2.9)], E_PHY)
    # kết hợp -> đầu ra
    poly(ax, [(7.52, 3.45), (7.72, 3.45), (7.72, 4.47), (7.95, 4.47)], E_LOSS)
    arrow(ax, (7.52, 3.22), (7.95, 3.22), color=E_LOSS)
    poly(ax, [(7.52, 3.0), (7.72, 3.0), (7.72, 1.92), (7.95, 1.92)], E_LOSS)
    box(ax, 0.05, 0.08, 9.9, 0.92, 'Huấn luyện trên cell train, lịch sử đo thật',
        'mỗi cell 40 gốc × 40 tầm; GBDT chọn số cây, PINN-dự báo dừng sớm theo MAE trên cell validation;'
        ' cell test không dùng khi học', '#ffffff', '#9aa5ad', fs=7.0)
    save(fig, out)


C_LIFE, E_LIFE = '#efe9fb', '#7b5cd6'


def hipinn_arch(out='results/fig_hipinn_arch', font='TeX Gyre Heros'):
    """Mô hình HiPINN (E33, pinnsoh/forecast_combo.py): đặc trưng gốc và đặc trưng lịch sử tại gốc k -> ba nhánh (GBDT
    lịch sử và nhánh theo đời nhận cả hai, PINN-dự báo chỉ nhận đặc trưng gốc) -> trung vị -> SOH, dung lượng, RUL,
    khoảng 90 %."""
    apply_theme(font=font)
    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 6.25); ax.axis('off')
    GREY = '#6b7680'
    region(ax, 0.05, 1.2, 1.95, 4.95, 'ĐẦU VÀO TẠI GỐC k')
    region(ax, 2.12, 1.2, 5.83, 4.95, 'HiPINN')
    region(ax, 8.07, 1.2, 1.88, 4.95, 'ĐẦU RA')
    box(ax, 0.17, 3.85, 1.71, 1.25, 'Lịch sử SOH tới k', 'đo được, hoặc ước\nlượng PINN-lai khi\ncell không được đo',
        fs=7.0)
    box(ax, 0.17, 2.0, 1.71, 1.25, 'Đặc trưng sạc tới k', '16 đặc trưng\nmỗi chu kỳ,\nđã chuẩn hoá', fs=7.0)
    box(ax, 2.27, 1.55, 1.5, 1.55, 'Đặc trưng gốc $\\mathbf{z}_k$',
        '23 giá trị: mức $\\ell_k$,\nđộ dốc 30, 100 hàng,\n$k$, $1/T$, đặc trưng\nsạc gần nhất', C_OUT, '#9aa5ad',
        fs=6.8)
    box(ax, 2.27, 3.4, 1.5, 2.15, 'Đặc trưng lịch sử',
        'SOH tại $k$; cửa sổ\n5, 10, 30, 100, 200\nhàng: trung bình,\nđộ lệch chuẩn, độ dốc;\nđộ trễ 1–63 hàng',
        C_OUT, '#9aa5ad', fs=6.8)
    box(ax, 4.05, 4.45, 2.35, 1.1, 'GBDT lịch sử', '$\\to \\ell_k - \\mathrm{SOH}(k+h)$ theo $h$\n6 cấu hình × 3 số cây',
        C_NET, E_NET, fs=7.0)
    box(ax, 4.05, 2.9, 2.35, 1.3, 'Nhánh theo đời', 'GBDT đời còn lại $\\to \\hat N$\nkhuôn quỹ đạo của K cell\n'
        'train theo $u = c/\\hat N$', C_LIFE, E_LIFE, fs=7.0)
    box(ax, 4.05, 1.4, 2.35, 1.2, 'PINN-dự báo', '$\\mathbf{z}_k \\to (a \\geq 0,\\ \\lambda)$, nghiệm đóng\n'
        '$\\ell_k + \\ln(1-\\lambda a h)/\\lambda$', C_PHY, E_PHY, fs=7.0)
    box(ax, 6.7, 2.7, 1.1, 1.75, 'Trung vị', 'của ba nhánh\ntại mỗi gốc\nvà tầm $h$', C_LOSS, E_LOSS, fs=7.0)
    box(ax, 8.2, 4.3, 1.62, 1.25, 'SOH, dung lượng', '$\\hat y(k+h)$ mọi $h$;\n$\\hat Q = \\hat y\\,\\times$\ndanh định',
        C_OUT, '#9aa5ad', fs=7.0)
    box(ax, 8.2, 3.0, 1.62, 1.0, 'RUL', '$h$ đầu tiên có\n$\\hat y \\leq \\tau$', C_OUT, '#9aa5ad', fs=7.0)
    box(ax, 8.2, 1.45, 1.62, 1.25, 'Khoảng 90 %', '$\\hat y \\pm q\\sqrt{1+h/L}$,\n$q$ từ nửa cell\nvalidation còn lại',
        C_OUT, '#9aa5ad', fs=7.0)
    # đầu vào -> đặc trưng
    poly(ax, [(1.88, 4.475), (2.27, 4.475)], GREY)
    poly(ax, [(1.88, 2.625), (2.06, 2.625), (2.06, 2.33), (2.27, 2.33)], GREY)
    poly(ax, [(1.88, 4.2), (2.08, 4.2), (2.08, 2.75), (2.27, 2.75)], GREY)
    # đặc trưng gốc và đặc trưng lịch sử -> thanh góp -> GBDT lịch sử, nhánh theo đời; đặc trưng gốc -> PINN-dự báo
    ax.plot([3.77, 3.9], [4.475, 4.475], color=GREY, lw=0.9)
    ax.plot([3.77, 3.9], [2.85, 2.85], color=GREY, lw=0.9)
    ax.plot([3.9, 3.9], [2.85, 5.0], color=GREY, lw=0.9)
    arrow(ax, (3.9, 5.0), (4.05, 5.0), color=E_NET)
    arrow(ax, (3.9, 3.55), (4.05, 3.55), color=E_LIFE)
    arrow(ax, (3.77, 2.0), (4.05, 2.0), color=E_PHY)
    # ba nhánh -> trung vị
    poly(ax, [(6.4, 5.0), (6.55, 5.0), (6.55, 4.2), (6.7, 4.2)], E_NET)
    arrow(ax, (6.4, 3.55), (6.7, 3.55), color=E_LIFE)
    poly(ax, [(6.4, 2.0), (6.55, 2.0), (6.55, 2.95), (6.7, 2.95)], E_PHY)
    # trung vị -> đầu ra
    poly(ax, [(7.8, 4.1), (8.0, 4.1), (8.0, 4.92), (8.2, 4.92)], E_LOSS)
    arrow(ax, (7.8, 3.5), (8.2, 3.5), color=E_LOSS)
    poly(ax, [(7.8, 2.95), (8.0, 2.95), (8.0, 2.07), (8.2, 2.07)], E_LOSS)
    box(ax, 0.05, 0.08, 9.9, 0.95, 'Huấn luyện trên cell train, lịch sử đo thật',
        'mỗi nhánh học trên các cặp (gốc, tầm) của cell train; cấu hình và số cây của từng nhánh chọn theo MAE trên một '
        'nửa cell validation;\nnửa còn lại để hiệu chỉnh khoảng; trung vị không có tham số cần chọn; cell test không '
        'dùng khi học', '#ffffff', '#9aa5ad', fs=7.0)
    save(fig, out)


if __name__ == '__main__':
    arch()
    protocol()
    pipeline()
    forecast_arch()
    hipinn_arch()
