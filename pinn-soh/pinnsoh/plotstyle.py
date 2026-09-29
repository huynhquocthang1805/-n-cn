"""
Bộ khuôn vẽ hình cho báo cáo khoa học — lấy nguyên tắc từ Flint (Microsoft Research,
arXiv 2607.20775) và bảng màu đã kiểm định độ phân biệt cho người mù màu.

Bốn quy tắc của Flint được áp dụng cụ thể ở đây:

1. NGỮ NGHĨA QUYẾT ĐỊNH THANG ĐO, không phải mặc định của thư viện.
   MAE là đại lượng cộng tính, có số 0 nghĩa lý  -> trục bắt đầu từ 0.
   Tỉ số PINN/MLP là đại lượng CÓ DẤU quanh 1.0  -> thang PHÂN KỲ, midpoint = 1.
   Trước đây mọi hình đều dùng thang tuần tự, làm mất thông tin "tốt hơn hay tệ hơn".

2. BANKING TO 45° cho tỉ lệ khung của biểu đồ đường (Cleveland).
   Tỉ lệ khung được tính sao cho trung vị độ dốc các đoạn xấp xỉ 45°, là góc mắt
   người phân biệt độ dốc tốt nhất. Không còn đặt figsize bằng tay.

3. MIỀN CÓ THỨ TỰ NỘI TẠI, không sắp theo bảng chữ cái.
   Bốn bộ dữ liệu xếp theo hoá học (NCM -> NCA/NCM -> LFP -> LFP), không phải A-Z.

4. GÁN NHÃN TRỰC TIẾP thay cho hộp chú giải khi số chuỗi ít.
   Mắt không phải nhảy qua lại giữa hộp màu và đường.

Mọi hình xuất đồng thời .pdf (vector, cho LaTeX) và .png (cho HTML/Word).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --------------------------------------------------------------------------- bảng màu
# Slot phân loại theo thứ tự cố định, không bao giờ xoay vòng.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MODEL_COLOR = {'mlp': SERIES[0], 'pinn_sup': SERIES[1], 'pinn_semi': SERIES[2],
               'pinn_bb': SERIES[3], 'truth': '#3f4a52'}
MODEL_LABEL = {'mlp': 'MLP (chỉ dữ liệu)', 'pinn_sup': 'PINN-sup', 'pinn_semi': 'PINN-semi',
               'pinn_bb': 'PINN động học đen', 'truth': 'thực đo'}

INK, INK2, MUTED, GRID = '#1b2330', '#4a5563', '#78848d', '#e7ebe9'

# Miền có thứ tự nội tại: theo hoá học, không theo bảng chữ cái (Flint, quy tắc 3)
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}

# Thang phân kỳ cho tỉ số quanh 1.0: xanh lục = tốt hơn, cam = tệ hơn, xám ở đúng 1.0
RATIO_CMAP = LinearSegmentedColormap.from_list(
    'ratio', ['#0d6e56', '#4fb499', '#b9ded3', '#eceeed', '#f7cdb4', '#e08a5a', '#b8531f'])
SEQ_CMAP = LinearSegmentedColormap.from_list(
    'seq', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def ratio_norm(vmin=None, vmax=None, center=1.0):
    """Thang phân kỳ có tâm tại 1.0 — 'tốt hơn baseline' và 'tệ hơn' đọc được ngay."""
    lo = min(vmin, center - 1e-3) if vmin is not None else center - 0.5
    hi = max(vmax, center + 1e-3) if vmax is not None else center + 0.5
    return TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)


# --------------------------------------------------------------------------- khuôn chung
def apply_theme(base=8.5, font='TeX Gyre Heros'):
    """Kiểu trình bày tối giản cho hình in: chỉ trục trái và dưới, lưới ngang mảnh."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font, 'DejaVu Sans', 'Liberation Sans'],
        # công thức trong nhãn trục dùng ĐÚNG phông của phần chữ, không nhảy sang DejaVu
        'mathtext.fontset': 'custom', 'mathtext.default': 'regular',
        'mathtext.rm': font, 'mathtext.it': f'{font}:italic', 'mathtext.bf': f'{font}:bold',
        'mathtext.sf': font, 'mathtext.cal': font, 'mathtext.tt': font,
        'font.size': base,
        'axes.titlesize': base + 1, 'axes.labelsize': base,
        'xtick.labelsize': base - 0.5, 'ytick.labelsize': base - 0.5,
        'legend.fontsize': base - 0.5,
        'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': INK2, 'ytick.color': INK2,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.color': GRID, 'grid.linewidth': 0.55, 'axes.axisbelow': True,
        'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.columnspacing': 1.2,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'figure.dpi': 130, 'savefig.dpi': 400,
        'lines.solid_capstyle': 'round', 'lines.linewidth': 1.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42,      # phông nhúng dạng TrueType, không vẽ đường
    })


def banked_height(x, ys, width, lo=0.62, hi=1.15):
    """Chiều cao khung theo nguyên tắc banking to 45° (Flint, quy tắc 2).

    Chọn tỉ lệ khung sao cho TRUNG VỊ độ dốc các đoạn đường xấp xỉ 45°, tức góc mà
    mắt phân biệt sai khác độ dốc chính xác nhất. Kẹp trong [lo, hi] để hình không
    bẹt hay cao quá so với khổ trang."""
    x = np.asarray(x, dtype=float)
    slopes = []
    for y in ys:
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2:
            continue
        dx = np.diff(x[m]); dy = np.diff(y[m])
        rx = np.ptp(x[m]); ry = np.ptp(y[m])
        if rx <= 0 or ry <= 0:
            continue
        slopes += list(np.abs((dy / ry) / (dx / rx)))
    if not slopes:
        return width * 0.75
    ar = float(np.median(slopes))                # tỉ lệ cao/rộng để trung vị dốc = 45°
    return width * float(np.clip(ar, lo, hi))


def direct_label(ax, x, y, text, color, dx=4, dy=0, size=None, weight=500, ha='left', va='center'):
    """Gán nhãn ngay cạnh đường (Flint, quy tắc 4) — mắt không phải tra hộp chú giải."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                color=color, fontsize=size or plt.rcParams['font.size'] - 0.5,
                fontweight=weight, ha=ha, va=va, annotation_clip=False)


def panel_title(ax, text, sub=None, pad=6):
    """Tiêu đề panel; phụ đề đặt NGAY SAU tiêu đề trên cùng dòng, không chồng lên nhau."""
    t = ax.set_title(text, loc='left', fontweight='bold', pad=pad)
    if sub:
        ax.annotate('  ' + sub, xycoords=t, xy=(1, 0), xytext=(2, 0),
                    textcoords='offset points', color=MUTED,
                    fontsize=plt.rcParams['font.size'] - 1.5, va='bottom', ha='left')


def spread_labels(ax, items, min_gap_px=10):
    """Tách các nhãn trực tiếp theo trục dọc để chúng không đè nhau.

    items: list các (y_dữ_liệu, text, color). Trả về list (y_đã_dịch, text, color)."""
    if not items:
        return []
    inv = ax.transData.inverted()
    to_px = lambda y: ax.transData.transform((0, y))[1]
    to_dat = lambda py: inv.transform((0, py))[1]
    arr = sorted(((to_px(y), t, c) for y, t, c in items), key=lambda r: r[0])
    out = [list(arr[0])]
    for py, t, c in arr[1:]:
        if py - out[-1][0] < min_gap_px:
            py = out[-1][0] + min_gap_px
        out.append([py, t, c])
    # căn lại vào giữa để cụm nhãn không trôi lên trên
    shift = (arr[0][0] + arr[-1][0]) / 2 - (out[0][0] + out[-1][0]) / 2
    return [(to_dat(py + shift), t, c) for py, t, c in out]


def tidy(ax, ynonneg=True, nbins=4):
    """Trục thưa tick, bắt đầu từ 0 khi đại lượng có số 0 nghĩa lý (Flint, quy tắc 1)."""
    ax.locator_params(axis='y', nbins=nbins)
    if ynonneg:
        ax.set_ylim(bottom=0)
    ax.tick_params(length=2.6, pad=2)


# ------------------------------------------------------- trục MAE dạng nhân (Flint, quy tắc 1)
MAE_TICKS = (4, 6, 10, 15, 25, 40)


def log_mae_axis(ax, ticks=MAE_TICKS, lo=None, hi=None):
    """Trục MAE theo thang LOG, đơn vị 10^-3 SOH.

    Vì sao không lấy mốc 0: mọi kết luận của báo cáo đều phát biểu dưới dạng TỈ SỐ
    ('PINN bằng 0,72 lần sai số MLP'), tức đại lượng NHÂN TÍNH. Trên thang log, cùng
    một khoảng cách dọc luôn là cùng một tỉ số, nên mắt đọc đúng thứ mà văn bản nói.
    Thang tuyến tính neo ở 0 làm bốn bộ dữ liệu lệch nhau 4 lần không thể đặt chung
    một trục, và dồn toàn bộ đường vào một dải hẹp. Trục log dùng chung cho cả bốn
    panel để so sánh chéo được ngay.
    """
    ax.set_yscale('log')
    ax.set_yticks(list(ticks))
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if lo is not None and hi is not None:
        ax.set_ylim(lo, hi)
    ax.tick_params(length=2.6, pad=2)


def guide_vline(ax, x, label=None, color=None):
    """Vạch mốc mờ đánh dấu điểm vận hành được bàn trong bài (30 % cell mang nhãn)."""
    ax.axvline(x, color=color or '#d7dedb', lw=0.8, ls=(0, (2.2, 2.2)), zorder=0)
    if label:
        ax.annotate(label, xy=(x, 0.0), xycoords=('data', 'axes fraction'),
                    xytext=(3, 3), textcoords='offset points',
                    color=MUTED, fontsize=plt.rcParams['font.size'] - 2,
                    ha='left', va='bottom')


def figure_note(fig, text, y=-0.02):
    fig.text(0.005, y, text, ha='left', va='top', color=MUTED,
             fontsize=plt.rcParams['font.size'] - 1.5)


def save(fig, path_noext, also_pdf=True, dpi=None):
    """Lưu .png cho HTML/Word và .pdf vector cho LaTeX."""
    fig.savefig(f'{path_noext}.png', dpi=dpi or plt.rcParams['savefig.dpi'])
    if also_pdf:
        fig.savefig(f'{path_noext}.pdf')
    plt.close(fig)
