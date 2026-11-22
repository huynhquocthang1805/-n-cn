"""
Dựng báo cáo đồ án chuyên ngành (tiếng Việt, theo mẫu hcmut-thesis của Khoa KH&KT Máy tính): bảng và đoạn văn có số
(report/generated/*.tex), hình (report/graphics/), rồi biên dịch report/main.tex bằng pdfLaTeX và BibTeX (latexmk -pdf)
-> report/bao-cao-do-an.pdf. Thư mục report/ tự đủ: các tệp sinh ra được đưa lên git, nên main.tex biên dịch được ngay
mà không cần chạy lại gì.

    python report/build.py            # hoặc: python run.py report
    python report/build.py --no-pdf   # chỉ bảng, đoạn văn và hình

Nguồn số liệu, không số nào gõ tay vào bảng:
  * bảng của báo cáo đồ án đã nộp (archive/thesis/do-an/main.tex, đọc bằng paper/thesis_tables.py) cho các thí nghiệm
    trên bốn bộ dữ liệu (bỏ mọi phần của bộ NASA); các bảng đó sinh từ results/ của lần chạy ở commit 2d66f75;
  * benchmark/results/ (benchmark/compare.py): chạy lại PINN4SOH và so trên đúng cách chia của bài;
  * results/features_*.csv (analysis/features.py): phân tích đặc trưng; một bản sao ở report/generated/data/;
  * results/thesis_leakage.csv (analysis/analyze_thesis.py): kiểm toán rò rỉ chạy lại; bản sao ở report/generated/data/;
  * số lượt chạy của từng thí nghiệm: pinnsoh/experiments.py;
  * dự báo SOH, dung lượng, RUL với HiGBDT và HiPINN (E33): runs/E33 -> analysis/analyze_e33.py,
    analysis/figures_report.py; bản sao bảng ở report/generated/data/e33_*.csv, hình ở report/graphics/fig_e33_*.pdf;
  * HiGBDT cho bài toán cập nhật (E34): runs/E34 -> analysis/analyze_e34.py; bản sao bảng ở
    report/generated/data/e34_*.csv.
Các con số trong lời văn của Chương 5 (kết quả từ báo cáo đã nộp) chép từ lời văn của báo cáo đó, trừ mục dự báo
(sinh từ E33). Mục dự báo E22 của báo cáo đã nộp không còn trong báo cáo: E33 thay nó, với PINN-GBDT của E22 học lại
làm mốc so sánh.
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'paper'), _os.path.join(_ROOT, 'analysis')]
_os.chdir(_ROOT)
import argparse, glob, json, os, re, shutil, subprocess
import numpy as np
import pandas as pd

from thesis_tables import THESIS, table
from pinnsoh.paper import PAPER_MAE

REP = 'report'
GEN = os.path.join(REP, 'generated')
DATA = os.path.join(GEN, 'data')                 # bản sao các bảng kết quả nhỏ (results/ không lên git)
FIG = os.path.join(REP, 'graphics')
BENCH = 'benchmark/results'
OLD_FIG = 'archive/thesis/do-an/hinh'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
ROWS = [(ds, 'full') for ds in DS] + [(ds, f'n{k}') for ds in ('XJTU', 'HUST') for k in (1, 2, 3, 4)]


# --------------------------------------------------------------------------- định dạng số kiểu Việt Nam
def v(x, n=2):
    """0.845 -> '0{,}85' (dấu phẩy thập phân, dùng được cả trong và ngoài chế độ toán); số âm dùng dấu trừ −."""
    return f'{x:.{n}f}'.replace('.', '{,}').replace('-', '−')


def rng(vals, n=2):
    vals = [x for x in vals if x is not None and pd.notna(x)]
    if not vals:
        return '--'
    lo, hi = v(min(vals), n), v(max(vals), n)
    return lo if lo == hi else (f'{lo} đến {hi}' if min(vals) < 0 else f'{lo}--{hi}')


def tex_esc(t):
    """Tên cell trong lời văn: thoát các ký tự đặc biệt của LaTeX."""
    return ''.join('\\' + c if c in '_#%&$' else c for c in str(t))


def pct(x, n=0):
    return v(100 * x, n) + r'\,\%'


def thou(n):
    return f'{int(n):,}'.replace(',', '.')


# Thuật ngữ thống nhất cho mọi văn bản sinh tự động (và bảng chép từ báo cáo đã nộp): tập huấn luyện, tập xác thực,
# tập kiểm tra, bộ chuẩn hoá, hàm mất mát; "có ý nghĩa" luôn là "có ý nghĩa thống kê". Thứ tự: cụm dài trước.
TERMS = [
    ('hai nửa cell validation', 'hai nửa tập xác thực'), ('nửa validation', 'nửa tập xác thực'),
    ('cell VALIDATION', 'cell xác thực'), ('cell validation', 'cell xác thực'), ('tập VALIDATION', 'tập xác thực'),
    ('tập validation', 'tập xác thực'), ('MAE validation', 'MAE trên tập xác thực'),
    ('sai số validation', 'sai số trên tập xác thực'), ('dự đoán validation', 'dự đoán trên tập xác thực'),
    ('tỉ số validation', 'tỉ số trên tập xác thực'), ('validation của', 'tập xác thực của'),
    ('validation chọn', 'tập xác thực chọn'), ('validation', 'tập xác thực'),
    ('cell test', 'cell kiểm tra'), ('tập test', 'tập kiểm tra'), ('hàng test', 'hàng kiểm tra'),
    ('dự đoán test', 'dự đoán trên tập kiểm tra'), ('kết quả test', 'kết quả kiểm tra'),
    ('trên test', 'trên tập kiểm tra'), ('cell train', 'cell huấn luyện'), ('hàng train', 'hàng huấn luyện'),
    ('chu kỳ train', 'chu kỳ huấn luyện'), ('tập train', 'tập huấn luyện'),
    ('normaliser', 'bộ chuẩn hoá'), ('loss vật lý', 'hàm mất mát vật lý'), ('trùng từng bit', 'giống hệt'),
    ('theo dự đoán phù hợp', 'theo phương pháp conformal'), ('dự đoán phù hợp', 'phương pháp conformal'),
    ('mỗi cell có một phiếu', 'mỗi cell được tính với trọng số như nhau'),
    ('mỗi cell một phiếu', 'mỗi cell có trọng số như nhau'), ('biến thể cắt bỏ', 'biến thể lược bỏ thành phần'),
    ('mặt mất mát', 'bề mặt hàm mất mát'), ('mỗi cell được test', 'mỗi cell được kiểm tra'),
    ('Trên test', 'Trên tập kiểm tra'), ('cùng chiều với test', 'cùng chiều với tập kiểm tra'),
]


def split_semicolons(text):
    """Tách các mệnh đề nối bằng dấu chấm phẩy thành câu riêng: chỉ ở mức ngoài cùng (không trong ngoặc, ngoặc nhọn hay
    chế độ toán), khi chữ sau là chữ thường; bỏ qua môi trường bảng. Dấu chấm phẩy giữa các số (0{,}5; 0{,}2) hay trong
    ngoặc vuông của khoảng tin cậy không bị đổi."""
    out, i, n = [], 0, len(text)
    par = brace = 0; math = False; in_tab = 0
    while i < n:
        if text.startswith('\\begin{tabular', i) or text.startswith('\\begin{tabularx', i):
            in_tab += 1
        elif text.startswith('\\end{tabular', i):
            in_tab = max(0, in_tab - 1)
        ch = text[i]
        if ch == '\\' and i + 1 < n:
            out.append(text[i:i + 2]); i += 2; continue
        if ch == '$':
            math = not math
        elif not math:
            if ch in '([':
                par += 1
            elif ch in ')]':
                par = max(0, par - 1)
            elif ch == '{':
                brace += 1
            elif ch == '}':
                brace = max(0, brace - 1)
        if (ch == ';' and not math and par == 0 and brace == 0 and not in_tab and i + 2 < n and text[i + 1] == ' '
                and text[i + 2].isalpha() and text[i + 2].islower()):
            out.append('. ' + text[i + 2].upper()); i += 3; continue
        out.append(ch); i += 1
    return ''.join(out)


def terms(text):
    for a, b in TERMS:
        text = text.replace(a, b)
    text = re.sub(r'có ý nghĩa(?! thống kê| vật lý)', 'có ý nghĩa thống kê', text)
    return split_semicolons(text)


def write(name, text):
    os.makedirs(GEN, exist_ok=True)
    open(os.path.join(GEN, name), 'w').write(terms(text).rstrip() + '\n')


def keep_copy(src, name):
    """results/ không lên git: giữ một bản trong report/generated/data/ và đọc từ đó khi results/ không có."""
    dst = os.path.join(DATA, name)
    if os.path.exists(src):
        os.makedirs(DATA, exist_ok=True)
        shutil.copy(src, dst)
    return dst if os.path.exists(dst) else None


# --------------------------------------------------------------------------- bảng lấy nguyên từ báo cáo đã nộp
def thesis_block(label, edits=()):
    s = open(THESIS, encoding='utf-8').read()
    i = s.index(f'\\label{{{label}}}')
    a = s.rfind('\\begin{table}', 0, i)
    b = s.index('\\end{table}', i) + len('\\end{table}')
    blk = s[a:b]
    for old, new in edits:
        assert old in blk, (label, old)
        blk = blk.replace(old, new)
    return blk + '\n'


# --------------------------------------------------------------------------- độ đo, mũi tên, in đậm số tốt nhất
UP, DN = r'$\uparrow$', r'$\downarrow$'
_NUM = r'([−-]?\d+(?:\{,\}\d+)?)'


def _num(cell):
    """Số đứng đầu một ô bảng ('6{,}10$^{*}$' -> 6.10; '−0{,}99' -> -0.99); None nếu ô không bắt đầu bằng số."""
    m = re.match(r'\s*(?:\\textbf\{)?' + _NUM, cell)
    return float(m.group(1).replace('{,}', '.').replace('−', '-')) if m else None


def metric_row(blk, anchor, row):
    """Chèn hàng nêu độ đo (kèm mũi tên) ngay trên dòng tiêu đề cột bắt đầu bằng `anchor`."""
    assert blk.count(anchor) == 1, anchor
    i = blk.index(anchor)
    return blk[:i] + row + '\n' + blk[i:]


def bold_best(blk, cols, best=min, by='col', groups=False, skip=0, drop_last=0, key=None):
    """In đậm số tốt nhất trong các cột `cols` (chỉ số từ 0). by='col': theo từng cột, trong mỗi nhóm hàng nếu groups=True
    (nhóm mới bắt đầu ở hàng có ô đầu khác rỗng); by='row': theo từng hàng. skip/drop_last: bỏ qua bao nhiêu hàng dữ liệu
    đầu/cuối. best chọn giá trị tốt nhất (min hoặc max) sau khi áp `key` (vd. abs). Chỉ in đậm khi có ít nhất hai số để
    so; số bằng nhau ở độ chính xác hiển thị cùng được in đậm. Ô đã in đậm giữ nguyên."""
    key = key or (lambda x: x)
    head, rest = blk.split('\\midrule', 1)
    body, tail = rest.split('\\bottomrule', 1)
    lines = body.split('\n')
    data = [i for i, ln in enumerate(lines) if ' & ' in ln and ln.rstrip().endswith('\\\\')]
    data = data[skip:len(data) - drop_last]
    cells = {i: lines[i].rstrip()[:-2].rstrip().split(' & ') for i in data}

    def mark(i, c):
        if '\\textbf' not in cells[i][c]:
            cells[i][c] = re.sub(r'^(\s*)([−-]?\d+(?:\{,\}\d+)?(?:\\,\\%)?)', r'\1\\textbf{\2}', cells[i][c], count=1)

    def pick(pairs):
        pairs = [(k, key(x)) for k, x in pairs if x is not None]
        if len(pairs) >= 2:
            b = best([x for _, x in pairs])
            for k, x in pairs:
                if x == b:
                    yield k
    if by == 'row':
        for i in data:
            for c in list(pick([(c, _num(cells[i][c])) for c in cols if c < len(cells[i])])):
                mark(i, c)
    else:
        grp, cur = [], []
        for i in data:
            if groups and cells[i][0].strip() and cur:
                grp.append(cur); cur = []
            cur.append(i)
        grp.append(cur)
        for g in grp:
            for c in cols:
                for i in list(pick([(i, _num(cells[i][c])) for i in g if c < len(cells[i])])):
                    mark(i, c)
    for i in data:
        lines[i] = ' & '.join(cells[i]) + ' \\\\'
    return head + '\\midrule' + '\n'.join(lines) + '\\bottomrule' + tail


# --------------------------------------------------------------------------- vai trò của phần vật lý
def physics_params():
    """λ và Eₐ học được của mọi PINN-semi trong E20 (3 thành viên, 4 mức nhãn, 3 lần lặp, 5 fold) và nhiệt độ của từng
    bộ; bản sao ở generated/data/physics_params.csv để dựng được khi không có runs/."""
    path = os.path.join(DATA, 'physics_params.csv')
    files = glob.glob('runs/E20/P_*.json')
    if len(files) >= 720:
        rows = []
        for f in files:
            d = json.load(open(f))
            rows.append(dict(ds=os.path.basename(f).split('_')[1], lam=d['physics_params']['lam'],
                             Ea=d['physics_params']['Ea_kJ_mol']))
        P = pd.DataFrame(rows).groupby('ds').agg(lam_min=('lam', 'min'), lam_max=('lam', 'max'), Ea_min=('Ea', 'min'),
                                                 Ea_max=('Ea', 'max'), n=('lam', 'size')).reset_index()
        P.to_csv(path, index=False)
    return pd.read_csv(path) if os.path.exists(path) else None


def physics_section(P):
    """Mục Thảo luận: vì sao phần vật lý đóng góp chủ yếu qua ràng buộc đơn điệu, và λ, Eₐ không định danh được."""
    g = lambda ds, c: float(P[P.ds == ds][c].iloc[0])
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    lam = jv([f"{rng([g(d, 'lam_min'), g(d, 'lam_max')])} ở {d}" for d in DS])
    far = max(DS, key=lambda d: max(abs(g(d, 'lam_min') - 1), abs(g(d, 'lam_max') - 1)))
    ea_x = rng([g('XJTU', 'Ea_min'), g('XJTU', 'Ea_max')], 1)
    shift = max(max(abs(30 - g(d, 'Ea_min')), abs(30 - g(d, 'Ea_max'))) for d in ('TJU', 'MIT', 'HUST'))
    return (
        r'\subsection{Vai trò của phần vật lý}\label{sec:vatly}' '\n'
        'Đề tài nhấn mạnh sự hỗ trợ của thông tin hoá lý, nên cần nói rõ phần vật lý trong PINN-semi thực sự đóng góp gì. '
        'Ba quan sát ở các mục trước cho thấy đóng góp này hẹp hơn tên gọi. Thứ nhất, chỉ riêng ràng buộc đơn điệu đã '
        r'mang lại 62--95\,\% mức giảm sai số của PINN-semi so với MLP ở 30\,\% nhãn, còn bỏ phần dư động học và ràng buộc '
        r'miền chỉ làm MAE tăng 1--11\,\% (\Cref{sec:source}). Thứ hai, các tham số hoá lý dịch chuyển ít so với giá trị '
        f'khởi tạo: $\\lambda$ (khởi tạo bằng 1) của các PINN-semi nằm trong khoảng {lam}, còn $E_a$ khởi tạo ở '
        f'30\\,kJ/mol thì dịch không quá {v(shift, 1)}\\,kJ/mol. Thứ ba, tốc độ $r$ học được trung bình chỉ 0{{,}}001--0{{,}}063, trong khi '
        'tốc độ suy giảm thực đo là 0{,}137--0{,}356 SOH trên 1\\,000 chu kỳ. Cả ba có nguyên nhân trong cách đặt bài toán, '
        'không phải do tối ưu chưa hội tụ.\n\n'
        r'Phần dư động học ràng buộc đạo hàm riêng $\partial u/\partial t$ khi giữ nguyên đặc trưng sạc. Trên quỹ đạo thật '
        r'của một cell, SOH thay đổi chủ yếu vì đặc trưng sạc thay đổi theo tuổi, tức qua số hạng $\nabla_{\mathbf{x}} u '
        r'\cdot \mathrm{d}\mathbf{x}/\mathrm{d}n$, còn $t$ chỉ là một trong 17 đầu vào. Mạng nghiệm vì vậy thoả được phương '
        'trình với một đạo hàm riêng nhỏ theo $t$, và $r$ học được nhỏ hơn nhiều so với tốc độ suy giảm thật. Với trọng số '
        r'$\alpha = 0{,}02$, phần dư này chủ yếu buộc đạo hàm theo $t$ không dương, điều mà ràng buộc đơn điệu đã làm trực '
        'tiếp hơn.\n\n'
        r'Hệ số $\lambda$ khó xác định vì thừa số $g(u) = e^{\lambda(1-u)}$ chỉ thay đổi khoảng 22\,\% khi SOH đi từ 1 xuống '
        r'0,8 với $\lambda \approx 1$. Sự phụ thuộc vào mức suy giảm này đã có sẵn trong $k(\tilde{\mathbf{x}})$, vốn nhận '
        'các đặc trưng sạc biến thiên theo tuổi. Hai thừa số thay thế được cho nhau, nên dữ liệu chỉ kéo $\\lambda$ đi '
        f'một đoạn ngắn khỏi giá trị khởi tạo, xa nhất ở {far}.' '\n\n'
        r'Năng lượng hoạt hoá $E_a$ thì không định danh được về mặt toán học trên ba bộ. Thừa số Arrhenius '
        r'$A(T) = \exp[-(E_a/R_g)(1/T - 1/T_\text{ref})]$ chỉ khác 1 khi nhiệt độ khác $T_\text{ref}$. Ở XJTU, mọi cell '
        r'chạy ở 298{,}15\,K, đúng bằng $T_\text{ref}$, nên $1/T - 1/T_\text{ref} = 0$, $A \equiv 1$ và gradient của hàm mất '
        f'mát theo $E_a$ bằng 0. Thực tế $E_a$ của mọi PINN-semi trên XJTU giữ đúng {ea_x}\\,kJ/mol. Ở MIT và HUST, nhiệt '
        r'độ là hằng 303{,}15\,K, nên $A$ chỉ là một hằng số nhân và bị $k(\tilde{\mathbf{x}}) = '
        r'\operatorname{softplus}(G_\theta(\tilde{\mathbf{x}}))$ hấp thụ: mọi giá trị $E_a$ cho cùng một $r$ nếu $G_\theta$ '
        r'bù lại. Chỉ TJU có ba mức nhiệt độ (298{,}15, 308{,}15 và 318{,}15\,K), nơi $E_a$ về nguyên tắc định danh được. '
        r'Nhưng ở đó nhiệt độ chỉ làm sai số thay đổi dưới 1\,\% (thêm $1/T$ vào GBDT cho tỉ số 0{,}99--1{,}00, '
        r'\Cref{sec:dt_bo}), nên dữ liệu cũng không đủ để ước lượng $E_a$ chính xác.' '\n\n'
        'Dù vậy, phần vật lý vẫn có giá trị ở ba điểm. Một là ràng buộc đơn điệu và ràng buộc miền là tiên nghiệm vật lý '
        'đúng, và chúng được áp lên cả cell không nhãn. Ở 10\\,\\% nhãn, phần lợi ích do áp hàm mất mát vật lý lên cell '
        'không nhãn (PINN-semi so với PINN-sup, tỉ số 0{,}62--0{,}94) chiếm 50--64\\,\\% mức giảm sai số so với MLP. Hai là '
        'dạng tốc độ có cấu trúc, luôn không âm, ngăn lời giải tầm thường $G \\approx u_t$ của động học hộp đen kiểu '
        'PINN4SOH, nên các ràng buộc thực sự tác động lên mạng nghiệm. Ba là ở bài toán dự báo, nghiệm đóng của cùng động '
        'học xám cho PINN-dự báo một đường ngoại suy có dạng vật lý: luôn không tăng, có thể gãy dần hoặc chậm dần, với chỉ '
        'hai tham số. PINN-dự báo là một trong ba nhánh của HiPINN. Vì vậy cụm “thông tin hoá lý” trong đồ án nên được hiểu '
        'là các tiên nghiệm có cấu trúc (đơn điệu, miền giá trị, tốc độ không âm, dạng Arrhenius, dạng nghiệm đóng), chứ '
        'không phải việc xác định các tham số điện hoá của cell.')


def tau_text():
    """Đoạn giải thích ngưỡng cuối đời τ của từng bộ (Bảng tham số dự báo), số tính từ dữ liệu; bản sao ở
    generated/data/tau.csv."""
    path = os.path.join(DATA, 'tau.csv')
    try:
        from pinnsoh.baselines import cells_of
        from pinnsoh.forecast import TAU
        rows = []
        for ds in DS:
            mins = np.array([np.nanmin(c.soh) for c in cells_of(ds)])
            grid = np.round(np.arange(0.70, 1.00, 0.01), 2)
            low = min(g for g in grid if np.mean(mins <= g) >= 0.8)
            rows.append(dict(ds=ds, tau=TAU[ds], rule=float(low), reach_tau=float(np.mean(mins <= TAU[ds])),
                             reach_08=float(np.mean(mins <= 0.8))))
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception as e:                                       # không có dữ liệu: dùng bản đã lưu
        print('tau: dùng bản đã lưu', e)
    T = pd.read_csv(path)
    assert (np.isclose(T.tau, T.rule)).all(), 'τ không theo quy tắc đã nêu'
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    return ('Ngưỡng cuối đời $\\tau$ không lấy chung 0,8 như quy ước xe điện ở \\Cref{sec:bai_toan}, vì nhiều cell trong '
            'các thí nghiệm này dừng thử trước khi SOH xuống 0,8. Tỉ lệ cell có SOH đo chạm 0,8 chỉ là '
            + jv([f'{pct(r.reach_08)} ở {r.ds}' for r in T.itertuples()])
            + ', nên RUL theo ngưỡng 0,8 không xác định được cho phần lớn cell. Với mỗi bộ, $\\tau$ được chọn trước khi học '
            'mô hình theo một quy tắc chỉ dựa trên phạm vi dữ liệu: mức thấp nhất trên lưới bước 0,01 mà ít nhất 80\\,\\% số '
            'cell của bộ có SOH đo chạm tới. Quy tắc này cho $\\tau$ = '
            + jv([f'{v(r.tau)} ({r.ds})' for r in T.itertuples()])
            + '. Quy tắc dùng SOH đo của mọi cell trong bộ, kể cả cell kiểm tra, nhưng chỉ để định nghĩa mục tiêu RUL chung '
            'cho mọi mô hình, không thay đổi đầu vào hay tham số của mô hình nào. '
            "Cột ``cell có chu kỳ cuối đời'' của \\Cref{tab:fcset} đếm theo định nghĩa chặt hơn dùng khi chấm, tức "
            'trung vị trượt 9 chu kỳ của SOH đo xuống dưới $\\tau$, nên có bộ thấp hơn 80\\,\\% một chút. Vì $\\tau$ khác nhau '
            'giữa các bộ, RUL của các bộ không so trực tiếp với nhau.')


def cost_text():
    """Mục chi phí tính toán (Chương 4), từ analysis/compute_cost.py; bản sao ở generated/data/compute_cost.csv."""
    p = keep_copy('results/compute_cost.csv', 'compute_cost.csv')
    if p is None:
        return '(Chưa có số đo chi phí: chạy \\texttt{python analysis/compute\\_cost.py}.)'
    C = pd.read_csv(p)
    kb = lambda x: v(x / 1024, 0) + '\\,KB'
    mb = lambda x: v(x / 1024 ** 2, 1) + '\\,MB'
    sz = lambda x: kb(x) if x < 1024 ** 2 else mb(x)
    szr = lambda a, b: (f'{v(a / 1024 ** 2, 1)}--{mb(b)}' if a >= 1024 ** 2 else f'{sz(a)}--{sz(b)}')
    rows = [f'{r.ds} & {thou(r.sol_params)} & {r.gbdt_trees} & {sz(r.lai_bytes)} & {v(r.ms_lai, 2)} & '
            f'{sz(r.hipinn_bytes)} & {v(r.ms_hipinn, 0)} \\\\' for r in C.itertuples()]
    tab = (r'\begin{table}[!htbp]\centering\small' '\n'
           r'\caption[Chi phí tính toán của PINN-lai và HiPINN]{Chi phí tính toán trên CPU một luồng của mô hình trong công '
           r'cụ dòng lệnh (học ở 30\,\% nhãn trên một fold của mỗi bộ). Tham số mạng nghiệm tính cho một trong ba mạng của '
           r'PINN-lai; dung lượng là kích thước khi lưu; độ trễ là trung vị của nhiều lần gọi. PINN-lai: một chu kỳ mới (ba '
           r'mạng nghiệm, GBDT cho một hàng, bộ lọc xu thế). HiPINN: một lần dự báo từ một gốc cho mọi tầm tới $3L$, gồm cả '
           r'tính đặc trưng lịch sử của cell.}\label{tab:cost}' '\n'
           r'\begin{tabular}{lrrrrrr}\toprule' '\n'
           r' & \multicolumn{4}{c}{PINN-lai} & \multicolumn{2}{c}{HiPINN} \\\cmidrule(lr){2-5}\cmidrule(lr){6-7}' '\n'
           r'Bộ & \begin{tabular}[b]{@{}r@{}}tham số\\mạng nghiệm\end{tabular} & \begin{tabular}[b]{@{}r@{}}số cây\\GBDT\end{tabular}'
           r' & \begin{tabular}[b]{@{}r@{}}dung\\lượng\end{tabular} & \begin{tabular}[b]{@{}r@{}}độ trễ\\(ms/chu kỳ)\end{tabular}'
           r' & \begin{tabular}[b]{@{}r@{}}dung\\lượng\end{tabular} & \begin{tabular}[b]{@{}r@{}}độ trễ\\(ms/dự báo)\end{tabular} \\\midrule'
           '\n' + '\n'.join(rows) + '\n' r'\bottomrule\end{tabular}\end{table}')
    big = C.loc[C.hipinn_bytes.idxmax()]
    share = lambda r: r.hip_life_bytes / r.hipinn_bytes
    gb_first = C.ms_gbdt.median() >= C.ms_nn3.median()
    parts = [f'lượt dự đoán của GBDT ({rng(C.ms_gbdt, 2)}\\,ms)', f'ba lượt truyền thẳng ({rng(C.ms_nn3, 2)}\\,ms)']
    parts = parts if gb_first else parts[::-1]
    mem = C.hipinn_bytes.max() / 1024 ** 2
    mem_txt = f'{int(np.ceil(mem))}\\,MB trở lên'
    return ('BMS thường chạy trên vi điều khiển hoặc một máy tính nhúng có bộ nhớ và năng lực tính toán hạn chế, nên chi '
            'phí suy luận là một tiêu chí khi triển khai. \\Cref{tab:cost} đo chi phí của hai mô hình cuối trên CPU một '
            f'luồng. Mỗi mạng nghiệm của PINN-lai có {thou(int(C.sol_params.iloc[0]))} tham số. Khi suy luận chỉ cần mạng '
            'nghiệm, còn mạng tốc độ chỉ dùng lúc học. GBDT của PINN-lai có '
            f'{rng(C.gbdt_trees.astype(float), 0)} cây tuỳ bộ. Toàn bộ PINN-lai chiếm '
            f'{szr(C.lai_bytes.min(), C.lai_bytes.max())} và cần {rng(C.ms_lai, 2)}\\,ms cho mỗi chu kỳ mới, chủ yếu cho {parts[0]}, sau đó là '
            f'{parts[1]}. Bộ lọc xu thế chỉ cần '
            'nhớ 10 ước lượng gần nhất của cell. Với một lần đo mỗi chu kỳ sạc, chi phí này nhỏ so với thời gian một lần sạc. '
            + ('HiPINN nặng hơn: ' if C.hipinn_bytes.min() > C.lai_bytes.max() else 'HiPINN chiếm ')
            + f'{szr(C.hipinn_bytes.min(), C.hipinn_bytes.max())}, lớn nhất ở {big.ds}, trong đó '
            f'nhánh theo đời (GBDT đời còn lại và khuôn quỹ đạo của các cell huấn luyện) chiếm '
            f'{v(100 * min(share(r) for r in C.itertuples()), 0)}--{pct(max(share(r) for r in C.itertuples()))}. Một lần dự báo cho mọi tầm tới $3L$ mất {rng(C.ms_hipinn, 0)}\\,ms. '
            f'Dự báo không cần chạy ở mọi chu kỳ, nên HiPINN phù hợp với một máy chủ hay một bộ điều khiển có bộ nhớ từ '
            f'{mem_txt}, còn PINN-lai đủ nhẹ để chạy sau mỗi lần sạc trên một máy tính nhúng. Các số đo trên một lõi CPU của '
            'máy chạy thí nghiệm, không phải trên phần cứng của BMS. Trên vi điều khiển, độ trễ sẽ lớn hơn và cần đo lại.'
            '\n\n' + tab)


def xjtu_band_note(cid):
    """Câu cho chú thích hình quỹ đạo dự báo: SOH đo của cell XJTU được vẽ lặp lại theo một vòng vài chu kỳ (nhóm xả thay
    đổi), nên các điểm đo xếp thành các dải song song; bản sao ở generated/data/xjtu_bands.json."""
    path = os.path.join(DATA, 'xjtu_bands.json')
    try:
        from pinnsoh.baselines import cells_of
        c = {x.cid: x for x in cells_of('XJTU')}[cid]
        s = c.soh
        d = {lag: float(np.nanmean(np.abs(s[lag:] - s[:-lag]))) for lag in range(1, 9)}
        per = min(range(2, 9), key=lambda k: d[k])
        json.dump(dict(cid=cid, period=per, d=d), open(path, 'w'), indent=1)
    except Exception as e:
        print('xjtu_bands: dùng bản đã lưu', e)
    if not os.path.exists(path):
        return ''
    J = json.load(open(path))
    d = {int(k): x for k, x in J['d'].items()}; per = J['period']
    others = [d[k] for k in range(1, per)]
    grp = J['cid'].split('/')[-1].split('_battery')[0]
    return (f' Ở XJTU, cell được chọn thuộc nhóm {tex_esc(grp)}, nhóm có chế độ xả thay đổi. Dung lượng đo của cell này lặp '
            f'lại theo vòng {per} chu kỳ: chênh lệch trung bình giữa hai chu kỳ cách nhau {per} là {m3(d[per], 1)} '
            f'($\\times10^{{-3}}$), so với {rng([1e3 * x for x in others], 1)} ở các khoảng cách 1--{per - 1} chu kỳ. Vì vậy '
            f'các điểm đo xếp thành {per} dải song song; đó là đặc điểm của dữ liệu, không phải nhiều cell vẽ chồng lên nhau.')


# --------------------------------------------------------------------------- PINN-lai và HiPINN-lai, mọi lựa chọn trong fold
FR = (0.1, 0.3, 0.5, 0.7)


def e20f_load():
    """Bảng của analysis/analyze_e20_folds.py: PINN-lai cố định, PINN-lai tinh chỉnh và HiPINN-lai với mọi lựa chọn làm
    riêng trong từng fold; bản sao ở generated/data/."""
    out = {}
    for name in ('summary', 'tests', 'choices', 'units'):
        p = keep_copy(f'results/e20f_{name}.csv', f'e20f_{name}.csv')
        if p is None:
            return None
        out[name] = pd.read_csv(p)
    if len(out['summary'].f.unique()) < 4:                     # chưa đủ bốn mức nhãn
        return None
    return out


def fval(F, ds, f, m, col='mae'):
    S = F['summary']
    r = S[(S.ds == ds) & np.isclose(S.f, f) & (S.model == m)]
    return float(r[col].iloc[0]) if len(r) else None


def ftest(F, a, b, ds, f):
    T = F['tests']
    r = T[(T.a == a) & (T.b == b) & (T.ds == ds) & np.isclose(T.f, f)]
    return r.iloc[0] if len(r) else None


def _rows_of(blk):
    head, rest = blk.split('\\midrule', 1)
    body, tail = rest.split('\\bottomrule', 1)
    return head, body.split('\n'), tail


def _unbold(c):
    """Bỏ \\textbf{...} quanh một ô, kể cả khi bên trong có ngoặc nhọn lồng một mức ('\\textbf{14{,}37}')."""
    return re.sub(r'\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', c)


def allmae_table(F):
    """Bảng MAE của mọi mô hình chính: các cột của báo cáo đã nộp, hai cột cuối là PINN-lai cố định và HiPINN-lai tính
    lại với mọi lựa chọn trong fold. Cột GBDT và PINN-semi phải trùng phân tích mới (kiểm tra tái lập)."""
    blk = thesis_block('tab:allmae', [(
        r'PINN-lai là bản tinh chỉnh trên validation; mô hình cuối là PINN-lai dùng hay không dùng lịch sử theo validation '
        r'(Mục~\ref{sec:histres}).',
        r'PINN-lai là cấu hình cố định ($\omega = \tfrac12$, $w = 10$), chốt trước khi đọc tập kiểm tra. HiPINN-lai là '
        r'PINN-lai có thêm đặc trưng lịch sử của đặc trưng sạc ở những fold mà tập xác thực chọn; trên bộ không có dạng '
        r'lịch sử nào được chọn (MIT), hai cột trùng nhau. HiPINN-lai là phân tích hậu kiểm (\Cref{sec:pinnlai}).'),
        ('Mô hình cuối', 'HiPINN-lai')])
    head, lines, tail = _rows_of(blk)
    ds = None
    for i, ln in enumerate(lines):
        if ' & ' not in ln or not ln.rstrip().endswith('\\\\'):
            continue
        cells = [_unbold(c.strip()) for c in ln.rstrip()[:-2].split(' & ')]
        ds = cells[0] or ds
        f = int(cells[1].split('\\')[0]) / 100
        for col, m in ((6, 'gbdt'), (7, 'pinn')):
            assert abs(_num(cells[col]) - 1e3 * fval(F, ds, f, m)) < 0.006, ('không tái lập', ds, f, m, cells[col])
        cells[8], cells[9] = v(1e3 * fval(F, ds, f, 'pinn_lai')), v(1e3 * fval(F, ds, f, 'hipinn_lai'))
        lines[i] = ' & '.join(cells) + ' \\\\'
    blk = head + '\\midrule' + '\n'.join(lines) + '\\bottomrule' + tail
    return bold_best(blk, list(range(2, 10)), by='row')


def allrm_table(F, MAE3):
    """RMSE theo cell và MAPE ở 30 % nhãn: các hàng của báo cáo đã nộp, hai hàng PINN-lai và HiPINN-lai tính lại."""
    head = (' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{d}}}' for d in DS) + r' \\'
            + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))) + '\n'
            + 'Mô hình & ' + ' & '.join([f'RMSE {DN} & MAPE {DN}'] * len(DS)) + r' \\')
    blk = thesis_block('tab:allrm', [
        ('MAPE (\\%, gộp mọi chu kỳ) trên test; in đậm là thấp nhất mỗi cột.',
         'MAPE (\\%, tính chung mọi chu kỳ) trên test; in đậm là thấp nhất mỗi cột. PINN-lai và HiPINN-lai như ở '
         'Bảng~\\ref{tab:allmae}.'),
        ('Mô hình cuối &', 'HiPINN-lai &'),
        (r'Mô hình & XJTU RMSE & MAPE & TJU RMSE & MAPE & MIT RMSE & MAPE & HUST RMSE & MAPE \\', head)])
    head_, lines, tail = _rows_of(blk)
    for i, ln in enumerate(lines):
        if ' & ' not in ln or not ln.rstrip().endswith('\\\\'):
            continue
        cells = [_unbold(c.strip()) for c in ln.rstrip()[:-2].split(' & ')]
        m = {'PINN-lai': 'pinn_lai', 'HiPINN-lai': 'hipinn_lai', 'GBDT': 'gbdt'}.get(cells[0])
        if m == 'gbdt':                                       # kiểm tra cách tính: hàng GBDT phải trùng
            for j, ds in enumerate(DS):
                assert abs(_num(cells[1 + 2 * j]) - 1e3 * fval(F, ds, 0.3, 'gbdt', 'rmse_cell')) < 0.006, ('RMSE', ds)
                assert abs(_num(cells[2 + 2 * j]) - fval(F, ds, 0.3, 'gbdt', 'mape')) < 0.006, ('MAPE', ds)
        elif m:
            cells = [cells[0]] + [x for ds in DS for x in (v(1e3 * fval(F, ds, 0.3, m, 'rmse_cell')),
                                                           v(fval(F, ds, 0.3, m, 'mape')))]
        lines[i] = ' & '.join(cells) + ' \\\\'
    blk = head_ + '\\midrule' + '\n'.join(lines) + '\\bottomrule' + tail
    return bold_best(blk, list(range(1, 9)))


def rmse_text(F, blk):
    """Đoạn ngắn về RMSE, MAPE và R^2 ở 30 % nhãn, số đọc từ chính bảng."""
    head, lines, tail = _rows_of(blk)
    rows = [[c.strip() for c in ln.rstrip()[:-2].split(' & ')] for ln in lines
            if ' & ' in ln and ln.rstrip().endswith('\\\\')]
    best = {}
    for j, ds in enumerate(DS):
        for k, met in ((1, 'RMSE'), (2, 'MAPE')):
            vals = [(_num(r[k + 2 * j]), r[0]) for r in rows]
            lo = min(x for x, _ in vals)
            best[(ds, met)] = [n for x, n in vals if x == lo]
    worst = {ds: max(rows, key=lambda r: _num(r[1 + 2 * j]))[0] for j, ds in enumerate(DS)}
    ours = {'PINN-lai', 'HiPINN-lai'}
    n_ours = sum(any(n in ours for n in best[k]) for k in best)
    others = sorted({n for k in best for n in best[k] if n not in ours})
    r2 = [fval(F, ds, 0.3, 'pinn_lai', 'r2_10') for ds in DS]
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    t = (r'Ở 30\,\% nhãn (Bảng~\ref{tab:allrm}), PINN-lai hoặc HiPINN-lai có RMSE hoặc MAPE thấp nhất ở '
         f'{n_ours}/8 cột' + (f'; ở các cột còn lại, mô hình thấp nhất là {jv(others)}' if others else '') + '. '
         + ('MLP có RMSE cao nhất trên cả bốn bộ. ' if all(w == 'MLP' for w in worst.values()) else '')
         + f'$R^2$ của PINN-lai ở 30\\,\\% nhãn, tính từ chu kỳ 10 của mỗi cell, là {rng(r2, 3)}.')
    return t.replace('; ở', '. Ở')


def history_table_f(F):
    """Bảng thêm lịch sử của đặc trưng sạc (E29) với PINN-lai cố định; dấu theo Wilcoxon, Holm qua các bộ có lịch sử."""
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Thêm lịch sử của cell vào đầu vào]{MAE theo cell ($\times10^{-3}$) trên tập kiểm tra khi thêm lịch sử '
           r'của đặc trưng sạc vào đầu vào, 3 lần lặp $\times$ 5 fold. Dạng lịch sử của mỗi bộ được chốt ở giai đoạn phát '
           r'triển trên tập xác thực; trên MIT không dạng nào giảm sai số trên tập xác thực đủ 2\,\% nên không có cột lịch '
           r'sử. PINN-lai là cấu hình cố định ở cả hai cột. $^{*}$/$^{\dagger}$: thấp hơn/cao hơn có ý nghĩa thống kê so '
           r'với cột không lịch sử ngay trước (Wilcoxon ghép cặp theo cell, hiệu chỉnh Holm qua các bộ có lịch sử, '
           r'$p < 0{,}05$). In đậm là thấp hơn trong mỗi cặp không/có lịch sử.}\label{tab:hist}',
           r'\begin{tabular}{llrrrr}\toprule',
           r' & & \multicolumn{4}{c}{MAE theo cell ($\times10^{-3}$) ' + DN + r'} \\\cmidrule(lr){3-6}',
           r' & & \multicolumn{2}{c}{PINN-semi} & \multicolumn{2}{c}{PINN-lai} \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
           r'Bộ & Nhãn & không lịch sử & có lịch sử & không lịch sử & có lịch sử \\\midrule']

    def mk(a, b, ds, f):
        x = fval(F, ds, f, a)
        if x is None:
            return '--'
        t = ftest(F, a, b, ds, f)
        mark = ((r'$^{*}$' if t.ratio < 1 else r'$^{\dagger}$') if t is not None and t.p_holm < 0.05 else '')
        return v(1e3 * x) + mark
    for ds in DS:
        for j, f in enumerate(FR):
            out.append(f'{ds if j == 0 else ""} & {int(100 * f)}\\,\\% & {v(1e3 * fval(F, ds, f, "pinn"))} & '
                       f'{mk("pinn_h", "pinn", ds, f)} & {v(1e3 * fval(F, ds, f, "pinn_lai"))} & '
                       f'{mk("lai_h", "pinn_lai", ds, f)} \\\\')
    out += [r'\bottomrule\end{tabular}\end{table}']
    return bold_best(bold_best('\n'.join(out), [2, 3], by='row'), [4, 5], by='row')


def history_text_f(F):
    """Đoạn văn về HiPINN-lai: lịch sử giúp ở đâu, fold nào chọn lịch sử, và kết quả của HiPINN-lai."""
    C, U = F['choices'], F['units']
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    hist_ds = [ds for ds in DS if C[C.ds == ds].hist_avail.max() > 0]
    if not hist_ds:
        return '(Chưa có kết quả E29: chạy E29 rồi \\texttt{python analysis/analyze\\_e20\\_folds.py}.)'
    rat = {(ds, f): fval(F, ds, f, 'lai_h') / fval(F, ds, f, 'pinn_lai') for ds in hist_ds for f in FR}
    sig = lambda a, b, s: [(ds, f) for ds in hist_ds for f in FR
                           if (t := ftest(F, a, b, ds, f)) is not None and t.p_holm < 0.05 and (t.ratio - 1) * s > 0]
    good, bad = sig('lai_h', 'pinn_lai', -1), sig('lai_h', 'pinn_lai', +1)
    case = lambda ks: jv([f'{ds} {int(100 * f)}\\,\\%' for ds, f in ks]) if ks else 'không trường hợp nào'
    share = {ds: [float(C[(C.ds == ds) & np.isclose(C.f, f)].hist_used.iloc[0]) for f in FR] for ds in hist_ds}
    rh = [fval(F, ds, f, 'hipinn_lai') / fval(F, ds, f, 'pinn_lai') for ds in DS for f in FR]
    hg = [(ds, f) for ds in DS for f in FR
          if (t := ftest(F, 'hipinn_lai', 'pinn_lai', ds, f)) is not None and t.p_holm < 0.05 and t.ratio < 1]
    hb = [(ds, f) for ds in DS for f in FR
          if (t := ftest(F, 'hipinn_lai', 'pinn_lai', ds, f)) is not None and t.p_holm < 0.05 and t.ratio > 1]
    # so HiPINN-lai với bản luôn dùng lịch sử: ở các trường hợp lịch sử giúp và các trường hợp lịch sử làm kém đi
    r_hip = lambda ks: [fval(F, ds, f, 'hipinn_lai') / fval(F, ds, f, 'pinn_lai') - 1 for ds, f in ks]
    r_his = lambda ks: [rat[k] - 1 for k in ks]
    end = ''
    if good:
        keep = np.mean(r_hip(good)) / np.mean(r_his(good))
        end += (f'Ở {len(good)} trường hợp lịch sử giúp có ý nghĩa thống kê, HiPINN-lai giữ được {pct(keep)} mức giảm '
                f'sai số của bản luôn dùng lịch sử. ')
    if bad:
        head = (f'Ở trường hợp duy nhất lịch sử làm kém đi có ý nghĩa thống kê ({case(bad)}), mức tăng sai số là '
                if len(bad) == 1 else
                f'Ở {len(bad)} trường hợp lịch sử làm kém đi có ý nghĩa thống kê, mức tăng sai số trung bình là ')
        end += (head + f'{pct(np.mean(r_hip(bad)), 1)} với HiPINN-lai so với {pct(np.mean(r_his(bad)), 1)} với bản luôn '
                'dùng lịch sử. ')
    return ('Bảng~\\ref{tab:hist} cho sai số khi thêm lịch sử của đặc trưng sạc vào đầu vào. Với PINN-lai, bản có lịch sử '
            f'có MAE bằng {rng(list(rat.values()))} lần bản không lịch sử. Bản có lịch sử thấp hơn có ý nghĩa thống kê ở '
            f'{case(good)}, và cao hơn có ý nghĩa thống kê ở {case(bad)}. Trên MIT không dạng lịch sử nào được tập xác thực '
            'chọn. HiPINN-lai chọn riêng trong từng fold có dùng lịch sử hay không: bản có lịch sử chỉ được dùng khi MAE trên cell xác thực của '
            'fold đó giảm ít nhất 2\\,\\%. Tỉ lệ fold chọn lịch sử là '
            + jv([f'{v(100 * min(share[ds]), 0)}--{pct(max(share[ds]))} ở {ds}' if min(share[ds]) != max(share[ds])
                  else f'{pct(share[ds][0])} ở {ds}' for ds in hist_ds]) + ' (tuỳ mức nhãn). '
            f'Kết quả là HiPINN-lai có MAE bằng {rng(rh)} lần PINN-lai, thấp hơn có ý nghĩa thống kê ở {case(hg)} '
            + (f'và cao hơn có ý nghĩa thống kê ở {case(hb)}' if hb else 'và không cao hơn có ý nghĩa thống kê ở trường hợp nào')
            + ' (\\Cref{tab:allmae}). ' + end
            + 'Quyết định dùng lịch sử chỉ dựa trên vài cell xác thực mỗi fold nên không phải lúc nào cũng đúng.')


def seq_mae():
    """MAE theo cell (×10⁻³) của CNN, LSTM, Transformer theo (bộ, mức nhãn), đọc từ bảng MAE chính của báo cáo đã nộp
    (dự đoán theo cell của ba mô hình này không có trong kho nên không kiểm định lại được)."""
    head, lines, tail = _rows_of(thesis_block('tab:allmae'))
    out, ds = {}, None
    for ln in lines:
        if ' & ' not in ln or not ln.rstrip().endswith('\\\\'):
            continue
        cells = [_unbold(c.strip()) for c in ln.rstrip()[:-2].split(' & ')]
        ds = cells[0] or ds
        f = int(cells[1].split('\\')[0]) / 100
        out[(ds, f)] = dict(CNN=_num(cells[3]), LSTM=_num(cells[4]), Transformer=_num(cells[5]))
    return out


def pinn_lai_text(F, rel):
    """Đoạn về PINN-lai cố định và bản tinh chỉnh theo từng fold. rel: tỉ số trên tập xác thực của biến thể đặc trưng
    tương đối (tính lại với mốc nhân quả), hoặc None."""
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    cases = [(ds, f) for ds in DS for f in FR]
    tp = [ftest(F, 'pinn_lai', 'pinn', ds, f) for ds, f in cases]
    tg = [ftest(F, 'pinn_lai', 'gbdt', ds, f) for ds, f in cases]
    n_lo_p = sum(t.ratio < 1 for t in tp); n_sig_p = sum(t.ratio < 1 and t.p_holm < 0.05 for t in tp)
    n_lo_g = sum(t.ratio < 1 for t in tg); n_sig_g = sum(t.ratio < 1 and t.p_holm < 0.05 for t in tg)
    assert n_lo_g > 8, 'Chương 6 (Hạn chế) nói PINN-lai thấp hơn GBDT ở phần lớn trường hợp: sửa lời văn'
    g_wins = [f'{ds} {int(100 * f)}\\,\\%' for (ds, f), t in zip(cases, tg) if t.ratio > 1 and t.p_holm < 0.05]
    g10 = [t for (ds, f), t in zip(cases, tg) if f == 0.1]
    all10 = all(t.ratio < 1 and t.p_holm < 0.05 for t in g10)
    tt = [ftest(F, 'pinn_lai_tc', 'pinn_lai', ds, f) for ds, f in cases]
    up_t = sum(t.ratio < 1 and t.p_holm < 0.05 for t in tt); dn_t = sum(t.ratio > 1 and t.p_holm < 0.05 for t in tt)
    tcg = [ftest(F, 'pinn_lai_tc', 'gbdt', ds, f) for ds, f in cases]
    tcg_lo = sum(t.ratio < 1 for t in tcg); tcg_up = sum(t.ratio < 1 and t.p_holm < 0.05 for t in tcg)
    tcg_dn = sum(t.ratio > 1 and t.p_holm < 0.05 for t in tcg)
    nval = F['units'].groupby('ds').n_val
    lo_v, hi_v = int(nval.min().min()), int(nval.max().max())
    nv = f'{lo_v}' if lo_v == hi_v else f'{lo_v}--{hi_v}'
    inc = ['ràng buộc đơn điệu mạnh hơn (1{,}065)', 'thêm phần dư Euler (1{,}289)']
    if rel is not None and rel > 1:
        inc.append(f'đặc trưng tương đối với mốc nhân quả ({v(rel, 3)})')
    inc_txt = f'{"Ba" if len(inc) == 3 else "Hai"} hướng làm sai số tăng là {jv(inc)}'
    if rel is not None and rel <= 1:
        inc_txt += (f'. Đặc trưng tương đối, khi tính lại với mốc nhân quả, cho tỉ số {v(rel, 3)} trên tập xác thực, '
                    'nhưng không có trong cấu hình đã chốt')
    SQ = seq_mae()
    beat = {(ds, f, m): 1e3 * fval(F, ds, f, 'pinn_lai') < SQ[(ds, f)][m] for ds, f in cases for m in SQ[(ds, f)]}
    lose = {}                                     # (bộ, các mức nhãn) -> các mô hình chuỗi thấp hơn PINN-lai
    for ds in DS:
        for m in ('CNN', 'LSTM', 'Transformer'):
            fs = tuple(f'{int(100 * f)}' for f in FR if not beat[(ds, f, m)])
            if fs:
                lose.setdefault((ds, fs), []).append(m)
    lose_txt = jv([f'{jv(ms)} ở {ds} {jv(list(fs))}\\,\\%' for (ds, fs), ms in lose.items()])
    seq_txt = (f'Tính theo MAE trung bình ở \\Cref{{tab:allmae}}, PINN-lai thấp hơn cả ba mô hình chuỗi ở '
               f'{sum(beat.values())}/{len(beat)} phép so'
               + (f'. Các trường hợp mô hình chuỗi thấp hơn PINN-lai chỉ gồm {lose_txt}. ' if lose else '. '))
    geo = lambda xs: float(np.exp(np.mean(np.log(xs))))
    dev_test = geo([fval(F, ds, f, 'pinn_lai') / fval(F, ds, f, 'pinn') for ds in DS for f in (0.1, 0.5)])
    return (
        'PINN-lai được xây dựng qua hai giai đoạn. Ở giai đoạn phát triển, chín hướng cải tiến được so trên tập xác thực '
        'của lần lặp 0, ở 10 và 50\\,\\% nhãn. Kết hợp với GBDT cho mức giảm lớn nhất (tỉ số trên tập xác thực so với '
        'PINN-semi 0{,}843 ở 10\\,\\% và 0{,}804 ở 50\\,\\% nhãn), còn nhóm ba mạng giảm vừa phải (0{,}963). '
        + inc_txt + '. Cấu hình chốt là '
        'ba PINN-semi ghép với GBDT theo $\\omega = \\tfrac12$ rồi lọc xu thế với $w = 10$, ấn định trước khi đọc tập kiểm '
        'tra.\n\n'
        f'Trên tập kiểm tra, PINN-lai thấp hơn PINN-semi ở {n_lo_p}/16 trường hợp, trong đó {n_sig_p} có ý nghĩa thống '
        f'kê, với tỉ số {rng([t.ratio for t in tp])}. So với GBDT, PINN-lai thấp hơn ở {n_lo_g}/16 trường hợp ({n_sig_g} '
        'có ý nghĩa thống kê)'
        + (', và ở 10\\,\\% nhãn thấp hơn có ý nghĩa thống kê trên cả bốn bộ' if all10 else '') + '. '
        + (f'GBDT chỉ còn tốt hơn có ý nghĩa thống kê ở {jv(g_wins)}. ' if g_wins else
           'GBDT không tốt hơn có ý nghĩa thống kê ở trường hợp nào. ')
        + seq_txt + 'Trên hai mức nhãn dùng khi phát triển, tỉ số trung bình nhân so với PINN-semi trên tập kiểm tra là '
        f'{v(dev_test, 3)}, gần 0{{,}}823 trên tập xác thực, nên việc chọn trên tập xác thực không làm kết quả kiểm tra '
        'lạc quan đáng kể. Tính trung bình nhân trên 16 trường hợp, nhóm ba mạng đưa '
        'tỉ số so với PINN-semi xuống 0{,}961, kết hợp với GBDT xuống 0{,}798 và bộ lọc xuống 0{,}793. Bước kết hợp mang phần '
        'lớn lợi ích, vì lệch mức theo cell của nhóm PINN và của GBDT chỉ tương quan vừa (Pearson 0{,}50--0{,}80) và ngược '
        'dấu nhau ở 16--31\\,\\% số cell. Áp cùng công thức với ba MLP thay cho ba PINN-semi (MLP-lai), PINN-lai tốt hơn '
        'MLP-lai có ý nghĩa thống kê ở 7/16 trường hợp và kém hơn có ý nghĩa thống kê ở 1, với tỉ số trung bình nhân 0{,}716 ở 10\\,\\% và 1{,}002 ở 70\\,\\% nhãn. '
        'Phần vật lý vẫn đóng góp khi ít nhãn.\n\n'
        'Sau khi có kết quả của bản cố định, đồ án thử thêm PINN-lai tinh chỉnh, chọn $(\\omega, w)$ riêng trong từng fold '
        'chỉ bằng cell xác thực của fold đó (\\Cref{sec:pinnlai}). Bản tinh chỉnh có MAE bằng '
        f'{rng([t.ratio for t in tt])} lần bản cố định, thấp hơn có ý nghĩa thống kê ở {up_t}/16 và cao hơn có ý nghĩa '
        f'thống kê ở {dn_t}/16 trường hợp. So với GBDT, bản tinh chỉnh thấp hơn ở {tcg_lo}/16 trường hợp ({tcg_up} có ý '
        'nghĩa thống kê)' + (f' và cao hơn có ý nghĩa thống kê ở {tcg_dn}. ' if tcg_dn else
                              ' và không cao hơn có ý nghĩa thống kê ở trường hợp nào. ')
        + f'Mỗi fold chỉ có {nv} cell xác thực nên cặp $(\\omega, w)$ chọn được dao động giữa các fold. Bản tinh chỉnh là '
        'phân tích hậu kiểm và không được dùng ở các phần sau.')


def rel_dev():
    """Tỉ số MAE trên tập xác thực của PINN-semi có 16 đặc trưng tương đối (mốc nhân quả) so với PINN-semi gốc, giai
    đoạn phát triển (lần lặp 0, nhãn 10 và 50 %): trung bình nhân qua bộ × mức nhãn. Bản sao ở generated/data/."""
    path = os.path.join(DATA, 'rel_dev.json')
    from pinnsoh import experiments as EX
    jobs = [j for j in EX.E20dev() if j.args.get('tag') == 'REL']
    base = {(j.args['dataset'], j.args['label_frac'], j.args['fold']): j.out for j in EX.E20()
            if j.kind == 'train' and j.args['tag'] == 'P' and j.args['member'] == 0 and j.args['repeat'] == 0}
    if all(os.path.exists(j.out) for j in jobs) and all(os.path.exists(base.get((j.args['dataset'],
            j.args['label_frac'], j.args['fold']), '')) for j in jobs):
        acc = {}
        for j in jobs:
            a = j.args
            rv = lambda p: json.load(open(p))['val_predictions']
            A, B = rv(j.out), rv(base[(a['dataset'], a['label_frac'], a['fold'])])
            for x, s in ((A, 'rel'), (B, 'base')):         # như analysis/dev_e20.py: MAE theo cell của fold, rồi
                y, p, c = np.asarray(x['y']), np.asarray(x['p']), np.asarray(x['cell'])    # trung bình qua các fold
                acc.setdefault((a['dataset'], a['label_frac'], s), []).append(
                    float(np.mean([np.abs(p[c == i] - y[c == i]).mean() for i in np.unique(c)])))
        r = {f'{ds}|{f}': float(np.mean(acc[(ds, f, 'rel')]) / np.mean(acc[(ds, f, 'base')]))
             for ds, f, s in acc if s == 'rel'}
        json.dump(dict(ratios=r, geo=float(np.exp(np.mean(np.log(list(r.values())))))), open(path, 'w'), indent=1)
    return json.load(open(path)) if os.path.exists(path) else None


def kalman_text(K, F):
    """Đoạn về cập nhật khi cell mới được đo (bộ lọc Kalman, GBDT-cập nhật), từ analysis/analyze_e32.py với L tính
    trên cell huấn luyện của fold và PINN-lai cố định."""
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    q = lambda ds, f, t, m: float(K[(K.ds == ds) & np.isclose(K.f, f) & (K.tier == t)][m].iloc[0])
    kal = [q(ds, 0.7, 'q10', 'kalman') for ds in DS]
    vs_g = [q(ds, 0.7, 'q10', 'kalman') / fval(F, ds, 0.7, 'gbdt') for ds in DS]
    cases = [(ds, f, t) for ds in DS for f in FR for t in ('q20', 'q10', 'q05')]
    r_k = [q(ds, f, t, 'capnhat') / q(ds, f, t, 'kalman') for ds, f, t in cases]
    r_h = [q(ds, f, t, 'capnhat') / q(ds, f, t, 'giu_nguyen') for ds, f, t in cases]
    n_k = sum(x < 1 for x in r_k); n_h = sum(x < 1 for x in r_h)
    return ('Khi cell mới cũng được đo theo lịch $0{,}1L$ (1--3\\,\\% số chu kỳ của nó) và ước lượng của PINN-lai được cập '
            'nhật bằng bộ lọc Kalman, MAE trên các chu kỳ không được đo còn '
            f'{rng([1e3 * x for x in kal], 1)} ($\\times10^{{-3}}$), bằng {rng(vs_g)} lần GBDT học với 70\\,\\% số cell có '
            'nhãn đầy đủ nhưng không đo cell mới (cột cuối của Bảng~\\ref{tab:e28}). GBDT-cập nhật, mô hình học thẳng '
            f'thay đổi SOH kể từ lần đo gần nhất, thấp hơn bộ lọc Kalman ở {n_k}/{len(cases)} trường hợp bộ $\\times$ mức '
            f'nhãn $\\times$ lịch đo (tỉ số {rng(r_k)}) và thấp hơn việc giữ nguyên giá trị đo gần nhất ở '
            f'{n_h}/{len(cases)} trường hợp (tỉ số {rng(r_h)}). Trong cả hai cách, lịch đo của cell kiểm tra dùng thang đời '
            '$L$ tính trên cell huấn luyện của fold.'), \
        dict(n_k=n_k, n_h=n_h, n=len(cases), r_k=r_k, r_h=r_h, kal=kal, vs_g=vs_g)


def e32_load():
    p = keep_copy('results/e32_tiers.csv', 'e32_tiers.csv')
    return pd.read_csv(p) if p is not None else None


def thesis_tables(F=None, K=None):
    """Bảng chép từ báo cáo đã nộp, kèm hàng độ đo và số tốt nhất in đậm. F: phân tích theo fold của PINN-lai và
    HiPINN-lai (thay hai cột cuối của bảng MAE chính, hai hàng của bảng RMSE, bảng lịch sử); K: phân tích cập nhật
    (thay cột Kalman của bảng ngân sách đo)."""
    MAE3 = r'MAE theo cell ($\times10^{-3}$) ' + DN
    write('bang_du_lieu.tex', thesis_block('tab:datastats'))
    # MAE của mọi mô hình theo lượng nhãn: in đậm sẵn (thấp nhất trong cột của mỗi bộ)
    big = thesis_block('tab:big')
    write('bang_nhan.tex', metric_row(big, 'Bộ & Mô hình &',
                                      r' & & \multicolumn{4}{c}{' + MAE3 + r'} \\\cmidrule(lr){3-6}'))
    # tỉ số PINN-semi so với MLP và GBDT: tỉ số thấp và tỉ lệ cell tốt hơn cao là tốt cho PINN-semi
    cmp_ = thesis_block('tab:cmp', [
        ('(nhỏ hơn 1: PINN-semi tốt hơn).',
         '(nhỏ hơn 1: PINN-semi tốt hơn); “cell tốt hơn” là tỉ lệ cell kiểm tra mà PINN-semi có sai số thấp hơn. In '
         'đậm: tốt nhất trong mỗi bộ (tỉ số thấp nhất, tỉ lệ cell tốt hơn cao nhất).'),
        (r'Bộ & Nhãn & so với MLP: tỉ số [KTC] & cell tốt hơn & $p_\text{Holm}$ & so với GBDT: tỉ số [KTC] & cell tốt '
         r'hơn & $p_\text{Holm}$ \\',
         r' & & \multicolumn{3}{c}{PINN-semi so với MLP} & \multicolumn{3}{c}{PINN-semi so với GBDT} \\'
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}' '\n'
         r'Bộ & Nhãn & tỉ số MAE [KTC] ' + DN + r' & cell tốt hơn ' + UP + r' & $p_\text{Holm}$ & tỉ số MAE [KTC] ' + DN
         + r' & cell tốt hơn ' + UP + r' & $p_\text{Holm}$ \\')])
    cmp_ = bold_best(cmp_, [2, 5], best=min, groups=True)
    write('bang_ti_so.tex', bold_best(cmp_, [3, 6], best=max, groups=True))
    # tách nguồn lợi ích: thấp nhất mỗi cột
    src = thesis_block('tab:source', [('trong ngoặc là tỉ số so với MLP.',
                                       'trong ngoặc là tỉ số so với MLP; in đậm là thấp nhất mỗi cột.')])
    src = metric_row(src, 'Mô hình (30', r' & \multicolumn{4}{c}{' + MAE3 + r' (tỉ số so với MLP)} \\'
                                          r'\cmidrule(lr){2-5}')
    write('bang_nguon_loi.tex', bold_best(src, [1, 2, 3, 4]))
    if F is not None:
        allmae = allmae_table(F)
    else:
        allmae = thesis_block('tab:allmae', [('Mô hình cuối', 'HiPINN-lai')])
    write('bang_mae_chinh.tex', metric_row(allmae, 'Bộ & Nhãn & MLP',
                                           r' & & \multicolumn{8}{c}{' + MAE3 + r'} \\\cmidrule(lr){3-10}'))
    if F is not None:
        rm = allrm_table(F, MAE3)
        write('bang_rmse.tex', rm)
        write('doan_rmse.tex', rmse_text(F, rm))
        write('bang_lich_su.tex', history_table_f(F) + '\n\n' + history_text_f(F))
    # ngân sách đo: so năm cột ước lượng; cột cuối có đo thêm cell kiểm tra nên không so cùng
    e28 = thesis_block('tab:e28', [
        ('Cột cuối: đo thêm cell test theo cùng lịch, cập nhật bằng bộ lọc Kalman, sai số trên các chu kỳ không được đo.',
         'Cột cuối: PINN-lai học với 70\\,\\% số cell, đo thêm cell kiểm tra mỗi $0{,}1L$ chu kỳ và cập nhật bằng bộ lọc '
         'Kalman; sai số trên các chu kỳ không được đo; lịch đo dùng $L$ tính trên cell huấn luyện của fold. In đậm là '
         'thấp nhất trong năm cột đầu của mỗi hàng; cột cuối có đo thêm cell kiểm tra nên không so cùng.'),
        (r'Bộ & PINN-lai, 30\,\% cell & PINN-lai, 70\,\% cell & GBDT, 70\,\% cell & PINN-lai, '
         r'đo mỗi $0{,}1L$ & PINN-lai, đo mỗi $0{,}05L$ & đo cả cell mới mỗi $0{,}1L$ \\',
         'Bộ & ' + ' & '.join(head2(a, b) for a, b in [
             ('PINN-lai,', r'30\,\% cell'), ('PINN-lai,', r'70\,\% cell'), ('GBDT,', r'70\,\% cell'),
             ('PINN-lai, đo', r'mỗi $0{,}1L$'), ('PINN-lai, đo', r'mỗi $0{,}05L$'),
             (r'PINN-lai 70\,\%, đo cả', r'cell mới mỗi $0{,}1L$')]) + r' \\')])
    if K is not None:
        head_, lines, tail = _rows_of(e28)
        for i, ln in enumerate(lines):
            if ' & ' in ln and ln.rstrip().endswith('\\\\'):
                cells = [c.strip() for c in ln.rstrip()[:-2].split(' & ')]
                ds = cells[0]
                if F is not None:        # ba cột đầu là PINN-lai cố định và GBDT: phải trùng phân tích theo fold
                    for col, f, m in ((1, 0.3, 'pinn_lai'), (2, 0.7, 'pinn_lai'), (3, 0.7, 'gbdt')):
                        assert abs(_num(cells[col]) - 1e3 * fval(F, ds, f, m)) < 0.006, ('bảng ngân sách đo', ds, f, m)
                cells[-1] = v(1e3 * float(K[(K.ds == ds) & np.isclose(K.f, 0.7) & (K.tier == 'q10')].kalman.iloc[0]))
                lines[i] = ' & '.join(cells) + ' \\\\'
        e28 = head_ + '\\midrule' + '\n'.join(lines) + '\\bottomrule' + tail
    e28 = metric_row(e28, r'Bộ & \begin{tabular}', r' & \multicolumn{6}{c}{' + MAE3 + r'} \\\cmidrule(lr){2-7}')
    write('bang_ngan_sach.tex', bold_best(e28, [1, 2, 3, 4, 5], by='row'))
    write('bang_du_bao_tham_so.tex', fcset_table())


def fcset_table():
    """Tham số của từng bộ cho bài toán dự báo, lấy từ bảng của báo cáo đã nộp, bỏ cột ba tầm báo cáo của E22."""
    blk = thesis_block('tab:fcset', [
        ('$L$ là trung vị số chu kỳ ghi được, $\\tau$ là ngưỡng cuối đời.',
         '$L$ là trung vị số chu kỳ ghi được của mọi cell trong bộ (khi học, $L$ tính lại trên cell train của từng '
         'fold), $\\tau$ là ngưỡng cuối đời.'),
        ('{lrrcrrr}', '{lrrrrr}'), (' & Tầm $0{,}1L$, $0{,}25L$, $0{,}5L$', '')])
    out = re.sub(r'^(\w+ & [^&]+ & [^&]+) & [0-9, ]+ &', r'\1 &', blk, flags=re.M)
    assert out.count('\n') == blk.count('\n') and len(re.findall(r'^\w+ & ', out, flags=re.M)) == 5, out
    return out


# --------------------------------------------------------------------------- độ nhạy của bộ lọc ngoại lai
def filter_text():
    """Câu về độ nhạy của bộ lọc ngoại lai nhân quả (Mục Tiền xử lý), từ results/filter_sensitivity.csv
    (analysis/filter_sensitivity.py); bản sao ở generated/data/."""
    p = keep_copy('results/filter_sensitivity.csv', 'filter_sensitivity.csv')
    if p is None:
        return ''
    R = pd.read_csv(p)
    drop = lambda name: [100 * float(R[(R.setting == name) & (R.ds == ds)]['drop'].iloc[0]) for ds in DS]
    K = R[R.setting.isin(['k3', 'k4', 'k6', 'k8'])]
    N = R[R.setting == 'none']
    H = R[R.setting.isin(['h5', 'h20'])]
    sig = lambda X: int((X.p_holm < 0.05).sum())
    worse = lambda X: int(((X.p_holm < 0.05) & (X.ratio > 1)).sum())
    pc = lambda xs: rng(xs, 1) + r'\,\%'

    def updown(X, paren=False):
        up = int(((X.p_holm < 0.05) & (X.ratio > 1)).sum()); dn = int(((X.p_holm < 0.05) & (X.ratio < 1)).sum())
        if not (up or dn):
            t = 'không trường hợp nào khác có ý nghĩa'
        else:
            t = ', '.join(([f'cao hơn có ý nghĩa ở {up} trên {len(X)} trường hợp'] if up else [])
                          + ([f'thấp hơn có ý nghĩa ở {dn}'] if dn else ['không trường hợp nào thấp hơn có ý nghĩa']))
        return f' ({t})' if paren else ', ' + t
    return (r'Ở cách đặt này, tỉ lệ chu kỳ bị loại là '
            + ', '.join(f'{v(x, 1)}' + r'\,\%' + f' ({ds})' for ds, x in zip(DS, drop('k5')))
            + r'. Một kiểm tra sau, không dùng để chọn tham số (GBDT, 30\,\% nhãn, 3 lần lặp $\times$ 5 fold, MAE chấm '
            r'trên cùng một tập chu kỳ kiểm tra cho mọi cách đặt, Wilcoxon ghép cặp theo cell, Holm qua bốn bộ): đổi ngưỡng thành 3, 4, 6 hoặc 8 làm tỉ lệ chu kỳ bị loại '
            r'đi từ ' + pc(drop('k3')) + r' (ngưỡng 3) tới ' + pc(drop('k8')) + r' (ngưỡng 8), còn MAE bằng '
            + rng(list(K.ratio), 3) + ' lần cách đặt của đồ án'
            + updown(K)
            + r'; bỏ hẳn bước lọc cho MAE bằng ' + rng(list(N.ratio), 3) + ' lần'
            + (f' (cao hơn có ý nghĩa ở {worse(N)} trên {len(N)} bộ)' if worse(N) else '')
            + r'; đổi số chu kỳ lịch sử tối thiểu thành 5 hoặc 20 cho ' + rng(list(H.ratio), 3) + ' lần'
            + updown(H, paren=True) + '.')


# --------------------------------------------------------------------------- bảng thí nghiệm
EXPS = [(['E13'], 'Kiểm định chéo chính: MLP, PINN-semi ở 4 mức nhãn; PINN-sup và hai biến thể lược bỏ thành phần ở 30\\,\\%'),
        (['E15'], 'Kiểm toán rò rỉ: 4 cách chia và chuẩn hoá sai, MLP 70\\,\\% nhãn'),
        (['E16'], 'Ridge, GPR, GBDT ở 4 mức nhãn'),
        (['E17'], 'PINN-sup ở 10 và 50\\,\\%; PINN hộp đen ở 4 mức nhãn'),
        (['E20dev', 'E20'], 'PINN-lai: 9 hướng cải tiến chọn trên validation; 3 PINN-semi, GBDT, 3 lần lặp, 4 mức nhãn'),
        (['E30dev', 'E30'], 'CNN, LSTM, Transformer: chọn độ dài cửa sổ trên validation; 3 lần lặp, 4 mức nhãn'),
        (['E29dev', 'E29'], 'Đầu vào có lịch sử của cell: chọn dạng trên validation; PINN-semi, MLP, GBDT'),
        (['E28'], 'Ngân sách đo: đo thưa mọi cell train theo 4 lịch đo'),
        (['E32'], 'GBDT-cập nhật: cập nhật SOH từ lần đo gần nhất'),
        (['E33'], 'Dự báo SOH, dung lượng, RUL: HiGBDT, HiPINN, các nhánh và mô hình đối chứng, 2 ngân sách nhãn, '
                  '3 lần lặp'),
        (['E34'], 'HiGBDT cho bài toán cập nhật: GBDT-cập nhật, HiGBDT-$\\Delta$, -R, -V trên 4 lịch đo'),
        (['E27'], 'PINN-lai, GBDT và PINN-semi trên đúng 39 nhóm của PINN4SOH, 2 cách tiền xử lý, 3 hạt giống')]


def experiments_table(n_rerun, n_feat_jobs, n_feat_fits):
    from pinnsoh.experiments import EXPERIMENTS
    out = [r'\begin{table}[!t]\centering\small',
           r'\caption[Các thí nghiệm có kết quả trong báo cáo]{Các thí nghiệm có kết quả trong báo cáo và số lượt huấn '
           r'luyện (mỗi lượt là một mô hình trên một fold, hoặc một lần chạy trên một nhóm; lượt trùng cấu hình chỉ tính '
           r'một lần).}\label{tab:exps}',
           r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r}\toprule',
           r'Thí nghiệm & Lượt \\\midrule']
    def sig(j):                                    # cấu hình của một lượt, không kể thư mục ghi kết quả
        a = dict(j.args); a.pop('out_dir', None)
        return j.kind, json.dumps(a, sort_keys=True, default=str)
    tot = 0
    for names, desc in EXPS:
        n, seen = 0, set()
        for x in names:                             # lượt trùng cấu hình với thí nghiệm trước cùng hàng chỉ tính một lần
            jobs = EXPERIMENTS[x]()
            n += sum(sig(j) not in seen for j in jobs)
            seen |= {sig(j) for j in jobs}
        tot += n
        out.append(f'{desc} & {thou(n)} \\\\')
    out.append(f'Mã gốc PINN4SOH trên 39 nhóm đánh giá của bài, 10 lần mỗi nhóm & {thou(n_rerun)} \\\\')
    tot += n_rerun
    if n_feat_jobs:
        out.append(f'Phân tích đặc trưng: {n_feat_jobs} fold $\\times$ {n_feat_fits // n_feat_jobs}--'
                   f'{-(-n_feat_fits // n_feat_jobs)} lần huấn luyện GBDT với các tập đầu vào khác nhau & {thou(n_feat_fits)} \\\\')
        tot += n_feat_fits
    fs = 'results/filter_sensitivity.csv'
    fs = fs if os.path.exists(fs) else os.path.join(DATA, 'filter_sensitivity.csv')
    if os.path.exists(fs):
        R = pd.read_csv(fs)
        alt = R[R.setting != 'k5']                     # cách đặt của đồ án trùng GBDT của E16: không tính lại
        n_fs = int(alt.n_fits.sum()) if 'n_fits' in R else 0
        if n_fs:
            out.append(f'Độ nhạy của bộ lọc ngoại lai: GBDT với {alt.setting.nunique()} cách đặt khác của bộ lọc, 4 bộ, '
                       f'30\\,\\% nhãn & {thou(n_fs)} \\\\')
            tot += n_fs
    out += [r'\midrule', f'Tổng & {thou(tot)} \\\\', r'\bottomrule\end{tabularx}\end{table}']
    write('bang_thi_nghiem.tex', '\n'.join(out))
    return tot


# --------------------------------------------------------------------------- kiểm toán rò rỉ
LEAK = [('row', 'chia theo chu kỳ'), ('minmax', 'min--max theo cell'), ('life3s', 'lọc dùng cả vòng đời'),
        ('normtest', 'chuẩn hoá gồm cả tập kiểm tra')]
LEAK_HEAD = {'row': ('chia theo', 'chu kỳ'), 'minmax': ('min--max', 'theo cell'), 'life3s': ('lọc dùng cả', 'vòng đời'),
             'normtest': ('chuẩn hoá gồm', 'cả tập kiểm tra')}


def leakage():
    path = keep_copy('results/thesis_leakage.csv', 'leakage.csv')
    T = pd.read_csv(path).set_index('bo').reindex(DS)
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Kiểm toán rò rỉ]{Kiểm toán rò rỉ: MAE theo cell của cùng một MLP (70\,\% nhãn, 2 lần lặp $\times$ 5 '
           r'fold) theo giao thức không rò rỉ, và tỉ số giữa MAE dưới mỗi cách làm sai với MAE đó (nhỏ hơn 1: sai số trông '
           r'thấp hơn thực tế). Cột cuối: chia theo chu kỳ rồi chép nhãn của chu kỳ train gần nhất trong cùng cell, không '
           r'dùng đặc trưng nào. Bảng đo độ lệch của đánh giá chứ không so mô hình, nên không có chiều tốt hơn và không '
           r'in đậm: tỉ số càng xa 1, cách làm sai càng làm lệch con số báo cáo.}\label{tab:leak}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{lrrrrrr}\toprule',
           r' & & \multicolumn{5}{c}{tỉ số MAE dưới cách làm sai so với MAE không rò rỉ} \\\cmidrule(lr){3-7}',
           r'Bộ & ' + head2('MAE không', r'rò rỉ ($\times10^{-3}$)') + ' & '
           + ' & '.join(head2(*LEAK_HEAD[k]) for k, _ in LEAK) + ' & ' + head2('chép', 'nhãn') + r' \\\midrule']
    for ds, r in T.iterrows():
        out.append(f'{ds} & {v(1e3 * r.chuan)} & ' + ' & '.join(v(r[f"ti_so_{s}"]) for s, _ in LEAK)
                   + f' & {v(r.ti_so_noi_suy_hang)} \\\\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    lo = lambda s: T[f'ti_so_{s}'].min(); hi = lambda s: T[f'ti_so_{s}'].max()
    m = T['ti_so_minmax']
    L = dict(row=f'{v(1 / hi("row"), 1)}--{v(1 / lo("row"), 1)}', copy=f'{v(lo("noi_suy_hang"))}--{v(hi("noi_suy_hang"))}',
             minmax=m, life=rng(T['ti_so_life3s']), normtest=rng(T['ti_so_normtest']),
             minmax_rng=f'{v(m.min())}--{v(m.max())}')
    text = (f'Bảng~\\ref{{tab:leak}} cho thấy mức độ của từng dạng rò rỉ khi chỉ đổi đúng một '
            f'yếu tố so với giao thức chuẩn. Chia theo chu kỳ làm sai số thấp đi {L["row"]} lần dù mô hình và dữ liệu '
            f'không đổi. Bộ “dự đoán” không dùng đặc trưng nào, chỉ chép nhãn của chu kỳ train gần nhất, có sai số '
            f'bằng {L["copy"]} lần mô hình thật được đánh giá đúng, đúng như Mệnh đề~\\ref{{md:rowleak}}. Chuẩn hoá '
            f'min--max theo cả vòng đời cell, như trong mã PINN4SOH, làm sai số thấp giả tạo trên XJTU ({v(m["XJTU"])}) '
            f'và HUST ({v(m["HUST"])}), nơi SOH cuối đời của mọi cell gần như bằng nhau nên vị trí trong vòng đời gần '
            f'như đủ để suy ra SOH; trên TJU ({v(m["TJU"])}) và MIT ({v(m["MIT"])}) nó làm sai số tăng vì xoá mất mức '
            f'tuyệt đối của đặc trưng. Dù theo chiều nào, con số đo được cũng không phản ánh sai số khi triển khai, vì '
            f'tuổi thọ của cell đang vận hành chưa biết trước. Lọc ngoại lai dùng cả vòng đời ({L["life"]}) và chuẩn hoá '
            f'gồm cả tập kiểm tra ({L["normtest"]}) tác động nhỏ hơn nhưng vẫn là rò rỉ.')
    return '\n'.join(out), text, L


# --------------------------------------------------------------------------- so với PINN4SOH
def bench_load():
    S = pd.read_csv(f'{BENCH}/summary.csv').set_index(['dataset', 'mode'])
    T = pd.read_csv(f'{BENCH}/tests.csv')
    L = json.load(open(f'{BENCH}/logged_constant.json'))
    return S, T, L


def _have(S, key, col):
    c = f'{col}_complete'
    return col in S.columns and pd.notna(S.loc[key, col]) and (c not in S.columns or bool(S.loc[key, c]))


def wins(T, a, b, keys):
    g = T[(T.a == a) & (T.b == b)]
    g = g[[(d, m) in keys for d, m in zip(g.dataset, g['mode'])]]
    return int(((g.p_holm < 0.05) & (g.ratio < 1)).sum()), int(((g.p_holm < 0.05) & (g.ratio > 1)).sum())


def sig(T, a, b, key):
    g = T[(T.a == a) & (T.b == b) & (T.dataset == key[0]) & (T['mode'] == key[1])]
    return None if g.empty else (float(g.p_holm.iloc[0]), float(g.ratio.iloc[0]))


def setting(key):
    return 'đầy đủ' if key[1] == 'full' else f'$K = {key[1][1]}$'


def bench_parts(S, T, L):
    full = [k for k in ROWS if k[1] == 'full']
    few = [k for k in ROWS if k[1] != 'full']
    have = lambda k, c: _have(S, k, c)
    ratio = lambda a, b, ks: [S.loc[k, a] / S.loc[k, b] for k in ks if have(k, a) and have(k, b)]
    nruns = lambda col: int(np.nanmin([S.loc[k, f'{col}_runs'] for k in ROWS if have(k, col)]))
    n_const, n_eq = sum(x['constant'] for x in L), sum(x['equal'] for x in L)
    ex = next(x for x in L if x['key'] == 'XJTU|full|0-0')
    rep_full, rep_few = ratio('rerun_published', 'published_paper', full), ratio('rerun_published', 'published_paper', few)
    rep_sig = wins(T, 'rerun_published', 'published', full)
    sd_rel = [S.loc[k, 'rerun_published_sd'] / S.loc[k, 'rerun_published'] for k in ROWS if have(k, 'rerun_published')]
    xj = [k for k in few if k[0] == 'XJTU' and k[1] != 'n1']
    main_keys = [k for k in full + few if k not in xj and k != ('XJTU', 'n1')]
    cur_main, cur_xjtu = ratio('rerun_current', 'rerun_published', main_keys), ratio('rerun_current', 'rerun_published', xj)
    cur_sig_main, cur_sig_xjtu = sum(wins(T, 'rerun_current', 'rerun_published', main_keys)), \
        wins(T, 'rerun_current', 'rerun_published', xj)
    B = {}
    B['hyb_full'], B['hyb_few'] = ratio('paper_pinn_lai', 'rerun_published', full), ratio('paper_pinn_lai', 'rerun_published', few)
    B['hyb_pub_full'], B['hyb_pub_few'] = ratio('paper_pinn_lai', 'published_paper', full), \
        ratio('paper_pinn_lai', 'published_paper', few)
    B['w_full'], B['l_full'] = wins(T, 'paper_pinn_lai', 'rerun_published', full)
    B['w_few'], B['l_few'] = wins(T, 'paper_pinn_lai', 'rerun_published', few)
    B['gb_vs_hyb'] = ratio('paper_gbdt', 'paper_pinn_lai', full + few)
    B['gw'], B['gl'] = wins(T, 'paper_gbdt', 'paper_pinn_lai', full + few)
    B['gb_sig_keys'] = [f'{d} {setting((d, m))}' for d, m in full + few
                        if (s := sig(T, 'paper_gbdt', 'paper_pinn_lai', (d, m))) and s[0] < 0.05 and s[1] < 1]
    B['strict'] = {k: S.loc[k, 'strict_pinn_lai'] / S.loc[k, 'published_paper'] for k in ROWS if have(k, 'strict_pinn_lai')}
    B['sw'], B['sl'] = wins(T, 'strict_pinn_lai', 'published', full)
    B['rep_full'], B['rep_few'], B['rep_sig'] = rep_full, rep_few, rep_sig
    B['n_groups'], B['n_const'], B['n_eq'] = len(L), n_const, n_eq
    B['runs'], B['runs_cur'] = nruns('rerun_published'), nruns('rerun_current')
    B['beat_rerun'] = sum(1 for k in ROWS if have(k, 'paper_pinn_lai') and S.loc[k, 'paper_pinn_lai'] < S.loc[k, 'rerun_published'])
    B['beat_pub'] = [k for k in ROWS if S.loc[k, 'paper_pinn_lai'] < S.loc[k, 'published_paper']]
    B['lose_pub'] = [k for k in ROWS if S.loc[k, 'paper_pinn_lai'] >= S.loc[k, 'published_paper']]
    table_text = ''
    return table_text, B


def _vn_ticks():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ','))


VN_TICKS = _vn_ticks()


BENCH_SERIES = [('published_paper', 'PINN4SOH, số công bố', '#9aa5ad', (0, (3, 1.6)), 'o'),
                ('rerun_published', 'PINN4SOH, chạy lại', '#eb6834', '-', 's'),
                ('paper_pinn_lai', 'PINN-lai, tiền xử lý của bài', '#2a78d6', '-', 'o'),
                ('strict_pinn_lai', 'PINN-lai, không rò rỉ', '#0b3d73', '-', '^')]


def fig_benchmark(S):
    """Hình so với PINN4SOH: (a) chế độ đầy đủ, cột nhóm theo bộ trên một trục chung, ghi số trên đầu cột;
    (b), (c) chế độ ít mẫu của XJTU và HUST theo số cell có nhãn K. Một chú giải chung ở trên cùng."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    from pinnsoh.plotstyle import apply_theme, save, INK, INK2, GRID
    apply_theme(base=8.5)
    fig = plt.figure(figsize=(6.3, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.22)
    n = len(BENCH_SERIES)
    w = 0.19
    ax = fig.add_subplot(gs[0, :])
    for j, (c, lab, color, _, _) in enumerate(BENCH_SERIES):
        xs = np.arange(len(DS)) + (j - (n - 1) / 2) * (w + 0.012)
        ys = [1e3 * S.loc[(ds, 'full'), c] for ds in DS]
        ax.bar(xs, ys, width=w, color=color, label=lab, zorder=2, linewidth=0)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.18, f'{y:.1f}'.replace('.', ','), ha='center', va='bottom', fontsize=6.6, color=INK2)
    ax.set_xticks(np.arange(len(DS)))
    ax.set_xticklabels([f'{ds}' for ds in DS], fontweight='bold', color=INK)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.55, len(DS) - 0.45)
    top = max(1e3 * S.loc[(ds, 'full'), c] for ds in DS for c, *_ in BENCH_SERIES)
    ax.set_ylim(0, top * 1.14)
    ax.yaxis.set_major_formatter(VN_TICKS)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.grid(axis='x', visible=False); ax.set_axisbelow(True)
    ax.set_ylabel(r'MAE theo cell ($\times10^{-3}$)')
    ax.set_title('(a) Chế độ đầy đủ', loc='left', fontweight='bold', fontsize=9)
    for i, ds in enumerate(('XJTU', 'HUST')):
        bx = fig.add_subplot(gs[1, i])
        K = [1, 2, 3, 4]
        for c, lab, color, ls, mk in BENCH_SERIES:
            bx.plot(K, [1e3 * S.loc[(ds, f'n{k}'), c] for k in K], color=color, ls=ls, marker=mk, ms=4, lw=1.4,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=3)
        bx.set_yscale('log')
        lo, hi = bx.get_ylim()
        bx.yaxis.set_major_locator(FixedLocator([t for t in (5, 7, 10, 15, 20, 30, 50) if lo <= t <= hi]))
        bx.yaxis.set_minor_locator(NullLocator())
        bx.yaxis.set_major_formatter(VN_TICKS)
        bx.set_xticks(K); bx.set_xlim(0.8, 4.2)
        bx.grid(axis='y', color=GRID, lw=0.6); bx.grid(axis='x', visible=False)
        bx.set_xlabel('số cell có nhãn $K$' + (' mỗi nhóm thí nghiệm' if ds == 'XJTU' else ''))
        bx.set_title(f'({"bc"[i]}) {ds}, chế độ ít mẫu', loc='left', fontweight='bold', fontsize=9)
        if i == 0:
            bx.set_ylabel(r'MAE theo cell ($\times10^{-3}$, thang log)')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01), handlelength=1.2,
               columnspacing=1.4, fontsize=7.8)
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.09, right=0.99)
    save(fig, os.path.join(FIG, 'fig_benchmark'))
    plt.close(fig)


FIG_BENCH = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_benchmark.pdf}' '\n'
             r'\caption[So với PINN4SOH trên đúng cách chia của bài]{MAE theo cell trên đúng các nhóm đánh giá của bài '
             r'PINN4SOH. (a) Chế độ đầy đủ, mỗi bộ bốn cột: số công bố của bài, mã gốc của bài chạy lại, PINN-lai với tiền '
             r'xử lý của bài và PINN-lai với tiền xử lý không rò rỉ. (b), (c) Chế độ ít mẫu: MAE theo số cell có nhãn $K$.}'
             r'\label{fig:bench}' '\n' r'\end{figure}')


def bench_section(S, T, L):
    """Số liệu so với PINN4SOH (dùng ở phần so sánh, kết luận, tóm tắt); hình vẽ ở fig_benchmark."""
    _, B = bench_parts(S, T, L)
    return B


def compare_table(S, T):
    """Bảng so với PINN4SOH: số công bố của bài (PINN4SOH, MLP, CNN), mã gốc chạy lại, PINN-lai của đồ án với hai cách
    tiền xử lý. In đậm: thấp nhất trong năm cột cùng điều kiện; dấu: PINN-lai (tiền xử lý của bài) so với mã gốc chạy lại."""
    pub = {k: PAPER_MAE[k] for k in ROWS}
    mark = {}
    for r in T[(T.a == 'paper_pinn_lai') & (T.b == 'rerun_published')].itertuples():
        if r.p_holm < 0.05:
            mark[(r.dataset, r.mode)] = '*' if r.ratio < 1 else r'\dagger'
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{5pt}',
           r'\caption[So với PINN4SOH trên đúng cách chia của bài]{MAE theo cell ($\times10^{-3}$) trên đúng 39 nhóm đánh '
           r'giá của bài PINN4SOH, trung bình qua cell test, lần chạy và nhóm như bài; $K$ là số cell có nhãn trong thí '
           r'nghiệm ít mẫu. Ba cột đầu là số công bố của bài cho PINN4SOH và hai mô hình đối chứng MLP, CNN của bài. Cột '
           r'chạy lại: mã gốc của bài chạy trên đúng các nhóm đó, siêu tham số theo nhật ký của bài, 10 lần mỗi nhóm. Hai '
           r'cột cuối: PINN-lai của đồ án (3 hạt giống) với tiền xử lý của bài, cùng điều kiện với các cột trước, và với '
           r'tiền xử lý không rò rỉ. In đậm: thấp nhất trong năm cột cùng điều kiện. $^{*}$/$^{\dagger}$: PINN-lai với '
           r'tiền xử lý của bài thấp hơn/cao hơn có ý nghĩa so với mã gốc chạy lại (Wilcoxon ghép cặp theo cell test, Holm '
           r'qua 12 hàng).}\label{tab:pub}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{llrrrrrr}\toprule',
           r' & & \multicolumn{6}{c}{MAE theo cell ($\times10^{-3}$) ' + DN + r'} \\\cmidrule(lr){3-8}',
           r' & & \multicolumn{3}{c}{số công bố của bài} & PINN4SOH & \multicolumn{2}{c}{PINN-lai của đồ án} \\'
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}',
           r'Bộ & Chế độ & PINN4SOH & MLP & CNN & chạy lại & tiền xử lý của bài & không rò rỉ \\\midrule']
    for key in ROWS:
        vals = [pub[key]['PINN4SOH'], pub[key]['MLP'], pub[key]['CNN'], S.loc[key, 'rerun_published'],
                S.loc[key, 'paper_pinn_lai']]
        best = int(np.nanargmin(vals))
        cells = [(r'\textbf{%s}' if i == best else '%s') % v(1e3 * x) for i, x in enumerate(vals)]
        if key in mark:
            cells[-1] += f'$^{{{mark[key]}}}$'
        cells.append(v(1e3 * S.loc[key, 'strict_pinn_lai']))
        out.append(f'{key[0] if key[1] in ("full", "n1") else ""} & {setting(key)} & ' + ' & '.join(cells) + r' \\')
        if key in (('HUST', 'full'), ('XJTU', 'n4')):
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    return '\n'.join(out)


# --------------------------------------------------------------------------- đặc trưng
NAMES_VI = {'voltage mean': 'trung bình điện áp', 'voltage std': 'độ lệch chuẩn điện áp',
            'voltage kurtosis': 'độ nhọn điện áp', 'voltage skewness': 'độ bất đối xứng điện áp',
            'voltage slope': 'độ dốc điện áp', 'voltage entropy': 'entropy điện áp', 'current mean': 'trung bình dòng điện',
            'current std': 'độ lệch chuẩn dòng điện', 'current kurtosis': 'độ nhọn dòng điện',
            'current skewness': 'độ bất đối xứng dòng điện', 'current slope': 'độ dốc dòng điện',
            'current entropy': 'entropy dòng điện', 'CC Q': 'điện lượng pha CC', 'CC charge time': 'thời gian pha CC',
            'CV Q': 'điện lượng pha CV', 'CV charge time': 'thời gian pha CV', 't': '$t$ (chỉ số chu kỳ)'}
GROUP_OF = {**{n: 'V' for n in ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'voltage slope',
                                 'voltage entropy']},
            **{n: 'I' for n in ['current mean', 'current std', 'current kurtosis', 'current skewness', 'current slope',
                                 'current entropy']},
            **{n: 'QT' for n in ['CC Q', 'CC charge time', 'CV Q', 'CV charge time']}, 't': 't'}
ORDER = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'voltage slope', 'voltage entropy',
         'current mean', 'current std', 'current kurtosis', 'current skewness', 'current slope', 'current entropy',
         'CC Q', 'CC charge time', 'CV Q', 'CV charge time', 't']
VARIANTS = [('no_t', 'bỏ $t$'), ('t_only', 'chỉ $t$'), ('no_V', 'bỏ nhóm V'), ('no_I', 'bỏ nhóm I'), ('no_QT', 'bỏ nhóm QT'),
            ('QT_t', 'chỉ QT và $t$'), ('rel', 'thêm 16 đặc trưng tương đối'), ('prune', 'chọn trên validation')]


def features_load():
    F = {}
    for name in ('relevance', 'redundancy', 'ablation', 'permutation', 'prune'):
        p = keep_copy(f'results/features_{name}.csv', f'features_{name}.csv')
        if p is None:
            return None
        F[name] = pd.read_csv(p)
    return F


def _ratio_cell(r):
    """Tỉ số so với bộ đầy đủ, kèm dấu * (thấp hơn có ý nghĩa) hoặc † (cao hơn có ý nghĩa) sau Holm."""
    m = ''
    if pd.notna(r.p_holm) and r.p_holm < 0.05:
        m = '$^{*}$' if r.ratio < 1 else '$^{\\dagger}$'
    return v(r.ratio) + m


def features_parts(F, L_leak, rel_info=None, F20=None):
    R, D, A, P, K = F['relevance'], F['redundancy'], F['ablation'], F['permutation'], F['prune']
    fracs = sorted(A.frac.unique())
    # ---------------- liên quan và trùng lặp
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption[Tương quan hạng giữa từng đầu vào và SOH]{Hệ số Spearman $\rho$ giữa từng đầu vào và SOH: trung vị '
           r'qua các cell của $\rho$ tính trong từng cell (cột “trong cell”), và $\rho$ trên mọi chu kỳ của bộ (cột '
           r'“gộp”). Dấu âm: đầu vào giảm khi SOH tăng. $|\rho|$ càng lớn, đầu vào càng biến thiên chặt theo SOH; in '
           r'đậm là $|\rho|$ lớn nhất trong 16 đặc trưng sạc ở mỗi cột.}\label{tab:relevance}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'rr' * len(DS) + r'}\toprule',
           r' & \multicolumn{' + str(2 * len(DS)) + r'}{c}{hệ số Spearman $\rho$ với SOH, $|\rho|$ ' + UP + r'} \\'
           r'\cmidrule(lr){2-' + str(1 + 2 * len(DS)) + '}',
           ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{d}}}' for d in DS) + r' \\'
           + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))),
           'Đầu vào & ' + ' & '.join(['trong cell & gộp'] * len(DS)) + r' \\\midrule']
    Rg = R.set_index(['dataset', 'feature'])
    prev = None
    for f in ORDER:
        if prev is not None and GROUP_OF[f] != GROUP_OF[prev]:
            out.append(r'\addlinespace[2pt]')
        out.append(NAMES_VI[f] + ' & ' + ' & '.join(f'{v(Rg.loc[(d, f), "rho_within"])} & {v(Rg.loc[(d, f), "rho_pooled"])}'
                                                    for d in DS) + r' \\')
        prev = f
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    tab_rel = bold_best('\n'.join(out), list(range(1, 1 + 2 * len(DS))), best=max, key=abs,
                        drop_last=len(ORDER) - 16)                     # hàng t không tính
    q = {f: {d: Rg.loc[(d, f), 'rho_within'] for d in DS} for f in ORDER}
    top = lambda d: sorted(ORDER[:16], key=lambda f: -abs(q[f][d]))[:3]
    n90 = {d: sum(abs(q[f][d]) >= 0.9 for f in ORDER[:16]) for d in DS}
    lfp, ni = ['MIT', 'HUST'], ['XJTU', 'TJU']
    flip = [f for f in ORDER[:16] if (all(q[f][d] > 0.3 for d in lfp) and all(q[f][d] < -0.3 for d in ni))
            or (all(q[f][d] < -0.3 for d in lfp) and all(q[f][d] > 0.3 for d in ni))]
    weak = {d: [f for f in ORDER[:16] if abs(q[f][d]) < 0.3] for d in DS}
    cnt = D.groupby(['f1', 'f2']).dataset.apply(list).reset_index()
    every = [r for r in cnt.itertuples() if len(r.dataset) == len(DS)]
    some = [r for r in cnt.itertuples() if len(r.dataset) < len(DS)]
    per_ds = D.groupby('dataset').size().reindex(DS).fillna(0).astype(int)
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    top1 = {d: top(d)[0] for d in DS}
    tq = list(q['t'].values())
    assert max(tq) < -0.9                                     # "gần như tỉ lệ nghịch"
    weak_all = [f for f in ORDER[:16] if all(f in weak[d] for d in DS)]
    ex = sorted(flip, key=lambda f: -min(abs(q[f][d]) for d in DS))[:1]
    flip_txt = ''
    if ex:
        f = ex[0]; up_lfp = q[f]['MIT'] > 0
        flip_txt = (f'Có đặc trưng còn đổi chiều theo hệ hoá học: {NAMES_VI[f]} '
                    + ('tăng' if up_lfp else 'giảm') + f' theo SOH ở hai bộ LFP ($\\rho$ {v(q[f]["MIT"])} và '
                    f'{v(q[f]["HUST"])}) nhưng ' + ('giảm' if up_lfp else 'tăng')
                    + f' ở XJTU và TJU ({v(q[f]["XJTU"])} và {v(q[f]["TJU"])}). ')
    text_rel = (
        'Bảng~\\ref{tab:relevance} cho thấy trong từng cell, chỉ số chu kỳ gần như tỉ lệ nghịch với SOH ($\\rho$ trung vị '
        f'từ {v(min(tq))} đến {v(max(tq))}), điều dễ hiểu vì SOH giảm dần theo thời gian. Nhiều đặc trưng sạc cũng biến '
        'thiên chặt theo SOH; số đặc trưng có $|\\rho|$ trung vị từ 0,9 trở lên là '
        + jv([f'{n90[d]} ở {d}' for d in DS]) + '. Nhưng đặc trưng mạnh nhất không giống nhau giữa các bộ: '
        + jv([f'{NAMES_VI[top1[d]]} ở {d}' for d in DS]) + '. ' + flip_txt
        + ('Không đặc trưng nào yếu ($|\\rho| < 0{,}3$) ở cả bốn bộ, nên không có lý do bỏ một đặc trưng chỉ vì tương '
           'quan thấp.' if not weak_all else
           'Yếu ($|\\rho| < 0{,}3$) ở cả bốn bộ chỉ có ' + jv([NAMES_VI[f] for f in weak_all]) + '.')
        + '\n\n'
        'Một số cặp đặc trưng gần như trùng nhau ($|\\rho| \\ge 0{,}95$ trên mọi chu kỳ của bộ): '
        + jv([f'{per_ds[d]} cặp ở {d}' for d in DS]) + ', nhưng '
        + (('chỉ có cặp ' if len(every) == 1 else f'chỉ có {len(every)} cặp ')
           + jv([f'{NAMES_VI[r.f1]} và {NAMES_VI[r.f2]}' for r in every]) + ' trùng ở cả bốn bộ. ' if every else
           'không cặp nào trùng ở cả bốn bộ. ')
        + 'Một cặp trùng ở bộ này lại mang hai thông tin khác nhau ở bộ khác, nên với một tập đầu vào chung không thể bỏ '
        'cột nào chỉ vì trùng lặp. Trùng lặp cũng làm phép hoán vị từng cột đánh giá thấp tầm quan trọng của mỗi cột trong '
        'cặp, vì vậy các phép thử dưới đây bỏ và hoán vị theo nhóm. Ngoài ra, tương quan đơn biến không cho biết một đặc '
        'trưng có giúp phân biệt các cell khi đi cùng các đặc trưng khác hay không, nên việc giữ hay bỏ phải được thử trực '
        'tiếp bằng mô hình.')
    # ---------------- bỏ / thêm nhóm
    Ai = A.set_index(['dataset', 'frac', 'variant'])
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption[Bỏ hoặc thêm nhóm đầu vào với GBDT]{Bỏ hoặc thêm nhóm đầu vào: GBDT học lại với từng tập đầu vào '
           r'trên đúng các fold không rò rỉ (3 lần lặp $\times$ 5 fold), ở 10 và 70\,\% nhãn. Mỗi hàng là một tập đầu vào; '
           r'các nhóm V, I, QT và $t$ như \Cref{tab:features}. Hàng đầu: MAE theo cell ($\times10^{-3}$) với đủ 17 đầu '
           r'vào. Các hàng sau: MAE của tập đó chia cho MAE với đủ 17 đầu vào; lớn hơn 1 là kém hơn, tức phần bị bỏ có '
           r'ích; nhỏ hơn 1 là tốt hơn. $^{*}$: thấp hơn có ý nghĩa, $^{\dagger}$: cao hơn có ý nghĩa (Wilcoxon ghép cặp '
           r'theo cell, Holm qua bốn bộ, $p < 0{,}05$); không có dấu: khác biệt không có ý nghĩa. Hàng thêm $1/T$ chỉ có '
           r'ở TJU nên giá trị $p$ của nó không cần hiệu chỉnh. In đậm là tỉ số thấp nhất mỗi cột, tức tập đầu vào tốt '
           r'nhất ở trường hợp đó.}\label{tab:ablation}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'rr' * len(DS) + r'}\toprule',
           ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{d}}}' for d in DS) + r' \\'
           + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))),
           'Tập đầu vào & ' + ' & '.join([' & '.join(f'{int(100 * f)}\\,\\%' for f in fracs)] * len(DS)) + r' \\\midrule',
           group_row(r'MAE theo cell ($\times10^{-3}$) ' + DN, 1 + len(fracs) * len(DS)),
           'đủ 17 đầu vào & ' + ' & '.join(v(1e3 * Ai.loc[(d, f, 'all'), 'mae']) for d in DS for f in fracs) + r' \\',
           r'\addlinespace[2pt]',
           group_row('tỉ số MAE so với đủ 17 đầu vào ' + DN, 1 + len(fracs) * len(DS))]
    for var, lab in VARIANTS:
        out.append(lab + ' & ' + ' & '.join(_ratio_cell(Ai.loc[(d, f, var)]) for d in DS for f in fracs) + r' \\')
    if ('TJU', fracs[0], 'with_T') in Ai.index:
        out.append(r'thêm $1/T$ (chỉ TJU) & -- & -- & ' + ' & '.join(_ratio_cell(Ai.loc[('TJU', f, 'with_T')]) for f in fracs)
                   + ' & ' + ' & '.join(['--'] * 2 * (len(DS) - 2)) + r' \\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    tab_abl = bold_best('\n'.join(out), list(range(1, 1 + len(fracs) * len(DS))), skip=1)
    # hoán vị theo nhóm: tỉ số nêu trong lời văn
    Pt = P[P.part == 'test'].set_index(['dataset', 'frac', 'feature'])
    # chọn trên validation
    kept = K.groupby(['dataset', 'frac']).n_kept.median()
    drop_counts = {}
    for s in K.dropped.fillna(''):
        for f in filter(None, s.split(';')):
            drop_counts[f] = drop_counts.get(f, 0) + 1
    often = sorted(drop_counts.items(), key=lambda x: -x[1])[:3]
    r_ = lambda var, f=None: [Ai.loc[(d, ff, var), 'ratio'] for d in DS for ff in (fracs if f is None else [f])]
    sig_ = lambda var, sign: [(d, ff) for d in DS for ff in fracs if pd.notna(Ai.loc[(d, ff, var), 'p_holm'])
                              and Ai.loc[(d, ff, var), 'p_holm'] < 0.05 and (Ai.loc[(d, ff, var), 'ratio'] - 1) * sign > 0]
    def lst(ks):
        """[(bộ, mức nhãn)] -> 'XJTU, TJU và HUST với cả hai mức nhãn và ở MIT 70 %'."""
        if not ks:
            return 'không trường hợp nào'
        if len(ks) == len(DS) * len(fracs):
            return 'cả bốn bộ với cả hai mức nhãn'
        join = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
        both = [d for d in DS if all((d, f) in ks for f in fracs)]
        single = [f'{d} {int(100 * f)}\\,\\%' for d, f in ks if d not in both]
        parts = ([join(both) + ' với cả hai mức nhãn'] if both else []) + ([join(single)] if single else [])
        return '; '.join(parts)
    perm = lambda g: [Pt.loc[(d, f, g), 'mean'] for d in DS for f in fracs]
    n_cases = len(DS) * len(fracs)

    def say(var, what):
        """'<what> cho tỉ số a–b; cao hơn có ý nghĩa ở ..., thấp hơn có ý nghĩa ở ...'"""
        up, dn = sig_(var, +1), sig_(var, -1)
        return (f'{what} cho tỉ số {rng(r_(var))}; '
                + (f'cao hơn có ý nghĩa ở {len(up)} trên {n_cases} trường hợp ({lst(up)})' if 0 < len(up) < n_cases else
                   f'cao hơn có ý nghĩa ở cả {n_cases} trường hợp' if up else
                   'không trường hợp nào cao hơn có ý nghĩa')
                + (f', thấp hơn có ý nghĩa ở {lst(dn)}' if dn else ', không trường hợp nào thấp hơn có ý nghĩa'))
    reduced = ['no_t', 't_only', 'no_V', 'no_I', 'no_QT', 'QT_t']
    always_hurt = [x for x in reduced if sig_(x, +1)]
    helps = [(x, sig_(x, -1)) for x in reduced + ['rel', 'prune'] if sig_(x, -1)]
    labv = dict(VARIANTS)
    rat = lambda var, d, f: Ai.loc[(d, f, var), 'ratio']
    up_t, dn_t = sig_('no_t', +1), sig_('no_t', -1)
    t_txt = (f'Biến thời gian $t$ mang thông tin rõ nhất. Bỏ $t$ làm sai số tăng có ý nghĩa ở {len(up_t)} trên '
             f'{n_cases} trường hợp, nhiều nhất {v(max(r_("no_t")))} lần; ' + (
                 ('trường hợp duy nhất có lợi là ' if len(dn_t) == 1 else 'các trường hợp có lợi là ')
                 + lst(dn_t) + ' (tỉ số ' + ', '.join(v(rat('no_t', d, f)) for d, f in dn_t) + '). ' if dn_t else
                 'không trường hợp nào có lợi. '))
    assert all(min(r_(x)) > 1 for x in ['t_only'])
    groups = ['no_V', 'no_I', 'no_QT', 'QT_t']
    assert not any(sig_(x, -1) for x in groups)               # bỏ nhóm không có lợi ở trường hợp nào
    nV, nI, nQT, nQt = (len(sig_(x, +1)) for x in groups)
    assert nV > nI and nQT > nI                               # nhóm V, QT quan trọng hơn nhóm I
    mQT = max(r_('no_QT'))
    pV, pI, pQT = (max(perm(g)) for g in ('[V]', '[I]', '[QT]'))
    same_order = pI < min(pV, pQT)
    pr_up, pr_dn = sig_('prune', +1), sig_('prune', -1)
    text_abl = (
        'Bảng~\\ref{tab:ablation} cho kết quả bỏ hoặc thêm nhóm ở 8 trường hợp (4 bộ $\\times$ 2 mức nhãn). Độ quan '
        'trọng hoán vị của mô hình đủ 17 đầu vào (tỉ số MAE khi xáo trộn một nhóm giữa các hàng test so với khi không '
        'xáo trộn) được nêu bằng số trong các đoạn dưới.\n\n'
        + t_txt
        + f'Ngược lại, chỉ dùng $t$ mà bỏ hết đặc trưng sạc cho sai số gấp {rng(r_("t_only"))} lần: tuổi của cell một mình '
        'không đủ để biết SOH, vì các cell già đi với tốc độ rất khác nhau (tuổi thọ trong mỗi bộ chênh nhau nhiều lần, '
        f'Bảng~\\ref{{tab:datastats}}). Hoán vị $t$ giữa các hàng test làm sai số tăng {rng(perm("[t]"))} lần, nên mô hình '
        'thật sự dùng biến này.\n\n'
        'Với ba nhóm đặc trưng sạc, không có trường hợp nào bỏ một nhóm mà sai số giảm có ý nghĩa. Nhóm V (điện '
        'áp pha CC) và nhóm QT (điện lượng, thời gian) mang nhiều thông tin nhất: bỏ nhóm V làm sai số tăng có ý nghĩa ở '
        f'{nV} trên {n_cases} trường hợp, bỏ nhóm QT ở {nQT} trên {n_cases}, '
        + (f'có trường hợp gần gấp đôi ({v(mQT)}). ' if mQT >= 1.8 else f'nhiều nhất {v(mQT)} lần. ')
        + f'Bỏ nhóm I (dòng điện pha CV) chỉ làm sai số tăng có ý nghĩa ở {nI} trường hợp. Chỉ giữ nhóm QT và $t$, tức 5 '
        f'đầu vào, kém bộ đầy đủ có ý nghĩa ở {nQt} trên {n_cases} trường hợp. '
        + (f'Phép hoán vị cho cùng nhận xét: hoán vị nhóm QT làm sai số tăng tới {v(pQT)} lần, nhóm V tới {v(pV)} lần, '
           f'nhóm I chỉ tới {v(pI)} lần.' if same_order else
           f'Hoán vị nhóm QT làm sai số tăng tới {v(pQT)} lần, nhóm V tới {v(pV)} lần, nhóm I tới {v(pI)} lần.')
        + '\n\n'
        'Phép thử cuối là chọn đặc trưng tự động. Khi bỏ các đầu vào mà hoán vị trên validation làm sai số tăng '
        'dưới 1\\,\\%, '
        f'mỗi fold giữ lại trung vị {rng(kept.values, 0)} trên 17 đầu vào tuỳ bộ và mức nhãn; hay bị bỏ nhất là '
        + jv([NAMES_VI[f] for f, c in often]) + '. Trên test, cách chọn này cho sai số bằng '
        + f'{rng(r_("prune"))} lần bộ đầy đủ'
        + (', không khác có ý nghĩa ở trường hợp nào, tức bớt đầu vào không làm mô hình tốt lên.' if not (pr_up or pr_dn)
           else (', kém hơn có ý nghĩa ở ' + lst(pr_up) if pr_up else '')
           + (', tốt hơn có ý nghĩa ở ' + lst(pr_dn) if pr_dn else '') + '.')
        + '\n\n'
        + ('Tóm lại, mọi tập rút gọn đều kém bộ đầy đủ ở ít nhất một trường hợp' if len(always_hurt) == len(reduced) else
           f'Tóm lại, {len(always_hurt)} trên {len(reduced)} tập rút gọn kém bộ đầy đủ ở ít nhất một trường hợp')
        + ((', và các trường hợp có lợi (' + '; '.join(f'{labv[x]} ở {lst(k)}' for x, k in helps)
            + ') không lặp lại ở bộ khác. Chọn đầu vào riêng cho từng bộ dựa trên những kết quả này là chọn mô hình trên '
            'tập test, còn chọn riêng cho từng fold bằng validation thì không cải thiện. ')
           if helps and all(len({d for d, _ in k}) == 1 for _, k in helps) else
           ('. Các trường hợp có lợi: ' + '; '.join(f'{labv[x]} ở {lst(k)}' for x, k in helps) + '. ' if helps else
            ', và không tập rút gọn hay mở rộng nào tốt hơn có ý nghĩa ở trường hợp nào. '))
        + 'Vì vậy đồ án dùng chung bộ 17 đầu vào cho mọi bộ dữ liệu và mọi mô hình.')
    # ---------------- đã bỏ
    assert rel_info is None or rel_info['geo'] > 1, 'đặc trưng tương đối không còn làm sai số tăng: sửa lời văn'
    rel = r_('rel')
    sigrel, relgood = sig_('rel', +1), sig_('rel', -1)
    has_T = ('TJU', fracs[0], 'with_T') in Ai.index
    tT = [Ai.loc[('TJU', f, 'with_T'), 'ratio'] for f in fracs] if has_T else []
    tT_txt = ', '.join(f'{_ratio_cell(Ai.loc[("TJU", f, "with_T")])} ở {int(100 * f)}\\,\\%' for f in fracs) if has_T else ''
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Các đặc trưng và cách biến đổi đã thử và bỏ]{Các đặc trưng và cách biến đổi đầu vào đã thử và bỏ, lý do '
           r'và số liệu. Tỉ số là MAE so với cấu hình được dùng (lớn hơn 1: kém hơn; nhỏ hơn 1 ở hai hàng đầu: sai số '
           r'trông thấp hơn thực tế vì rò rỉ).}\label{tab:dropped}',
           r'\begin{tabularx}{\textwidth}{>{\raggedright}p{3.6cm}>{\raggedright}p{4.2cm}>{\raggedright\arraybackslash}X}\toprule',
           r'Đã thử & Lý do bỏ & Số liệu \\\midrule',
           r'Chuẩn hoá min--max đặc trưng và $t$ theo cả vòng đời từng cell (như PINN4SOH) & dùng điểm cuối đời của cell, '
           r'không có khi triển khai; $t$ sau chuẩn hoá là phần đời đã qua & MLP, 70\,\% nhãn: tỉ số XJTU '
           + v(L_leak['minmax']['XJTU']) + ', TJU ' + v(L_leak['minmax']['TJU']) + ', MIT ' + v(L_leak['minmax']['MIT'])
           + ', HUST ' + v(L_leak['minmax']['HUST']) + r' (Bảng~\ref{tab:leak}) \\',
           r'Lọc ngoại lai $3\sigma$ theo cả vòng đời, trên cả cột dung lượng & dùng tương lai của cell và cả nhãn & tỉ số '
           + L_leak['life'] + r' \\',
           r'Chuẩn hoá đặc trưng gồm cả cell test & dùng thống kê của tập test & tỉ số ' + L_leak['normtest'] + r' \\',
           r'16 đặc trưng tương đối so với trung bình các chu kỳ đầu đã có của cell & làm sai số trên tập xác thực của '
           r'PINN-semi tăng; với GBDT tác động trái chiều giữa các bộ & '
           + (r'PINN-semi, tập xác thực: tỉ số ' + v(rel_info['geo'], 3) + ' (lớn nhất ' + v(max(rel_info['ratios'].values()), 2)
              + r' lần); ' if rel_info else '') + r'GBDT, tập kiểm tra: ' + rng(rel)
           + (', cao hơn có ý nghĩa ở ' + lst(sigrel) if sigrel else '')
           + (', thấp hơn có ý nghĩa ở ' + lst(relgood) if relgood else '') + r' \\',
           r'Đặc trưng có lịch sử của cell (trung bình trượt mũ, độ dốc) & chỉ có lợi ở một số bộ khi đủ nhãn & '
           + (r'Bảng~\ref{tab:hist}: PINN-lai có lịch sử bằng ' + rng([fval(F20, d, f, 'lai_h') / fval(F20, d, f, 'pinn_lai')
                                                                    for d in DS for f in FR if fval(F20, d, f, 'lai_h')])
              + r' lần bản không lịch sử, tuỳ bộ và mức nhãn' if F20 is not None else r'Bảng~\ref{tab:hist}') + r' \\']
    if has_T:
        out.append(r'Nhiệt độ làm đầu vào của mô hình & hằng số trên XJTU, MIT, HUST; trên TJU chỉ thay đổi sai số dưới '
                   r'1\,\% & GBDT, TJU, thêm $1/T$: tỉ số ' + tT_txt + r' \\')
    out += [r'Đặc trưng lấy từ lần xả, hoặc dung lượng đo của các chu kỳ trước & lần xả đầy chính là phép đo nhãn; khi ước '
            r'lượng, dùng chúng là rò rỉ nhãn & chép nhãn chu kỳ gần nhất có sai số ' + L_leak['copy'] + r' lần mô hình '
            r'thật (Bảng~\ref{tab:leak}); khi có đo thì dùng ở bước cập nhật (Mục~\ref{sec:budgetres}) \\',
            r'Đặc trưng cần đường cong thô (dung lượng gia tăng, điện áp hồi phục) & bản dữ liệu đã trích đặc trưng không '
            r'có đường cong thô; chưa xử lý dữ liệu thô của từng bộ & chưa thử \\',
            r'\bottomrule\end{tabularx}\end{table}']
    tab_drop = '\n'.join(out)
    text_drop = (
        'Bảng~\\ref{tab:dropped} tóm tắt những gì đã thử và bỏ. Ba dạng đầu là rò rỉ: chúng làm sai số đo được khác sai '
        'số thật khi triển khai, theo cả hai chiều tuỳ bộ (Bảng~\\ref{tab:leak}), nên bị loại dù trên một số bộ chúng làm '
        'con số đẹp hơn. Đặc trưng tương đối bị loại ở bước chọn hướng cải tiến của PINN-lai vì làm sai số của PINN-semi '
        'trên tập xác thực tăng' + (f' (tỉ số {v(rel_info["geo"], 3)}, tính với mốc chỉ dùng các chu kỳ đã qua)'
                                     if rel_info else '') + '. Phép chia cho mức ban đầu của cell khuếch đại các đặc trưng '
        'có giá trị ban đầu gần 0. Với GBDT, tác động của chúng trái chiều giữa các bộ'
        + (' (thấp hơn có ý nghĩa ở ' + lst(relgood) + ', cao hơn có ý nghĩa ở ' + lst(sigrel) + ')' if relgood and sigrel
           else (' (cao hơn có ý nghĩa ở ' + lst(sigrel) + ')' if sigrel else ''))
        + ', nên không dùng làm mặc định cho mọi bộ. Đặc trưng có lịch sử (Bảng~\\ref{tab:hist} ở \\Cref{sec:e20res}) '
        'chỉ giúp ở một số bộ khi đủ nhãn và làm kém đi ở những trường hợp khác, nhất là khi số chiều đầu vào tăng gấp đôi '
        'tới gấp bốn mà chỉ còn vài cell có nhãn. Vì vậy PINN-lai không dùng chúng, còn HiPINN-lai chỉ dùng chúng ở những '
        'fold mà tập xác thực chọn (\\Cref{sec:pinnlai}, phân tích hậu kiểm). '
        + ('Nhiệt độ là hằng số trên ba bộ nên không mang thông tin; trên TJU, thêm $1/T$ vào GBDT cho tỉ số '
           + tT_txt.replace('$^{*}$', ' (có ý nghĩa)').replace('$^{\\dagger}$', ' (có ý nghĩa)')
           + ', tức thay đổi dưới 1\\,\\%, nên nhiệt độ chỉ được giữ trong thừa số Arrhenius của động học. '
           if has_T else '')
        + 'Đặc trưng từ lần xả và dung lượng các chu kỳ trước bị loại khỏi bài toán ước lượng vì đó chính là nhãn.')
    text_drop = text_drop + '\n\n' + tab_drop
    # ---------------- kết luận về đầu vào
    concl = (
        r'Từ các kết quả trên, đồ án dùng đủ 16 đặc trưng đoạn sạc và $t = n/1000$: cả 17 tính được ở mọi chu kỳ từ dữ liệu '
        f'sạc, {len(always_hurt)} trên {len(reduced)} tập rút gọn kém hơn có ý nghĩa ở ít nhất một trường hợp, không tập '
        f'nào tốt hơn ở mọi bộ, và chọn bớt đặc trưng trên validation cho tỉ số {rng(r_("prune"))}. Đồ án không chuẩn hoá '
        r'theo cả vòng đời cell, không lọc ngoại lai bằng tương lai của cell và không fit normaliser trên cell test, vì đây '
        r'là rò rỉ và tác động của chúng lên sai số lên tới ' + L_leak['minmax_rng'] + r' lần tuỳ bộ. Đặc trưng tương '
        r'đối không được dùng vì tác động trái chiều giữa các bộ; đặc trưng có lịch sử chỉ dùng khi validation chọn; nhiệt '
        r'độ chỉ được dùng trong thừa số Arrhenius, vì thêm vào đầu vào chỉ thay đổi sai số dưới 1\,\%; thông tin của lần xả không '
        r'được dùng khi ước lượng.')
    return text_rel + '\n\n' + tab_rel, text_abl + '\n\n' + tab_abl, text_drop, concl





# --------------------------------------------------------------------------- Chương 6: đối chiếu với bài báo đã công bố
PUB_CAND = [('paper_pinn_lai', 'PINN-lai'), ('paper_gbdt', 'GBDT'), ('paper_pinn3', 'PINN-semi')]


def best_rmse_table(S):
    """RMSE công bố của PINN4SOH so với RMSE thấp nhất của đồ án trên mỗi bộ (đúng cách chia, cùng tiền xử lý với bài,
    chế độ đầy đủ), kèm tên mô hình đạt được. Trả (bảng LaTeX, các dòng {bộ, mô hình, tỉ lệ chênh})."""
    out, rows = [], []
    for ds in DS:
        k = (ds, 'full')
        pub = float(S.loc[k, 'published_rmse'])
        col, name = min(PUB_CAND, key=lambda c: float(S.loc[k, f'{c[0]}_rmse']))
        best = float(S.loc[k, f'{col}_rmse'])
        d = best / pub - 1
        cells = [(r'\textbf{' + v(1e3 * x) + '}') if x == min(pub, best) else v(1e3 * x) for x in (pub, best)]
        out.append(f'{ds} & {cells[0]} & {cells[1]} & {name} & ' + (r'$-$' if d < 0 else r'$+$') + pct(abs(d), 1) + r' \\')
        rows.append(dict(ds=ds, model=name, d=d))
    tab = (r'\begin{table}[!htbp]\centering\small' '\n'
           r'\caption[RMSE tốt nhất của đồ án so với PINN4SOH]{RMSE ($\times10^{-3}$) trên đúng cách chia của bài PINN4SOH, '
           r'chế độ đầy đủ, cùng tiền xử lý với bài. Cột thứ hai tính từ tệp kết quả công bố cùng mã nguồn của bài; cột '
           r'thứ ba là RMSE thấp nhất trong ' + ', '.join(n for _, n in PUB_CAND[:-1]) + ' và ' + PUB_CAND[-1][1]
           + r' của đồ án, cột Mô hình ghi mô hình đạt được. Cả hai cột lấy trung bình qua cell test, lần chạy và nhóm như '
           r'Bảng~\ref{tab:pub}. In đậm: thấp hơn trong hai cột. Chênh lệch là RMSE của đồ án so với số công bố, âm là '
           r'đồ án thấp hơn.}\label{tab:pubrmse}'
           '\n'
           r'\begin{tabular}{lrrlr}\toprule' '\n'
           r'Bộ dữ liệu & RMSE PINN4SOH công bố ' + DN + r' & RMSE tốt nhất của đồ án ' + DN + r' & Mô hình & '
           r'Chênh lệch ' + DN + r' \\\midrule' '\n'
           + '\n'.join(out) + '\n' r'\bottomrule\end{tabular}\end{table}')
    return tab, rows


def compare_chapter(S, T, B, L_leak, E33=None, PH=None):
    pub = {k: PAPER_MAE[k] for k in ROWS}
    beat = {m: [k for k in ROWS if S.loc[k, 'paper_pinn_lai'] < pub[k][m]] for m in ('PINN4SOH', 'MLP', 'CNN')}
    tab = compare_table(S, T) + '\n\n' + FIG_BENCH
    full = [k for k in ROWS if k[1] == 'full']
    few = [k for k in ROWS if k[1] != 'full']
    name = lambda k: f'{k[0]} {"đầy đủ" if k[1] == "full" else "$K = " + k[1][1] + "$"}'
    miss = lambda m: [k for k in ROWS if k not in beat[m]]
    st = B['strict']
    st_better = [k for k in full if st[k] < 0.95]
    st_tie = [k for k in full if 0.95 <= st[k] <= 1.05]
    st_worse = [k for k in full if st[k] > 1.05]
    st_few_better = [k for k in few if st[k] < 1]
    st_few_worse = [k for k in few if st[k] >= 1]
    hyb_rerun = [S.loc[k, 'paper_pinn_lai'] / S.loc[k, 'rerun_published'] for k in ROWS]
    s = []
    s.append(r'\subsection{Mốc so sánh}' '\n'
             r'Trên bốn bộ dữ liệu này, công trình cùng bài toán, cùng đầu vào và có công bố sai số theo từng nhóm cell test '
             r'là PINN4SOH~\cite{wang2024}; bài đó còn báo cáo hai mô hình đối chứng MLP và CNN trên cùng các nhóm. Các bài '
             r'công bố bộ dữ liệu gốc giải bài toán khác hoặc dùng đầu vào khác: Severson và cộng sự~\cite{severson2019} dự '
             r'đoán tuổi thọ của cả cell từ các chu kỳ đầu, Zhu và cộng sự~\cite{zhu2022} dùng điện áp hồi phục sau sạc, '
             r'Ma và cộng sự~\cite{ma2022} dùng học chuyển giao với cách chia riêng. Vì cách chia và đầu vào khác nhau, sai '
             r'số của các bài đó không so trực tiếp được với đồ án, và báo cáo không trích chúng. Mọi so sánh dưới đây dùng '
             r'đúng 39 nhóm đánh giá của bài, dựng lại từ nhật ký huấn luyện công bố kèm mã: mỗi nhóm thí nghiệm (batch) của '
             r'XJTU và TJU là một nhóm đánh giá ở chế độ đầy đủ, một nhóm cho MIT (23 cell test) và cho HUST (20 cell test), và các nhóm ít mẫu '
             r'$K = 1, \dots, 4$ của XJTU và HUST. Sai số tính đúng như bài: MAE từng cell test, trung bình qua lần chạy, '
             r'qua cell, qua nhóm. PINN-lai của đồ án chạy với hai cách tiền xử lý: bản sao tiền xử lý của bài (có rò rỉ, '
             r'chỉ để so cùng điều kiện) và tiền xử lý không rò rỉ của đồ án. Bảng~\ref{tab:pub} và \Cref{fig:bench} tập '
             r'hợp các con số.' '\n\n' + tab)
    rtab, rrows = best_rmse_table(S)
    lower = [r['ds'] for r in rrows if r['d'] < 0]
    higher = [r for r in rrows if r['d'] >= 0]
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    s.append(r'Theo RMSE, Bảng~\ref{tab:pubrmse} lấy trên mỗi bộ mô hình có sai số thấp nhất của đồ án và ghi tên mô hình '
             r'đó. ' + (f'RMSE thấp nhất của đồ án thấp hơn số công bố ở {jv(lower)} ('
                        + '; '.join(f'{r["ds"]}: {r["model"]}, {pct(-r["d"], 1)}' for r in rrows if r['d'] < 0) + ')'
                        if lower else 'RMSE thấp nhất của đồ án không thấp hơn số công bố ở bộ nào')
             + (('; ở ' + jv([f'{r["ds"]}, {r["model"]} cao hơn {pct(r["d"], 1)}' for r in higher])) if higher else '')
             + '.' '\n\n' + rtab)
    good = [
        rf'Cùng cách chia và cùng tiền xử lý với bài, PINN-lai có sai số thấp hơn mã gốc PINN4SOH chạy lại ở '
        rf'{B["beat_rerun"]} trên 12 chế độ, thấp hơn có ý nghĩa ở {B["w_full"] + B["w_few"]} (tỉ số {rng(B["hyb_full"])} '
        rf'ở chế độ đầy đủ, {rng(B["hyb_few"])} ở chế độ ít mẫu) và không chế độ nào cao hơn có ý nghĩa '
        rf'(Bảng~\ref{{tab:pub}}).',
        rf'So với số công bố của PINN4SOH, PINN-lai thấp hơn ở {len(beat["PINN4SOH"])} trên 12 chế độ (tỉ số '
        rf'{rng(B["hyb_pub_full"])} ở chế độ đầy đủ, {rng(B["hyb_pub_few"])} ở chế độ ít mẫu). Lợi thế lớn nhất ở chế độ '
        rf'ít mẫu, nơi hàm mất mát vật lý được tính cả trên các cell huấn luyện không nhãn.',
        rf'So với MLP và CNN mà bài PINN4SOH báo cáo, PINN-lai thấp hơn MLP ở {len(beat["MLP"])} trên 12 chế độ và thấp '
        rf'hơn CNN ở {len(beat["CNN"])} trên 12.',
        rf'Với tiền xử lý không rò rỉ, PINN-lai vẫn thấp hơn số công bố của PINN4SOH trên '
        + ', '.join(f'{k[0]} (tỉ số {v(st[k])})' for k in st_better)
        + rf', thấp hơn có ý nghĩa theo cell ở {B["sw"]} trên 4 bộ đầy đủ'
        + (', và ở ' + ', '.join(f'{name(k)} (tỉ số {v(st[k])})' for k in st_few_better) if st_few_better else '') + '.',
    ]
    bad = [
        'Cùng điều kiện với bài, PINN-lai chưa thấp hơn số công bố của PINN4SOH ở '
        + ', '.join(f'{name(k)} ({v(1e3 * S.loc[k, "paper_pinn_lai"])} so với {v(1e3 * pub[k]["PINN4SOH"])})'
                    for k in B['lose_pub'])
        + ('' if not miss('CNN') else ', và chưa thấp hơn CNN của bài ở '
           + ', '.join(f'{name(k)} ({v(1e3 * pub[k]["CNN"])})' for k in miss('CNN'))) + '.',
        rf'Một GBDT đơn giản với cùng tiền xử lý rò rỉ tốt ngang PINN-lai: tỉ số GBDT/PINN-lai {rng(B["gb_vs_hyb"])}, GBDT '
        rf'thấp hơn có ý nghĩa ở {B["gw"]} chế độ ({", ".join(B["gb_sig_keys"])}) và không chế độ nào cao hơn. Với '
        r'min--max theo cell, chỉ số chu kỳ đã chuẩn hoá là phần đời đã qua và mô hình cây khai thác thẳng thông tin này, '
        r'nên lợi thế so với mã gốc trong điều kiện đó không thể quy cho phần vật lý.',
        r'Với tiền xử lý không rò rỉ, PINN-lai cao hơn số công bố trên '
        + ', '.join(f'{k[0]} (tỉ số {v(st[k])})' for k in st_worse)
        + (', ngang trên ' + ', '.join(f'{k[0]} ({v(st[k])})' for k in st_tie) if st_tie else '')
        + rf', cao hơn có ý nghĩa theo cell ở {B["sl"]} trên 4 bộ đầy đủ; ở chế độ ít mẫu cao hơn ở '
        + f'{len(st_few_worse)} trên {len(few)} (tỉ số {rng([st[k] for k in st_few_worse])}). Đây là các chế độ mà tiền '
        r'xử lý của bài có lợi nhiều nhất: khi chỉ có vài cell có nhãn, biết cell đã đi được bao nhiêu phần đời gần như là '
        r'biết SOH. Đồ án không dùng tiền xử lý đó vì nó không thực hiện được khi triển khai (Mục~\ref{sec:dt_bo}).',
        rf'Đồ án không tái lập được chính xác số công bố của PINN4SOH: mã gốc chạy lại cho sai số bằng {rng(B["rep_full"])} '
        rf'lần ở chế độ đầy đủ, cao hơn có ý nghĩa theo cell ở {B["rep_sig"][1]} trên 4 bộ (Bảng~\ref{{tab:pub}}); các hàng train dựng lại trùng '
        rf'với dữ liệu của bài, nên phần chênh đến từ phần cứng và phiên bản thư viện.',
    ]
    internal = [
        r'GBDT đơn lẻ có sai số thấp hơn PINN-semi ở 14/16 trường hợp bộ $\times$ mức nhãn (10 có ý nghĩa). PINN-semi chỉ '
        r'vượt GBDT có ý nghĩa ở HUST 30\,\% nhãn (Mục~\ref{sec:labels}). PINN-lai cố định vẫn kém GBDT có ý nghĩa ở TJU 70\,\%.',
        r'Ở 70\,\% nhãn, PINN-semi kém MLP có ý nghĩa trên TJU (tỉ số 1{,}09); với kiểm định bảo thủ ở mức fold, lợi ích ở '
        r'30\,\% nhãn chỉ đứng vững trên MIT.',
        r'Khoảng dự đoán 90\,\% của ước lượng SOH có độ phủ thực tế 84{,}9--92{,}6\,\% (trung bình 88{,}8\,\%), '
        r'14--25\,\% số cell có độ phủ dưới 80\,\%.',
        r'Tham số vật lý $\lambda$, $E_a$ không được dữ liệu xác định, và tốc độ $r$ học được thấp hơn nhiều tốc độ suy giảm '
        r'thực đo (\Cref{sec:vatly}).',
        r'Bộ lọc xu thế không bảo đảm SOH dự đoán không tăng (đầu ra tăng ở 28--44\,\% số bước); ép không tăng làm sai số '
        r'trên XJTU tăng 1{,}76--1{,}94 lần vì dung lượng đo được của XJTU tăng ở đầu đời.',
    ]
    if E33 is not None:
        assert min(E33['cov_h']) < 0.9 or min(E33['cov_rul']) < 0.9, 'khoảng dự báo đã đạt độ phủ: sửa câu "chưa đạt"'
        internal.insert(3, r'Với HiPINN (\Cref{sec:e33res}), khoảng SOH 90\,\% tính bằng phương pháp conformal phủ '
                        + pct_rng(E33['cov_h']) + r' tuỳ bộ và ngân sách, khoảng RUL 90\,\% phủ '
                        + pct_rng(E33['cov_rul']) + r'; tỉ lệ dự báo RUL nằm trong $\pm 20\,\%$ RUL thật là '
                        + pct_rng(E33['in20']) + r'. Ở 10\,\% đời cuối của ' + E33['worst']
                        + r', MAE của HiPINN vẫn là ' + v(E33['tail_h'], 1) + r' ($\times10^{-3}$).')
    para = lambda xs: ' '.join(xs)
    s.append(r'\subsection{Các chỉ số tốt hơn các bài báo đã công bố}' '\n' + para(good))
    s.append(r'\subsection{Các chỉ số chưa tốt}' '\n' 'So với các bài báo đã công bố, đồ án còn một số điểm chưa đạt. '
             + para(bad) + '\n\n' 'Ngay trong các so sánh của chính đồ án, một số chỉ số cũng chưa đạt. ' + para(internal))
    if PH is not None:
        s.append(physics_section(PH))
    s.append(r'\subsection{Nhận xét}' '\n'
             r'Trên đúng cách chia của PINN4SOH và cùng cách tính sai số, PINN-lai của đồ án có sai số thấp hơn số công bố ở '
             r'phần lớn các chế độ và thấp hơn mã gốc chạy lại ở mọi chế độ. Tuy vậy, trong điều kiện có rò rỉ của bài, một '
             r'GBDT cũng đạt kết quả tương tự, còn khi bỏ rò rỉ thì hơn hay kém tuỳ bộ dữ liệu. Vì thế, những đóng góp có cơ sở '
             r'vững nhất của đồ án là giao thức đánh giá không rò rỉ, các so sánh ghép cặp có kiểm định trong cùng một giao '
             r'thức, và các mô hình có đặc trưng lịch sử HiGBDT, HiPINN cho bài toán dự báo.')
    write('doan_so_sanh.tex', '\n\n'.join(s))
    return dict(beat=beat, st_better=st_better, st_worse=st_worse, good=good, bad=bad)


# --------------------------------------------------------------------------- chạy lại các thí nghiệm của đồ án
def reproducibility(B):
    """Kiểm tra tái lập: kiểm toán rò rỉ so với các lượt tham chiếu; 20 lượt mẫu; E27 chạy lại so với lần chạy đầu
    (generated/data/e27_lan_dau.json). Kết quả được lưu ở generated/data/tai_lap.json để dựng được khi không có runs/."""
    path = os.path.join(DATA, 'tai_lap.json')
    rec = json.load(open(path)) if os.path.exists(path) else {}
    import hashlib
    sys_path = _sys.path[:]
    _sys.path.insert(0, 'tools')
    try:
        from kiem_tai_lap import REFERENCE
    finally:
        _sys.path[:] = sys_path
    refs = [(e, f, m, h) for e, f, m, h in REFERENCE if e in ('E15', 'E13') and os.path.exists(f) and '/F_' in f
            or e == 'E15' and os.path.exists(f)]
    if refs:
        same = 0
        for e, f, m, h in refs:
            r = json.load(open(f))
            got = r['test']['MAE_cell']
            hh = hashlib.sha1(json.dumps(r['test_predictions']['p']).encode()).hexdigest()[:16] \
                if r.get('test_predictions') else None
            same += got == m and (h is None or hh == h)
        rec['leak_refs'] = dict(n=len(refs), same=int(same))
    ref27 = os.path.join(DATA, 'e27_lan_dau.json')
    if os.path.exists('runs/E27') and len(os.listdir('runs/E27')) >= 1002 and os.path.exists(ref27):
        # so với lần chạy E27 đầu tiên: trung bình theo (nguồn, bộ, chế độ) ở độ chính xác đầy đủ; bảng theo cell ở 6 chữ
        # số có nghĩa (đúng như benchmark/compare.py ghi cells.csv), so bằng mã băm
        _sys.path.insert(0, 'benchmark')
        import compare
        ref = json.load(open(ref27))
        C = pd.DataFrame(compare.ours_rows())
        S = compare.summarize(C).set_index(['dataset', 'mode'])
        same = sum(float(S.loc[(r['dataset'], r['mode']), col]) == r[stat] for r in ref['tong_hop']
                   for stat, col in (('mae', r['source']), ('rmse', r['source'] + '_rmse'), ('sd', r['source'] + '_sd')))
        k = ['source', 'dataset', 'mode', 'group', 'run', 'cell']
        D = C[k + ['mae', 'rmse']].sort_values(k, kind='mergesort', ignore_index=True)
        sha = hashlib.sha1(D.to_csv(index=False, float_format='%.6g').encode()).hexdigest()
        rec['e27'] = dict(runs=len(os.listdir('runs/E27')), summary_n=3 * len(ref['tong_hop']), summary_same=int(same),
                          cells=len(D), cells_same=bool(sha == ref['theo_cell']['sha1']
                                                        and len(D) == ref['theo_cell']['dong']))
    rec.pop('e22', None)                                  # mục dự báo E22 không còn trong báo cáo (E33 thay)
    log = os.path.join(DATA, 'kiem_tai_lap.txt')
    if os.path.exists(log):
        mm = re.search(r'(\d+)/(\d+) lượt giống từng bit', open(log).read())
        if mm:
            rec['sample'] = dict(same=int(mm.group(1)), n=int(mm.group(2)), text=open(log).read()[-2000:])
    os.makedirs(DATA, exist_ok=True)
    json.dump(rec, open(path, 'w'), ensure_ascii=False, indent=1)
    s = [r'Mọi lượt chạy dùng một luồng và hạt giống cố định, nên trên cùng loại CPU và cùng phiên bản thư viện, chạy lại '
         r'phải cho dự đoán giống hệt. Các kiểm tra dưới đây xác nhận điều đó.']
    items = []
    if 'sample' in rec:
        items.append(f'Hai mươi lượt mẫu phủ mọi loại thí nghiệm chính (MLP, PINN-semi, PINN-sup, PINN hộp đen, Ridge, GPR, '
                     f'GBDT, kiểm toán rò rỉ, chuyển miền), chạy lại vào thư mục tạm và so với giá trị lưu của lần chạy báo '
                     f'cáo đã nộp: {rec["sample"]["same"]} trên {rec["sample"]["n"]} lượt có MAE theo cell và mã băm của toàn '
                     'bộ dự đoán trên tập kiểm tra giống hệt'
                     + ('.' if rec['sample']['same'] == rec['sample']['n'] else
                        '; lượt còn lại lệch ở mức làm tròn số thực.'))
    if 'leak_refs' in rec:
        items.append('Kiểm toán rò rỉ (160 lượt của bốn cách làm sai và 40 lượt tham chiếu theo cell) được chạy lại toàn bộ; '
                     f'{rec["leak_refs"]["same"]} trên {rec["leak_refs"]["n"]} lượt có giá trị tham chiếu lưu sẵn giống '
                     'hệt, và các tỉ số ở Bảng~\\ref{tab:leak} khớp với báo cáo đã nộp.')
    if 'e27' in rec:
        e = rec['e27']
        items.append(f'PINN-lai, GBDT và PINN-semi trên 39 nhóm của PINN4SOH ({thou(e["runs"])} lượt) được chạy lại toàn bộ và so với '
                     f'kết quả của lần chạy trước: {e["summary_same"]} trên {e["summary_n"]} giá trị trung bình (MAE, RMSE và '
                     'độ lệch chuẩn qua ba hạt giống của PINN-lai, GBDT và trung bình ba PINN-semi, với hai cách tiền xử lý, '
                     'trên 12 chế độ) giống hệt. '
                     + (f'MAE và RMSE theo cell của cả {thou(e["cells"])} dòng trùng ở 6 chữ số có nghĩa đã lưu.'
                        if e['cells_same'] else 'bảng MAE và RMSE theo cell không trùng hoàn toàn ở 6 chữ số có nghĩa đã lưu.'))
    items.append('GBDT với đủ 17 đầu vào trong phân tích đặc trưng (Mục~\\ref{sec:dt_pp}) cho MAE theo cell giống hệt '
                 'với lượt GBDT tham chiếu của thí nghiệm chính (XJTU, 30\\,\\% nhãn, lần lặp 0, fold 0).')
    items.append(f'So với PINN4SOH trên đúng dữ liệu huấn luyện của bài: một đại lượng ghi theo từng vòng huấn luyện (epoch) trong nhật ký huấn '
                 f'luyện công bố của bài, tính lại trên các hàng train dựng lại ở đây, khớp ở cả {B["n_eq"]} nhóm; sai số '
                 'khi chạy lại mã gốc không trùng số công bố vì khác phần cứng và phiên bản thư viện.')
    s.append(' '.join(items))
    s.append('Khác CPU (ví dụ AVX-512 so với AVX2) hoặc khác phiên bản PyTorch có thể làm kết quả lệch ở mức làm tròn số '
             'thực; khi đó các kết luận không đổi nhưng chữ số cuối trong bảng có thể khác.')
    write('doan_tai_lap.tex', '\n\n'.join(s))
    return rec


# --------------------------------------------------------------------------- kết luận và tóm tắt
def conclusion(B, L_leak, C6, F_ok, n_total, E33=None, F20=None, KS=None):
    tp = [ftest(F20, 'pinn_lai', 'pinn', ds, f) for ds in DS for f in FR]
    tg = [ftest(F20, 'pinn_lai', 'gbdt', ds, f) for ds in DS for f in FR]
    lai = (f'có sai số bằng {rng([t.ratio for t in tp])} lần PINN-semi và thấp hơn GBDT ở '
           f'{sum(t.ratio < 1 for t in tg)}/16 trường hợp')
    p1 = (r'Đồ án đã xây dựng đường ống dữ liệu cho bốn bộ dữ liệu công khai (387 cell, 305.802 chu kỳ sau lọc) với bộ lọc '
          r'ngoại lai nhân quả, và kiểm tra bằng số liệu vì sao dùng 16 đặc trưng sạc cùng chỉ số chu kỳ, vì sao bỏ các cách '
          r'biến đổi khác (Mục~\ref{ch:dactrung}). Giao thức đánh giá không rò rỉ cho thấy rò rỉ làm lệch kết quả đáng kể: '
          rf'chia theo chu kỳ làm sai số thấp đi {L_leak["row"]} lần, còn chuẩn hoá theo cả vòng đời cell làm sai số đổi '
          rf'{L_leak["minmax_rng"]} lần tuỳ bộ. Trong giao thức đó, PINN bán giám sát với động học xám luôn không âm có sai '
          r'số bằng 0{,}39--0{,}90 lần MLP ở 10\,\% nhãn, tức tiết kiệm 1{,}7--4{,}5 lần số nhãn, và mô hình kết hợp PINN-lai '
          + lai + '. Các phép so đều được kiểm định '
          r'thống kê, với chín mô hình đối chứng, trong đó có bốn mô hình học sâu của đề cương (MLP, CNN, LSTM, '
          r'Transformer), và bốn độ đo MAE, RMSE, MAPE, $R^2$. So với PINN4SOH trên đúng '
          rf'{B["n_groups"]} nhóm đánh giá của bài và cùng tiền xử lý, PINN-lai thấp hơn số công bố ở '
          rf'{len(B["beat_pub"])}/12 chế độ và thấp hơn mã gốc của bài chạy lại ở {B["beat_rerun"]}/12.')
    p2 = (r'Ở phần mở rộng, đồ án đặt lại bài toán ít nhãn theo số lần đo dung lượng: đo thưa mọi cell với '
          r'1{,}2--6{,}0\,\% số chu kỳ cho sai số bằng 0{,}85--0{,}93 lần gắn nhãn đầy đủ cho 30\,\% số cell. '
          + (e33_item(E33) if E33 else ''))
    p3 = (rf'Mã nguồn được tổ chức thành các khối dữ liệu, mô hình, thí nghiệm, phân tích, so sánh với PINN4SOH, bản thảo '
          rf'bài báo và báo cáo; {thou(n_total)} lượt huấn luyện có kết quả trong báo cáo, mọi bảng sinh tự động từ kết quả '
          r'và chạy lại cho kết quả giống hệt (Mục~\ref{sec:tailap}).')
    write('doan_ket_luan.tex', p1 + '\n\n' + p2 + '\n\n' + p3 + '\n\n' + final_table(B, C6, E33, F20, KS))
    tomtat = (
        r'Đồ án ước lượng trạng thái sức khoẻ (SOH) của cell lithium-ion từ 16 đặc trưng đoạn sạc khi chỉ một phần nhỏ cell '
        r'được đo dung lượng, trên bốn bộ dữ liệu công khai (387 cell). Phân tích đặc trưng bằng tương quan hạng, bỏ từng '
        r'nhóm đầu vào, độ quan trọng hoán vị và chọn trên validation cho thấy cả ba nhóm đặc trưng sạc và chỉ số chu kỳ '
        r'đều mang thông tin; chuẩn hoá theo cả vòng đời cell và lọc ngoại lai bằng tương lai của cell bị loại vì là rò rỉ. '
        rf'Kiểm toán cho thấy chia theo chu kỳ làm sai số thấp đi {L_leak["row"]} lần và chuẩn hoá theo cả vòng đời cell làm '
        rf'sai số đổi {L_leak["minmax_rng"]} lần tuỳ bộ. Trong giao thức không rò rỉ (kiểm định chéo 5 fold theo cell, lặp '
        r'lại, kiểm định ghép cặp), PINN bán giám sát có sai số bằng 0{,}39--0{,}90 lần MLP ở 10\,\% nhãn, nhưng GBDT vẫn tốt '
        r'hơn nó khi có từ 30\,\% nhãn. Mô hình kết hợp PINN-lai, cấu hình chốt trước khi đọc tập kiểm tra, ' + lai
        + rf'. Trên đúng {B["n_groups"]} nhóm đánh giá và với tiền xử lý của bài PINN4SOH, PINN-lai thấp '
        rf'hơn số công bố ở {len(B["beat_pub"])}/12 chế độ và thấp hơn mã gốc của bài chạy lại ở {B["beat_rerun"]}/12, nhưng '
        r'một GBDT cùng tiền xử lý đạt '
        r'kết quả tương tự; với tiền xử lý không rò rỉ, PINN-lai thấp hơn số công bố trên '
        + ' và '.join(k[0] for k in C6['st_better']) + ', cao hơn trên ' + ' và '.join(k[0] for k in C6['st_worse'])
        + r'. Đo thưa mọi cell với 1{,}2--6{,}0\,\% số chu kỳ cho sai số bằng 0{,}85--0{,}93 lần gắn nhãn đầy đủ cho 30\,\% '
        r'số cell.' + (e33_abstract(E33) if E33 else ''))
    return terms(tomtat)               # \TomTat nằm trong ngoặc nhọn nên write() không tách câu: tách ở đây


def final_table(B, C6, E33=None, F20=None, KS=None):
    """Chương 6: mô hình cuối của từng bài toán. Hàng ước lượng và HiPINN-lai từ phân tích theo fold (F20), hàng cập nhật
    từ phân tích cập nhật (KS), hàng dự báo và HiGBDT-V từ E33, E34; hàng PINN4SOH từ phần so sánh; hàng nhãn thưa chép
    từ lời văn Chương 5."""
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    cases = [(ds, f) for ds in DS for f in FR]
    tp = [ftest(F20, 'pinn_lai', 'pinn', ds, f) for ds, f in cases]
    tg = [ftest(F20, 'pinn_lai', 'gbdt', ds, f) for ds, f in cases]
    th = [ftest(F20, 'hipinn_lai', 'pinn_lai', ds, f) for ds, f in cases]
    gw = [f'{ds} {int(100 * f)}\\,\\%' for (ds, f), t in zip(cases, tg) if t.ratio > 1 and t.p_holm < 0.05]
    hup = sum(t.ratio < 1 and t.p_holm < 0.05 for t in th); hdn = sum(t.ratio > 1 and t.p_holm < 0.05 for t in th)
    thg = [ftest(F20, 'hipinn_lai', 'gbdt', ds, f) for ds, f in cases]
    hg_lo = sum(t.ratio < 1 for t in thg); hg_up = sum(t.ratio < 1 and t.p_holm < 0.05 for t in thg)
    hg_dn = sum(t.ratio > 1 and t.p_holm < 0.05 for t in thg)
    none = lambda k, word: (f'{word} có ý nghĩa thống kê ở {k}/16' if k else f'không {word} có ý nghĩa thống kê ở trường hợp nào')
    rows = [
        ('Ước lượng SOH từ đoạn sạc khi ít nhãn', 'PINN-lai',
         r'trung bình ba PINN-semi ghép với GBDT ($\omega = \tfrac12$), rồi lọc xu thế nhân quả 10 hàng',
         f'MAE bằng {rng([t.ratio for t in tp])} lần PINN-semi ({sum(t.ratio < 1 and t.p_holm < 0.05 for t in tp)}/16 '
         f'có ý nghĩa thống kê). Thấp hơn GBDT ở {sum(t.ratio < 1 for t in tg)}/16 trường hợp bộ $\\times$ mức nhãn '
         f'({sum(t.ratio < 1 and t.p_holm < 0.05 for t in tg)} có ý nghĩa thống kê)'
         + (f', kém hơn có ý nghĩa thống kê ở {len(gw)} ({jv(gw)})' if gw else ''), r'\ref{sec:e20res}'),
        ('Như trên, khi theo dõi được đặc trưng sạc của cell qua thời gian', 'HiPINN-lai',
         r'PINN-lai có thêm trung bình trượt mũ và độ dốc của đặc trưng sạc, chỉ dùng ở những fold mà tập xác thực chọn',
         f'MAE bằng {rng([t.ratio for t in th])} lần PINN-lai, thấp hơn có ý nghĩa thống kê ở {hup}/16 trường hợp và '
         + none(hdn, 'cao hơn') + f'. So với GBDT, thấp hơn ở {hg_lo}/16 ({hg_up} có ý nghĩa thống kê) và '
         + none(hg_dn, 'cao hơn') + '. Phân tích hậu kiểm', r'\ref{sec:e20res}'),
        ('Học khi chỉ đo thưa dung lượng', 'PINN-lai, nhãn thưa',
         r'mọi cell train được đo dung lượng mỗi $0{,}05L$ chu kỳ, tức 1{,}2--6{,}0\,\% số chu kỳ',
         r'MAE bằng 0{,}85--0{,}93 lần PINN-lai học với 30\,\% số cell có nhãn đầy đủ, thấp hơn có ý nghĩa trên cả bốn '
         r'bộ', r'\ref{sec:budgetres}'),
    ]
    upd = (f'thấp hơn bộ lọc Kalman ở {KS["n_k"]}/{KS["n"]} trường hợp bộ $\\times$ mức nhãn $\\times$ lịch đo và '
           f'thấp hơn giữ lần đo ở {KS["n_h"]}/{KS["n"]}')
    if E33 is not None and E33.get('D34') is not None:
        GA = E33['D34']['anchor']
        ga = lambda ds, t, m: float(GA[(GA.ds == ds) & (GA.tier == t)][m].iloc[0])
        low = {t: sum(ga(ds, t, 'selected') < ga(ds, t, 'e32_style') for ds in DS) for t in ('q20', 'q10')}
        assert low['q20'] < 3 and low['q10'] < 3, 'HiGBDT-V tốt hơn GBDT-cập nhật khi đo thưa: đổi mô hình cuối'
        upd += (f'. HiGBDT-V, bản có lịch sử, chỉ thấp hơn nó ở {low["q20"]}/4 bộ khi đo mỗi $0{{,}}2L$ và '
                f'{low["q10"]}/4 bộ khi đo mỗi $0{{,}}1L$')
    rows.append(('Cập nhật SOH khi cell mới được đo vài lần', 'GBDT-cập nhật',
                 'GBDT học SOH đã đổi bao nhiêu kể từ lần đo dung lượng gần nhất', upd,
                 r'\ref{sec:budgetres}'))
    if E33 is not None:
        E = E33
        rows.append(('Dự báo SOH, dung lượng và RUL', 'HiPINN',
                     'trung vị của HiGBDT, PINN-dự báo và nhánh theo đời tại từng gốc và tầm',
                     f'SOH: MAE thấp nhất trong tám mô hình ở {E["best_n"]}/{E["n_all"]} trường hợp, bằng {rng(E["r_hl"])} '
                     f'lần HiGBDT (tốt hơn có ý nghĩa thống kê ở {E["up_l"]}/{E["n_all"]}, kém hơn ở {E["dn_l"]}) và '
                     f'{rng(E["r_hk"])} lần PINN-GBDT (tốt hơn có ý nghĩa thống kê ở {E["up_k"]}/{E["n_do"]}). Khoảng SOH '
                     f'90\\,\\% phủ {pct_rng(E["cov_h"])}. RUL: '
                     + ('được quy tắc chọn khi học trên mọi cell huấn luyện' if E['rul_b'].get('all') == 'hipinn' else
                        'không được quy tắc chọn khi học trên mọi cell huấn luyện')
                     + (', không mô hình nào thoả ở 30\\,\\% nhãn' if E['rul_b'].get('lab30') is None else '')
                     + (f'. Trên {jv(E["worse_rul"])}, PINN-GBDT có MAE RUL thấp hơn' if E['worse_rul'] else '') + '.',
                     r'\ref{sec:e33res}'))
    rows.append(('So với PINN4SOH trên đúng cách chia của bài', 'PINN-lai',
                 'cùng tiền xử lý với bài, và tiền xử lý không rò rỉ',
                 f'MAE thấp hơn số công bố ở {len(B["beat_pub"])}/12 chế độ và thấp hơn mã gốc chạy lại ở '
                 f'{B["beat_rerun"]}/12; với tiền xử lý không rò rỉ, thấp hơn số công bố trên '
                 + jv([k[0] for k in C6['st_better']]) + ', cao hơn trên ' + jv([k[0] for k in C6['st_worse']]),
                 r'\ref{ch:sosanh}'))
    L = lambda w: r'>{\raggedright\arraybackslash}' + w
    out = [r'\Cref{tab:final} tóm tắt mô hình cuối của đồ án cho từng bài toán và kết quả chính trên test. Các mô hình '
           r'khác trong báo cáo là mốc so sánh hoặc bước trung gian để đi tới các mô hình này.',
           r'\begin{table}[!htbp]\centering\footnotesize',
           r'\caption[Mô hình cuối của đồ án cho từng bài toán]{Mô hình cuối của đồ án cho từng bài toán. Tỉ số là MAE '
           r'theo cell trên test của mô hình cuối chia cho mô hình được so (nhỏ hơn 1: mô hình cuối tốt hơn); ý nghĩa '
           r'theo kiểm định Wilcoxon ghép cặp theo cell.}\label{tab:final}',
           r'\setlength{\tabcolsep}{4pt}'
           r'\begin{tabularx}{\textwidth}{' + L('p{2.0cm}') + L('p{1.6cm}') + L('p{3.3cm}') + L('X') + r'l}\toprule',
           r'Bài toán & Mô hình cuối & Cách làm & Kết quả chính trên test & Mục \\\midrule']
    out += [' & '.join(r) + r' \\' + (r'\addlinespace' if i < len(rows) - 1 else '') for i, r in enumerate(rows)]
    out += [r'\bottomrule\end{tabularx}\end{table}']
    return '\n'.join(out)


def e33_item(E):
    """Đoạn kết luận về bài toán dự báo (HiGBDT, HiPINN)."""
    t = (r'Với bài toán dự báo, đồ án thiết kế hai mô hình dùng lịch sử SOH đo của cell: HiGBDT, một GBDT có đặc trưng '
         r'lịch sử SOH, và HiPINN, trung vị của HiGBDT, PINN-dự báo và nhánh theo đời (\Cref{sec:e33pp,sec:e33res}). '
         f'HiPINN có MAE thấp nhất trong tám mô hình ở {E["best_n"]}/{E["n_all"]} trường hợp, bằng {rng(E["r_hl"])} lần '
         f'HiGBDT và {rng(E["r_hk"])} lần PINN-GBDT, mô hình dự báo không có đặc trưng lịch sử. Đặc trưng lịch sử giúp '
         f'GBDT ở mọi trường hợp ({rng(E["r_g"])} lần) '
         + (f'nhưng giúp PINN-dự báo không đều ({rng(E["r_p"])} lần).' if max(E['r_p']) >= 1 else
            f'và cũng giúp PINN-dự báo ({rng(E["r_p"])} lần).'))
    if E['mit_h'] < E['mit_k']:
        t += (f' Ở 10\\,\\% đời cuối của MIT, MAE giảm từ {v(E["mit_k"], 1)} với PINN-GBDT xuống {v(E["mit_h"], 1)} với '
              f'HiPINN ($\\times10^{{-3}}$).')
    return t


def e33_abstract(E):
    return (r' Đưa lịch sử của cell vào dự báo, đồ án xây dựng HiGBDT (GBDT có đặc trưng lịch sử SOH) và HiPINN, mô hình '
            r'lấy trung vị của HiGBDT, PINN-dự báo và một nhánh theo đời cell; khi có lịch sử đo dung lượng, HiPINN có sai '
            rf'số dự báo SOH bằng {rng(E["r_hk"])} lần PINN-GBDT, mô hình dự báo không có đặc trưng lịch sử.')


# --------------------------------------------------------------------------- hình
def m3(x, n=1):
    """SOH (hoặc sai số SOH) theo đơn vị 10⁻³."""
    return v(1e3 * x, n)


def pct_rng(vals, n=0):
    """Khoảng phần trăm: [0.02, 0.09] -> '2--9\\,\\%'."""
    lo, hi = v(100 * min(vals), n), v(100 * max(vals), n)
    return (lo if lo == hi else f'{lo}--{hi}') + r'\,\%'


# --------------------------------------------------------------------------- E33: dự báo có đặc trưng lịch sử (HiPINN)
E33_CSV = ['summary', 'tests', 'rul', 'rul_tests', 'choices', 'traj_pick',
           'tail', 'cal', 'val', 'val_tests']
E33_TAB = ['gbdt', 'pinn', 'ket_hop', 'gbdt_ls', 'pinn_ls', 'doi', 'hipinn_w', 'hipinn']      # cột của bảng chính
E33_HEAD = {'gbdt': ('GBDT', 'theo tầm'), 'pinn': ('PINN-', 'dự báo'), 'ket_hop': ('PINN-', 'GBDT'),
            'gbdt_ls': ('', 'HiGBDT'), 'pinn_ls': ('PINN có', 'lịch sử'), 'doi': ('Theo', 'đời'),
            'tb3': ('', 'HiPINN-tb'), 'hipinn_w': ('', 'HiPINN-w'), 'hipinn': ('', 'HiPINN')}
E33_TXT = {'giu': 'giữ nguyên', 'tuyen': 'ngoại suy tuyến tính', 'gbdt': 'GBDT theo tầm', 'gbdt_ls': 'HiGBDT',
           'pinn': 'PINN-dự báo', 'pinn_ls': 'PINN-dự báo có lịch sử', 'ket_hop': 'PINN-GBDT',
           'ket_hop_ls': 'HiGBDT ghép PINN theo trọng số chọn', 'doi': 'nhánh theo đời',
           'tb3': 'HiPINN-tb', 'hipinn_w': 'HiPINN-w', 'hipinn': 'HiPINN'}
E33_SET = [('lab30', 'do', r'30\,\% nhãn'), ('all', 'do', 'mọi cell train')]      # lịch sử SOH tới gốc: dung lượng đo
E33_RUNS = 120
E33_FIGS = ('e33_mae', 'e33_traj', 'e33_tail')


def e33_load():
    """Khi runs/E33 đủ 120 lượt: tính lại bảng (analysis/analyze_e33.py) và vẽ lại hình; bảng giữ bản sao ở
    generated/data/, nên không có runs/ vẫn dựng được."""
    from pinnsoh.forecast_combo import ROOT as R33
    fresh = len(glob.glob(f'{R33}/fc_*.npz')) >= E33_RUNS
    if fresh:
        import analyze_e33, figures_report as FR
        analyze_e33.main()
        for name in E33_FIGS:
            FR.FIGS[name]()
            shutil.copy(f'results/fig_{name}.pdf', os.path.join(FIG, f'fig_{name}.pdf'))
    D = {}
    for name in E33_CSV:
        p = keep_copy(f'results/e33_{name}.csv', f'e33_{name}.csv')
        if p is None:
            return None
        D[name] = pd.read_csv(p)
    if 'hipinn_w' not in D['summary'].columns:           # bản sao của lần chạy trước khi HiPINN dùng trung vị
        print('generated/data/e33_*.csv là bản cũ (HiPINN chưa dùng trung vị): chạy python run.py E33 rồi dựng lại')
        return None
    return D


def head2(a, b):
    """Ô tiêu đề hai dòng, căn phải như cột số."""
    return r'\begin{tabular}[b]{@{}r@{}}' + a + r'\\' + b + r'\end{tabular}'


def group_row(text, ncol):
    return r'\multicolumn{' + str(ncol) + r'}{@{}l}{\textit{' + text + r'}} \\'


class E33Data:
    """Truy cập gọn các bảng kết quả E33."""

    def __init__(self, D):
        self.D = D
        self.S, self.T, self.R, self.RT, self.CH = D['summary'], D['tests'], D['rul'], D['rul_tests'], D['choices']
        self.TL, self.TP, self.V, self.VT = D['tail'], D['traj_pick'], D['val'], D['val_tests']

    def val(self, b, sc, ds, m):
        S = self.S
        return float(S[(S.budget == b) & (S.scen == sc) & (S.ds == ds)][m].iloc[0])

    def rv(self, b, sc, ds, col):
        R = self.R
        return float(R[(R.budget == b) & (R.scen == sc) & (R.ds == ds)][col].iloc[0])

    def tail(self, b, sc, ds, pb, col):
        T = self.TL
        return float(T[(T.budget == b) & (T.scen == sc) & (T.ds == ds) & (T.pb == pb)][col].iloc[0])

    def half(self, b, ds, part, m):
        """MAE theo cell trên một nửa validation ('chon' hoặc 'hieu_chinh')."""
        V = self.V
        return float(V[(V.budget == b) & (V.ds == ds) & (V.part == part)][m].iloc[0])

    def half_test(self, b, ds, part, other):
        """Kiểm định HiPINN so với `other` trên một nửa validation: (p_holm, HiPINN tốt hơn, số cell tốt hơn, n)."""
        T = self.VT
        r = T[(T.budget == b) & (T.ds == ds) & (T.part == part) & (T.b == other)].iloc[0]
        return float(r.p_holm), r.a_wins == 1, int(r.a_better), int(r.n)

    def row(self, b, sc, ds, x, y, rul=False):
        """Dòng kiểm định của cặp (x, y) dù bảng lưu theo thứ tự nào, kèm hướng: (p_holm, x tốt hơn, tỉ số x/y, dòng)."""
        tab = self.RT if rul else self.T
        r = tab[(tab.budget == b) & (tab.scen == sc) & (tab.ds == ds)
                & (((tab.a == x) & (tab.b == y)) | ((tab.a == y) & (tab.b == x)))].iloc[0]
        return float(r.p_holm), (r.a_wins == 1) == (r.a == x), float(r.ratio if r.a == x else 1 / r.ratio), r

    def better(self, b, sc, x, y, rul=False):
        """Các bộ mà x tốt hơn / kém hơn y có ý nghĩa (Wilcoxon, Holm)."""
        up, dn = [], []
        for ds in DS:
            p, xw, _, _ = self.row(b, sc, ds, x, y, rul)
            if p < 0.05:
                (up if xw else dn).append(ds)
        return up, dn

    def ratios(self, sets, x, y):
        return [self.val(b, sc, ds, x) / self.val(b, sc, ds, y) for b, sc in sets for ds in DS]


def e33_table(E):
    ncol = 1 + len(E33_TAB)
    rows = []
    for b, sc, lab in E33_SET:
        rows.append(group_row(lab, ncol))
        for ds in DS:
            vals = [E.val(b, sc, ds, m) for m in E33_TAB]
            cells = [(r'\textbf{' + m3(x, 2) + '}') if x == min(vals) else m3(x, 2) for x in vals]
            p, xw, _, _ = E.row(b, sc, ds, 'hipinn', 'gbdt_ls')
            cells[-1] += (r'$^{*}$' if xw else r'$^{\dagger}$') if p < 0.05 else r'\phantom{$^{*}$}'
            rows.append(' & '.join([ds] + cells) + r' \\')
        rows.append(r'\addlinespace[3pt]')
    return (r'\begin{table}[!htbp]\centering\small' '\n'
            r'\caption[MAE của SOH dự báo theo mô hình]{MAE theo cell ($\times10^{-3}$) của SOH dự báo trên mọi chu kỳ đo được '
            r'sau gốc ($0{,}2L$, $0{,}4L$, $0{,}6L$), trung bình qua 3 lần lặp; in đậm là thấp nhất mỗi hàng. '
            r'$^{*}$/$^{\dagger}$: HiPINN tốt hơn/kém hơn có ý nghĩa so với HiGBDT theo kiểm định Wilcoxon ghép '
            r'cặp theo cell (Holm qua bốn bộ, $p<0{,}05$).}\label{tab:e33}' '\n'
            r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * len(E33_TAB) + r'}\toprule' '\n'
            r' & \multicolumn{' + str(len(E33_TAB)) + r'}{c}{MAE của SOH dự báo ($\times10^{-3}$) ' + DN + r'} \\'
            r'\cmidrule(lr){2-' + str(1 + len(E33_TAB)) + '}\n'
            r' & \multicolumn{3}{c}{chỉ đặc trưng gốc} & \multicolumn{5}{c}{có đặc trưng lịch sử} \\'
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-9}' '\n'
            'Bộ & ' + ' & '.join(head2(*E33_HEAD[m]) for m in E33_TAB) + r' \\\midrule' '\n'
            + '\n'.join(rows[:-1]) + '\n' r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')


def e33_rul_table(E):
    ncol = 4 + len(E33_TAB)
    rr = []
    for b, sc, lab in E33_SET:
        rr.append(group_row(lab, ncol))
        for ds in DS:
            vals = [E.rv(b, sc, ds, f'e_{m}') for m in E33_TAB]
            cells = [(r'\textbf{' + v(x, 0) + '}') if x == min(vals) else v(x, 0) for x in vals]
            p, hw, _, _ = E.row(b, sc, ds, 'hipinn', 'ket_hop', rul=True)
            cells[-1] += (r'$^{*}$' if hw else r'$^{\dagger}$') if p < 0.05 else r'\phantom{$^{*}$}'
            rr.append(' & '.join([ds, v(E.rv(b, sc, ds, 'true'), 0)] + cells
                                 + [pct(E.rv(b, sc, ds, 'in20_hipinn')), pct(E.rv(b, sc, ds, 'cov_h'))]) + r' \\')
        rr.append(r'\addlinespace[3pt]')
    return (r'\begin{table}[!htbp]\centering\small' '\n'
            r'\caption[MAE của RUL dự báo theo mô hình]{MAE của RUL (chu kỳ), trung bình qua các gốc ở 25, 50 và 75\,\% chu kỳ '
            r'cuối đời thật và qua 3 lần lặp; in đậm là thấp nhất mỗi hàng. Cột RUL thực là RUL thật trung bình, để đối '
            r'chiếu cỡ sai số. Hai cột cuối là của HiPINN: tỉ lệ RUL dự báo nằm trong $\pm20\,\%$ RUL thực (cao hơn là tốt '
            r'hơn) và độ phủ của khoảng RUL 90\,\% (tốt nhất là gần 90\,\%). $^{*}$/$^{\dagger}$: RUL của HiPINN tốt hơn/kém '
            r'hơn có ý nghĩa thống kê so với PINN-GBDT (Wilcoxon ghép cặp theo cell, hiệu chỉnh Holm qua bốn bộ, '
            r'$p<0{,}05$); không có dấu: khác biệt không có ý nghĩa thống kê.}\label{tab:e33rul}' '\n'
            r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * (3 + len(E33_TAB)) + r'}\toprule' '\n'
            r' & & \multicolumn{' + str(len(E33_TAB)) + r'}{c}{MAE của RUL (chu kỳ) ' + DN + r'} & '
            r'\multicolumn{2}{c}{HiPINN} \\\cmidrule(lr){3-' + str(2 + len(E33_TAB)) + r'}\cmidrule(lr){'
            + str(3 + len(E33_TAB)) + '-' + str(4 + len(E33_TAB)) + '}\n'
            r'Bộ & ' + head2('RUL', 'thực') + ' & ' + ' & '.join(head2(*E33_HEAD[m]) for m in E33_TAB)
            + ' & ' + head2('Trong', r'$\pm20\,\%$ ' + UP) + ' & ' + head2('Độ phủ', r'(90\,\%)') + r' \\\midrule' '\n'
            + '\n'.join(rr[:-1]) + '\n' r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')


E33_VAL = [('chon', ['gbdt_ls', 'hipinn_w', 'hipinn']), ('hieu_chinh', ['gbdt_ls', 'hipinn_w', 'tb3', 'hipinn'])]


def e33_val_table(E):
    """MAE trên hai nửa cell validation (mỗi cell lấy trung bình qua các lượt nó thuộc nửa đó)."""
    cols = [(part, m) for part, ms in E33_VAL for m in ms]
    rows = []
    for b, sc, lab in E33_SET:
        rows.append(group_row(lab, 1 + len(cols)))
        for ds in DS:
            cells = []
            for part, ms in E33_VAL:
                vals = [E.half(b, ds, part, m) for m in ms]
                cells += [(r'\textbf{' + m3(x, 2) + '}') if x == min(vals) else m3(x, 2) for x in vals]
            p, hw, _, _ = E.half_test(b, ds, 'hieu_chinh', 'gbdt_ls')
            cells[-1] += (r'$^{*}$' if hw else r'$^{\dagger}$') if p < 0.05 else r'\phantom{$^{*}$}'
            rows.append(' & '.join([ds] + cells) + r' \\')
        rows.append(r'\addlinespace[3pt]')
    n1, n2 = len(E33_VAL[0][1]), len(E33_VAL[1][1])
    return (r'\begin{table}[!htbp]\centering\small' '\n'
            r'\caption[MAE của SOH dự báo trên hai nửa validation]{MAE theo cell ($\times10^{-3}$) của SOH dự báo trên '
            r'hai nửa cell validation; mỗi cell lấy trung bình qua các lượt mà nó thuộc nửa đó. Nửa chọn là nơi chọn trọng '
            r'số của HiPINN-w; nửa hiệu chỉnh không dùng để chọn gì cho các mô hình trong bảng. In đậm là thấp nhất trong '
            r'mỗi nửa. $^{*}$/$^{\dagger}$: trên nửa hiệu chỉnh, HiPINN tốt hơn/kém hơn có ý nghĩa so với HiGBDT (Wilcoxon '
            r'ghép cặp theo cell, Holm qua bốn bộ, $p<0{,}05$).}\label{tab:e33val}' '\n'
            r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * len(cols) + r'}\toprule' '\n'
            r' & \multicolumn{' + str(len(cols)) + r'}{c}{MAE theo cell ($\times10^{-3}$) ' + DN + r'} \\'
            r'\cmidrule(lr){2-' + str(1 + len(cols)) + '}\n'
            r' & \multicolumn{' + str(n1) + r'}{c}{nửa chọn} & \multicolumn{' + str(n2) + r'}{c}{nửa hiệu chỉnh} \\'
            r'\cmidrule(lr){2-' + str(1 + n1) + r'}\cmidrule(lr){' + str(2 + n1) + '-' + str(1 + n1 + n2) + '}\n'
            'Bộ & ' + ' & '.join(head2(*E33_HEAD[m]) for _, m in cols) + r' \\\midrule' '\n'
            + '\n'.join(rows[:-1]) + '\n' r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')


E34_CSV = ['anchor', 'cost']
E34_RUNS = 60


def e34_load():
    """HiGBDT cho bài toán cập nhật. Khi runs/E34 đủ 60 lượt: tính lại bảng và hình (analysis/analyze_e34.py); bảng giữ
    bản sao ở generated/data/, nên không có runs/ vẫn dựng được."""
    from pinnsoh.update_hist import ROOT as R34
    if len(glob.glob(f'{R34}/up_*.npz')) >= E34_RUNS:
        import analyze_e34
        analyze_e34.main()
    if os.path.exists('results/fig_e34_cost.pdf'):
        shutil.copy('results/fig_e34_cost.pdf', os.path.join(FIG, 'fig_e34_cost.pdf'))
    D = {}
    for name in E34_CSV:
        p = keep_copy(f'results/e34_{name}.csv', f'e34_{name}.csv')
        D[name] = pd.read_csv(p) if p else None
    return D if D['anchor'] is not None else None


def e34_anchor(D34):
    """Phụ lục: HiGBDT cho bài toán cập nhật (E34)."""
    GA, C = D34['anchor'].copy(), D34['cost']
    from analyze_e33 import holm                   # Holm qua bốn bộ trong từng lịch đo, như các bảng khác
    GA['p_selected'] = GA.p_selected.astype(float)
    for t_, g_ in GA.groupby('tier'):
        GA.loc[g_.index, 'p_selected'] = holm(list(g_.p_selected))
    assert 'source' not in GA.columns or GA.source.iloc[0] == 'E34', 'bảng E34 không phải của lần chạy E34'
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    tiers = [('q20', r'mỗi $0{,}2L$'), ('q10', r'mỗi $0{,}1L$'), ('q05', r'mỗi $0{,}05L$'), ('M2', 'mọi lần trước')]
    am = ['hold', 'linear', 'e32_style', 'history_delta', 'history_residual', 'selected']
    alab = [('Giữ', 'lần đo'), ('Tuyến', 'tính'), ('GBDT-', 'cập nhật'), ('', r'HiGBDT-$\Delta$'), ('', 'HiGBDT-R'),
            ('', 'HiGBDT-V')]
    ga = lambda ds, t, m: float(GA[(GA.ds == ds) & (GA.tier == t)][m].iloc[0])
    row = lambda ds, t: GA[(GA.ds == ds) & (GA.tier == t)].iloc[0]
    ar = []
    for t, tl in tiers:
        ar.append(group_row('lịch đo ' + tl if t != 'M2' else 'biết mọi lần đo trước', 1 + len(am)))
        for ds in DS:
            vals = [ga(ds, t, m) for m in am]
            cells = [(r'\textbf{' + m3(x, 2) + '}') if x == min(vals) else m3(x, 2) for x in vals]
            r_ = row(ds, t)
            cells[-1] += ((r'$^{*}$' if r_.wins_selected == 1 else r'$^{\dagger}$') if r_.p_selected < 0.05
                          else r'\phantom{$^{*}$}')
            ar.append(' & '.join([ds] + cells) + r' \\')
        ar.append(r'\addlinespace[3pt]')
    tab = (r'\begin{table}[!htbp]\centering\small' '\n'
           r'\caption[HiGBDT cho bài toán cập nhật]{Cập nhật SOH ở các chu kỳ không đo từ lần đo dung lượng gần nhất, '
           r'theo lịch đo: MAE theo cell ($\times10^{-3}$), 30\,\% nhãn, 3 lần lặp $\times$ 5 fold. '
           r'$^{*}$/$^{\dagger}$: HiGBDT-V tốt hơn/kém hơn có ý nghĩa so với GBDT-cập nhật '
           r'(Wilcoxon ghép cặp theo cell, hiệu chỉnh Holm qua bốn bộ trong từng lịch đo, $p<0{,}05$).}\label{tab:anchor}' '\n'
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * len(am) + r'}\toprule' '\n'
           r' & \multicolumn{' + str(len(am)) + r'}{c}{MAE của SOH cập nhật ($\times10^{-3}$) ' + DN + r'} \\'
           r'\cmidrule(lr){2-' + str(1 + len(am)) + '}\n'
           r'Bộ & ' + ' & '.join(head2(*x) for x in alab) + r' \\\midrule' '\n' + '\n'.join(ar[:-1]) + '\n'
           r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    low = {t: [ds for ds in DS if ga(ds, t, 'selected') < ga(ds, t, 'e32_style')] for t, _ in tiers}
    worse = {t: [ds for ds in DS if row(ds, t).p_selected < 0.05 and row(ds, t).wins_selected == 0] for t, _ in tiers}
    worse3 = [ds for ds in DS if all(ds in worse[t] for t in ('q20', 'q10', 'q05'))]
    sparse_no = len(low['q20']) <= 1 and len(low['q10']) <= 1
    keep = len(low['q10']) < 3 and len(low['q20']) < 3
    odd = [r_ for r_ in GA.itertuples() if r_.tier != 'M2' and r_.p_selected < 0.05 and r_.flip_selected > 0]
    tl = dict(tiers)
    q20 = [ga(ds, 'q20', 'e32_style') for ds in DS]
    q20_hold = [ga(ds, 'q20', 'e32_style') / ga(ds, 'q20', 'hold') for ds in DS]
    m2_r = [ga(ds, 'M2', 'history_residual') / ga(ds, 'M2', 'e32_style') for ds in DS]
    text = (r'Đồ án đưa đặc trưng lịch sử của HiGBDT vào bài toán cập nhật của \Cref{sec:mo_rong}: ước lượng SOH ở các '
            r'chu kỳ không đo từ lần đo dung lượng gần nhất, trên cùng fold với các thí nghiệm khác, 30\,\% nhãn, $L$ và '
            r'chuẩn hoá tính trên cell train. GBDT-cập nhật dùng đặc trưng sạc hiện tại và độ đổi của chúng kể từ lần đo, '
            r'SOH ở lần đo, khoảng chu kỳ tới lần đo và độ dốc giữa hai lần đo gần nhất. Hai mô hình dùng lịch sử thêm hai '
            r'trung bình trượt mũ nhân quả của đặc trưng sạc ($\rho = 0{,}1$ và $0{,}03$), log khoảng chu kỳ và chỉ số chu '
            r'kỳ: HiGBDT-$\Delta$ học độ giảm SOH kể từ lần đo, HiGBDT-R học phần dư so với ngoại suy tuyến tính từ hai lần '
            r'đo gần nhất. HiGBDT-V chọn trên validation, theo từng lịch đo, một trong ba mô hình đó và trọng số trộn nó '
            r'với giữ lần đo hoặc ngoại suy tuyến tính. Mỗi mô hình học trên tối đa khoảng 60 nghìn cặp (chu kỳ, lần đo), '
            r'cân theo cell và lịch đo, có thêm hai pha đo ngẫu nhiên cho mỗi lịch thưa; số cây chọn trên validation. '
            r'\Cref{tab:anchor} cho kết quả. Chỉ với một lần đo mỗi $0{,}2L$ chu kỳ, GBDT-cập nhật đạt MAE '
            + rng([1e3 * x for x in q20]) + r' ($\times10^{-3}$), bằng ' + rng(q20_hold) + r' lần việc giữ nguyên giá trị '
            r'đo gần nhất. Khi biết mọi lần đo trước, HiGBDT-R có MAE bằng ' + rng(m2_r) + ' lần GBDT-cập nhật. ')
    if sparse_no:
        text += (r'Khi đo thưa, lịch sử không giúp: HiGBDT-V chỉ có MAE trung bình thấp hơn GBDT-cập nhật ở '
                 f'{len(low["q20"])}/4 bộ với lịch đo mỗi $0{{,}}2L$ và {len(low["q10"])}/4 bộ với lịch mỗi $0{{,}}1L$')
    else:
        text += (r'Khi đo thưa, HiGBDT-V có MAE trung bình thấp hơn GBDT-cập nhật ở '
                 f'{len(low["q20"])}/4 bộ với lịch đo mỗi $0{{,}}2L$ và {len(low["q10"])}/4 bộ với lịch mỗi $0{{,}}1L$')
    text += ((f', và kém hơn có ý nghĩa trên {jv(worse3)} ở cả ba lịch đo thưa. ') if worse3 else '. ')
    text += (f'Lịch sử có lợi hơn khi đo dày: {len(low["q05"])}/4 bộ'
             + (f' ({jv(low["q05"])})' if 0 < len(low['q05']) < 4 else '') + r' ở lịch mỗi $0{,}05L$ và '
             f'{len(low["M2"])}/4 bộ khi biết mọi lần đo trước, nhưng mức sau không phải là đo thưa.')
    for ds in DS:                                  # kiểm định nghiêng về HiGBDT-V dù MAE trung bình cao hơn
        oo = [r_ for r_ in odd if r_.ds == ds and r_.wins_selected == 1]
        if oo:
            one = all(r_.flip_selected == 1 for r_ in oo)
            text += (f' Ở {ds} với lịch {jv([tl[r_.tier] for r_ in oo])}, HiGBDT-V tốt hơn ở '
                     + jv([f'{int(r_.better_selected)}/{int(r_.n_cell)}' for r_ in oo])
                     + f' cell nên kiểm định nghiêng về nó, dù MAE trung bình cao hơn vì '
                     + ('một cell sai lớn.' if one else 'vài cell sai lớn.'))
    text += (
             ' Vì vậy đồ án giữ GBDT-cập nhật cho lịch đo thưa.' if keep else '')
    fig = ''
    if C is not None and os.path.exists(os.path.join(FIG, 'fig_e34_cost.pdf')):
        cc = lambda ds, t, col: float(C[(C.ds == ds) & (C.tier == t)][col].iloc[0])
        beat = [ds for ds in DS if cc(ds, 'q20', 'e32_style') < cc(ds, 'q05', 'hold')]
        fewer = [cc(ds, 'q05', 'n_measured') / cc(ds, 'q20', 'n_measured') for ds in DS]
        text += (r' \Cref{fig:e34cost} đặt sai số cạnh số lần đo dung lượng trung bình mỗi cell. Với lịch mỗi $0{,}2L$, '
                 r'GBDT-cập nhật có MAE ' + (f'thấp hơn việc giữ lần đo với lịch mỗi $0{{,}}05L$ trên {"cả bốn bộ" if len(beat) == 4 else jv(beat)}, '
                                            f'tức với số lần đo ít hơn {rng(fewer, 1)} lần.' if beat else
                                            r'chưa thấp hơn việc giữ lần đo với lịch mỗi $0{,}05L$ ở bộ nào.'))
        fig = ('\n\n' r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e34_cost.pdf}'
               '\n' r'\caption[Sai số cập nhật theo số lần đo]{MAE ($\times10^{-3}$, thang log) của SOH cập nhật theo số '
               r'lần đo dung lượng trung bình mỗi cell, bốn lịch đo (mỗi $0{,}2L$, $0{,}1L$, $0{,}05L$ và biết mọi lần đo '
               r'trước), 30\,\% nhãn, trung bình qua 3 lần lặp. Đường nối chỉ để dễ so, không phải quan hệ liên tục đã '
               r'ước lượng.}\label{fig:e34cost}' '\n' r'\end{figure}')
    return text, tab + fig


def e33_section(D):
    """Mục kết quả E33 và đoạn HiGBDT cho bài toán cập nhật (Chương 5). Lời văn viết tay; số lấy từ bảng kết quả, nhận định định tính được
    kiểm bằng điều kiện trên số liệu (sai thì dừng dựng báo cáo hoặc đổi câu, thay vì in câu sai)."""
    E = E33Data(D)
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    where = lambda xs: 'cả bốn bộ' if len(xs) == 4 else (jv(xs) if xs else 'không bộ nào')
    do = [('lab30', 'do'), ('all', 'do')]
    ALL = do
    CH = E.CH
    sgn = lambda x: ('+' if round(x, 1) > 0 else '') + v(x, 1)

    def sig_txt(up, dn, good='thấp hơn', bad='cao hơn'):
        parts = ([f'{good} có ý nghĩa trên {where(up)}'] if up else []) + ([f'{bad} có ý nghĩa trên {where(dn)}'] if dn else [])
        return ', '.join(parts) if parts else 'không khác có ý nghĩa trên bộ nào'

    def sig2(x, y):
        """Kết quả kiểm định x so với y ở hai ngân sách (lịch sử đo), thành lời."""
        U = [set(E.better(b, sc, x, y)[0]) for b, sc in do]
        Dn = [set(E.better(b, sc, x, y)[1]) for b, sc in do]
        both = [ds for ds in DS if all(ds in u for u in U)]
        one = [ds for ds in DS if any(ds in u for u in U) and ds not in both]
        dn = [ds for ds in DS if any(ds in d for d in Dn)]
        parts = (([f'thấp hơn có ý nghĩa ở cả hai ngân sách trên {where(both)}'] if both else [])
                 + ([f'chỉ ở một ngân sách trên {jv(one)}'] if one else [])
                 + ([f'cao hơn có ý nghĩa trên {jv(dn)}'] if dn else []))
        return ', '.join(parts) if parts else 'không khác có ý nghĩa trên bộ nào'

    def count_sig(x, y, sets):
        return (sum(len(E.better(b, sc, x, y)[0]) for b, sc in sets),
                sum(len(E.better(b, sc, x, y)[1]) for b, sc in sets))

    def n_lower(x, y, sets):
        return sum(E.val(b, sc, ds, x) < E.val(b, sc, ds, y) for b, sc in sets for ds in DS)

    n_all = len(ALL) * len(DS)
    # ---------------- bảng và mở đầu
    own = [E.val(b, sc, ds, m) / E.val(b, sc, ds, 'hipinn') for b, sc in ALL for ds in DS for m in ('giu', 'tuyen')]
    p0 = (r'\Cref{tab:e33} cho MAE của SOH dự báo ở hai ngân sách nhãn, 30\,\% nhãn hoặc mọi cell train; lịch sử SOH '
          r'của cell test tới gốc là dung lượng đo. Hai mô hình chỉ dựa vào chính cell là giữ nguyên và ngoại suy '
          f'tuyến tính có sai số gấp {rng(own, 1)} lần HiPINN nên không đưa vào bảng (có trong \\Cref{{fig:e33mae}}).')
    # ---------------- lịch sử trong từng nhánh
    r_g = E.ratios(do, 'gbdt_ls', 'gbdt')
    r_p = E.ratios(do, 'pinn_ls', 'pinn'); up_, dp_ = count_sig('pinn_ls', 'pinn', do)
    assert max(r_g) < 1, 'câu "lịch sử giúp GBDT ở mọi trường hợp" không còn đúng'
    p1 = (r'Trước hết, xét tác động của việc thêm lịch sử vào từng nhánh. Với GBDT, đặc trưng lịch sử giúp rõ rệt: HiGBDT có MAE bằng '
          f'{rng(r_g)} lần GBDT theo tầm, {sig2("gbdt_ls", "gbdt")}. Cây quyết định dùng '
          'được trực tiếp những thông tin như độ dốc trên nhiều cửa sổ hay mức dao động gần đây để nhận ra cell đang suy '
          'giảm nhanh hay chậm. ')
    if max(r_p) < 1 and not dp_:
        p1 += f'PINN-dự báo cũng tốt lên khi có lịch sử, MAE bằng {rng(r_p)} lần bản không có lịch sử. '
    else:
        p1 += (f'Với PINN-dự báo, lợi ích không đều: bản có lịch sử có MAE bằng {rng(r_p)} lần bản không có lịch sử, '
               + sig2('pinn_ls', 'pinn') + '. Một cách giải thích: mạng chỉ chọn hai tham số $(a, \\lambda)$ của một '
               'đường suy giảm có dạng cố định, nên thông tin thêm từ lịch sử chủ yếu giúp ước lượng tốc độ hiện tại, '
               'điều mà hai độ dốc trong $\\mathbf{z}_k$ đã có một phần, trong khi số đầu vào tăng hơn gấp đôi. Vì vậy '
               'HiPINN giữ PINN-dự báo không có lịch sử làm nhánh PINN, còn lịch sử được đưa vào mô hình qua hai nhánh kia.')
    # ---------------- HiPINN-w: trọng số chọn trên vài cell
    halves = [(b, ds) for b in ('lab30', 'all') for ds in DS]
    r_sel = [E.half(b, ds, 'chon', 'hipinn_w') / E.half(b, ds, 'chon', 'gbdt_ls') for b, ds in halves]
    r_cal = [E.half(b, ds, 'hieu_chinh', 'hipinn_w') / E.half(b, ds, 'hieu_chinh', 'gbdt_ls') for b, ds in halves]
    r_wt = E.ratios(do, 'hipinn_w', 'gbdt_ls'); k_wt = n_lower('gbdt_ls', 'hipinn_w', do)
    assert np.mean(r_sel) < np.mean(r_cal), 'HiPINN-w không tốt hơn trên nửa chọn so với nửa hiệu chỉnh'
    lost = np.mean(r_cal) >= 1 or k_wt > len(do) * len(DS) / 2
    p2 = (r'HiPINN-w, phương án đầu của HiPINN, chọn trọng số của ba nhánh và nhánh PINN trên nửa '
          f'validation dùng để chọn; nửa này chỉ có {int(CH.n_sel.min())}--{int(CH.n_sel.max())} cell mỗi lượt. Trên '
          f'chính nửa đó (\\Cref{{tab:e33val}}), HiPINN-w có MAE bằng {rng(r_sel)} lần HiGBDT; trên nửa hiệu chỉnh, '
          f'tỉ số này là '
          f'{rng(r_cal)}, và trên cell test là '
          f'{rng(r_wt)}, cao hơn HiGBDT ở {k_wt}/{len(do) * len(DS)} trường hợp. Trọng số chọn trên vài cell phản '
          'ánh chính các cell đó nhiều hơn là quy luật chung, nên '
          + ('ưu thế mất đi' if lost else 'ưu thế giảm đi') + ' khi chuyển sang cell khác. HiPINN vì vậy dùng trung vị, '
          r'không có trọng số cần chọn (\Cref{sec:e33pp}).')
    # ---------------- HiPINN
    best_n = sum(min(E33_TAB, key=lambda m: E.val(b, sc, ds, m)) == 'hipinn' for b, sc in ALL for ds in DS)
    r_hl = E.ratios(ALL, 'hipinn', 'gbdt_ls'); n_hl = n_lower('hipinn', 'gbdt_ls', ALL)
    up_l, dn_l = count_sig('hipinn', 'gbdt_ls', ALL)
    frac_l = [E.row(b, sc, ds, 'hipinn', 'gbdt_ls')[3] for b, sc in ALL for ds in DS]
    frac_l = [(r.a_better if r.a == 'hipinn' else r.n - r.a_better) / r.n for r in frac_l]
    r_hk = E.ratios(do, 'hipinn', 'ket_hop'); up_k, dn_k = count_sig('hipinn', 'ket_hop', do)
    r_hw = E.ratios(ALL, 'hipinn', 'hipinn_w'); n_hw = n_lower('hipinn', 'hipinn_w', ALL)
    r_ht = E.ratios(ALL, 'hipinn', 'tb3'); n_ht = n_lower('hipinn', 'tb3', ALL)
    c_w = sum(E.half(b, ds, 'hieu_chinh', 'hipinn') < E.half(b, ds, 'hieu_chinh', 'hipinn_w') for b, ds in halves)
    c_l = sum(E.half(b, ds, 'hieu_chinh', 'hipinn') < E.half(b, ds, 'hieu_chinh', 'gbdt_ls') for b, ds in halves)
    s_l = [E.half_test(b, ds, 'hieu_chinh', 'gbdt_ls') for b, ds in halves]
    su_l, sd_l = sum(p < 0.05 and w for p, w, _, _ in s_l), sum(p < 0.05 and not w for p, w, _, _ in s_l)
    p3 = (r'Xét tiếp HiPINN, mô hình lấy trung vị của ba nhánh. '
          f'HiPINN có MAE thấp nhất trong tám mô hình của \\Cref{{tab:e33}} ở {best_n}/{n_all} trường hợp (bộ $\\times$ '
          f'ngân sách). So với HiGBDT, nhánh tốt nhất khi đứng riêng, MAE của HiPINN bằng {rng(r_hl)} lần và thấp '
          f'hơn ở {n_hl}/{n_all} trường hợp; theo kiểm định ghép cặp, HiPINN tốt hơn có ý nghĩa ở {up_l}/{n_all} và kém '
          f'hơn có ý nghĩa ở {dn_l}/{n_all} trường hợp, với {pct_rng(frac_l)} số cell có sai số thấp hơn. So với '
          f'PINN-GBDT, MAE của HiPINN bằng {rng(r_hk)} lần (thấp hơn có ý nghĩa ở '
          f'{up_k}/{len(do) * len(DS)} trường hợp'
          + (f', cao hơn có ý nghĩa ở {dn_k}' if dn_k else '') + '). '
          f'Trung vị cũng tốt hơn hai cách ghép còn lại: MAE của HiPINN bằng {rng(r_hw)} lần HiPINN-w (thấp hơn ở '
          f'{n_hw}/{n_all} trường hợp) và {rng(r_ht)} lần HiPINN-tb, tức trung bình đều ba nhánh (thấp hơn ở '
          f'{n_ht}/{n_all}). Khi một '
          'nhánh sai xa, trung bình bị kéo theo, còn trung vị đi theo hai nhánh còn lại.')
    lab_b = {'all': 'khi học trên mọi cell train', 'lab30': r'với 30\,\% nhãn'}
    cl_b = {bb: [E.half(bb, ds, 'hieu_chinh', 'hipinn') < E.half(bb, ds, 'hieu_chinh', 'gbdt_ls') for ds in DS]
            for bb in ('all', 'lab30')}
    sg_b = {bb: [E.half_test(bb, ds, 'hieu_chinh', 'gbdt_ls') for ds in DS] for bb in ('all', 'lab30')}
    part_b = []
    for bb in ('all', 'lab30'):
        up_b = sum(p < 0.05 and w for p, w, _, _ in sg_b[bb]); dn_b = sum(p < 0.05 and not w for p, w, _, _ in sg_b[bb])
        part_b.append(f'{sum(cl_b[bb])}/4 bộ {lab_b[bb]}'
                      + (f' (tốt hơn có ý nghĩa ở {up_b}' + (f', kém hơn có ý nghĩa ở {dn_b}' if dn_b else '') + ')'
                         if up_b or dn_b else ''))
    weak = [bb for bb in ('all', 'lab30') if sum(cl_b[bb]) <= 2]
    p3b = (r'Dạng trung vị được chọn sau khi đã thấy kết quả kiểm tra của '
           'HiPINN-w, nên cần một kiểm tra không dùng cell test. Nửa hiệu chỉnh không dùng để chọn gì cho các '
           'mô hình HiGBDT, HiPINN-w, HiPINN-tb và HiPINN, và sai số của trung vị trên nửa này chưa được xem khi quyết '
           'định. Trên '
           f'nửa này (\\Cref{{tab:e33val}}), HiPINN có MAE thấp hơn HiPINN-w ở {c_w}/{len(halves)} trường hợp; so với '
           'HiGBDT, HiPINN thấp hơn ở '
           + ' và '.join(part_b) + '. ')
    if not weak:
        p3b += 'Kết quả trên hai phần dữ liệu cùng chiều, nên ưu thế của trung vị không chỉ do cách chọn trên test.'
    elif len(weak) < 2:
        p3b += ('Kết quả cùng chiều với test ' + ' và '.join(lab_b[bb] for bb in ('all', 'lab30') if bb not in weak)
                + '; ' + ' và '.join(lab_b[bb] for bb in weak) + ', ưu thế của HiPINN so với HiGBDT trên test chưa được '
                'nửa hiệu chỉnh xác nhận, nên một phần ưu thế đó có thể do chính việc chọn dạng trung vị sau khi xem '
                'kết quả.')
    else:
        p3b += ('Ở cả hai ngân sách, ưu thế của HiPINN so với HiGBDT trên test chưa được nửa hiệu chỉnh xác nhận, nên '
                'một phần ưu thế đó có thể do chính việc chọn dạng trung vị sau khi xem kết quả.')
    # ---------------- phần cuối đời
    tb = '0.9-1'
    pbs = ['<0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1']
    tail = lambda ds, m, b='all': 1e3 * E.tail(b, 'do', ds, tb, m)
    bias = lambda ds, m, b='all': 1e3 * E.tail(b, 'do', ds, tb, f'b_{m}')
    peak = [ds for ds in DS if max(pbs, key=lambda b_: E.tail('all', 'do', ds, b_, 'hipinn')) == tb]
    mono = all(np.all(np.diff([E.tail('all', 'do', ds, b_, m) for b_ in pbs]) > 0)
               for ds in DS for m in ('gbdt_ls', 'pinn', 'doi', 'hipinn'))
    M = 'MIT'
    tk, th, tl_, td = tail(M, 'ket_hop'), tail(M, 'hipinn'), tail(M, 'gbdt_ls'), tail(M, 'doi')
    best_tail = min(E33_TAB, key=lambda m: tail(M, m))
    assert bias(M, 'gbdt_ls') > 0 and bias(M, 'hipinn') > 0, 'ở MIT, dự báo cuối đời không còn cao hơn thực tế'
    p4 = (r'Sai số dự báo không đều theo tuổi của cell. \Cref{fig:e33tail} chia sai số theo phần đời của chính cell (mọi cell train, lịch '
          'sử đo). Sai số của ' + ('cả ba nhánh và HiPINN tăng dần về cuối đời trên mọi bộ' if mono else
                                   'các mô hình nhìn chung tăng về cuối đời')
          + (', và với HiPINN lớn nhất ở 10\\,\\% đời cuối trên ' + where(peak) if peak else '') + '. '
          f'Ở MIT, MAE trên 10\\,\\% đời cuối là {v(tk, 1)} với PINN-GBDT và {v(th, 1)} với HiPINN'
          + (f', giảm {pct(1 - th / tk)}' if th < tk else '') + f' ($\\times10^{{-3}}$; HiGBDT {v(tl_, 1)}). '
          f'Phần sai số còn lại chủ yếu là dự báo cao hơn thực tế: độ lệch trung bình ở đoạn này là '
          f'{sgn(bias(M, "hipinn"))} với HiPINN và {sgn(bias(M, "gbdt_ls"))} với HiGBDT. Thời điểm gãy mỗi cell '
          'một khác, nên dự báo trực tiếp thành một đường trung bình thoải hơn thực tế. ')
    if abs(bias(M, 'doi')) < abs(bias(M, 'gbdt_ls')):
        p4 += (f'Nhánh theo đời đặt hình dạng suy giảm lên đời dự đoán của từng cell nên độ lệch chỉ còn '
               f'{sgn(bias(M, "doi"))}, nhưng MAE của nó ({v(td, 1)}) '
               + ('không thấp hơn' if td >= th else 'chỉ thấp hơn một ít')
               + ' vì khi đời bị đoán sai thì điểm gãy cũng bị đặt sai chỗ. ')
    if best_tail != 'hipinn':
        p4 += f'Ở đoạn này, {E33_TXT[best_tail]} có MAE thấp nhất ({v(tail(M, best_tail), 1)}). '
    lt_k = {b: [ds for ds in DS if tail(ds, 'hipinn', b) < tail(ds, 'ket_hop', b)] for b in ('all', 'lab30')}
    p4 += (f'Trên bốn bộ, MAE ở 10\\,\\% đời cuối của HiPINN thấp hơn PINN-GBDT ở {len(lt_k["all"])}/4 '
           f'bộ với mọi cell train và {len(lt_k["lab30"])}/4 bộ với 30\\,\\% nhãn.')
    worst = max(DS, key=lambda ds: tail(ds, 'hipinn'))
    # ---------------- ví dụ
    p5 = (r'\Cref{fig:e33traj} minh hoạ dự báo cho một cell mỗi bộ. Ở mỗi tầm, HiPINN trùng với nhánh nằm giữa trong '
          'ba nhánh, nên đường của nó gãy khúc ở chỗ hai nhánh cắt nhau. Hình chỉ vẽ hai nhánh có đặc trưng lịch sử; ở '
          'những đoạn HiPINN không trùng đường nào trong hai đường đó, nhánh nằm giữa là PINN-dự báo.')
    # ---------------- kiểm định và trung bình ngược chiều
    odd = []
    for b, sc, lab in E33_SET:
        for x, y in (('hipinn', 'gbdt_ls'), ('hipinn', 'ket_hop'), ('gbdt_ls', 'ket_hop')):
            for ds in DS:
                p, xw, ratio, r = E.row(b, sc, ds, x, y)
                if p < 0.05 and ((xw and ratio > 1) or (not xw and ratio < 1)):
                    odd.append((b, sc, ds, x, y, r))
    p6 = ''
    if odd:
        dss = sorted({o[2] for o in odd})
        tops = {o[5].top_cell for o in odd}
        p6 = (r'Có một điểm cần lưu ý khi đọc các kiểm định. '
              f'Ở {jv(dss)}, có {len(odd)} phép so mà kiểm định ghép cặp và MAE trung bình chỉ hai hướng khác nhau: phần '
              'lớn cell đi theo hướng của kiểm định, còn trung bình bị kéo bởi vài cell sai rất lớn; bỏ '
              + rng([o[5].flip_k for o in odd], 0) + ' cell như vậy thì trung bình cũng đổi chiều. ')
        if len(tops) == 1:
            t = tops.pop(); r = odd[0][5]
            p6 += (f'Cell gây chênh lệch lớn nhất là {tex_esc(t.split("/")[-1])} của {t.split("/")[0]}: MAE của '
                   f'PINN-dự báo {v(1e3 * r.top_pinn, 0)}, của HiGBDT {v(1e3 * r.top_gbdt_ls, 0)}, của HiPINN '
                   f'{v(1e3 * r.top_hipinn, 0)} ($\\times10^{{-3}}$).')
    # ---------------- RUL và khoảng
    rk = E.better('all', 'do', 'hipinn', 'ket_hop', rul=True)
    rr = [E.rv('all', 'do', ds, 'e_hipinn') / E.rv('all', 'do', ds, 'e_ket_hop') for ds in DS]
    rl = [E.rv('all', 'do', ds, 'e_hipinn') / E.rv('all', 'do', ds, 'e_gbdt_ls') for ds in DS]
    rlk = E.better('all', 'do', 'hipinn', 'gbdt_ls', rul=True)
    over = [ds for ds in DS if E.rv('all', 'do', ds, 's_gbdt_ls') > 0.8 * E.rv('all', 'do', ds, 'e_gbdt_ls')]
    cov_do = [E.val(b, sc, ds, 'cov_h') for b, sc in do for ds in DS]
    cov_r = [E.rv('all', 'do', ds, 'cov_h') for ds in DS]
    in20 = [E.rv('all', 'do', ds, 'in20_hipinn') for ds in DS]
    p7 = (r'Về RUL và khoảng dự báo, \Cref{tab:e33rul} cho MAE của RUL. Với mọi cell '
          f'train, RUL của HiPINN có MAE bằng {rng(rr)} lần PINN-GBDT ({sig_txt(*rk)}) và {rng(rl)} lần HiGBDT '
          f'({sig_txt(*rlk)}); tỉ lệ RUL dự báo nằm trong $\\pm 20\\,\\%$ RUL thực là {pct_rng(in20)}. ')
    if over:
        p7 += (f'HiGBDT dự báo RUL dài hơn thực tế trên {jv(over)} (sai số có dấu trung bình '
               + jv([f'{ds} +{v(E.rv("all", "do", ds, "s_gbdt_ls"), 0)}' for ds in over])
               + ' chu kỳ), cùng nguyên nhân với độ lệch dương ở phần cuối đời. ')
    p7 += (f'Khoảng SOH 90\\,\\% của HiPINN phủ {pct_rng(cov_do)} và khoảng RUL 90\\,\\% phủ {pct_rng(cov_r)} tuỳ bộ, '
           'tức có bộ chưa đạt độ phủ danh nghĩa; phân vị của khoảng chỉ tính trên vài cell hiệu chỉnh mỗi lượt.'
           if min(cov_do + cov_r) < 0.9 else
           f'Khoảng SOH 90\\,\\% của HiPINN phủ {pct_rng(cov_do)} và khoảng RUL 90\\,\\% phủ {pct_rng(cov_r)}.')
    # ---------------- chọn mô hình
    def verdict(b, sc, cand, get, rul=False):
        for a in cand:
            if all(sum(get(b, sc, ds, a) < get(b, sc, ds, c) for ds in DS) >= 3 and not E.better(b, sc, a, c, rul)[1]
                   for c in cand if c != a):
                return a
        return None
    cand = ['hipinn', 'gbdt_ls', 'ket_hop']
    soh = {(b, sc): verdict(b, sc, cand, E.val) for b, sc, _ in E33_SET}
    rget = lambda b, s, ds, m: E.rv(b, s, ds, f'e_{m}')
    rul_b = {b: verdict(b, sc, cand, rget, True) for b, sc, _ in E33_SET}
    rul = rul_b['all']
    name = lambda m: E33_TXT[m] if m else None
    lab = {(b, sc): l for b, sc, l in E33_SET}
    p8 = ('Để gợi ý một mô hình cho lần dùng sau, ba mô hình HiPINN, HiGBDT và PINN-GBDT được so theo cùng một quy tắc, '
          'áp riêng cho từng ngân sách nhãn: mô hình được chọn phải có MAE thấp hơn từng mô hình còn lại trên ít nhất ba '
          'bộ và không kém có ý nghĩa thống kê trên bộ nào. Quy tắc này và dạng trung vị của HiPINN đều được đặt sau khi '
          'đã có kết quả trên tập kiểm tra, nên lựa chọn dưới đây chỉ là gợi ý, chưa được kiểm chứng trên dữ liệu mới. ')
    if len(set(soh.values())) == 1 and None not in soh.values():
        p8 += f'Với SOH, quy tắc chọn {name(next(iter(soh.values())))} ở cả hai ngân sách. '
    else:
        for k, m in soh.items():
            p8 += (f'Với SOH và {lab[k]}, quy tắc chọn {name(m)}. ' if m else
                   f'Với SOH và {lab[k]}, không mô hình nào thoả quy tắc. ')
    for b, sc, l in E33_SET:
        m = rul_b[b]
        if m:
            p8 += f'Với RUL khi học trên {l if b != "all" else "mọi cell huấn luyện"}, quy tắc chọn {name(m)}. '
        else:
            nl = sum(E.rv(b, sc, ds, 'e_hipinn') < E.rv(b, sc, ds, 'e_ket_hop') for ds in DS)
            p8 += (f'Với RUL ở {l}, không mô hình nào thoả quy tắc: MAE RUL của HiPINN chỉ thấp hơn PINN-GBDT ở '
                   f'{nl}/4 bộ. ')
    worse_rul = [ds for ds in DS if all(E.rv(b, sc, ds, 'e_hipinn') > E.rv(b, sc, ds, 'e_ket_hop') for b, sc, _ in E33_SET)]
    for ds in worse_rul:
        _, _, ratio_k, rk = E.row('all', 'do', ds, 'hipinn', 'ket_hop', rul=True)
        better_k = int(rk.a_better if rk.a == 'hipinn' else rk.n - rk.a_better)
        big = [m for m in ('gbdt_ls', 'doi') if E.rv('all', 'do', ds, f'e_{m}') > 2 * E.rv('all', 'do', ds, 'e_ket_hop')]
        p8 += (f'Riêng trên {ds}, MAE RUL của HiPINN cao hơn PINN-GBDT ở cả hai ngân sách '
               f'({v(E.rv("all", "do", ds, "e_hipinn"), 0)} so với {v(E.rv("all", "do", ds, "e_ket_hop"), 0)} chu kỳ khi '
               f'học trên mọi cell huấn luyện, gấp {v(ratio_k, 2)} lần). Khác biệt này không có ý nghĩa thống kê vì HiPINN '
               f'vẫn tốt hơn ở {better_k}/{int(rk.n)} cell, còn MAE trung bình bị vài cell sai rất lớn kéo lên'
               + ((': ' + ' và '.join({'gbdt_ls': 'HiGBDT', 'doi': 'nhánh theo đời'}[m] for m in big)
                   + ' cùng dự báo RUL dài hơn thực tế ở các cell đó, nên trung vị đi theo chúng') if len(big) == 2 else '')
               + f'. Với các cell giống {ds}, mô hình không dùng lịch sử như PINN-GBDT cho RUL an toàn hơn.')
    figs = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_mae.pdf}' '\n'
            r'\caption[MAE của SOH dự báo theo mô hình và ngân sách nhãn]{MAE theo cell (trung bình qua 3 lần lặp, thang log) của các mô hình '
            r'ở hai ngân sách nhãn, 30\,\% nhãn hoặc mọi cell train; lịch sử SOH tới gốc là dung lượng đo.}'
            r'\label{fig:e33mae}'
            '\n' r'\end{figure}')
    fig_tail = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_tail.pdf}'
                '\n' r'\caption[Sai số dự báo theo phần đời của cell]{MAE ($\times10^{-3}$) của SOH dự báo theo phần đời '
                r'của chính cell (chu kỳ chia chu kỳ cuối ghi được), mọi cell train, lịch sử đo, trung bình theo cell. '
                r'Phần đời chỉ dùng để chia nhóm khi chấm, mô hình không biết nó.}\label{fig:e33tail}' '\n'
                r'\end{figure}')
    TP = E.TP
    fig_traj = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_traj.pdf}'
                '\n' r'\caption[Dự báo của HiPINN so với SOH đo được]{Dự báo của HiPINN, HiGBDT và nhánh theo đời '
                r'so với SOH đo được cho một cell test mỗi bộ: mọi cell train, lịch sử đo, lần lặp 0, gốc ở $0{,}4L$; trong '
                r'các cell còn ít nhất $' + v(float(TP.min_h.iloc[0]), 2) + r'L$ chu kỳ đo sau gốc, chọn cell có sai số '
                r'của HiPINN ở trung vị. Khoảng 90\,\% tính bằng phương pháp conformal. PINN-dự báo, nhánh thứ ba của HiPINN, không '
                r'vẽ để hình dễ đọc.' + xjtu_band_note(str(TP[TP.ds == 'XJTU'].cid.iloc[0])) + r'}\label{fig:e33traj}'
                '\n' r'\end{figure}')
    body = ('\\FloatBarrier\n\\subsection{Dự báo SOH, dung lượng và RUL: HiGBDT và HiPINN}\\label{sec:e33res}\n\n' + p0 + '\n\n'
            + e33_table(E) + '\n\n' + p1 + '\n\n' + p2 + '\n\n' + p3 + '\n\n' + p3b + '\n\n' + e33_val_table(E)
            + '\n\n' + figs + '\n\n'
            + p4 + '\n\n'
            + fig_tail + '\n\n' + p5 + '\n\n' + fig_traj + '\n\n' + (p6 + '\n\n' if p6 else '') + p7 + '\n\n'
            + e33_rul_table(E) + '\n\n' + p8 + '\n')
    write('doan_e33.tex', body)
    # ---------------- HiGBDT cho bài toán cập nhật (Chương 5, mục ngân sách đo và cập nhật)
    D34 = e34_load()
    a_text, a_tab = e34_anchor(D34) if D34 is not None else (r'(Chưa có kết quả E34: chạy E34 rồi dựng lại báo cáo.)', '')
    write('doan_cap_nhat_hi.tex', r'\label{sec:anchor}' + a_text + '\n\n' + a_tab)
    return dict(soh=soh, rul=rul, rul_b=rul_b, worse_rul=worse_rul, name=name, r_hk=r_hk, r_hl=r_hl, r_g=r_g, r_p=r_p, best_n=best_n,
                n_all=n_all, n_hl=n_hl, cov_h=cov_do, cov_rul=cov_r, in20=in20, worst=worst,
                tail_ls=tail(worst, 'gbdt_ls'), tail_h=tail(worst, 'hipinn'), mit_k=tk, mit_h=th,
                up_l=up_l, dn_l=dn_l, up_k=up_k, dn_k=dn_k, n_do=len(do) * len(DS), D34=D34)


def demo_branch_txt(Dm):
    """So HiPINN với nhánh tốt nhất của chính cell minh hoạ ở mỗi bộ (đúng với số trong demo/hipinn_bon_bo.csv)."""
    lab = {'gbdt_ls': 'HiGBDT', 'pinn': 'PINN-dự báo', 'doi': 'nhánh theo đời'}
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    best = {r.bo: min(lab, key=lambda k: getattr(r, f'mae_{k}')) for r in Dm.itertuples()}
    beat = [r.bo for r in Dm.itertuples() if r.mae <= getattr(r, f'mae_{best[r.bo]}')]
    rest = [r for r in Dm.itertuples() if r.bo not in beat]
    t = ('ở ' + jv(beat) + ', HiPINN không kém nhánh tốt nhất của chính cell đó' if beat else '')
    if rest:
        t += ('; ' if t else '') + '; '.join(f'ở {r.bo}, nhánh tốt nhất là {lab[best[r.bo]]} '
                                             f'({v(1e3 * getattr(r, "mae_" + best[r.bo]), 1)})' for r in rest)
    return t + '.'


def demo_section():
    """Chương 4: công cụ dự báo cho một cell (tools/demo.py), HiPINN cho một cell test của mỗi bộ. Số và hình lấy từ
    demo/hipinn_bon_bo.* (lệnh `python run.py demo tat-ca`); bản sao ở generated/data/ và graphics/."""
    p = keep_copy('demo/hipinn_bon_bo.csv', 'demo_hipinn.csv')
    if os.path.exists('demo/hipinn_bon_bo.pdf'):
        shutil.copy('demo/hipinn_bon_bo.pdf', os.path.join(FIG, 'fig_demo_hipinn.pdf'))
    text = (r'\section{Công cụ dự báo cho một cell}\label{sec:congcu}' '\n\n'
            r'Công cụ dòng lệnh của đồ án gói PINN-semi, HiPINN và GBDT-cập nhật để dùng cho một cell bất kỳ, chỉ cần tệp đặc trưng sạc của cell '
            r'đó. Lệnh thứ nhất học PINN-semi ở 30\,\% nhãn cho một bộ dữ liệu, lệnh thứ hai ước lượng SOH ở từng chu kỳ, '
            r'lệnh thứ ba dự báo SOH, dung lượng và RUL từ một chu kỳ bất kỳ bằng HiPINN (\Cref{sec:e33pp}). Lần đầu dự báo '
            r'cho một bộ, HiPINN được học trên các cell có nhãn của cùng fold với PINN-semi của công cụ, chọn cấu hình '
            r'từng nhánh trên cell validation rồi lưu lại; cell được dự báo '
            r'không tham gia khi học. Lịch sử SOH tới gốc là dung lượng đo của chính cell; cell chưa được đo dung '
            r'lượng thì công cụ chỉ ước lượng SOH hiện tại (lệnh thứ hai), không dự báo.')
    up = keep_copy('demo/2C_battery-8_capnhat.csv', 'demo_capnhat.csv')
    if up is not None:                             # lệnh cập nhật: ví dụ cell 2C_battery-8 của XJTU, lịch đo mỗi 0,1L
        U = pd.read_csv(up); m = U.soh_do.notna(); ev = U.soh_cap_nhat.notna() & ~m
        last = np.maximum.accumulate(np.where(m, np.arange(len(U)), -1)); prev = np.r_[-1, last[:-1]]
        hold = np.where(prev >= 0, U.soh_do.values[np.maximum(prev, 0)], np.nan)
        e_up = float((U.soh_cap_nhat - U.soh_thuc).abs()[ev].mean()); e_hold = float(np.abs(hold - U.soh_thuc)[ev].mean())
        text += (r' Lệnh thứ tư cập nhật SOH khi cell được đo dung lượng ở vài chu kỳ. Ở mỗi chu kỳ không đo, GBDT-cập '
                 r'nhật ước lượng SOH từ lần đo gần nhất trước nó và đặc trưng sạc, đúng mô hình mà \Cref{sec:anchor} giữ '
                 r'cho lịch đo thưa; chỉ khi các lần đo dày tới mức coi như đo mọi chu kỳ trước, công cụ mới dùng HiGBDT-R. '
                 r'Với cell 2C\_battery-8 '
                 f'của XJTU đo {int(m.sum())} lần trên {len(U)} chu kỳ theo lịch mỗi $0{{,}}1L$, MAE ở các chu kỳ không đo là '
                 f'{v(1e3 * e_up, 1)} ($\\times10^{{-3}}$), so với {v(1e3 * e_hold, 1)} khi giữ nguyên lần đo gần nhất.')
    if p is None or not os.path.exists(os.path.join(FIG, 'fig_demo_hipinn.pdf')):
        write('doan_cong_cu.tex', text)
        return
    Dm = pd.read_csv(p)
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    name = lambda cid: tex_esc(cid.split('/')[-1])
    text += ('\n\n' r'Lệnh thứ năm áp dụng HiPINN cho cả bốn bộ: với mỗi bộ, lấy cell test có số chu kỳ ở trung vị '
             r'và dự báo từ 40\,\% số chu kỳ của nó (\Cref{fig:demo4}). MAE trên phần đời còn lại là '
             + jv([f'{v(1e3 * r.mae, 1)} ở {r.bo} (cell {name(r.cell)})' for r in Dm.itertuples()])
             + r' ($\times10^{-3}$); ' + demo_branch_txt(Dm) + r' Mô hình của công cụ học trên 30\,\% cell có nhãn, ít '
               r'hơn mức mọi cell train ở \Cref{sec:e33res}, và mỗi bộ chỉ có một cell, nên đây là minh hoạ chứ không phải số '
               r'đánh giá.' '\n\n'
             r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_demo_hipinn.pdf}' '\n'
             r'\caption[HiPINN cho một cell test của mỗi bộ dữ liệu]{HiPINN áp dụng cho bốn bộ dữ liệu qua công cụ dòng '
             r'lệnh: với mỗi bộ, cell test có số chu kỳ ở trung vị, dự báo từ 40\,\% số chu kỳ của cell, lịch sử là dung '
             r'lượng đo; mô hình học trên 30\,\% cell có nhãn của fold dùng cho công cụ. PINN-dự báo, nhánh thứ ba của '
             r'HiPINN, không vẽ để hình dễ đọc.}\label{fig:demo4}' '\n'
             r'\end{figure}')
    write('doan_cong_cu.tex', text)


def pred_traj_text():
    """Vẽ lại hình quỹ đạo dự đoán (MLP, PINN-semi, PINN-lai; analysis/figures_report.py) khi có đủ lượt chạy, rồi viết
    câu về sai số của ba mô hình trên chính các cell trong hình; bản sao bảng ở generated/data/pred_traj_cells.csv."""
    try:
        import figures_report as FR
        FR.FIGS['pred_traj']()
        shutil.copy('results/fig_pred_traj.pdf', os.path.join(FIG, 'fig_pred_traj.pdf'))
    except Exception as e:                                       # thiếu lượt chạy: dùng hình và bảng đã lưu
        print('pred_traj: dùng hình đã lưu', type(e).__name__, e)
    p = keep_copy('results/pred_traj_cells.csv', 'pred_traj_cells.csv')
    if p is None:
        return ''
    C = pd.read_csv(p)
    m = C.groupby(['model', 'frac']).mae.mean()
    best = C.loc[C.groupby(['ds', 'frac']).mae.idxmin()].model
    n_lai, n = int((best == 'pinn_lai').sum()), len(best)
    smooth = C.loc[C.groupby(['ds', 'frac']).tv.idxmin()].model
    n_sm = int((smooth == 'pinn_lai').sum())
    tv = C.groupby('model').tv.mean()
    fr = sorted(C.frac.unique())
    jv = lambda xs: ', '.join(xs[:-1]) + ' và ' + xs[-1]
    vals = lambda mod: jv([v(1e3 * m[(mod, f)]) for f in fr])
    words = {1: 'một', 2: 'hai', 3: 'ba', 4: 'bốn'}
    return (f'Hình cũng vẽ PINN-lai, mô hình cuối của bài toán ước lượng. Trên {words.get(C.ds.nunique(), C.ds.nunique())} cell của hình, MAE trung '
            f'bình của PINN-lai ở {jv([str(int(round(100 * f))) for f in fr])}\\,\\% nhãn là {vals("pinn_lai")} '
            f'($\\times10^{{-3}}$), so với {vals("pinn_semi")} của PINN-semi và {vals("mlp")} của MLP. PINN-lai có MAE '
            f'thấp nhất trong ba mô hình ở {n_lai}/{n} ô của hình. Đường của PINN-lai dao động ít nhất ở {n_sm}/{n} ô: '
            f'chênh lệch trung bình giữa hai chu kỳ liền nhau là {v(1e3 * tv["pinn_lai"])}, so với '
            f'{v(1e3 * tv["pinn_semi"])} của PINN-semi và {v(1e3 * tv["mlp"])} của MLP ($\\times10^{{-3}}$), nhờ trung '
            'bình ba mạng với GBDT và bộ lọc xu thế.')


def figures():
    import diagrams
    os.makedirs(FIG, exist_ok=True)
    # kiến trúc PINN-semi, PINN-lai, tổng quan HiPINN và quy trình thực nghiệm: hình do người dùng vẽ, lưu sẵn ở graphics/
    diagrams.hipinn_arch(os.path.join(FIG, 'fig_hipinn_arch'))
    for src, dst in [('fig_e28_budget', 'fig_e28_budget')]:
        shutil.copy(os.path.join(OLD_FIG, src + '.pdf'), os.path.join(FIG, dst + '.pdf'))


def compile_pdf():
    """pdfLaTeX + BibTeX như mẫu hcmut-thesis (latexmk -pdf), rồi chép main.pdf thành bao-cao-do-an.pdf."""
    if not shutil.which('latexmk') or not shutil.which('pdflatex'):
        print('không có latexmk/pdflatex: bỏ qua bước biên dịch')
        return
    r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=REP,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')   # log T5 không phải UTF-8
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit('biên dịch lỗi: xem report/main.log')
    shutil.copy(os.path.join(REP, 'main.pdf'), os.path.join(REP, 'bao-cao-do-an.pdf'))
    print('đã dựng report/bao-cao-do-an.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pdf', action='store_true')
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    F20, K32 = e20f_load(), e32_load()
    assert F20 is not None and K32 is not None, 'thiếu analysis/analyze_e20_folds.py hoặc analysis/analyze_e32.py'
    thesis_tables(F20, K32)
    REL = rel_dev()
    write('doan_quy_dao.tex', pred_traj_text())
    write('doan_pinn_lai.tex', pinn_lai_text(F20, REL['geo'] if REL else None))
    k_txt, KS = kalman_text(K32, F20)
    write('doan_kalman.tex', k_txt)
    write('doan_tau.tex', tau_text())
    write('doan_chi_phi.tex', cost_text())
    leak_tab, leak_text, L_leak = leakage()
    write('doan_ro_ri.tex', leak_text + '\n\n' + leak_tab)
    write('doan_loc.tex', filter_text())
    S, T, L = bench_load()
    B = bench_section(S, T, L)
    fig_benchmark(S)
    F = features_load()
    n_feat_jobs = n_feat_fits = 0
    if F is not None:
        rel, abl, drop, concl = features_parts(F, L_leak, REL, F20)
        jobs = F['prune'][['dataset', 'frac', 'repeat', 'fold']].drop_duplicates()
        n_feat_jobs = len(jobs)
        n_feat_fits = n_feat_jobs * 9 + int((jobs.dataset == 'TJU').sum())     # 9 tập đầu vào, thêm 1/T ở TJU
    else:
        rel = abl = drop = concl = r'(Chưa có kết quả phân tích đặc trưng: chạy \texttt{python analysis/features.py}.)'
    write('doan_lien_quan.tex', rel); write('doan_bo_nhom.tex', abl); write('doan_da_bo.tex', drop)
    write('doan_ket_luan_dt.tex', concl)
    n_rerun = int(B['runs'] * B['n_groups'])
    n_total = experiments_table(n_rerun, n_feat_jobs, n_feat_fits)
    demo_section()
    D33 = e33_load()
    E33 = None
    if D33 is not None:
        E33 = e33_section(D33)
    else:
        write('doan_e33.tex', r'(Chưa có kết quả E33 với HiPINN: chạy E33 rồi dựng lại báo cáo.)')
        write('doan_cap_nhat_hi.tex', '')
    C6 = compare_chapter(S, T, B, L_leak, E33, physics_params())
    reproducibility(B)
    tomtat = conclusion(B, L_leak, C6, F is not None, n_total, E33, F20, KS)
    write('so_lieu.tex', r'\newcommand{\TomTat}{' + tomtat + '}')
    figures()
    if not a.no_pdf:
        compile_pdf()


if __name__ == '__main__':
    main()
