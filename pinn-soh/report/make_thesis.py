"""
Dựng báo cáo dạng đồ án (LaTeX → PDF) từ results/*.csv.

    python report/make_thesis.py         # -> do-an/main.tex, do-an/hinh/, do-an/bao-cao-do-an.pdf

Chạy sau `python run.py analyze` (hoặc gọi thẳng `python run.py thesis`).
Mọi con số trong văn bản đọc từ results/, không gõ tay.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import math, os, shutil, subprocess
import numpy as np
import pandas as pd

R, OUT = 'results', 'do-an'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}
LBL = {'mlp': 'MLP', 'ridge': 'Ridge', 'gbdt': 'GBDT', 'gpr': 'GPR', 'pinn_sup': 'PINN-sup',
       'pinn_bb': 'PINN hộp đen', 'pinn_semi': 'PINN-semi', 'mono_only': 'chỉ đơn điệu', 'euler': 'PINN-semi (Euler)'}


# ───────────────────────────────────────────────────────────── định dạng số kiểu Việt
def f(x, n=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return '--'
    if isinstance(x, float) and math.isinf(x):
        return '$>70$'
    s = f'{x:.{n}f}'.replace('.', '{,}')
    return '−' + s[1:] if s.startswith('-') else s                 # dấu trừ thật, không phải gạch nối


def pc(x, n=0):
    return f(100 * x, n) + r'\,\%'


def m3(x, n=2):
    """MAE theo đơn vị 10^-3 SOH."""
    return f(1e3 * x, n)


def pv(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '--'
    return r'$<10^{-4}$' if p < 1e-4 else f(p, 4)


def num(x):
    return f'{int(x):,}'.replace(',', '.')


def load():
    d = {}
    for k in ['thesis_split_design', 'thesis_strat', 'thesis_repeats', 'thesis_curve', 'thesis_compare',
              'thesis_equivalence', 'thesis_leakage', 'thesis_transfer', 'E13_tests', 'E13_secondary',
              'pred_metrics', 'verify_pipeline', 'E13_physics_params', 'thesis_rowbound', 'thesis_leakage_raw', 'thesis_rowsim',
              'metrics_summary', 'metrics_tests', 'metrics_transfer', 'physics_summary', 'physics_rate_soh',
              'physics_cells', 'physics_check', 'e20_summary', 'e20_tests', 'e20_conformal', 'e20_dev',
              'e20_error_decomp', 'e20_check', 'e20_biascorr', 'e21_base', 'e21_chain', 'e21_chain_tests',
              'e22_horizon', 'e22_traj', 'e22_tests', 'e22_rul', 'e22_rul_tests', 'e22_lambda', 'e22_setting',
              'e23_tests', 'e23_horizon', 'e23_rul', 'e23_rul_tests', 'e23_cover', 'e23_decomp', 'e23_periodic_xjtu',
              'e24_cv', 'e24_tests', 'e24_rest', 'e24_samba']:
        p = f'{R}/{k}.csv'
        if os.path.exists(p):
            d[k] = pd.read_csv(p, header=[0, 1], index_col=[0, 1]) if k == 'E13_physics_params' else pd.read_csv(p)
    return d


def dataset_stats():
    from pinnsoh.data import load_dataset
    rows = []
    for ds in DS:
        cs = load_dataset('PINN4SOH/data', ds, clean='causal')
        life = np.array([c.cycle.max() + 1 for c in cs])
        rows.append(dict(bo=ds, N=len(cs), n=sum(len(c.soh) for c in cs), batch=len({c.batch for c in cs}),
                         lmin=life.min(), lmed=int(np.median(life)), lmax=life.max(),
                         s0a=min(c.soh[0] for c in cs), s0b=max(c.soh[0] for c in cs),
                         s1a=min(c.soh[-1] for c in cs), s1b=max(c.soh[-1] for c in cs)))
    return pd.DataFrame(rows)


def run_counts():
    import glob
    c = dict(E13=len(glob.glob('runs/E13/*.json')), E16=len(glob.glob('runs/E16/*.json')),
             E17=len(glob.glob('runs/E17/*.json')), E15=len(glob.glob('runs/E15/*.json')),
             E14=len(glob.glob('runs/E14/samenorm_*.json')),
             E20=len({os.path.basename(p) for p in glob.glob('runs/E20/*.json') + glob.glob('runs/E20dev/*.json')}),
             E21=len(glob.glob('runs/E21/*.json')), E22=len(glob.glob('runs/E22/*.npz')),
             E23=len(glob.glob('runs/E23/*.npz')), E24=len(glob.glob('runs/E24/*.json')))
    c['tong'] = sum(c.values())
    return c


# ───────────────────────────────────────────────────────────── khung tài liệu
PREAMBLE = r"""\documentclass[12pt,a4paper,oneside]{report}
\usepackage{amsmath,amsthm}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{DejaVu Sans Mono}[Scale=0.82]
\setmathfont{TeX Gyre Termes Math}
\usepackage{polyglossia}
\setdefaultlanguage{vietnamese}
\usepackage[left=3cm,right=2cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{graphicx,booktabs,longtable,array,tabularx,multirow,float,enumitem}
\usepackage[section]{placeins}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{5}
\usepackage[table]{xcolor}
\usepackage{adjustbox}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\usepackage{titlesec}
\titleformat{\chapter}[display]{\normalfont\bfseries\Large\centering}{\MakeUppercase{\chaptername}~\thechapter}{6pt}{\LARGE}
\titlespacing*{\chapter}{0pt}{-10pt}{18pt}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyhead[R]{\small\nouppercase{\leftmark}}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\headheight}{15pt}
\usepackage[colorlinks=true,linkcolor=blue!40!black,citecolor=green!35!black,urlcolor=blue!40!black]{hyperref}
\linespread{1.25}
\setlength{\parskip}{4pt}
\theoremstyle{plain}
\newtheorem{menhde}{Mệnh đề}[chapter]
\newtheorem{bode}[menhde]{Bổ đề}
\theoremstyle{definition}
\newtheorem{dinhnghia}[menhde]{Định nghĩa}
\newtheorem{nhanxet}[menhde]{Nhận xét}
\renewcommand{\proofname}{Chứng minh}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\setlist{itemsep=2pt,topsep=4pt}
\emergencystretch=1.5em
\setcounter{tocdepth}{1}
\begin{document}
"""

TITLE = r"""\begin{titlepage}
\centering
{\large ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\\[2pt]
\textbf{TRƯỜNG ĐẠI HỌC BÁCH KHOA}\\[2pt]
\textbf{KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH}}\\[1.2cm]
\includegraphics[width=3.2cm]{hinh/bk.png}\\[1.4cm]
{\Large\textbf{ĐỒ ÁN CHUYÊN NGÀNH}}\\[1.2cm]
{\LARGE\bfseries Dự đoán thông số sức khoẻ của pin\\[4pt] bằng mô hình học sâu được hỗ trợ\\[4pt] từ thông tin hoá lý của hệ pin}\\[1.5cm]
\begin{tabular}{rl}
Sinh viên thực hiện: & Huỳnh Quốc Thắng\\
Nhóm: & 15\\
Học kỳ: & HK261
\end{tabular}\\[\fill]
{\large TP. Hồ Chí Minh, tháng 10/2026}
\end{titlepage}
"""


def _cap(caption):
    """\\caption[câu đầu]{đầy đủ}: danh mục hình và bảng chỉ in câu đầu của chú thích, dưới hình vẫn đủ."""
    i = caption.find('. ')
    short = caption[:i] if 0 < i < len(caption) - 2 else ''
    ok = short and short.count('$') % 2 == 0 and short.count('{') == short.count('}') and '[' not in short
    return rf'\caption[{short}]{{{caption}}}' if ok else rf'\caption{{{caption}}}'


def fig(name, caption, label, width=r'\textwidth'):
    return (r'\begin{figure}[!htbp]\centering' + '\n' + rf'\includegraphics[width={width}]{{hinh/{name}.pdf}}' + '\n'
            + _cap(caption) + rf'\label{{{label}}}' + '\n' + r'\end{figure}' + '\n')


def table(header, rows, spec, caption, label, size=r'\small'):
    head = ' & '.join(header) + r' \\'
    body = '\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows)
    return (r'\begin{table}[!htbp]\centering' + size + '\n' + _cap(caption) + rf'\label{{{label}}}' + '\n'
            + r'\begin{adjustbox}{max width=\textwidth}' + rf'\begin{{tabular}}{{{spec}}}\toprule' + '\n' + head
            + r'\midrule' + '\n' + body + '\n' + r'\bottomrule\end{tabular}\end{adjustbox}\end{table}' + '\n')


def T(text, **kw):
    """Thay <<khoá>> bằng giá trị — tránh phải nhân đôi dấu ngoặc nhọn của LaTeX trong f-string."""
    for k, v in kw.items():
        text = text.replace(f'<<{k}>>', str(v))
    assert '<<' not in text, text[text.index('<<'):text.index('<<') + 40]
    return text


# ═════════════════════════════════════════════════════════════ CHƯƠNG 1
def ch1(d, ctx):
    return T(r"""
\chapter{Giới thiệu}\label{ch:gioithieu}

\section{Bối cảnh}

Pin lithium-ion là nguồn năng lượng chính của xe điện, thiết bị di động và hệ thống lưu trữ điện. Trong
quá trình sử dụng, dung lượng của pin giảm dần do các phản ứng phụ bên trong cell: lớp SEI dày lên, lithium
bị mất khỏi chu trình, vật liệu điện cực bị suy thoái. Trạng thái sức khoẻ (State of Health, SOH) đo mức
suy giảm này và thường được định nghĩa bằng tỉ số giữa dung lượng hiện tại và dung lượng danh định:
\begin{equation}
\mathrm{SOH}_n = \frac{Q_n}{Q_\text{danh định}},
\end{equation}
trong đó $Q_n$ là dung lượng đo được ở chu kỳ thứ $n$. Hệ thống quản lý pin (BMS) cần SOH để ước tính quãng
đường còn lại, điều chỉnh chiến lược sạc và cảnh báo thời điểm thay pin; với pin xe điện, mốc cuối đời
thường được chọn ở SOH khoảng 0,8.

Khó khăn thực tế nằm ở nhãn. Muốn biết chính xác $Q_n$ phải xả đầy cell theo một quy trình chuẩn, mất nhiều
giờ và phải dừng vận hành. Ngược lại, dữ liệu điện áp và dòng điện trong lúc sạc được BMS ghi lại ở mọi chu
kỳ, của mọi cell, gần như không tốn chi phí. Bài toán vì vậy có dạng điển hình của học ít nhãn: rất nhiều dữ
liệu đầu vào nhưng chỉ một phần nhỏ có nhãn.

\section{Phát biểu bài toán}

Xét tập cell $\mathcal{C}$. Mỗi cell $c$ có chuỗi chu kỳ $n = 0,1,\dots,L_c$; ở mỗi chu kỳ, từ đoạn sạc
CC--CV ta trích được vector đặc trưng $\mathbf{x}_{c,n} \in \R^{16}$. Nhãn $y_{c,n} = \mathrm{SOH}_{c,n}$
chỉ được biết trên một tập con các cell huấn luyện $\mathcal{C}_L \subset \mathcal{C}_\text{train}$; các cell
còn lại $\mathcal{C}_U = \mathcal{C}_\text{train} \setminus \mathcal{C}_L$ chỉ có đặc trưng sạc. Mục tiêu là
học hàm
\begin{equation}
\hat y = F(\mathbf{x}_{c,n}, n)
\end{equation}
sao cho sai số trên các cell \emph{chưa từng xuất hiện} khi huấn luyện là nhỏ nhất. Mô hình chỉ dùng dữ liệu
của chu kỳ hiện tại, nên có thể chạy trực tuyến trong BMS.

Bài toán thứ hai là dự báo: đứng ở chu kỳ $k$ của một cell, chỉ dùng dữ liệu tới chu kỳ $k$, dự báo SOH và dung
lượng ở các chu kỳ $k + h$ sắp tới, cùng số chu kỳ còn lại trước khi SOH xuống dưới một ngưỡng cuối đời $\tau$
(remaining useful life, RUL).

\section{Câu hỏi nghiên cứu}

\begin{enumerate}[label=\textbf{CH\arabic*.}]
\item Khi cùng một lượng cell có nhãn, mô hình PINN bán giám sát có sai số thấp hơn mô hình học có nhãn
      thuần tuý không?
\item Lợi ích đó thay đổi thế nào khi lượng nhãn đi từ rất ít (10\,\% số cell) tới nhiều (70\,\%)? Mô hình
      học có nhãn cần bao nhiêu nhãn để đạt sai số của PINN?
\item So với các mô hình học có nhãn mạnh khác (hồi quy tuyến tính, hồi quy Gauss, gradient boosting),
      PINN đứng ở đâu?
\item Lợi ích đến từ đâu: từ vai trò chuẩn hoá (regularization) của vật lý, từ dữ liệu không nhãn, từ
      phương trình động học hay chỉ từ ràng buộc đơn điệu?
\item Mô hình học trên một bộ dữ liệu dùng được cho bộ khác tới đâu, và thích nghi không nhãn cùng học lặp
      bằng nhãn giả giúp được bao nhiêu?
\item Có thể giảm sai số thêm mà không dùng thêm nhãn nào không, và mô hình có cho biết dự đoán của nó đáng tin
      tới đâu không?
\item Từ SOH ước lượng được, có dự báo được SOH, dung lượng ở các chu kỳ tương lai và số chu kỳ còn lại tới ngưỡng
      cuối đời không, kể cả khi dung lượng hồi phục sau khoảng nghỉ, và dạng phương trình động học giúp gì cho dự báo?
\end{enumerate}

Mọi câu hỏi trên chỉ có ý nghĩa nếu giao thức đánh giá không rò rỉ thông tin. Vì vậy một phần lớn của đồ án
dành cho việc thiết kế, lý giải và chứng minh giao thức chia dữ liệu (Chương~\ref{ch:thietke}).

\section{Đóng góp}

\begin{enumerate}
\item Một mô hình PINN bán giám sát cho bài toán ước lượng SOH: mạng nghiệm kết hợp động học suy giảm dạng
      xám (tốc độ suy giảm luôn không âm, có hệ số đầu gối và hệ số Arrhenius), cùng ba hàm mất mát vật lý
      không dùng nhãn nên áp được lên cell chưa đo dung lượng.
\item Một giao thức đánh giá không rò rỉ: chia theo cell, kiểm định chéo <<K>> fold phân tầng theo chế độ thí
      nghiệm, lặp <<R>> lần, với lý do chọn từng con số và chứng minh các tính chất của cách chia.
\item Kiểm toán rò rỉ định lượng: đo mức sai số bị đánh giá thấp giả tạo khi chia theo chu kỳ, khi chuẩn hoá
      theo cả vòng đời cell, khi lọc ngoại lai bằng dữ liệu tương lai.
\item So sánh hệ thống PINN với sáu mô hình đối chứng ở bốn mức nhãn trên bốn bộ dữ liệu công khai
      (<<NCELL>> cell), tổng cộng <<NRUN>> lượt huấn luyện, có kiểm định thống kê và hiệu chỉnh đa so sánh.
\item Hai giao thức huấn luyện trên một bộ dữ liệu và kiểm tra trên ba bộ còn lại (70/30 và 70/15/15), cùng một
      quy trình thích nghi không nhãn và tự huấn luyện nhiều vòng có ràng buộc vật lý.
\item Mô hình cải tiến PINN-lai: nhóm ba mạng PINN-semi kết hợp với GBDT và một bộ lọc quỹ đạo nhân quả mang giả
      định SOH không tăng vào bước suy luận, được chọn chỉ trên tập validation theo đề cương ấn định trước, kèm
      khoảng dự đoán phù hợp 90\,\%.
\item Mở rộng sang dự báo SOH tương lai, dung lượng còn lại và RUL: mạng PINN-dự báo dùng nghiệm đóng của động học
      xám làm lớp đầu ra (luôn đơn điệu, mô tả được cả điểm đầu gối lẫn suy giảm chậm dần, RUL có dạng đóng), kết hợp
      với GBDT, đánh giá với lịch sử SOH đo và lịch sử SOH do mô hình ước lượng; bản cải tiến tách phần hồi phục và
      dao động đều ra khỏi xu thế suy giảm.
\item Áp dụng toàn bộ quy trình cho bộ NASA PCoE (trích lại 16 đặc trưng sạc từ dữ liệu gốc), kiểm chứng ý tưởng dùng
      thời gian nghỉ của SambaMixer và so trên đúng cách chia của công trình đó.
\end{enumerate}

\section{Bố cục}

Chương~\ref{ch:lythuyet} trình bày cơ sở lý thuyết và các công trình liên quan. Chương~\ref{ch:dulieu} mô tả
bốn bộ dữ liệu và cách tiền xử lý. Chương~\ref{ch:phuongphap} trình bày mô hình, các hàm mất mát và các mô
hình đối chứng. Chương~\ref{ch:thietke} là thiết kế thí nghiệm: cách chọn số chia, cách chống rò rỉ khi đánh
giá trong cùng bộ và khi chuyển miền, cùng các phương pháp thống kê. Chương~\ref{ch:ketqua} trình bày và
thảo luận kết quả. Chương~\ref{ch:caitien} phân tích sai số còn lại, thử sáu hướng cải tiến và đánh giá mô hình
PINN-lai cùng khoảng dự đoán. Chương~\ref{ch:ketluan} kết luận và nêu hướng phát triển.
""", K=5, R=5, NCELL=num(ctx['ncell']), NRUN=num(ctx['runs']['tong']))


# ═════════════════════════════════════════════════════════════ CHƯƠNG 2
def ch2(d, ctx):
    return r"""
\chapter{Cơ sở lý thuyết và các công trình liên quan}\label{ch:lythuyet}

\section{Suy giảm dung lượng của pin lithium-ion}

Dung lượng của cell lithium-ion giảm theo hai nhóm cơ chế chính~\cite{birkl2017}: mất lithium khả dụng (loss
of lithium inventory, LLI), chủ yếu do lớp SEI phát triển trên bề mặt anode, và mất vật liệu hoạt động (loss
of active material, LAM) do nứt hạt, bong tách hoặc hoà tan kim loại chuyển tiếp. Trong giai đoạn đầu, dung
lượng thường giảm chậm và gần tuyến tính; ở giai đoạn sau nhiều cell đi vào vùng “đầu gối” (knee), nơi
tốc độ suy giảm tăng nhanh~\cite{attia2022knee}. Các phản ứng phụ phụ thuộc nhiệt độ theo quy luật dạng
Arrhenius: nhiệt độ cao làm SEI phát triển nhanh hơn~\cite{waldmann2014}. Hai quan sát này là cơ sở cho
phần động học xám ở Chương~\ref{ch:phuongphap}: tốc độ suy giảm luôn không âm, tăng khi SOH giảm và tăng theo
nhiệt độ.

\paragraph{Các quy luật thực nghiệm.} Trước khi có học máy, tuổi thọ pin thường được mô tả bằng các hàm đóng
khớp trên dữ liệu thử nghiệm. Khi SEI dày lên và quá trình bị giới hạn bởi khuếch tán dung môi qua lớp SEI,
lượng lithium mất đi tỉ lệ với căn bậc hai của thời gian~\cite{broussely2001,pinson2013}. Wang và cộng
sự~\cite{wang2011lfp} khớp mất mát dung lượng của cell graphite/LFP bằng một hàm luỹ thừa của lượng điện tích
trao đổi nhân với thừa số Arrhenius, với số mũ khoảng 0,5. Schmalstieg và cộng sự~\cite{schmalstieg2014} tách lão
hoá theo lịch và theo chu kỳ của cell NMC 18650, mỗi phần là một hàm luỹ thừa có hệ số phụ thuộc nhiệt độ, điện
áp và độ sâu xả. Các mô hình này gọn và dễ diễn giải, nhưng mỗi hệ hoá học và mỗi chế độ vận hành cần một bộ
hệ số riêng, và chúng không mô tả được đoạn đầu gối. Fermín-Cueto và cộng sự~\cite{fermin2020} định nghĩa điểm
đầu gối và điểm bắt đầu đầu gối trên đường suy giảm rồi dùng học máy dự đoán hai điểm này từ các chu kỳ đầu.
Phương trình xám của đồ án đi theo cùng tinh thần: giữ dạng nhân của các quy luật trên (thừa số theo trạng
thái và thừa số Arrhenius), còn phần phụ thuộc vào từng cell để mạng học.

\section{Các hướng ước lượng SOH}

\paragraph{Mô hình cơ chế.} Mô hình điện hoá giả hai chiều (P2D) của Doyle, Fuller và Newman~\cite{doyle1993}
và các dạng rút gọn như mô hình hạt đơn mô tả chi tiết quá trình bên trong cell; mô hình mạch tương
đương~\cite{plett2015} đơn giản hơn và được dùng nhiều trong BMS. Ưu điểm là diễn giải được; nhược điểm là
cần nhiều tham số khó đo, việc hiệu chỉnh tham số theo lão hoá tốn kém.

\paragraph{Học máy dựa trên đặc trưng.} Severson và cộng sự~\cite{severson2019} cho thấy đặc trưng trích từ
đường cong xả của 100 chu kỳ đầu đủ để dự đoán tuổi thọ của 124 cell LFP bằng hồi quy tuyến tính có chuẩn
hoá. Richardson, Osborne và Howey~\cite{richardson2017} dùng hồi quy Gauss để dự báo SOH kèm độ bất định.
Roman và cộng sự~\cite{roman2021} xây dựng một quy trình học máy hoàn chỉnh cho ước lượng SOH. Zhu và cộng
sự~\cite{zhu2022} dùng đặc trưng của điện áp hồi phục sau sạc và công bố bộ dữ liệu TJU. Nhóm phương pháp này
cho kết quả tốt khi có đủ nhãn, nhưng phụ thuộc nhiều vào lượng nhãn và phân phối dữ liệu huấn luyện.

\paragraph{Học sâu.} Mạng hồi quy LSTM~\cite{zhang2018lstm}, mạng tích chập và các kiến trúc dựa trên chú ý
học trực tiếp từ chuỗi đo. Các mô hình này mạnh khi có nhiều dữ liệu có nhãn và dễ quá khớp khi ít nhãn. Gần đây,
SambaMixer~\cite{olalde2024samba} dùng mô hình không gian trạng thái Mamba trên chuỗi điện áp, dòng điện và nhiệt độ
của lần xả, kèm mã hoá vị trí theo thời gian lấy mẫu và theo khoảng thời gian giữa hai chu kỳ để học hiện tượng hồi
phục dung lượng sau khoảng nghỉ; trên bộ NASA, mô hình này đạt MAE khoảng 1\,\% SOH.

\paragraph{Học chuyển giao.} Ma và cộng sự~\cite{ma2022} dùng học chuyển giao sâu để cá nhân hoá dự báo
sức khoẻ cho từng cell và công bố bộ dữ liệu HUST. Hướng này giảm nhu cầu nhãn ở miền đích nhưng vẫn cần một
phần nhãn để thích nghi. Lu và cộng sự~\cite{lu2023} đi xa hơn: một nhóm mạng sâu kết hợp thích nghi miền ước
lượng SOH cho pin đích không có nhãn nào, thử trên 65 cell của 5 nhà sản xuất. Kết quả này cho thấy dữ liệu
không nhãn của chính pin đích mang nhiều thông tin, và cũng là lý do đồ án đưa cell không nhãn vào huấn luyện.

\paragraph{Dữ liệu thực tế và dữ liệu dạng bảng.} Sulzer và cộng sự~\cite{sulzer2021} chỉ ra rằng dữ liệu
ngoài hiện trường có điều kiện vận hành không kiểm soát, cảm biến kém chính xác và rất ít lần đo dung lượng
chuẩn; họ cho rằng kết hợp học máy với mô hình vật lý là hướng triển vọng vì giúp suy luận từ dữ liệu nhiễu và
ngoại suy sang điều kiện mới. Một điểm ít được nhắc trong các bài về PINN cho pin là đặc trưng trích từ đường
sạc là dữ liệu dạng bảng. Grinsztajn, Oyallon và Varoquaux~\cite{grinsztajn2022} so sánh trên 45 bộ dữ liệu
bảng cỡ vừa và thấy mô hình cây như gradient boosting vẫn vượt mạng nơ-ron; mạng nơ-ron khó học các hàm mục tiêu
không trơn và bị ảnh hưởng nhiều hơn bởi đặc trưng không mang thông tin. Vì vậy đồ án luôn đặt GBDT làm đối
chứng, và kết quả ở Chương~\ref{ch:ketqua} xác nhận đây là đối thủ khó nhất khi có đủ nhãn.

\paragraph{Mô hình lai và PINN.} Mạng nơ-ron có thông tin vật lý (PINN) của Raissi, Perdikaris và
Karniadakis~\cite{raissi2019} đưa phần dư của phương trình vi phân vào hàm mất mát; phần dư này được tính tại
các điểm không cần nhãn. Karniadakis và cộng sự~\cite{karniadakis2021} tổng kết hướng học máy có thông tin
vật lý, còn Aykol và cộng sự~\cite{aykol2021} thảo luận việc kết hợp vật lý với học máy cho bài toán tuổi
thọ pin. Wen và cộng sự~\cite{wen2023} áp dụng PINN cho dự báo và quản lý sức khoẻ pin. Gần với đồ án nhất là
PINN4SOH của Wang và cộng sự~\cite{wang2024}: dùng chính bốn bộ dữ liệu của đồ án, 16 đặc trưng đoạn sạc, mạng
nghiệm $u(\mathbf{x}, t)$ và một mạng động học dạng hộp đen $G(\mathbf{x}, t, u, u_\mathbf{x}, u_t)$.

\section{Mạng nơ-ron có thông tin vật lý}

Xét phương trình $\mathcal{N}[u] = 0$ với nghiệm $u(\mathbf{z})$. PINN xấp xỉ $u$ bằng mạng $u_\varphi$ và cực
tiểu
\begin{equation}
\mathcal{L}(\varphi) = \frac{1}{|D|}\sum_{(\mathbf{z}, y) \in D}\big(u_\varphi(\mathbf{z}) - y\big)^2
 + \lambda \frac{1}{|P|}\sum_{\mathbf{z} \in P}\big(\mathcal{N}[u_\varphi](\mathbf{z})\big)^2,
\end{equation}
trong đó $D$ là tập điểm có nhãn và $P$ là tập điểm “collocation” chỉ cần đầu vào. Khi phương trình đã biết,
số hạng thứ hai đóng vai trò tín hiệu học bổ sung không cần nhãn. Với pin, phương trình suy giảm không được
biết chính xác, nên một phần của $\mathcal{N}$ phải học từ dữ liệu: đây là chỗ khác nhau giữa động học hộp đen
(học hoàn toàn) và động học xám (giữ cấu trúc vật lý, chỉ học phần chưa biết).

Huấn luyện PINN khó hơn huấn luyện mạng thường. Nghiên cứu của Wang, Teng và Perdikaris~\cite{wang2021grad} cho thấy
gradient của các số hạng trong hàm mất mát có thể chênh nhau nhiều bậc, khiến mạng bỏ qua một số hạng; họ đề
xuất cân lại trọng số trong lúc học. Krishnapriyan và cộng sự~\cite{krishnapriyan2021} chỉ ra rằng ràng buộc
mềm làm mặt mất mát khó tối ưu khi hệ số của phương trình lớn. Hai kết quả này giải thích vì sao đồ án tăng dần
trọng số vật lý trong 20\,\% số bước đầu và giữ hệ số của $\mathcal{L}_\text{ode}$ nhỏ.

Braun, Raible và Huber~\cite{braun2026} tổng hợp 212 công trình học máy có thông tin vật lý trong lĩnh vực dự báo
và quản lý sức khoẻ thiết bị, xếp chúng thành bốn nhóm: đưa vật lý vào dữ liệu (thiên lệch quan sát), vào kiến
trúc (thiên lệch quy nạp), vào hàm mất mát (thiên lệch học), và kết hợp. Pin lithium-ion là đối tượng được nghiên
cứu nhiều nhất. Theo cách xếp này, PINN của đồ án thuộc nhóm thứ ba; bộ lọc quỹ đạo ở Chương~\ref{ch:caitien}
đưa cùng giả định đơn điệu vào bước suy luận. Navidi và cộng sự~\cite{navidi2024} so sánh bốn cách xây mô hình
lai cho chẩn đoán suy giảm trên 24 cell theo dõi bốn năm, theo độ chính xác, độ phức tạp và khả năng ngoại suy;
không có cách nào thắng ở mọi tiêu chí.

\section{Học bán giám sát}

Học bán giám sát tận dụng dữ liệu không nhãn qua một giả định về cấu trúc của dữ liệu, ví dụ tính trơn của
hàm cần học hoặc tính nhất quán của dự đoán. Trong đồ án, giả định được dùng là giả định vật lý: quỹ đạo SOH
của mọi cell, kể cả cell chưa đo dung lượng, phải thoả phương trình suy giảm, không tăng theo thời gian và nằm
trong miền giá trị hợp lý. Các ràng buộc này tính được chỉ từ đặc trưng sạc, nên mỗi cell không nhãn đều
đóng góp gradient cho mạng nghiệm.

Hai họ phương pháp bán giám sát phổ biến khác dựa trên chính dự đoán của mô hình. Gán nhãn giả~\cite{lee2013}
dùng dự đoán của mô hình trên mẫu không nhãn làm nhãn để huấn luyện tiếp; chưng cất tri thức~\cite{hinton2015}
dùng dự đoán của một mô hình thầy làm mục tiêu mềm cho mô hình trò. Mean Teacher~\cite{tarvainen2017} giữ một
bản trung bình trượt của trọng số làm thầy và phạt độ lệch giữa thầy và trò. Điểm chung là nhãn giả chỉ tốt
bằng mô hình sinh ra nó; nếu thầy sai có hệ thống trên một nhóm cell, trò học lại đúng sai số đó. Ràng buộc vật
lý không có nhược điểm này vì nó không phụ thuộc vào mô hình nào, nhưng bù lại nó chỉ cho biết dạng của quỹ đạo
chứ không cho biết mức SOH. Chương~\ref{ch:caitien} thử kết hợp hai nguồn tín hiệu này.

\section{Kết hợp mô hình và độ bất định}

Trung bình dự đoán của nhiều mô hình là cách đơn giản nhất để giảm phương sai. Bagging~\cite{breiman1996} huấn
luyện cùng một mô hình trên các mẫu bootstrap; stacking~\cite{wolpert1992} học trọng số kết hợp các mô hình khác
loại trên dữ liệu không dùng để huấn luyện chúng. Với mạng nơ-ron, Lakshminarayanan, Pritzel và
Blundell~\cite{lakshminarayanan2017} cho thấy chỉ cần huấn luyện vài mạng giống nhau với khởi tạo ngẫu nhiên khác
nhau rồi lấy trung bình là đủ cải thiện độ chính xác, và độ phân tán giữa các mạng là một ước lượng độ bất định
dùng được.

Độ phân tán đó lại không đảm bảo khoảng tin cậy có độ phủ đúng. Dự đoán phù hợp (conformal
prediction)~\cite{vovk2005,lei2018} giải quyết việc này mà không cần giả định về phân phối: trên một tập hiệu
chỉnh tách riêng, tính phần dư của mô hình, lấy phân vị phù hợp làm nửa độ rộng khoảng. Nếu các mẫu hiệu chỉnh
và mẫu mới hoán đổi được cho nhau, khoảng thu được có độ phủ ít nhất bằng mức đặt ra~\cite{angelopoulos2023}.
Thelen và cộng sự~\cite{thelen2024} tổng quan học máy xác suất cho chẩn đoán và dự báo pin, nhấn mạnh rằng
phần lớn công trình vẫn chỉ đưa ra một con số, bỏ qua biến động giữa các cell, trong khi các quyết định như
thay pin hay giới hạn công suất cần biết độ tin cậy của con số đó.

\section{Đánh giá mô hình và rò rỉ dữ liệu}

Kiểm định chéo là cách chuẩn để ước lượng sai số tổng quát~\cite{arlot2010}. Khi dữ liệu có cấu trúc nhóm, ở
đây là các chu kỳ thuộc cùng một cell, việc chia phải theo nhóm; nếu không, các mẫu gần như trùng nhau sẽ nằm
ở cả hai phía train và test. Kapoor và Narayanan~\cite{kapoor2023} tổng kết rò rỉ dữ liệu là một nguyên nhân
phổ biến khiến kết quả học máy trong khoa học không tái lập được; các dạng thường gặp gồm chia không đúng đơn
vị, tiền xử lý dùng thống kê của tập test và chọn mô hình trên tập test~\cite{varma2006}.

Khi so sánh hai mô hình bằng kiểm định chéo lặp lại, các tập huấn luyện chồng lấp nhau nên kết quả các fold
không độc lập; phép kiểm $t$ thông thường vì vậy quá lạc quan~\cite{dietterich1998}. Nadeau và
Bengio~\cite{nadeau2003} đề xuất hiệu chỉnh phương sai, được Bouckaert và Frank~\cite{bouckaert2004} kiểm
chứng cho kiểm định chéo lặp lại. Khi thực hiện nhiều phép so sánh, phương pháp Holm~\cite{holm1979} kiểm
soát xác suất mắc ít nhất một sai lầm loại I.

\section{Tổng hợp và khoảng trống}

\begin{table}[!htbp]\centering\small
\caption{Tóm tắt các hướng tiếp cận liên quan.}\label{tab:related}
\begin{tabularx}{\textwidth}{>{\raggedright}p{3.2cm}>{\raggedright}p{3.6cm}cc>{\raggedright\arraybackslash}X}
\toprule
Công trình & Phương pháp & Vật lý & Không nhãn & Ghi chú \\
\midrule
Severson 2019~\cite{severson2019} & đặc trưng xả + hồi quy tuyến tính & không & không & dự đoán tuổi thọ sớm, bộ MIT\\
Richardson 2017~\cite{richardson2017} & hồi quy Gauss & không & không & có độ bất định\\
Roman 2021~\cite{roman2021} & quy trình học máy & không & không & so sánh nhiều mô hình\\
Zhu 2022~\cite{zhu2022} & điện áp hồi phục + học máy & không & không & bộ TJU\\
Ma 2022~\cite{ma2022} & học chuyển giao sâu & không & một phần & bộ HUST\\
Lu 2023~\cite{lu2023} & nhóm mạng sâu + thích nghi miền & không & có & không cần nhãn pin đích\\
Wang 2011~\cite{wang2011lfp} & hàm luỹ thừa + Arrhenius & có & không & mô hình thực nghiệm, cell LFP\\
Raissi 2019~\cite{raissi2019} & PINN tổng quát & có & có & phương trình đã biết\\
Navidi 2024~\cite{navidi2024} & so sánh bốn cách lai & có & không & chẩn đoán suy giảm\\
Wang 2024~\cite{wang2024} & PINN, động học hộp đen & có & không & cùng bốn bộ dữ liệu\\
Thelen 2024~\cite{thelen2024} & học máy xác suất (tổng quan) & một phần & -- & độ bất định\\
Đồ án & PINN bán giám sát, động học xám, nhóm mạng + lọc quỹ đạo & có & có & loss vật lý không dùng nhãn,
khoảng dự đoán phù hợp\\
\bottomrule
\end{tabularx}
\end{table}

Từ các công trình trên có thể rút ra bốn khoảng trống. Thứ nhất, các mô hình học có nhãn phụ thuộc mạnh vào
lượng nhãn, trong khi dữ liệu sạc không nhãn hầu như chưa được khai thác. Thứ hai, các PINN cho pin hiện có
dùng nhãn trong phần vật lý hoặc chỉ áp vật lý lên cell có nhãn, nên vật lý chỉ đóng vai trò chuẩn hoá chứ
không thay được nhãn. Thứ ba, kết quả công bố thường dựa trên một cách chia duy nhất, ít khi kiểm định
thống kê, ít khi kiểm tra rò rỉ và ít khi so với mô hình cây, vốn mạnh trên dữ liệu dạng bảng. Thứ tư, hầu hết
chỉ đưa ra một giá trị SOH mà không kèm khoảng tin cậy có độ phủ được kiểm chứng. Đồ án nhắm vào cả bốn điểm này.
"""


# ═════════════════════════════════════════════════════════════ CHƯƠNG 3
def ch3(d, ctx):
    st = ctx['stats']
    rows = []
    nom = {'XJTU': '2,0', 'TJU': '3,5 / 2,5', 'MIT': '1,1', 'HUST': '1,1'}
    for _, r in st.iterrows():
        rows.append([r.bo, CHEM[r.bo], num(r.N), num(r.n), str(r.batch),
                     f'{num(r.lmin)} / {num(r.lmed)} / {num(r.lmax)}',
                     f'{f(r.s0a, 2)}--{f(r.s0b, 2)}', f'{f(r.s1a, 2)}--{f(r.s1b, 2)}', nom[r.bo]])
    rows.append([r'\textbf{Tổng}', '', num(st.N.sum()), num(st.n.sum()), '', '', '', '', ''])
    tab = table(['Bộ', 'Hoá học', 'Cell', 'Chu kỳ', 'Nhóm', 'Tuổi thọ (min / trung vị / max)', 'SOH đầu',
                 'SOH cuối', '$Q_\\text{dđ}$ (Ah)'], rows, 'llrrrcccc',
                'Thống kê bốn bộ dữ liệu sau tiền xử lý. “Nhóm” là số chế độ thí nghiệm khác nhau dùng để phân tầng.',
                'tab:datastats', size=r'\footnotesize')
    return T(r"""
\chapter{Dữ liệu}\label{ch:dulieu}

Bốn bộ dữ liệu được chọn vì chúng phủ ba hệ hoá học khác nhau, nhiều chế độ sạc--xả và nhiệt độ, và vì chúng đã
được dùng trong công trình gốc của PINN4SOH, nên kết quả của đồ án so được với tài liệu. Chương này mô tả nguồn
gốc, đặc trưng đầu vào và cách làm sạch dữ liệu; mọi bước làm sạch chỉ dùng quá khứ của chính cell để không đưa
thông tin tương lai vào mô hình.

\section{Nguồn dữ liệu}

Đồ án dùng bốn bộ dữ liệu lão hoá pin công khai. Tất cả là cell hình trụ 18650 được sạc--xả lặp lại trong
phòng thí nghiệm cho tới khi dung lượng giảm sâu.

\begin{itemize}
\item \textbf{XJTU} --- Đại học Giao thông Tây An, công bố cùng PINN4SOH~\cite{wang2024}: 55 cell NCM, dung
      lượng danh định 2,0\,Ah, sáu chế độ thí nghiệm (sạc 2C, sạc 3C, xả ngẫu nhiên R2.5 và R3, bước ngẫu nhiên
      RW, mô phỏng quỹ đạo vệ tinh) ở nhiệt độ phòng.
\item \textbf{TJU} --- Zhu và cộng sự~\cite{zhu2022}: 130 cell thuộc ba nhóm hoá học (NCA, NCM và hỗn hợp
      NCM+NCA), thử ở 25, 35 và 45\,\textdegree C với nhiều tốc độ sạc, tạo thành chín nhóm điều kiện.
\item \textbf{MIT} --- Severson và cộng sự~\cite{severson2019}: cell LFP/graphite 1,1\,Ah, nhiều giao thức sạc
      nhanh khác nhau, chia thành ba đợt thí nghiệm, buồng nhiệt 30\,\textdegree C.
\item \textbf{HUST} --- Ma và cộng sự~\cite{ma2022}: 77 cell LFP/graphite 1,1\,Ah, cùng chế độ sạc nhanh và 77
      chế độ xả nhiều bước khác nhau, 30\,\textdegree C.
\end{itemize}

Đặc trưng đoạn sạc của cả bốn bộ được trích theo cùng một quy trình và công bố kèm PINN4SOH~\cite{wang2024};
nhờ vậy bốn bộ có chung không gian đặc trưng và có thể thử chuyển miền giữa chúng.

Bộ thứ năm, \textbf{NASA} --- Saha và Goebel~\cite{saha2007nasa}, là bộ thường dùng nhất trong các công trình về SOH
và RUL. Đồ án dùng 18 cell 18650 NCA (danh định 2,0\,Ah) có chế độ xả cố định, đúng tập cell mà
SambaMixer~\cite{olalde2024samba} dùng (bỏ cell số 45 vì chỉ có 17 lần sạc đầy đủ), thử ở 4, 24 và 43\,\textdegree C
với dòng xả 1--4\,A, chia thành sáu nhóm theo giao thức. Dữ liệu gốc chỉ có chuỗi đo, nên đồ án tự trích 16 đặc trưng
đoạn sạc theo đúng định nghĩa của PINN4SOH (đoạn CC từ 4,0 tới 4,2\,V, đoạn CV từ 0,5 xuống 0,1\,A) từ lần sạc ngay
trước mỗi lần xả; nhãn là dung lượng của lần xả chia cho 2,0\,Ah. Các lần xả hỏng mà tài liệu của NASA đã cảnh báo
(dung lượng thấp hơn trung vị bảy lần xả lân cận quá 10\,\%) bị loại. Khác với bốn bộ kia, giữa các lần xả của NASA
có những khoảng nghỉ dài tới hàng trăm giờ, và dung lượng hồi phục rõ sau mỗi khoảng nghỉ đó; khoảng nghỉ trước mỗi lần
xả được giữ lại để dùng như một đặc trưng thứ 17.

\section{Đặc trưng đầu vào}

Mỗi chu kỳ được mô tả bằng 16 đặc trưng thống kê của đoạn sạc CC--CV:
\begin{itemize}
\item điện áp trong pha dòng không đổi: trung bình, độ lệch chuẩn, độ nhọn, độ lệch, độ dốc, entropy;
\item dòng điện trong pha áp không đổi: trung bình, độ lệch chuẩn, độ nhọn, độ lệch, độ dốc, entropy;
\item điện lượng nạp và thời gian của pha CC và của pha CV.
\end{itemize}
Ngoài 16 đặc trưng, mô hình nhận thêm biến thời gian $t_n = n/1000$, với $n$ là chỉ số chu kỳ gốc. Hằng số
1000 dùng chung cho mọi cell; không chuẩn hoá theo tuổi thọ riêng của từng cell, vì làm vậy sẽ đưa thông tin
tương lai vào đầu vào (Mục~\ref{sec:audit}).

\section{Tiền xử lý}

\paragraph{Nhãn.} $y_{c,n} = Q_{c,n}/Q_\text{danh định}$ với dung lượng danh định của từng bộ ở
Bảng~\ref{tab:datastats}. Chu kỳ thiếu dung lượng được giữ lại với nhãn trống và chỉ dùng cho loss vật lý.

\paragraph{Lọc ngoại lai nhân quả.} Chu kỳ $n$ bị loại nếu có đặc trưng không hữu hạn, hoặc nếu có đặc trưng
lệch xa so với 50 chu kỳ liền trước của chính cell đó. Với đặc trưng thứ $j$:
\begin{equation}
m_{n,j} = \operatorname{median}(x_{n-50..n-1,\,j}), \quad
s_{n,j} = \max\!\Big(\tfrac{\mathrm{IQR}(x_{n-50..n-1,\,j})}{1{,}349},\, 0{,}01\,|m_{n,j}|\Big), \quad
z_{n,j} = \frac{|x_{n,j} - m_{n,j}|}{s_{n,j}},
\end{equation}
và chu kỳ bị loại nếu $\max_j z_{n,j} > 5$ (cần ít nhất 10 chu kỳ lịch sử). Hệ số 1,349 đổi khoảng tứ phân vị
sang độ lệch chuẩn của phân phối chuẩn; ngưỡng 5 được chọn trước khi huấn luyện để tỉ lệ chu kỳ bị loại
(1,6\,\%) tương đương bộ lọc $3\sigma$ dùng cả vòng đời (2,2\,\%). Sau lọc còn <<NROW>> chu kỳ.

\paragraph{Chuẩn hoá.} Mỗi đặc trưng được chuẩn hoá z-score bằng trung bình $\boldsymbol\mu$ và độ lệch chuẩn
$\boldsymbol\sigma$ tính trên đặc trưng của các cell huấn luyện (cả có nhãn và không nhãn):
$\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol\mu)/(\boldsymbol\sigma + 10^{-6})$.

\section{Thống kê mô tả}

<<TAB>>

Bốn bộ khác nhau rõ rệt về hoá học, tuổi thọ và mức đa dạng điều kiện vận hành. XJTU và TJU (họ niken) có
tuổi thọ ngắn và nhiều chế độ; MIT và HUST (LFP) có tuổi thọ dài hơn nhiều. SOH đầu đời của HUST lớn hơn 1 do
dung lượng thực tế của cell mới cao hơn dung lượng danh định. TJU là bộ duy nhất có nhiều mức nhiệt độ, nên
cũng là bộ duy nhất mà hệ số Arrhenius có thể học được (Mục~\ref{sec:physparams}).
""", NROW=num(st.n.sum()), TAB=tab)


# ═════════════════════════════════════════════════════════════ CHƯƠNG 4
def ch4(d, ctx):
    return r"""
\chapter{Phương pháp}\label{ch:phuongphap}

\section{Tổng quan}

Mô hình gồm hai mạng nối với nhau qua hàm mất mát (Hình~\ref{fig:arch}). Mạng nghiệm $F_\varphi$ dự đoán SOH;
mạng tốc độ $G_\theta$ cùng hai tham số vật lý $\lambda, E_a$ mô tả tốc độ suy giảm. Khi suy luận chỉ cần
$F_\varphi$.

\begin{figure}[!htbp]\centering
\includegraphics[width=\textwidth]{hinh/fig_thesis_arch.pdf}
\caption{Kiến trúc mô hình: mạng nghiệm dự đoán SOH, mạng tốc độ mô tả động học suy giảm; hai mạng gắn với
nhau qua các hàm mất mát vật lý.}\label{fig:arch}
\end{figure}

\section{Mạng nghiệm}

\begin{equation}
u = F_\varphi(\tilde{\mathbf{x}}, t), \qquad F_\varphi:\R^{17} \to \R,
\end{equation}
là perceptron nhiều lớp 17--128--128--64--1 với hàm kích hoạt SiLU.

\section{Động học suy giảm dạng xám}

\begin{equation}\label{eq:rate}
r(\tilde{\mathbf{x}}, u, T) =
\underbrace{\operatorname{softplus}\big(G_\theta(\tilde{\mathbf{x}})\big)}_{k(\mathbf{x})}\;
\underbrace{e^{\lambda\,\operatorname{clip}(1-u,\,-0{,}5,\,1)}}_{g(u)}\;
\underbrace{e^{-\frac{E_a}{R_g}\left(\frac{1}{T}-\frac{1}{T_\text{ref}}\right)}}_{A(T)},
\end{equation}
với $G_\theta$ là perceptron 16--32--32--1, $\lambda = \operatorname{softplus}(\lambda_\text{raw})$,
$E_a = 10^4 \operatorname{softplus}(e_\text{raw})$\,J/mol, $R_g = 8{,}314$\,J/(mol·K), $T_\text{ref} = 298{,}15$\,K.
Thừa số $k$ là tốc độ cơ sở phụ thuộc trạng thái điện hoá (thể hiện qua đặc trưng sạc), $g$ làm suy giảm tăng
nhanh khi SOH thấp (hiện tượng đầu gối), $A$ là hệ số Arrhenius. Phương trình mà mạng nghiệm phải thoả:
\begin{equation}\label{eq:ode}
\frac{\partial u}{\partial t} = -\,r(\tilde{\mathbf{x}}, u, T).
\end{equation}

\section{Hàm mất mát}

Với cặp chu kỳ $(n_1, n_2)$ của cùng một cell, $n_2 \ge n_1 + h$, $h \sim \mathcal{U}\{5,\dots,50\}$, ký hiệu
$u_i = F_\varphi(\tilde{\mathbf{x}}_{n_i}, t_{n_i})$:
\begin{align}
\mathcal{L}_\text{data} &= \tfrac{1}{B}\textstyle\sum (u - y)^2 && \text{(chỉ chu kỳ có nhãn)},\\
\mathcal{L}_\text{ode} &= \tfrac{1}{B}\textstyle\sum \big(\partial_t u_1 + r(\tilde{\mathbf{x}}_{n_1}, u_1, T)\big)^2, \\
\mathcal{L}_\text{mono} &= \tfrac{1}{B}\textstyle\sum \operatorname{ReLU}(u_2 - u_1 - \varepsilon), && \varepsilon = 0{,}002,\\
\mathcal{L}_\text{range} &= \tfrac{1}{B}\textstyle\sum \big[\operatorname{ReLU}(u_1 - 1{,}15) + \operatorname{ReLU}(0{,}4 - u_1)\notag\\
&\qquad\qquad + \mathbb{1}_\text{đầu}\big(\operatorname{ReLU}(u_1 - 1{,}10) + \operatorname{ReLU}(0{,}85 - u_1)\big)\big],
\end{align}
\begin{equation}
\mathcal{L} = \mathcal{L}_\text{data} + w(s)\big(\alpha\mathcal{L}_\text{ode} + \beta\mathcal{L}_\text{mono}
+ \gamma\mathcal{L}_\text{range}\big), \qquad w(s) = \min\big(1, s/(0{,}2S)\big),
\end{equation}
với $\alpha = 0{,}02$, $\beta = 0{,}5$, $\gamma = 1$, $S = 4000$ bước. $\mathbb{1}_\text{đầu} = 1$ nếu $n_1$ là
chu kỳ đầu của cell. Đạo hàm $\partial_t u$ tính bằng vi phân tự động.

\section{Cơ chế bán giám sát}

PINN-semi lấy một nửa số cặp chu kỳ từ cell có nhãn và một nửa từ cell không nhãn. Ba loss vật lý không chứa
$y$ (Mệnh đề~\ref{md:labelfree}), nên gradient của chúng theo $\varphi$ trên cell không nhãn là tín hiệu học
thật sự: mạng nghiệm bị buộc phải cho quỹ đạo hợp lý vật lý trên chính những cell mà nó sẽ phải tổng quát tới.

\section{Phần vật lý nằm ở đâu}\label{sec:wherephys}

Mạng nghiệm $F_\varphi$ tự nó là một perceptron nhiều lớp bình thường, giống hệt MLP đối chứng. Điều làm mô hình
trở thành PINN nằm ở hàm mất mát và ở các điểm mà hàm mất mát được tính, không nằm ở kiến trúc mạng hay ở bước
suy luận. Vật lý ở đây là vật lý ở mức hiện tượng: những quy luật định tính đã biết về lão hoá pin (dung lượng
không tự hồi phục, suy giảm nhanh dần khi pin già, phản ứng phụ tăng theo nhiệt độ), được viết thành phương trình
và bất đẳng thức. Đồ án không giải mô hình điện hoá chi tiết (P2D, mô hình hạt đơn) và không mô hình hoá đặc
trưng sạc bằng phương trình nào. Bảng~\ref{tab:physmap} nối từng hiểu biết vật lý với thành phần tương ứng.

\begin{table}[!htbp]\centering\small
\caption{Hiểu biết vật lý và thành phần mô hình tương ứng.}\label{tab:physmap}
\begin{tabularx}{\textwidth}{>{\raggedright}p{3.6cm}>{\raggedright}p{3.4cm}>{\raggedright}X c}
\toprule
Hiểu biết vật lý & Biểu diễn & Tác động vào đâu & Cần nhãn \\
\midrule
Dung lượng không tăng theo thời gian vận hành & $\partial u/\partial t = -r$, $r \ge 0$ & đạo hàm riêng
$\partial F_\varphi/\partial t$ tại điểm $(\tilde{\mathbf{x}}_{n_1}, t_{n_1})$, qua $\mathcal{L}_\text{ode}$ & không \\
Quỹ đạo SOH thực tế không tăng (cho phép hồi phục nhỏ $\varepsilon$) & $u_2 \le u_1 + \varepsilon$ & cặp chu kỳ
$(n_1, n_2)$ thật của cùng cell, qua $\mathcal{L}_\text{mono}$ & không \\
Suy giảm nhanh dần khi pin già (đầu gối) & $g(u) = e^{\lambda(1-u)}$ & thừa số trong $r$ & không \\
Phản ứng phụ tăng theo nhiệt độ & $A(T)$ dạng Arrhenius & thừa số trong $r$ & không \\
SOH nằm trong miền hợp lý, cell mới gần 1 & các hàm ReLU chặn & giá trị $u_1$, qua $\mathcal{L}_\text{range}$ & không \\
Dung lượng đo được & $(u - y)^2$ & chu kỳ có nhãn, qua $\mathcal{L}_\text{data}$ & có \\
\bottomrule
\end{tabularx}
\end{table}

Có ba điểm cần phân biệt. Thứ nhất, các ràng buộc được tính tại \emph{điểm collocation}: mọi chu kỳ của mọi cell
train, kể cả cell chưa đo dung lượng, ghép thành cặp $(n_1, n_2)$; đây là chỗ dữ liệu không nhãn đi vào mô hình.
Thứ hai, $\mathcal{L}_\text{ode}$ ràng buộc đạo hàm riêng theo $t$ khi giữ nguyên đặc trưng sạc, còn
$\mathcal{L}_\text{mono}$ ràng buộc tổng biến thiên dọc quỹ đạo thật, nơi cả $\mathbf{x}$ lẫn $t$ cùng đổi; hai
ràng buộc này tác động vào hai hướng khác nhau của hàm $F_\varphi$, và Mục~\ref{sec:physcheck} đo riêng từng hướng.
Thứ ba, khi suy luận chỉ dùng $F_\varphi$: vật lý định hình hàm được học, chứ không tham gia vào việc tính dự đoán.

\section{Tính chất của mô hình}

\begin{menhde}[Tốc độ suy giảm không âm]\label{md:rate}
Với mọi tham số $\theta, \lambda_\text{raw}, e_\text{raw}$ và mọi $(\tilde{\mathbf{x}}, u, T)$, $T > 0$:
$r(\tilde{\mathbf{x}}, u, T) \ge 0$. Do đó mọi nghiệm của \eqref{eq:ode} không tăng theo $t$.
\end{menhde}
\begin{proof}
$\operatorname{softplus}(z) = \ln(1+e^z) > 0$ với mọi $z$ thực vì $1 + e^z > 1$. Hai thừa số còn lại là hàm
mũ của số thực nên dương. Tích của ba số dương là dương; trong số học dấu phẩy động, softplus có thể tràn
dưới về $0$ nên phát biểu đúng là $r \ge 0$. Khi đó $\partial u/\partial t = -r \le 0$.
\end{proof}

\begin{menhde}[Loss vật lý không dùng nhãn]\label{md:labelfree}
$\mathcal{L}_\text{ode}$, $\mathcal{L}_\text{mono}$, $\mathcal{L}_\text{range}$ chỉ phụ thuộc đặc trưng và thời
điểm của cặp chu kỳ ($\tilde{\mathbf{x}}_{n_i}$, $t_{n_i}$), nhiệt độ $T$ và tham số
$(\varphi, \theta, \lambda, E_a)$; không phụ thuộc $y$. Vì vậy chúng tính được trên mọi cell chỉ có dữ liệu sạc.
\end{menhde}
\begin{proof}
Trực tiếp từ định nghĩa: không biểu thức nào chứa $y$; $\partial_t u_1$ là đạo hàm của $F_\varphi$ theo đầu vào
$t$, không qua nhãn.
\end{proof}

\begin{menhde}[Tập không của loss đơn điệu]\label{md:mono}
$\mathcal{L}_\text{mono} = 0$ khi và chỉ khi $u_2 \le u_1 + \varepsilon$ với mọi cặp trong batch; khi vi phạm,
gradient theo $u_2 - u_1$ bằng $1/B$ cho mỗi cặp vi phạm, không phụ thuộc mức vi phạm.
\end{menhde}
\begin{proof}
$\operatorname{ReLU}(z) \ge 0$ và bằng $0$ khi và chỉ khi $z \le 0$, nên tổng các số hạng không âm bằng $0$ khi
và chỉ khi từng số hạng bằng $0$. Đạo hàm của $\operatorname{ReLU}$ bằng $1$ trên $z > 0$.
\end{proof}

\begin{menhde}[Dạng tích phân của phần dư]\label{md:euler}
Nếu $u$ thoả \eqref{eq:ode} dọc một quỹ đạo và $r$ khả vi liên tục theo $t$ trên $[t_1, t_2]$, thì phần dư dạng
Euler $\rho = u_2 - u_1 + r_1\Delta t$ thoả $|\rho| \le \tfrac12 \max|\dot r|\,\Delta t^2$.
\end{menhde}
\begin{proof}
$u_2 - u_1 = -\int_{t_1}^{t_2} r(s)\,ds = -r_1\Delta t - \int_{t_1}^{t_2}(r(s) - r_1)\,ds$, và
$|r(s) - r_1| \le \max|\dot r|\,(s - t_1)$. Lấy tích phân được cận $\tfrac12\max|\dot r|\Delta t^2$.
\end{proof}

Mệnh đề~\ref{md:euler} cho thấy dạng Euler dọc quỹ đạo và dạng đạo hàm riêng $\partial_t u$ là hai ràng buộc
khác nhau: dạng thứ nhất ràng buộc tổng biến thiên dọc chuỗi chu kỳ thực (khi đặc trưng $\mathbf{x}$ cũng thay
đổi), dạng thứ hai ràng buộc đạo hàm theo $t$ khi giữ $\mathbf{x}$ cố định. Cả hai được so sánh ở
Chương~\ref{ch:ketqua}.

\section{Các mô hình đối chứng}\label{sec:baselines}

Mọi mô hình dùng cùng đầu vào $[\tilde{\mathbf{x}}, t]$, cùng cách chia, cùng tập cell có nhãn và cùng
normaliser.

\begin{table}[!htbp]\centering\small
\caption{Các mô hình được so sánh.}\label{tab:models}
\begin{tabularx}{\textwidth}{l>{\raggedright}p{3.2cm}>{\raggedright\arraybackslash}X}
\toprule
Mô hình & Nhóm & Mô tả \\
\midrule
Ridge & học có nhãn & hồi quy tuyến tính chuẩn hoá L2; $\alpha \in \{10^{-3},\dots,10^{3}\}$ chọn theo validation\\
GPR & học có nhãn & hồi quy Gauss~\cite{rasmussen2006}, nhân RBF + nhiễu trắng, 1\,500 điểm có nhãn chọn ngẫu nhiên\\
GBDT & học có nhãn & gradient boosting trên cây~\cite{friedman2001}, dạng histogram~\cite{ke2017}; số cây chọn theo validation\\
MLP & học có nhãn & cùng kiến trúc với mạng nghiệm, chỉ $\mathcal{L}_\text{data}$\\
PINN-sup & vật lý trên cell có nhãn & như PINN-semi nhưng loss vật lý chỉ tính trên cell có nhãn\\
PINN hộp đen & vật lý, bán giám sát & động học dạng PINN4SOH~\cite{wang2024}: $u_t = G(\mathbf{x}, t, u, u_\mathbf{x}, u_t)$, cùng loss không nhãn\\
PINN-semi & vật lý, bán giám sát & mô hình đề xuất\\
\bottomrule
\end{tabularx}
\end{table}

Hai biến thể cắt bỏ dùng để tách nguồn gốc lợi ích: \emph{chỉ đơn điệu} ($\alpha = \gamma = 0$) và
\emph{PINN-semi dạng Euler} (dùng phần dư dọc quỹ đạo thay cho $\partial_t u$, $\alpha = 2$).

\section{Cấu hình huấn luyện}

\begin{table}[!htbp]\centering\small
\caption{Cấu hình huấn luyện chung cho MLP và các biến thể PINN.}\label{tab:train}
\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X}\toprule
Thành phần & Giá trị \\\midrule
Bộ tối ưu & AdamW~\cite{loshchilov2019}, tốc độ học $2\cdot10^{-3}$, weight decay $10^{-5}$ \\
Lịch tốc độ học & cosine, giảm về $10^{-4}$ trong 4\,000 bước \\
Batch & 512 điểm có nhãn; 512 cặp chu kỳ cho loss vật lý \\
Cắt gradient & chuẩn $L_2 \le 5$ \\
Dừng sớm & MAE trên validation mỗi 100 bước, dừng sau 12 lần không cải thiện, giữ trọng số tốt nhất \\
\bottomrule\end{tabularx}\end{table}

Trọng số vật lý ($\alpha, \beta$) và dạng phần dư được chọn trước theo MAE validation ở một thí nghiệm sơ bộ
và giữ cố định trong toàn bộ các thí nghiệm của báo cáo; không có siêu tham số nào được chọn theo tập test.
"""


# ═════════════════════════════════════════════════════════════ CHƯƠNG 5
def ch5(d, ctx):
    sd = d['thesis_split_design']
    rows = []
    for _, r in sd.iterrows():
        rows.append([r.bo if r.K == 3 else '', str(r.K), f'{r.train_min} ({pc(r.frac_train)})', str(r.val),
                     f'{r.test_min}--{r.test_max}' if r.test_min != r.test_max else str(r.test_min),
                     'đạt' if r.kha_thi_70 else 'không', str(r.luot)])
    tabK = table(['Bộ', '$K$', 'Train (cell)', 'Val', 'Test/fold', '70\\,\\% nhãn', 'Lượt/mô hình/mức'], rows,
                 'llrrrcr', 'Khả thi của số fold $K$ với validation 10\\,\\% và mức nhãn cao nhất 70\\,\\% số cell '
                 '(số lượt tính với 5 lần lặp).', 'tab:kfeas', size=r'\small')
    # đánh đổi tỉ lệ validation
    vrows = []
    for v in [0.05, 0.10, 0.15, 0.20]:
        kmin = math.ceil(1 / (1 - v - 0.70) - 1e-9)
        vrows.append([pc(v), str(kmin), str(int(round(v * 55))), f'{55 / kmin:.1f}'.replace('.', '{,}'),
                      str(kmin * 5)])
    tabV = table(['Validation $v$', '$K_{\\min}$', 'Val (cell XJTU)', 'Test/fold XJTU', 'Lượt/mô hình/mức'], vrows,
                 'rrrrr', 'Đánh đổi khi chọn tỉ lệ validation (mức nhãn cao nhất 70\\,\\%, 5 lần lặp).', 'tab:vtrade')
    st = d['thesis_strat']
    strat_ok = all(st.moi_cell_test_dung_1_lan) and st.lech_lon_nhat.max() <= 1
    rp = d['thesis_repeats']
    r1 = rp[rp.R == 1]; r4 = rp[rp.R == 4]; r5 = rp[rp.R == 5]
    rprows = []
    for ds in DS:
        a, b, c = r1[r1.bo == ds].iloc[0], r4[r4.bo == ds].iloc[0], r5[r5.bo == ds].iloc[0]
        rprows.append([ds, f(a.ti_so_sd, 3), f(b.ti_so_sd, 3), f(a.rong_KTC, 3), f(c.rong_KTC, 3),
                       f'{f(a.ti_so_min, 3)}--{f(a.ti_so_max, 3)}', f(c.ti_so_tb, 3)])
    tabR = table(['Bộ', 'SD, $R{=}1$', 'SD, $R{=}4$', 'KTC, $R{=}1$', 'KTC, $R{=}5$', 'Dải ước lượng, $R{=}1$',
                  'Ước lượng, $R{=}5$'], rprows, 'lrrrrcr',
                 'Mô phỏng lấy lại theo số lần lặp: tỉ số MAE PINN-semi/MLP ở 30\\,\\% nhãn. SD là độ lệch chuẩn của '
                 'ước lượng giữa các tổ hợp lần lặp; KTC là độ rộng khoảng tin cậy 95\\,\\% bootstrap theo cell.',
                 'tab:repeats', size=r'\small')
    rb = d.get('thesis_rowbound')
    lk = d.get('thesis_leakage')
    rbrows = []
    for ds in DS:
        b = rb[rb.bo == ds].iloc[0]
        obs = lk.set_index('bo').loc[ds, 'noi_suy_hang'] if lk is not None and 'noi_suy_hang' in lk.columns else float('nan')
        std = lk.set_index('bo').loc[ds, 'chuan'] if lk is not None and 'chuan' in lk.columns else float('nan')
        rbrows.append([ds, m3(b.dy_tb, 2), m3(b.can_tren, 2), m3(obs, 2), m3(std, 2)])
    tabB = table(['Bộ', '$\\overline{|\\Delta y|}$ (×$10^{-3}$)', 'Cận $\\E[d]\\,\\overline{|\\Delta y|}$',
                  'Nội suy, đo được', 'MLP, chia theo cell'], rbrows, 'lrrrr',
                 'Sai số của phép “dự đoán” chỉ chép nhãn của chu kỳ train gần nhất trong cùng cell khi chia theo '
                 'chu kỳ, so với cận của Mệnh đề~\\ref{md:rowleak} và với sai số thật khi chia theo cell '
                 '(MAE theo cell, ×$10^{-3}$).', 'tab:rowbound')
    rc = ctx['runs']
    exrows = [['Kiểm định chéo chính', 'MLP, PINN-semi ở 4 mức; PINN-sup, 2 biến thể cắt bỏ ở 30\\,\\%', num(rc['E13'])],
              ['Đối chứng học có nhãn', 'Ridge, GPR, GBDT ở 4 mức nhãn', num(rc['E16'])],
              ['Mở rộng biến thể vật lý', 'PINN-sup ở 10, 50\\,\\%; PINN hộp đen ở 4 mức', num(rc['E17'])],
              ['Kiểm toán rò rỉ', '4 cách chia/chuẩn hoá sai, MLP 70\\,\\% nhãn, 2 lần lặp', num(rc['E15'])],
              ['Chuyển miền', '12 cặp nguồn--đích, $k \\in \\{0, 3\\}$, 5 seed; MLP, GBDT, PINN-semi', num(rc['E14'])],
              ['Cải tiến mô hình (Chương~\\ref{ch:caitien})', '9 hướng trên validation; PINN-lai, 3 lần lặp, 4 mức nhãn',
               num(rc['E20'])],
              ['Huấn luyện một bộ, kiểm tra bộ khác', '2 giao thức × 4 bộ nguồn × 3 hạt giống; thích nghi, 2 vòng học lặp',
               num(rc['E21'])],
              ['Dự báo SOH tương lai và RUL', 'GBDT + 3 PINN-dự báo mỗi fold, 5 fold × 3 lần lặp × 4 bộ', num(rc['E22'])],
              ['Dự báo có xét hồi phục', 'cùng fold với E22, thêm bộ NASA; học lại cả mô hình E22 để so ghép cặp',
               num(rc['E23'])],
              ['Bộ NASA', '2 bộ đặc trưng × 2 mức nhãn × 15 fold; cách chia của SambaMixer, 3 hạt giống', num(rc['E24'])],
              [r'\textbf{Tổng}', '', r'\textbf{' + num(rc['tong']) + '}']]
    tabE = table(['Thí nghiệm', 'Nội dung', 'Lượt'], exrows, 'llr', 'Các thí nghiệm và số lượt huấn luyện.', 'tab:exps')
    rs = d['thesis_rowsim']; mc = rs[rs.bo == 'mo_phong']; real = rs[rs.bo != 'mo_phong']
    mcerr = float((100 * (mc.E_d_mc / mc.E_d - 1).abs()).max())
    inb = int((real.mae_chep <= 1.05 * real.can).sum())
    lkc = d['thesis_leakage'].set_index('bo')['chuan']
    hi = real[real.q >= 0.5]
    fold = float(min(lkc[r.bo] / r.mae_chep for r in hi.itertuples()))
    return T(r"""
\chapter{Thiết kế thí nghiệm và giao thức đánh giá}\label{ch:thietke}

Chương này trả lời ba câu hỏi về cách đánh giá: chia dữ liệu theo đơn vị nào và thành bao nhiêu phần, vì sao
chọn đúng các con số đó, và làm sao bảo đảm không có thông tin của tập test lọt vào quá trình huấn luyện,
kể cả khi chuyển mô hình từ bộ dữ liệu này sang bộ khác.

\section{Các yêu cầu đối với giao thức}

\begin{enumerate}[label=\textbf{Y\arabic*.}]
\item \emph{Không rò rỉ}: mô hình và mọi bước tiền xử lý không được dùng bất kỳ thông tin nào của cell test.
\item \emph{Phủ hết dữ liệu}: mỗi cell đều phải được dùng làm test, để kết quả không phụ thuộc vào việc chọn may
      rủi một nhóm cell test.
\item \emph{Phủ hết điều kiện thí nghiệm}: mỗi fold phải có cell của mọi chế độ thí nghiệm, để không có fold
      nào chỉ kiểm tra trên một chế độ lạ.
\item \emph{Đủ nhãn ở mức thấp nhất}: ở 10\,\% nhãn, bộ nhỏ nhất (XJTU, 55 cell) vẫn phải có vài cell có nhãn.
\item \emph{Đủ cell validation} để dừng sớm và chọn siêu tham số ổn định.
\item \emph{Đủ chỗ cho mức nhãn cao nhất}: tập train phải chứa được 70\,\% số cell có nhãn.
\item \emph{Ước lượng ổn định}: đủ số lần lặp để kết quả không phụ thuộc một cách xáo ngẫu nhiên.
\item \emph{Ghép cặp}: mọi mô hình dùng đúng cùng các tập cell, để so sánh được theo từng cell.
\end{enumerate}

\section{Đơn vị chia: cell, không phải chu kỳ}\label{sec:unit}

Các chu kỳ liên tiếp của một cell gần như giống nhau: SOH chỉ thay đổi rất ít giữa hai chu kỳ kề nhau. Nếu chia
ngẫu nhiên theo chu kỳ, gần như mọi chu kỳ test đều có một chu kỳ “hàng xóm” trong tập train, và bài toán
suy biến thành nội suy dọc một quỹ đạo đã biết nhãn.

\begin{menhde}[Cận sai số khi chia theo chu kỳ]\label{md:rowleak}
Giả sử mỗi chu kỳ được đưa vào train độc lập với xác suất $q$. Xét bộ dự đoán không dùng đặc trưng nào, chỉ
chép nhãn của chu kỳ train gần nhất trong cùng cell. Khi bỏ qua hiệu ứng ở hai đầu cell, khoảng cách $d$ tới
chu kỳ train gần nhất thoả
\begin{equation}
\E[d] = \frac{1}{1 - (1-q)^2},\label{eq:Ed}
\end{equation}
và sai số tuyệt đối kỳ vọng không vượt quá $\E[d]\cdot\overline{|\Delta y|}$, với $\overline{|\Delta y|}$ là biến
thiên nhãn trung bình giữa hai chu kỳ liền kề.
\end{menhde}
\begin{proof}
$d > j$ khi và chỉ khi $2j$ chu kỳ ở khoảng cách $1, \dots, j$ về hai phía đều không thuộc train, xác suất
$(1-q)^{2j}$. Vì $d$ nhận giá trị nguyên dương, $\E[d] = \sum_{j \ge 0} P(d > j) = \sum_j (1-q)^{2j} = 1/(1-(1-q)^2)$.
Gọi $n^\ast = n + s d$, $s \in \{-1, +1\}$, là chu kỳ train gần nhất. Theo bất đẳng thức tam giác,
$|y_n - y_{n^\ast}| \le \sum_{i=0}^{d-1} |\Delta y_{n + s i}|$, với $\Delta y_m = y_{m+s} - y_m$. Lấy kỳ vọng theo
cách chia và theo vị trí $n$ của chu kỳ test, phân bố đều trên cell:
\begin{equation*}
\E\Big[\sum_{i=0}^{d-1} |\Delta y_{n+si}|\Big] = \sum_{i \ge 0} \E\big[\mathbb{1}\{d > i\}\,|\Delta y_{n+si}|\big]
= \sum_{i \ge 0} P(d > i)\,\overline{|\Delta y|} = \E[d]\cdot\overline{|\Delta y|},
\end{equation*}
vì sự kiện $\{d > i\}$ chỉ phụ thuộc cách chia nên độc lập với $n$, và khi $n$ phân bố đều (bỏ qua hai đầu cell)
thì $\E_n|\Delta y_{n+si}| = \overline{|\Delta y|}$ với mọi $i$.
\end{proof}

Với $q = 0{,}7$, $\E[d] = <<ED>>$ chu kỳ. Bảng~\ref{tab:rowbound} so sánh cận này với sai số đo được: bộ chép
nhãn không học gì vẫn đạt sai số nhỏ hơn mô hình thật nhiều lần, và khớp với cận lý thuyết. Một mô hình được
đánh giá theo cách chia này vì vậy có thể đạt kết quả rất tốt chỉ bằng cách nhớ quỹ đạo, không phải bằng cách
học quan hệ giữa đặc trưng sạc và SOH. Đơn vị chia của đồ án là cell: mọi chu kỳ của một cell nằm trọn trong
train, validation hoặc test.

<<TABB>>

Để kiểm tra mệnh đề trên toàn bộ dải $q$, đồ án mô phỏng hai lần (Hình~\ref{fig:rowsim}). Thứ nhất, gán ngẫu nhiên
$10^6$ vị trí vào train với xác suất $q$ và đo khoảng cách trung bình tới vị trí train gần nhất: kết quả Monte Carlo
lệch công thức \eqref{eq:Ed} không quá <<MCERR>>\,\%. Thứ hai, áp phép chép nhãn lên quỹ đạo SOH thật của cả 387
cell với $q$ từ 0,1 tới 0,9: <<INB>> trong <<NPT>> điểm nằm dưới hoặc trong khoảng 5\,\% quanh cận; các điểm vượt
nhẹ đến từ hai đầu cell, nơi chu kỳ train gần nhất chỉ có ở một phía. Ở mọi $q \ge 0{,}5$, sai số chép nhãn
nhỏ hơn MLP được đánh giá đúng cách từ <<FOLD>> lần trở lên.

<<FIGRS>>

\section{Chọn số fold \texorpdfstring{$K$}{K}}

Mỗi fold gồm ba phần: test $1/K$, validation $v$, train $1 - 1/K - v$. Yêu cầu Y6 buộc train chứa được
$f_{\max} = 70\,\%$ số cell.

\begin{menhde}[Số fold nhỏ nhất]\label{md:kmin}
Điều kiện $1 - 1/K - v \ge f_{\max}$ tương đương $K \ge 1/(1 - v - f_{\max})$. Với $v = 0{,}10$ và
$f_{\max} = 0{,}70$, số fold nhỏ nhất là $K = 5$.
\end{menhde}
\begin{proof}
Biến đổi tương đương: $1 - v - f_{\max} \ge 1/K \Leftrightarrow K \ge 1/(1 - v - f_{\max}) = 1/0{,}2 = 5$.
\end{proof}

Trong các giá trị khả thi ($K \ge 5$), $K = 5$ được chọn vì ba lý do. Một là mỗi fold có nhiều cell test nhất
(XJTU: 11 cell), nên MAE của từng fold ổn định hơn. Hai là chi phí tính toán nhỏ nhất: số lượt huấn luyện tỉ lệ
với $K$. Ba là tỉ lệ train 70\,\% trùng với giao thức 70/15/15 thường dùng trong các công trình trước, nên mức
70\,\% nhãn ở đây tương ứng với trường hợp “dùng hết nhãn” của các công trình đó. $K = 10$ cũng khả thi nhưng
chỉ còn 5--6 cell test mỗi fold ở XJTU và gấp đôi chi phí; kiểm định chéo bỏ-một-cell cần tới 55--130 lượt cho
mỗi cấu hình. Bảng~\ref{tab:kfeas} tóm tắt các lựa chọn.

<<TABK>>

\section{Chọn tỉ lệ validation}

Validation dùng để dừng sớm và chọn siêu tham số của các đối chứng (Ridge, GBDT). Bảng~\ref{tab:vtrade} cho thấy
đánh đổi: $v = 5\,\%$ chỉ còn 3 cell validation ở XJTU, quá ít để dừng sớm ổn định; $v = 15\,\%$ buộc
$K \ge 7$ (Mệnh đề~\ref{md:kmin}), mỗi fold XJTU chỉ còn khoảng 8 cell test và chi phí tăng 40\,\%.
$v = 10\,\%$ là điểm cân bằng: 6 cell validation ở bộ nhỏ nhất và vẫn giữ được $K = 5$. Cần lưu ý cell validation
có nhãn và nằm ngoài “ngân sách nhãn” của các mức 10--70\,\%; điều này giống nhau cho mọi mô hình.

<<TABV>>

\section{Phân tầng theo chế độ thí nghiệm}

Các cell trong cùng một bộ được thử ở nhiều chế độ khác nhau (Bảng~\ref{tab:datastats}). Nếu chia hoàn toàn ngẫu
nhiên, có thể một fold không có cell nào của một chế độ, trong khi fold khác lại toàn cell của chế độ đó. Thuật
toán chia dùng bộ đếm chạy liên tục qua các nhóm:

\begin{enumerate}
\item Xáo thứ tự cell trong từng nhóm (với hạt giống của lần lặp $r$); chọn ngẫu nhiên điểm bắt đầu $k_0$.
\item Duyệt các nhóm theo thứ tự cố định; cell thứ $i$ trong thứ tự duyệt toàn cục được gán vào fold
      $(k_0 + i) \bmod K$.
\item Ở fold $k$: test là các cell thuộc fold $k$; từ phần còn lại lấy $\operatorname{round}(vN)$ cell làm
      validation theo vòng tròn qua các nhóm; phần còn lại là train.
\item Ở mức nhãn $f$: lấy $\operatorname{round}(fN)$ cell train làm cell có nhãn theo một thứ tự vòng tròn cố
      định qua các nhóm; các cell train còn lại là cell không nhãn.
\end{enumerate}

\begin{menhde}[Phân hoạch và cân bằng]\label{md:partition}
Với mỗi lần lặp: (i) các tập test của $K$ fold rời nhau và hợp lại bằng toàn bộ $\mathcal{C}$, nên mỗi cell
được test đúng một lần; (ii) số cell test của hai fold bất kỳ lệch nhau không quá 1; (iii) trong mỗi nhóm thí
nghiệm $b$ có $n_b$ cell, số cell của nhóm trong mỗi fold bằng $\lfloor n_b/K\rfloor$ hoặc $\lceil n_b/K\rceil$.
\end{menhde}
\begin{proof}
(i) Mỗi cell nhận đúng một chỉ số fold, nên các tập test là các lớp tương đương của ánh xạ gán fold. (ii) Chỉ số
$i$ chạy từ $0$ tới $N-1$; trong $N$ số nguyên liên tiếp, số phần tử có cùng số dư modulo $K$ bằng
$\lfloor N/K\rfloor$ hoặc $\lceil N/K\rceil$. (iii) Các cell của nhóm $b$ chiếm $n_b$ chỉ số liên tiếp trong thứ
tự duyệt, nên lập luận của (ii) áp dụng với $n_b$ thay cho $N$.
\end{proof}

\begin{menhde}[Tập có nhãn lồng nhau]\label{md:nested}
Với cùng fold và lần lặp, nếu $f_1 < f_2$ thì tập cell có nhãn ở mức $f_1$ là tập con của tập ở mức $f_2$.
\end{menhde}
\begin{proof}
Thứ tự vòng tròn ở bước 4 chỉ phụ thuộc hạt giống, không phụ thuộc $f$; tập có nhãn là $\operatorname{round}(fN)$
phần tử đầu của cùng một thứ tự, và $\operatorname{round}(f_1N) \le \operatorname{round}(f_2N)$.
\end{proof}

Mệnh đề~\ref{md:nested} bảo đảm rằng khi tăng mức nhãn, khác biệt duy nhất là có thêm nhãn; đường cong theo
lượng nhãn vì vậy không bị lẫn với việc đổi sang một nhóm cell khác. Kiểm tra trên cả 5 lần lặp của 4 bộ
dữ liệu cho thấy <<STRAT>>. Hình~\ref{fig:split} minh hoạ cách chia thật của XJTU và Hình~\ref{fig:strat} là số cell
test của từng nhóm trong từng fold.

<<FIGSPLIT>>
<<FIGSTRAT>>

\section{Chọn số lần lặp \texorpdfstring{$R$}{R}}

Một lần kiểm định chéo cho mỗi cell đúng một dự đoán, nên kết quả vẫn phụ thuộc một cách xáo ngẫu nhiên. Lặp
lại với $R$ cách xáo khác nhau cho mỗi cell $R$ dự đoán từ $R$ mô hình khác nhau; sai số của cell được lấy
trung bình qua các lần lặp. Để chọn $R$, đồ án mô phỏng lấy lại trên kết quả thật: với mỗi $R' \in \{1, \dots, 5\}$,
lấy mọi tổ hợp $R'$ trong 5 lần lặp, tính lại tỉ số MAE giữa PINN-semi và MLP ở 30\,\% nhãn
(Bảng~\ref{tab:repeats}, Hình~\ref{fig:repeats}).

<<TABR>>
<<FIGREP>>

Có hai quan sát. Độ dao động của ước lượng giữa các tổ hợp giảm gần như $1/\sqrt{R}$: một lần lặp duy nhất cho
tỉ số dao động trong một dải đủ rộng để đổi kết luận ở XJTU và TJU, còn với 4--5 lần lặp ước lượng gần như cố
định. Ngược lại, độ rộng khoảng tin cậy chỉ giảm chậm rồi bão hoà, vì phần còn lại đến từ khác biệt thật giữa các
cell, không giảm được bằng cách lặp thêm. $R = 5$ vì vậy là điểm mà thêm lần lặp không còn đổi kết luận, trong
khi chi phí (25 lượt cho mỗi mô hình, mỗi mức nhãn, mỗi bộ) vẫn chạy được trên CPU.

\section{Ghép cặp và hạt giống}

Ở fold $k$ của lần lặp $r$, mọi mô hình dùng chung hạt giống $100r + k$ cho việc chọn cell có nhãn, khởi tạo
trọng số và lấy mẫu batch. Mọi mô hình vì vậy thấy đúng cùng cell train, cùng cell có nhãn, cùng cell
validation và được đánh giá trên đúng cùng cell test. Nhờ đó so sánh được theo từng cell (Mục~\ref{sec:stats}).

\section{Quy trình chống rò rỉ trong cùng bộ dữ liệu}\label{sec:noleak}

\begin{dinhnghia}[Rò rỉ]
Một giao thức bị rò rỉ nếu tham số của mô hình đã huấn luyện, hoặc bất kỳ quyết định tiền xử lý hay chọn mô
hình nào, phụ thuộc vào đặc trưng hoặc nhãn của cell test, hoặc vào thông tin không có tại thời điểm dự đoán
(chẳng hạn các chu kỳ tương lai của chính cell).
\end{dinhnghia}

Hình~\ref{fig:usage} liệt kê phần dữ liệu nào được dùng cho bước nào. Bốn quy tắc:
\begin{enumerate}
\item Normaliser chỉ fit trên đặc trưng của cell train (có nhãn và không nhãn).
\item Bộ lọc ngoại lai quyết định giữ hay bỏ chu kỳ $n$ chỉ bằng các chu kỳ trước $n$ của chính cell đó.
\item Biến thời gian $t = n/1000$ dùng hằng số chung, không chuẩn hoá theo tuổi thọ từng cell.
\item Dừng sớm và chọn siêu tham số chỉ dựa trên validation; tập test chỉ được dùng một lần để đánh giá.
\end{enumerate}

<<FIGUSAGE>>

\begin{menhde}[Ước lượng không chệch]\label{md:unbiased}
Giả sử các cell là mẫu độc lập cùng phân phối từ một tổng thể. Theo bốn quy tắc trên, tham số $\hat\theta$ của
mô hình là hàm của dữ liệu cell train và cell validation, độc lập với dữ liệu cell test. Khi đó
$\E[\mathrm{MAE}_\text{test} \mid \hat\theta]$ bằng sai số kỳ vọng của $\hat\theta$ trên một cell mới của tổng thể.
\end{menhde}
\begin{proof}
Bộ lọc của một cell test chỉ dùng dữ liệu của chính cell đó và không thay đổi khi dữ liệu của cell khác thay
đổi; normaliser, các hàm mất mát và tiêu chí dừng sớm không chứa dữ liệu cell test. Do đó $\hat\theta$ đo được
theo $\sigma$-đại số sinh bởi dữ liệu train và validation. Vì các cell độc lập, dữ liệu một cell test $c$ độc
lập với $\hat\theta$, nên $\E[\mathrm{MAE}_c \mid \hat\theta] = \E_{c' \sim \text{tổng thể}}[\mathrm{MAE}_{c'}(\hat\theta)]$.
Lấy trung bình qua các cell test được điều phải chứng minh.
\end{proof}

\section{Chuyển miền giữa các bộ dữ liệu}\label{sec:transfer}

Ở thí nghiệm chuyển miền, mô hình học trên bộ nguồn và được đánh giá trên bộ đích. Có hai thiết lập: không có
cell đích nào có nhãn ($k = 0$) và có $k = 3$ cell đích có nhãn. Quy trình:

\begin{enumerate}
\item Bộ nguồn được chia 70/15/15 theo cell; mọi cell train của bộ nguồn (70\,\% số cell) đều có nhãn. Phần
      validation và test của bộ nguồn không tham gia huấn luyện hay chọn mô hình.
\item Bộ đích được chia 70/15/15 theo cell với cùng hạt giống. Chỉ 15\,\% cell test của bộ đích được dùng để
      đánh giá; chúng không tham gia bất kỳ bước nào khác. Phần validation của bộ đích không được dùng.
\item $k$ cell có nhãn của bộ đích được lấy từ phần train của bộ đích; các cell train còn lại của bộ đích là cell
      không nhãn.
\item Normaliser của \emph{mọi} mô hình (MLP, GBDT và PINN-semi) fit trên cùng một tập: đặc trưng của cell train
      bộ nguồn và cell train bộ đích. Như vậy khác biệt giữa các mô hình chỉ còn là mô hình và hàm mất mát.
\item $\mathcal{L}_\text{data}$ dùng cell có nhãn của bộ nguồn và $k$ cell có nhãn của bộ đích; khi $k > 0$,
      một nửa mỗi batch lấy từ $k$ cell đích để chúng không bị dữ liệu nguồn lấn át. PINN-semi áp loss vật lý lên
      mọi cell train của cả hai bộ.
\item MLP và PINN-semi huấn luyện một số bước cố định (2\,500), không dừng sớm; số cây của GBDT chọn trên
      validation của bộ nguồn. Không mô hình nào được chọn theo bất kỳ nhãn nào của bộ đích.
\end{enumerate}

\begin{menhde}[Chuyển miền không rò rỉ]\label{md:transfer}
Với quy trình trên, tham số mô hình độc lập với đặc trưng và nhãn của các cell test bộ đích.
\end{menhde}
\begin{proof}
Cell test bộ đích không thuộc tập fit normaliser (bước 4), không thuộc tập của $\mathcal{L}_\text{data}$ hay loss
vật lý (bước 5), và không có bước chọn mô hình nào (bước 6). Áp dụng lập luận của Mệnh đề~\ref{md:unbiased}.
\end{proof}

Ở một thiết kế trước đó, MLP fit normaliser chỉ trên dữ liệu nguồn còn PINN fit trên cả nguồn và đích, nên một
phần khác biệt có thể đến từ normaliser chứ không phải từ vật lý. Bước 4 loại bỏ yếu tố này. Ngoài 4 cặp cùng hoá
học (HUST $\leftrightarrow$ MIT, XJTU $\leftrightarrow$ TJU), đồ án chạy đủ 12 cặp nguồn--đích có hướng, 5 hạt
giống cho mỗi cặp.

\section{Kiểm toán rò rỉ}\label{sec:audit}

Để đo định lượng tác động của các dạng rò rỉ thường gặp, cùng một mô hình (MLP, 70\,\% nhãn) được huấn luyện với
bốn cách xử lý sai, mỗi cách chỉ đổi đúng một yếu tố so với giao thức chuẩn:
\begin{itemize}
\item \emph{chia theo chu kỳ}: mỗi chu kỳ được gán ngẫu nhiên vào train/validation/test;
\item \emph{min--max theo cell}: đặc trưng và chỉ số chu kỳ được chuẩn hoá min--max theo toàn bộ vòng đời của
      từng cell, như trong mã nguồn PINN4SOH~\cite{wang2024}; chỉ số chu kỳ sau chuẩn hoá chính là “phần trăm tuổi
      thọ đã trôi qua”;
\item \emph{lọc dùng cả vòng đời}: bộ lọc $3\sigma$ dùng trung bình và độ lệch chuẩn của cả vòng đời cell;
\item \emph{chuẩn hoá gồm cả test}: normaliser fit trên cả đặc trưng cell validation và test.
\end{itemize}

\section{Chỉ số đánh giá}

Với $y_i$ là SOH thực đo và $\hat y_i$ là SOH dự đoán:
\begin{gather}
\mathrm{MAE} = \tfrac{1}{N}\textstyle\sum|\hat y_i - y_i|, \quad
\mathrm{RMSE} = \sqrt{\tfrac{1}{N}\textstyle\sum(\hat y_i - y_i)^2}, \quad
\mathrm{MAPE} = \tfrac{100\,\%}{N}\textstyle\sum\big|\tfrac{\hat y_i - y_i}{y_i}\big|, \\
R^2 = 1 - \frac{\sum(\hat y_i - y_i)^2}{\sum(y_i - \bar y)^2}, \qquad
\mathrm{MAE}_\text{cell} = \frac{1}{|\mathcal{C}_\text{test}|}\sum_{c}\frac{1}{n_c}\sum_{n}|\hat y_{c,n} - y_{c,n}|.
\end{gather}
Chỉ số chính là $\mathrm{MAE}_\text{cell}$: mỗi cell có một phiếu, nên cell có tuổi thọ dài không lấn át kết
quả. Ngoài ra đồ án dùng sai số ở giai đoạn cuối đời ($\mathrm{SOH} \le 0{,}9$), sai số thời điểm SOH chạm 0,85
(theo chu kỳ), và tổng mức tăng ngược của quỹ đạo dự đoán trên 100 chu kỳ.

\section{Phương pháp thống kê}\label{sec:stats}

\paragraph{Khoảng tin cậy.} Tỉ số $\rho = \overline{\mathrm{MAE}}_\text{PINN}/\overline{\mathrm{MAE}}_\text{đối chứng}$
có khoảng tin cậy 95\,\% theo bootstrap phân vị~\cite{efron1993}: lấy lại mẫu \emph{theo cell} (ghép cặp),
10\,000 lần.

\paragraph{Kiểm định theo cell.} Với $d_c = \mathrm{MAE}_c^\text{PINN} - \mathrm{MAE}_c^\text{đối chứng}$,
kiểm định dấu--hạng Wilcoxon~\cite{wilcoxon1945} hai phía cho giả thuyết trung vị của $d_c$ bằng 0. Kiểm định này
không giả định phân phối chuẩn.

\paragraph{Hiệu chỉnh đa so sánh.} Mỗi phép so sánh được thực hiện trên 4 bộ dữ liệu; $p$ được hiệu chỉnh bằng
phương pháp Holm~\cite{holm1979}: sắp $p_{(1)} \le \dots \le p_{(m)}$, đặt
$\tilde p_{(i)} = \max_{j \le i}\min\big(1, (m - j + 1)p_{(j)}\big)$.

\paragraph{Kiểm định bảo thủ ở mức fold.} Các tập train của các fold chồng lấp nhau nên MAE của các fold không
độc lập. Phép $t$ hiệu chỉnh của Nadeau và Bengio~\cite{nadeau2003,bouckaert2004} dùng
\begin{equation}
t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \frac{n_\text{test}}{n_\text{train}}\right)s_d^2}}, \qquad
\text{bậc tự do } J - 1,
\end{equation}
với $J = KR = 25$ hiệu theo fold. Hệ số $n_\text{test}/n_\text{train} \approx 0{,}29$ làm phương sai lớn hơn nhiều
so với phép $t$ thông thường, nên đây là kiểm định rất bảo thủ; kết luận được xem là vững khi qua được cả hai
kiểm định.

<<TABE>>
""", ED=f(1 / (1 - 0.3 ** 2), 3), MCERR=f(mcerr, 1), INB=inb, NPT=len(real), FOLD=f(fold, 0),
             FIGRS=fig('fig_thesis_rowsim', 'Mô phỏng chia theo chu kỳ. Trái: khoảng cách trung bình tới chu kỳ train gần '
                       'nhất theo $q$. Phải: sai số của phép chép nhãn trên quỹ đạo SOH thật (điểm) so với cận '
                       '$\\E[d]\\cdot\\overline{|\\Delta y|}$ (nét đứt); trục dọc thang log.', 'fig:rowsim'),
             TABB=tabB, TABK=tabK, TABV=tabV, TABR=tabR, TABE=tabE,
             STRAT=('mọi cell đều được test đúng một lần ở mỗi lần lặp và số cell của một nhóm trong các fold lệch '
                    'nhau không quá 1, đúng như Mệnh đề~\\ref{md:partition}') if strat_ok else 'có sai lệch so với mệnh đề',
             FIGSPLIT=fig('fig_thesis_split', 'Cách chia thật của bộ XJTU ở lần lặp đầu tiên, mức 30\\,\\% nhãn. Mỗi cột '
                          'là một cell, mỗi hàng là một fold; cell được xếp theo nhóm chế độ thí nghiệm.', 'fig:split'),
             FIGSTRAT=fig('fig_thesis_strat', 'Số cell test của từng nhóm thí nghiệm trong từng fold (lần lặp đầu tiên).',
                          'fig:strat'),
             FIGREP=fig('fig_thesis_repeats', 'Trái: tỉ số MAE PINN-semi/MLP ở 30\\,\\% nhãn theo số lần lặp; đường là '
                        'trung bình, dải là khoảng nhỏ nhất--lớn nhất giữa các tổ hợp lần lặp. Phải: độ rộng khoảng tin '
                        'cậy 95\\,\\%.', 'fig:repeats'),
             FIGUSAGE=fig('fig_thesis_usage', 'Phần dữ liệu nào được dùng cho bước nào (ô màu: có dùng). Trái: đánh giá '
                          'trong cùng bộ; phải: chuyển miền. Với MLP, cột loss vật lý không áp dụng.', 'fig:usage',
                          width=r'0.95\textwidth'))


# ═════════════════════════════════════════════════════════════ TÀI LIỆU THAM KHẢO
REFS = r"""
\begin{thebibliography}{99}\small
\bibitem{angelopoulos2023} A. N. Angelopoulos, S. Bates. Conformal prediction: a gentle introduction.
  \emph{Foundations and Trends in Machine Learning}, 16(4):494--591, 2023.
\bibitem{arlot2010} S. Arlot, A. Celisse. A survey of cross-validation procedures for model selection.
  \emph{Statistics Surveys}, 4:40--79, 2010.
\bibitem{attia2022knee} P. M. Attia, A. Bills, F. Brosa Planella, et al. Review---“Knees” in lithium-ion battery
  aging trajectories. \emph{Journal of The Electrochemical Society}, 169:060517, 2022.
\bibitem{aykol2021} M. Aykol, C. B. Gopal, A. Anapolsky, et al. Perspective---Combining physics and machine
  learning to predict battery lifetime. \emph{Journal of The Electrochemical Society}, 168:030525, 2021.
\bibitem{barlow1972} R. E. Barlow, D. J. Bartholomew, J. M. Bremner, H. D. Brunk. \emph{Statistical Inference under
  Order Restrictions}. Wiley, 1972.
\bibitem{birkl2017} C. R. Birkl, M. R. Roberts, E. McTurk, P. G. Bruce, D. A. Howey. Degradation diagnostics for
  lithium ion cells. \emph{Journal of Power Sources}, 341:373--386, 2017.
\bibitem{bouckaert2004} R. R. Bouckaert, E. Frank. Evaluating the replicability of significance tests for comparing
  learning algorithms. \emph{Advances in Knowledge Discovery and Data Mining (PAKDD)}, 3--12, 2004.
\bibitem{braun2026} C. Braun, J. Raible, M. F. Huber. Physics-informed machine learning in prognostics and health
  management: a systematic literature review. arXiv:2608.10047, 2026.
\bibitem{breiman1996} L. Breiman. Bagging predictors. \emph{Machine Learning}, 24(2):123--140, 1996.
\bibitem{broussely2001} M. Broussely, S. Herreyre, P. Biensan, et al. Aging mechanism in Li ion cells and calendar
  life predictions. \emph{Journal of Power Sources}, 97--98:13--21, 2001.
\bibitem{cleveland1979} W. S. Cleveland. Robust locally weighted regression and smoothing scatterplots.
  \emph{Journal of the American Statistical Association}, 74(368):829--836, 1979.
\bibitem{dietterich1998} T. G. Dietterich. Approximate statistical tests for comparing supervised classification
  learning algorithms. \emph{Neural Computation}, 10(7):1895--1923, 1998.
\bibitem{doyle1993} M. Doyle, T. F. Fuller, J. Newman. Modeling of galvanostatic charge and discharge of the
  lithium/polymer/insertion cell. \emph{Journal of The Electrochemical Society}, 140(6):1526--1533, 1993.
\bibitem{efron1993} B. Efron, R. J. Tibshirani. \emph{An Introduction to the Bootstrap}. Chapman \& Hall, 1993.
\bibitem{fermin2020} P. Fermín-Cueto, E. McTurk, M. Allerhand, et al. Identification and machine learning
  prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells. \emph{Energy and AI},
  1:100006, 2020.
\bibitem{friedman2001} J. H. Friedman. Greedy function approximation: a gradient boosting machine.
  \emph{Annals of Statistics}, 29(5):1189--1232, 2001.
\bibitem{grinsztajn2022} L. Grinsztajn, E. Oyallon, G. Varoquaux. Why do tree-based models still outperform deep
  learning on typical tabular data? \emph{Advances in Neural Information Processing Systems}, 35, 2022.
\bibitem{hinton2015} G. Hinton, O. Vinyals, J. Dean. Distilling the knowledge in a neural network.
  arXiv:1503.02531, 2015.
\bibitem{he2011} W. He, N. Williard, M. Osterman, M. Pecht. Prognostics of lithium-ion batteries based on
  Dempster--Shafer theory and the Bayesian Monte Carlo method. \emph{Journal of Power Sources}, 196(23):10314--10321,
  2011.
\bibitem{olalde2024samba} J. I. Olalde-Verano, S. Kirch, C. Pérez-Molina, S. Martín. SambaMixer: State of health
  prediction of Li-ion batteries using Mamba state space models. \emph{IEEE Access}, 2024.
  doi:10.1109/ACCESS.2024.3524321.
\bibitem{holm1979} S. Holm. A simple sequentially rejective multiple test procedure. \emph{Scandinavian Journal of
  Statistics}, 6(2):65--70, 1979.
\bibitem{hu2020joule} X. Hu, L. Xu, X. Lin, M. Pecht. Battery lifetime prognostics. \emph{Joule}, 4(2):310--346,
  2020.
\bibitem{kapoor2023} S. Kapoor, A. Narayanan. Leakage and the reproducibility crisis in machine-learning-based
  science. \emph{Patterns}, 4(9):100804, 2023.
\bibitem{karniadakis2021} G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang.
  Physics-informed machine learning. \emph{Nature Reviews Physics}, 3:422--440, 2021.
\bibitem{ke2017} G. Ke, Q. Meng, T. Finley, et al. LightGBM: a highly efficient gradient boosting decision tree.
  \emph{Advances in Neural Information Processing Systems}, 30, 2017.
\bibitem{krishnapriyan2021} A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, M. W. Mahoney. Characterizing
  possible failure modes in physics-informed neural networks. \emph{Advances in Neural Information Processing
  Systems}, 34, 2021.
\bibitem{lakshminarayanan2017} B. Lakshminarayanan, A. Pritzel, C. Blundell. Simple and scalable predictive
  uncertainty estimation using deep ensembles. \emph{Advances in Neural Information Processing Systems}, 30, 2017.
\bibitem{lee2013} D.-H. Lee. Pseudo-label: the simple and efficient semi-supervised learning method for deep neural
  networks. \emph{ICML Workshop on Challenges in Representation Learning}, 2013.
\bibitem{lei2018} J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. Distribution-free predictive
  inference for regression. \emph{Journal of the American Statistical Association}, 113(523):1094--1111, 2018.
\bibitem{li2018adabn} Y. Li, N. Wang, J. Shi, X. Hou, J. Liu. Adaptive batch normalization for practical domain
  adaptation. \emph{Pattern Recognition}, 80:109--117, 2018.
\bibitem{loshchilov2019} I. Loshchilov, F. Hutter. Decoupled weight decay regularization. \emph{International
  Conference on Learning Representations}, 2019.
\bibitem{lu2023} J. Lu, R. Xiong, J. Tian, C. Wang, F. Sun. Deep learning to estimate lithium-ion battery state of
  health without additional degradation experiments. \emph{Nature Communications}, 14:2760, 2023.
\bibitem{ma2022} G. Ma, S. Xu, B. Jiang, et al. Real-time personalized health status prediction of lithium-ion
  batteries using deep transfer learning. \emph{Energy \& Environmental Science}, 15:4083--4094, 2022.
\bibitem{nadeau2003} C. Nadeau, Y. Bengio. Inference for the generalization error. \emph{Machine Learning},
  52:239--281, 2003.
\bibitem{navidi2024} S. Navidi, A. Thelen, T. Li, C. Hu. Physics-informed machine learning for battery degradation
  diagnostics: a comparison of state-of-the-art methods. arXiv:2404.04429, 2024.
\bibitem{pinson2013} M. B. Pinson, M. Z. Bazant. Theory of SEI formation in rechargeable batteries: capacity fade,
  accelerated aging and lifetime prediction. \emph{Journal of The Electrochemical Society}, 160(2):A243--A250, 2013.
\bibitem{plett2015} G. L. Plett. \emph{Battery Management Systems, Volume I: Battery Modeling}. Artech House, 2015.
\bibitem{raissi2019} M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks: a deep learning
  framework for solving forward and inverse problems involving nonlinear partial differential equations.
  \emph{Journal of Computational Physics}, 378:686--707, 2019.
\bibitem{rasmussen2006} C. E. Rasmussen, C. K. I. Williams. \emph{Gaussian Processes for Machine Learning}.
  MIT Press, 2006.
\bibitem{richardson2017} R. R. Richardson, M. A. Osborne, D. A. Howey. Gaussian process regression for forecasting
  battery state of health. \emph{Journal of Power Sources}, 357:209--219, 2017.
\bibitem{roman2021} D. Roman, S. Saxena, V. Robu, M. Pecht, D. Flynn. Machine learning pipeline for battery
  state-of-health estimation. \emph{Nature Machine Intelligence}, 3:447--456, 2021.
\bibitem{saxena2008} A. Saxena, J. Celaya, E. Balaban, et al. Metrics for evaluating performance of prognostic
  techniques. In \emph{International Conference on Prognostics and Health Management}, IEEE, 2008.
\bibitem{saha2007nasa} B. Saha, K. Goebel. Battery data set. \emph{NASA Ames Prognostics Data Repository}, NASA
  Ames Research Center, Moffett Field, CA, 2007.
\bibitem{schmalstieg2014} J. Schmalstieg, S. Käbitz, M. Ecker, D. U. Sauer. A holistic aging model for
  Li(NiMnCo)O$_2$ based 18650 lithium-ion batteries. \emph{Journal of Power Sources}, 257:325--334, 2014.
\bibitem{severson2019} K. A. Severson, P. M. Attia, N. Jin, et al. Data-driven prediction of battery cycle life before
  capacity degradation. \emph{Nature Energy}, 4:383--391, 2019.
\bibitem{sulzer2021} V. Sulzer, P. Mohtat, A. Aitio, et al. The challenge and opportunity of battery lifetime
  prediction from field data. \emph{Joule}, 5(8):1934--1955, 2021.
\bibitem{tarvainen2017} A. Tarvainen, H. Valpola. Mean teachers are better role models: weight-averaged consistency
  targets improve semi-supervised deep learning results. \emph{Advances in Neural Information Processing Systems},
  30, 2017.
\bibitem{thelen2024} A. Thelen, X. Huan, N. Paulson, S. Onori, Z. Hu, C. Hu. Probabilistic machine learning for
  battery health diagnostics and prognostics---review and perspectives. \emph{npj Materials Sustainability}, 2:14,
  2024.
\bibitem{varma2006} S. Varma, R. Simon. Bias in error estimation when using cross-validation for model selection.
  \emph{BMC Bioinformatics}, 7:91, 2006.
\bibitem{vovk2005} V. Vovk, A. Gammerman, G. Shafer. \emph{Algorithmic Learning in a Random World}. Springer, 2005.
\bibitem{waldmann2014} T. Waldmann, M. Wilka, M. Kasper, M. Fleischhammer, M. Wohlfahrt-Mehrens. Temperature
  dependent ageing mechanisms in lithium-ion batteries -- a post-mortem study. \emph{Journal of Power Sources},
  262:129--135, 2014.
\bibitem{wang2011lfp} J. Wang, P. Liu, J. Hicks-Garner, et al. Cycle-life model for graphite-LiFePO$_4$ cells.
  \emph{Journal of Power Sources}, 196(8):3942--3948, 2011.
\bibitem{wang2021grad} S. Wang, Y. Teng, P. Perdikaris. Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks. \emph{SIAM Journal on Scientific Computing}, 43(5):A3055--A3081, 2021.
\bibitem{wang2024} F. Wang, Z. Zhai, Z. Zhao, Y. Di, X. Chen. Physics-informed neural network for lithium-ion
  battery degradation stable modeling and prognosis. \emph{Nature Communications}, 15:4332, 2024.
\bibitem{wen2023} P. Wen, Z.-S. Ye, Y. Li, S. Chen, P. Xie, S. Zhao. Physics-informed neural networks for
  prognostics and health management of lithium-ion batteries. arXiv:2301.00776, 2023.
\bibitem{wilcoxon1945} F. Wilcoxon. Individual comparisons by ranking methods. \emph{Biometrics Bulletin},
  1(6):80--83, 1945.
\bibitem{wolpert1992} D. H. Wolpert. Stacked generalization. \emph{Neural Networks}, 5(2):241--259, 1992.
\bibitem{zhang2018lstm} Y. Zhang, R. Xiong, H. He, M. G. Pecht. Long short-term memory recurrent neural network for
  remaining useful life prediction of lithium-ion batteries. \emph{IEEE Transactions on Vehicular Technology},
  67(7):5695--5705, 2018.
\bibitem{zhu2022} J. Zhu, Y. Wang, Y. Huang, et al. Data-driven capacity estimation of commercial lithium-ion
  batteries from voltage relaxation. \emph{Nature Communications}, 13:2261, 2022.
\end{thebibliography}
"""


# ═════════════════════════════════════════════════════════════ CHƯƠNG 6
def _best(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return min(v) if v else None


def ch6(d, ctx):
    cur, cmp_, eq = d['thesis_curve'], d['thesis_compare'], d['thesis_equivalence']
    MODS = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']
    FR = [0.1, 0.3, 0.5, 0.7]
    val = lambda m, fr, ds: (lambda g: float(g.MAE_cell.iloc[0]) if len(g) else float('nan'))(
        cur[(cur.model == m) & (cur.frac == fr) & (cur.bo == ds)])

    # ---------- 6.1 rò rỉ
    lk = d['thesis_leakage'].set_index('bo')
    sets = [('chuan', 'Chia theo cell (chuẩn)'), ('normtest', 'Chuẩn hoá gồm cả test'), ('life3s', 'Lọc dùng cả vòng đời'),
            ('minmax', 'Min--max theo cell'), ('row', 'Chia theo chu kỳ'), ('noi_suy_hang', 'Chia theo chu kỳ, chép nhãn')]
    lrows = []
    for key, name in sets:
        if key not in lk.columns:
            continue
        row = [name]
        for ds in DS:
            v = lk.loc[ds, key]
            rr = '' if key == 'chuan' else f' ({f(v / lk.loc[ds, "chuan"], 2)})'
            row.append(m3(v, 2) + rr)
        lrows.append(row)
    tabL = table(['Thiết lập'] + DS, lrows, 'lrrrr',
                 'Kiểm toán rò rỉ: MAE theo cell (×$10^{-3}$) của MLP với 70\\,\\% nhãn, trung bình 2 lần lặp × 5 fold; '
                 'trong ngoặc là tỉ số so với giao thức chuẩn.', 'tab:leak')
    rmin = min(lk.loc[ds, 'row'] / lk.loc[ds, 'chuan'] for ds in DS) if 'row' in lk.columns else float('nan')
    rmax = max(lk.loc[ds, 'row'] / lk.loc[ds, 'chuan'] for ds in DS) if 'row' in lk.columns else float('nan')
    mm = [lk.loc[ds, 'minmax'] / lk.loc[ds, 'chuan'] for ds in DS] if 'minmax' in lk.columns else [float('nan')]
    nt = [lk.loc[ds, 'normtest'] / lk.loc[ds, 'chuan'] for ds in DS] if 'normtest' in lk.columns else [float('nan')]
    lf = [lk.loc[ds, 'life3s'] / lk.loc[ds, 'chuan'] for ds in DS] if 'life3s' in lk.columns else [float('nan')]

    # ---------- 6.2 bảng lớn theo lượng nhãn
    brows = []
    for ds in DS:
        for i, m in enumerate(MODS):
            cells = []
            for fr in FR:
                v = val(m, fr, ds)
                best = _best([val(mm_, fr, ds) for mm_ in MODS])
                s = m3(v, 2) if not math.isnan(v) else '--'
                if best is not None and not math.isnan(v) and abs(v - best) < 1e-12:
                    s = r'\textbf{' + s + '}'
                cells.append(s)
            brows.append([ds if i == 0 else '', LBL[m]] + cells)
        brows.append([r'\addlinespace[2pt]'] + [''] * 5) if ds != DS[-1] else None
    brows = [r for r in brows if r is not None]
    tabBig = table(['Bộ', 'Mô hình', '10\\,\\%', '30\\,\\%', '50\\,\\%', '70\\,\\%'],
                   [r if r[0] != r'\addlinespace[2pt]' else [r'\addlinespace[2pt]'] for r in brows], 'llrrrr',
                   'MAE theo cell (×$10^{-3}$) của mọi mô hình theo lượng nhãn (phần trăm số cell có nhãn); in đậm là '
                   'thấp nhất trong cột của mỗi bộ. Trung bình 5 lần lặp × 5 fold; mỗi cell được test 5 lần.',
                   'tab:big', size=r'\small')
    tabBig = tabBig.replace(r'\addlinespace[2pt] \\', r'\addlinespace[3pt]')

    # ---------- 6.3 kiểm định PINN-semi so với MLP và GBDT
    crows = []
    for ds in DS:
        for fr in FR:
            a = cmp_[(cmp_.doi_chung == 'mlp') & (cmp_.frac == fr) & (cmp_.bo == ds)]
            b = cmp_[(cmp_.doi_chung == 'gbdt') & (cmp_.frac == fr) & (cmp_.bo == ds)]
            def cellfmt(g):
                if g.empty:
                    return ['--', '--', '--']
                g = g.iloc[0]
                star = r'$^{*}$' if g.p_holm < 0.05 else ''
                star2 = r'$^{\dagger}$' if g.p_nb_holm < 0.05 else ''
                return [f'{f(g.ti_so, 2)} [{f(g.lo, 2)}; {f(g.hi, 2)}]{star}{star2}', pc(g.cell_pinn_tot_hon), pv(g.p_holm)]
            crows.append([ds if fr == 0.1 else '', pc(fr)] + cellfmt(a) + cellfmt(b))
    tabC = table(['Bộ', 'Nhãn', 'so với MLP: tỉ số [KTC]', 'cell tốt hơn', '$p_\\text{Holm}$',
                  'so với GBDT: tỉ số [KTC]', 'cell tốt hơn', '$p_\\text{Holm}$'], crows, 'llrrrrrr',
                 'Tỉ số MAE theo cell của PINN-semi so với MLP và GBDT ở từng mức nhãn (nhỏ hơn 1: PINN-semi tốt hơn). '
                 '$^{*}$: $p_\\text{Holm} < 0{,}05$ theo cell; $^{\\dagger}$: còn có ý nghĩa với kiểm định bảo thủ mức fold.',
                 'tab:cmp', size=r'\scriptsize')

    # ---------- 6.4 lượng nhãn tương đương
    def feq(x):
        if x == -1:
            return '$\\le 10\\,\\%$'
        if math.isinf(x):
            return '$> 70\\,\\%$'
        return pc(x)
    erows = []
    for ds in DS:
        row = [ds]
        for fr in [0.1, 0.3]:
            for who in ['mlp', 'tot_nhat_co_nhan']:
                g = eq[(eq.bo == ds) & (eq.frac_pinn == fr) & (eq.doi_chung == who)]
                row.append(feq(float(g.frac_tuong_duong.iloc[0])) if len(g) else '--')
        erows.append(row)
    tabEq = table(['Bộ', 'PINN 10\\,\\% ≈ MLP', 'PINN 10\\,\\% ≈ tốt nhất', 'PINN 30\\,\\% ≈ MLP',
                   'PINN 30\\,\\% ≈ tốt nhất'], erows, 'lrrrr',
                  'Lượng nhãn tương đương: mô hình học có nhãn cần bao nhiêu phần trăm cell có nhãn để đạt MAE của '
                  'PINN-semi ở 10\\,\\% và 30\\,\\% nhãn (“tốt nhất”: mô hình học có nhãn tốt nhất ở từng mức, '
                  'trong Ridge, GPR, GBDT, MLP). Nội suy log-tuyến tính giữa các mức đo.', 'tab:equiv')

    # ---------- 6.5 nguồn gốc lợi ích ở 30 %
    srows = []
    for m in ['mlp', 'pinn_sup', 'mono_only', 'euler', 'pinn_bb', 'pinn_semi']:
        row = [LBL[m]]
        for ds in DS:
            v, base = val(m, 0.3, ds), val('mlp', 0.3, ds)
            row.append(m3(v, 2) + ('' if m == 'mlp' else f' ({f(v / base, 2)})'))
        srows.append(row)
    tabS = table(['Mô hình (30\\,\\% nhãn)'] + DS, srows, 'lrrrr',
                 'Tách nguồn gốc lợi ích ở 30\\,\\% nhãn: MAE theo cell (×$10^{-3}$), trong ngoặc là tỉ số so với MLP.',
                 'tab:source')

    # ---------- 6.6 chỉ số mở rộng
    pmx = d['pred_metrics']
    prow = []
    for ds in DS:
        for arm, name in [('A', 'MLP 30\\,\\%'), ('C', 'PINN-semi 30\\,\\%'), ('F', 'MLP 70\\,\\%'), ('G', 'PINN-semi 70\\,\\%')]:
            g = pmx[(pmx.bo == ds) & (pmx.arm == arm)]
            if g.empty:
                continue
            g = g.iloc[0]
            prow.append([ds if arm == 'A' else '', name, f(100 * g.MAE, 2), f(100 * g.RMSE, 2), f(g.MAPE, 2),
                         f(g.R2, 3), f(g.R2_tu10, 3), f(100 * g.MaxAE, 1)])
    tabP = table(['Bộ', 'Mô hình', 'MAE (\\%)', 'RMSE (\\%)', 'MAPE (\\%)', '$R^2$', '$R^2$ từ chu kỳ 10', 'MaxAE (\\%)'],
                 prow, 'llrrrrrr', 'Chỉ số gộp mọi chu kỳ trên tập test (MAE, RMSE, MaxAE theo điểm phần trăm SOH).',
                 'tab:pred', size=r'\small')

    # ---------- 6.7 chuyển miền
    tr = d['thesis_transfer']
    tmods = [m for m in ['mlp', 'gbdt', 'gpr', 'pinn_semi'] if m in set(tr.model)]
    trows = []
    for s_ in DS:
        for t_ in DS:
            if s_ == t_:
                continue
            g = lambda m, k: tr[(tr.nguon == s_) & (tr.dich == t_) & (tr.model == m) & (tr.k == k)]
            vals = []
            for k in [0, 3]:
                xs = {m: (float(g(m, k)['mean'].iloc[0]) if len(g(m, k)) else NAN) for m in tmods}
                bk = _best(list(xs.values()))
                for m in tmods:
                    sv = f(xs[m], 3)
                    vals.append(r'\textbf{' + sv + '}' if _ok(xs[m]) and xs[m] == bk else sv)
            same = CHEM[s_] == CHEM[t_] or {s_, t_} <= {'XJTU', 'TJU'}
            trows.append([f'{s_} → {t_}', 'có' if same else 'không'] + vals)
    short = {'mlp': 'MLP', 'gbdt': 'GBDT', 'gpr': 'GPR', 'pinn_semi': 'PINN'}
    tabT = table(['Nguồn → đích', 'Cùng họ'] + [f'{short[m]}, $k{{=}}{k}$' for k in [0, 3] for m in tmods], trows,
                 'll' + 'r' * (2 * len(tmods)),
                 'Chuyển miền: MAE theo cell trên tập test của bộ đích, trung bình 5 hạt giống; in đậm là thấp nhất trong '
                 'cùng một $k$. Mọi mô hình dùng cùng cách chia và cùng normaliser.', 'tab:transfer', size=r'\small')

    # ---------- 6.8 tham số vật lý
    php = d.get('E13_physics_params')
    phrows = []
    if php is not None:
        for ds in DS:
            try:
                lam_m, lam_s = php.loc[('C', ds), ('lam', 'mean')], php.loc[('C', ds), ('lam', 'std')]
                ea_m, ea_s = php.loc[('C', ds), ('Ea', 'mean')], php.loc[('C', ds), ('Ea', 'std')]
                phrows.append([ds, f'{f(lam_m, 2)} ± {f(lam_s, 2)}', f'{f(ea_m, 1)} ± {f(ea_s, 1)}',
                               'có' if ds == 'TJU' else 'không'])
            except KeyError:
                pass
    tabPh = table(['Bộ', '$\\lambda$', '$E_a$ (kJ/mol)', 'Nhiều mức nhiệt độ'], phrows, 'lrrc',
                  'Tham số vật lý học được của PINN-semi ở 30\\,\\% nhãn (trung bình ± độ lệch chuẩn qua 25 lượt). Giá trị '
                  'khởi tạo: $\\lambda = 1$, $E_a = 30$\\,kJ/mol.', 'tab:phys')

    ctx['_tables'] = dict(L=tabL, BIG=tabBig, C=tabC, EQ=tabEq, S=tabS, P=tabP, T=tabT, PH=tabPh)
    ctx['_leak'] = dict(rmin=rmin, rmax=rmax, mm=mm, nt=nt, lf=lf)
    return ctx


FRS = [0.1, 0.3, 0.5, 0.7]
NAN = float('nan')


def _sig(cmp_, other, fr, better=True):
    g = cmp_[(cmp_.doi_chung == other) & (cmp_.frac == fr)]
    g = g[(g.p_holm < 0.05) & ((g.ti_so < 1) if better else (g.ti_so > 1))]
    return [ds for ds in DS if ds in set(g.bo)]


def _lst(xs):
    xs = list(xs)
    if not xs:
        return 'không bộ nào'
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]


def _tren(xs):
    """'trên XJTU và MIT' / 'trên cả bốn bộ'; rỗng nếu danh sách rỗng."""
    xs = list(xs)
    if not xs:
        return ''
    return 'trên cả bốn bộ' if len(xs) == len(DS) else 'trên ' + _lst(xs)


def _ok(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _pairs(dct, n=2):
    return ', '.join(f'{ds} {f(dct[ds], n)}' for ds in DS if ds in dct and _ok(dct[ds]))


def _rng(vals, n=2):
    v = [x for x in vals if _ok(x)]
    if not v:
        return '--'
    a, b = f(min(v), n), f(max(v), n)
    return a if a == b else f'{a}--{b}'


def _gm(vals):
    v = [x for x in vals if _ok(x) and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else NAN


def _levels(frs):
    """[0.1, 0.3, 0.5] -> '10--50 %'; cả bốn mức -> 'cả bốn mức nhãn'."""
    if len(frs) == len(FRS):
        return 'cả bốn mức nhãn'
    out, run = [], [frs[0]]
    for x in frs[1:]:
        if FRS.index(x) == FRS.index(run[-1]) + 1:
            run.append(x)
        else:
            out.append(run); run = [x]
    out.append(run)
    return ', '.join(pc(r[0]) if len(r) == 1 else f'{int(round(100 * r[0]))}--{pc(r[-1])}' for r in out)


def _n_labelled(fr):
    import glob, json
    out = {}
    for ds in DS:
        fs = sorted(glob.glob(f'runs/E13/*_{ds}_mlp_f{fr}_*cv0-0of5.json'))
        if fs:
            out[ds] = json.load(open(fs[0]))['n_cells']['train_labelled']
    return out


def facts(d, ctx):
    """Mọi nhận định định lượng trong Chương 6, 7 và phần tóm tắt, tính một lần từ results/."""
    cur, cmp_ = d['thesis_curve'], d['thesis_compare']
    F = {}
    val = lambda m, fr, ds: (lambda g: float(g.MAE_cell.iloc[0]) if len(g) else NAN)(
        cur[(cur.model == m) & (cur.frac == fr) & (cur.bo == ds)])
    ratio = lambda other, fr, ds: (lambda g: float(g.ti_so.iloc[0]) if len(g) else NAN)(
        cmp_[(cmp_.doi_chung == other) & (cmp_.frac == fr) & (cmp_.bo == ds)])
    F['val'] = val
    F['r'] = {fr: {ds: ratio('mlp', fr, ds) for ds in DS} for fr in FRS}
    F['sig'] = {fr: _sig(cmp_, 'mlp', fr) for fr in FRS}
    F['worse'] = {fr: _sig(cmp_, 'mlp', fr, better=False) for fr in FRS}
    F['nb'] = {fr: [ds for ds in DS if ((cmp_.doi_chung == 'mlp') & (cmp_.frac == fr) & (cmp_.bo == ds) &
                                        (cmp_.p_nb_holm < 0.05) & (cmp_.ti_so < 1)).any()] for fr in FRS}
    F['gm'] = {fr: _gm(F['r'][fr].values()) for fr in FRS}
    F['nlab10'] = _n_labelled(0.1)

    # mô hình tốt nhất (PINN-sup ở 70 % trùng PINN-semi nên không tính riêng)
    MODS = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']
    best = {}
    for ds in DS:
        for fr in FRS:
            c = {m: val(m, fr, ds) for m in MODS if not (m == 'pinn_sup' and fr == 0.7)}
            c = {m: v for m, v in c.items() if _ok(v)}
            best[(ds, fr)] = min(c, key=c.get) if c else None
    F['best'] = best
    parts = []
    for ds in DS:
        groups = {}
        for fr in FRS:
            if best[(ds, fr)]:
                groups.setdefault(best[(ds, fr)], []).append(fr)
        parts.append(f'{ds}: ' + ', '.join(f'{LBL[m]} ở {_levels(frs)}' for m, frs in groups.items()))
    F['best_txt'] = '; '.join(parts)
    F['gb_best30'] = [ds for ds in DS if best[(ds, 0.3)] == 'gbdt']

    # so với các mô hình học có nhãn
    def wins(other, frs=FRS):
        pairs = [(ds, fr) for ds in DS for fr in frs if _ok(val(other, fr, ds)) and _ok(val('pinn_semi', fr, ds))]
        return sum(val('pinn_semi', fr, ds) < val(other, fr, ds) for ds, fr in pairs), len(pairs)
    F['wins'] = {m: wins(m) for m in ['mlp', 'ridge', 'gpr', 'gbdt']}
    F['gb_sig_better'] = {fr: _sig(cmp_, 'gbdt', fr, better=False) for fr in FRS}
    F['pinn_sig_better_gb'] = {fr: _sig(cmp_, 'gbdt', fr) for fr in FRS}
    F['n_gb_sig'] = sum(len(v) for v in F['gb_sig_better'].values())
    F['r_gb10'] = {ds: ratio('gbdt', 0.1, ds) for ds in DS}
    F['ridge10_better'] = [ds for ds in DS if val('ridge', 0.1, ds) < val('mlp', 0.1, ds)]
    F['ridge_gain'] = [val('ridge', 0.7, ds) / val('ridge', 0.1, ds) for ds in DS]
    F['mlp_gain'] = [val('mlp', 0.7, ds) / val('mlp', 0.1, ds) for ds in DS]
    gp = [(ds, fr) for ds in DS for fr in FRS if _ok(val('gpr', fr, ds)) and _ok(val('gbdt', fr, ds))]
    F['gpr_vs_gb'] = (sum(val('gbdt', fr, ds) < val('gpr', fr, ds) for ds, fr in gp), len(gp))

    # nguồn gốc lợi ích
    F['sup10'] = {ds: val('pinn_sup', 0.1, ds) / val('mlp', 0.1, ds) for ds in DS}
    F['semisup10'] = {ds: val('pinn_semi', 0.1, ds) / val('pinn_sup', 0.1, ds) for ds in DS}
    F['semisup30'] = {ds: val('pinn_semi', 0.3, ds) / val('pinn_sup', 0.3, ds) for ds in DS}
    share = [math.log(F['semisup10'][ds]) / math.log(F['r'][0.1][ds]) for ds in DS
             if _ok(F['semisup10'][ds]) and F['r'][0.1][ds] < 0.97]
    F['share_unl10'] = share
    F['mono_semi'] = {ds: val('mono_only', 0.3, ds) / val('pinn_semi', 0.3, ds) for ds in DS}
    F['share_mono'] = [math.log(val('mono_only', 0.3, ds) / val('mlp', 0.3, ds)) / math.log(F['r'][0.3][ds])
                       for ds in DS if F['r'][0.3][ds] < 0.97]
    F['euler_semi'] = {ds: val('euler', 0.3, ds) / val('pinn_semi', 0.3, ds) for ds in DS}
    F['bb30'] = {ds: val('pinn_bb', 0.3, ds) / val('pinn_semi', 0.3, ds) for ds in DS}
    F['bb10'] = {ds: val('pinn_bb', 0.1, ds) / val('pinn_semi', 0.1, ds) for ds in DS}
    F['bb_worse'] = [ds for ds in DS if _ok(F['bb30'][ds]) and F['bb30'][ds] > 1.02]
    F['bb_sig_semi'] = {fr: _sig(cmp_, 'pinn_bb', fr) for fr in FRS}                 # PINN-semi tốt hơn hộp đen
    F['bb_sig_bb'] = {fr: _sig(cmp_, 'pinn_bb', fr, better=False) for fr in FRS}     # hộp đen tốt hơn
    F['bb_nsemi'] = sum(len(v) for v in F['bb_sig_semi'].values())
    F['bb_nsig'] = F['bb_nsemi'] + sum(len(v) for v in F['bb_sig_bb'].values())
    cands = [(ds, fr, ratio('pinn_bb', fr, ds)) for ds in DS for fr in FRS if _ok(ratio('pinn_bb', fr, ds))]
    F['bb_max'] = min(cands, key=lambda x: x[2]) if cands else ('--', 0.1, NAN)       # tỉ số PINN-semi / hộp đen nhỏ nhất
    F['bb_mono_gap'] = max((abs(F['bb30'][ds] - F['mono_semi'][ds]) for ds in DS
                            if _ok(F['bb30'][ds]) and _ok(F['mono_semi'][ds])), default=NAN)
    F['bb_have'] = [ds for ds in DS if _ok(F['bb30'][ds])]

    # chuyển miền
    tr = d['thesis_transfer']
    fam = lambda s_, t_: (CHEM[s_] == CHEM[t_]) or ({s_, t_} <= {'XJTU', 'TJU'})
    def tval(s_, t_, k, m):
        x = tr[(tr.nguon == s_) & (tr.dich == t_) & (tr.k == k) & (tr.model == m)]['mean'] if len(tr) else []
        return float(x.iloc[0]) if len(x) else NAN
    pairs = [(s_, t_) for s_ in DS for t_ in DS if s_ != t_]
    T_ = {}
    for same in (True, False):
        pp = [p for p in pairs if fam(*p) == same]
        for k in (0, 3):
            T_[(same, k, 'ratio')] = _gm([tval(*p, k, 'pinn_semi') / tval(*p, k, 'mlp') for p in pp])
            T_[(same, k, 'pinn')] = _gm([tval(*p, k, 'pinn_semi') for p in pp])
            T_[(same, k, 'mlp')] = _gm([tval(*p, k, 'mlp') for p in pp])
    F['tr'] = T_
    ni, lfp = {'XJTU', 'TJU'}, {'MIT', 'HUST'}
    g_ni_lfp = _gm([tval(s_, t_, 0, 'pinn_semi') for s_, t_ in pairs if s_ in ni and t_ in lfp])
    g_lfp_ni = _gm([tval(s_, t_, 0, 'pinn_semi') for s_, t_ in pairs if s_ in lfp and t_ in ni])
    F['tr_dir'] = ''
    if _ok(g_ni_lfp) and _ok(g_lfp_ni) and max(g_ni_lfp, g_lfp_ni) > 1.5 * min(g_ni_lfp, g_lfp_ni):
        a_, b_ = ('họ niken sang LFP', 'LFP sang họ niken') if g_ni_lfp > g_lfp_ni else ('LFP sang họ niken',
                                                                                         'họ niken sang LFP')
        F['tr_dir'] = (f' Chiều chuyển cũng quan trọng: từ {a_} khó hơn (MAE trung bình nhân '
                       f'{f(max(g_ni_lfp, g_lfp_ni), 3)}) so với từ {b_} ({f(min(g_ni_lfp, g_lfp_ni), 3)}).')
    F['tr_n'] = sum(_ok(tval(*p, 0, 'pinn_semi')) for p in pairs)
    for k in (0, 3):
        both = [p for p in pairs if _ok(tval(*p, k, 'gbdt')) and _ok(tval(*p, k, 'pinn_semi'))]
        F[f'tr_gb{k}_n'] = len(both)
        F[f'tr_gb{k}_win'] = sum(tval(*p, k, 'pinn_semi') < tval(*p, k, 'gbdt') for p in both)
        F[f'tr_gb{k}_gm'] = _gm([tval(*p, k, 'pinn_semi') / tval(*p, k, 'gbdt') for p in both])
        F[f'tr_gb{k}_rng'] = [tval(*p, k, 'gbdt') for p in both]
        for fam_name, fam_set in (('lfp', lfp), ('ni', ni)):
            pp = [p for p in both if p[0] in fam_set]
            F[f'tr_gb{k}_{fam_name}'] = (sum(tval(*p, k, 'pinn_semi') < tval(*p, k, 'gbdt') for p in pp), len(pp))
    F['tr_k0_mlp'] = [tval(*p, 0, 'mlp') for p in pairs]
    F['tr_k0_pinn'] = [tval(*p, 0, 'pinn_semi') for p in pairs]
    F['tr_k3_win'] = sum(tval(*p, 3, 'pinn_semi') < tval(*p, 3, 'mlp') for p in pairs
                         if _ok(tval(*p, 3, 'pinn_semi')) and _ok(tval(*p, 3, 'mlp')))
    F['tr_k0_win'] = sum(tval(*p, 0, 'pinn_semi') < tval(*p, 0, 'mlp') for p in pairs
                         if _ok(tval(*p, 0, 'pinn_semi')) and _ok(tval(*p, 0, 'mlp')))

    # bộ có lợi ích nhỏ nhất
    F['weak'] = max(DS, key=lambda ds: _gm([F['r'][fr][ds] for fr in FRS]))
    ctx['F'] = F
    return ctx


def ch6_metrics(d, ctx):
    """Mục RMSE và R² cho mọi mô hình (đọc results/metrics_*.csv)."""
    from scipy.stats import spearmanr
    sm, ts, tr = d['metrics_summary'], d['metrics_tests'], d['metrics_transfer']
    MODS = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']
    get = lambda ds, m, fr, col: (lambda g: float(g[col].iloc[0]) if len(g) else NAN)(
        sm[(sm.dataset == ds) & (sm.model == m) & (sm.frac == fr)])
    rows = []
    for ds in DS:
        for i, m in enumerate(MODS):
            cells = []
            for col, best, fmt in [('RMSE_cell', min, lambda v: m3(v, 2)), ('R2_10', max, lambda v: f(v, 3))]:
                for fr in FRS:
                    v = get(ds, m, fr, col)
                    bv = best([x for x in (get(ds, mm, fr, col) for mm in MODS) if _ok(x)])
                    s = fmt(v) if _ok(v) else '--'
                    cells.append(r'\textbf{' + s + '}' if _ok(v) and v == bv else s)
            rows.append([ds if i == 0 else '', LBL[m]] + cells)
    head = ['Bộ', 'Mô hình'] + [f'{int(100 * fr)}\\,\\%' for fr in FRS] * 2
    tab = table(head, rows, 'll' + 'r' * 8,
                'RMSE theo cell (×$10^{-3}$, bốn cột đầu) và $R^2$ tính từ chu kỳ thứ 10 (bốn cột sau) theo lượng nhãn; '
                'in đậm là tốt nhất trong cột của mỗi bộ. $R^2$ ghép dự đoán của 5 fold thành toàn bộ bộ dữ liệu, '
                'trung bình qua 5 lần lặp.', 'tab:rmser2', size=r'\footnotesize')
    tab = tab.replace(r'\toprule' + '\n', r'\toprule' + '\n' + r' & & \multicolumn{4}{c}{RMSE theo cell} & '
                      r'\multicolumn{4}{c}{$R^2$} \\ \cmidrule(lr){3-6}\cmidrule(lr){7-10}' + '\n', 1)
    c = d['thesis_curve'].rename(columns={'bo': 'dataset'})
    mm = sm.merge(c, on=['dataset', 'model', 'frac'])
    rho_all = spearmanr(mm.MAE_cell, mm.RMSE_cell).statistic
    rho_ds = {ds: spearmanr(g.MAE_cell, g.RMSE_cell).statistic for ds, g in mm.groupby('dataset')}
    sig = lambda other, fr, better: [ds for ds in DS if ((ts.doi_chung == other) & (ts.frac == fr) & (ts.bo == ds)
                                                          & (ts.p_holm < 0.05) & ((ts.ti_so < 1) if better else (ts.ti_so > 1))).any()]
    rr = lambda other, fr: {ds: (lambda g: float(g.ti_so.iloc[0]) if len(g) else NAN)(
        ts[(ts.doi_chung == other) & (ts.frac == fr) & (ts.bo == ds)]) for ds in DS}
    gb_w = [f'{pc(fr)} ({_lst(sig("gbdt", fr, False))})' for fr in FRS if sig('gbdt', fr, False)]
    pn_w = [f'{pc(fr)} ({_lst(sig("gbdt", fr, True))})' for fr in FRS if sig('gbdt', fr, True)]
    r2_10 = {ds: (get(ds, 'pinn_semi', 0.1, 'R2_10'), get(ds, 'mlp', 0.1, 'R2_10')) for ds in DS}
    worst = min(DS, key=lambda ds: get(ds, 'mlp', 0.1, 'R2'))
    best10 = {ds: max(MODS, key=lambda m: get(ds, m, 0.1, 'R2_10')) for ds in DS}
    # chuyển miền
    trs = tr.groupby(['k', 'model']).R2.agg(['median', 'min', 'max']).reset_index()
    trows = []
    for m in ['mlp', 'gbdt', 'pinn_semi']:
        row = [LBL[m]]
        for k in [0, 3]:
            g = trs[(trs.k == k) & (trs.model == m)]
            row += [f(float(g['median'].iloc[0]), 2), f'{f(float(g["min"].iloc[0]), 2)} ; {f(float(g["max"].iloc[0]), 2)}'] \
                if len(g) else ['--', '--']
        trows.append(row)
    ttab = table(['Mô hình', 'trung vị, $k{=}0$', 'nhỏ nhất ; lớn nhất', 'trung vị, $k{=}3$', 'nhỏ nhất ; lớn nhất'],
                 trows, 'lrrrr', '$R^2$ trên tập test của bộ đích qua 12 cặp chuyển miền (mỗi cặp trung bình 5 hạt giống).',
                 'tab:transr2')
    med = lambda m, k: float(trs[(trs.k == k) & (trs.model == m)]['median'].iloc[0])
    return T(r"""
\section{\texorpdfstring{RMSE và $R^2$ cho mọi mô hình}{RMSE và R2 cho mọi mô hình}}\label{sec:rmser2}

MAE theo cell ở các mục trên cho mỗi cell một phiếu. Bảng~\ref{tab:rmser2} và Hình~\ref{fig:rmser2} bổ sung hai
chỉ số quen thuộc: RMSE theo cell, nhạy với sai số lớn hơn MAE, và hệ số xác định $R^2$. $R^2$ được tính trên toàn
bộ bộ dữ liệu: trong một lần lặp, năm fold phủ mỗi cell đúng một lần, nên dự đoán của năm fold ghép lại thành dự
đoán cho mọi chu kỳ của mọi cell; $R^2$ tính trên đó rồi lấy trung bình qua năm lần lặp. Bảng dùng $R^2$ tính từ
chu kỳ thứ 10 của mỗi cell, vì vài chu kỳ đầu của một số cell XJTU có đặc trưng bất thường và riêng chúng đã đủ
kéo $R^2$ gộp xuống rất thấp: tính cả các chu kỳ đó, $R^2$ của MLP trên <<WORST>> ở 10\,\% nhãn chỉ còn
<<WORSTV>>, tức kém hơn cả việc đoán bằng giá trị trung bình.

<<TAB>>
<<FIG>>

Thứ hạng của các mô hình theo RMSE gần như trùng với theo MAE: hệ số tương quan hạng Spearman giữa hai chỉ số là
<<RHO>> trên <<N>> tổ hợp bộ × mô hình × mức nhãn (<<RHODS>>). Vì vậy các kết luận ở Mục~\ref{sec:labels} giữ
nguyên khi đổi thước đo. Ở 10\,\% nhãn, $R^2$ của PINN-semi so với MLP là <<R210>>; RMSE theo cell của PINN-semi
bằng <<RR10>> lần MLP, có ý nghĩa thống kê <<SIG10>> (Wilcoxon theo cell, hiệu chỉnh Holm). So với GBDT, theo RMSE
<<GBTXT>>. Mô hình có $R^2$ cao nhất ở 10\,\% nhãn là <<BEST10>>.

Với chuyển miền (Bảng~\ref{tab:transr2}), $R^2$ trên bộ đích âm với mọi mô hình khi bộ đích không có nhãn: trung vị
<<TRM0>> cho MLP, <<TRG0>> cho GBDT và <<TRP0>> cho PINN-semi. Khi chuyển sang bộ khác, dự đoán lệch đều một khoảng
lớn hơn cả độ biến thiên của SOH trong bộ đích, nên $R^2$ không phân biệt được một mô hình lệch ít với một mô hình
lệch nhiều; MAE ở Mục~\ref{sec:transferres} là thước đo phù hợp hơn. Với 3 cell đích có nhãn, độ lệch này được
hiệu chỉnh và $R^2$ trung vị lên <<TRP3>> cho PINN-semi, <<TRG3>> cho GBDT, trong khi MLP vẫn ở <<TRM3>>.

<<TTAB>>
""", WORST=worst, WORSTV=f(get(worst, 'mlp', 0.1, 'R2'), 2), TAB=tab,
             FIG=fig('fig_metrics_rmse_r2', 'RMSE theo cell (hàng trên, thang log) và $R^2$ tính từ chu kỳ thứ 10 (hàng '
                     'dưới) theo lượng nhãn.', 'fig:rmser2'),
             RHO=f(rho_all, 2), N=len(mm), RHODS=', '.join(f'{ds} {f(rho_ds[ds], 2)}' for ds in DS),
             R210=', '.join(f'{ds} {f(r2_10[ds][0], 2)} so với {f(r2_10[ds][1], 2)}' for ds in DS),
             RR10=_rng(rr('mlp', 0.1).values()), SIG10=_tren(sig('mlp', 0.1, True)) or 'trên không bộ nào',
             GBTXT=('GBDT tốt hơn có ý nghĩa ở ' + '; '.join(gb_w) if gb_w else 'GBDT không tốt hơn có ý nghĩa ở mức nào')
             + ('; PINN-semi tốt hơn có ý nghĩa ở ' + '; '.join(pn_w) if pn_w else '; PINN-semi không tốt hơn có ý nghĩa ở mức nào'),
             BEST10='; '.join(f'{ds}: {LBL[best10[ds]]} ({f(get(ds, best10[ds], 0.1, "R2_10"), 3)})' for ds in DS),
             TRM0=f(med('mlp', 0), 0), TRG0=f(med('gbdt', 0), 1), TRP0=f(med('pinn_semi', 0), 1),
             TRM3=f(med('mlp', 3), 1), TRG3=f(med('gbdt', 3), 2), TRP3=f(med('pinn_semi', 3), 2), TTAB=ttab)


def ch6_physcheck(d, ctx):
    """Mục kiểm tra phần vật lý (đọc results/metrics_summary.csv, results/physics_*.csv)."""
    sm, ph, bins = d['metrics_summary'], d['physics_summary'], d['physics_rate_soh']
    mono = lambda ds, m: float(sm[(sm.dataset == ds) & (sm.model == m) & (sm.frac == 0.3)].vi_pham_don_dieu_pct.iloc[0])
    true = {ds: float(sm[(sm.dataset == ds) & (sm.model == 'mlp') & (sm.frac == 0.3)].vi_pham_nhan_that_pct.iloc[0])
            for ds in DS}
    MM = ['mlp', 'gbdt', 'pinn_sup', 'pinn_bb', 'mono_only', 'pinn_semi']
    mrows = [[ds, f(true[ds], 1)] + [f(mono(ds, m), 1) for m in MM] for ds in DS]
    tabM = table(['Bộ', 'Nhãn thật'] + [LBL[m] for m in MM], mrows, 'l' + 'r' * 7,
                 'Tỉ lệ cặp chu kỳ cách nhau ít nhất 10 chu kỳ mà SOH dự đoán tăng quá 0,002 (\\%), 30\\,\\% nhãn, mọi '
                 'cell test của 5 lần lặp; cột đầu là cùng phép đếm trên chính nhãn thật.', 'tab:mono')
    P = lambda ds, tag, col: float(ph[(ph.bo == ds) & (ph.nhanh == tag)][col].iloc[0])
    prows = [[ds] + [f(P(ds, t, 'dudt_duong_pct'), 1) for t in 'ADCE']
             + [f(100 * P(ds, t, 'phan_thoi_gian_trung_vi'), 0) for t in 'ACE']
             + [f(P(ds, 'C', 'toc_do_r'), 3), f(P(ds, 'E', 'toc_do_r'), 3), f(P(ds, 'C', 'toc_do_that'), 3)] for ds in DS]
    tabP = table(['Bộ', 'MLP', 'chỉ đ.điệu', 'PINN-semi', 'Euler', 'MLP', 'PINN-semi', 'Euler', '$r$ PINN-semi',
                  '$r$ Euler', 'thực đo'], prows, 'l' + 'r' * 10,
                 'Đo trên cell test của mô hình đã huấn luyện (30\\,\\% nhãn, lần lặp đầu, 5 fold). Bốn cột đầu: tỉ lệ '
                 'điểm có $\\partial F_\\varphi/\\partial t > 0$ (\\%). Ba cột giữa: trung vị qua các cell của phần giảm SOH '
                 'dự đoán do riêng biến thời gian (\\%). Ba cột cuối: tốc độ suy giảm trung bình (SOH trên 1\\,000 chu kỳ).',
                 'tab:physcheck', size=r'\footnotesize')
    tabP = tabP.replace(r'\toprule' + '\n', r'\toprule' + '\n' + r' & \multicolumn{4}{c}{$\partial F_\varphi/\partial t > 0$ (\%)}'
                        r' & \multicolumn{3}{c}{phần do thời gian (\%)} & \multicolumn{3}{c}{tốc độ suy giảm} \\'
                        r' \cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}' + '\n', 1)
    # số liệu cho văn bản
    m_mlp = [mono(ds, 'mlp') for ds in DS]; m_semi = [mono(ds, 'pinn_semi') for ds in DS]
    m_only = [mono(ds, 'mono_only') for ds in DS]
    close = all(abs(a - b) <= 1.5 for a, b in zip(m_only, m_semi))
    dpos = {t: {ds: P(ds, t, 'dudt_duong_pct') for ds in DS} for t in 'ACE'}
    lower = [ds for ds in DS if dpos['C'][ds] < 0.5 * dpos['A'][ds]]
    trivial = [ds for ds in DS if P(ds, 'C', 'toc_do_r') < 0.01]
    share = {t: [100 * P(ds, t, 'phan_thoi_gian_trung_vi') for ds in DS] for t in 'ACE'}
    rel = {ds: P(ds, 'C', 'phan_du_tuong_doi') for ds in DS if ds not in trivial}
    rhoC = {ds: P(ds, 'C', 'rho_r') for ds in DS}; rhoE = {ds: P(ds, 'E', 'rho_r') for ds in DS}
    good_E = [ds for ds in DS if rhoE[ds] > 0.5]
    rt = {ds: P(ds, 'C', 'rho_tong') for ds in DS}; rtA = {ds: P(ds, 'A', 'rho_tong') for ds in DS}

    def curve(ds, tag, col):
        return bins[(bins.ds == ds) & (bins.tag == tag) & (bins.n >= 200)].sort_values('bin')[col].to_numpy()
    span = lambda v: float(v.max() / v.min()) if len(v) >= 2 and v.min() > 0 else NAN
    kt = {ds: span(curve(ds, 'A', 'rate_true')) for ds in DS}
    kr = {ds: span(curve(ds, 'C', 'r')) for ds in DS if ds not in trivial}
    rise = {ds: curve(ds, 'A', 'rate_true')[0] / curve(ds, 'A', 'rate_true')[-1] for ds in DS}  # SOH thấp / SOH cao
    knee_ds = [ds for ds in DS if rise[ds] > 1.5]; no_knee = [ds for ds in DS if ds not in knee_ds]
    cells_ = d['physics_cells'][d['physics_cells'].tag == 'A']
    cv = (cells_.groupby('ds').rate_true.std() / cells_.groupby('ds').rate_true.mean()).to_dict()
    flat = min(DS, key=cv.get)
    shC = {ds: 100 * P(ds, 'C', 'phan_thoi_gian_trung_vi') for ds in DS}
    low25 = [ds for ds in DS if shC[ds] < 25]; top = max(DS, key=shC.get)
    pc = d.get('physics_check')
    n_same = f'{int(pc.n_same.iloc[0])}/{int(pc.n_ref.iloc[0])}' if pc is not None else '--'
    return T(r"""
\section{Kiểm tra phần vật lý}\label{sec:physcheck}

Mục~\ref{sec:wherephys} chỉ ra rằng ba hàm mất mát vật lý tác động vào hai hướng khác nhau của hàm $F_\varphi$:
$\mathcal{L}_\text{mono}$ tác động lên tổng biến thiên dọc quỹ đạo thật của cell, $\mathcal{L}_\text{ode}$ tác động
lên đạo hàm riêng theo thời gian khi giữ nguyên đặc trưng sạc. Mục này đo riêng từng hướng trên các cell test.

\paragraph{Dọc quỹ đạo: tính đơn điệu của dự đoán.} Với mỗi cell test, đếm các cặp chu kỳ cách nhau ít nhất 10 chu
kỳ mà SOH dự đoán ở chu kỳ sau cao hơn chu kỳ trước quá 0,002, tức đúng ngưỡng của $\mathcal{L}_\text{mono}$. Phép
đếm này không cần nhãn, nên làm được với mọi mô hình; làm trên chính nhãn thật cho mức tham chiếu, vì dung lượng
thật có thể hồi phục nhẹ sau thời gian nghỉ (Bảng~\ref{tab:mono}, Hình~\ref{fig:mono}). Ở 30\,\% nhãn, dự đoán của
MLP tăng ngược ở <<MMLP>>\,\% số cặp, của PINN-semi chỉ <<MSEMI>>\,\%, trong khi nhãn thật là <<MTRUE>>\,\%.
<<CLOSE>>

<<TABM>>
<<FIGM>>

\paragraph{Theo thời gian: đạo hàm riêng và phần dư ODE.} Đại lượng mà $\mathcal{L}_\text{ode}$ ràng buộc là
$\partial F_\varphi/\partial t$, chỉ tính được từ chính mô hình. Trọng số mô hình không được lưu, nhưng huấn luyện
là tất định, nên đồ án huấn luyện lại các lượt của MLP, PINN-semi, các biến thể cắt bỏ và PINN hộp đen ở 30\,\% nhãn,
lần lặp đầu, cả 5 fold trên 4 bộ: <<NSAME>> lượt cho MAE trùng từng bit với lượt đã báo cáo. Trên cell test của
từng lượt, đo $\partial F_\varphi/\partial t$ bằng vi phân tự động, tốc độ $r$ của động học xám, và tách tổng mức
giảm SOH dự đoán trên vòng đời cell thành phần do biến thời gian
$\sum_i [F_\varphi(\tilde{\mathbf{x}}_i, t_{i+1}) - F_\varphi(\tilde{\mathbf{x}}_i, t_i)]$ và phần do đặc trưng sạc
$\sum_i [F_\varphi(\tilde{\mathbf{x}}_{i+1}, t_{i+1}) - F_\varphi(\tilde{\mathbf{x}}_i, t_{i+1})]$; hai phần cộng lại
đúng bằng tổng (Bảng~\ref{tab:physcheck}).

<<TABP>>

Ba nhận xét. Thứ nhất, MLP có $\partial F_\varphi/\partial t > 0$, tức SOH tăng theo thời gian khi đặc trưng giữ
nguyên, ở <<DPA>>\,\% số điểm test; PINN-semi giảm tỉ lệ này xuống <<DPC>>\,\%<<LOWER>>. <<TRIVIAL>>Thứ hai, ở mọi
mô hình, phần lớn mức giảm SOH dự đoán đi qua đặc trưng sạc chứ không qua biến thời gian: phần do thời gian chỉ
chiếm <<SHA>>\,\% với MLP và <<SHC>>\,\% với PINN-semi; với PINN-semi, phần này dưới 25\,\% <<LOW25>> và cao
nhất <<SHMAX>>. Mạng nghiệm nhận ra pin đã già chủ yếu nhờ đường sạc đã đổi dạng, và $\mathcal{L}_\text{ode}$ chỉ
ràng buộc phần còn lại. Thứ ba, phần dư ODE nhỏ so với chính đạo hàm
(<<REL>>), nghĩa là ràng buộc được thoả trên cell chưa từng thấy.

\paragraph{Tốc độ suy giảm học được.} Hình~\ref{fig:ratecell} so tốc độ suy giảm thực đo của từng cell test (độ
dốc của SOH theo chu kỳ) với tốc độ của mô hình. Độ dốc của SOH dự đoán bám sát thực đo ở cả MLP lẫn PINN-semi
(tương quan hạng Spearman <<RT>>)<<FLAT>>. Tốc độ $r$ của
PINN-semi thì không: trung bình chỉ <<RC>> so với <<RTRUE>> thực đo, và gần như không tương quan với tốc độ của từng
cell (<<RHOC>>). Lý do nằm ở nhận xét thứ hai: ở dạng đạo hàm riêng, $r$ được khớp với
$-\partial F_\varphi/\partial t$, tức chỉ với phần suy giảm mà mạng gán cho biến thời gian. Dạng Euler khớp $r$ với
tổng biến thiên dọc quỹ đạo, nên $r$ của nó xấp xỉ tốc độ thật (trung bình <<RE>>) và tương quan với tốc độ từng cell
(<<RHOE>>)<<GOODE>>. Đổi lại, dạng Euler có MAE cao hơn dạng đạo hàm riêng trên TJU và MIT (Mục~\ref{sec:source}).

<<FIGRC>>

Hình~\ref{fig:ratesoh} xếp tốc độ suy giảm theo khoảng SOH thực đo. <<KNEE>> Mỗi khoảng SOH gồm một tập cell khác
nhau, nên đây là đường trung bình chứ không phải đường của một cell. Độ dốc dự đoán của PINN-semi theo sát đường
thực đo trên cả bốn bộ, nhưng $r$ gần như phẳng: giữa các khoảng SOH, $r$ chỉ thay đổi <<KR>> lần<<KRX>>, trong khi
tốc độ thực đo thay đổi <<KT>> lần. Với $\lambda \approx 0{,}96$ (Mục~\ref{sec:physparams}), thừa số $g(u) = e^{\lambda(1-u)}$ chỉ tăng khoảng
1,2 lần khi SOH giảm từ 1 xuống 0,8. Đầu gối trong dự đoán đến từ đặc trưng sạc, không đến từ thừa số đầu gối của
động học.

<<FIGRS>>

Tóm lại, phần vật lý giúp mô hình chủ yếu qua ràng buộc dọc quỹ đạo: dự đoán của PINN-semi gần đơn điệu như nhãn
thật, và đây cũng là phần mang lại phần lớn lợi ích về sai số (Mục~\ref{sec:source}). Ràng buộc ODE dạng đạo hàm riêng
giữ cho SOH không tăng theo thời gian, nhưng tốc độ $r$ của nó không phải là ước lượng tốc độ suy giảm của cell; muốn
$r$ diễn giải được theo nghĩa vật lý cần dạng Euler.
""", MMLP=_rng(m_mlp, 1), MSEMI=_rng(m_semi, 1), MTRUE=_rng(true.values(), 1),
             CLOSE=(f'Biến thể chỉ đơn điệu cho tỉ lệ gần như bằng PINN-semi ({_rng(m_only, 1)}\\,\\%): chính '
                    '$\\mathcal{L}_\\text{mono}$, áp lên cặp chu kỳ của cả cell không nhãn, làm quỹ đạo dự đoán đơn điệu.'
                    if close else ''),
             TABM=tabM, FIGM=fig('fig_metrics_mono', 'Tỉ lệ cặp chu kỳ có SOH dự đoán tăng ngược ở 30\\,\\% nhãn; vạch '
                                 'đứt và vùng tô xám là cùng phép đếm trên nhãn thật.', 'fig:mono'),
             NSAME=n_same, TABP=tabP, DPA=_rng(dpos['A'].values(), 0), DPC=_rng(dpos['C'].values(), 0),
             LOWER=(f', giảm rõ {_tren(lower)}' if lower else ''),
             TRIVIAL=(f'Riêng {_lst(trivial)}, ràng buộc được thoả theo cách tầm thường: $r$ học được gần bằng 0 '
                      f'({_pairs({ds: P(ds, "C", "toc_do_r") for ds in trivial}, 3)}) và đạo hàm theo thời gian cũng gần '
                      'bằng 0, nên dấu của đạo hàm chỉ còn là nhiễu quanh 0. ' if trivial else ''),
             SHA=_rng(share['A'], 0), SHC=_rng(share['C'], 0),
             REL=('$|\\partial_t F_\\varphi + r|$ bằng ' + _pairs({ds: 100 * v for ds, v in rel.items()}, 0)
                  + '\\,\\% của $|\\partial_t F_\\varphi|$'),
             RT=f'{_rng(rtA.values(), 2)} với MLP, {_rng(rt.values(), 2)} với PINN-semi',
             RC=_rng([P(ds, 'C', 'toc_do_r') for ds in DS], 3), RTRUE=_rng([P(ds, 'C', 'toc_do_that') for ds in DS], 3),
             RHOC='Spearman ' + _pairs(rhoC, 2), RE=_rng([P(ds, 'E', 'toc_do_r') for ds in DS], 3),
             RHOE='Spearman ' + _pairs(rhoE, 2),
             GOODE=(f', rõ trên {_lst(good_E)}' if good_E and len(good_E) < len(DS) else ''),
             FIGRC=fig('fig_physics_rate_cell', 'Tốc độ suy giảm của từng cell test: thực đo (trục ngang) so với độ dốc '
                       'dự đoán và tốc độ $r$ học được (trục dọc); đường chéo là trùng khớp.', 'fig:ratecell'),
             KT=_rng(kt.values(), 0), KR=_rng(kr.values(), 1),
             KRX=(f' (bỏ qua {_lst(trivial)}, nơi $r \\approx 0$)' if trivial else ''),
             KNEE=((f'Trên {_lst(knee_ds)}, tốc độ thực đo ở khoảng SOH thấp nhất cao gấp '
                    f'{_rng([rise[ds] for ds in knee_ds], 0)} lần ở khoảng SOH cao nhất: đây là hiện tượng đầu gối.')
                   + (f' Trên {_lst(no_knee)}, đường trung bình không đi lên đều theo SOH.' if no_knee else '')),
             LOW25=_tren(low25) if low25 else '', SHMAX=f'trên {top} ({f(shC[top], 0)}\\,\\%)',
             FLAT=(f'; trên {flat} tương quan thấp hơn vì tốc độ của các cell gần như nhau (hệ số biến thiên '
                   f'{f(cv[flat], 2)}, so với {_rng([cv[ds] for ds in DS if ds != flat], 2)} ở các bộ khác)'),
             FIGRS=fig('fig_physics_rate_soh', 'Tốc độ suy giảm theo khoảng SOH thực đo (trục ngang giảm dần): thực đo, '
                       'độ dốc dự đoán và tốc độ $r$ học được.', 'fig:ratesoh'))


def ch6_text(d, ctx):
    t, F = ctx['_tables'], ctx['F']
    st = ctx['stats'].set_index('bo')
    eq = d['thesis_equivalence']

    # ---- rò rỉ
    lk = d['thesis_leakage'].set_index('bo')
    lr = lambda key: {ds: lk.loc[ds, key] / lk.loc[ds, 'chuan'] for ds in DS} if key in lk.columns else {}
    row_r, mm, lf, nt = lr('row'), lr('minmax'), lr('life3s'), lr('normtest')
    mm_lo = [ds for ds in DS if mm.get(ds, 1) < 0.95]
    mm_hi = [ds for ds in DS if mm.get(ds, 1) > 1.05]
    spread = {ds: st.loc[ds, 's1b'] - st.loc[ds, 's1a'] for ds in DS}
    tight = sorted(DS, key=spread.get)[:len(mm_lo)]
    endsoh = ', '.join(f'{ds} {f(st.loc[ds, "s1a"], 2)}--{f(st.loc[ds, "s1b"], 2)}' for ds in mm_lo)
    mm_txt = ''
    if mm_lo:
        mm_txt += (f'Chuẩn hoá min--max theo vòng đời cell làm sai số thấp giả tạo {_tren(mm_lo)} '
                   f'(tỉ số {_pairs({ds: mm[ds] for ds in mm_lo})}), vì đầu vào đã chứa vị trí tương đối của chu kỳ '
                   'trong vòng đời')
        mm_txt += (f'; ở {"hai bộ" if len(mm_lo) == 2 else "các bộ"} này SOH cuối đời của mọi cell gần như bằng nhau '
                   f'({endsoh}; Bảng~\\ref{{tab:datastats}}), nên biết vị trí trong vòng đời gần như đủ để suy ra SOH. '
                   if set(tight) == set(mm_lo) else '. ')
    if mm_hi:
        endhi = ', '.join(f'{ds} {f(st.loc[ds, "s1a"], 2)}--{f(st.loc[ds, "s1b"], 2)}' for ds in mm_hi)
        mm_txt += (f'Ngược lại, {_tren(mm_hi)} cùng phép chuẩn hoá đó làm sai số tăng '
                   f'(tỉ số {_pairs({ds: mm[ds] for ds in mm_hi})}): SOH cuối đời ở đây trải rộng hơn ({endhi}), nên '
                   'vị trí trong vòng đời không đủ để suy ra SOH, trong khi chuẩn hoá theo từng cell lại xoá mất mức '
                   'tuyệt đối của đặc trưng, vốn là thông tin phân biệt cell này với cell khác. '
                   'Dù theo chiều nào, con số đo được cũng không phản ánh sai số khi triển khai, vì tuổi thọ của một cell '
                   'đang vận hành chưa biết trước. ')
    small = [f'lọc dùng cả vòng đời: {_rng(lf.values())}'] if lf else []
    small += [f'chuẩn hoá gồm cả test: {_rng(nt.values())}'] if nt else []
    small_txt = (f'{"Hai dạng" if len(small) == 2 else "Dạng"} còn lại tác động nhỏ hơn ({"; ".join(small)}), nhưng '
                 'vẫn là rò rỉ và vẫn được loại bỏ trong giao thức chuẩn. ') if small else ''

    # ---- 30 %
    r30 = F['r'][0.3]
    lo30, hi30 = min(DS, key=r30.get), max(DS, key=r30.get)
    nb30 = F['nb'][0.3]; rest30 = [ds for ds in DS if ds not in nb30]
    if len(nb30) == len(DS):
        nb_txt = 'Kết luận vẫn đứng vững trên cả bốn bộ với kiểm định bảo thủ ở mức fold.'
    else:
        nb_txt = ((f'Với kiểm định bảo thủ ở mức fold, kết luận chỉ còn đứng vững {_tren(nb30)}; '
                   if nb30 else 'Không bộ nào qua được kiểm định bảo thủ ở mức fold; ')
                  + f'{_lst(rest30)} có hướng cải thiện nhất quán (Hình~\\ref{{fig:cells}}: phần lớn cell nằm dưới '
                  'đường chéo) nhưng biến thiên giữa các lần huấn luyện còn lớn so với mức cải thiện.')

    # ---- lượng nhãn tương đương
    def eqv(ds, fr, who):
        g = eq[(eq.bo == ds) & (eq.frac_pinn == fr) & (eq.doi_chung == who)]
        return float(g.frac_tuong_duong.iloc[0]) if len(g) else NAN
    def feq(x):
        return '$\\le 10\\,\\%$' if x == -1 else ('$> 70\\,\\%$' if math.isinf(x) else pc(x))
    e10m = {ds: eqv(ds, 0.1, 'mlp') for ds in DS}
    e30m = {ds: eqv(ds, 0.3, 'mlp') for ds in DS}
    e30b = {ds: eqv(ds, 0.3, 'tot_nhat_co_nhan') for ds in DS}
    e10b = {ds: eqv(ds, 0.1, 'tot_nhat_co_nhan') for ds in DS}
    fin = [x for x in e10m.values() if _ok(x) and x > 0]
    eq_txt = (f'So với MLP, PINN-semi dùng 10\\,\\% nhãn đạt sai số mà MLP cần '
              f'{", ".join(f"{ds} {feq(e10m[ds])}" for ds in DS)} nhãn mới đạt'
              + (f', tức tiết kiệm {f(min(fin) / 0.1, 1)}--{f(max(fin) / 0.1, 1)} lần số nhãn' if fin else '')
              + f'; PINN-semi dùng 30\\,\\% nhãn tương đương MLP dùng '
              f'{", ".join(f"{ds} {feq(e30m[ds])}" for ds in DS)}. ')
    b10_none = [ds for ds in DS if e10b[ds] == -1]
    save30 = [ds for ds in DS if _ok(e30b[ds]) and e30b[ds] > 0.3 or math.isinf(e30b[ds])]
    lose30 = [ds for ds in DS if _ok(e30b[ds]) and (e30b[ds] < 0.3)]
    if b10_none:
        eq_txt += (f'So với mô hình học có nhãn tốt nhất ở từng mức thì khác: {_tren(b10_none)}, mô hình học có nhãn '
                   'tốt nhất đã đạt sai số của PINN-semi ngay ở 10\\,\\% nhãn')
        eq_txt += (f'; ở 30\\,\\% nhãn, PINN-semi vẫn tiết kiệm nhãn {_tren(save30)} '
                   f'(tương đương {", ".join(feq(e30b[ds]) for ds in save30)})' if save30 else '')
        eq_txt += (f', còn {_tren(lose30)} mô hình tốt nhất chỉ cần '
                   f'{_rng([100 * e30b[ds] for ds in lose30], 0)}\\,\\% nhãn để bằng PINN-semi 30\\,\\%.'
                   if lose30 else '.')

    # ---- so với GBDT
    gbl = [f'{pc(fr)} ({_lst(F["gb_sig_better"][fr])})' for fr in FRS if F['gb_sig_better'][fr]]
    pbl = [f'{pc(fr)} ({_lst(F["pinn_sig_better_gb"][fr])})' for fr in FRS if F['pinn_sig_better_gb'][fr]]
    gb_txt = ('mô hình này tốt hơn PINN-semi có ý nghĩa thống kê ở ' + '; '.join(gbl) if gbl
              else 'dù vậy nó không tốt hơn PINN-semi có ý nghĩa thống kê ở mức nhãn nào')
    gb_txt += ('; ngược lại PINN-semi tốt hơn GBDT có ý nghĩa ở ' + '; '.join(pbl) if pbl
               else '; ngược lại không có trường hợp nào PINN-semi tốt hơn GBDT có ý nghĩa thống kê')
    gb_txt += f'. Ở 10\\,\\% nhãn, tỉ số MAE của PINN-semi so với GBDT là {_pairs(F["r_gb10"])}'
    ridge_txt = (f'Ridge, dù chỉ là mô hình tuyến tính, tốt hơn MLP ở 10\\,\\% nhãn {_tren(F["ridge10_better"])}: '
                 'khi rất ít nhãn, mô hình đơn giản ít quá khớp hơn. Nhưng Ridge cải thiện chậm khi thêm nhãn: MAE ở '
                 f'70\\,\\% bằng {_rng(F["ridge_gain"])} lần MAE ở 10\\,\\%, so với {_rng(F["mlp_gain"])} lần của MLP. '
                 if F['ridge10_better'] else '')
    gw, gn = F['gpr_vs_gb']
    pw, pn = F['wins']['gpr']
    gpr_all = [ds for ds in DS if all(_ok(F['val']('gpr', fr, ds)) and F['val']('gpr', fr, ds) < F['val']('pinn_semi', fr, ds)
                                      for fr in FRS)]
    gpr_txt = (f'GPR chỉ dùng được 1\\,500 điểm huấn luyện do chi phí $O(n^3)$ theo số điểm, và kém GBDT trong {gw}/{gn} '
               f'trường hợp (bộ × mức nhãn); dù vậy GPR vẫn có MAE thấp hơn PINN-semi trong {pn - pw}/{pn} trường hợp'
               + (f', và thấp hơn ở cả bốn mức nhãn {_tren(gpr_all)}' if gpr_all else '') + '. ' if gn else '')

    # ---- nguồn gốc lợi ích
    sup_bad = [ds for ds in DS if _ok(F['sup10'][ds]) and F['sup10'][ds] > 1]
    sup_txt = (f'ở 10\\,\\% nhãn tỉ số so với MLP là {_pairs(F["sup10"])}. Vật lý đóng vai trò bộ chuẩn hoá có giúp, '
               'nhưng còn xa mới bằng PINN-semi' + (f', và trên {_lst(sup_bad)} còn kém MLP' if sup_bad else '') + '.')
    unl_txt = (f'tỉ số {_pairs(F["semisup10"])} ở 10\\,\\% và {_pairs(F["semisup30"])} ở 30\\,\\% nhãn. Tính theo '
               f'thang log, phần này chiếm {_rng([100 * x for x in F["share_unl10"]], 0)}\\,\\% mức giảm sai số của '
               'PINN-semi so với MLP ở 10\\,\\% nhãn, phần còn lại đến từ vật lý trên cell có nhãn. '
               + ('Dữ liệu không nhãn là nguồn lợi ích lớn nhất ở mức nhãn thấp' if min(F['share_unl10']) > 0.55 else
                  'Ở mức nhãn thấp, dữ liệu không nhãn đóng góp ít nhất ngang với vật lý trên cell có nhãn')
               + ', và nó chỉ có ích khi được nối với mô hình qua loss vật lý không dùng nhãn.')
    ms = F['mono_semi']; worst_m = max(DS, key=lambda ds: ms[ds] if _ok(ms[ds]) else 0)
    mono_txt = (f'chỉ với ràng buộc đơn điệu, mô hình đã đạt {_rng([100 * x for x in F["share_mono"]], 0)}\\,\\% mức '
                'giảm sai số (thang log) của PINN-semi so với MLP ở 30\\,\\% nhãn. Khi bỏ $\\mathcal{L}_\\text{ode}$ và '
                f'$\\mathcal{{L}}_\\text{{range}}$, MAE bằng {_pairs(ms)} lần PINN-semi đầy đủ, chênh lệch lớn nhất ở '
                f'{worst_m}: phần còn lại của lợi ích đến từ cấu trúc động học.')
    eu = F['euler_semi']
    eu_eq = [ds for ds in DS if _ok(eu[ds]) and abs(eu[ds] - 1) <= 0.03]
    eu_w = [ds for ds in DS if _ok(eu[ds]) and eu[ds] > 1.03]
    eu_b = [ds for ds in DS if _ok(eu[ds]) and eu[ds] < 0.97]
    eu_txt = (f'MAE của phần dư Euler dọc quỹ đạo so với dạng đạo hàm riêng theo $t$ là {_pairs(eu)}: '
              + '; '.join(x for x in [f'ngang nhau {_tren(eu_eq)}' if eu_eq else '',
                                      f'kém hơn {_tren(eu_w)}' if eu_w else '',
                                      f'tốt hơn {_tren(eu_b)}' if eu_b else ''] if x) + '.')
    if F['bb_have']:
        gw = '; '.join(f'{pc(fr)} ({_lst(F["bb_sig_semi"][fr])})' for fr in FRS if F['bb_sig_semi'][fr])
        bw = '; '.join(f'{pc(fr)} ({_lst(F["bb_sig_bb"][fr])})' for fr in FRS if F['bb_sig_bb'][fr])
        bb_txt = (f'tỉ số MAE của PINN hộp đen so với PINN-semi là {_pairs(F["bb10"])} ở 10\\,\\% và '
                  f'{_pairs(F["bb30"])} ở 30\\,\\% nhãn. Theo kiểm định Wilcoxon có hiệu chỉnh Holm, động học xám tốt '
                  f'hơn có ý nghĩa ở {gw if gw else "không trường hợp nào"}'
                  + (f', hộp đen tốt hơn có ý nghĩa ở {bw}' if bw else '')
                  + f'; {16 - F["bb_nsig"]} trường hợp còn lại (trong 16 cặp bộ × mức nhãn) khác biệt không có ý nghĩa. '
                  f'Khác biệt lớn nhất ở {F["bb_max"][0]} với {pc(F["bb_max"][1])} nhãn, khi MAE của PINN hộp đen bằng '
                  f'{f(1 / F["bb_max"][2], 2)} lần PINN-semi. '
                  + (f'Ở 30\\,\\% nhãn, sai số của PINN hộp đen gần như trùng với biến thể chỉ đơn điệu (chênh không quá '
                     f'{f(100 * F["bb_mono_gap"], 0)}\\,\\%). Điều này phù hợp với giả thuyết: khi $G$ nhận cả $u_t$ làm '
                     'đầu vào, mạng có thể học $G \\approx u_t$ nên phần dư động học bị thoả mãn tầm thường, chỉ còn ràng '
                     'buộc đơn điệu và miền giá trị có tác dụng; động học xám không có lối thoát này vì $r$ không phụ thuộc '
                     '$u_t$ và luôn không âm.' if F['bb_mono_gap'] <= 0.03 else
                     'Một khả năng là khi $G$ nhận cả $u_t$ làm đầu vào, mạng có thể học $G \\approx u_t$ nên phần dư động '
                     'học bị thoả mãn tầm thường; động học xám không có lối thoát này vì $r$ không phụ thuộc $u_t$.'))
    else:
        bb_txt = 'chưa có kết quả.'

    # ---- chỉ số mở rộng
    pmx = d['pred_metrics'].set_index(['bo', 'arm'])
    r2b30 = [ds for ds in DS if pmx.loc[(ds, 'C'), 'R2'] > pmx.loc[(ds, 'A'), 'R2']]
    r2b70 = [ds for ds in DS if pmx.loc[(ds, 'G'), 'R2'] > pmx.loc[(ds, 'F'), 'R2']]
    r2_txt = (f'Ở 30\\,\\% nhãn, $R^2$ của PINN-semi cao hơn MLP {_tren(r2b30) or "trên không bộ nào"}; ở 70\\,\\% nhãn '
              f'{_tren(r2b70) or "trên không bộ nào"}'
              + (f', trừ {_lst([ds for ds in DS if ds not in r2b70])}' if 0 < len(r2b70) < len(DS) else '') + '. ')
    xj = [pmx.loc[('XJTU', a), 'R2_tu10'] for a in 'ACFG']
    xj0 = [pmx.loc[('XJTU', a), 'R2'] for a in 'ACFG']
    r2_txt += (f'$R^2$ gộp của XJTU thấp ({_rng(xj0)}) vì vài chu kỳ đầu của một số cell có đặc trưng khác xa dữ liệu '
               f'huấn luyện và cho sai số rất lớn; tính từ chu kỳ thứ 10, $R^2$ của XJTU là {_rng(xj)}.')

    # ---- chuyển miền
    Tr = F['tr']
    if F['tr_n']:
        hard = Tr[(False, 0, 'pinn')] > 1.5 * Tr[(True, 0, 'pinn')]
        tr_txt = (f'Khi bộ đích không có nhãn ($k = 0$), MLP gần như không dùng được: MAE theo cell '
                  f'{_rng(F["tr_k0_mlp"])} (đơn vị SOH), tức dự đoán lệch hẳn khỏi miền giá trị. PINN-semi giữ sai số '
                  f'ở mức {_rng(F["tr_k0_pinn"], 3)} nhờ loss vật lý áp lên dữ liệu sạc của bộ đích và ràng buộc miền giá '
                  f'trị, và tốt hơn MLP ở {F["tr_k0_win"]}/{F["tr_n"]} cặp. Trung bình nhân của tỉ số PINN/MLP ở $k = 0$ '
                  f'là {f(Tr[(True, 0, "ratio")], 3)} cho các cặp cùng họ hoá học và {f(Tr[(False, 0, "ratio")], 3)} '
                  'cho các cặp khác họ. Khi có 3 cell đích có nhãn, MLP cải thiện mạnh và khoảng cách thu hẹp (tỉ số '
                  f'{f(Tr[(True, 3, "ratio")], 2)} và {f(Tr[(False, 3, "ratio")], 2)}); PINN-semi vẫn tốt hơn MLP ở '
                  f'{F["tr_k3_win"]}/{F["tr_n"]} cặp. ')
        if F['tr_gb0_n']:
            tr_txt += (f'GBDT không ngoại suy ra ngoài miền nhãn của dữ liệu huấn luyện, nên ở $k = 0$ không lệch như MLP '
                       f'(MAE {_rng(F["tr_gb0_rng"], 3)}). So với GBDT, PINN-semi tốt hơn ở {F["tr_gb0_win"]}/'
                       f'{F["tr_gb0_n"]} cặp, tỉ số MAE trung bình nhân {f(F["tr_gb0_gm"], 2)}. Kết quả phụ thuộc rõ vào '
                       f'bộ nguồn: khi nguồn là LFP, PINN-semi tốt hơn GBDT ở {F["tr_gb0_lfp"][0]}/{F["tr_gb0_lfp"][1]} '
                       f'cặp; khi nguồn thuộc họ niken, chỉ {F["tr_gb0_ni"][0]}/{F["tr_gb0_ni"][1]} cặp. Ở $k = 3$, '
                       f'PINN-semi tốt hơn GBDT ở {F["tr_gb3_win"]}/{F["tr_gb3_n"]} cặp (tỉ số {f(F["tr_gb3_gm"], 2)}). ')
        if _ok(Tr[(False, 0, 'pinn')]):
            tr_txt += ((f'Chuyển giữa hai họ hoá học (NCM/NCA và LFP) khó hơn hẳn: MAE trung bình nhân của PINN-semi ở '
                        f'$k = 0$ là {f(Tr[(True, 0, "pinn")], 3)} cho cặp cùng họ và {f(Tr[(False, 0, "pinn")], 3)} cho '
                        'cặp khác họ, nhiều khả năng do đặc trưng sạc của hai họ khác nhau về bản chất.' + F['tr_dir']) if hard else
                       (f'MAE trung bình nhân của PINN-semi ở $k = 0$ là {f(Tr[(True, 0, "pinn")], 3)} cho cặp cùng họ '
                        f'và {f(Tr[(False, 0, "pinn")], 3)} cho cặp khác họ.'))
    else:
        tr_txt = ''

    # ---- tham số vật lý
    php = d.get('E13_physics_params')
    try:
        lam = [php.loc[('C', ds), ('lam', 'mean')] for ds in DS]
        dea = {ds: php.loc[('C', ds), ('Ea', 'mean')] - 30.0 for ds in DS}
        ph_txt = (f'Cả hai tham số vật lý gần như không rời giá trị khởi tạo: $\\lambda$ nằm trong khoảng {_rng(lam)} '
                  f'trên mọi bộ (khởi tạo 1), $E_a$ dịch {_pairs({ds: abs(v) for ds, v in dea.items()}, 1)}\\,kJ/mol. '
                  'Trên XJTU, nhiệt độ thí nghiệm trùng $T_\\text{ref} = 25$\\,\\textdegree C nên $A(T) = 1$ và gradient '
                  'theo $E_a$ bằng 0; trên MIT và HUST, mọi cell cùng 30\\,\\textdegree C nên $A(T)$ chỉ là một hằng số '
                  'nhân, không tách được khỏi $k(\\mathbf{x})$. Ngay cả trên TJU, nơi có ba mức nhiệt độ, $E_a$ cũng chỉ '
                  'dịch chừng 1\\,kJ/mol: với trọng số $\\alpha = 0{,}02$, tín hiệu từ $\\mathcal{L}_\\text{ode}$ quá yếu '
                  'để ước lượng $E_a$, và mạng $k(\\mathbf{x})$ có thể tự hấp thụ ảnh hưởng của nhiệt độ qua đặc trưng '
                  'sạc. Vì vậy hai tham số này nên được hiểu là phần cấu trúc của ràng buộc (dấu và dạng hàm của tốc độ '
                  'suy giảm), không phải đại lượng vật lý được ước lượng từ dữ liệu.')
    except Exception:
        ph_txt = ''

    weak = F['weak']
    weak_txt = (f'Khi đã có đủ nhãn (70\\,\\%), và trên {weak}, nơi tỉ số so với MLP chỉ ở mức '
                f'{_rng([F["r"][fr][weak] for fr in FRS])} trên mọi mức nhãn.'
                + (' TJU gộp ba nhóm hoá học và chín nhóm điều kiện; có thể phần biến thiên giữa các nhóm lớn hơn phần '
                   'mà ràng buộc vật lý dùng chung cho mọi cell giải thích được.' if weak == 'TJU' else ''))

    return T(r"""
\chapter{Kết quả và thảo luận}\label{ch:ketqua}

Trước khi so mô hình, cần biết những con số sắp đọc có đáng tin không. Vì vậy chương này mở đầu bằng kiểm toán
rò rỉ: cùng một mô hình được đánh giá dưới các cách chia đúng và sai để thấy cách chia sai làm sai số trông tốt hơn
thực tế bao nhiêu. Sau đó các mục đi lần lượt theo câu hỏi nghiên cứu: PINN so với MLP ở 30\,\% nhãn, rồi khi
lượng nhãn thay đổi, so với các mô hình học có nhãn khác, lợi ích đến từ phần nào của hàm mất mát, chuyển miền
giữa các bộ dữ liệu, và cuối cùng là kiểm tra trực tiếp xem phần vật lý tác động lên mô hình đã huấn luyện ra sao.

\section{Kiểm toán rò rỉ}\label{sec:leakres}

<<TL>>
<<FIGL>>

Bảng~\ref{tab:leak} cho thấy mức độ nghiêm trọng của từng dạng rò rỉ. Chia theo chu kỳ làm sai số thấp đi
<<RMIN>>--<<RMAX>> lần so với chia theo cell, dù mô hình và dữ liệu không đổi. Hàng cuối của bảng còn rõ hơn: một
bộ “dự đoán” không dùng đặc trưng nào, chỉ chép nhãn của chu kỳ train gần nhất trong cùng cell, đạt sai số
thấp hơn mọi mô hình thật được đánh giá đúng cách, đúng như Mệnh đề~\ref{md:rowleak} dự báo. <<MMTXT>><<SMALL>>Mọi
kết quả từ đây trở đi dùng giao thức chuẩn của Chương~\ref{ch:thietke}.

\section{\texorpdfstring{So sánh ở 30\,\% nhãn}{So sánh ở 30\% nhãn}}

Ở mức 30\,\% số cell có nhãn, PINN-semi có MAE theo cell thấp hơn MLP có ý nghĩa thống kê (Wilcoxon theo cell,
hiệu chỉnh Holm) <<SIG30>>, với tỉ số MAE từ <<R30LO>> (<<DSLO>>) tới <<R30HI>> (<<DSHI>>). <<NBTXT>>

<<FIGCELLS>>

\section{PINN và học có nhãn khi thay đổi lượng nhãn}\label{sec:labels}

Bảng~\ref{tab:big} và Hình~\ref{fig:curvephys}, \ref{fig:curvebase} là kết quả trung tâm của đồ án: sai số của
mọi mô hình ở bốn mức nhãn.

<<TBIG>>
<<FIGCP>>
<<FIGCB>>

\paragraph{Khi có ít nhãn.} Ở 10\,\% nhãn, PINN-semi vượt MLP với khoảng cách lớn nhất: tỉ số MAE là <<R10>>,
có ý nghĩa <<SIG10>>. Ở mức này mỗi fold chỉ có <<NLAB>> cell có nhãn, và MLP dễ quá khớp; loss vật lý trên các
cell không nhãn buộc mạng phải cho quỹ đạo hợp lý trên chính những cell nó chưa có nhãn, nên bù được phần lớn
thông tin thiếu.

\paragraph{Khi có nhiều nhãn.} Ở 70\,\% nhãn, mọi cell train đều có nhãn và không còn cell không nhãn, nên
PINN-semi và PINN-sup trùng nhau; vật lý lúc này chỉ còn vai trò một bộ chuẩn hoá. Tỉ số MAE so với MLP là
<<R70>>; <<SIG70>><<WORSE70>>. Trung bình nhân của tỉ số qua bốn bộ tiến dần về 1 khi lượng nhãn tăng:
<<GMTXT>>. Dữ liệu có nhãn càng đủ, ràng buộc vật lý càng ít thông tin mới.

\paragraph{Lượng nhãn tương đương.} Bảng~\ref{tab:equiv} quy đổi kết quả sang số nhãn tiết kiệm được. Một giá trị
“$> 70\,\%$” nghĩa là mô hình học có nhãn chưa đạt được sai số đó ngay cả khi dùng toàn bộ nhãn train; “$\le
10\,\%$” nghĩa là mô hình học có nhãn đã đạt sai số đó ngay ở mức nhãn thấp nhất. <<EQTXT>>

<<TEQ>>
<<FIGRATIO>>

<<TC>>

\section{So với các mô hình học có nhãn khác}

Mô hình có MAE thấp nhất ở từng bộ (Bảng~\ref{tab:big}): <<BEST>>.

<<RIDGE>><<GPR>>GBDT là đối thủ mạnh nhất: <<GBTXT>>. Cây quyết định xử lý tốt các đặc trưng có phân phối lệch và
các chế độ thí nghiệm khác hẳn nhau mà không cần chuẩn hoá cẩn thận; đây là ưu thế quen thuộc của mô hình cây trên
dữ liệu dạng bảng. Phát biểu đúng mức vì vậy là: vật lý cải thiện rõ rệt \emph{mô hình mạng nơ-ron} khi ít nhãn và
đưa nó tới gần mô hình cây ở 10\,\% nhãn (tỉ số so với GBDT <<RGB10>>), nhưng chưa vượt được một mô hình cây
được tinh chỉnh tốt, cũng như hồi quy Gauss, trên phần lớn các bộ dữ liệu.

\section{Lợi ích đến từ đâu}\label{sec:source}

<<TS>>

\begin{itemize}
\item \emph{Vật lý chỉ trên cell có nhãn} (PINN-sup): <<SUPTXT>>
\item \emph{Thêm cell không nhãn} (PINN-semi so với PINN-sup): <<UNLTXT>>
\item \emph{Phương trình động học so với chỉ đơn điệu}: <<MONOTXT>>
\item \emph{Dạng phần dư}: <<EUTXT>>
\item \emph{Động học hộp đen} (kiểu PINN4SOH) so với động học xám: <<BBTXT>>
\end{itemize}

\section{Các chỉ số mở rộng và hình dự đoán}

<<TP>>

<<R2TXT>> Hình~\ref{fig:parity}--\ref{fig:errsoh} cho thấy dự đoán bám sát thực đo trong phần lớn vòng đời,
PINN-semi cho quỹ đạo trơn hơn, và sai số tăng ở giai đoạn cuối đời.

<<FIGPAR>>
Hình~\ref{fig:traj} đặt cạnh nhau dự đoán cho cùng một cell khi lượng nhãn tăng từ 10 lên 30 và 70\,\%. Khi ít nhãn,
cả hai mô hình bám được hình dạng của quỹ đạo nhưng lệch mức, rõ nhất ở TJU và HUST; thêm nhãn chủ yếu làm giảm phần
lệch mức này chứ không làm thay đổi hình dạng. Ở 10\,\% nhãn, dự đoán của MLP dao động mạnh hơn và có những đoạn đi
lên, còn PINN-semi giữ được xu hướng giảm; khoảng cách giữa hai mô hình thu hẹp dần khi nhãn tăng.

<<FIGTRAJ>>
<<FIGERR>>
<<METRICS>>
\section{Chuyển miền}\label{sec:transferres}

<<TT>>
<<FIGTR>>

<<TRTXT>>

\section{Tham số vật lý học được}\label{sec:physparams}

<<TPH>>

<<PHTXT>>
<<PHYSCHECK>>

\section{Thảo luận chung}

\paragraph{Khi nào nên dùng PINN bán giám sát?} Khi đã chọn mạng nơ-ron làm mô hình, PINN bán giám sát nên được
dùng thay cho MLP khi số cell có nhãn ít (khoảng 10--30\,\%) và khi có nhiều cell chưa đo dung lượng. So với GBDT
thì khác: trong cùng một bộ dữ liệu, GBDT có sai số thấp hơn ở phần lớn các trường hợp.<<TRGBD>> Trong số các mô hình đơn lẻ, nếu chỉ cần
sai số thấp nhất trên cùng một bộ dữ liệu thì GBDT là lựa chọn đơn giản và mạnh hơn; Chương~\ref{ch:caitien} cho thấy
ghép GBDT với nhóm PINN còn tốt hơn cả hai.

\paragraph{Khi nào lợi ích nhỏ?} <<WEAKTXT>>

\paragraph{Hạn chế.} (i) Mô hình ước lượng SOH tại chu kỳ hiện tại, chưa dự báo SOH tương lai hay tuổi thọ còn
lại. (ii) Trọng số vật lý được chọn ở một thí nghiệm sơ bộ và giữ cố định; chưa tối ưu riêng cho từng mức nhãn.
(iii) GBDT là đối thủ mạnh; Chương~\ref{ch:caitien} kết hợp nó với PINN ở mức dự đoán, nhưng ràng buộc vật lý
chưa được đưa vào chính mô hình cây. (iv) Cell validation cần nhãn ngoài ngân
sách nhãn. (v) Ở 30\,\% nhãn, kết luận ở mức fold chỉ vững <<NB30>>. (vi) Tham số vật lý $\lambda, E_a$ chưa
ước lượng được từ dữ liệu hiện có. (vii) Ở dạng đạo hàm riêng, tốc độ $r$ học được chỉ phản ánh phần suy giảm mà
mạng gán cho biến thời gian, không phải tốc độ suy giảm thật của cell (Mục~\ref{sec:physcheck}).
""", TL=t['L'], FIGL=fig('fig_thesis_leakage', 'MAE theo cell của cùng một mô hình (MLP, 70\\,\\% nhãn) dưới các cách chia '
                                           'và chuẩn hoá khác nhau; vạch đứt là cách chia đúng (theo cell). Điểm càng lệch trái so với vạch, sai '
                                           'số càng bị đánh giá thấp giả tạo.', 'fig:leak'),
             RMIN=f(1 / max(row_r.values()), 1), RMAX=f(1 / min(row_r.values()), 1), MMTXT=mm_txt, SMALL=small_txt,
             SIG30=_tren(F['sig'][0.3]) or 'trên không bộ nào', R30LO=f(r30[lo30], 2), DSLO=lo30, R30HI=f(r30[hi30], 2),
             DSHI=hi30, NBTXT=nb_txt,
             FIGCELLS=fig('fig_E13_cells', 'MAE từng cell ở 30\\,\\% nhãn, trung bình 5 lần lặp: trục ngang MLP, trục dọc '
                          'PINN-semi; điểm dưới đường chéo (xanh lục) là cell mà PINN-semi có sai số thấp hơn.', 'fig:cells'),
             TBIG=t['BIG'],
             FIGCP=fig('fig_thesis_curve_physics', 'MAE theo cell theo lượng nhãn: MLP và các biến thể có vật lý. Dải là '
                       'khoảng tin cậy 95\\,\\% bootstrap theo cell; trục dọc thang log.', 'fig:curvephys'),
             FIGCB=fig('fig_thesis_curve_baselines', 'MAE theo cell theo lượng nhãn: PINN-semi so với các mô hình học có '
                       'nhãn.', 'fig:curvebase'),
             R10=_pairs(F['r'][0.1]), SIG10=_tren(F['sig'][0.1]) or 'trên không bộ nào',
             NLAB=f"{_rng(F['nlab10'].values(), 0)} ({_pairs(F['nlab10'], 0)})",
             R70=_pairs(F['r'][0.7]),
             SIG70=('cải thiện có ý nghĩa ' + _tren(F['sig'][0.7])) if F['sig'][0.7] else 'không còn cải thiện có ý nghĩa',
             WORSE70=(f'; {_tren(F["worse"][0.7])} MLP lại tốt hơn có ý nghĩa' if F['worse'][0.7] else ''),
             GMTXT=', '.join(f'{pc(fr)}: {f(F["gm"][fr], 2)}' for fr in FRS),
             EQTXT=eq_txt, TEQ=t['EQ'], TC=t['C'],
             FIGRATIO=fig('fig_thesis_ratio', 'Tỉ số MAE theo cell của PINN-semi so với MLP, GBDT và PINN-sup theo lượng '
                          'nhãn (vạch: khoảng tin cậy 95\\,\\%). Dưới 1: PINN-semi tốt hơn.', 'fig:ratio'),
             BEST=F['best_txt'], RIDGE=ridge_txt, RGB10=_rng(F['r_gb10'].values()), GPR=gpr_txt, GBTXT=gb_txt,
             TS=t['S'], SUPTXT=sup_txt, UNLTXT=unl_txt, MONOTXT=mono_txt, EUTXT=eu_txt, BBTXT=bb_txt,
             TP=t['P'], R2TXT=r2_txt,
             FIGPAR=fig('fig_pred_parity', 'SOH dự đoán so với thực đo trên tập test (30\\,\\% nhãn). Hàng trên: MLP; '
                        'hàng dưới: PINN-semi.', 'fig:parity'),
             FIGTRAJ=fig('fig_pred_traj', 'Quỹ đạo SOH của cùng một cell test mỗi bộ khi có 10, 30 và 70\\,\\% số cell '
                         'mang nhãn (lần lặp 0). Cell được chọn là cell có sai số MLP ở trung vị của bộ tại 30\\,\\% nhãn, '
                         'bỏ qua các cell có nhãn nhảy bất thường giữa hai chu kỳ; trục dọc theo dải SOH thực đo nên vài '
                         'chu kỳ đầu của XJTU vượt khỏi khung.', 'fig:traj', width=r'0.92\textwidth'),
             FIGERR=fig('fig_pred_error_soh', 'MAE theo từng khoảng SOH thực đo (30\\,\\% nhãn).', 'fig:errsoh'),
             TT=t['T'], FIGTR=fig('fig_thesis_transfer', 'MAE theo cell trên tập test của bộ đích (thang log); hàng: số cell '
                                  'đích có nhãn $k$, cột: mô hình; ô gạch chéo: cùng một bộ, không chuyển miền.',
                                  'fig:transfer', width=r'\textwidth'),
             TRTXT=tr_txt, TPH=t['PH'], METRICS=ch6_metrics(d, ctx), PHYSCHECK=ch6_physcheck(d, ctx),
             TRGBD=e21_discussion(d), PHTXT=ph_txt, WEAKTXT=weak_txt, NB30=_tren(nb30) or 'trên không bộ nào')


# ═════════════════════════════════════════════════════════════ CHƯƠNG 7
DEV_ARMS = [   # (tên trong e20_dev.csv, mô tả)
    ('ENS3+GBDT|trend10', '3 PINN-semi + GBDT, lọc quỹ đạo ($w = 10$)'),
    ('ENS3+GBDT', '3 PINN-semi + GBDT'),
    ('HYB', '1 PINN-semi + GBDT'),
    ('GBDT', 'GBDT'),
    ('KD03', 'chưng cất từ GBDT, $\\lambda_\\text{KD} = 0{,}3$'),
    ('ENS3|trend10', '3 PINN-semi, lọc quỹ đạo'),
    ('ENS3', '3 PINN-semi'),
    ('KD1', 'chưng cất từ GBDT, $\\lambda_\\text{KD} = 1$'),
    ('P_m0|trend10', '1 PINN-semi, lọc quỹ đạo'),
    ('P_m0|iso', '1 PINN-semi, hồi quy đơn điệu'),
    ('M2', 'đơn điệu mạnh hơn, $\\beta = 2$'),
    ('EU', 'thêm phần dư Euler, $\\alpha_E = 2$'),
    ('REL', 'thêm đặc trưng tương đối'),
]
E20_LBL = {'mlp': 'MLP', 'gbdt': 'GBDT', 'pinn': 'PINN-semi', 'pinn_lai': 'PINN-lai'}


def e20_facts(d):
    """Mọi con số của Chương 7, tính từ results/e20_*.csv."""
    s, ts, cf = d['e20_summary'], d['e20_tests'], d['e20_conformal']
    E = {}
    v = lambda m, fr, ds, col='MAE_cell': float(s[(s.model == m) & (s.frac == fr) & (s.dataset == ds)][col].iloc[0])
    E['v'] = v
    t = lambda o, fr, ds: ts[(ts.doi_chung == o) & (ts.frac == fr) & (ts.bo == ds)].iloc[0]
    E['t'] = t
    for o in ['pinn', 'gbdt', 'mlp', 'mlp_lai', 'pinn3', 'lai_tho', 'gbdt_loc']:
        sub = ts[ts.doi_chung == o]
        E[f'{o}_better'] = int((sub.ti_so < 1).sum())
        E[f'{o}_sig'] = int(((sub.ti_so < 1) & (sub.p_holm < 0.05)).sum())
        E[f'{o}_worse_sig'] = int(((sub.ti_so > 1) & (sub.p_holm < 0.05)).sum())
        E[f'{o}_n'] = len(sub)
        E[f'{o}_r'] = {fr: [float(t(o, fr, ds).ti_so) for ds in DS] for fr in FRS}
        E[f'{o}_gm'] = {fr: _gm(E[f'{o}_r'][fr]) for fr in FRS}
        E[f'{o}_worse_list'] = [(r.bo, r.frac) for r in sub.itertuples() if r.ti_so > 1 and r.p_holm < 0.05]
    # dev: tỉ số validation so với PINN-semi đơn lẻ
    dv = d['e20_dev'].groupby(['arm', 'dataset', 'frac']).val_MAE_cell.mean().unstack(['dataset', 'frac'])
    base = dv.loc['P_m0']
    arms = [a for a, _ in DEV_ARMS] + ['KD03+GBDT', 'ENS3+GBDT_w0.6']
    E['dev'] = {a: {fr: _gm([dv.loc[a, (ds, fr)] / base[(ds, fr)] for ds in DS]) for fr in (0.1, 0.5)}
                | {'all': _gm([dv.loc[a, c] / base[c] for c in dv.columns])} for a in arms if a in dv.index}
    rel = (dv.loc['REL'] / base)
    E['rel_worst'] = (f'{rel.idxmax()[0]} ở {pc(rel.idxmax()[1])} nhãn', float(rel.max()))
    # test, cùng hai mức nhãn để so với dev
    E['test_dev'] = _gm([E['pinn_r'][fr][j] for fr in (0.1, 0.5) for j in range(4)])
    E['cf'] = cf
    return E


def ch7_improve(d, ctx):
    if 'e20_summary' not in d:
        return ''
    E = e20_facts(d)
    v, t = E['v'], E['t']
    # bảng phân rã sai số ở 30 % nhãn
    dc = d['e20_error_decomp']
    rows = []
    for ds in DS:
        for m in ['mlp', 'pinn_semi', 'gbdt']:
            r = dc[(dc.dataset == ds) & (dc.model == m) & (dc.frac == 0.3)].iloc[0]
            rows.append([ds if m == 'mlp' else '', LBL[m], m3(r.MAE_cell), pc(r.lech_muc), pc(r.troi_cham), pc(r.dao_dong)])
    tab_dc = table(['Bộ', 'Mô hình', r'MAE theo cell ($\times10^{-3}$)', 'lệch mức', 'trôi chậm', 'dao động nhanh'], rows,
                   'llrrrr', r'Phân rã MSE theo cell ở 30\,\% nhãn thành ba phần (tỉ lệ trên tổng MSE).', 'tab:decomp')
    sub = dc[dc.frac == 0.3]
    fast = sub.dao_dong; slow = sub.lech_muc + sub.troi_cham
    fast_nomit = sub[sub.dataset != 'MIT'].dao_dong
    # bảng dev
    rows = []
    for a, lab in DEV_ARMS:
        if a in E['dev']:
            g = E['dev'][a]
            rows.append([lab, f(g[0.1], 3), f(g[0.5], 3), f(g['all'], 3)])
    tab_dev = table(['Hướng thử', r'10\,\% nhãn', r'50\,\% nhãn', 'chung'], rows, 'lrrr',
                    r'Giai đoạn phát triển: MAE theo cell trên tập validation chia cho MAE của PINN-semi đơn lẻ, trung bình '
                    r'nhân trên bốn bộ (lần lặp 0, 5 fold). Nhỏ hơn 1 là tốt hơn.', 'tab:dev')
    dev = E['dev']
    # bảng kết quả chính: MAE theo cell
    rows = []
    for ds in DS:
        for m in ['mlp', 'gbdt', 'pinn', 'pinn_lai']:
            vals = [v(m, fr, ds) for fr in FRS]
            best = [min(v(mm, fr, ds) for mm in ['mlp', 'gbdt', 'pinn', 'pinn_lai']) for fr in FRS]
            cells = [(r'\textbf{' + m3(x) + '}') if x == b else m3(x) for x, b in zip(vals, best)]
            rows.append([ds if m == 'mlp' else '', E20_LBL[m]] + cells)
    tab_main = table(['Bộ', 'Mô hình', r'10\,\%', r'30\,\%', r'50\,\%', r'70\,\%'], rows, 'llrrrr',
                     r'MAE theo cell ($\times10^{-3}$) trên test, lần lặp 0--2; in đậm là thấp nhất ở mỗi bộ và mức nhãn.',
                     'tab:e20')
    # bảng kiểm định
    def cellv(o, fr, ds):
        r = t(o, fr, ds)
        mark = r'$^{*}$' if r.p_holm < 0.05 else ''
        return f(r.ti_so, 2) + mark
    rows = [[ds if j == 0 else '', pc(fr)] + [cellv(o, fr, ds) for o in ['pinn', 'gbdt', 'mlp', 'mlp_lai']]
            for ds in DS for j, fr in enumerate(FRS)]
    tab_t = table(['Bộ', 'Nhãn', 'so với PINN-semi', 'so với GBDT', 'so với MLP', 'so với MLP-lai'], rows, 'llrrrr',
                  r'Tỉ số MAE theo cell của PINN-lai so với từng mô hình (nhỏ hơn 1 là PINN-lai tốt hơn). '
                  r'$^{*}$: Wilcoxon ghép cặp theo cell có ý nghĩa sau hiệu chỉnh Holm trên bốn bộ ($p < 0{,}05$).',
                  'tab:e20tests')
    # R², tăng ngược ở 30 %
    r2 = {m: [v(m, 0.3, ds, 'R2_10') for ds in DS] for m in ['mlp', 'gbdt', 'pinn', 'pinn_lai']}
    mono = {m: [v(m, 0.3, ds, 'tang_nguoc') for ds in DS] for m in ['mlp', 'gbdt', 'pinn', 'pinn_lai']}
    # các bước
    steps = {}
    for m in ['pinn3', 'lai_tho', 'pinn_lai']:
        steps[m] = _gm([v(m, fr, ds) / v('pinn', fr, ds) for fr in FRS for ds in DS])
    gb_alone = _gm([v('gbdt', fr, ds) / v('pinn', fr, ds) for fr in FRS for ds in DS])
    # khoảng dự đoán
    cf = E['cf']
    rows = []
    for ds in DS:
        for fr in FRS:
            a = cf[(cf.dataset == ds) & (cf.frac == fr) & (cf.kieu == 'co_dinh')].iloc[0]
            b = cf[(cf.dataset == ds) & (cf.frac == fr) & (cf.kieu == 'thich_nghi')].iloc[0]
            rows.append([ds if fr == FRS[0] else '', pc(fr), pc(a.phu, 1), f(100 * a.rong, 2), pc(b.phu, 1),
                         f(100 * b.rong, 2), pc(b.phu_cell_duoi80)])
    tab_cf = table(['Bộ', 'Nhãn', 'độ phủ', 'độ rộng', 'độ phủ', 'độ rộng', r'cell phủ $<80\,\%$'], rows, 'llrrrrr',
                   r'Khoảng dự đoán 90\,\% của PINN-lai trên test: độ phủ thực tế và độ rộng trung bình (\% SOH). Hai cột '
                   r'đầu là độ rộng cố định, hai cột sau là độ rộng thích nghi; cột cuối là tỉ lệ cell có độ phủ dưới '
                   r'80\,\% với dạng thích nghi.', 'tab:conformal')
    cov_a = cf[cf.kieu == 'co_dinh'].phu; cov_b = cf[cf.kieu == 'thich_nghi'].phu
    wid_a = cf[cf.kieu == 'co_dinh'].rong; wid_b = cf[cf.kieu == 'thich_nghi'].rong
    bc = d['e20_biascorr']
    chk = d['e20_check']
    worse_gb = E['gbdt_worse_list']
    gb10 = [ds for ds in DS if t('gbdt', 0.1, ds).ti_so < 1 and t('gbdt', 0.1, ds).p_holm < 0.05]
    gb10_txt = ('; riêng ở 10\\,\\% nhãn, PINN-lai tốt hơn GBDT có ý nghĩa ' + _tren(gb10)) if gb10 else ''
    lose = {}
    for ds in DS:
        for fr in FRS:
            if t('gbdt', fr, ds).ti_so > 1:
                lose.setdefault(ds, []).append(fr)
    gblose_txt = ('GBDT có sai số thấp hơn PINN-lai ở ' + _lst([f'{ds} ({_levels(frs)})' for ds, frs in lose.items()])
                  + ', phần lớn các trường hợp này không có ý nghĩa thống kê.') if lose else ''
    worse_txt = (('GBDT vẫn tốt hơn có ý nghĩa ở ' + _lst([f'{ds} {pc(fr)}' for ds, fr in worse_gb]) + '.')
                 if worse_gb else 'Không có trường hợp nào GBDT tốt hơn PINN-lai có ý nghĩa thống kê.')
    rmed = float(bc.rho.median())
    rho_word = ('tương quan yếu' if rmed < 0.4 else 'chỉ tương quan ở mức vừa' if rmed < 0.75 else 'tương quan khá chặt')
    rho_word2 = ('gần như không tương quan' if rmed < 0.4 else 'chỉ tương quan ở mức vừa' if rmed < 0.75
                 else 'tương quan khá chặt nhưng vẫn khác nhau ở nhiều cell')
    rho_conc = ('Vì hai mô hình không lệch cùng một kiểu, lấy trung bình của chúng triệt tiêu được một phần lệch mức mà '
                r'Mục~\ref{sec:errdecomp} chỉ ra.')
    ml = E['mlp_lai_r']
    phys_sig = E['mlp_lai_sig']; phys_worse = E['mlp_lai_worse_sig']
    lowfr_ml = _gm(ml[0.1]); hifr_ml = _gm(ml[0.7])
    if phys_sig >= 4 and lowfr_ml < hifr_ml:
        phys_conc = ('Phần vật lý vẫn đóng góp khi đã có GBDT, rõ nhất khi ít nhãn, và đóng góp ít dần khi nhãn nhiều lên, '
                     r'cùng xu hướng như ở Chương~\ref{ch:ketqua}.')
        phys_disc = ' nên PINN-lai hơn MLP-lai khi ít nhãn'
    elif phys_sig > phys_worse:
        phys_conc = ('Khi đã có GBDT, phần vật lý chỉ còn đóng góp ở một số trường hợp; phần lớn lợi ích của PINN-lai so với '
                     'PINN-semi đến từ việc kết hợp chứ không từ vật lý.')
        phys_disc = ', nhưng khi đã kết hợp với GBDT phần đóng góp này nhỏ'
    else:
        phys_conc = ('Khi đã có GBDT, thay PINN-semi bằng MLP không làm sai số tăng: trong mô hình lai, lợi ích đến từ '
                     'việc kết hợp chứ không từ vật lý trong hàm mất mát.')
        phys_disc = ', nhưng trong mô hình lai đóng góp này không còn đo được'
    cb = [x for x in cf[cf.kieu == 'thich_nghi'].phu]
    cov_conc = (', sát mức 90\\,\\% đặt ra' if np.mean(cb) >= 0.895 else
                (f', trung bình {f(100 * np.mean(cb), 1)}\\,\\%, tức thấp hơn mức 90\\,\\% đặt ra vài điểm phần trăm. '
                 'Thiếu hụt này có nguyên nhân rõ: tập hiệu chỉnh của mỗi fold chỉ gồm vài đến mười mấy cell, và phần dư '
                 'của các chu kỳ trong cùng một cell tương quan rất mạnh, nên số mẫu hiệu chỉnh thực chất là số cell chứ '
                 'không phải số chu kỳ; phân vị ước lượng từ chừng ấy cell dao động nhiều giữa các fold') if np.mean(cb) >= 0.85
                else ', thấp hơn rõ mức 90\\,\\% đặt ra')
    return T(r"""
\chapter{Cải tiến mô hình}\label{ch:caitien}

Chương~\ref{ch:ketqua} để lại hai kết quả trái chiều. PINN-semi giảm rõ sai số so với mạng cùng kiến trúc khi ít
nhãn, nhưng GBDT, một mô hình không biết gì về vật lý, vẫn có sai số thấp nhất ở phần lớn các trường hợp có từ
30\,\% nhãn trở lên. Mục~\ref{sec:physcheck} cũng cho thấy vật lý giúp chủ yếu bằng cách giữ quỹ đạo dự đoán gần
đơn điệu. Chương này tìm cách giảm sai số thêm mà không dùng thêm nhãn nào và không đổi giao thức đánh giá. Quy
trình gồm hai giai đoạn tách bạch: chọn hướng cải tiến chỉ trên tập validation, chốt cấu hình bằng một đề cương
được commit trước, rồi mới đánh giá trên test.

\section{Sai số còn lại nằm ở đâu}\label{sec:errdecomp}

Trước khi thử cải tiến, cần biết sai số hiện tại có dạng gì. Với mỗi cell test, sai số theo chu kỳ
$e_n = \hat y_n - y_n$ được tách thành ba phần:
\begin{equation}
e_n = \underbrace{b}_{\text{lệch mức}} + \underbrace{s_n}_{\text{trôi chậm}} + \underbrace{f_n}_{\text{dao động nhanh}},
\end{equation}
trong đó $b$ là trung bình của $e_n$ trên cả cell, $s_n$ là trung bình trượt căn giữa 21 điểm của $e_n - b$, còn $f_n$
là phần còn lại. Bảng~\ref{tab:decomp} cho tỉ lệ của từng phần trong MSE ở 30\,\% nhãn.

<<TABDC>>

Phần dao động nhanh, tức nhiễu giữa các chu kỳ liền nhau, chỉ chiếm <<FAST>> tổng MSE ngoài bộ MIT; trên MIT, nơi
sai số tuyệt đối rất nhỏ, phần này lên tới <<FASTMIT>>. Phần lớn sai số là lệch mức và trôi chậm: mô hình đoán sai
SOH của cả một cell, hoặc đoán sai tốc độ suy giảm trên một đoạn dài. Hai hệ quả rút ra. Một là làm trơn dự đoán chỉ
giúp được một phần nhỏ. Hai là muốn giảm sai số đáng kể phải giảm phần lệch theo từng cell, và cách đơn giản nhất là
lấy trung bình những mô hình có lệch theo cell khác nhau.

\section{Các hướng thử}\label{sec:candidates}

Sáu hướng được thử, mỗi hướng dựa trên một kết quả đã công bố hoặc một quan sát ở Chương~\ref{ch:ketqua}.

\paragraph{Nhóm mạng.} Huấn luyện $M = 3$ mạng PINN-semi giống hệt nhau, chỉ khác hạt giống khởi tạo và thứ tự lấy
batch, rồi lấy trung bình $\bar F = \frac{1}{M}\sum_m F^{(m)}$~\cite{lakshminarayanan2017}. Cả ba mạng dùng đúng
cùng cách chia và cùng tập cell có nhãn, nên không có nhãn nào được thêm vào.

\paragraph{Chưng cất từ GBDT.} GBDT học trên cell có nhãn rồi dự đoán cho mọi chu kỳ của cell không nhãn; dự đoán
này làm mục tiêu mềm cho mạng~\cite{hinton2015,lee2013}, thêm số hạng
$\lambda_\text{KD}\,\overline{(F_\varphi(\tilde{\mathbf{x}}, t) - g(\tilde{\mathbf{x}}, t))^2}$ trên cell không nhãn.

\paragraph{Kết hợp với GBDT.} Lấy trung bình dự đoán của nhóm mạng và của GBDT, $\hat y = \omega \bar F + (1-\omega) g$.
Đây là dạng đơn giản nhất của kết hợp mô hình khác loại~\cite{wolpert1992}. Lý do để kỳ vọng nó giúp là hai mô hình
có thiên lệch quy nạp khác hẳn nhau: cây quyết định học tốt các hàm không trơn trên dữ liệu bảng~\cite{grinsztajn2022},
còn PINN-semi dùng được cell không nhãn và cho quỹ đạo trơn, gần đơn điệu.

\paragraph{Bộ lọc quỹ đạo.} Vật lý cũng có thể được đưa vào ở bước suy luận chứ không chỉ khi huấn luyện. Tại chu kỳ
$n$ của một cell, lấy $w$ dự đoán gần nhất $\hat y_{n-w+1}, \dots, \hat y_n$, khớp một đường thẳng bằng bình phương
tối thiểu~\cite{cleveland1979} với ràng buộc độ dốc không dương, rồi lấy giá trị đường thẳng tại $n$:
\begin{equation}\label{eq:trend}
\tilde y_n = \bar y + \min(0, \hat\beta)\,(t_n - \bar t), \qquad
\hat\beta = \frac{\sum_i (t_i - \bar t)(\hat y_i - \bar y)}{\sum_i (t_i - \bar t)^2},
\end{equation}
với $\bar t, \bar y$ là trung bình trên cửa sổ. Bộ lọc chỉ dùng dự đoán của các chu kỳ đã qua, không dùng nhãn,
nên chạy được trực tuyến. Một lựa chọn tự nhiên khác là hồi quy đơn điệu~\cite{barlow1972} trên toàn bộ quá khứ,
nhưng Mệnh đề~\ref{prop:filter} cho thấy nó lệch xuống.

\paragraph{Ràng buộc vật lý mạnh hơn.} Tăng trọng số đơn điệu lên $\beta = 2$, hoặc thêm phần dư Euler dọc quỹ đạo
(Mục~\ref{sec:physcheck}) với hệ số $\alpha_E = 2$ bên cạnh phần dư dạng đạo hàm riêng.

\paragraph{Đặc trưng tương đối.} Nối thêm vào đầu vào độ thay đổi tương đối của đặc trưng so với trung bình ba chu
kỳ đầu của chính cell, $(\mathbf{x}_n - \mathbf{x}_\text{ref})/|\mathbf{x}_\text{ref}|$, theo tinh thần của đặc
trưng $\Delta Q(V)$ trong~\cite{severson2019}: suy giảm thể hiện ở chỗ đường sạc đã đổi bao nhiêu so với lúc mới.

\section{Chọn hướng trên tập validation}\label{sec:devproto}

Giai đoạn phát triển dùng lần lặp 0, cả 5 fold, hai mức nhãn 10\,\% và 50\,\%. Mỗi fold có khoảng 10\,\% số cell
làm validation, tách khỏi cả train lẫn test; mọi quyết định chỉ dựa trên MAE theo cell của tập này. Bảng~\ref{tab:dev}
tóm tắt kết quả.

<<TABDEV>>

Ba nhóm kết quả hiện ra. Kết hợp với GBDT cho mức giảm lớn nhất: ở 10\,\% nhãn tỉ số chỉ còn <<D_HYB10>>, ở 50\,\%
nhãn còn <<D_HYB50>>, thấp hơn cả GBDT đứng một mình (<<D_GB>>). Nhóm mạng và chưng cất cho mức giảm vừa phải, khoảng
<<D_ENS>> và <<D_KD>>; khi đã kết hợp với GBDT, chưng cất không còn thêm lợi ích (tỉ số <<D_KDG>> so với <<D_HYB1>>
của một mạng PINN-semi kết hợp GBDT), nên bị bỏ để mô hình gọn hơn. Ba hướng còn lại làm sai số tăng: ràng buộc đơn điệu mạnh hơn
(<<D_M2>>), thêm phần dư Euler (<<D_EU>>) và nhất là đặc trưng tương đối (<<D_REL>>). Phần dư Euler buộc $r$ bằng tốc
độ dọc quỹ đạo trong khi phần dư dạng đạo hàm riêng buộc $\partial F/\partial t = -r$, nên hai số hạng kéo mạng
nghiệm về hai phía. Đặc trưng tương đối chia cho $|\mathbf{x}_\text{ref}|$, nên khuếch đại các đặc trưng có giá trị
ban đầu gần 0 và làm phân phối đầu vào khác hẳn giữa các cell; trường hợp tệ nhất là <<RELW>>, sai số tăng
<<RELX>> lần.

Trọng số $\omega = 0{,}5$ và $0{,}4$ cho GBDT trọng số $0{,}5$ và $0{,}6$, kết quả gần như nhau trên validation
(<<D_W5>> và <<D_W6>>), nên chọn $\omega = 0{,}5$
để không phải ước lượng thêm tham số. Cửa sổ lọc $w = 5, 10, 20$ cũng gần như nhau; chọn $w = 10$ vì nó gần tốt nhất
với mọi mô hình nền đã thử. Cấu hình cuối được ấn định thành văn bản trước khi đọc bất kỳ dự đoán test nào.

\section{Mô hình PINN-lai}\label{sec:pinnlai}

Mô hình cải tiến, gọi là PINN-lai, gồm ba bước:
\begin{align}
\bar F(\mathbf{x}, t) &= \tfrac{1}{3}\textstyle\sum_{m=1}^{3} F_{\varphi_m}(\tilde{\mathbf{x}}, t), \label{eq:lai1}\\
\hat y_n &= \tfrac{1}{2}\,\bar F(\mathbf{x}_n, t_n) + \tfrac{1}{2}\,g(\mathbf{x}_n, t_n), \label{eq:lai2}\\
\tilde y_n &= \text{bộ lọc \eqref{eq:trend} với } w = 10 \text{ trên } \hat y_{n-9}, \dots, \hat y_n, \label{eq:lai3}
\end{align}
trong đó $F_{\varphi_m}$ là ba mạng PINN-semi cấu hình như Mục~\ref{sec:pinn} và $g$ là GBDT như
Mục~\ref{sec:baselines}, cùng học trên đúng các cell
có nhãn của fold. Huấn luyện tốn gấp ba PINN-semi cộng một GBDT; suy luận vẫn chỉ cần đặc trưng sạc của chu kỳ hiện
tại và chín dự đoán trước đó của chính cell.

\begin{menhde}\label{prop:filter}
Xét một cửa sổ $w$ điểm cách đều.
(i) Nếu $\hat y_i = a - c\,t_i$ với $c \ge 0$ thì $\tilde y_n = \hat y_n$: bộ lọc không làm lệch một quỹ đạo giảm
tuyến tính.
(ii) Nếu $\hat y_i = a - c\,t_i + \epsilon_i$ với $\epsilon_i$ độc lập, kỳ vọng 0, phương sai $\sigma^2$, và ràng
buộc độ dốc không kích hoạt, thì $\tilde y_n$ không chệch và
$\operatorname{Var}(\tilde y_n) = \sigma^2\,\dfrac{2(2w-1)}{w(w+1)}$; với $w = 10$ là $0{,}345\,\sigma^2$.
(iii) Giá trị cuối của hồi quy đơn điệu giảm trên $\hat y_1, \dots, \hat y_n$ luôn không lớn hơn $\hat y_n$, nên
khi $\hat y$ có nhiễu nó lệch xuống.
\end{menhde}
\begin{proof}
(i) Đường thẳng khớp đúng dữ liệu, độ dốc $-c \le 0$ nên ràng buộc không tác động. (ii) Đây là phương sai của giá
trị dự đoán bình phương tối thiểu tại đầu mút: $\sigma^2\big(\frac{1}{w} + \frac{(t_n - \bar t)^2}{S_{tt}}\big)$
với $t_n - \bar t = \frac{w-1}{2}$ và $S_{tt} = \frac{w(w^2-1)}{12}$ khi bước bằng 1; rút gọn được biểu thức trên.
(iii) Với hồi quy đơn điệu giảm, giá trị tại điểm cuối bằng $\min_{j \le n}$ trung bình của $\hat y_j, \dots, \hat y_n$
\cite{barlow1972}; chọn $j = n$ cho thấy nó không vượt $\hat y_n$.
\end{proof}

\section{Kết quả trên tập test}\label{sec:e20res}

Đánh giá cuối dùng lần lặp 0, 1 và 2, cả 5 fold, 4 bộ và 4 mức nhãn. Mọi mô hình trong chương này, kể cả các mô
hình cũ, được tính lại trên đúng ba lần lặp đó. Thành viên thứ nhất của nhóm mạng chạy lại đúng cấu hình PINN-semi
và cho dự đoán trùng từng bit với kết quả ở Mục~\ref{sec:labels} ở <<CHK_P>> lượt; GBDT trùng ở <<CHK_G>> lượt.

<<TABMAIN>>
<<FIGCURVE>>

PINN-lai có MAE thấp hơn PINN-semi ở <<B_PINN>>/<<N_PINN>> trường hợp (bộ × mức nhãn), <<S_PINN>> trong đó có ý nghĩa
thống kê, với tỉ số <<R_PINN>>. So với GBDT, PINN-lai tốt hơn ở <<B_GB>>/<<N_GB>> trường hợp, <<S_GB>> có ý nghĩa<<GB10>>.
<<GBLOSE>> <<WORSE_GB>>
So với MLP, tỉ số là <<R_MLP>>; ở 10\,\% nhãn sai số chỉ còn <<R_MLP10>> của MLP. Trên hai mức nhãn đã dùng khi phát
triển (10 và 50\,\%), tỉ số so với PINN-semi trên test là <<TESTDEV>>, gần với <<DEVSEL>> trên validation, nên việc
chọn trên validation không làm kết quả test bị lạc quan đáng kể.

<<TABT>>

Các chỉ số khác cho cùng một bức tranh. Ở 30\,\% nhãn, $R^2$ tính từ chu kỳ 10 của PINN-lai là <<R2LAI>> trên bốn bộ,
so với <<R2GB>> của GBDT và <<R2PINN>> của PINN-semi. Tỉ lệ cặp chu kỳ có dự đoán tăng ngược của PINN-lai là
<<MONOLAI>>\,\%, thấp hơn PINN-semi (<<MONOPINN>>\,\%) và GBDT (<<MONOGB>>\,\%).

\section{Phân rã lợi ích}\label{sec:e20steps}

Hình~\ref{fig:e20steps} đi qua từng bước. Tính trung bình nhân trên 16 trường hợp, nhóm ba mạng đưa tỉ số so với
PINN-semi xuống <<ST_ENS>>, kết hợp với GBDT xuống <<ST_LAI>>, và bộ lọc quỹ đạo xuống <<ST_FIN>>; GBDT đứng một mình
là <<ST_GB>>. Bước kết hợp mang lại phần lớn lợi ích.

<<FIGSTEPS>>

Lệch mức theo cell của trung bình ba PINN-semi và của GBDT <<RHOWORD>>: hệ số Pearson <<RHO>> tùy bộ và mức nhãn,
và ở <<NGC>> số cell hai mô hình lệch ngược dấu nhau. <<RHOCONC>>

Câu hỏi còn lại là vật lý có còn cần khi đã có GBDT không. Để trả lời, cùng công thức được áp với ba mạng MLP thay
cho ba mạng PINN-semi (MLP-lai): cùng GBDT, cùng bộ lọc. PINN-lai tốt hơn MLP-lai có ý nghĩa ở <<PHYS_SIG>>/16 trường
hợp và kém hơn có ý nghĩa ở <<PHYS_WORSE>>; tỉ số trung bình nhân là <<PHYS10>> ở 10\,\% nhãn và <<PHYS70>> ở
70\,\% nhãn. <<PHYSCONC>>

\section{Khoảng dự đoán}\label{sec:conformal}

Một con số SOH đơn lẻ chưa đủ cho các quyết định như thay pin hay giảm công suất~\cite{thelen2024}. PINN-lai được
gắn thêm khoảng dự đoán bằng phương pháp dự đoán phù hợp tách tập~\cite{vovk2005,lei2018}. Trong mỗi fold, phần dư
$|\tilde y - y|$ được tính trên tập validation của chính fold đó; với mức $1 - \alpha = 90\,\%$, lấy phân vị
$\lceil (n+1)(1-\alpha) \rceil / n$ của các phần dư làm nửa độ rộng $q$, và khoảng cho một chu kỳ test là
$[\tilde y - q, \tilde y + q]$. Dạng thích nghi chia phần dư cho $\sigma_n + \delta$, với
$\sigma_n = |\bar F - g|$ là độ vênh giữa nhóm mạng và GBDT tại chu kỳ đó và $\delta$ là trung vị của $\sigma$ trên
validation; khoảng khi đó rộng ở những chu kỳ hai mô hình không đồng ý với nhau.

<<TABCF>>
<<FIGCF>>

Độ phủ thực tế trên test là <<COVA>> với độ rộng cố định và <<COVB>> với độ rộng thích nghi<<COVCONC>>.
Độ rộng trung bình của dạng thích nghi là <<WIDB>>\,\% SOH, so với <<WIDA>>\,\% của dạng cố định. Bảo đảm của phương
pháp là bảo đảm trung bình: nếu các cell hiệu chỉnh và cell test hoán đổi được cho nhau thì tỉ lệ chu kỳ nằm trong
khoảng xấp xỉ 90\,\%, nhưng không phải cell nào cũng được phủ 90\,\%. Với độ rộng thích nghi, <<C80>> số cell có độ
phủ dưới 80\,\%: đó là các cell bị lệch mức cả đời, đúng loại sai số chiếm phần lớn MSE. Hình~\ref{fig:e20traj} đặt PINN-lai cạnh PINN-semi trên đúng các cell của Hình~\ref{fig:traj}. Ở cả ba mức
nhãn, dự đoán của PINN-lai trơn hơn và ít đoạn đi lên hơn; trên TJU và HUST nó kéo phần lệch mức về gần thực đo hơn,
còn trên MIT hai mô hình gần như trùng nhau vì sai số vốn đã rất nhỏ. Dải tô quanh PINN-lai hẹp lại khi lượng nhãn
tăng, đúng như hàng dưới của Hình~\ref{fig:conformal}.

<<FIGTRAJ>>

\section{Thảo luận}\label{sec:e20disc}

Cải tiến lớn nhất không đến từ một ràng buộc vật lý mới mà từ việc ghép hai mô hình có điểm mạnh khác nhau. Kết quả
này khớp với hai quan sát: sai số còn lại chủ yếu là lệch theo từng cell, và lệch của PINN-semi với của GBDT
<<RHOWORD2>>. Vật lý có vai trò ở hai chỗ: trong huấn luyện, nó giúp nhóm mạng dùng được cell không nhãn<<PHYSDISC>>;
khi suy luận, bộ lọc quỹ đạo dùng đúng giả định SOH không tăng để làm trơn dự đoán mà không làm lệch quỹ đạo giảm. Hai hướng ràng buộc mạnh hơn thất bại cho thấy thêm vật lý vào hàm mất mát không phải
lúc nào cũng tốt: khi dạng phương trình còn thô, ép nó chặt hơn làm mạng mất tự do để khớp dữ liệu.

Về giới hạn: đánh giá cuối dùng ba lần lặp thay vì năm vì chi phí tính toán, và giai đoạn phát triển dùng cell
validation của lần lặp 0, vốn nằm trong test ở fold khác; thiên lệch chọn này nhỏ vì mức cải thiện trên test gần
với trên validation.
""",
             TABDC=tab_dc, FAST=_rng([100 * x for x in fast_nomit], 0) + r'\,\%',
             FASTMIT=_rng([100 * x for x in sub[sub.dataset == 'MIT'].dao_dong], 0) + r'\,\%',
             TABDEV=tab_dev, D_HYB10=f(dev['ENS3+GBDT|trend10'][0.1], 3), D_HYB50=f(dev['ENS3+GBDT|trend10'][0.5], 3),
             D_GB=f(dev['GBDT']['all'], 3), D_ENS=f(dev['ENS3']['all'], 3), D_KD=f(dev['KD03']['all'], 3),
             D_M2=f(dev['M2']['all'], 3), D_EU=f(dev['EU']['all'], 3), D_REL=f(dev['REL']['all'], 3),
             D_KDG=f(dev['KD03+GBDT']['all'], 3), D_HYB1=f(dev['HYB']['all'], 3),
             D_W5=f(dev['ENS3+GBDT']['all'], 3), D_W6=f(dev['ENS3+GBDT_w0.6']['all'], 3),
             RELW=E['rel_worst'][0], RELX=f(E['rel_worst'][1], 1),
             CHK_P=f'{int(chk.pinn_same.sum())}/{len(chk)}', CHK_G=f'{int(chk.gbdt_same.sum())}/{len(chk)}',
             TABMAIN=tab_main,
             FIGCURVE=fig('fig_e20_curve', 'MAE theo cell trên test theo lượng nhãn (thang log, lần lặp 0--2).',
                          'fig:e20curve'),
             B_PINN=E['pinn_better'], N_PINN=E['pinn_n'], S_PINN=E['pinn_sig'],
             R_PINN=_rng([x for fr in FRS for x in E['pinn_r'][fr]]),
             B_GB=E['gbdt_better'], N_GB=E['gbdt_n'], S_GB=E['gbdt_sig'], WORSE_GB=worse_txt, GB10=gb10_txt,
             GBLOSE=gblose_txt,
             R_MLP=_rng([x for fr in FRS for x in E['mlp_r'][fr]]), R_MLP10=_rng(E['mlp_r'][0.1]),
             TESTDEV=f(E['test_dev'], 3), DEVSEL=f(dev['ENS3+GBDT|trend10']['all'], 3),
             TABT=tab_t, R2LAI=_rng(r2['pinn_lai'], 3), R2GB=_rng(r2['gbdt'], 3), R2PINN=_rng(r2['pinn'], 3),
             MONOLAI=_rng(mono['pinn_lai'], 1), MONOPINN=_rng(mono['pinn'], 1), MONOGB=_rng(mono['gbdt'], 1),
             ST_ENS=f(steps['pinn3'], 3), ST_LAI=f(steps['lai_tho'], 3), ST_FIN=f(steps['pinn_lai'], 3),
             ST_GB=f(gb_alone, 3),
             FIGSTEPS=fig('fig_e20_steps', 'Tỉ số MAE theo cell so với PINN-semi qua từng bước của PINN-lai; mỗi đường là '
                          'một mức nhãn; cột bên phải là GBDT đứng một mình, mỗi hình thoi một mức nhãn (nhạt: 10\\,\\%, đậm: 70\\,\\%).',
                          'fig:e20steps'),
             RHO=_rng(bc.rho, 2), NGC=_rng([100 * x for x in bc.ti_le_ngc_dau], 0) + r'\,\%',
             RHOWORD=rho_word, RHOWORD2=rho_word2, RHOCONC=rho_conc, PHYSCONC=phys_conc, PHYSDISC=phys_disc,
             COVCONC=cov_conc,
             PHYS_SIG=phys_sig, PHYS_WORSE=phys_worse, PHYS10=f(lowfr_ml, 3), PHYS70=f(hifr_ml, 3),
             TABCF=tab_cf, C80=_rng([100 * x for x in cf[cf.kieu == 'thich_nghi'].phu_cell_duoi80], 0) + r'\,\%',
             FIGCF=fig('fig_e20_conformal', 'Độ phủ thực tế (hàng trên) và độ rộng trung bình (hàng dưới) của khoảng dự đoán '
                       '90\\,\\% trên test theo lượng nhãn; vạch đứt là mức 90\\,\\% đặt ra.', 'fig:conformal'),
             COVA=_rng([100 * x for x in cov_a], 1) + r'\,\%', COVB=_rng([100 * x for x in cov_b], 1) + r'\,\%',
             WIDA=_rng([100 * x for x in wid_a], 2), WIDB=_rng([100 * x for x in wid_b], 2),
             FIGTRAJ=fig('fig_e20_traj', 'Quỹ đạo SOH thực đo, dự đoán của PINN-semi và của PINN-lai trên cùng các cell '
                         'của Hình~\\ref{fig:traj} ở 10, 30 và 70\\,\\% nhãn (lần lặp 0). Dải tô là khoảng 90\\,\\% dạng '
                         'độ rộng cố định; trục dọc theo dải SOH thực đo.',
                         'fig:e20traj', width=r'0.92\textwidth'))


def ch6_concl(d):
    if 'e20_summary' not in d:
        return ''
    E = e20_facts(d)
    cf = E['cf']; tb = cf[cf.kieu == 'thich_nghi']
    return T(r"""\item Có. PINN-lai, nhóm ba PINN-semi kết hợp GBDT và bộ lọc quỹ đạo, có MAE thấp hơn PINN-semi ở
      <<B>>/<<N>> trường hợp (tỉ số <<R>>) và thấp hơn GBDT ở <<BG>>/<<N>> trường hợp, <<SG>> có ý nghĩa thống kê;
      GBDT chỉ còn tốt hơn có ý nghĩa ở <<WG>> trường hợp.
      Phần lớn lợi ích đến từ việc kết hợp hai mô hình có lệch theo cell chỉ tương quan ở mức vừa; phần vật lý vẫn giúp khi ít
      nhãn (so với cùng công thức dùng MLP, tỉ số <<PH>> ở 10\,\% nhãn). Khoảng dự đoán phù hợp 90\,\% đạt độ phủ
      <<COV>> trên test, nhưng chỉ theo nghĩa trung bình trên các chu kỳ, không phải cho từng cell.
""", B=E['pinn_better'], N=E['pinn_n'], R=_rng([x for fr in FRS for x in E['pinn_r'][fr]]), BG=E['gbdt_better'],
             SG=E['gbdt_sig'], WG=E['gbdt_worse_sig'], PH=f(_gm(E['mlp_lai_r'][0.1]), 2), COV=_rng([100 * x for x in tb.phu], 0) + r'\,\%')


def ch8(d, ctx):
    F = ctx['F']
    w = F['wins']
    ch3 = (f'PINN-semi có MAE thấp hơn MLP trong {w["mlp"][0]}/{w["mlp"][1]}, thấp hơn Ridge trong '
           f'{w["ridge"][0]}/{w["ridge"][1]}' + (f' và GPR trong {w["gpr"][0]}/{w["gpr"][1]}' if w['gpr'][1] else '')
           + ' trường hợp (bộ × mức nhãn), nhưng GBDT tốt hơn PINN-semi trong '
           f'{w["gbdt"][1] - w["gbdt"][0]}/{w["gbdt"][1]} trường hợp, {F["n_gb_sig"]} trong đó có ý nghĩa thống kê. '
           'GBDT được tinh chỉnh tốt vẫn là mô hình mạnh nhất khi đánh giá trong cùng một bộ dữ liệu.')
    bb = (f' Động học xám tốt hơn động học hộp đen kiểu PINN4SOH có ý nghĩa ở {F["bb_nsemi"]} trong 16 trường hợp '
          f'(bộ × mức nhãn) và kém hơn ở {F["bb_nsig"] - F["bb_nsemi"]}; khác biệt rõ nhất ở {F["bb_max"][0]} khi ít '
          'nhãn.' if F['bb_have'] else '')
    Tr = F['tr']
    ch5 = e21_concl(d) or (f'Khi bộ đích không có nhãn, PINN-semi giữ MAE ở mức {_rng(F["tr_k0_pinn"], 3)} trong khi MLP lệch tới '
           f'{_rng(F["tr_k0_mlp"])}; với 3 cell đích có nhãn, khoảng cách thu hẹp nhưng PINN-semi vẫn tốt hơn ở '
           f'{F["tr_k3_win"]}/{F["tr_n"]} cặp.' + (f' So với GBDT, PINN-semi tốt hơn ở {F["tr_gb0_win"]}/{F["tr_gb0_n"]} cặp '
                                                   f'khi $k = 0$ và {F["tr_gb3_win"]}/{F["tr_gb3_n"]} cặp khi $k = 3$.'
                                                   if F.get('tr_gb0_n') else '')) if F['tr_n'] else ''
    return T(r"""
\chapter{Kết luận và hướng phát triển}\label{ch:ketluan}

\section{Kết luận}

Đồ án xây dựng một mô hình PINN bán giám sát cho bài toán ước lượng SOH của cell lithium-ion và đánh giá nó bằng
một giao thức không rò rỉ trên bốn bộ dữ liệu công khai. Trả lời các câu hỏi nghiên cứu:

\begin{enumerate}[label=\textbf{CH\arabic*.}]
\item Cùng 30\,\% nhãn, PINN-semi có sai số thấp hơn MLP có ý nghĩa thống kê <<SIG30>> (tỉ số MAE <<R30>>).
\item Lợi ích lớn nhất khi ít nhãn (10\,\%: tỉ số <<R10>>, có ý nghĩa <<SIG10>>) và giảm dần khi lượng nhãn tăng
      (trung bình nhân qua bốn bộ: <<GM>>). Ở 70\,\% nhãn, khi không còn cell không nhãn, <<SIG70>><<WORSE70>>.
\item <<CH3>>
\item Ở 10\,\% nhãn, <<SHARE>>\,\% mức giảm sai số (thang log) của PINN-semi so với MLP đến từ việc áp loss vật
      lý lên cell không nhãn, phần còn lại từ vật lý trên cell có nhãn. Ở 30\,\% nhãn, riêng ràng buộc đơn điệu đã
      mang lại <<MONO>>\,\% mức giảm đó.<<BB>> Đo trực tiếp trên mô hình đã huấn luyện, vật lý tác động chủ yếu qua
      ràng buộc dọc quỹ đạo: dự đoán của PINN-semi gần đơn điệu như nhãn thật; tốc độ $r$ ở dạng đạo hàm riêng chỉ
      phản ánh phần suy giảm gán cho biến thời gian, còn dạng Euler cho $r$ sát tốc độ thật (Mục~\ref{sec:physcheck}).
\item <<CH5>>
<<CH6>><<CH7>>\end{enumerate}

Về phương pháp đánh giá, đồ án cho thấy cách chia theo chu kỳ làm sai số trông thấp hơn thực tế, và chứng minh bằng
một cận đơn giản rằng ngay cả bộ dự đoán không học gì cũng đạt sai số rất thấp dưới cách chia đó. Giao thức kiểm
định chéo 5 fold theo cell, phân tầng theo chế độ thí nghiệm, lặp 5 lần, được chọn từ các ràng buộc rõ ràng và có
các tính chất được chứng minh: mỗi cell được test đúng một lần mỗi lần lặp, các fold cân bằng theo từng chế độ, tập
nhãn lồng nhau giữa các mức, và ước lượng sai số không chệch.

\section{Hướng phát triển}

\begin{enumerate}
\item Đưa ràng buộc vật lý vào chính mô hình cây, ví dụ dùng loss đơn điệu và loss động học để tạo nhãn giả cho
      GBDT trên cell không nhãn, thay vì chỉ lấy trung bình dự đoán như PINN-lai.
\item Cập nhật ước lượng và dự báo khi có thêm một vài lần đo dung lượng của chính cell, ví dụ dùng lớp nghiệm đóng
      làm mô hình trạng thái cho một bộ lọc Bayes; nhờ đó khoảng dự đoán và khoảng dự báo có độ phủ đúng cho từng cell
      chứ không chỉ trung bình.
\item Dùng điện lượng tích luỹ thay cho chỉ số chu kỳ làm biến thời gian, để chuyển giữa các giao thức sạc tốt hơn.
\item Kết hợp phần dư dạng đạo hàm riêng với dạng Euler dọc quỹ đạo, tăng trọng số của phần dư động học và dùng dữ
      liệu nhiều mức nhiệt độ, để tốc độ $r$, $\lambda$ và $E_a$ diễn giải được theo nghĩa vật lý mà vẫn giữ sai số thấp.
\end{enumerate}
""", SIG30=_tren(F['sig'][0.3]) or 'trên không bộ nào', R30=_rng(F['r'][0.3].values()),
             R10=_rng(F['r'][0.1].values()), SIG10=_tren(F['sig'][0.1]) or 'trên không bộ nào',
             GM=', '.join(f'{pc(fr)}: {f(F["gm"][fr], 2)}' for fr in FRS),
             SIG70=('cải thiện chỉ còn có ý nghĩa ' + _tren(F['sig'][0.7])) if F['sig'][0.7]
             else 'không còn cải thiện có ý nghĩa',
             WORSE70=(f', và {_tren(F["worse"][0.7])} MLP tốt hơn' if F['worse'][0.7] else ''),
             CH3=ch3, SHARE=_rng([100 * x for x in F['share_unl10']], 0), MONO=_rng([100 * x for x in F['share_mono']], 0),
             BB=bb, CH5=ch5, CH6=ch6_concl(d), CH7=e22_concl(d))


# ═════════════════════════════════════════════════════════════ TÓM TẮT, PHỤ LỤC, DỰNG
def abstract_e21(d):
    if 'e21_base' not in d:
        return ''
    import thesis_bk as BK
    E = BK.e21_facts(d)
    t = (f' Khi học trên một bộ dữ liệu và kiểm tra trên ba bộ còn lại, sai số tăng {f(E["gap"]["ket_hop"], 1)} lần so '
         f'với trong cùng bộ, chủ yếu do lệch mức')
    if 'r0' in E:
        t += (f'; thích nghi không nhãn giúp ở {E["r0_better"]}/12 cặp nhưng làm kém đi ở {E["r0_worse"]} cặp, '
              f'còn học lặp bằng nhãn giả hội tụ về chính dự đoán ban đầu')
    return t + '.'


def abstract_e20(d):
    if 'e20_summary' not in d:
        return ''
    E = e20_facts(d)
    return T(r"""Từ phân tích sai số, đồ án đề xuất mô hình PINN-lai: ba mạng PINN bán giám sát kết hợp với GBDT, rồi qua
một bộ lọc quỹ đạo nhân quả không cho SOH tăng. Cấu hình được chọn chỉ trên tập validation; trên test, PINN-lai có
sai số bằng <<R>> lần PINN ban đầu và thấp hơn GBDT ở <<BG>>/<<N>> trường hợp bộ dữ liệu × mức nhãn, kèm khoảng
dự đoán 90\,\% có độ phủ thực tế <<COV>> trên test.""", R=_rng([x for fr in FRS for x in E['pinn_r'][fr]]),
             BG=E['gbdt_better'], N=E['gbdt_n'],
             COV=_rng([100 * x for x in E['cf'][E['cf'].kieu == 'thich_nghi'].phu], 0) + r'\,\%')


def abstract(d, ctx):
    F = ctx['F']
    gb = ('một mô hình gradient boosting được tinh chỉnh tốt vẫn có sai số thấp nhất trên phần lớn các bộ dữ liệu'
          if len(F['gb_best30']) >= 3 else
          'một mô hình gradient boosting được tinh chỉnh tốt vẫn cạnh tranh mạnh trên một số bộ dữ liệu')
    return T(r"""
\chapter*{Tóm tắt}\addcontentsline{toc}{chapter}{Tóm tắt}
Đồ án nghiên cứu bài toán ước lượng trạng thái sức khoẻ (SOH) của cell lithium-ion khi chỉ một phần nhỏ cell có
nhãn dung lượng. Mô hình đề xuất là một mạng nơ-ron có thông tin vật lý (PINN) bán giám sát: mạng nghiệm dự đoán
SOH từ 16 đặc trưng đoạn sạc, gắn với một động học suy giảm dạng xám luôn không âm, có hệ số đầu gối và hệ số
Arrhenius; ba hàm mất mát vật lý không dùng nhãn nên áp được lên cell chưa đo dung lượng. Đồ án thiết kế và chứng
minh một giao thức đánh giá không rò rỉ (kiểm định chéo 5 fold theo cell, phân tầng theo chế độ thí nghiệm, lặp 5
lần), lý giải từng con số của cách chia, đo định lượng tác động của các dạng rò rỉ thường gặp, và so sánh PINN với
sáu mô hình đối chứng ở bốn mức nhãn trên bốn bộ dữ liệu công khai (<<NCELL>> cell, <<NRUN>> lượt huấn luyện).
Ở 10\,\% nhãn, tỉ số sai số giữa PINN bán giám sát và mạng nơ-ron cùng kiến trúc chỉ học có nhãn là <<R10>>, trong
đó khoảng một nửa đến hai phần ba mức giảm đến từ cell không nhãn; khi tăng lên 70\,\% nhãn, lợi ích giảm dần (tỉ số <<R70>>). Tuy vậy, <<GB>>.<<TRGB>>
<<E20>>

\noindent\textbf{Từ khoá:} pin lithium-ion, trạng thái sức khoẻ, PINN, học bán giám sát, rò rỉ dữ liệu, kiểm định chéo,
kết hợp mô hình, dự đoán phù hợp, dự báo tuổi thọ còn lại.
""", NCELL=num(ctx['ncell']), NRUN=num(ctx['runs']['tong']), R10=_rng(F['r'][0.1].values()),
             R70=_rng(F['r'][0.7].values()), GB=gb,
             TRGB=abstract_e21(d), E20=abstract_e20(d) + abstract_e22(d))


def e21_concl(d):
    if 'e21_base' not in d:
        return ''
    import thesis_bk as BK
    E = BK.e21_facts(d)
    t = (f'Học trên một bộ và dùng nguyên cho bộ khác, sai số của mô hình kết hợp gấp {f(E["gap"]["ket_hop"], 1)} lần '
         f'sai số trong bộ (GBDT: {f(E["gap"]["gbdt"], 1)} lần, MLP: {f(E["gap"]["mlp"], 1)} lần); phần lớn là lệch mức '
         f'của cả quỹ đạo.')
    if 'r0' in E:
        t += (f' Chuẩn hoá theo từng bộ cùng loss vật lý trên cell đích giảm sai số có ý nghĩa ở {E["r0_better"]} cặp '
              f'nhưng làm tăng ở {E["r0_worse"]} cặp (tỉ số trung bình nhân {f(E["r0"], 2)}); hai vòng học lặp bằng nhãn '
              f'giả hội tụ về đúng dự đoán của vòng đầu (tỉ số {f(E["r2v0"], 3)}), nên không cải thiện thêm.')
    return t


def e21_discussion(d):
    """Một câu về chuyển sang bộ khác cho phần thảo luận chung (rỗng khi chưa có E21)."""
    if 'e21_base' not in d:
        return ''
    import thesis_bk as BK
    E = BK.e21_facts(d)
    return (f' Khi đem sang một bộ dữ liệu khác chưa có nhãn, mọi mô hình đều mất nhiều độ chính xác: sai số của mô hình '
            f'kết hợp gấp {f(E["gap"]["ket_hop"], 1)} lần sai số trong bộ, của GBDT gấp {f(E["gap"]["gbdt"], 1)} lần '
            f'(Mục~\\ref{{sec:transferres}}).')


def e21_sections(d):
    """Các mục kết quả E21 (rỗng khi chưa có results/e21_*.csv)."""
    if 'e21_base' not in d:
        return ''
    import thesis_bk as BK
    return BK.e21_text(d, sys_helpers())


def e22_sections(d):
    """(bảng tham số ở Chương 5, các mục kết quả ở Chương 6) của E22; rỗng khi chưa có results/e22_*.csv."""
    if 'e22_traj' not in d:
        return ('', '')
    import thesis_bk as BK
    H = sys_helpers()
    extra = (BK.e23_text(d, H) if 'e23_tests' in d else '') + (BK.e24_text(d, H) if 'e24_cv' in d else '')
    return BK.e22_setting_table(d, H), BK.e22_text(d, H) + extra


def e22_concl(d):
    if 'e22_traj' not in d:
        return ''
    import thesis_bk as BK
    E = BK.e22_facts(d)
    t = E['t']
    return T(r"""\item Có, với độ chính xác giảm dần theo tầm. Khi cell test không được đo dung lượng và lịch sử SOH là ước
      lượng PINN-lai, mô hình kết hợp (PINN-dự báo và GBDT) dự báo SOH trên toàn bộ phần đời còn lại với MAE
      <<MAE>> ($\times10^{-3}$), bằng <<RB>> lần sai số của các mô hình chỉ dựa vào lịch sử của chính cell. GBDT và
      mô hình kết hợp gần ngang nhau về SOH, nhưng GBDT không ngoại suy được ra ngoài tầm đã học nên sai số RUL gấp
      <<RG>> lần; nghiệm đóng của động học là thứ giúp dự báo RUL. Tỉ lệ RUL dự báo nằm trong $\pm20\,\%$ RUL thật
      là <<IN20>> tuỳ bộ. Dấu của hệ số $\lambda$ học được khớp với hình dạng quỹ đạo: dương ở các bộ có điểm đầu
      gối, âm ở bộ suy giảm chậm dần. Khoảng 90\,\% hiện chỉ đạt độ phủ <<COV>>. Tách phần hồi phục và dao
      động đều ra khỏi xu thế giảm sai số tầm ngắn có ý nghĩa trên cả bốn bộ chính và giảm sai số ngay sau hồi phục ở
      XJTU, nhưng chưa giúp được trên NASA, nơi hồi phục gắn với khoảng nghỉ dài và cell có ít chu kỳ.
""", MAE=_rng([1e3 * t(ds, 'uoc', 'e_ket_hop') for ds in DS], 1),
             RB=_rng([x for vs in ('giu', 'tuyen', 'vatly') for x in E['ratio'][vs]], 2),
             RG=_rng([E['rul_gb'][ds] for ds in DS], 1), IN20=_rng([100 * E['rul_in20'][ds] for ds in DS], 0) + r'\,\%',
             COV=_rng([100 * x for x in E['cov']], 0) + r'\,\%')


def abstract_e22(d):
    if 'e22_traj' not in d:
        return ''
    import thesis_bk as BK
    E = BK.e22_facts(d)
    return T(r""" Đồ án mở rộng mô hình sang dự báo SOH tương lai, dung lượng còn lại và số chu kỳ còn lại tới ngưỡng
cuối đời: mạng PINN-dự báo trả hai tham số của nghiệm đóng của động học xám, kết hợp với GBDT. Khi cell không được đo
dung lượng lần nào, mô hình kết hợp có sai số SOH chỉ bằng <<RB>> lần các mô hình ngoại suy theo từng cell, và sai số
RUL thấp hơn GBDT <<RG>> lần nhờ nghiệm đóng ngoại suy được ra ngoài tầm đã học. Tách phần hồi phục dung lượng
và dao động đều ra khỏi xu thế giảm thêm sai số tầm ngắn. Trên bộ NASA, PINN-lai có sai số thấp nhất trong các mô hình
dùng đặc trưng sạc, còn thời gian nghỉ không cải thiện ước lượng.""",
             RB=_rng([x for vs in ('giu', 'tuyen', 'vatly') for x in E['ratio'][vs]], 2),
             RG=_rng([E['rul_gb'][ds] for ds in DS], 1))


def sys_helpers():
    return dict(f=f, pc=pc, m3=m3, table=table, fig=fig, T=T, lst=_lst, rng=_rng, gm=_gm, tren=_tren, DS=DS, CHEM=CHEM)


def build():
    d = load()
    ctx = dict(stats=dataset_stats(), runs=run_counts())
    ctx['ncell'] = int(ctx['stats'].N.sum())
    os.makedirs(f'{OUT}/hinh', exist_ok=True)
    for fn in os.listdir(R):
        if fn.startswith(('fig_thesis_', 'fig_pred_', 'fig_E13_', 'fig_metrics_', 'fig_physics_', 'fig_e20_', 'fig_e21_', 'fig_e22_', 'fig_e23_',
                           'fig_e24_')) \
                and fn.endswith('.pdf'):
            shutil.copy(f'{R}/{fn}', f'{OUT}/hinh/{fn}')
    shutil.copy('slide/mau/bk.png', f'{OUT}/hinh/bk.png')
    ctx = ch6(d, ctx)
    ctx = facts(d, ctx)
    import thesis_bk as BK
    main_body = BK.assemble(ch1(d, ctx), ch2(d, ctx), ch3(d, ctx), ch4(d, ctx), ch5(d, ctx), ch6_text(d, ctx),
                            ch7_improve(d, ctx), ch8(d, ctx), e21_sections(d), fig, e22_sections(d))
    body = (PREAMBLE + BK.COVER + r'\pagenumbering{roman}' + BK.FRONT + abstract(d, ctx)
            + r'{\linespread{1.0}\selectfont\tableofcontents\listoffigures\listoftables}' + BK.ABBR
            + '\n\\clearpage\\pagenumbering{arabic}\n' + main_body + REFS + r'\end{document}' + '\n')
    open(f'{OUT}/main.tex', 'w').write(body)
    for _ in range(2):
        # SOURCE_DATE_EPOCH cố định: PDF dựng lại từ cùng dữ liệu giống từng byte
        env = dict(os.environ, SOURCE_DATE_EPOCH='1792800000', FORCE_SOURCE_DATE='1')
        p = subprocess.run(['xelatex', '-interaction=nonstopmode', 'main.tex'], cwd=OUT, capture_output=True, text=True,
                           env=env)
    log = open(f'{OUT}/main.log', encoding='utf-8', errors='replace').read()
    errs = [l for l in log.splitlines() if l.startswith('!')]
    print('lỗi LaTeX:', errs[:10] if errs else 'không')
    if os.path.exists(f'{OUT}/main.pdf'):
        shutil.copy(f'{OUT}/main.pdf', f'{OUT}/bao-cao-do-an.pdf')
        print('đã ghi', f'{OUT}/bao-cao-do-an.pdf')


if __name__ == '__main__':
    build()
