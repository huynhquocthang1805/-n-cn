"""
Dựng báo cáo dạng đồ án (LaTeX → PDF) từ results/*.csv.

    python make_thesis.py            # -> do-an/main.tex, do-an/hinh/, do-an/bao-cao-do-an.pdf

Chạy sau: analyze_e13.py, analyze_predictions.py, analyze_thesis.py, verify_pipeline.py --csv.
Mọi con số trong văn bản đọc từ results/, không gõ tay.
"""
from __future__ import annotations
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
    return f'{x:.{n}f}'.replace('.', '{,}')


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
              'pred_metrics', 'verify_pipeline', 'E13_physics_params', 'thesis_rowbound', 'thesis_leakage_raw', 'thesis_rowsim']:
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
             E14=len(glob.glob('runs/E14/samenorm_*.json')))
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


def fig(name, caption, label, width=r'\textwidth'):
    return (r'\begin{figure}[H]\centering' + '\n' + rf'\includegraphics[width={width}]{{hinh/{name}.pdf}}' + '\n'
            + rf'\caption{{{caption}}}\label{{{label}}}' + '\n' + r'\end{figure}' + '\n')


def table(header, rows, spec, caption, label, size=r'\small'):
    head = ' & '.join(header) + r' \\'
    body = '\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows)
    return (r'\begin{table}[H]\centering' + size + '\n' + rf'\caption{{{caption}}}\label{{{label}}}' + '\n'
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
\item Mô hình có chuyển được từ bộ dữ liệu này sang bộ khác mà không cần (hoặc chỉ cần rất ít) nhãn của bộ
      đích không?
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
\item Thí nghiệm chuyển miền đầy đủ giữa bốn bộ dữ liệu với quy trình chuẩn hoá không dùng dữ liệu test.
\end{enumerate}

\section{Bố cục}

Chương~\ref{ch:lythuyet} trình bày cơ sở lý thuyết và các công trình liên quan. Chương~\ref{ch:dulieu} mô tả
bốn bộ dữ liệu và cách tiền xử lý. Chương~\ref{ch:phuongphap} trình bày mô hình, các hàm mất mát và các mô
hình đối chứng. Chương~\ref{ch:thietke} là thiết kế thí nghiệm: cách chọn số chia, cách chống rò rỉ khi đánh
giá trong cùng bộ và khi chuyển miền, cùng các phương pháp thống kê. Chương~\ref{ch:ketqua} trình bày và
thảo luận kết quả. Chương~\ref{ch:ketluan} kết luận và nêu hướng phát triển.
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
học trực tiếp từ chuỗi đo. Các mô hình này mạnh khi có nhiều dữ liệu có nhãn và dễ quá khớp khi ít nhãn.

\paragraph{Học chuyển giao.} Ma và cộng sự~\cite{ma2022} dùng học chuyển giao sâu để cá nhân hoá dự báo
sức khoẻ cho từng cell và công bố bộ dữ liệu HUST. Hướng này giảm nhu cầu nhãn ở miền đích nhưng vẫn cần một
phần nhãn để thích nghi.

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

\section{Học bán giám sát}

Học bán giám sát tận dụng dữ liệu không nhãn qua một giả định về cấu trúc của dữ liệu, ví dụ tính trơn của
hàm cần học hoặc tính nhất quán của dự đoán. Trong đồ án, giả định được dùng là giả định vật lý: quỹ đạo SOH
của mọi cell, kể cả cell chưa đo dung lượng, phải thoả phương trình suy giảm, không tăng theo thời gian và nằm
trong miền giá trị hợp lý. Các ràng buộc này tính được chỉ từ đặc trưng sạc, nên mỗi cell không nhãn đều
đóng góp gradient cho mạng nghiệm.

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

\begin{table}[H]\centering\small
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
Raissi 2019~\cite{raissi2019} & PINN tổng quát & có & có & phương trình đã biết\\
Wang 2024~\cite{wang2024} & PINN, động học hộp đen & có & không & cùng bốn bộ dữ liệu\\
Đồ án & PINN bán giám sát, động học xám & có & có & loss vật lý không dùng nhãn\\
\bottomrule
\end{tabularx}
\end{table}

Từ các công trình trên có thể rút ra ba khoảng trống. Thứ nhất, các mô hình học có nhãn phụ thuộc mạnh vào
lượng nhãn, trong khi dữ liệu sạc không nhãn hầu như chưa được khai thác. Thứ hai, các PINN cho pin hiện có
dùng nhãn trong phần vật lý hoặc chỉ áp vật lý lên cell có nhãn, nên vật lý chỉ đóng vai trò chuẩn hoá chứ
không thay được nhãn. Thứ ba, kết quả công bố thường dựa trên một cách chia duy nhất, ít khi kiểm định
thống kê và ít khi kiểm tra rò rỉ. Đồ án nhắm vào cả ba điểm này.
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

\begin{figure}[H]\centering
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

\begin{table}[H]\centering\small
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

\begin{table}[H]\centering\small
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
\bibitem{arlot2010} S. Arlot, A. Celisse. A survey of cross-validation procedures for model selection.
  \emph{Statistics Surveys}, 4:40--79, 2010.
\bibitem{attia2022knee} P. M. Attia, A. Bills, F. Brosa Planella, et al. Review---“Knees” in lithium-ion battery
  aging trajectories. \emph{Journal of The Electrochemical Society}, 169:060517, 2022.
\bibitem{aykol2021} M. Aykol, C. B. Gopal, A. Anapolsky, et al. Perspective---Combining physics and machine
  learning to predict battery lifetime. \emph{Journal of The Electrochemical Society}, 168:030525, 2021.
\bibitem{birkl2017} C. R. Birkl, M. R. Roberts, E. McTurk, P. G. Bruce, D. A. Howey. Degradation diagnostics for
  lithium ion cells. \emph{Journal of Power Sources}, 341:373--386, 2017.
\bibitem{bouckaert2004} R. R. Bouckaert, E. Frank. Evaluating the replicability of significance tests for comparing
  learning algorithms. \emph{Advances in Knowledge Discovery and Data Mining (PAKDD)}, 3--12, 2004.
\bibitem{dietterich1998} T. G. Dietterich. Approximate statistical tests for comparing supervised classification
  learning algorithms. \emph{Neural Computation}, 10(7):1895--1923, 1998.
\bibitem{doyle1993} M. Doyle, T. F. Fuller, J. Newman. Modeling of galvanostatic charge and discharge of the
  lithium/polymer/insertion cell. \emph{Journal of The Electrochemical Society}, 140(6):1526--1533, 1993.
\bibitem{efron1993} B. Efron, R. J. Tibshirani. \emph{An Introduction to the Bootstrap}. Chapman \& Hall, 1993.
\bibitem{friedman2001} J. H. Friedman. Greedy function approximation: a gradient boosting machine.
  \emph{Annals of Statistics}, 29(5):1189--1232, 2001.
\bibitem{holm1979} S. Holm. A simple sequentially rejective multiple test procedure. \emph{Scandinavian Journal of
  Statistics}, 6(2):65--70, 1979.
\bibitem{kapoor2023} S. Kapoor, A. Narayanan. Leakage and the reproducibility crisis in machine-learning-based
  science. \emph{Patterns}, 4(9):100804, 2023.
\bibitem{karniadakis2021} G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang.
  Physics-informed machine learning. \emph{Nature Reviews Physics}, 3:422--440, 2021.
\bibitem{ke2017} G. Ke, Q. Meng, T. Finley, et al. LightGBM: a highly efficient gradient boosting decision tree.
  \emph{Advances in Neural Information Processing Systems}, 30, 2017.
\bibitem{loshchilov2019} I. Loshchilov, F. Hutter. Decoupled weight decay regularization. \emph{International
  Conference on Learning Representations}, 2019.
\bibitem{ma2022} G. Ma, S. Xu, B. Jiang, et al. Real-time personalized health status prediction of lithium-ion
  batteries using deep transfer learning. \emph{Energy \& Environmental Science}, 15:4083--4094, 2022.
\bibitem{nadeau2003} C. Nadeau, Y. Bengio. Inference for the generalization error. \emph{Machine Learning},
  52:239--281, 2003.
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
\bibitem{severson2019} K. A. Severson, P. M. Attia, N. Jin, et al. Data-driven prediction of battery cycle life before
  capacity degradation. \emph{Nature Energy}, 4:383--391, 2019.
\bibitem{varma2006} S. Varma, R. Simon. Bias in error estimation when using cross-validation for model selection.
  \emph{BMC Bioinformatics}, 7:91, 2006.
\bibitem{waldmann2014} T. Waldmann, M. Wilka, M. Kasper, M. Fleischhammer, M. Wohlfahrt-Mehrens. Temperature
  dependent ageing mechanisms in lithium-ion batteries -- a post-mortem study. \emph{Journal of Power Sources},
  262:129--135, 2014.
\bibitem{wang2024} F. Wang, Z. Zhai, Z. Zhao, Y. Di, X. Chen. Physics-informed neural network for lithium-ion
  battery degradation stable modeling and prognosis. \emph{Nature Communications}, 15:4332, 2024.
\bibitem{wen2023} P. Wen, Z.-S. Ye, Y. Li, S. Chen, P. Xie, S. Zhao. Physics-informed neural networks for
  prognostics and health management of lithium-ion batteries. arXiv:2301.00776, 2023.
\bibitem{wilcoxon1945} F. Wilcoxon. Individual comparisons by ranking methods. \emph{Biometrics Bulletin},
  1(6):80--83, 1945.
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

\section{Lợi ích đến từ đâu}

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
<<FIGTRAJ>>
<<FIGERR>>

\section{Chuyển miền}

<<TT>>
<<FIGTR>>

<<TRTXT>>

\section{Tham số vật lý học được}\label{sec:physparams}

<<TPH>>

<<PHTXT>>

\section{Thảo luận chung}

\paragraph{Khi nào nên dùng PINN bán giám sát?} Khi đã chọn mạng nơ-ron làm mô hình, PINN bán giám sát nên được
dùng thay cho MLP khi số cell có nhãn ít (khoảng 10--30\,\%), khi có nhiều cell chưa đo dung lượng, và nhất là khi
phải chạy trên một bộ dữ liệu mới chưa có nhãn: ở đó MLP thất bại hoàn toàn còn PINN-semi vẫn cho sai số dùng
được. So với GBDT thì khác: trong cùng một bộ dữ liệu, GBDT có sai số thấp hơn ở phần lớn các trường hợp; khi
chuyển miền, hai mô hình thắng thua tuỳ cặp<<TRGBD>>. Nếu chỉ cần sai số thấp nhất trên cùng một bộ dữ liệu,
GBDT là lựa chọn đơn giản và mạnh hơn.

\paragraph{Khi nào lợi ích nhỏ?} <<WEAKTXT>>

\paragraph{Hạn chế.} (i) Mô hình ước lượng SOH tại chu kỳ hiện tại, chưa dự báo SOH tương lai hay tuổi thọ còn
lại. (ii) Trọng số vật lý được chọn ở một thí nghiệm sơ bộ và giữ cố định; chưa tối ưu riêng cho từng mức nhãn.
(iii) GBDT là đối thủ mạnh và chưa được kết hợp với ràng buộc vật lý. (iv) Cell validation cần nhãn ngoài ngân
sách nhãn. (v) Ở 30\,\% nhãn, kết luận ở mức fold chỉ vững <<NB30>>. (vi) Tham số vật lý $\lambda, E_a$ chưa
ước lượng được từ dữ liệu hiện có.
""", TL=t['L'], FIGL=fig('fig_thesis_leakage', 'MAE theo cell của cùng một mô hình (MLP, 70\\,\\% nhãn) dưới các cách chia '
                                           'và chuẩn hoá khác nhau.', 'fig:leak'),
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
             FIGTRAJ=fig('fig_pred_traj', 'Quỹ đạo SOH trên ba cell test mỗi bộ (cell có sai số MLP ở phân vị 25, 50 và '
                         '75\\,\\%).', 'fig:traj', width=r'0.92\textwidth'),
             FIGERR=fig('fig_pred_error_soh', 'MAE theo từng khoảng SOH thực đo (30\\,\\% nhãn).', 'fig:errsoh'),
             TT=t['T'], FIGTR=fig('fig_thesis_transfer', 'MAE theo cell trên tập test của bộ đích (thang log); hàng: số cell '
                                  'đích có nhãn $k$, cột: mô hình; ô gạch chéo: cùng một bộ, không chuyển miền.',
                                  'fig:transfer', width=r'\textwidth'),
             TRTXT=tr_txt, TPH=t['PH'],
             TRGBD=(f' (PINN-semi tốt hơn ở {F["tr_gb0_win"]}/{F["tr_gb0_n"]} cặp khi không có nhãn đích)'
                    if F.get('tr_gb0_n') else ''), PHTXT=ph_txt, WEAKTXT=weak_txt, NB30=_tren(nb30) or 'trên không bộ nào')


# ═════════════════════════════════════════════════════════════ CHƯƠNG 7
def ch7(d, ctx):
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
    ch5 = (f'Khi bộ đích không có nhãn, PINN-semi giữ MAE ở mức {_rng(F["tr_k0_pinn"], 3)} trong khi MLP lệch tới '
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
      mang lại <<MONO>>\,\% mức giảm đó.<<BB>>
\item <<CH5>>
\end{enumerate}

Về phương pháp đánh giá, đồ án cho thấy cách chia theo chu kỳ làm sai số trông thấp hơn thực tế, và chứng minh bằng
một cận đơn giản rằng ngay cả bộ dự đoán không học gì cũng đạt sai số rất thấp dưới cách chia đó. Giao thức kiểm
định chéo 5 fold theo cell, phân tầng theo chế độ thí nghiệm, lặp 5 lần, được chọn từ các ràng buộc rõ ràng và có
các tính chất được chứng minh: mỗi cell được test đúng một lần mỗi lần lặp, các fold cân bằng theo từng chế độ, tập
nhãn lồng nhau giữa các mức, và ước lượng sai số không chệch.

\section{Hướng phát triển}

\begin{enumerate}
\item Kết hợp ràng buộc vật lý với mô hình cây: dùng dự đoán của GBDT làm đầu vào hoặc làm giá trị khởi đầu cho
      mạng nghiệm, hoặc dùng loss vật lý để tạo nhãn giả cho GBDT trên cell không nhãn.
\item Mở rộng sang dự báo SOH tương lai và tuổi thọ còn lại, với giao thức chia theo thời gian bên trong mỗi cell
      test.
\item Ước lượng độ bất định (tổ hợp nhiều mô hình hoặc dự đoán phù hợp) để BMS biết khi nào tin được dự đoán.
\item Dùng điện lượng tích luỹ thay cho chỉ số chu kỳ làm biến thời gian, để chuyển giữa các giao thức sạc tốt hơn.
\item Tăng trọng số của phần dư động học và dùng dữ liệu nhiều mức nhiệt độ để ước lượng được $\lambda$ và $E_a$.
\end{enumerate}
""", SIG30=_tren(F['sig'][0.3]) or 'trên không bộ nào', R30=_rng(F['r'][0.3].values()),
             R10=_rng(F['r'][0.1].values()), SIG10=_tren(F['sig'][0.1]) or 'trên không bộ nào',
             GM=', '.join(f'{pc(fr)}: {f(F["gm"][fr], 2)}' for fr in FRS),
             SIG70=('cải thiện chỉ còn có ý nghĩa ' + _tren(F['sig'][0.7])) if F['sig'][0.7]
             else 'không còn cải thiện có ý nghĩa',
             WORSE70=(f', và {_tren(F["worse"][0.7])} MLP tốt hơn' if F['worse'][0.7] else ''),
             CH3=ch3, SHARE=_rng([100 * x for x in F['share_unl10']], 0), MONO=_rng([100 * x for x in F['share_mono']], 0),
             BB=bb, CH5=ch5)


# ═════════════════════════════════════════════════════════════ TÓM TẮT, PHỤ LỤC, DỰNG
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
đó khoảng một nửa đến hai phần ba mức giảm đến từ cell không nhãn; khi tăng lên 70\,\% nhãn, lợi ích giảm dần (tỉ số <<R70>>). Trên bộ dữ liệu mới
không có nhãn, PINN vẫn cho sai số dùng được trong khi mạng học có nhãn thất bại<<TRGB>>. Tuy vậy, <<GB>>.

\noindent\textbf{Từ khoá:} pin lithium-ion, trạng thái sức khoẻ, PINN, học bán giám sát, rò rỉ dữ liệu, kiểm định chéo.
""", NCELL=num(ctx['ncell']), NRUN=num(ctx['runs']['tong']), R10=_rng(F['r'][0.1].values()),
             R70=_rng(F['r'][0.7].values()), GB=gb,
             TRGB=(f', và tốt hơn GBDT ở {F["tr_gb0_win"]}/{F["tr_gb0_n"]} cặp nguồn--đích'
                   if F.get('tr_gb0_n') and F['tr_gb0_win'] > F['tr_gb0_n'] / 2 else ''))


def build():
    d = load()
    ctx = dict(stats=dataset_stats(), runs=run_counts())
    ctx['ncell'] = int(ctx['stats'].N.sum())
    os.makedirs(f'{OUT}/hinh', exist_ok=True)
    for fn in os.listdir(R):
        if fn.startswith(('fig_thesis_', 'fig_pred_', 'fig_E13_')) and fn.endswith('.pdf'):
            shutil.copy(f'{R}/{fn}', f'{OUT}/hinh/{fn}')
    shutil.copy('slide/mau/bk.png', f'{OUT}/hinh/bk.png')
    ctx = ch6(d, ctx)
    ctx = facts(d, ctx)
    body = (PREAMBLE + TITLE + r'\pagenumbering{roman}' + abstract(d, ctx) + r'\tableofcontents\listoffigures\listoftables'
            + '\n\\clearpage\\pagenumbering{arabic}\n' + ch1(d, ctx) + ch2(d, ctx) + ch3(d, ctx) + ch4(d, ctx) + ch5(d, ctx)
            + ch6_text(d, ctx) + ch7(d, ctx) + REFS + r'\end{document}' + '\n')
    open(f'{OUT}/main.tex', 'w').write(body)
    for _ in range(2):
        p = subprocess.run(['xelatex', '-interaction=nonstopmode', 'main.tex'], cwd=OUT, capture_output=True, text=True)
    log = open(f'{OUT}/main.log', encoding='utf-8', errors='replace').read()
    errs = [l for l in log.splitlines() if l.startswith('!')]
    print('lỗi LaTeX:', errs[:10] if errs else 'không')
    if os.path.exists(f'{OUT}/main.pdf'):
        shutil.copy(f'{OUT}/main.pdf', f'{OUT}/bao-cao-do-an.pdf')
        print('đã ghi', f'{OUT}/bao-cao-do-an.pdf')


if __name__ == '__main__':
    build()
