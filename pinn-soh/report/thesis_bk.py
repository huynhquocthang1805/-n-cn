"""
Sắp xếp báo cáo theo khuôn đồ án chuyên ngành của Trường Đại học Bách khoa (ĐHQG-HCM):

    bìa — lời cam đoan — lời cảm ơn — tóm tắt — mục lục — danh mục hình, bảng, từ viết tắt
    1 Giới thiệu đề tài · 2 Cơ sở lý thuyết · 3 Các công trình liên quan · 4 Dữ liệu và phương pháp đề xuất
    5 Thiết kế thí nghiệm · 6 Kết quả và đánh giá · 7 Kết luận và hướng phát triển · Tài liệu tham khảo

Các chương cũ do make_thesis.py sinh ra (mọi con số đọc từ results/) được cắt theo mục rồi ghép lại; các bảng,
hình trùng thông tin được bỏ bớt để báo cáo nằm trong giới hạn trang.
"""
from __future__ import annotations
import re

COVER = r"""\begin{titlepage}
\begin{center}
{\large ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\\[2pt]
\textbf{TRƯỜNG ĐẠI HỌC BÁCH KHOA}\\[2pt]
\textbf{KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH}}\\[1.0cm]
\includegraphics[width=3.4cm]{hinh/bk.png}\\[1.2cm]
{\Large\textbf{ĐỒ ÁN CHUYÊN NGÀNH}}\\[1.4cm]
{\LARGE\bfseries ƯỚC LƯỢNG TRẠNG THÁI SỨC KHOẺ\\[4pt] CỦA PIN LITHIUM-ION BẰNG MẠNG NƠ-RON\\[4pt]
CÓ THÔNG TIN VẬT LÝ}\\[1.8cm]
\end{center}
\begin{tabular}{@{\hspace{3.2cm}}ll}
Giảng viên hướng dẫn: & ......................................................\\[4pt]
Sinh viên thực hiện: & Huỳnh Quốc Thắng \quad MSSV: ..................\\[4pt]
Nhóm: & 15\\[4pt]
Học kỳ: & HK261
\end{tabular}
\vfill
\begin{center}{\large TP. Hồ Chí Minh, tháng 10/2026}\end{center}
\end{titlepage}
"""

FRONT = r"""
\chapter*{Lời cam đoan}\addcontentsline{toc}{chapter}{Lời cam đoan}
Tôi xin cam đoan đồ án chuyên ngành ``Ước lượng trạng thái sức khoẻ của pin lithium-ion bằng mạng nơ-ron có thông
tin vật lý'' là kết quả làm việc của riêng tôi dưới sự hướng dẫn của giảng viên hướng dẫn. Các bộ dữ liệu sử dụng
đều là dữ liệu công khai và được trích dẫn nguồn đầy đủ. Mọi số liệu, bảng và hình trong báo cáo được tính trực
tiếp từ kết quả thực nghiệm do tôi thực hiện; các phần tham khảo từ công trình khác đều được ghi rõ trong danh mục
tài liệu tham khảo. Tôi xin chịu hoàn toàn trách nhiệm nếu có bất kỳ sai phạm nào so với lời cam đoan này.

\vspace{1.2cm}
\hfill\begin{tabular}{c}TP. Hồ Chí Minh, tháng 10 năm 2026\\ Sinh viên thực hiện\\[1.6cm] Huỳnh Quốc Thắng\end{tabular}

\chapter*{Lời cảm ơn}\addcontentsline{toc}{chapter}{Lời cảm ơn}
Tôi xin chân thành cảm ơn giảng viên hướng dẫn đã định hướng đề tài, góp ý về phương pháp đánh giá và theo sát
quá trình thực hiện đồ án. Tôi cũng cảm ơn quý thầy cô Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách
khoa -- ĐHQG-HCM, đã trang bị nền tảng kiến thức để tôi có thể thực hiện đề tài này. Cảm ơn các nhóm nghiên cứu
tại Đại học Giao thông Tây An, Đại học Thiên Tân, MIT--Stanford--Toyota Research Institute và Đại học Khoa học và
Công nghệ Hoa Trung đã công bố dữ liệu lão hoá pin, và nhóm tác giả PINN4SOH đã công bố mã nguồn tiền xử lý.
Cuối cùng, tôi cảm ơn gia đình và bạn bè đã luôn động viên trong suốt thời gian thực hiện đồ án.
"""

ABBR = r"""
\chapter*{Danh mục từ viết tắt}\addcontentsline{toc}{chapter}{Danh mục từ viết tắt}
\begin{center}\small
\begin{tabular}{>{\bfseries}l l}
\toprule
Viết tắt & Ý nghĩa \\
\midrule
BMS & Hệ thống quản lý pin (Battery Management System)\\
CC--CV & Sạc dòng không đổi -- áp không đổi (Constant Current -- Constant Voltage)\\
GBDT & Cây quyết định tăng cường gradient (Gradient Boosting Decision Tree)\\
GPR & Hồi quy quá trình Gauss (Gaussian Process Regression)\\
KTC & Khoảng tin cậy\\
LAM & Mất vật liệu hoạt động (Loss of Active Material)\\
LFP & Lithium sắt phốt phát (LiFePO$_4$)\\
LLI & Mất lithium khả dụng (Loss of Lithium Inventory)\\
MAE & Sai số tuyệt đối trung bình (Mean Absolute Error)\\
MLP & Perceptron nhiều lớp (Multilayer Perceptron)\\
NCA, NCM & Oxit niken--coban--nhôm, oxit niken--coban--mangan\\
ODE & Phương trình vi phân thường (Ordinary Differential Equation)\\
PINN & Mạng nơ-ron có thông tin vật lý (Physics-Informed Neural Network)\\
PINN-semi & PINN bán giám sát, loss vật lý áp lên cả cell không nhãn\\
PINN-lai & Mô hình kết hợp: ba PINN-semi, GBDT và bộ lọc quỹ đạo\\
RMSE & Căn bậc hai sai số bình phương trung bình (Root Mean Square Error)\\
SEI & Lớp điện phân rắn trên bề mặt anode (Solid Electrolyte Interphase)\\
SOH & Trạng thái sức khoẻ (State of Health)\\
\bottomrule
\end{tabular}
\end{center}
"""


# ───────────────────────────────────────────────────────────── công cụ cắt ghép
def split_sections(ch: str):
    """Tách chương thành phần đầu (dòng \\chapter và đoạn mở đầu) và danh sách (tiêu đề, nội dung mục)."""
    parts = re.split(r'(?m)^(?=\\section\{)', ch)
    head, out = parts[0], []
    for p in parts[1:]:
        m = re.match(r'\\section\{((?:[^{}]|\{[^{}]*\})*)\}', p)
        out.append((m.group(1), p))
    return head, out


def pick(secs, prefix):
    for t, body in secs:
        if t.startswith(prefix) or re.sub(r'\\texorpdfstring\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{.*', r'\1', t).startswith(prefix):
            return body
    raise KeyError(prefix)


def drop_float(tex: str, label: str) -> str:
    pat = re.compile(r'\\begin\{(figure|table)\}\[H\](?:(?!\\end\{(?:figure|table)\}).)*?\\label\{' + re.escape(label)
                     + r'\}.*?\\end\{\1\}\n?', re.S)
    new, n = pat.subn('', tex)
    assert n == 1, (label, n)
    return new


def demote(sec: str) -> str:
    """\\section -> \\subsection (đưa một chương cũ thành một mục)."""
    return re.sub(r'(?m)^\\section\{', r'\\subsection{', sec)


def retitle(sec: str, new: str) -> str:
    return re.sub(r'^\\section\{(?:[^{}]|\{[^{}]*\})*\}', lambda _: '\\section{' + new + '}', sec, count=1)


def sub1(tex: str, a: str, b: str) -> str:
    assert tex.count(a) == 1, a[:80]
    return tex.replace(a, b)


# ───────────────────────────────────────────────────────────── văn bản mới
CH1_GOALS = r"""\section{Mục tiêu, đối tượng và phạm vi}

Mục tiêu của đồ án là xây dựng và đánh giá một mô hình ước lượng SOH tận dụng được dữ liệu không nhãn và hiểu biết
vật lý về lão hoá pin, với ba yêu cầu: (i) sai số thấp hơn các mô hình học có nhãn thông dụng khi chỉ một phần nhỏ
cell có nhãn; (ii) kết quả được đo bằng một giao thức không rò rỉ thông tin, có kiểm định thống kê; (iii) đánh giá
được khả năng dùng mô hình học trên một bộ dữ liệu cho một bộ dữ liệu khác.

Đối tượng nghiên cứu là cell lithium-ion hình trụ 18650 thuộc ba hệ hoá học NCM, NCA/NCM và LFP, trong bốn bộ dữ
liệu lão hoá công khai. Đầu vào là 16 đặc trưng trích từ đoạn sạc CC--CV của từng chu kỳ; đầu ra là SOH tại chu kỳ
đó. Đồ án không xây dựng mô hình điện hoá chi tiết, không dự báo SOH ở tương lai và không ước lượng tuổi thọ còn
lại; các hướng này được nêu ở phần hướng phát triển.
"""

CH2_HEAD = r"""\chapter{Cơ sở lý thuyết}\label{ch:lythuyet}

Chương này trình bày các kiến thức nền mà mô hình và giao thức đánh giá của đồ án dựa vào: cơ chế suy giảm dung
lượng của pin lithium-ion, mạng nơ-ron có thông tin vật lý, học bán giám sát, kết hợp mô hình cùng ước lượng độ
bất định, thích nghi miền, và cách đánh giá mô hình khi dữ liệu có cấu trúc nhóm.
"""

CH2_DA = r"""\section{Thích nghi miền}\label{sec:da}

Khi mô hình học trên một bộ dữ liệu (miền nguồn) được dùng cho một bộ khác (miền đích), phân phối của đặc trưng và
quan hệ giữa đặc trưng với nhãn đều có thể thay đổi. Với pin, nguyên nhân là khác hệ hoá học, khác dung lượng danh
định, khác giao thức sạc và nhiệt độ. Thích nghi miền không nhãn dùng đặc trưng của miền đích, không dùng nhãn,
để giảm khoảng cách này. Cách đơn giản nhất là chuẩn hoá riêng từng miền bằng thống kê của chính miền đó, tương tự
việc tính lại thống kê của lớp chuẩn hoá theo batch trên miền đích (AdaBN~\cite{li2018adabn}). Các cách phức tạp
hơn học một biểu diễn chung bằng hàm mất mát đối kháng hoặc khoảng cách phân phối; Lu và cộng sự~\cite{lu2023}
dùng hướng này cho SOH. Một họ khác là tự huấn luyện: mô hình gán nhãn giả cho dữ liệu đích, rồi học lại trên chính
các nhãn giả đó qua nhiều vòng. Hiệu quả của tự huấn luyện phụ thuộc vào chất lượng nhãn giả; khi nhãn giả lệch có
hệ thống, mô hình học lại chính độ lệch đó. Đồ án dùng ràng buộc vật lý để làm sạch nhãn giả trước mỗi vòng.
"""

CH3_HEAD = r"""\chapter{Các công trình liên quan}\label{ch:lienquan}

Chương này tổng quan các hướng ước lượng SOH đã công bố, từ mô hình cơ chế tới học máy, học chuyển giao và mô hình
lai có thông tin vật lý, rồi rút ra các khoảng trống mà đồ án nhắm tới.
"""

CH4_HEAD = r"""\chapter{Dữ liệu và phương pháp đề xuất}\label{ch:dulieu}\label{ch:phuongphap}

Chương này mô tả bốn bộ dữ liệu và cách tiền xử lý, rồi trình bày ba thành phần của phương pháp: mô hình PINN bán
giám sát, mô hình kết hợp PINN-lai, và quy trình thích nghi không nhãn kèm tự huấn luyện để dùng mô hình trên một bộ
dữ liệu khác. Cuối chương là các mô hình đối chứng và cấu hình huấn luyện chung.
"""

CH4_DA = r"""\section{Thích nghi không nhãn và tự huấn luyện}\label{sec:selftrain}

Khi đem mô hình học trên bộ nguồn $S$ sang bộ đích $T$, đồ án dùng một quy trình gồm vòng 0 và các vòng tự huấn
luyện. Các cell của $T$ được chia làm hai nửa theo nhóm thí nghiệm: nửa thích nghi chỉ cung cấp đặc trưng sạc, nửa
còn lại dùng để kiểm tra; sau đó hai nửa đổi vai, nên mọi cell của $T$ đều được kiểm tra đúng một lần và không cell
nào vừa dùng để thích nghi vừa dùng để kiểm tra.

\paragraph{Vòng 0.} Đặc trưng của mỗi bộ được chuẩn hoá bằng thống kê của chính bộ đó: bộ nguồn dùng trung bình và
độ lệch chuẩn của cell train nguồn, bộ đích dùng của cell thích nghi (Mục~\ref{sec:da}). Ba mạng PINN-semi học trên
nhãn của $S$, còn các loss vật lý được tính trên cả cặp chu kỳ của cell thích nghi; GBDT học trên nhãn của $S$. Dự
đoán của mô hình kết hợp ở vòng 0 là $\tilde y^{(0)}$ theo \eqref{eq:lai1}--\eqref{eq:lai3}.

\paragraph{Vòng $k \ge 1$.} Nhãn giả trên cell thích nghi là $\tilde y^{(k-1)}$, đã qua bộ lọc quỹ đạo nên không
tăng theo chu kỳ. Mỗi mạng PINN-semi học lại từ đầu với thêm số hạng
$\lambda_\text{ps}\,\overline{(F_\varphi(\tilde{\mathbf{x}}, t) - \tilde y^{(k-1)})^2}$ trên cell thích nghi, cùng
các loss vật lý; GBDT học lại trên nhãn của $S$ cộng nhãn giả, trong đó tổng trọng số của nhãn giả bằng tổng trọng
số của nhãn nguồn. Đồ án dùng $\lambda_\text{ps} = 1$ và hai vòng tự huấn luyện; các giá trị này được ấn định trước,
không chọn theo nhãn của bộ đích vì nhãn đó không được phép dùng.

\begin{table}[H]\centering\small
\caption{Thuật toán thích nghi không nhãn và tự huấn luyện cho một cặp $S \to T$.}\label{tab:algo}
\begin{tabularx}{\textwidth}{>{\raggedleft}p{0.6cm}X}
\toprule
1 & Chia cell của $T$ thành nửa thích nghi $T_a$ và nửa kiểm tra $T_e$ (phân tầng theo nhóm thí nghiệm).\\
2 & Chuẩn hoá $S$ bằng thống kê cell train của $S$; chuẩn hoá $T$ bằng thống kê đặc trưng của $T_a$.\\
3 & Vòng 0: học 3 PINN-semi (nhãn $S$, vật lý trên $S$ và $T_a$) và GBDT (nhãn $S$); dừng sớm theo validation của $S$.\\
4 & Tính $\tilde y^{(0)}$ trên $T_a$: trung bình hai mô hình, qua bộ lọc quỹ đạo.\\
5 & Với $k = 1, 2$: học lại 3 PINN-semi và GBDT với thêm nhãn giả $\tilde y^{(k-1)}$ trên $T_a$; tính $\tilde y^{(k)}$.\\
6 & Báo cáo sai số của mô hình kết hợp mỗi vòng trên $T_e$; đổi vai $T_a$, $T_e$ và lặp lại.\\
\bottomrule
\end{tabularx}
\end{table}
"""

CH5_E21 = r"""\section{Huấn luyện một bộ, kiểm tra các bộ khác}\label{sec:transfer}

Bên cạnh đánh giá trong cùng một bộ, đồ án kiểm tra khả năng tổng quát sang bộ dữ liệu khác bằng hai giao thức trên
bộ nguồn $S \in \{\text{XJTU, TJU, MIT, HUST}\}$, mỗi giao thức với 3 hạt giống chia (Hình~\ref{fig:protocol}):
\begin{itemize}
\item Giao thức A: chia cell của $S$ thành 70\,\% train và 30\,\% validation; mô hình được kiểm tra trên toàn bộ cell
      của ba bộ còn lại. Mỗi bộ nguồn cho ba cặp $S \to T$, tổng cộng 12 cặp.
\item Giao thức B: chia cell của $S$ thành 70\,\% train, 15\,\% validation và 15\,\% test; mô hình được kiểm tra trên
      15\,\% cell test của chính $S$ (trong bộ) và trên toàn bộ cell của ba bộ còn lại (ngoài bộ).
\end{itemize}
Cả hai giao thức chia theo cell và phân tầng theo nhóm thí nghiệm như Mục~\ref{sec:unit}. Mọi cell train của bộ
nguồn đều có nhãn; validation của bộ nguồn dùng để dừng sớm và chọn số cây GBDT. Nhãn của bộ đích không được dùng ở
bất kỳ bước nào trước khi tính sai số. Ở giai đoạn không thích nghi, đặc trưng của mọi bộ được chuẩn hoá bằng thống
kê của cell train nguồn, và mọi dự đoán được giới hạn trong miền $[0{,}40;\,1{,}15]$ của $\mathcal{L}_\text{range}$:
khi đặc trưng của bộ đích nằm xa miền đã học, mạng nơ-ron ngoại suy không giới hạn, còn cây quyết định tự nhiên bị
chặn trong miền nhãn đã thấy. Quy trình thích nghi và tự huấn luyện ở Mục~\ref{sec:selftrain} được chạy trên giao
thức A với 2 hạt giống; sai số của nó được so ghép cặp với mô hình không thích nghi trên đúng các cell kiểm tra.

\FIGPROTO
"""

CH5_ENV = r"""\section{Môi trường thực nghiệm}

Mọi thí nghiệm chạy trên CPU (Intel Xeon 2,1\,GHz, 4 lõi), mỗi lượt huấn luyện một tiến trình một luồng để kết quả
tái lập được từng bit. Phần mềm: Python 3.11, PyTorch, scikit-learn, NumPy, SciPy và Matplotlib. Mã nguồn được tổ
chức thành một gói gồm các mô-đun đọc và làm sạch dữ liệu, mô hình, huấn luyện, mô hình đối chứng, hậu xử lý và
phân tích; mỗi thí nghiệm là một danh sách lượt chạy độc lập, có thể dừng và chạy tiếp. Chạy lại 20 lượt mẫu trên
máy khác cho dự đoán trùng từng bit với kết quả báo cáo.
"""


# ───────────────────────────────────────────────────────────── ghép
def assemble(c1, c2, c3, c4, c5, c6, c7, c8, e21, fig):
    """c1..c8: chuỗi các chương cũ (c7 = chương cải tiến); e21: các mục kết quả E21; fig: hàm sinh hình."""
    # Chương 1
    h1, s1 = split_sections(c1)
    ch1 = (sub1(h1, r'\chapter{Giới thiệu}', r'\chapter{Giới thiệu đề tài}')
           + retitle(pick(s1, 'Bối cảnh'), 'Đặt vấn đề') + pick(s1, 'Phát biểu bài toán') + CH1_GOALS
           + pick(s1, 'Câu hỏi nghiên cứu') + pick(s1, 'Đóng góp') + retitle(pick(s1, 'Bố cục'), 'Cấu trúc báo cáo'))
    # Chương 2, 3
    h2, s2 = split_sections(c2)
    ch2 = (CH2_HEAD + pick(s2, 'Suy giảm dung lượng') + pick(s2, 'Mạng nơ-ron có thông tin vật lý')
           + retitle(pick(s2, 'Học bán giám sát'), 'Học bán giám sát và tự huấn luyện')
           + pick(s2, 'Kết hợp mô hình') + CH2_DA + pick(s2, 'Đánh giá mô hình'))
    ch3 = (CH3_HEAD + retitle(pick(s2, 'Các hướng ước lượng SOH'), 'Các hướng ước lượng SOH')
           + retitle(pick(s2, 'Tổng hợp và khoảng trống'), 'Nhận xét và khoảng trống nghiên cứu'))
    # Chương 4: dữ liệu + PINN + PINN-lai + thích nghi + đối chứng
    h3, s3 = split_sections(c3)
    data_intro = h3.split('\\label{ch:dulieu}', 1)[1]
    data = '\\section{Dữ liệu}\\label{sec:data}\n' + data_intro + ''.join(demote(b) for _, b in s3)
    h4, s4 = split_sections(c4)
    keep4 = ['Tổng quan', 'Mạng nghiệm', 'Động học suy giảm', 'Hàm mất mát', 'Cơ chế bán giám sát',
             'Phần vật lý nằm ở đâu', 'Tính chất của mô hình']
    pinn = ('\\section{Mô hình PINN bán giám sát}\\label{sec:pinn}\n'
            + ''.join(demote(pick(s4, k)) for k in keep4))
    h7, s7 = split_sections(c7)
    lai = ('\\section{Mô hình kết hợp PINN-lai}\\label{sec:pinnlai_m}\n'
           'Chương~\\ref{ch:ketqua} sẽ cho thấy sai số còn lại của PINN-semi chủ yếu là lệch mức theo từng cell, và '
           'GBDT có sai số thấp nhất khi có đủ nhãn. Mục này mô tả các thành phần được xem xét để cải tiến mô hình và '
           'mô hình kết hợp được chọn.\n\n'
           + demote(retitle(pick(s7, 'Các hướng thử'), 'Các thành phần được xem xét'))
           + demote(pick(s7, 'Mô hình PINN-lai')))
    lai = sub1(lai, 'Sáu hướng được thử, mỗi hướng dựa trên một kết quả đã công bố hoặc một quan sát ở Chương~\\ref{ch:ketqua}.',
               'Sáu thành phần được xem xét, mỗi thành phần dựa trên một kết quả đã công bố hoặc một quan sát ở '
               'Chương~\\ref{ch:ketqua}; Mục~\\ref{sec:devproto} chọn trong số đó chỉ bằng tập validation.')
    ch4 = (CH4_HEAD + data + pinn + lai + CH4_DA + pick(s4, 'Các mô hình đối chứng') + pick(s4, 'Cấu hình huấn luyện'))
    # Chương 5: thiết kế thí nghiệm
    h5, s5 = split_sections(c5)
    unit = drop_float(pick(s5, 'Đơn vị chia'), 'tab:rowbound')
    unit = sub1(unit, 'Bảng~\\ref{tab:rowbound} so sánh cận này với sai số đo được: bộ chép\nnhãn không học gì vẫn đạt '
                'sai số nhỏ hơn mô hình thật nhiều lần, và khớp với cận lý thuyết.',
                'Trên dữ liệu thật, bộ chép nhãn không học gì đạt sai số nhỏ hơn mô hình được đánh giá đúng cách nhiều '
                'lần và khớp với cận lý thuyết.')
    val = drop_float(pick(s5, 'Chọn tỉ lệ validation'), 'tab:vtrade')
    val = sub1(val, 'Bảng~\\ref{tab:vtrade} cho thấy\nđánh đổi:', 'Có một đánh đổi:')
    strat = drop_float(pick(s5, 'Phân tầng'), 'fig:strat')
    strat = sub1(strat, ' và Hình~\\ref{fig:strat} là số cell\ntest của từng nhóm trong từng fold', '')
    rep = drop_float(pick(s5, 'Chọn số lần lặp'), 'fig:repeats')
    rep = sub1(rep, '(Bảng~\\ref{tab:repeats}, Hình~\\ref{fig:repeats})', '(Bảng~\\ref{tab:repeats})')
    ch5 = (h5 + pick(s5, 'Các yêu cầu') + unit + pick(s5, 'Chọn số fold') + val + strat + rep
           + pick(s5, 'Ghép cặp') + pick(s5, 'Quy trình chống rò rỉ')
           + CH5_E21.replace('\\FIGPROTO', fig('fig_e21_protocol', 'Hai giao thức huấn luyện trên một bộ, kiểm tra '
                                              'trên các bộ khác (trên) và quy trình thích nghi không nhãn, tự huấn '
                                              'luyện cho một cặp nguồn--đích (dưới).', 'fig:protocol'))
           + pick(s5, 'Kiểm toán rò rỉ') + pick(s5, 'Chỉ số đánh giá') + pick(s5, 'Phương pháp thống kê') + CH5_ENV)
    # Chương 6: kết quả
    h6, s6 = split_sections(c6)
    head6 = sub1(h6, '\\chapter{Kết quả và thảo luận}', '\\chapter{Kết quả và đánh giá}')
    leak = pick(s6, 'Kiểm toán rò rỉ')
    s30 = drop_float(pick(s6, 'So sánh ở 30'), 'fig:cells')
    labels = drop_float(pick(s6, 'PINN và học có nhãn'), 'fig:ratio')
    ext = pick(s6, 'Các chỉ số mở rộng')
    for lab in ('tab:pred', 'fig:parity', 'fig:errsoh'):
        ext = drop_float(ext, lab)
    rm = pick(s6, 'RMSE và')
    for lab in ('tab:rmser2', 'tab:transr2'):
        rm = drop_float(rm, lab)
    phys = drop_float(pick(s6, 'Kiểm tra phần vật lý'), 'fig:ratecell')
    improve = ('\\section{Mô hình kết hợp PINN-lai}\\label{sec:improve}\n'
               + ''.join(demote(pick(s7, k)) for k in ['Sai số còn lại', 'Chọn hướng trên tập validation',
                                                       'Kết quả trên tập test', 'Phân rã lợi ích', 'Khoảng dự đoán']))
    improve = drop_float(improve, 'tab:conformal')
    ch6 = (head6 + leak + s30 + labels + pick(s6, 'So với các mô hình học có nhãn khác') + pick(s6, 'Lợi ích đến từ đâu')
           + ext + rm + pick(s6, 'Tham số vật lý') + phys + improve + e21
           + pick(s6, 'Thảo luận chung') + demote(retitle(pick(s7, 'Thảo luận'), 'Về mô hình kết hợp')))
    ch7 = c8
    return fix_refs(ch1 + ch2 + ch3 + ch4 + ch5 + ch6 + ch7)


STRUCTURE = r"""\section{Cấu trúc báo cáo}

Báo cáo gồm bảy chương. Chương~\ref{ch:lythuyet} trình bày cơ sở lý thuyết về suy giảm dung lượng pin, mạng
nơ-ron có thông tin vật lý, học bán giám sát, kết hợp mô hình và thích nghi miền. Chương~\ref{ch:lienquan} tổng
quan các công trình liên quan và chỉ ra khoảng trống. Chương~\ref{ch:phuongphap} mô tả dữ liệu, mô hình PINN bán
giám sát, mô hình kết hợp PINN-lai, quy trình thích nghi và tự huấn luyện, cùng các mô hình đối chứng.
Chương~\ref{ch:thietke} là thiết kế thí nghiệm: cách chia dữ liệu và lý do chọn từng con số, quy trình chống rò rỉ,
hai giao thức huấn luyện trên một bộ và kiểm tra trên các bộ khác, chỉ số và kiểm định thống kê.
Chương~\ref{ch:ketqua} trình bày và đánh giá kết quả. Chương~\ref{ch:ketluan} kết luận và nêu hướng phát triển.
"""


def fix_refs(t: str) -> str:
    """Sửa các câu tham chiếu tới chương đã gộp hoặc tới bảng, hình đã bỏ."""
    t = re.sub(r'\\section\{Cấu trúc báo cáo\}.*?(?=\\chapter\{)', lambda _: STRUCTURE, t, count=1, flags=re.S)
    reps = [
        ('bộ lọc quỹ đạo ở Chương~\\ref{ch:caitien}', 'bộ lọc quỹ đạo ở Mục~\\ref{sec:pinnlai_m}'),
        ('Chương~\\ref{ch:caitien} thử kết hợp', 'Mục~\\ref{sec:pinnlai_m} thử kết hợp'),
        ('Cải tiến mô hình (Chương~\\ref{ch:caitien})', 'Cải tiến mô hình (Mục~\\ref{sec:improve})'),
        ('Chương~\\ref{ch:caitien} cho thấy', 'Mục~\\ref{sec:improve} cho thấy'),
        ('(iii) GBDT là đối thủ mạnh; Chương~\\ref{ch:caitien}', '(iii) GBDT là đối thủ mạnh; Mục~\\ref{sec:improve}'),
        (' (Hình~\\ref{fig:cells}: phần lớn cell nằm dưới đường chéo)', ''),
        ('Hình~\\ref{fig:parity}--\\ref{fig:errsoh} cho thấy', 'Hình~\\ref{fig:traj} cho thấy'),
        ('Bảng~\\ref{tab:rmser2} và Hình~\\ref{fig:rmser2} bổ sung hai', 'Hình~\\ref{fig:rmser2} bổ sung hai'),
        ('rồi lấy trung bình qua năm lần lặp. Bảng dùng $R^2$', 'rồi lấy trung bình qua năm lần lặp. Hình dùng $R^2$'),
        ('Hình~\\ref{fig:ratecell} so tốc độ suy giảm', 'Đồ án so tốc độ suy giảm'),
    ]
    for a, b in reps:
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    # đoạn R² của chuyển miền kiểu cũ (E18) — thay bằng Mục kết quả E21
    t, n = re.subn(r'Với chuyển miền \(Bảng~\\ref\{tab:transr2\}\).*?\n\n', '', t, count=1, flags=re.S)
    assert n == 1
    return t


# ───────────────────────────────────────────────────────────── kết quả E21
E21_MODELS = [('mlp', 'MLP'), ('gbdt', 'GBDT'), ('pinn3', '3 PINN'), ('ket_hop', 'Kết hợp')]
SAME_CHEM = {('XJTU', 'TJU'), ('TJU', 'XJTU'), ('MIT', 'HUST'), ('HUST', 'MIT')}


def e21_facts(d):
    import numpy as np
    b = d['e21_base']
    v = lambda proto, s, t, m, col='MAE_cell': float(b[(b.proto == proto) & (b.src == s) & (b.tgt == t) & (b.model == m)][col].iloc[0])
    DS = ['XJTU', 'TJU', 'MIT', 'HUST']
    pairs = [(s, t) for s in DS for t in DS if s != t]
    E = dict(v=v, pairs=pairs)
    E['win'] = {m: 0 for m, _ in E21_MODELS}
    for s, t in pairs:
        best = min(E21_MODELS, key=lambda mm: v('a7030', s, t, mm[0]))[0]
        E['win'][best] += 1
    gm = lambda xs: float(np.exp(np.mean(np.log(xs))))
    E['in_dom'] = {m: [v('b701515', s, s, m) for s in DS] for m, _ in E21_MODELS}
    E['out_dom'] = {m: [v('b701515', s, t, m) for s, t in pairs] for m, _ in E21_MODELS}
    E['gap'] = {m: gm(E['out_dom'][m]) / gm(E['in_dom'][m]) for m, _ in E21_MODELS}
    E['same'] = {m: gm([v('a7030', s, t, m) for s, t in pairs if (s, t) in SAME_CHEM]) for m, _ in E21_MODELS}
    E['diff'] = {m: gm([v('a7030', s, t, m) for s, t in pairs if (s, t) not in SAME_CHEM]) for m, _ in E21_MODELS}
    E['ab'] = float(np.mean([abs(v('a7030', s, t, 'ket_hop') / v('b701515', s, t, 'ket_hop') - 1) for s, t in pairs]))
    E['kh_vs_gb'] = gm([v('a7030', s, t, 'ket_hop') / v('a7030', s, t, 'gbdt') for s, t in pairs])
    E['kh_vs_p3'] = gm([v('a7030', s, t, 'ket_hop') / v('a7030', s, t, 'pinn3') for s, t in pairs])
    E['bias'] = [v('a7030', s, t, 'ket_hop', 'bias') for s, t in pairs]
    E['mae'] = [v('a7030', s, t, 'ket_hop') for s, t in pairs]
    if 'e21_chain' in d:
        c = d['e21_chain']
        cv = lambda s, t, m, r: float(c[(c.src == s) & (c.tgt == t) & (c.model == m)][str(r)].iloc[0])
        E['cv'] = cv
        tt = d['e21_chain_tests']
        E['ct'] = lambda s, t, r: tt[(tt.src == s) & (tt.tgt == t) & (tt['round'] == r)].iloc[0]
        E['r0'] = gm([cv(s, t, 'ket_hop', 0) / cv(s, t, 'ket_hop', -1) for s, t in pairs])
        E['r2'] = gm([cv(s, t, 'ket_hop', 2) / cv(s, t, 'ket_hop', -1) for s, t in pairs])
        E['r2v0'] = gm([cv(s, t, 'ket_hop', 2) / cv(s, t, 'ket_hop', 0) for s, t in pairs])
        sig = lambda r, better: sum(1 for s, t in pairs if (E['ct'](s, t, r).p < 0.05)
                                    and ((E['ct'](s, t, r).ti_so < 1) == better))
        E['r0_better'], E['r0_worse'] = sig(0, True), sig(0, False)
        E['r2_better'], E['r2_worse'] = sig(2, True), sig(2, False)
        E['r2_over_r0'] = sum(1 for s, t in pairs if cv(s, t, 'ket_hop', 2) < cv(s, t, 'ket_hop', 0))
    return E


def e21_text(d, H):
    f, pc, m3, table, fig, T, rng, lst = (H[k] for k in ('f', 'pc', 'm3', 'table', 'fig', 'T', 'rng', 'lst'))
    CHEM = H['CHEM']
    E = e21_facts(d)
    v = E['v']
    rows = []
    for s, t in E['pairs']:
        vals = [v('a7030', s, t, m) for m, _ in E21_MODELS]
        best = min(vals)
        rows.append([f'{s} $\\to$ {t}', f'{CHEM[s]} $\\to$ {CHEM[t]}']
                    + [(r'\textbf{' + m3(x, 1) + '}') if x == best else m3(x, 1) for x in vals])
    tab_a = table(['Cặp', 'Hoá học'] + [lab for _, lab in E21_MODELS], rows, 'llrrrr',
                  r'Giao thức A (70\,\% train, 30\,\% validation trên bộ nguồn): MAE theo cell ($\times10^{-3}$) trên toàn bộ '
                  r'cell của bộ đích, chưa thích nghi, trung bình 3 hạt giống; in đậm là thấp nhất của mỗi cặp.', 'tab:e21a')
    rows = []
    for i, s in enumerate(['XJTU', 'TJU', 'MIT', 'HUST']):
        others = [t for t in ['XJTU', 'TJU', 'MIT', 'HUST'] if t != s]
        r = [s]
        for m in ('gbdt', 'ket_hop'):
            r += [m3(v('b701515', s, s, m), 1), rng([1e3 * v('b701515', s, t, m) for t in others], 1)]
        rows.append(r)
    tab_b = table(['Bộ nguồn', 'GBDT, trong bộ', 'GBDT, bộ khác', 'Kết hợp, trong bộ', 'Kết hợp, bộ khác'], rows,
                  'lrrrr', r'Giao thức B (70/15/15): MAE theo cell ($\times10^{-3}$) trên 15\,\% cell test của chính bộ '
                  r'nguồn và trên ba bộ còn lại (khoảng nhỏ nhất--lớn nhất), trung bình 3 hạt giống.', 'tab:e21b')
    win_txt = lst([f'{lab} ở {E["win"][m]} cặp' for m, lab in E21_MODELS if E['win'][m]])
    out = T(r"""
\section{Huấn luyện một bộ, kiểm tra các bộ khác}\label{sec:transferres}

Bảng~\ref{tab:e21a} cho sai số khi mô hình học trên bộ nguồn theo giao thức A rồi đem dùng nguyên cho toàn bộ cell của
từng bộ đích, chưa có bước thích nghi nào. Mô hình có sai số thấp nhất là <<WIN>>. So với GBDT, mô hình kết hợp có
tỉ số MAE trung bình nhân <<KHGB>> trên 12 cặp; so với trung bình ba PINN là <<KHP3>>. Sai số trên bộ khác lớn hơn
nhiều so với trong cùng bộ: cặp cùng nhóm hoá học (XJTU--TJU, MIT--HUST) có MAE trung bình nhân của mô hình kết hợp
<<SAME>>, cặp khác hoá học <<DIFF>> (đơn vị $10^{-3}$). Phần lớn sai số này là lệch mức: độ lệch trung bình của mô
hình kết hợp trên từng cặp nằm trong khoảng <<BIAS>>, cùng bậc với chính MAE (<<MAE>>). Mô hình không biết dung lượng
ban đầu và mốc SOH của bộ đích, nên dù bắt được hình dạng quỹ đạo, cả quỹ đạo bị dịch lên hoặc xuống.

<<TABA>>

Giao thức B trả lời câu hỏi cùng một mô hình mất bao nhiêu độ chính xác khi rời bộ dữ liệu của nó (Bảng~\ref{tab:e21b},
Hình~\ref{fig:e21matrix}). Trung bình nhân trên các bộ, sai số ngoài bộ của mô hình kết hợp gấp <<GAPKH>> lần sai số
trong bộ, của GBDT gấp <<GAPGB>> lần và của MLP gấp <<GAPMLP>> lần. Kết quả ngoài bộ của hai giao thức gần như trùng
nhau (chênh trung bình <<AB>>), tức 15\,\% cell giữ lại để test trong bộ không làm thay đổi kết luận về chuyển miền.

<<TABB>>
<<FIGM>>
""", WIN=win_txt, KHGB=f(E['kh_vs_gb'], 2), KHP3=f(E['kh_vs_p3'], 2), SAME=f(1e3 * E['same']['ket_hop'], 1),
             DIFF=f(1e3 * E['diff']['ket_hop'], 1), BIAS=rng([1e3 * x for x in E['bias']], 1) + r'$\times10^{-3}$',
             MAE=rng([1e3 * x for x in E['mae']], 1) + r'$\times10^{-3}$', TABA=tab_a,
             GAPKH=f(E['gap']['ket_hop'], 1), GAPGB=f(E['gap']['gbdt'], 1), GAPMLP=f(E['gap']['mlp'], 1),
             AB=pc(E['ab'], 1), TABB=tab_b,
             FIGM=fig('fig_e21_matrix', 'Giao thức B: MAE theo cell (thang log) khi học trên bộ ở hàng và kiểm tra trên '
                      'bộ ở cột; ô viền đậm trên đường chéo là test trong cùng bộ.', 'fig:e21matrix'))
    if 'cv' not in E:
        return out
    cv, ct = E['cv'], E['ct']
    rows = []
    for s, t in E['pairs']:
        r = [f'{s} $\\to$ {t}'] + [m3(cv(s, t, 'ket_hop', k), 1) for k in (-1, 0, 1, 2)]
        x = ct(s, t, 2)
        r.append(f(x.ti_so, 2) + (r'$^{*}$' if x.p < 0.05 else ''))
        rows.append(r)
    tab_c = table(['Cặp', 'chưa thích nghi', 'vòng 0', 'vòng 1', 'vòng 2', 'tỉ số vòng 2'], rows, 'lrrrrr',
                  r'Thích nghi không nhãn và tự huấn luyện (giao thức A): MAE theo cell ($\times10^{-3}$) của mô hình kết hợp '
                  r'trên nửa bộ đích không dùng để thích nghi, trung bình 2 hạt giống $\times$ 2 nửa. Cột cuối: tỉ số vòng 2 so '
                  r'với chưa thích nghi; $^{*}$: Wilcoxon ghép cặp theo cell có $p < 0{,}05$.', 'tab:e21chain')
    out += T(r"""
\section{Thích nghi không nhãn và học lặp}\label{sec:selftrainres}

Bảng~\ref{tab:e21chain} và Hình~\ref{fig:e21rounds} theo dõi sai số của mô hình kết hợp qua từng vòng của quy trình ở
Mục~\ref{sec:selftrain}, luôn trên nửa bộ đích không dùng để thích nghi. Vòng 0, gồm chuẩn hoá theo từng bộ và loss vật
lý trên cell đích, đưa tỉ số MAE so với chưa thích nghi xuống <<R0>> (trung bình nhân trên 12 cặp); giảm có ý nghĩa
thống kê ở <<R0B>> cặp và tăng có ý nghĩa ở <<R0W>> cặp. Sau hai vòng tự huấn luyện, tỉ số là <<R2>>, giảm có ý nghĩa ở
<<R2B>> cặp và tăng có ý nghĩa ở <<R2W>> cặp. So riêng với vòng 0, hai vòng học lặp cho tỉ số <<R2V0>>, và sai số
vòng 2 thấp hơn vòng 0 ở <<R2O>>/12 cặp.

<<TABC>>
<<FIGR>>

Kết quả này cho thấy giới hạn của việc học lặp bằng nhãn giả. Nhãn giả đã qua bộ lọc nên không còn đoạn đi lên, tức
đúng về hình dạng; nhưng nếu cả quỹ đạo dự đoán của một cell bị lệch mức, nhãn giả mang đúng độ lệch đó, và vòng sau
học lại nó. Hình~\ref{fig:e21traj} minh hoạ trên một cell của mỗi bộ đích: bước chuẩn hoá theo bộ ở vòng 0 kéo quỹ
đạo về gần thực đo hơn nhiều, còn các vòng tự huấn luyện chủ yếu làm quỹ đạo trơn hơn chứ không sửa được phần lệch
mức còn lại. Muốn sửa phần này cần một chút thông tin về mức SOH của bộ đích, ví dụ dung lượng đo ở chu kỳ đầu của
mỗi cell, vốn thường có sẵn khi xuất xưởng.

<<FIGT>>
""", R0=f(E['r0'], 3), R0B=E['r0_better'], R0W=E['r0_worse'], R2=f(E['r2'], 3), R2B=E['r2_better'], R2W=E['r2_worse'],
             R2V0=f(E['r2v0'], 3), R2O=E['r2_over_r0'], TABC=tab_c,
             FIGR=fig('fig_e21_rounds', 'MAE theo cell của mô hình kết hợp trên bộ đích qua các vòng thích nghi (thang log); '
                      'mỗi đường là một bộ nguồn.', 'fig:e21rounds'),
             FIGT=fig('fig_e21_traj', 'Quỹ đạo SOH của một cell bộ đích (cell có sai số trung vị khi chưa thích nghi) với '
                      'mô hình kết hợp chưa thích nghi, sau vòng 0 và sau vòng 2.', 'fig:e21traj'))
    return out
