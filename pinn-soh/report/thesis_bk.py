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
    pat = re.compile(r'\\begin\{(figure|table)\}\[!htbp\](?:(?!\\end\{(?:figure|table)\}).)*?\\label\{' + re.escape(label)
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

Chương này mô tả bốn bộ dữ liệu và cách tiền xử lý, rồi trình bày bốn thành phần của phương pháp: mô hình PINN bán
giám sát, mô hình kết hợp PINN-lai, quy trình thích nghi không nhãn kèm tự huấn luyện để dùng mô hình trên một bộ
dữ liệu khác, và phần mở rộng dự báo SOH tương lai, dung lượng còn lại và RUL. Cuối chương là các mô hình đối chứng
và cấu hình huấn luyện chung.
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

\begin{table}[!htbp]\centering\small
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

CH2_FC = r"""\section{Dự báo suy giảm và tuổi thọ còn lại}\label{sec:rul_theory}

Ước lượng SOH cho biết dung lượng hiện tại; dự báo cho biết dung lượng ở các chu kỳ sắp tới và số chu kỳ còn lại
trước khi pin chạm một ngưỡng cuối đời $\tau$ (remaining useful life, RUL). Các cách tiếp cận chia làm ba
nhóm~\cite{hu2020joule}. Nhóm thứ nhất khớp một hàm thực nghiệm, thường là tổng hai hàm mũ, lên dung lượng đã đo của
chính cell rồi ngoại suy, có thể kèm bộ lọc Bayes để cập nhật tham số khi có thêm số đo~\cite{he2011}; cách này không
cần dữ liệu của cell khác nhưng rất nhạy với độ dài và nhiễu của lịch sử, nhất là trước điểm đầu gối, nơi quỹ đạo đã
thấy chưa chứa dấu hiệu tăng tốc~\cite{attia2022knee}. Nhóm thứ hai học trực tiếp từ nhiều cell một ánh xạ từ lịch sử
tới dung lượng tương lai hoặc tới RUL, bằng mạng hồi quy~\cite{zhang2018lstm} hay mô hình dựa trên đặc trưng của các chu
kỳ đầu~\cite{severson2019}. Nhóm thứ ba ghép hai nhóm trên: dạng hàm lấy từ hiểu biết về cơ chế suy giảm, còn tham số
của nó được học từ dữ liệu của nhiều cell. Đánh giá dự báo dùng thêm các chỉ số riêng cho RUL, chẳng hạn tỉ lệ dự báo
rơi trong một dải sai số tương đối quanh RUL thật~\cite{saxena2008}.
"""

CH4_FC = r"""\section{Dự báo SOH tương lai, dung lượng còn lại và RUL}\label{sec:forecast_m}

Mô hình ở các mục trên trả lời câu hỏi pin đang ở đâu. Mục này mở rộng sang câu hỏi pin sẽ đi tới đâu: tại chu kỳ
gốc $k$ của một cell, dự báo SOH ở mọi chu kỳ $k + h$, $h > 0$, rồi suy ra dung lượng còn lại
$\hat Q(k+h) = \hat y(k+h)\,C_\text{dđ}$ (Ah, $C_\text{dđ}$ là dung lượng danh định) và
$\text{RUL}_k = \min\{h : \hat y(k+h) \le \tau\}$.

\paragraph{Thông tin tại gốc.} Bộ dự báo chỉ dùng các chu kỳ $\le k$: đặc trưng sạc $\mathbf{x}_1, \dots,
\mathbf{x}_k$ và một chuỗi SOH lịch sử $s_1, \dots, s_k$. Chuỗi lịch sử có hai nguồn, ứng với hai kịch bản: dung lượng
đo thật, khi BMS có đo dung lượng định kỳ; hoặc SOH do PINN-lai ước lượng từ đặc trưng sạc, khi cell không được đo
dung lượng lần nào. Từ đó tạo vectơ $\mathbf{z}_k$ gồm 23 thành phần: mức hiện tại $\ell_k$ (bộ lọc
\eqref{eq:trend} với $w = 10$ áp lên $s$), độ dốc bình phương tối thiểu trên 30 và 100 chu kỳ gần nhất, mức ban đầu $\ell_0$ (trung
bình 5 chu kỳ đầu) và độ giảm $\ell_0 - \ell_k$, $k/1000$, $1000/T$, cùng trung bình 16 đặc trưng sạc đã chuẩn hoá
trên 10 chu kỳ gần nhất.

\paragraph{Nghiệm đóng của động học xám.} Mạng tốc độ ở Mục~\ref{sec:pinn} mô tả
$\partial u/\partial t = -r(\mathbf{x})\,e^{\lambda(1-u)}A(T)$. Đặc trưng sạc của các chu kỳ tương lai chưa biết, nên
trong tầm dự báo coi $r(\mathbf{x})A(T)$ là hằng số. Đặt $v = 1 - u$, phương trình trở thành
$\mathrm{d}v/\mathrm{d}N = a_0 e^{\lambda v}$; tích phân từ gốc $k$ cho
\begin{equation}\label{eq:fcdrop}
  \Delta v(h) = v(k+h) - v(k) = -\frac{1}{\lambda}\ln\bigl(1 - \lambda a h\bigr),
\end{equation}
trong đó $a$ là tốc độ suy giảm tại gốc (trên 1000 chu kỳ) và $h$ tính bằng nghìn chu kỳ. Khi $\lambda > 0$ suy giảm
tăng tốc và đường cong dốc đứng khi $h$ tiến tới $1/(\lambda a)$, đúng hình dạng của điểm đầu gối; khi $\lambda < 0$ suy
giảm chậm dần; khi $\lambda \to 0$ ta có $\Delta v = a h$. Với mọi $a \ge 0$, $\Delta v$ không âm và không giảm theo
$h$, nên dự báo luôn đơn điệu mà không cần số hạng phạt. RUL cũng có dạng đóng:
\begin{equation}\label{eq:fcrul}
  h_\tau = \frac{1 - e^{-\lambda(\ell_k - \tau)}}{\lambda a}.
\end{equation}

\paragraph{PINN-dự báo.} Một mạng hai lớp ẩn 64 nút nhận $\mathbf{z}_k$ và trả $(a, \lambda)$, với $a$ qua hàm
softplus và $\lambda = 80\tanh(\cdot/80)$; dự báo là $\hat y(k+h) = \ell_k - \Delta v(h; a, \lambda)$, giới hạn trong
$[0{,}40;\,1{,}15]$. Hàm mất mát là sai số bình phương giữa $\hat y(k+h)$ và SOH đo trên các cặp (gốc, tầm) lấy từ
cell train: mỗi cell cho 40 gốc cách đều, mỗi gốc 40 điểm đích cách đều trong phần đời còn lại, nên các cell có trọng
số gần bằng nhau bất kể độ dài. Khác với PINN-semi, vật lý ở đây là ràng buộc cứng: nghiệm của phương trình động học
chính là lớp đầu ra. Ba thành viên với hạt giống khác nhau được lấy trung bình, dừng sớm theo MAE trên cell
validation.

\paragraph{GBDT trực tiếp và mô hình kết hợp.} GBDT học ánh xạ $(\mathbf{z}_k, h) \mapsto \ell_k - y(k+h)$ trên cùng
các cặp, với ràng buộc đơn điệu tăng theo $h$, cùng cấu hình và cách chọn số cây như GBDT đối chứng. Mô hình kết hợp
lấy trung bình hai dự báo, theo đúng lý do của PINN-lai: GBDT nội suy tốt trong miền tầm dự báo đã thấy nhưng đi
ngang bên ngoài miền đó, còn PINN-dự báo ngoại suy theo dạng hàm của động học.

\paragraph{Mô hình đối chứng.} (i) Giữ nguyên: $\hat y(k+h) = \ell_k$. (ii) Ngoại suy tuyến tính theo độ dốc 100 chu
kỳ gần nhất, chặn để không dương. (iii) Khớp vật lý theo cell: khớp \eqref{eq:fcdrop} với ba tham số
$(u_0, a_0, \lambda)$ lên nửa sau lịch sử của chính cell, có phạt nhẹ $\lambda$, rồi ngoại suy, đại diện cho nhóm
mô hình thực nghiệm khớp theo từng cell~\cite{he2011}. Khớp trên toàn bộ lịch sử và không phạt cho đường dự báo đổ
xuống ngay sau gốc; cấu hình trên là tốt nhất trong bốn biến thể có điểm đầu gối khi thử trên cell validation của
fold đầu tiên.

\paragraph{Khoảng dự báo.} Sai số có dấu (thật trừ dự báo) của mô hình kết hợp trên cell validation được chia theo
tầm tương đối $h/L$ ($L$ là trung vị số chu kỳ của bộ, các nhóm cận 0,05; 0,1; 0,25; 0,5; 1); cộng phân vị 5\,\% và
95\,\% của mỗi nhóm vào dự báo cho khoảng 90\,\%. Cận dưới và cận trên của RUL là lúc biên dưới và biên trên của
khoảng cắt ngưỡng $\tau$.
"""

CH5_FC = r"""\section{Giao thức đánh giá dự báo}\label{sec:fcproto}

Dự báo dùng đúng các fold của Mục~\ref{sec:unit} (5 fold theo cell, 10\,\% cell validation, lần lặp 0--2) ở mức 70\,\%
nhãn, nên mọi cell được dự báo đúng một lần mỗi lần lặp. Bộ dự báo học trên lịch sử đo của cell train. Ở kịch bản
không đo dung lượng, lịch sử của cell test là ước lượng PINN-lai của chính fold đó, tức từ mô hình chưa thấy cell này.

Gốc dự báo SOH đặt ở 20, 40 và 60\,\% số chu kỳ ghi được của cell; tỉ lệ này chỉ dùng để chọn điểm đánh giá, không
phải đầu vào của mô hình. Sai số SOH được báo cáo ở ba tầm $h = 0{,}1L$, $0{,}25L$, $0{,}5L$ và trên toàn bộ phần đời
còn lại sau gốc. Ngưỡng cuối đời $\tau$ của mỗi bộ được ấn định trước khi dự báo, là mức SOH thấp nhất (bước 0,01) mà
ít nhất 80\,\% cell của bộ đã chạm tới; chu kỳ cuối đời thật là chu kỳ đầu tiên mà trung vị trượt 9 chu kỳ của SOH đo
không vượt quá $\tau$. Ngưỡng 0,8 thường dùng không áp được vì phần lớn cell của MIT và XJTU dừng đo trước mức đó. Gốc
dự báo RUL đặt ở 25, 50 và 75\,\% chu kỳ cuối đời thật của cell. Bảng~\ref{tab:fcset} tóm tắt các tham số.

<<TABFC>>

Chỉ số gồm MAE của SOH theo cell ở từng tầm, quy ra dung lượng (mAh); MAE của RUL (chu kỳ) và tỉ lệ dự báo RUL nằm
trong $\pm 20\,\%$ RUL thật~\cite{saxena2008}; độ phủ thực tế của khoảng 90\,\%. Sai số của một cell lấy trung bình
qua lần lặp và các gốc; so sánh dùng Wilcoxon ghép cặp theo cell, hiệu chỉnh Holm trong mỗi bộ.
"""

CH5_ENV = r"""\section{Môi trường thực nghiệm}

Mọi thí nghiệm chạy trên CPU (Intel Xeon 2,1\,GHz, 4 lõi), mỗi lượt huấn luyện một tiến trình một luồng để kết quả
tái lập được từng bit. Phần mềm: Python 3.11, PyTorch, scikit-learn, NumPy, SciPy và Matplotlib. Mã nguồn được tổ
chức thành một gói gồm các mô-đun đọc và làm sạch dữ liệu, mô hình, huấn luyện, mô hình đối chứng, hậu xử lý và
phân tích; mỗi thí nghiệm là một danh sách lượt chạy độc lập, có thể dừng và chạy tiếp. Chạy lại 20 lượt mẫu trên
máy khác cho dự đoán trùng từng bit với kết quả báo cáo.
"""


# ───────────────────────────────────────────────────────────── ghép
def assemble(c1, c2, c3, c4, c5, c6, c7, c8, e21, fig, e22=('', '')):
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
           + pick(s2, 'Kết hợp mô hình') + CH2_DA + CH2_FC + pick(s2, 'Đánh giá mô hình'))
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
    ch4 = (CH4_HEAD + data + pinn + lai + CH4_DA + CH4_FC + pick(s4, 'Các mô hình đối chứng')
           + pick(s4, 'Cấu hình huấn luyện'))
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
           + (CH5_FC.replace('<<TABFC>>', e22[0]) if e22[0] else '')
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
    improve = drop_float(drop_float(improve, 'tab:conformal'), 'fig:e20traj')
    improve, n = re.subn(r' Hình~\\ref\{fig:e20traj\} đặt PINN-lai cạnh PINN-semi.*?Hình~\\ref\{fig:conformal\}\.', '',
                         improve, count=1, flags=re.S)
    assert n == 1
    ch6 = (head6 + leak + s30 + labels + pick(s6, 'So với các mô hình học có nhãn khác') + pick(s6, 'Lợi ích đến từ đâu')
           + ext + rm + pick(s6, 'Tham số vật lý') + phys + improve + e21 + e22[1]
           + pick(s6, 'Thảo luận chung') + demote(retitle(pick(s7, 'Thảo luận'), 'Về mô hình kết hợp')))
    ch7 = c8
    return fix_refs(ch1 + ch2 + ch3 + ch4 + ch5 + ch6 + ch7)


STRUCTURE = r"""\section{Cấu trúc báo cáo}

Báo cáo gồm bảy chương. Chương~\ref{ch:lythuyet} trình bày cơ sở lý thuyết về suy giảm dung lượng pin, mạng
nơ-ron có thông tin vật lý, học bán giám sát, kết hợp mô hình, thích nghi miền và dự báo tuổi thọ còn lại.
Chương~\ref{ch:lienquan} tổng quan các công trình liên quan và chỉ ra khoảng trống. Chương~\ref{ch:phuongphap} mô tả
dữ liệu, mô hình PINN bán giám sát, mô hình kết hợp PINN-lai, quy trình thích nghi và tự huấn luyện, mô hình dự báo
SOH tương lai và RUL, cùng các mô hình đối chứng. Chương~\ref{ch:thietke} là thiết kế thí nghiệm: cách chia dữ liệu và
lý do chọn từng con số, quy trình chống rò rỉ, hai giao thức huấn luyện trên một bộ và kiểm tra trên các bộ khác, giao
thức đánh giá dự báo, chỉ số và kiểm định thống kê.
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
    E['n_level'] = sum(1 for s, t in pairs if v('a7030', s, t, 'ket_hop', 'absbias') > 0.8 * v('a7030', s, t, 'ket_hop'))
    E['anchor'] = [v('a7030', s, t, 'ket_hop', 'MAE_anchor') for s, t in pairs]
    E['in_kh'] = E['in_dom']['ket_hop']
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
        E['r1v0'] = {m: gm([cv(s, t, m, 1) / cv(s, t, m, 0) for s, t in pairs]) for m in ('pinn3', 'gbdt', 'ket_hop')}
        E['r0_up'] = [f'{s} $\\to$ {t}' for s, t in pairs if E['ct'](s, t, 0).p < 0.05 and E['ct'](s, t, 0).ti_so > 1]
        E['r0_down'] = [f'{s} $\\to$ {t}' for s, t in pairs if E['ct'](s, t, 0).p < 0.05 and E['ct'](s, t, 0).ti_so < 1]
    if 'thesis_transfer' in d:
        tr = d['thesis_transfer']
        med = lambda m, k: float(tr[(tr.model == m) & (tr.k == k)]['mean'].median())
        E['k3'] = dict(p0=med('pinn_semi', 0), p3=med('pinn_semi', 3), g0=med('gbdt', 0), g3=med('gbdt', 3))
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
<<SAME>>, cặp khác hoá học <<DIFF>> (đơn vị $10^{-3}$), trong khi sai số trong bộ chỉ <<INKH>>.<<CHEMTXT>> Mạng nơ-ron hỏng nặng
nhất: sau khi chuẩn hoá bằng thống kê của bộ nguồn, đặc trưng của bộ đích nằm xa miền đã học, mạng ngoại suy và dự đoán
chạm biên của miền giới hạn ở phần lớn chu kỳ; GBDT tự nhiên bị chặn trong miền nhãn đã thấy nên ít hỏng hơn. Ở
<<NLEV>>/12 cặp, độ lệch trung bình theo cell chiếm trên 80\,\% MAE của mô hình kết hợp, tức cả quỹ đạo bị dịch lên
hoặc xuống. Tuy vậy, ngay cả khi biết dung lượng ở 5 chu kỳ đầu của mỗi cell đích và trừ đi độ lệch tương ứng, MAE vẫn
còn <<ANCH>> ($\times10^{-3}$): ngoài mức ban đầu, quan hệ giữa đặc trưng sạc và tốc độ suy giảm cũng khác giữa các
bộ dữ liệu.

<<TABA>>

Giao thức B trả lời câu hỏi cùng một mô hình mất bao nhiêu độ chính xác khi rời bộ dữ liệu của nó (Bảng~\ref{tab:e21b},
Hình~\ref{fig:e21matrix}). Trung bình nhân trên các bộ, sai số ngoài bộ của mô hình kết hợp gấp <<GAPKH>> lần sai số
trong bộ, của GBDT gấp <<GAPGB>> lần và của MLP gấp <<GAPMLP>> lần. Kết quả ngoài bộ của hai giao thức gần như trùng
nhau (chênh trung bình <<AB>>), tức 15\,\% cell giữ lại để test trong bộ không làm thay đổi kết luận về chuyển miền.

<<TABB>>
<<FIGM>>
""", WIN=win_txt, KHGB=f(E['kh_vs_gb'], 2), KHP3=f(E['kh_vs_p3'], 2), SAME=f(1e3 * E['same']['ket_hop'], 1),
             DIFF=f(1e3 * E['diff']['ket_hop'], 1), INKH=rng([1e3 * x for x in E['in_kh']], 1), NLEV=E['n_level'],
             CHEMTXT=(' Cùng hệ hoá học không làm việc chuyển dễ hơn: hai bộ LFP (MIT, HUST) khác nhau về giao thức sạc, '
                      'dải tuổi thọ và mức SOH ban đầu, nên khoảng cách giữa chúng không nhỏ hơn khoảng cách giữa hai hệ '
                      'hoá học khác nhau.' if E['same']['ket_hop'] >= E['diff']['ket_hop'] else
                      ' Cặp cùng hệ hoá học dễ chuyển hơn, nhưng sai số vẫn gấp nhiều lần sai số trong bộ.'),
             ANCH=rng([1e3 * x for x in E['anchor']], 1), TABA=tab_a,
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
lý trên cell đích, cho tỉ số MAE so với chưa thích nghi là <<R0>> (trung bình nhân trên 12 cặp), nhưng kết quả chia hai
chiều rõ rệt: giảm có ý nghĩa thống kê ở <<R0B>> cặp (<<R0DOWN>>) và tăng có ý nghĩa ở <<R0W>> cặp (<<R0UP>>). Chuẩn hoá
theo từng bộ đưa đặc trưng của bộ đích về cùng miền giá trị với bộ nguồn nên mạng nơ-ron không còn ngoại suy; đổi lại,
nó ngầm giả định cell trung bình của bộ đích có cùng SOH với cell trung bình của bộ nguồn. Khi hai bộ khác nhau về mức
SOH, chẳng hạn HUST có SOH ban đầu trung vị 1,08 trong khi ba bộ còn lại chỉ 0,93--0,98, giả định này đẩy dự
đoán lệch thêm;
nhiều khả năng đó là lý do mọi cặp có đích HUST đều kém đi.

Hai vòng tự huấn luyện gần như không làm thay đổi sai số của mô hình kết hợp: tỉ số vòng 2 so với vòng 0 là <<R2V0>>,
sau hai vòng tỉ số so với chưa thích nghi là <<R2>> (giảm có ý nghĩa ở <<R2B>> cặp, tăng có ý nghĩa ở <<R2W>> cặp).
Nguyên nhân nằm ở cấu trúc của quy trình. Ở vòng 1, mỗi mạng PINN và GBDT học khớp nhãn giả, tức chính dự đoán của mô
hình kết hợp ở vòng 0; các thành phần vì vậy hội tụ về nhãn giả (sai số của trung bình ba PINN đổi <<P31>> lần, của GBDT
<<G31>> lần giữa vòng 0 và vòng 1), còn mô hình kết hợp ở vòng sau trùng với vòng trước, một điểm bất động. Học lặp chỉ
có ích khi mỗi vòng đưa vào một thông tin mà nhãn giả chưa chứa.

<<TABC>>
<<FIGR>>

Hình~\ref{fig:e21traj} minh hoạ trên một cell của bốn cặp. Đường vòng 0 và vòng 2 gần như trùng nhau; vòng 0 kéo
quỹ đạo về gần thực đo ở các cặp có sai số giảm và đẩy xa hơn ở cặp có sai số tăng, đúng như Bảng~\ref{tab:e21chain}.
Nhãn giả đã qua bộ lọc nên đúng về hình dạng, nhưng mang nguyên độ lệch mức của mô hình, và vòng sau học lại độ lệch đó.
Thông tin còn thiếu là mức SOH của bộ đích. Một thí nghiệm bổ sung trên cùng bốn bộ (mô hình học trên bộ nguồn, thêm 3
cell đích có nhãn) cho thấy chỉ ba cell có nhãn đã đưa MAE trung vị qua 12 cặp của PINN-semi từ <<K0P>> xuống <<K3P>>,
của GBDT từ <<K0G>> xuống <<K3G>>. Khi triển khai, cách rẻ nhất để có thông tin này là đo dung lượng của một vài cell ở
bộ đích, hoặc dùng dung lượng đo khi xuất xưởng.

<<FIGT>>
""", R0=f(E['r0'], 3), R0B=E['r0_better'], R0W=E['r0_worse'], R2=f(E['r2'], 3), R2B=E['r2_better'], R2W=E['r2_worse'],
             R2V0=f(E['r2v0'], 3), R2O=E['r2_over_r0'], TABC=tab_c, R0DOWN=lst(E['r0_down']), R0UP=lst(E['r0_up']),
             P31=f(E['r1v0']['pinn3'], 2), G31=f(E['r1v0']['gbdt'], 2),
             K0P=m3(E['k3']['p0'], 1) + r'$\times10^{-3}$', K3P=m3(E['k3']['p3'], 1) + r'$\times10^{-3}$',
             K0G=m3(E['k3']['g0'], 1) + r'$\times10^{-3}$', K3G=m3(E['k3']['g3'], 1) + r'$\times10^{-3}$',
             FIGR=fig('fig_e21_rounds', 'MAE theo cell của mô hình kết hợp trên bộ đích qua các vòng thích nghi (thang log); '
                      'mỗi đường là một bộ nguồn.', 'fig:e21rounds'),
             FIGT=fig('fig_e21_traj', 'Quỹ đạo SOH của một cell bộ đích (cell có sai số trung vị khi chưa thích nghi) với '
                      'mô hình kết hợp chưa thích nghi, sau vòng 0 và sau vòng 2.', 'fig:e21traj'))
    return out


# ───────────────────────────────────────────────────────────── kết quả E22: dự báo
E22_MODELS = [('giu', 'Giữ nguyên'), ('tuyen', 'Tuyến tính'), ('vatly', 'Vật lý theo cell'), ('gbdt', 'GBDT'),
              ('pinn', 'PINN-dự báo'), ('ket_hop', 'Kết hợp')]


def e22_facts(d):
    import numpy as np
    hz, tj, ts, ru, rt, lam, st = (d[k] for k in ('e22_horizon', 'e22_traj', 'e22_tests', 'e22_rul', 'e22_rul_tests',
                                                     'e22_lambda', 'e22_setting'))
    DS = ['XJTU', 'TJU', 'MIT', 'HUST']
    E = {}
    E['h'] = lambda ds, sc, hr, m: float(hz[(hz.ds == ds) & (hz.scen == sc) & (hz.h_rel == hr)][m].iloc[0])
    E['t'] = lambda ds, sc, m: float(tj[(tj.ds == ds) & (tj.scen == sc)][m].iloc[0])
    E['r'] = lambda ds, sc, m: float(ru[(ru.ds == ds) & (ru.scen == sc)][m].iloc[0])
    E['test'] = lambda ds, sc, vs: ts[(ts.ds == ds) & (ts.scen == sc) & (ts.vs == vs)].iloc[0]
    E['rtest'] = lambda ds, sc, vs: rt[(rt.ds == ds) & (rt.scen == sc) & (rt.vs == vs)].iloc[0]
    E['ratio'] = {vs: [float(E['test'](ds, 'uoc', vs).ratio) for ds in DS] for vs in ('giu', 'tuyen', 'vatly', 'gbdt', 'pinn')}
    sig = lambda f, ds, sc, vs, better: (f(ds, sc, vs).p_holm < 0.05) and ((f(ds, sc, vs).better > f(ds, sc, vs).n / 2) == better)
    E['beats_base'] = all(sig(E['test'], ds, sc, vs, True) for ds in DS for sc in ('do', 'uoc') for vs in ('giu', 'tuyen', 'vatly'))
    E['soh_vs'] = {vs: dict(better=[ds for ds in DS if sig(E['test'], ds, 'uoc', vs, True)],
                            worse=[ds for ds in DS if sig(E['test'], ds, 'uoc', vs, False)]) for vs in ('gbdt', 'pinn')}
    E['rul_vs'] = {vs: dict(better=[ds for ds in DS if sig(E['rtest'], ds, 'uoc', vs, True)],
                            worse=[ds for ds in DS if sig(E['rtest'], ds, 'uoc', vs, False)])
                   for vs in ('tuyen', 'vatly', 'gbdt', 'pinn')}
    E['s2s1'] = {ds: E['t'](ds, 'uoc', 'e_ket_hop') / E['t'](ds, 'do', 'e_ket_hop') for ds in DS}
    E['cov'] = [E['t'](ds, sc, 'cov') for ds in DS for sc in ('do', 'uoc')]
    E['rcov'] = [E['r'](ds, sc, 'cov') for ds in DS for sc in ('do', 'uoc')]
    E['lam'] = {r.ds: (float(r.lam_median), float(r.frac_pos)) for r in lam.itertuples()}
    E['st'] = {r.ds: r for r in st.itertuples()}
    E['rul_rel'] = {ds: E['r'](ds, 'uoc', 'rel_ket_hop') for ds in DS}
    E['rul_in20'] = {ds: E['r'](ds, 'uoc', 'in20_ket_hop') for ds in DS}
    E['rul_gb'] = {ds: E['r'](ds, 'uoc', 'e_gbdt') / E['r'](ds, 'uoc', 'e_ket_hop') for ds in DS}
    return E


def e22_setting_table(d, H):
    f, table = H['f'], H['table']
    rows = []
    for r in d['e22_setting'].itertuples():
        nom = f(r.nom_min, 1) if r.nom_min == r.nom_max else f'{f(r.nom_min, 1)}--{f(r.nom_max, 1)}'
        rows.append([r.ds, r.n_cell, int(r.L), f'{r.h1}, {r.h2}, {r.h3}', f(r.tau, 2), H['pc'](r.reach, 0), nom])
    return table(['Bộ', 'Số cell', '$L$ (chu kỳ)', 'Tầm $0{,}1L$, $0{,}25L$, $0{,}5L$', r'$\tau$',
                  'Cell có chu kỳ cuối đời', r'$C_\text{dđ}$ (Ah)'], rows, 'lrrcrrr',
                 r'Tham số đánh giá dự báo của từng bộ: $L$ là trung vị số chu kỳ ghi được, $\tau$ là ngưỡng cuối đời.',
                 'tab:fcset')


def e22_text(d, H):
    f, pc, m3, table, fig, T, rng, lst = (H[k] for k in ('f', 'pc', 'm3', 'table', 'fig', 'T', 'rng', 'lst'))
    import numpy as np
    E = e22_facts(d)
    DS = ['XJTU', 'TJU', 'MIT', 'HUST']
    h, t, r = E['h'], E['t'], E['r']
    rows = []
    for ds in DS:
        for k, hr in enumerate((0.1, 0.25, 0.5)):
            vals = [h(ds, 'uoc', hr, f'e_{m}') for m, _ in E22_MODELS]
            best = min(vals)
            rows.append([ds if k == 0 else '', f(hr, 2 if hr == 0.25 else 1) + '$L$']
                        + [(r'\textbf{' + m3(v, 1) + '}') if v == best else m3(v, 1) for v in vals]
                        + [m3(h(ds, 'do', hr, 'e_ket_hop'), 1), f(h(ds, 'uoc', hr, 'mah'), 0)])
    tab_soh = table(['Bộ', 'Tầm'] + [lab for _, lab in E22_MODELS] + ['Kết hợp, đo', 'Kết hợp (mAh)'], rows,
                    'llrrrrrrrr',
                    r'MAE theo cell ($\times10^{-3}$) của SOH dự báo ở ba tầm, kịch bản lịch sử ước lượng (không đo dung '
                    r'lượng cell test); in đậm là thấp nhất. Hai cột cuối: mô hình kết hợp khi lịch sử là dung lượng đo, '
                    r'và sai số của mô hình kết hợp quy ra dung lượng (mAh) ở kịch bản ước lượng.', 'tab:e22soh')
    rows = []
    for ds in DS:
        vals = [r(ds, 'uoc', f'e_{m}') for m, _ in E22_MODELS[1:]]
        best = min(vals)
        rows.append([ds, f(r(ds, 'uoc', 'true'), 0)] + [(r'\textbf{' + f(v, 0) + '}') if v == best else f(v, 0) for v in vals]
                    + [f(r(ds, 'do', 'e_ket_hop'), 0), pc(r(ds, 'uoc', 'in20_ket_hop'), 0), pc(r(ds, 'uoc', 'cov'), 0)])
    tab_rul = table(['Bộ', 'RUL thực'] + [lab for _, lab in E22_MODELS[1:]] + ['Kết hợp, đo', r'Trong $\pm20\,\%$',
                                                                               'Độ phủ khoảng'], rows, 'lrrrrrrrrr',
                    r'MAE của RUL (chu kỳ), kịch bản lịch sử ước lượng, trung bình qua gốc ở 25, 50, 75\,\% đời; cột '
                    r'RUL thực là trung bình RUL thật tại các gốc. Ba cột cuối của mô hình kết hợp: MAE khi lịch sử là '
                    r'dung lượng đo, tỉ lệ dự báo nằm trong $\pm20\,\%$ RUL thật và độ phủ thực tế của khoảng RUL 90\,\%.',
                    'tab:e22rul')
    sv, rv = E['soh_vs'], E['rul_vs']
    gb_txt = ('tốt hơn có ý nghĩa ' + ('trên ' + lst(sv['gbdt']['better']) if sv['gbdt']['better'] else 'trên không bộ nào')
              + (', kém hơn có ý nghĩa trên ' + lst(sv['gbdt']['worse']) if sv['gbdt']['worse'] else ''))
    pi_txt = ('thấp hơn có ý nghĩa trên ' + lst(sv['pinn']['better']) if sv['pinn']['better'] else
              'không khác có ý nghĩa trên bộ nào')
    rul_pi = ('không khác có ý nghĩa trên bộ nào' if not (rv['pinn']['better'] or rv['pinn']['worse']) else
              ('thấp hơn có ý nghĩa trên ' + lst(rv['pinn']['better']) if rv['pinn']['better'] else '')
              + ('; cao hơn có ý nghĩa trên ' + lst(rv['pinn']['worse']) if rv['pinn']['worse'] else ''))
    lam = E['lam']
    acc = [ds for ds in DS if lam[ds][0] > 0]; dec = [ds for ds in DS if lam[ds][0] <= 0]
    out = T(r"""
\section{Dự báo SOH tương lai, dung lượng còn lại và RUL}\label{sec:forecastres}

\paragraph{Sai số SOH theo tầm dự báo.} Bảng~\ref{tab:e22soh} cho sai số ở kịch bản
thực tế hơn, khi cell test không được đo dung lượng lần nào và lịch sử SOH là ước lượng PINN-lai. Ba mô hình chỉ dựa
vào lịch sử của chính cell đều kém xa: trên toàn bộ phần đời còn lại, sai số của mô hình kết hợp chỉ bằng
<<RGIU>> lần sai số của dự báo giữ nguyên, <<RTUY>> lần của ngoại suy tuyến tính và <<RVL>> lần của khớp vật lý theo
cell<<BEATS>>. Khớp vật lý theo cell còn kém cả ngoại suy tuyến tính ở <<VLBAD>>: ba tham số khớp trên vài trăm điểm có
nhiễu không đủ để phân biệt một quỹ đạo sắp gãy với một quỹ đạo đang chậm dần, đúng nhược điểm đã biết của cách tiếp
cận này. Các mô hình học từ nhiều cell tránh được điều đó vì dạng đường cong được rút ra từ cell khác có đặc trưng
sạc tương tự. Giữa ba mô hình học từ nhiều cell, khác biệt nhỏ hơn nhiều: so với GBDT, mô hình kết hợp có tỉ số sai
số <<RGB>> và <<GBTXT>>; so với PINN-dự báo, tỉ số <<RPI>>, <<PITXT>>. Quy ra dung lượng, sai số của mô hình kết hợp
ở tầm $0{,}5L$ là <<MAH>> mAh.

<<TABSOH>>

\paragraph{Cái giá của việc không đo dung lượng.} Khi lịch sử là dung lượng đo thật, sai số trên toàn bộ phần đời còn
lại bằng <<S2S1>> sai số khi lịch sử là ước lượng (lần lượt XJTU, TJU, MIT, HUST). Chênh lệch lớn nhất ở HUST,
nơi ở tầm ngắn $0{,}1L$ sai số tăng từ <<HU1>> lên <<HU2>> ($\times10^{-3}$): sai số của dự báo ngắn hạn gần như bằng
sai số ước lượng mức SOH tại gốc, nên chất lượng của mô hình ước lượng quyết định chất lượng dự báo ở tầm ngắn. Ở MIT,
nơi PINN-lai ước lượng SOH rất sát, hai kịch bản gần như trùng nhau. Hình~\ref{fig:e22traj} minh hoạ một cell mỗi bộ:
lịch sử ước lượng, dự báo từ gốc ở 40\,\% đời và khoảng 90\,\%, với trục phải là dung lượng theo Ah.

<<FIGT>>

\paragraph{RUL.} Bảng~\ref{tab:e22rul} và Hình~\ref{fig:e22rul} cho thấy khác biệt rõ nhất giữa các mô hình nằm ở
RUL. GBDT không ngoại suy được ra ngoài miền tầm dự báo đã thấy khi học, nên khi RUL dài nó dự báo SOH đi ngang và
cắt ngưỡng quá muộn; sai số RUL của GBDT gấp <<RULGB>> lần của mô hình kết hợp, và mô hình kết hợp tốt hơn GBDT có ý
nghĩa <<RULGBS>>. Nghiệm đóng của PINN-dự báo tiếp tục cong theo hệ số $\lambda$ đã học nên cắt ngưỡng đúng lúc
hơn; giữa mô hình kết hợp và PINN-dự báo, sai số RUL <<RULPI>>. Như vậy mô hình kết hợp gần bằng GBDT về sai số SOH và
gần bằng PINN-dự báo về RUL, là mô hình duy nhất không kém ở cả hai chỉ số. Tỉ lệ dự báo RUL nằm trong $\pm20\,\%$
RUL thật là <<IN20>> và sai số tương đối trung vị <<REL>> (cùng thứ tự bốn bộ); sai số giảm khi gốc dự báo tiến gần
cuối đời, vì phần quỹ đạo còn phải ngoại suy ngắn lại.

<<TABRUL>>
<<FIGR>>

\paragraph{Hệ số $\lambda$ học được.} Mạng chỉ thấy đặc trưng tại gốc, không được cho biết bộ dữ liệu, nhưng dấu
của $\lambda$ khớp với hình dạng quỹ đạo của từng bộ. Trung vị $\lambda$ là <<LAMS>> (lần lượt XJTU, TJU, MIT, HUST),
với tỉ lệ gốc có $\lambda > 0$ là <<LAMP>>. <<LAMTXT>>

\paragraph{Khoảng dự báo.} Khoảng 90\,\% có độ phủ thực tế <<COV>> trên toàn bộ phần đời còn lại và <<RCOV>> với
khoảng RUL, thấp hơn mức đặt ra. Tập validation chỉ có 10\,\% số cell, trong khi sai số dự báo của một cell tương quan
mạnh giữa các tầm; phân vị ước lượng từ vài cell vì vậy hẹp hơn phân bố sai số của cell mới. Khoảng này cho biết bậc
độ lớn của bất định, nhưng chưa đủ tin cậy để dùng như một cam kết 90\,\%.
""", RGIU=rng(E['ratio']['giu'], 2), RTUY=rng(E['ratio']['tuyen'], 2), RVL=rng(E['ratio']['vatly'], 2),
            BEATS=(', khác biệt có ý nghĩa trên cả bốn bộ ở cả hai kịch bản' if E['beats_base'] else ''),
            VLBAD=lst([ds for ds in DS if t(ds, 'uoc', 'e_vatly') > t(ds, 'uoc', 'e_tuyen')]) or 'không bộ nào',
            RGB=rng(E['ratio']['gbdt'], 2), GBTXT=gb_txt, RPI=rng(E['ratio']['pinn'], 2), PITXT=pi_txt,
            MAH=rng([h(ds, 'uoc', 0.5, 'mah') for ds in DS], 0), TABSOH=tab_soh,
            S2S1='; '.join(f(1 / E['s2s1'][ds], 2) for ds in DS) + ' lần',
            HU1=m3(h('HUST', 'do', 0.1, 'e_ket_hop'), 1), HU2=m3(h('HUST', 'uoc', 0.1, 'e_ket_hop'), 1),
            FIGT=fig('fig_e22_traj', 'Dự báo cho một cell test mỗi bộ (lần lặp 0, gốc ở 40\\,\\% đời, cell có sai số ở '
                     'trung vị): lịch sử SOH ước lượng tới gốc, dự báo của GBDT, PINN-dự báo và mô hình kết hợp kèm '
                     'khoảng 90\\,\\%; trục phải là dung lượng (Ah).', 'fig:e22traj'),
            RULGB=rng([E['rul_gb'][ds] for ds in DS], 1), RULGBS=('trên cả bốn bộ' if len(rv['gbdt']['better']) == 4 else
                    'trên ' + lst(rv['gbdt']['better']) if rv['gbdt']['better'] else 'trên không bộ nào'),
            RULPI=rul_pi, IN20='; '.join(pc(E['rul_in20'][ds], 0) for ds in DS),
            REL='; '.join(pc(E['rul_rel'][ds], 0) for ds in DS), TABRUL=tab_rul,
            FIGR=fig('fig_e22_rul', 'RUL dự báo của mô hình kết hợp so với RUL thật, kịch bản lịch sử ước lượng; mỗi '
                     'điểm là một cell ở một gốc, trung bình qua 3 lần lặp; dải xám là sai lệch $\\pm20\\,\\%$.',
                     'fig:e22rul'),
            LAMS='; '.join(f(lam[ds][0], 1) for ds in DS), LAMP='; '.join(pc(lam[ds][1], 0) for ds in DS),
            LAMTXT=((f'Ở {lst(acc)}, $\\lambda$ dương ở phần lớn gốc: tốc độ suy giảm được dự báo tăng dần, đúng với '
                     'quỹ đạo có điểm đầu gối. ' if acc else '')
                    + (f'Ở {lst(dec)}, nơi nhiều cell giảm nhanh ở đầu đời rồi chậm lại, trung vị $\\lambda$ âm và chỉ '
                       'khoảng một nửa số gốc có $\\lambda > 0$: mạng tự chọn nhánh suy giảm chậm dần của nghiệm đóng. '
                       if dec else '')
                    + 'Cùng một dạng phương trình bao được cả hai kiểu quỹ đạo nhờ để dấu của $\\lambda$ tự do.'),
            COV=rng([100 * x for x in E['cov']], 0) + r'\,\%', RCOV=rng([100 * x for x in E['rcov']], 0) + r'\,\%')
    return out
