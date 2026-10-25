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
import numpy as np

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

Đối tượng nghiên cứu là cell lithium-ion hình trụ 18650 trong bốn bộ dữ liệu lão hoá công khai (NCM, NCA/NCM, LFP)
và bộ NASA PCoE. Đầu vào là 16 đặc trưng trích từ đoạn sạc CC--CV của từng chu kỳ; đầu ra là SOH tại chu kỳ đó và
SOH, dung lượng, RUL ở các chu kỳ sau. Đồ án không xây dựng mô hình điện hoá chi tiết.
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

CH4_HIST = r"""\section{Đầu vào có lịch sử của cell}\label{sec:hist_m}

PINN-semi, MLP và GBDT đều dự đoán SOH ở chu kỳ $n$ chỉ từ 16 đặc trưng sạc của chính chu kỳ đó và chỉ số chu kỳ.
Đặc trưng của một chu kỳ có nhiễu đo, trong khi các chu kỳ liền trước mang thông tin về gần như cùng một trạng thái của
cell. Đồ án thêm lịch sử vào đầu vào bằng các đại lượng nhân quả, chỉ dùng chu kỳ $n$ và các chu kỳ trước của cùng
cell nên vẫn chạy trực tuyến được: trung bình trượt mũ
\begin{equation}\label{eq:ema}
\mathbf{z}^{(a)}_1 = \mathbf{x}_1, \qquad \mathbf{z}^{(a)}_n = a\,\mathbf{x}_n + (1-a)\,\mathbf{z}^{(a)}_{n-1},
\end{equation}
nhớ khoảng $1/a$ chu kỳ, và độ dốc bình phương tối thiểu của từng đặc trưng trên 20 chu kỳ gần nhất. Bốn dạng được
thử: chỉ $\mathbf{z}^{(0{,}3)}$ thay cho $\mathbf{x}$; $\mathbf{x}$ nối với $\mathbf{z}^{(0{,}1)}$; $\mathbf{x}$ nối
với $\mathbf{z}^{(0{,}3)}, \mathbf{z}^{(0{,}1)}, \mathbf{z}^{(0{,}03)}$; và $\mathbf{x}$ nối với $\mathbf{z}^{(0{,}1)},
\mathbf{z}^{(0{,}03)}$ cùng độ dốc. GBDT thử thêm $\mathbf{x}$ nối với độ dốc, và $\mathbf{x}, \mathbf{z}^{(0{,}1)}$ nối
với độ lệch của $\mathbf{x}$ so với trung bình 5 chu kỳ đầu của cell. Đây là một lớp hồi quy tuyến tính với hệ số cố định, cùng ý tưởng với mô hình không
gian trạng thái của SambaMixer~\cite{olalde2024samba} nhưng không học hệ số. Mọi mô hình dùng được cùng đầu vào, nên
phép so sánh tách được tác động của lịch sử khỏi tác động của vật lý. Với mỗi bộ và mỗi loại mô hình, dạng lịch sử được
chọn trên tập validation và chỉ được giữ khi nó giảm MAE validation ít nhất 2\,\%; nếu không, mô hình dùng đặc trưng
gốc. PINN-lai có lịch sử dùng đúng công thức \eqref{eq:lai1}--\eqref{eq:lai3} với ba PINN-semi và GBDT đã chọn cho bộ.

"""


def mono_text(d, H):
    """Tiểu mục: bộ lọc xu thế không bảo đảm không tăng; bản có bảo đảm được và mất gì (analysis/analyze_mono.py)."""
    f, pc, T, rng, lst, fig = (H[k] for k in ('f', 'pc', 'T', 'rng', 'lst', 'fig'))
    S = d['mono_stats'].set_index('dataset'); F = d['mono_filter']
    r = lambda ds: list(F[F.dataset == ds].ratio)
    worse = [ds for ds in E28_DS if ((F.dataset == ds) & (F.p < 0.05) & (F.ratio > 1)).sum() == (F.dataset == ds).sum()]
    flat = [ds for ds in E28_DS if max(abs(x - 1) for x in r(ds)) < 0.02]
    return T(r"""
\subsection{Bộ lọc có giữ SOH không tăng không}\label{sec:monores}

Theo Mệnh đề~\ref{prop:filter}(iv), bộ lọc \eqref{eq:trend} không bảo đảm dãy đầu ra không tăng, và trên test điều này
xảy ra thường xuyên: đầu ra của PINN-lai tăng ở <<UP>> số bước giữa hai chu kỳ liên tiếp, bước tăng lớn nhất là
<<UPMAX>> SOH. Dãy \eqref{eq:mono} thì không bao giờ tăng. Khi $(w, a)$ của nó được chọn trên validation riêng cho từng
bộ rồi so trên test với bộ lọc xu thế, tỉ số MAE gần như bằng 1 trên <<FLAT>> (<<RFLAT>>), nhưng là <<RMIT>> trên MIT và
<<RXJTU>> trên XJTU<<WORSE>>.

Nguyên nhân nằm ở nhãn, không ở bộ lọc. Dung lượng đo được của XJTU tăng trong vài chục chu kỳ đầu (trung vị
<<EARLY>> SOH), và <<XUP>> số bước tăng quá 0,5\,\% SOH; trên NASA, tỉ lệ này là <<NUP>> vì dung lượng hồi phục sau
mỗi lần nghỉ. TJU và HUST gần như không có bước tăng thật. Dãy không tăng bắt đầu từ dự đoán ở chu kỳ đầu nên không theo
được đoạn dung lượng tăng (Hình~\ref{fig:mono}). PINN-lai vì vậy giữ bộ lọc xu thế, và đồ án mô tả nó đúng là làm trơn
theo xu thế giảm, không phải ràng buộc không tăng. Ràng buộc cứng chỉ hợp với các bộ mà dung lượng đo được thật sự
không tăng, như TJU và HUST.

<<FIG>>
""", UP=rng([100 * x for x in S.frac_up.dropna()], 0) + r'\,\%', UPMAX=f(float(S.up_max.max()), 2),
             FLAT=lst(flat) or 'không bộ nào', RFLAT=rng([x for ds in flat for x in r(ds)], 2),
             RMIT=rng(r('MIT'), 2), RXJTU=rng(r('XJTU'), 2),
             WORSE=(f', kém hơn có ý nghĩa ở mọi mức nhãn trên {lst(worse)}' if worse else ''),
             EARLY=f(float(S.loc['XJTU', 'early_rise_med']), 2), XUP=pc(float(S.loc['XJTU', 'label_up']), 0),
             NUP=pc(float(S.loc['NASA', 'label_up']), 0),
             FIG=fig('fig_mono_example', 'Một cell test của XJTU có dung lượng tăng đầu đời và một cell TJU (30\\,\\% nhãn, '
                     'lần lặp 0): SOH đo được, dự đoán của PINN-lai trước khi lọc, sau bộ lọc xu thế và sau bộ lọc không '
                     'tăng \\eqref{eq:mono} với $(w, a)$ chọn trên validation.', 'fig:mono'))


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
"""

CH5_E21 = r"""\section{Huấn luyện một bộ, kiểm tra các bộ khác}\label{sec:transfer}

Bên cạnh đánh giá trong cùng một bộ, đồ án kiểm tra khả năng tổng quát sang bộ dữ liệu khác bằng hai giao thức trên
bộ nguồn $S \in \{\text{XJTU, TJU, MIT, HUST}\}$, mỗi giao thức với 3 hạt giống chia. Giao thức A chia cell của $S$
thành 70\,\% train và 30\,\% validation rồi kiểm tra trên toàn bộ cell của ba bộ còn lại (12 cặp $S \to T$). Giao thức
B chia 70/15/15 và kiểm tra cả trên 15\,\% cell test của chính $S$ lẫn trên ba bộ còn lại. Cả hai chia theo cell, phân
tầng theo nhóm thí nghiệm; nhãn của bộ đích không được dùng ở bất kỳ bước nào trước khi tính sai số. Khi chưa thích
nghi, đặc trưng của mọi bộ được chuẩn hoá bằng thống kê của cell train nguồn và dự đoán được giới hạn trong
$[0{,}40;\,1{,}15]$. Quy trình thích nghi và tự huấn luyện ở Mục~\ref{sec:selftrain} chạy trên giao thức A với 2 hạt
giống, so ghép cặp với mô hình không thích nghi trên đúng các cell kiểm tra.
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

Một khó khăn riêng của dự báo là dung lượng đo được không giảm đều. Sau một khoảng nghỉ, các ion lithium phân bố lại
trong điện cực và cell lấy lại một phần dung lượng, rồi mất dần trong vài chu kỳ tiếp theo; một số giao thức sạc--xả
còn làm dung lượng dao động đều theo chu kỳ của chính giao thức. Phần thay đổi thuận nghịch này chồng lên xu thế suy
giảm không thuận nghịch. Mô hình dự báo coi mọi biến động là xu thế sẽ hiểu sai cả mức lẫn tốc độ suy giảm ngay sau
một lần hồi phục; SambaMixer~\cite{olalde2024samba} xử lý bằng cách đưa khoảng thời gian giữa hai chu kỳ vào mô hình.
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

\paragraph{Xét hồi phục dung lượng.} Bản cải tiến tách lịch sử tại gốc thành ba phần, chỉ dùng 60 chu kỳ gần nhất:
(i) xu thế, là đường thẳng khớp bằng hồi quy Huber nên vài điểm hồi phục không kéo lệch được mức và độ dốc;
(ii) dao động đều, tìm chu kỳ $P \in [2{,}2;\,15]$ với hai hài trên phần dư và chỉ giữ khi giải thích được ít nhất
60\,\% phương sai trên cả cửa sổ lẫn từng nửa cửa sổ, tức dao động thật sự lặp lại; (iii) phần hồi phục hiện tại $e_k$,
là phần dư của chính chu kỳ cuối sau khi trừ hai phần trên (ngay sau một lần nghỉ dài chỉ chu kỳ cuối nhảy lên), co về 0
một khoảng bằng hai lần mức nhiễu cục bộ ($1{,}4826$ lần độ lệch tuyệt đối trung vị của phần dư) để nhiễu đo không bị
coi là hồi phục. Dự báo trở thành
\begin{equation}\label{eq:fcrec}
  \hat y(k+h) = \ell_k + e_k\,g(h) - \Delta v(h; a, \lambda) + p_k(h), \qquad g(h) = \rho + (1 - \rho)\,e^{-h/\tau_r},
\end{equation}
trong đó $\ell_k$ là giá trị xu thế tại gốc, $\Delta v$ là nghiệm đóng \eqref{eq:fcdrop} chỉ cho phần suy giảm không
thuận nghịch, $p_k(h)$ là dao động ngoại suy, $\rho$ là phần hồi phục được giữ lại và $\tau_r$ là thời gian tắt của phần
còn lại. Cặp $(\rho, \tau_r)$ được ước lượng trực tiếp trên cell train của từng fold: tại mỗi lần hồi phục (SOH đo vượt
trung vị năm chu kỳ trước đó quá 0,004), phần lệch của SOH so với xu thế cộng dao động trong 30 chu kỳ sau được khớp
bằng $e_k\,g(h)$, dò lưới $\rho \in \{0; 0{,}1; \dots; 1\}$ và $\tau_r \in \{0{,}5; 1; \dots; 40\}$ chu kỳ; bộ có dưới năm
lần hồi phục nhận $\rho = 0$. Bản đầu tiên để mạng học $\tau_r$ bằng gradient, nhưng vì rất ít mẫu có $e_k$ khác 0 nên
$\tau_r$ đứng yên quanh giá trị khởi tạo 10 chu kỳ ở mọi bộ, trong khi dữ liệu cho thấy phần hồi phục hoặc được giữ gần
như trọn, hoặc mất ngay ở chu kỳ sau. GBDT học phần giảm của xu thế sau khi trừ hai thành phần thuận nghịch; trọng
số của GBDT trong mô hình kết hợp tăng tuyến tính từ 0 tại $h = 0$ tới $\tfrac{1}{2}$ tại $h = 0{,}1L$, vì cây gom các
tầm rất ngắn vào cùng một ngăn nên kém chính xác ngay sau gốc. Các lựa chọn này được chốt trên cell validation của lần
lặp đầu tiên, trước khi chạy trên test.

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

Bản có xét hồi phục được chạy trên đúng các fold đó, cùng lượt với các mô hình của bản đầu (học lại với cùng hạt giống),
nên hai bản được so ghép cặp trên cùng gốc. Ngoài các gốc trên, mỗi cell có thêm tối đa ba gốc đặt ngay sau một lần hồi
phục: SOH đo vượt trung vị năm chu kỳ trước đó quá 0,004, chỉ xét sau đỉnh của giai đoạn kích hoạt đầu đời (ở nhiều cell,
SOH tăng trong vài chục chu kỳ đầu, không phải hồi phục). Sai số ngắn hạn tính trên các chu kỳ trong tầm $0{,}1L$ sau
gốc. Bộ NASA dùng cùng giao thức kiểm định chéo cho cả ước lượng lẫn dự báo, với ngưỡng cuối đời 70\,\% như tài liệu
của NASA và SambaMixer; để so với SambaMixer, đồ án chạy thêm đúng cách chia NASA-L của bài đó.
"""

CH5_NEW = r"""\section{So với công trình đã công bố và ngân sách đo}\label{sec:newproto}

\paragraph{Cách chia của PINN4SOH.} Để so trực tiếp với PINN4SOH~\cite{wang2024}, đồ án dựng lại đúng cách chia của
bài từ nhật ký huấn luyện mà nhóm tác giả công bố kèm mã nguồn, nơi ghi danh sách cell train và test của từng lần chạy.
XJTU và TJU học một mô hình riêng cho mỗi batch: cell test của XJTU là cell số 4 và 8 (thêm cell 14 ở batch 3C), của
TJU là các cell chọn theo thứ tự tệp; MIT và HUST học một mô hình cho cả bộ, với 23 và 20 cell test cố định. Thí nghiệm
ít mẫu của bài chỉ cho $K = 1, \dots, 4$ cell train đầu tiên có nhãn (XJTU theo từng batch, HUST cả bộ). Validation
như bài: 20\,\% hàng ngẫu nhiên của các cell có nhãn. Ở chế độ ít mẫu, đặc trưng sạc của các cell train còn lại đi vào
các loss vật lý, đúng ý học bán giám sát; cell test không được dùng dưới dạng nào. Mỗi nhóm chạy với 3 hạt giống và hai
cách tiền xử lý: cách của đồ án (lọc nhân quả, chuẩn hoá theo cell train) và bản sao tiền xử lý của bài (lọc $3\sigma$
trên cả vòng đời, kể cả cột dung lượng; min--max theo từng cell cho đặc trưng và chỉ số chu kỳ). Cách thứ hai có rò rỉ
(Mục~\ref{sec:audit}) và chỉ dùng để so cùng điều kiện với số công bố. Số của PINN4SOH lấy từ các tệp kết quả công bố
cùng mã nguồn; sai số tính như bài: MAE từng cell test, trung bình qua cell, qua lần chạy, rồi qua nhóm.

\paragraph{Ngân sách đo.} Thí nghiệm ngân sách đo dùng đúng các fold và hạt giống của PINN-lai (3 lần lặp $\times$ 5
fold), nhưng thay vì đo đầy đủ một phần cell, mọi cell train được đo theo lịch $q \in \{0{,}5;\ 0{,}2;\ 0{,}1;\
0{,}05\}$ của Mục~\ref{sec:kalman_m}. Cell validation giữ nhãn đầy đủ như mọi thí nghiệm khác, cell test không có nhãn.
Trục so sánh là phần trăm số chu kỳ của cell train có đo dung lượng. Ở chế độ đo thưa cả khi dùng, cell test cũng được
đo theo cùng lịch $q$ (không có lần đo cuối) và ước lượng được cập nhật bằng \eqref{eq:kf}; sai số khi đó chỉ tính trên
các chu kỳ không được đo.
"""

CH5_ENV = r"""\section{Môi trường thực nghiệm}

Mọi thí nghiệm chạy trên CPU (Intel Xeon 2,1\,GHz, 4 lõi), mỗi lượt huấn luyện một tiến trình một luồng để kết quả
tái lập được từng bit. Phần mềm: Python 3.11, PyTorch, scikit-learn, NumPy, SciPy và Matplotlib. Mã nguồn được tổ
chức thành một gói gồm các mô-đun đọc và làm sạch dữ liệu, mô hình, huấn luyện, mô hình đối chứng, hậu xử lý và
phân tích; mỗi thí nghiệm là một danh sách lượt chạy độc lập, có thể dừng và chạy tiếp. Chạy lại 20 lượt mẫu trên
máy khác cho dự đoán trùng từng bit với kết quả báo cáo.
"""


# ───────────────────────────────────────────────────────────── ghép
def assemble(c1, c2, c3, c4, c5, c6, c7, c8, e21, fig, e22=('', ''), e25=('', ''), e26='', e27='', e28='', overview='',
             tuned='', e25s='', mono='', e29=''):
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
    ch4 = (CH4_HEAD + data + pinn + lai + (CH4_HIST if e29 else '') + CH4_DA + CH4_FC + e25[0] + pick(s4, 'Các mô hình đối chứng')
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
           + CH5_E21
           + (CH5_FC.replace('<<TABFC>>', e22[0]) if e22[0] else '') + (CH5_NEW if (e27 or e28) else '')
           + pick(s5, 'Kiểm toán rò rỉ') + pick(s5, 'Chỉ số đánh giá') + pick(s5, 'Phương pháp thống kê') + CH5_ENV)
    # Chương 6: kết quả
    h6, s6 = split_sections(c6)
    head6 = sub1(h6, '\\chapter{Kết quả và thảo luận}', '\\chapter{Kết quả và đánh giá}')
    leak = drop_float(pick(s6, 'Kiểm toán rò rỉ'), 'tab:leak')            # hình kiểm toán rò rỉ có đủ sáu thiết lập
    leak = sub1(leak, 'Bảng~\\ref{tab:leak} cho thấy', 'Hình~\\ref{fig:leak} cho thấy')
    leak = sub1(leak, 'Hàng cuối của bảng còn rõ hơn', 'Thiết lập cuối còn rõ hơn')
    s30 = drop_float(pick(s6, 'So sánh ở 30'), 'fig:cells')
    labels = drop_float(pick(s6, 'PINN và học có nhãn'), 'fig:ratio')
    ext = pick(s6, 'Các chỉ số mở rộng')
    for lab in ('tab:pred', 'fig:parity', 'fig:errsoh'):
        ext = drop_float(ext, lab)
    rm = pick(s6, 'RMSE và')
    for lab in ('tab:rmser2', 'tab:transr2', 'fig:rmser2'):
        rm = drop_float(rm, lab)
    phys = drop_float(drop_float(drop_float(pick(s6, 'Kiểm tra phần vật lý'), 'fig:ratecell'), 'fig:mono'), 'fig:ratesoh')
    phys = sub1(phys, 'Hình~\\ref{fig:ratesoh} xếp tốc độ suy giảm', 'Đồ án xếp tốc độ suy giảm')
    phys = drop_float(phys, 'tab:mono')
    phys = sub1(phys, ' (Bảng~\\ref{tab:mono}, Hình~\\ref{fig:mono})', '')
    improve = ('\\section{Mô hình kết hợp PINN-lai}\\label{sec:improve}\n'
               + ''.join(demote(pick(s7, k)) for k in ['Sai số còn lại', 'Chọn hướng trên tập validation',
                                                       'Kết quả trên tập test', 'Phân rã lợi ích', 'Khoảng dự đoán']))
    improve = drop_float(drop_float(improve, 'tab:conformal'), 'fig:e20traj')
    improve, n = re.subn(r' Hình~\\ref\{fig:e20traj\} đặt PINN-lai cạnh PINN-semi.*?Hình~\\ref\{fig:conformal\}\.', '',
                         improve, count=1, flags=re.S)
    assert n == 1
    ch6 = (head6 + leak + s30 + labels + pick(s6, 'So với các mô hình học có nhãn khác') + pick(s6, 'Lợi ích đến từ đâu')
           + ext + rm + pick(s6, 'Tham số vật lý') + e26 + phys + improve + tuned + mono + e29 + e27 + e21 + e22[1] + e28
           + (e25s if (e28 and e25s) else e25[1])
           + sub1(pick(s6, 'Thảo luận chung'), '\\section{Thảo luận chung}\n', '\\section{Thảo luận chung}\n' + overview)
           + ('' if e28 else demote(retitle(pick(s7, 'Thảo luận'), 'Về mô hình kết hợp'))))
    ch7 = c8
    out = fix_refs(ch1 + ch2 + ch3 + ch4 + ch5 + ch6 + ch7)
    # giữ báo cáo trong 75 trang: bỏ các hình mà bảng hoặc lời văn đã có cùng thông tin
    for lab in ('fig:e20curve', 'fig:conformal', 'fig:curvebase', 'fig:split', 'fig:rowsim', 'fig:usage', 'fig:e20steps',
                'fig:e23ex', 'tab:dev', 'tab:phys'):
        out = drop_float(out, lab)
    out = sub1(out, ' Bảng~\\ref{tab:dev}\ntóm tắt kết quả.', '')
    out = sub1(out, 'Hình~\\ref{fig:curvephys}, \\ref{fig:curvebase}', 'Hình~\\ref{fig:curvephys}')
    out = sub1(out, ' Hình~\\ref{fig:split} minh hoạ cách chia thật của XJTU.', '')
    # hình vẽ lại to hơn để dễ đọc khi in: bỏ bốn hình minh hoạ mà lời văn đã nêu đủ số liệu
    out = sub1(out, 'đồ án mô phỏng hai lần (Hình~\\ref{fig:rowsim}).', 'đồ án mô phỏng hai lần.')
    out = sub1(out, 'Hình~\\ref{fig:usage} liệt kê phần dữ liệu nào được dùng cho bước nào. Bốn quy tắc:',
               'Bốn quy tắc về phần dữ liệu được dùng ở từng bước:')
    out = sub1(out, 'Hình~\\ref{fig:e20steps} đi qua từng bước. Tính trung bình nhân', 'Tính trung bình nhân')
    return out


STRUCTURE = r"""\section{Cấu trúc báo cáo}

Chương~\ref{ch:lythuyet} và~\ref{ch:lienquan} trình bày cơ sở lý thuyết và các công trình liên quan,
Chương~\ref{ch:phuongphap} mô tả dữ liệu và các mô hình, Chương~\ref{ch:thietke} là thiết kế thí nghiệm,
Chương~\ref{ch:ketqua} là kết quả, Chương~\ref{ch:ketluan} là kết luận và hướng phát triển.
"""


def fix_refs(t: str) -> str:
    """Sửa các câu tham chiếu tới chương đã gộp hoặc tới bảng, hình đã bỏ."""
    t = re.sub(r'\\section\{Cấu trúc báo cáo\}.*?(?=\\chapter\{)', lambda _: STRUCTURE, t, count=1, flags=re.S)
    reps = [
        ('bộ lọc quỹ đạo ở Chương~\\ref{ch:caitien}', 'bộ lọc quỹ đạo ở Mục~\\ref{sec:pinnlai_m}'),
        ('Chương~\\ref{ch:caitien} thử kết hợp', 'Mục~\\ref{sec:pinnlai_m} thử kết hợp'),
        ('Cải tiến mô hình (Chương~\\ref{ch:caitien})', 'Cải tiến mô hình (Mục~\\ref{sec:improve})'),
        ('Chương~\\ref{ch:caitien} cho thấy', 'Mục~\\ref{sec:improve} cho thấy'),
        (' (Hình~\\ref{fig:cells}: phần lớn cell nằm dưới đường chéo)', ''),
        (' Hình~\\ref{fig:parity}--\\ref{fig:errsoh} cho thấy dự đoán bám sát thực đo trong phần lớn vòng đời,\n'
         'PINN-semi cho quỹ đạo trơn hơn, và sai số tăng ở giai đoạn cuối đời.', ''),
        ('Bảng~\\ref{tab:rmser2} và Hình~\\ref{fig:rmser2} bổ sung hai', 'Mục này bổ sung hai'),
        ('rồi lấy trung bình qua năm lần lặp. Bảng dùng $R^2$', 'rồi lấy trung bình qua năm lần lặp. Mục này dùng $R^2$'),
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
    E = e21_facts(d)
    v = E['v']
    rows = []
    for s in ['XJTU', 'TJU', 'MIT', 'HUST']:
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

Khi học trên bộ nguồn theo giao thức A rồi dùng nguyên cho toàn bộ cell của từng bộ đích, mô hình có sai số thấp nhất là
<<WIN>> (trên 12 cặp). Mô hình kết hợp có tỉ số MAE trung bình nhân <<KHGB>> so với GBDT và <<KHP3>> so với trung bình ba
PINN. Sai số trên bộ khác lớn hơn nhiều so với trong bộ: MAE trung bình nhân của mô hình kết hợp là <<SAME>> ở cặp cùng
nhóm hoá học và <<DIFF>> ở cặp khác hoá học (đơn vị $10^{-3}$), trong khi trong bộ chỉ <<INKH>>.<<CHEMTXT>> Ở <<NLEV>>/12
cặp, độ lệch trung bình theo cell chiếm trên 80\,\% MAE, tức cả quỹ đạo bị dịch lên hoặc xuống; ngay cả khi biết dung
lượng ở 5 chu kỳ đầu của mỗi cell đích và trừ độ lệch tương ứng, MAE vẫn còn <<ANCH>> ($\times10^{-3}$). Theo giao thức
B (Bảng~\ref{tab:e21b}), sai số ngoài bộ của mô hình kết hợp gấp <<GAPKH>> lần sai số trong bộ, của GBDT <<GAPGB>> lần
và của MLP <<GAPMLP>> lần (trung bình nhân trên các bộ).

<<TABB>>
""", WIN=win_txt, KHGB=f(E['kh_vs_gb'], 2), KHP3=f(E['kh_vs_p3'], 2), SAME=f(1e3 * E['same']['ket_hop'], 1),
             DIFF=f(1e3 * E['diff']['ket_hop'], 1), INKH=rng([1e3 * x for x in E['in_kh']], 1), NLEV=E['n_level'],
             CHEMTXT=(' Cùng hệ hoá học không làm việc chuyển dễ hơn: hai bộ LFP khác nhau về giao thức sạc, dải tuổi thọ '
                      'và mức SOH ban đầu.' if E['same']['ket_hop'] >= E['diff']['ket_hop'] else
                      ' Cặp cùng hệ hoá học dễ chuyển hơn, nhưng sai số vẫn gấp nhiều lần sai số trong bộ.'),
             ANCH=rng([1e3 * x for x in E['anchor']], 1),
             GAPKH=f(E['gap']['ket_hop'], 1), GAPGB=f(E['gap']['gbdt'], 1), GAPMLP=f(E['gap']['mlp'], 1), TABB=tab_b)
    if 'cv' not in E:
        return out
    out += T(r"""
\paragraph{Thích nghi không nhãn và học lặp.} Vòng 0 của quy trình ở Mục~\ref{sec:selftrain} (chuẩn hoá theo từng bộ,
loss vật lý trên cell đích) cho tỉ số MAE so với chưa thích nghi là <<R0>> (trung bình nhân trên 12 cặp), nhưng chia hai
chiều: giảm có ý nghĩa thống kê ở <<R0B>> cặp (<<R0DOWN>>) và tăng có ý nghĩa ở <<R0W>> cặp (<<R0UP>>). Chuẩn hoá theo
từng bộ ngầm giả định cell trung bình của bộ đích có cùng SOH với cell trung bình của bộ nguồn; HUST có SOH ban đầu trung
vị 1,08 trong khi ba bộ còn lại chỉ 0,93--0,98, nên mọi cặp có đích HUST đều kém đi. Hai vòng tự huấn luyện gần như
không đổi sai số (tỉ số vòng 2 so với vòng 0 là <<R2V0>>): mỗi thành phần học khớp nhãn giả, tức chính dự đoán của vòng
trước, nên mô hình kết hợp rơi vào một điểm bất động. Nhãn giả mang nguyên độ lệch mức của mô hình; thông tin còn thiếu
là mức SOH của bộ đích, và chỉ ba cell đích có nhãn đã đưa MAE trung vị qua 12 cặp của PINN-semi từ <<K0P>> xuống
<<K3P>>, của GBDT từ <<K0G>> xuống <<K3G>>.
""", R0=f(E['r0'], 3), R0B=E['r0_better'], R0W=E['r0_worse'], R2V0=f(E['r2v0'], 3), R0DOWN=lst(E['r0_down']),
             R0UP=lst(E['r0_up']),
             K0P=m3(E['k3']['p0'], 1) + r'$\times10^{-3}$', K3P=m3(E['k3']['p3'], 1) + r'$\times10^{-3}$',
             K0G=m3(E['k3']['g0'], 1) + r'$\times10^{-3}$', K3G=m3(E['k3']['g3'], 1) + r'$\times10^{-3}$')
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

\paragraph{RUL.} Bảng~\ref{tab:e22rul} cho thấy khác biệt rõ nhất giữa các mô hình nằm ở
RUL. GBDT không ngoại suy được ra ngoài miền tầm dự báo đã thấy khi học, nên khi RUL dài nó dự báo SOH đi ngang và
cắt ngưỡng quá muộn; sai số RUL của GBDT gấp <<RULGB>> lần của mô hình kết hợp, và mô hình kết hợp tốt hơn GBDT có ý
nghĩa <<RULGBS>>. Nghiệm đóng của PINN-dự báo tiếp tục cong theo hệ số $\lambda$ đã học nên cắt ngưỡng đúng lúc
hơn; giữa mô hình kết hợp và PINN-dự báo, sai số RUL <<RULPI>>. Như vậy mô hình kết hợp gần bằng GBDT về sai số SOH và
gần bằng PINN-dự báo về RUL, là mô hình duy nhất không kém ở cả hai chỉ số. Tỉ lệ dự báo RUL nằm trong $\pm20\,\%$
RUL thật là <<IN20>> và sai số tương đối trung vị <<REL>> (cùng thứ tự bốn bộ); sai số giảm khi gốc dự báo tiến gần
cuối đời, vì phần quỹ đạo còn phải ngoại suy ngắn lại.

<<TABRUL>>

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
            LAMS='; '.join(f(lam[ds][0], 1) for ds in DS), LAMP='; '.join(pc(lam[ds][1], 0) for ds in DS),
            LAMTXT=((f'Ở {lst(acc)}, $\\lambda$ dương ở phần lớn gốc: tốc độ suy giảm được dự báo tăng dần, đúng với '
                     'quỹ đạo có điểm đầu gối. ' if acc else '')
                    + (f'Ở {lst(dec)}, nơi nhiều cell giảm nhanh ở đầu đời rồi chậm lại, trung vị $\\lambda$ âm và chỉ '
                       'khoảng một nửa số gốc có $\\lambda > 0$: mạng tự chọn nhánh suy giảm chậm dần của nghiệm đóng. '
                       if dec else '')
                    + 'Cùng một dạng phương trình bao được cả hai kiểu quỹ đạo nhờ để dấu của $\\lambda$ tự do.'),
            COV=rng([100 * x for x in E['cov']], 0) + r'\,\%', RCOV=rng([100 * x for x in E['rcov']], 0) + r'\,\%')
    return out


# ───────────────────────────────────────────────────────────── kết quả E24: bộ NASA
E24_MODELS = [('mlp', 'MLP'), ('gbdt', 'GBDT'), ('pinn', 'PINN-semi'), ('pinn3', '3 PINN-semi'), ('pinn_lai', 'PINN-lai')]
SAMBA_REF = {'B0006': (2.448, 1.173), 'B0007': (1.861, 1.197), 'B0047': (2.549, 0.512)}   # (CNN-BiGRU, SambaMixer-L), MAE %


def e24_text(d, H):
    f, table, fig, T, rng, lst = (H[k] for k in ('f', 'table', 'fig', 'T', 'rng', 'lst'))
    import numpy as np
    cv, ts, rs, sb = d['e24_cv'], d['e24_tests'], d['e24_rest'], d.get('e24_samba')
    v = lambda ds, fr, m, col='mae': 100 * float(cv[(cv.ds == ds) & (cv.frac == fr) & (cv.model == m)][col].iloc[0])
    t = lambda kind, fr, m, ds=None: ts[(ts.kind == kind) & (ts.frac == fr) & (ts.model == m)
                                        & ((ts.ds == ds) if ds else ts.ds.isna())].iloc[0]
    rows = []
    for m, lab in E24_MODELS:
        vals = [v(ds, fr, m) for fr in (0.3, 0.7) for ds in ('NASA', 'NASAdt')]
        rows.append([lab] + [f(x, 2) for x in vals])
    best = [min(range(len(rows)), key=lambda i: float(rows[i][j].replace('{,}', '.'))) for j in range(1, 5)]
    for j, i in enumerate(best):
        rows[i][j + 1] = r'\textbf{' + rows[i][j + 1] + '}'
    tab_cv = table(['Mô hình', '30\\,\\%, 16 đặc trưng', '30\\,\\%, thêm giờ nghỉ', '70\\,\\%, 16 đặc trưng',
                    '70\\,\\%, thêm giờ nghỉ'], rows, 'lrrrr',
                   r'Bộ NASA, kiểm định chéo 5 fold theo cell $\times$ 3 lần lặp: MAE theo cell (\% SOH) ở hai mức nhãn, '
                   r'không và có thời gian nghỉ trước lần xả làm đặc trưng thứ 17; in đậm là thấp nhất mỗi cột.', 'tab:e24cv')
    lai_sig = [fr for fr in (0.3, 0.7) for m in ('mlp',) if t('lai_vs_mlp', fr, 'mlp', 'NASA').p < 0.05]
    out = T(r"""
\section{Bộ dữ liệu NASA}\label{sec:nasares}

\paragraph{Ước lượng SOH.} Bộ NASA được đánh giá bằng đúng giao thức của Chương~\ref{ch:thietke}: kiểm định chéo 5 fold
theo cell, phân tầng theo sáu nhóm giao thức, 3 lần lặp, ở 30 và 70\,\% nhãn (Bảng~\ref{tab:e24cv}). PINN-lai có sai số
thấp nhất ở cả hai mức nhãn, <<L3>>\,\% và <<L7>>\,\% SOH. So với MLP, tỉ số MAE theo cell là <<RM3>> và <<RM7>>; so với
một PINN-semi là <<RP3>> và <<RP7>> (đều có ý nghĩa thống kê); so với GBDT là <<RG3>> và <<RG7>>, không có ý nghĩa. Sai
số trên NASA lớn hơn nhiều so với bốn bộ kia vì bộ này chỉ có 18 cell trải trên sáu nhóm giao thức và ba mức nhiệt độ:
khi một nhóm chỉ có hai đến bốn cell, cell test thường không có cell train nào cùng điều kiện, và SOH tính theo danh
định 2,0\,Ah của cell 4\,\textdegree C thấp hơn cell 24\,\textdegree C ở cùng số chu kỳ tới 0,4.

\paragraph{Thời gian nghỉ.} Thêm thời gian nghỉ trước lần xả làm đặc trưng thứ 17, theo ý tưởng của
SambaMixer~\cite{olalde2024samba}, không cải thiện có ý nghĩa: tỉ số MAE của PINN-lai có và không có đặc trưng này là
<<DT3>> (30\,\% nhãn) và <<DT7>> (70\,\% nhãn). Lý do nằm ở đầu vào. Ở các chu kỳ ngay sau khoảng nghỉ dài (dài hơn hai
lần trung vị của cell), sai số của PINN-lai không lớn hơn các chu kỳ khác (<<LONG>> so với <<NORM>>\,\% ở 70\,\% nhãn):
đặc trưng sạc được đo sau khoảng nghỉ, nên đã phản ánh trạng thái vừa hồi phục, và thời gian nghỉ không mang thêm thông
tin.

<<TABCV>>
""", L3=f(v('NASA', 0.3, 'pinn_lai'), 2), L7=f(v('NASA', 0.7, 'pinn_lai'), 2),
            RM3=f(t('lai_vs_mlp', 0.3, 'mlp', 'NASA').ratio, 2), RM7=f(t('lai_vs_mlp', 0.7, 'mlp', 'NASA').ratio, 2),
            RP3=f(t('lai_vs_pinn', 0.3, 'pinn', 'NASA').ratio, 2), RP7=f(t('lai_vs_pinn', 0.7, 'pinn', 'NASA').ratio, 2),
            RG3=f(t('lai_vs_gbdt', 0.3, 'gbdt', 'NASA').ratio, 2), RG7=f(t('lai_vs_gbdt', 0.7, 'gbdt', 'NASA').ratio, 2),
            DT3=f(t('dt_vs_nodt', 0.3, 'pinn_lai').ratio, 2) + f' ($p = {f(t("dt_vs_nodt", 0.3, "pinn_lai").p, 2)}$)',
            DT7=f(t('dt_vs_nodt', 0.7, 'pinn_lai').ratio, 2) + f' ($p = {f(t("dt_vs_nodt", 0.7, "pinn_lai").p, 2)}$)',
            LONG=f(100 * float(rs[(rs.ds == 'NASA') & (rs.frac == 0.7) & (rs.model == 'pinn_lai')].mae_long.iloc[0]), 2),
            NORM=f(100 * float(rs[(rs.ds == 'NASA') & (rs.frac == 0.7) & (rs.model == 'pinn_lai')].mae_norm.iloc[0]), 2),
            TABCV=tab_cv)
    if sb is None:
        return out
    sv = lambda ds, m, c: 100 * float(sb[(sb.ds == ds) & (sb.model == m) & (sb.cell == f'NASA/{c}')].mae.iloc[0])
    rows = [[c, f(SAMBA_REF[c][0], 2), f(SAMBA_REF[c][1], 2), f(sv('NASA', 'gbdt', c), 2), f(sv('NASA', 'pinn_lai', c), 2),
             f(sv('NASAdt', 'pinn_lai', c), 2)] for c in ('B0006', 'B0007', 'B0047')]
    tab_s = table(['Cell', 'CNN-BiGRU', 'SambaMixer-L', 'GBDT', 'PINN-lai', 'PINN-lai, thêm giờ nghỉ'], rows, 'lrrrrr',
                  r'Cách chia NASA-L của SambaMixer: MAE (\% SOH) trên ba cell đánh giá. Hai cột đầu lấy từ bài của '
                  r'SambaMixer~\cite{olalde2024samba}, đầu vào là đường xả của chính chu kỳ cần ước lượng; ba cột sau là '
                  r'của đồ án, đầu vào là 16 đặc trưng của lần sạc trước đó, trung bình 3 hạt giống.', 'tab:e24samba')
    out += T(r"""
\paragraph{So với SambaMixer.} Bảng~\ref{tab:e24samba} dùng đúng cách chia NASA-L của SambaMixer: tám cell train, hai cell
giữ lại để dừng sớm, đánh giá trên các cell 6, 7 và 47. PINN-lai có MAE <<S6>>, <<S7>> và <<S47>>\,\% SOH, cao hơn
SambaMixer trên cả ba cell và chỉ tốt hơn CNN-BiGRU trên cell 47. SambaMixer nhận chuỗi điện áp, dòng điện, nhiệt độ của chính lần xả cần ước lượng; với dòng xả không đổi, thời gian xả
nhân với dòng điện chính là dung lượng, nên nhãn gần như nằm sẵn trong đầu vào. Đồ án dùng đoạn sạc trước lần xả, là
dữ liệu BMS ghi được ở mọi chu kỳ mà không phải xả sâu cell. Sai số lớn nhất ở cell 6, vốn bắt đầu ở 101\,\% SOH trong khi
các cell train cùng nhóm bắt đầu quanh 92\,\%, với 75--84\,\% số chu kỳ có đặc trưng sạc nằm ngoài miền của cell
train.<<HIST>>

<<TABS>>
<<FIGS>>
""", S6=f(sv('NASA', 'pinn_lai', 'B0006'), 2), S7=f(sv('NASA', 'pinn_lai', 'B0007'), 2),
             S47=f(sv('NASA', 'pinn_lai', 'B0047'), 2), TABS=tab_s,
             FIGS='', HIST=nasa_hist(d, f, rng))
    return out


def nasa_hist(d, f, rng):
    """Một câu: đầu vào có lịch sử trên NASA (tỉ số MAE validation của dạng tốt nhất so với đặc trưng gốc, trung bình
    qua kiểm định chéo và cách chia NASA-L; analysis/dev_e29.py)."""
    if 'e29_choice' not in d or 'e29_summary' not in d:
        return ''
    C = d['e29_choice']
    C = C[C.dataset.isin(['NASA', 'NASAdt'])]
    p, g = C[C.model == 'pinn_semi'].ratio, C[C.model == 'gbdt'].ratio
    if not len(p):
        return ''
    txt = (r' Đầu vào có lịch sử (Mục~\ref{sec:hist_m}) không được chọn cho NASA: dạng tốt nhất trên validation, tính '
           r'trung bình qua kiểm định chéo và cách chia NASA-L, vẫn có MAE bằng ' + rng(p, 2) + ' lần đặc trưng gốc với '
           'PINN-semi' + (f' và {rng(g, 2)} lần với GBDT' if len(g) else ''))
    return txt + (r'. Có thể vì dung lượng NASA nhảy lên sau mỗi lần nghỉ: đặc trưng sạc của chu kỳ hiện tại phản ánh '
                  r'ngay cú nhảy đó, còn trung bình trượt làm nó trễ đi.')


# ───────────────────────────────────────────────────────────── kết quả E23: dự báo có xét hồi phục
E23_DS = ['XJTU', 'TJU', 'MIT', 'HUST', 'NASA']


def e23_facts(d):
    ts, rt, dc = d['e23_tests'], d['e23_rul_tests'], d['e23_decomp']
    g = lambda ds, sc, ot, part, a='e_ket_hop': ts[(ts.ds == ds) & (ts.scen == sc) & (ts.otype == ot) & (ts.part == part)
                                                   & (ts.old == a)]
    E = dict(g=g)
    E['row'] = lambda ds, sc, ot, part, a='e_ket_hop': (g(ds, sc, ot, part, a).iloc[0] if len(g(ds, sc, ot, part, a)) else None)
    E['rul'] = lambda ds, sc: (rt[(rt.ds == ds) & (rt.scen == sc)].iloc[0] if len((rt[(rt.ds == ds) & (rt.scen == sc)])) else None)
    E['dec'] = {r.ds: r for r in dc.itertuples()}
    v1 = d.get('e23_tests_v1')                      # bản đầu (tau học bằng gradient), kết quả ở runs/E23_v1
    E['v1'] = (lambda ds, sc, ot, part: None) if v1 is None else (
        lambda ds, sc, ot, part: (lambda x: x.iloc[0] if len(x) else None)(
            v1[(v1.ds == ds) & (v1.scen == sc) & (v1.otype == ot) & (v1.part == part) & (v1.old == 'e_ket_hop')]))
    return E


def e23_text(d, H):
    f, m3, table, fig, T, rng, lst, pc = (H[k] for k in ('f', 'm3', 'table', 'fig', 'T', 'rng', 'lst', 'pc'))
    E = e23_facts(d)
    R = E['row']
    cell = lambda r, which: '--' if r is None else (m3(r[which], 1) + (r'$^{*}$' if which == 'mae_new' and r.p < 0.05
                                                                       and r.mae_new < r.mae_old else '')
                                                    + (r'$^{\dagger}$' if which == 'mae_new' and r.p < 0.05
                                                       and r.mae_new > r.mae_old else ''))
    rows = []
    for ds in E23_DS:
        rr = [R(ds, 'uoc', 0, 'toan_bo'), R(ds, 'do', 0, 'toan_bo'), R(ds, 'do', 1, 'toan_bo'), R(ds, 'do', 1, 'ngan')]
        rows.append([ds] + [cell(r, w) for r in rr for w in ('mae_old', 'mae_new')])
    tab = table(['Bộ', 'Ước lượng, E22', 'Ước lượng, mới', 'Đo, E22', 'Đo, mới', 'Sau hồi phục, E22', 'Sau hồi phục, mới',
                 'Ngắn hạn, E22', 'Ngắn hạn, mới'], rows, 'lrrrrrrrr',
                r'MAE theo cell ($\times10^{-3}$) của mô hình kết hợp trên toàn bộ phần đời còn lại, bản E22 và bản có xét '
                r'hồi phục, ghép cặp trên cùng gốc. Bốn cột đầu: gốc thường, lịch sử ước lượng và lịch sử đo; bốn cột sau: '
                r'gốc ngay sau một lần hồi phục (lịch sử đo), trên toàn bộ phần đời còn lại và trong tầm $0{,}1L$. '
                r'$^{*}$/$^{\dagger}$: tốt hơn/kém hơn có ý nghĩa (Wilcoxon ghép cặp theo cell, $p < 0{,}05$); --: không có '
                r'gốc sau hồi phục.', 'tab:e23')
    better = lambda sc, ot, part: [ds for ds in E23_DS if R(ds, sc, ot, part) is not None and R(ds, sc, ot, part).p < 0.05
                                   and R(ds, sc, ot, part).mae_new < R(ds, sc, ot, part).mae_old]
    worse = lambda sc, ot, part: [ds for ds in E23_DS if R(ds, sc, ot, part) is not None and R(ds, sc, ot, part).p < 0.05
                                  and R(ds, sc, ot, part).mae_new > R(ds, sc, ot, part).mae_old]
    ratios = lambda sc, ot, part: [R(ds, sc, ot, part).ratio for ds in E23_DS if R(ds, sc, ot, part) is not None]
    dec = E['dec']
    rul = [(ds, E['rul'](ds, 'uoc')) for ds in E23_DS if E['rul'](ds, 'uoc') is not None]
    rb = [ds for ds, x in rul if x.p < 0.05 and x.mae_new < x.mae_old]
    rw = [ds for ds, x in rul if x.p < 0.05 and x.mae_new > x.mae_old]
    txt = lambda xs, empty: ('trên ' + lst(xs)) if xs else empty
    sig = lambda r, good: r is not None and r.p < 0.05 and ((r.mae_new < r.mae_old) == good)
    sh_b = [ds for ds in E23_DS if sig(R(ds, 'do', 0, 'ngan'), True)]
    sh_r = [R(ds, 'do', 0, 'ngan').ratio for ds in sh_b]
    x_w, x_s = R('XJTU', 'do', 1, 'toan_bo'), R('XJTU', 'do', 1, 'ngan')
    m_w, n_s, n_w = R('MIT', 'do', 1, 'toan_bo'), R('NASA', 'do', 1, 'ngan'), R('NASA', 'do', 1, 'toan_bo')
    has_ev = [ds for ds in E23_DS if ds in dec and dec[ds].n_ev_train >= 5]
    v1n = E['v1']('NASA', 'do', 1, 'ngan')
    ev_worse = sorted({ds for ds in E23_DS for sc in ('do', 'uoc') for part in ('toan_bo', 'ngan')
                       if R(ds, sc, 1, part) is not None and R(ds, sc, 1, part).p < 0.05
                       and R(ds, sc, 1, part).mae_new > R(ds, sc, 1, part).mae_old})
    keep = [ds for ds in has_ev if dec[ds].rho >= 0.5]; drop = [ds for ds in has_ev if dec[ds].rho < 0.5]
    return T(r"""
\section{Dự báo khi dung lượng hồi phục}\label{sec:recoveryres}

Bảng~\ref{tab:e23} so bản dự báo có xét hồi phục với bản ở Mục~\ref{sec:forecastres}, ghép cặp trên cùng gốc. Rõ nhất là
tầm ngắn: với lịch sử đo, trong tầm $0{,}1L$ sau gốc thường, bản mới giảm sai số có ý nghĩa <<SHB>> (tỉ số <<SHR>>), vì
mức xu thế bền xuất phát đúng hơn và GBDT không còn làm lệch đoạn đầu của đường dự báo. Trên toàn bộ phần đời còn lại,
khác biệt nhỏ hơn: tỉ số là <<RD>> với lịch sử đo và <<RU>> với lịch sử ước lượng; bản mới tốt hơn có ý nghĩa <<BD>>
(lịch sử đo) và <<BU>> (lịch sử ước lượng), kém hơn có ý nghĩa <<WU>>.

Ngay sau một lần hồi phục, bộ lọc quỹ đạo của bản cũ trộn điểm vừa nhảy vào đường hồi quy, nên cú nhảy làm lệch cả mức
lẫn độ dốc của xu thế; bản mới tách cú nhảy khỏi xu thế rồi giữ lại hay bỏ đi theo $\rho$. Trên XJTU, bộ có nhiều gốc
loại này nhất (<<NX>> cell), bản mới giảm sai số <<XW>> trên toàn bộ phần đời còn lại<<XWSIG>> và <<XS>> trong tầm
ngắn<<XSSIG>>; trên MIT, tỉ số là <<MW>> nhưng chỉ có <<NM>> cell nên chưa đủ để kết luận. Trên NASA, tỉ số là <<NW>>
trên toàn bộ phần đời còn lại và <<NS>> trong tầm ngắn<<NSIG>>.

Cặp $(\rho, \tau_r)$ ước lượng trên cell train cho thấy hai kiểu hồi phục: ở <<KEEP>>, phần lớn cú nhảy được giữ lại
(trung vị $\rho$: <<RKEEP>>), còn ở <<DROP>>, phần hồi phục mất ngay ở chu kỳ sau (<<RDROP>>); $\tau_r = $ <<TR>> chu kỳ
ở mọi bộ, tức không có phần hồi phục tắt dần chậm. Trên NASA, điều này khớp với nhận xét của
SambaMixer~\cite{olalde2024samba} rằng hồi phục gắn với thời gian nghỉ. Dao động đều chỉ được phát hiện ở <<PERX>> số gốc
của XJTU, hầu hết thuộc nhóm R2.5. Về RUL, bản mới <<RULTXT>>; khoảng 90\,\% có độ phủ <<COV>> trên toàn bộ phần đời còn
lại, thấp hơn mức đặt ra vì cùng lý do ở Mục~\ref{sec:forecastres}.

<<TAB>>
<<FIG>>
""", SHB=('trên cả bốn bộ ' + lst([d for d in sh_b if d != 'NASA']) if set(sh_b) >= {'XJTU', 'TJU', 'MIT', 'HUST'}
          else txt(sh_b, 'trên không bộ nào')), SHR=rng(sh_r, 2),
             RU=rng(ratios('uoc', 0, 'toan_bo'), 2), RD=rng(ratios('do', 0, 'toan_bo'), 2),
             BU=txt(better('uoc', 0, 'toan_bo'), 'trên không bộ nào'), BD=txt(better('do', 0, 'toan_bo'), 'trên không bộ nào'),
             WU=txt(sorted(set(worse('uoc', 0, 'toan_bo')) | set(worse('do', 0, 'toan_bo'))), 'trên không bộ nào'),
             NX=int(x_w.n), XW=pc(1 - x_w.ratio, 0), XS=pc(1 - x_s.ratio, 0), MW=f(m_w.ratio, 2), NM=int(m_w.n),
             XWSIG=(' (có ý nghĩa thống kê)' if x_w.p < 0.05 else ' (chưa có ý nghĩa thống kê)'),
             XSSIG=(' (có ý nghĩa thống kê)' if x_s.p < 0.05 else ' (chưa có ý nghĩa thống kê)'),
             NS=f(n_s.ratio, 2), NW=f(n_w.ratio, 2),
             NSIG=(', có ý nghĩa thống kê' if (n_s.p < 0.05 or n_w.p < 0.05) else ', chưa có ý nghĩa thống kê'),
             KEEP=lst(keep) or 'không bộ nào', RKEEP=', '.join(f'{ds} {f(dec[ds].rho, 1)}' for ds in keep),
             DROP=lst(drop) or 'không bộ nào',
             RDROP=', '.join(f'{ds} $\\rho$ = {f(dec[ds].rho, 1)}' for ds in drop),
             TR=rng([dec[ds].tau for ds in E23_DS if ds in dec and dec[ds].n_ev_train >= 5], 1),
             PERX=pc(dec['XJTU'].per, 0) if 'XJTU' in dec else '--',
             RULTXT=(('có sai số thấp hơn có ý nghĩa ' + txt(rb, '')) if rb else 'không khác bản cũ có ý nghĩa ở bộ nào')
                    + ((', cao hơn có ý nghĩa ' + txt(rw, '')) if rw else ''),
             COV=rng([100 * x for x in d['e23_cover'].query('otype == 0').cov_r], 0) + r'\,\%', TAB=tab,
             FIG=fig('fig_e23_examples', 'Ba ví dụ với lịch sử đo: dao động đều của nhóm R2.5 (XJTU, gốc ở 40\\,\\% đời), gốc '
                     'ngay sau một lần hồi phục ở XJTU và ở NASA. Đường xám là bản E22, đường đậm là bản có xét hồi phục.',
                     'fig:e23ex'))


# ───────────────────────────────────────────────────────────── E25: cập nhật theo vài lần đo dung lượng
CH4_KF = r"""\section{Học và cập nhật với vài lần đo dung lượng}\label{sec:kalman_m}

Nhãn của bài toán là dung lượng đo bằng một lần xả đầy ở điều kiện chuẩn, nên chi phí nhãn nằm ở số lần đo chứ không ở
số cell. Giao thức chính đếm nhãn theo cell: 30\,\% cell có nhãn nghĩa là mọi chu kỳ của 30\,\% số cell đều được đo.
Mục này mô tả cách mô hình dùng một số ít lần đo rải trên nhiều cell, ở hai thời điểm: khi huấn luyện và khi dùng mô
hình cho một cell mới.

\paragraph{Nhãn thưa khi huấn luyện.} Mỗi cell train được đo dung lượng ở chu kỳ đầu, sau mỗi $qL$ chu kỳ ($L$ là trung
vị số chu kỳ của các cell train) và ở chu kỳ cuối của thử nghiệm; các chu kỳ còn lại chỉ có đặc trưng sạc. Lần đo cuối
là lần kiểm tra khi kết thúc lão hoá, luôn có với cell thử trong phòng thí nghiệm; thiếu nó, một cell ngắn hơn một
khoảng đo chỉ có đúng một lần đo ở chu kỳ đầu. Với PINN-semi, $\mathcal{L}_\text{data}$ chỉ tính trên các chu kỳ được
đo, còn ba loss vật lý vẫn tính trên cặp chu kỳ của mọi hàng của mọi cell: ràng buộc đơn điệu và phần dư động học nối
các lần đo với nhau, nên thông tin của một lần đo lan sang các chu kỳ lân cận mà không cần nhãn ở đó. GBDT không có cơ
chế này; với nó, SOH ở các chu kỳ giữa hai lần đo được lấy theo nội suy tuyến tính trước khi học (lan nhãn). PINN-lai
ở chế độ này giữ nguyên công thức \eqref{eq:lai1}--\eqref{eq:lai3}, với ba mạng PINN-semi học từ nhãn thưa và GBDT học
nhãn nội suy.

\paragraph{Cập nhật theo cell khi dùng.} Sai số còn lại của PINN-lai chủ yếu là độ lệch mức riêng của từng cell
(Mục~\ref{sec:improve}), không thể sửa nếu không có thông tin nào về chính cell. Khi cell mới cũng được đo vài lần (lúc
nhận pin, khi bảo dưỡng định kỳ), gọi $p_n$ là SOH do mô hình ước lượng ở chu kỳ $n$ và $y_n$ là SOH đo được; sai số
được mô hình hoá
\begin{equation}\label{eq:kf}
  y_n - p_n = b_n + \varepsilon_n,\qquad
  b_{n'} - \mu_0 = e^{-(n'-n)/\ell}\,(b_n - \mu_0) + \eta,\qquad
  \operatorname{Var}\varepsilon = R,\ \operatorname{Var} b = P_0,
\end{equation}
tức độ lệch $b_n$ của cell trôi chậm và hồi quy về trung bình tổng thể $\mu_0$ với độ dài tương quan $\ell$ (quá trình
Ornstein--Uhlenbeck), còn $\varepsilon_n$ là nhiễu nhanh. Bộ lọc Kalman cập nhật $b$ mỗi khi có một lần đo; giữa hai lần
đo, thông tin của lần đo phai dần theo khoảng cách chu kỳ. Ước lượng hiệu chỉnh là $p_n + \hat b_n$, với $\hat b_n$ chỉ
dùng các lần đo ở chu kỳ $\le n$, kèm khoảng 90\,\% $\pm 1{,}645\sqrt{\hat P_n + R}$. Bốn tham số lấy trên cell validation
của chính fold: $\mu_0$ là sai số trung bình, $R$ là nửa phương sai của sai phân bậc một, $P_0$ là phần phương sai còn
lại, $\ell$ chọn trên lưới theo MAE riêng cho từng lịch đo. Với dự báo, đường dự báo của mô hình kết hợp có xét hồi phục
được cộng thêm độ lệch hậu nghiệm tại gốc, trôi dần về $\mu_0$ theo \eqref{eq:kf} trong tầm dự báo.
"""


def e26_text(d, H):
    """Đoạn nối vào Mục tham số vật lý: λ, Ea khi tăng trọng số ODE và đổi khởi tạo (E26)."""
    f, T, rng, lst = (H[k] for k in ('f', 'T', 'rng', 'lst'))
    P, M = d['e26_params'], d['e26_mae']
    sh = P.dropna(subset=['s_lam'])
    oth = M[~M.arm.isin(['A0', 'A3'])]
    o_sig = sorted({r.ds for r in oth.itertuples() if r.p < 0.05})
    eu = [f'{f(r.ratio, 2)} trên {r.ds}' + (' (có ý nghĩa)' if r.p < 0.05 else '') for r in M[M.arm == 'A3'].itertuples()]
    return T(r"""
Hai giải thích trên được tách bằng một thí nghiệm bổ sung trên NASA (4, 24, 43\,\textdegree C) và TJU (25, 35,
45\,\textdegree C): huấn luyện lại PINN-semi ở 70\,\% nhãn với $\alpha$ tăng lên 0{,}2 và 1, có hoặc không thêm ràng buộc
Euler, và với khởi tạo $\lambda = 3$, $E_a = 60$\,kJ/mol. Hiệu giữa giá trị học được của hai cách khởi tạo bằng <<SL>> lần
hiệu khởi tạo với $\lambda$ và <<SE>> lần với $E_a$, kể cả khi ràng buộc ODE mạnh gấp 50 lần: tham số đứng yên ở nơi được
đặt. Giải thích hợp lý nhất nằm ở cấu trúc: tốc độ suy giảm chỉ xuất hiện dưới dạng tích $k(\mathbf{x})\,g(u)\,A(T)$, mà
$k(\mathbf{x})$ nhận chính các đặc trưng sạc vốn đã mang thông tin về nhiệt độ và mức SOH, nên $k$ bù lại mọi thay đổi
của $\lambda$ hay $E_a$. Sai số của các nhánh <<OS>> (tỉ số MAE <<RO>>); thêm ràng buộc Euler cho tỉ số <<EU>>.
""", SL=rng(list(sh.s_lam), 2), SE=rng(list(sh.s_ea), 2),
             OS=('không khác cấu hình chuẩn có ý nghĩa' if not o_sig else 'khác cấu hình chuẩn có ý nghĩa trên '
                 + lst(o_sig)), RO=rng(list(oth.ratio), 2), EU=lst(eu))


# ───────────────────────────────────────────────────────────── tinh chỉnh PINN-lai trên validation
def tuned_text(d, H):
    f, m3, table, T, rng, lst = (H[k] for k in ('f', 'm3', 'table', 'T', 'rng', 'lst'))
    e29 = 'e29_summary' in d                # bảng E29 có PINN-lai tinh chỉnh ở cả bốn mức nhãn
    R = d['e20_tuned']
    DSX = ['XJTU', 'TJU', 'MIT', 'HUST']
    r = lambda ds, fr: R[(R.ds == ds) & (R.f == fr)].iloc[0]
    rows = []
    for ds in DSX:
        a, b = r(ds, 0.3), r(ds, 0.7)
        st = lambda x: r'$^{*}$' if x.p_v1 < 0.05 else ''
        rows.append([ds, f'{f(a.omega, 1)}; {int(a.w)}', m3(a.v1, 2), m3(a.v2, 2) + st(a),
                     f'{f(b.omega, 1)}; {int(b.w)}', m3(b.v1, 2), m3(b.v2, 2) + st(b), m3(b.gbdt, 2)])
    tab = table(['Bộ', '$\\omega$; $w$ (30\\,\\%)', 'gốc', 'tinh chỉnh', '$\\omega$; $w$ (70\\,\\%)', 'gốc', 'tinh chỉnh',
                 'GBDT 70\\,\\%'], rows, 'lrrrrrrr',
                r'PINN-lai với $(\omega, w)$ chọn trên validation: MAE theo cell ($\times10^{-3}$) trên test ở 30 và 70\,\% nhãn, '
                r'cạnh bản gốc ($\omega = 0{,}5$, $w = 10$) và GBDT ở 70\,\% nhãn. $^{*}$: khác bản gốc có ý nghĩa (Wilcoxon '
                r'ghép cặp theo cell, $p < 0{,}05$).', 'tab:tuned')
    sig = R[(R.p_v1 < 0.05) & (R.r_v1 < 1)]
    gbw = R[(R.p_gbdt < 0.05) & (R.r_gbdt < 1)]
    gbl = R[(R.p_gbdt < 0.05) & (R.r_gbdt > 1)]
    r50 = R[R.f == 0.5]
    ns50 = [x.ds for x in r50.itertuples() if x.p_gbdt70 >= 0.05]
    return T(r"""
\subsection{Tinh chỉnh trọng số ghép trên validation}\label{sec:tuned}

Đề cương của PINN-lai cố định $\omega = \tfrac12$ trong \eqref{eq:lai2} và $w = 10$ trong \eqref{eq:lai3}. Hai giá trị
này chọn được trên validation mà không phải huấn luyện lại: với mỗi bộ và mức nhãn, $(\omega, w)$ được chọn trên lưới
$\omega \in \{0{,}3;\ 0{,}4;\ 0{,}5;\ 0{,}6;\ 0{,}7\}$, $w \in \{5, 10, 20, 40\}$ bằng MAE theo cell trung bình trên
validation của 3 lần lặp $\times$ 5 fold, rồi đánh giá đúng một lần trên test (<<WHERE>>). Validation nghiêng
về GBDT ($\omega = 0{,}3$) ở XJTU, TJU, MIT khi có từ 30\,\% nhãn và chọn cửa sổ dài ($w = 40$) ở HUST. Trên test, sai
số còn <<RV1>> lần bản gốc (trung bình nhân <<GM>>), giảm có ý nghĩa ở <<NSIG>>/16 trường hợp bộ × mức nhãn; PINN-lai
tinh chỉnh thấp hơn GBDT cùng mức nhãn có ý nghĩa ở <<NGB>>/16 trường hợp<<GBL>>. Ở 50\,\% nhãn, nó không khác GBDT học với
70\,\% nhãn có ý nghĩa <<NS50>>.

<<TAB>>
""", RV1=rng(list(R.r_v1), 2), GM=f(float(np.exp(np.log(R.r_v1).mean())), 3), NSIG=len(sig), NGB=len(gbw),
             GBL=(f', cao hơn có ý nghĩa ở {len(gbl)}' if len(gbl) else ' và không trường hợp nào cao hơn có ý nghĩa'),
             NS50=('trên ' + lst(ns50) if len(ns50) < 4 else 'trên cả bốn bộ') if ns50 else 'trên không bộ nào',
             WHERE=r'cột PINN-lai của Bảng~\ref{tab:e29}' if e29 else r'Bảng~\ref{tab:tuned}', TAB='' if e29 else tab)


# ───────────────────────────────────────────────────────────── E27: so với PINN4SOH
E27_DS = ['XJTU', 'TJU', 'MIT', 'HUST']


def e27_facts(d):
    from pinnsoh.paper import PAPER_MAE
    S, Tt = d['e27_summary'], d['e27_tests']
    nan = float('nan')

    def g(ds, mode, prep, model):
        r = S[(S.dataset == ds) & (S['mode'] == mode) & (S.prep == prep) & (S.model == model)]
        return float(r.mae.iloc[0]) if len(r) else nan
    P = lambda ds, mode, ref='PINN4SOH': PAPER_MAE[(ds, mode)][ref]
    tt = lambda ds, prep: Tt[(Tt.dataset == ds) & (Tt.prep == prep)].iloc[0]
    E = dict(g=g, P=P, tt=tt)
    E['rep'] = {ds: g(ds, 'full', 'paper', 'pinn_bb') / P(ds, 'full') for ds in E27_DS}
    E['lfl'] = {ds: g(ds, 'full', 'paper', 'pinn_lai') / P(ds, 'full') for ds in E27_DS}
    E['strict'] = {ds: g(ds, 'full', 'strict', 'pinn_lai') / P(ds, 'full') for ds in E27_DS}
    E['leak'] = {ds: g(ds, 'full', 'strict', 'pinn_bb') / g(ds, 'full', 'paper', 'pinn_bb') for ds in E27_DS}
    E['small'] = {(ds, n, prep): g(ds, f'n{n}', prep, 'pinn_lai') / P(ds, f'n{n}')
                  for ds in ('XJTU', 'HUST') for n in (1, 2, 3, 4) for prep in ('strict', 'paper')}
    E['small_gb'] = {(ds, n): g(ds, f'n{n}', 'strict', 'gbdt') / P(ds, f'n{n}') for ds in ('XJTU', 'HUST') for n in (1, 2, 3, 4)}
    return E


def e27_text(d, H):
    f, m3, table, fig, T, rng, lst = (H[k] for k in ('f', 'm3', 'table', 'fig', 'T', 'rng', 'lst'))
    E = e27_facts(d)
    g, P, tt = E['g'], E['P'], E['tt']
    rows = []
    for ds in E27_DS:
        rows.append([ds, m3(P(ds, 'full'), 2), m3(P(ds, 'full', 'MLP'), 2), m3(g(ds, 'full', 'paper', 'pinn_bb'), 2),
                     m3(g(ds, 'full', 'paper', 'pinn_lai'), 2), m3(g(ds, 'full', 'strict', 'pinn_bb'), 2),
                     m3(g(ds, 'full', 'strict', 'gbdt'), 2), m3(g(ds, 'full', 'strict', 'pinn_lai'), 2)])
    tab = table(['Bộ', 'PINN4SOH', 'MLP', 'Hộp đen', 'PINN-lai', 'Hộp đen', 'GBDT', 'PINN-lai'], rows, 'lrrrrrrr',
                r'MAE theo cell ($\times10^{-3}$) trên đúng cách chia của PINN4SOH. Hai cột đầu là số công bố của bài; hai cột '
                r'tiếp theo là mô hình của đồ án với bản sao tiền xử lý của bài (có rò rỉ); ba cột cuối dùng tiền xử lý '
                r'không rò rỉ của đồ án. Hộp đen: PINN động học hộp đen kiểu PINN4SOH. Trung bình 3 hạt giống, rồi qua các '
                r'batch như bài.', 'tab:e27')
    sig = lambda prep, better: [ds for ds in E27_DS if tt(ds, prep).p < 0.05 and (tt(ds, prep).ours < tt(ds, prep).theirs) == better]
    lk = E['leak']
    up = [ds for ds in E27_DS if lk[ds] > 1.05]
    down = [ds for ds in E27_DS if lk[ds] < 0.95]
    st = E['strict']
    win = [ds for ds in E27_DS if st[ds] < 0.95]
    tie = [ds for ds in E27_DS if 0.95 <= st[ds] <= 1.05]
    lose = [ds for ds in E27_DS if st[ds] > 1.05]
    sb, sw = sig('strict', True), sig('strict', False)
    sm = E['small']
    small_paper = [x for (ds, n, prep), x in sm.items() if prep == 'paper']
    sp_all = all(x < 1 for x in small_paper)
    sw_strict = [(ds, n) for (ds, n, prep), x in sm.items() if prep == 'strict' and x < 1]
    rest = [x for (ds, n, prep), x in sm.items() if prep == 'strict' and x >= 1]
    return T(r"""
\section{So với PINN4SOH trên đúng cách chia của bài}\label{sec:paperres}

Bảng~\ref{tab:e27} và Hình~\ref{fig:e27} so mô hình của đồ án với số công bố của PINN4SOH trên đúng các cell test của
bài. Trước hết, bản dựng lại có đáng tin không: PINN động học hộp đen của đồ án, khi dùng bản sao tiền xử lý của bài, có
MAE bằng <<REP>> lần số công bố, tức cùng cỡ với bài dù cấu hình mạng và lịch học khác. Cùng điều kiện đó, PINN-lai có
MAE bằng <<LFL>> lần PINN4SOH<<LFLSIG>>.

<<TAB>>

Tiền xử lý của bài vừa thêm một thông tin rò rỉ vừa bỏ đi một thông tin thật, và hai tác động này cân nhau khác nhau theo
bộ. Đổi sang tiền xử lý không rò rỉ, sai số của PINN hộp đen tăng <<LUP>> lần trên <<UPDS>>: chỉ số chu kỳ sau min--max
theo từng cell chính là phần đời đã qua của cell, thông tin mà một mô hình triển khai thật không thể có. Ngược lại, sai
số giảm còn <<LDOWN>> lần trên <<DOWNDS>>: min--max theo từng cell xoá mức tuyệt đối của đặc trưng sạc, vốn mang thông
tin về SOH. Với tiền xử lý không rò rỉ, PINN-lai có sai số thấp hơn số công bố của PINN4SOH trên <<WIN>><<TIE>><<LOSE>>.

<<FIG>>

Ở chế độ ít mẫu (Hình~\ref{fig:e27}, dưới), với tiền xử lý của bài, PINN-lai có sai số <<SPTXT>> (tỉ số <<SMP>>): các
cell train không nhãn đi vào loss vật lý và bù cho số nhãn ít, rõ nhất khi chỉ một cell có nhãn. Với tiền xử lý không rò
rỉ, <<SMS>>. Khi chỉ có vài cell có nhãn, biết cell đã đi được bao nhiêu phần đời gần như là biết SOH, nên con số ít
mẫu của bài phần lớn đến từ việc chuẩn hoá theo cả vòng đời cell.
""", REP=rng(list(E['rep'].values()), 2), LFL=rng(list(E['lfl'].values()), 2),
             LFLSIG=((', thấp hơn có ý nghĩa thống kê theo cell trên ' + lst(sig('paper', True))) if sig('paper', True)
                     else '') + ((', cao hơn có ý nghĩa trên ' + lst(sig('paper', False))) if sig('paper', False) else ''),
             TAB=tab, LUP=rng([lk[ds] for ds in up], 2) if up else '--', UPDS=lst(up) if up else 'không bộ nào',
             LDOWN=rng([lk[ds] for ds in down], 2) if down else '--', DOWNDS=lst(down) if down else 'không bộ nào',
             WIN=(lst(win) + ' (tỉ số ' + rng([st[ds] for ds in win], 2)
                  + (', có ý nghĩa thống kê theo cell trên ' + lst([x for x in win if x in sb]) if [x for x in win if x in sb] else '')
                  + ')') if win else 'không bộ nào',
             TIE=(f', ngang trên {lst(tie)} (tỉ số {rng([st[ds] for ds in tie], 2)})' if tie else ''),
             LOSE=(f', và chỉ cao hơn trên {lst(lose)} (tỉ số {rng([st[ds] for ds in lose], 2)}), đúng bộ mà tiền xử lý của '
                   'bài có lợi nhất' if lose else ''),
             FIG=fig('fig_e27_compare', 'So với PINN4SOH trên đúng cách chia của bài. Trên: chế độ đầy đủ, mỗi bộ năm cột '
                     '(số công bố, rồi PINN hộp đen và PINN-lai của đồ án với tiền xử lý của bài và với tiền xử lý không rò '
                     'rỉ). Dưới: chế độ ít mẫu, MAE theo số cell có nhãn (thang log).', 'fig:e27'),
             SPTXT=(r'thấp hơn PINN4SOH ở mọi mức $K = 1, \dots, 4$ trên cả XJTU và HUST' if sp_all else
                    'so với PINN4SOH'), SMP=rng(small_paper, 2),
             SMS=(('PINN-lai chỉ thấp hơn số công bố ở ' + lst([f'{ds} với $K = {n}$' for ds, n in sw_strict])
                   + (f' (tỉ số {rng([sm[(ds, n, "strict")] for ds, n in sw_strict], 2)})') if sw_strict
                   else 'PINN-lai không thấp hơn số công bố ở mức nào')
                  + (f'; ở các mức còn lại tỉ số là {rng(rest, 2)}' if rest else '')))


# ───────────────────────────────────────────────────────────── E28: ngân sách đo
E28_DS = ['XJTU', 'TJU', 'MIT', 'HUST']


def fq(f, q):
    """Lịch đo q không có số 0 thừa: 0,05; 0,1; 0,5."""
    return f(q, 1) if abs(10 * q - round(10 * q)) < 1e-9 else f(q, 2)


def e28_facts(d):
    S, Tt = d['e28_summary'], d['e28_tests']
    nan = float('nan')

    def row(ds, kind, regime, model):
        r = S[(S.dataset == ds) & (S.kind == kind) & (S.regime == regime) & (S.model == model)]
        return r.iloc[0] if len(r) else None
    mae = lambda *a: float(row(*a).mae) if row(*a) is not None else nan
    fr = lambda *a: float(row(*a).frac_rows) if row(*a) is not None else nan
    tt = lambda ds, q, model, ref: (lambda r: r.iloc[0] if len(r) else None)(
        Tt[(Tt.dataset == ds) & (Tt.q == q) & (Tt.model == model) & (Tt.ref == ref)])
    qs = sorted({float(x[1:]) for x in S[S.kind == 'do_thua'].regime.unique()}, reverse=True)
    q = min(qs)
    cmp = lambda model, ref: {ds: r for ds in E28_DS for r in [tt(ds, q, model, ref)] if r is not None}
    E = dict(mae=mae, fr=fr, tt=tt, qs=qs, qmin=q, qmax=max(qs),
             c30=cmp('pinn_lai', 'pinn_lai@30'), c70=cmp('pinn_lai', 'pinn_lai@70'),
             t30=cmp('pinn_lai_v2', 'pinn_lai_v2@30'), t70=cmp('pinn_lai_v2', 'pinn_lai_v2@70'),
             gi70=cmp('gbdt_interp', 'gbdt@70'),
             fewer=[fr(ds, 'theo_cell', 'f0.3', 'pinn_lai') / fr(ds, 'do_thua', f'q{q}', 'pinn_lai') for ds in E28_DS])
    if 'e28_tuned' in d:
        om = d['e28_tuned'].set_index(['dataset', 'q']).om
        E['omega'] = lambda ds, q_: float(om[(ds, q_)])
    if 'e28_deploy' in d:
        D = d['e28_deploy']
        D = D.groupby(['dataset', 'q', 'src', 'cell'])[['goc', 'kalman', 'n_meas', 'n_rows']].mean().reset_index()
        E['dep'] = lambda ds, q_, src, col='kalman': float(D[(D.dataset == ds) & (D.q == q_) & (D.src == src)][col].mean())
        E['dep_frac'] = lambda ds, q_: float((D[(D.dataset == ds) & (D.q == q_) & (D.src == 'sparse')].n_meas
                                             / D[(D.dataset == ds) & (D.q == q_) & (D.src == 'sparse')].n_rows).mean())
    return E


def _verdict(C, lst):
    """'tốt hơn có ý nghĩa ở TJU, kém hơn có ý nghĩa ở MIT, khác biệt không có ý nghĩa ở XJTU và HUST'"""
    b = [ds for ds in E28_DS if ds in C and C[ds].p < 0.05 and C[ds].ratio < 1]
    w = [ds for ds in E28_DS if ds in C and C[ds].p < 0.05 and C[ds].ratio > 1]
    n = [ds for ds in E28_DS if ds in C and C[ds].p >= 0.05]
    return ', '.join(([f'tốt hơn có ý nghĩa ở {lst(b)}'] if b else []) + ([f'kém hơn có ý nghĩa ở {lst(w)}'] if w else [])
                     + ([f'khác biệt không có ý nghĩa ở {lst(n)}'] if n else []))


def e28_text(d, H):
    f, m3, table, fig, T, rng, lst, tren = (H[k] for k in ('f', 'm3', 'table', 'fig', 'T', 'rng', 'lst', 'tren'))
    E = e28_facts(d)
    mae, fr, qmin, qmax = E['mae'], E['fr'], E['qmin'], E['qmax']
    DSX = E28_DS
    q2 = sorted(E['qs'])[1] if len(E['qs']) > 1 else qmin
    cell = lambda v, p: f'{m3(v, 2)} ({f(100 * p, 1)})'
    rows = []
    for ds in DSX:
        r = [ds, cell(mae(ds, 'theo_cell', 'f0.3', 'pinn_lai'), fr(ds, 'theo_cell', 'f0.3', 'pinn_lai')),
             m3(mae(ds, 'theo_cell', 'f0.7', 'pinn_lai'), 2), m3(mae(ds, 'theo_cell', 'f0.7', 'gbdt'), 2),
             cell(mae(ds, 'do_thua', f'q{q2}', 'pinn_lai'), fr(ds, 'do_thua', f'q{q2}', 'pinn_lai')),
             cell(mae(ds, 'do_thua', f'q{qmin}', 'pinn_lai'), fr(ds, 'do_thua', f'q{qmin}', 'pinn_lai'))]
        if 'dep' in E:
            r.append(m3(E['dep'](ds, q2, 'sparse'), 2))
        rows.append(r)
    hdr = ['Bộ', 'PINN-lai, 30\\,\\% cell', 'PINN-lai, 70\\,\\% cell', 'GBDT, 70\\,\\% cell',
           f'PINN-lai, đo mỗi ${fq(f, q2)}L$', f'PINN-lai, đo mỗi ${fq(f, qmin)}L$']
    if 'dep' in E:
        hdr.append(f'đo cả cell mới mỗi ${fq(f, q2)}L$')
    tab = table(hdr, rows, 'l' + 'r' * (len(hdr) - 1),
                r'MAE theo cell ($\times10^{-3}$) trên test, trung bình 3 lần lặp; trong ngoặc là phần trăm số chu kỳ của cell '
                r'train có đo dung lượng (70\,\% số cell là toàn bộ). Hai cột đo thưa: mọi cell train được đo ở chu kỳ đầu, '
                r'sau mỗi $qL$ chu kỳ và ở chu kỳ cuối. Cột cuối: đo thêm cell test theo cùng lịch, cập nhật bằng bộ lọc '
                r'Kalman, sai số trên các chu kỳ không được đo.', 'tab:e28')
    c30, c70, t30, t70 = E['c30'], E['c70'], E['t30'], E['t70']
    better30 = [ds for ds in DSX if ds in c30 and c30[ds].ratio < 1]
    sig30 = [ds for ds in better30 if c30[ds].p < 0.05]
    s30 = (', cả bốn có ý nghĩa thống kê' if len(sig30) == len(DSX) else
           (f', có ý nghĩa thống kê ở {lst(sig30)}' if sig30 else ', không bộ nào có ý nghĩa thống kê'))
    p3 = lambda ds, q_: mae(ds, 'do_thua', f'q{q_}', 'pinn3')
    gi = lambda ds, q_: mae(ds, 'do_thua', f'q{q_}', 'gbdt_interp')
    gb_lo = [ds for ds in DSX if gi(ds, qmin) < p3(ds, qmin)]
    gb_hi = [ds for ds in DSX if gi(ds, qmax) > p3(ds, qmax)]
    mlp_hi = [mae(ds, 'do_thua', f'q{qmax}', 'mlp') / p3(ds, qmax) for ds in DSX]
    out = T(r"""
\section{Ngân sách đo dung lượng}\label{sec:budgetres}

\subsection{Đo thưa khi huấn luyện}

Hình~\ref{fig:e28} đặt hai cách gắn nhãn lên cùng một trục: phần trăm số chu kỳ của cell train có đo dung lượng. Gắn
nhãn đầy đủ cho 30\,\% số cell tương ứng <<F30>> số chu kỳ train, 70\,\% số cell là toàn bộ. Đo mỗi $<<QMIN>>L$ chu kỳ
trên mọi cell chỉ dùng <<FQ>> số chu kỳ, ít hơn <<FEWER>> lần, nhưng PINN-lai học theo cách này có MAE thấp hơn PINN-lai
học với 30\,\% số cell <<B30>> (tỉ số <<R30>><<S30>>; Bảng~\ref{tab:e28}). So với 70\,\% số cell, tỉ số là <<R70>>:
<<V70>>. Phân tích thêm sau khi có các số trên: khi trọng số trộn và cửa sổ lọc được chọn trên validation cho cả hai
cách gắn nhãn như Mục~\ref{sec:tuned}, hai tỉ số này là <<T30>> và <<T70>>.

<<FIG>>

Lợi ích nằm ở chỗ đặt nhãn hơn là ở mô hình. Phần lớn sai số là lệch mức và trôi chậm riêng của từng cell
(Mục~\ref{sec:errdecomp}); mô hình chỉ học được phần này khi thấy nhãn của chính cell đó, trong khi các chu kỳ liền nhau
của một cell gần như lặp lại thông tin của nhau. Đo kín một phần cell dồn công đo vào thông tin lặp lại và bỏ trống độ
lệch của các cell còn lại. Vì vậy GBDT học nhãn nội suy tuyến tính giữa các lần đo ở lịch $<<QMIN>>L$ cũng có sai số
bằng <<GI70>> lần GBDT học với 70\,\% số cell, và Hình~\ref{fig:overview} cho thấy điều tương tự ở mọi mô hình.

Không thành phần nào của PINN-lai tốt nhất ở mọi lịch đo. Ở lịch $<<QMIN>>L$, GBDT học nhãn nội suy tốt hơn trung bình
ba PINN-semi <<GBLO>>. Ở lịch thưa nhất ($<<QMAX>>L$, khoảng ba lần đo mỗi cell), nội suy tuyến tính qua nửa đời cell bỏ
qua đoạn suy giảm nhanh về cuối, và GBDT kém hơn ba PINN-semi <<GBHI>>; MLP chỉ học trên các lần đo cũng kém ba PINN-semi
<<MLPR>> lần, vì ràng buộc đơn điệu và động học trên mọi hàng là thứ nối các lần đo với nhau.<<OM>>

<<TAB>>
""", F30=rng([100 * fr(ds, 'theo_cell', 'f0.3', 'pinn_lai') for ds in DSX], 0) + r'\,\%',
             QMIN=fq(f, qmin), QMAX=fq(f, qmax),
             FQ=rng([100 * fr(ds, 'do_thua', f'q{qmin}', 'pinn_lai') for ds in DSX], 1) + r'\,\%',
             FEWER=rng(E['fewer'], 0),
             B30=tren(better30) or 'trên không bộ nào', R30=rng([c30[ds].ratio for ds in c30], 2), S30=s30,
             R70=rng([c70[ds].ratio for ds in c70], 2), V70=_verdict(c70, lst),
             T30=rng([t30[ds].ratio for ds in t30], 2), T70=rng([t70[ds].ratio for ds in t70], 2),
             GI70=rng([E['gi70'][ds].ratio for ds in E['gi70']], 2),
             GBLO=tren(gb_lo) or 'trên không bộ nào',
             GBHI=(tren(gb_hi) + f' (tới {f(max(gi(ds, qmax) / p3(ds, qmax) for ds in gb_hi), 1)} lần)') if gb_hi
             else 'trên không bộ nào',
             MLPR=rng(mlp_hi, 1), TAB=tab,
             OM=_omega_sentence(E, f, lst, qmin, qmax) if 'omega' in E else '',
             FIG=fig('fig_e28_budget', 'MAE theo cell trên test theo phần trăm số chu kỳ của cell train có đo dung lượng '
                     '(hai trục thang log). Nét đứt: nhãn theo cell (10, 30, 50, 70\\,\\% số cell có nhãn đầy đủ); nét liền: đo '
                     'thưa mọi cell train theo lịch $qL$, $q$ = 0{,}5; 0{,}2; 0{,}1; 0{,}05.', 'fig:e28'))
    if 'dep' in E:
        dep = {ds: E['dep'](ds, q2, 'sparse') for ds in DSX}
        dep70 = {ds: E['dep'](ds, q2, 'full70') for ds in DSX}
        g70 = {ds: mae(ds, 'theo_cell', 'f0.7', 'gbdt') for ds in DSX}
        below = [ds for ds in DSX if dep[ds] < g70[ds]]
        out += T(r"""
\subsection{Đo thưa cả khi dùng}

Khi cell mới cũng được đo theo lịch $<<Q2>>L$ (<<DF>> số chu kỳ của nó) và ước lượng được cập nhật bằng bộ lọc Kalman
\eqref{eq:kf}, MAE trên các chu kỳ không được đo của PINN-lai học từ nhãn thưa còn <<DEP>> ($\times10^{-3}$), bằng
<<DEPR>> lần GBDT học với 70\,\% số cell có nhãn đầy đủ nhưng không đo cell mới<<BELOW>>. Với cùng lịch đo khi dùng,
PINN-lai học với 70\,\% số cell cho <<DEP70>>: khi cell mới đã được đo, chênh lệch giữa hai cách huấn luyện không quá
<<DIFF>> ($\times10^{-3}$).
""", Q2=fq(f, q2), DF=rng([100 * E['dep_frac'](ds, q2) for ds in DSX], 0) + r'\,\%',
                 DEP=rng([1e3 * v for v in dep.values()], 1), DEPR=rng([dep[ds] / g70[ds] for ds in DSX], 2),
                 BELOW='' if len(below) == len(DSX) else f' (thấp hơn {tren(below) or "trên không bộ nào"})',
                 DEP70=rng([1e3 * v for v in dep70.values()], 1),
                 DIFF=f(max(abs(1e3 * (dep[ds] - dep70[ds])) for ds in DSX), 1))
    return out


def _omega_sentence(E, f, lst, qmin, qmax):
    lo = [ds for ds in E28_DS if E['omega'](ds, qmin) <= 0.3]
    hi = [ds for ds in E28_DS if E['omega'](ds, qmax) >= 0.7]
    parts = ([rf'$\omega = 0{{,}}3$ (nghiêng về GBDT) ở lịch ${fq(f, qmin)}L$ trên {lst(lo)}'] if lo else []) + \
            ([rf'$\omega = 0{{,}}7$ ở lịch ${fq(f, qmax)}L$ trên {lst(hi)}'] if hi else [])
    return (' Trọng số chọn trên validation đi theo lịch đo: ' + ', '.join(parts) + '.') if parts else ''


# ───────────────────────────────────────────────────────────── E29: đầu vào có lịch sử của cell
E29_DSA = ['XJTU', 'TJU', 'MIT', 'HUST']


def e29_facts(d):
    S, Tt = d['e29_summary'], d['e29_tests']

    def mae(ds, fr, m):
        r = S[(S.dataset == ds) & (S.f == fr) & (S.model == m)]
        return float(r.mae.iloc[0]) if len(r) else None
    pair = lambda m, ref: Tt[(Tt.model == m) & (Tt.ref == ref)]
    one = lambda m, ref, ds, fr: (lambda r: r.iloc[0] if len(r) else None)(
        Tt[(Tt.model == m) & (Tt.ref == ref) & (Tt.dataset == ds) & (Tt.f == fr)])
    DSH = [ds for ds in E29_DSA if (S[S.dataset == ds].model == 'pinn_h').any()]
    return dict(S=S, DS=DSH, mae=mae, pair=pair, one=one, fracs=sorted(S.f.unique()))


def _sig(X):
    """(số trường hợp tốt hơn có ý nghĩa, kém hơn có ý nghĩa, tổng)"""
    return int(((X.p < 0.05) & (X.ratio < 1)).sum()), int(((X.p < 0.05) & (X.ratio > 1)).sum()), len(X)


def e29_groups(E):
    """Chia các trường hợp (bộ, mức nhãn) có lượt E29 theo tác động của lịch sử lên PINN-lai tinh chỉnh trên test:
    'good' — tốt hơn có ý nghĩa; phần còn lại."""
    lv = E['pair']('pinn_lai_h_v2', 'pinn_lai_v2')
    good = lv[(lv.p < 0.05) & (lv.ratio < 1)]
    gds = [ds for ds in E29_DSA if ds in set(good.dataset)]
    fmin = float(good.f.min()) if len(good) else None
    return lv, good, gds, fmin


def e29_text(d, H):
    f, pc, m3, table, fig, T, rng, lst = (H[k] for k in ('f', 'pc', 'm3', 'table', 'fig', 'T', 'rng', 'lst'))
    E = e29_facts(d)
    mae, pair, one, DSH = E['mae'], E['pair'], E['one'], E['DS']
    mark = lambda r: '' if r is None else (r'$^{*}$' if r.p < 0.05 and r.ratio < 1 else
                                           (r'$^{\dagger}$' if r.p < 0.05 and r.ratio > 1 else ''))
    val = lambda x: '--' if x is None else m3(x, 2)
    rows = []
    for ds in E29_DSA:
        for fr in E['fracs']:
            h = ds in DSH
            rows.append([ds if fr == E['fracs'][0] else '', f'{int(round(100 * fr))}\\,\\%',
                         val(mae(ds, fr, 'pinn')),
                         val(mae(ds, fr, 'pinn_h')) + (mark(one('pinn_h', 'pinn', ds, fr)) if h else ''),
                         val(mae(ds, fr, 'gbdt_best')),
                         val(mae(ds, fr, 'pinn_lai_v2')),
                         val(mae(ds, fr, 'pinn_lai_h_v2')) + (mark(one('pinn_lai_h_v2', 'pinn_lai_v2', ds, fr)) if h else ''),
                         val(mae(ds, fr, 'pinn_lai_sel_v2')) + mark(one('pinn_lai_sel_v2', 'gbdt_best', ds, fr))])
    tab = table(['Bộ', 'Nhãn', 'PINN-semi', '+ lịch sử', 'GBDT', 'PINN-lai', '+ lịch sử', 'Mô hình cuối'], rows,
                'llrrrrrr',
                r'MAE theo cell ($\times10^{-3}$) trên test, 3 lần lặp $\times$ 5 fold, cùng fold và hạt giống với '
                r'Mục~\ref{sec:e20res}. PINN-lai là bản tinh chỉnh trên validation (Mục~\ref{sec:tuned}); GBDT là bản có '
                r'hoặc không lịch sử, chọn trên validation; mô hình cuối là PINN-lai dùng hay không dùng lịch sử theo '
                r'validation của từng bộ và mức nhãn. $^{*}$/$^{\dagger}$: tốt hơn/kém hơn có ý nghĩa (Wilcoxon ghép cặp '
                r'theo cell, $p < 0{,}05$) so với cột không lịch sử ngay trước; ở cột cuối, so với GBDT.', 'tab:e29',
                size=r'\footnotesize')
    # lựa chọn dạng lịch sử ở E29dev (lần lặp 0, 30 và 70 % nhãn)
    C = d['e29_choice'].set_index(['model', 'dataset'])
    chosen = lambda m: [ds for ds in E29_DSA if (m, ds) in C.index and C.loc[(m, ds), 'chosen'] != 'global']
    kp, kg = chosen('pinn_semi'), chosen('gbdt')
    dp = [ds for ds in E29_DSA if ds not in kp]; dg = [ds for ds in E29_DSA if ds not in kg]
    # tác động trên test
    lv, good, gds, fmin = e29_groups(E)
    ph = pair('pinn_h', 'pinn')
    inA = lambda X: X[X.dataset.isin(gds) & (X.f >= fmin)]
    bad = [ds for ds in DSH if ds not in gds]
    phX, lvX = ph[ph.dataset.isin(bad)], lv[lv.dataset.isin(bad)]
    ph10, lv10 = ph[ph.f < fmin], lv[lv.f < fmin]
    cnt = lambda X: (lambda b, w, n: f'{b}/{n}')(*_sig(X))
    # vì sao chọn sai: tỉ số validation theo lần lặp (analysis/analyze_e29.py: val_ratios)
    V = d['e29_val']
    vb = V[V.dataset.isin(bad) & V.f.isin([0.3, 0.7])]
    m10 = pair('mlp_h', 'mlp'); m10 = m10[m10.f < fmin]
    # chọn hậu kiểm và so với GBDT
    Sel = d['e29_select']
    sv2 = Sel[(Sel.kind == 'pinn_lai_v2') & (Sel.src == 'hist')]
    selh = sorted({ds for ds in sv2.dataset}, key=E29_DSA.index)
    sfr = sorted(sv2.f.unique())
    fin, finf = pair('pinn_lai_sel_v2', 'gbdt_best'), pair('pinn_lai_sel_v2', 'gbdt_best_f')
    nb, _, n = _sig(fin)
    nbf, nwf, nf = _sig(finf)
    # vật lý khi đã có lịch sử
    pm_h, pm = pair('pinn_h', 'mlp_h'), pair('pinn', 'mlp')
    lo, hi = pm_h[pm_h.f < fmin], pm_h[pm_h.f >= fmin]
    weak = [ds for ds in DSH if (hi[hi.dataset == ds].ratio >= 1).any()]
    strong = [ds for ds in DSH if ds not in weak]
    # các nhận định trong đoạn văn phải đúng với số liệu
    assert bad == ['XJTU'] and weak == ['TJU'], (bad, weak)
    assert (phX.ratio > 1).all() and _sig(ph10)[0] == 0 and _sig(lv10)[0] == 0
    assert set(zip(sv2.dataset, sv2.f)) == set(zip(good.dataset, good.f))
    return T(r"""
\section{Đầu vào có lịch sử của cell}\label{sec:histres}

Dạng lịch sử được chọn ở bước khám phá, trên validation của lần lặp 0 với 30 và 70\,\% nhãn, rồi chốt trước khi chạy
đánh giá cuối. Lịch sử giảm MAE validation của PINN-semi trên <<KP>> (tỉ số <<RKP>>) và của GBDT trên <<KG>>
(<<RKG>>); với PINN-semi trên <<DP>> và với GBDT trên <<DG>>, không có dạng nào đạt ngưỡng 2\,\% nên các mô hình này giữ
đặc trưng gốc. Trên NASA, tính trung bình qua các cách chia, mọi dạng đều kém hơn đặc trưng gốc
(Mục~\ref{sec:nasares}).

<<FIG>>
Trên test (Bảng~\ref{tab:e29}, Hình~\ref{fig:e29}), lịch sử giúp rõ trên <<GDS>> khi có từ <<FMIN>> nhãn: MAE của
PINN-semi còn <<RPA>> lần bản gốc (tốt hơn có ý nghĩa ở <<CPA>> trường hợp) và của PINN-lai còn <<RLA>> lần (<<CLA>>).
Trên <<BAD>> thì ngược lại: PINN-semi có lịch sử kém bản gốc ở mọi mức nhãn (<<RPX>> lần), PINN-lai kém <<RLX>> lần. Ở
10\,\% nhãn, lịch sử không giảm sai số có ý nghĩa trên bộ nào (PINN-semi <<RP10>> lần, PINN-lai <<RL10>> lần).

<<TAB>>
Lựa chọn ở bước khám phá sai trên <<BAD>> vì nó chỉ dựa trên một lần lặp. Với lần lặp 0, tỉ số MAE validation của
PINN-semi có và không có lịch sử trên <<BAD>> là <<V0>> ở 30 và 70\,\% nhãn; với lần lặp 2, nó là <<V2>>; trung bình ba
lần lặp là <<VA>>, gần với tỉ số trên test (<<TA>>). Mỗi fold của XJTU chỉ có 6 cell validation, nên tác động của lịch sử
đổi theo cách chia cell. Ở 10\,\% nhãn, đầu vào có lịch sử có số chiều gấp đôi đến gấp bốn trong khi chỉ còn vài cell có
nhãn; ngay cả MLP có lịch sử cũng kém MLP gốc <<RM10>> lần ở mức nhãn này.

Validation của đủ ba lần lặp đã có sẵn trong các lượt trên, nên có thể chọn lại mà không dùng test: với mỗi bộ và mức
nhãn, PINN-lai dùng hay không dùng lịch sử, cùng trọng số trộn và cửa sổ lọc, được chọn theo MAE validation trung bình
của 3 lần lặp $\times$ 5 fold; GBDT được chọn có hay không lịch sử theo cùng cách. Bước này được đặt ra sau khi đã thấy
kết quả test ở trên, nên là phân tích hậu kiểm chứ không thuộc đề cương. Validation chọn lịch sử ở <<SELH>> với
<<SFR>> nhãn và bỏ nó ở mọi trường hợp còn lại, trùng với các trường hợp lịch sử có lợi trên test. Với lựa chọn này,
PINN-lai (cột cuối của Bảng~\ref{tab:e29}) có MAE thấp hơn GBDT ở <<NB>>/<<N>> trường hợp bộ $\times$ mức nhãn (tỉ số
<<RB>>), có ý nghĩa ở <<NS>>. Khi GBDT cũng được qua
bộ lọc xu thế với cửa sổ chọn trên validation, tỉ số là <<RF>>, thấp hơn có ý nghĩa ở <<NSF>>/<<NF>><<WF>>.

Ràng buộc vật lý giúp nhiều nhất khi ít nhãn và không giúp ở mọi bộ. Khi cả hai cùng có lịch sử, PINN-semi có MAE bằng
<<RLO>> lần MLP ở 10\,\% nhãn, tốt hơn có ý nghĩa <<SLO>>. Từ 30\,\% nhãn, tỉ số là <<RSTRONG>> trên <<STRONG>> nhưng
<<RWEAK>> trên <<WEAK>>, nơi MLP có lịch sử tốt hơn PINN-semi có lịch sử (có ý nghĩa ở <<NWEAK>>). Không có lịch sử,
<<WEAK>> cũng như vậy
(<<RWEAK0>> ở 50 và 70\,\% nhãn): bộ này gộp ba nhóm hoá học và nhiều nhóm điều kiện, nên một động học dùng chung cho
mọi cell có thể không khớp với từng nhóm.
""", KP=lst(kp), RKP=rng([C.loc[('pinn_semi', ds), 'ratio'] for ds in kp], 2), KG=lst(kg),
             RKG=rng([C.loc[('gbdt', ds), 'ratio'] for ds in kg], 2),
             DP=lst(dp), DG=lst(dg),
             GDS=lst(gds), FMIN=pc(fmin), RPA=rng(inA(ph).ratio, 2), CPA=cnt(inA(ph)), RLA=rng(inA(lv).ratio, 2),
             CLA=cnt(inA(lv)), BAD=lst(bad), RPX=rng(phX.ratio, 2), RLX=rng(lvX.ratio, 2), RP10=rng(ph10.ratio, 2),
             RL10=rng(lv10.ratio, 2), V0=rng(vb.val_r0, 2), V2=rng(vb.val_r2, 2), VA=rng(vb.val, 2), TA=rng(vb.test, 2),
             RM10=rng(m10.ratio, 2), SELH=lst(selh), SFR=lst([pc(x) for x in sfr]),
             NB=int((fin.ratio < 1).sum()), N=n, RB=rng(fin.ratio, 2), NS=nb, RF=rng(finf.ratio, 2), NSF=nbf, NF=nf,
             WF=(f', cao hơn có ý nghĩa ở {nwf}' if nwf else ' và không trường hợp nào cao hơn có ý nghĩa'),
             RLO=rng(lo.ratio, 2), SLO=('trên cả ba bộ' if _sig(lo)[0] == len(lo) else f'ở {_sig(lo)[0]}/{len(lo)}'),
             RSTRONG=rng(hi[hi.dataset.isin(strong)].ratio, 2), STRONG=lst(strong),
             RWEAK=rng(hi[hi.dataset.isin(weak)].ratio, 2), WEAK=lst(weak),
             NWEAK=lst([pc(x) for x in hi[hi.dataset.isin(weak) & (hi.p < 0.05) & (hi.ratio > 1)].f]) + ' nhãn',
             RWEAK0=rng(pm[pm.dataset.isin(weak) & (pm.f >= 0.5)].ratio, 2),
             TAB=tab, FIG=fig('fig_e29_hist', 'MAE theo cell trên test theo mức nhãn: PINN-semi và PINN-lai tinh chỉnh '
                              'không có (nét đứt) và có lịch sử (nét liền), cạnh GBDT có hoặc không lịch sử chọn trên '
                              'validation. Trên MIT không dạng lịch sử nào được chọn.', 'fig:e29'))


def e25_short(d, H):
    """Bản gọn của kết quả E25 (lịch đo khi dùng, mô hình học với nhãn theo cell) — tiểu mục trong mục ngân sách đo."""
    f, T, rng, lst = (H[k] for k in ('f', 'T', 'rng', 'lst'))
    S = d['e25_estimation']
    DSX = ['XJTU', 'TJU', 'MIT', 'HUST', 'NASA']
    rq = lambda sc: [x for x in S[S.sched == sc].ratio]
    worse_dau = sorted({r.ds for r in S[S.sched == 'dau'].itertuples() if r.p < 0.05 and r.kalman > r.goc})
    nk = S[S.sched == 'q10']
    gan_better = sorted({r.ds for r in nk.itertuples() if r.p_vs_gan < 0.05 and r.gan_nhat < r.kalman})
    s25 = S[S.sched == 'q25']
    kal_better25 = sorted({r.ds for r in s25.itertuples() if r.p_vs_gan < 0.05 and r.kalman < r.gan_nhat})
    return T(r"""
\subsection{Đo cell mới: một lần lúc nhận hay đo định kỳ}\label{sec:kalmanres}

Với mô hình học từ nhãn theo cell (10--70\,\% cell có nhãn, năm bộ kể cả NASA), đo cell mới mỗi $0{,}1L$ đưa sai số
xuống còn <<R10>> lần không hiệu chỉnh, mỗi $0{,}25L$ còn <<R25>> lần. Một lần đo duy nhất lúc nhận pin gần như không giúp (tỉ số
<<RD>>) và còn làm kém đi có ý nghĩa ở <<WD>>: dung lượng của nhiều cell còn tăng trong vài chục chu kỳ đầu, nên độ lệch
đo ở chu kỳ đầu không đại diện cho về sau. Khi đo dày, dịch ước lượng theo sai số của lần đo gần nhất đã đủ tốt
(<<GAN>>); bộ lọc có lợi khi đo thưa (mỗi $0{,}25L$, tốt hơn có ý nghĩa trên <<K25>>), vì nó trả độ lệch về mức chung
khi lần đo đã xa. Khoảng 90\,\% của bộ lọc hẹp hơn nhiều so với khoảng phù hợp cố định nhưng độ phủ thực tế chỉ <<COV>>.
""", R10=rng(rq('q10'), 2), R25=rng(rq('q25'), 2), RD=rng(rq('dau'), 2),
             WD=lst([x for x in DSX if x in worse_dau]) or 'không bộ nào',
             GAN=(('hơn bộ lọc có ý nghĩa ở ' + lst([x for x in DSX if x in gan_better])) if gan_better
                  else 'không kém bộ lọc ở bộ nào'),
             K25=lst([x for x in DSX if x in kal_better25]) or 'không bộ nào',
             COV=rng([100 * x for x in nk.cov_kalman], 0) + r'\,\%')


def e25_text(d, H):
    f, m3, table, T, rng, lst, pc = (H[k] for k in ('f', 'm3', 'table', 'T', 'rng', 'lst', 'pc'))
    S, F = d['e25_estimation'], d['e25_forecast']
    DSX = ['XJTU', 'TJU', 'MIT', 'HUST', 'NASA']
    g = lambda ds, fr, sc: S[(S.ds == ds) & (S.frac == fr) & (S.sched == sc)].iloc[0]
    star = lambda r, col='kalman': (r'$^{*}$' if r.p < 0.05 and r[col] < r.goc else
                                    (r'$^{\dagger}$' if r.p < 0.05 and r[col] > r.goc else ''))
    rows = []
    for ds in DSX:
        a, b, c = g(ds, 0.3, 'dau'), g(ds, 0.3, 'q25'), g(ds, 0.3, 'q10')
        rows.append([ds, m3(a.goc, 1), m3(a.kalman, 1) + star(a), m3(b.kalman, 1) + star(b), m3(c.kalman, 1) + star(c),
                     m3(c.gan_nhat, 1), f(c.n_meas, 1)])
    tab = table(['Bộ', 'Không đo', 'Đo lúc nhận', 'Mỗi $0{,}25L$', 'Mỗi $0{,}1L$', 'Mỗi $0{,}1L$, lần gần nhất',
                 'Số lần đo ($0{,}1L$)'], rows, 'lrrrrrr',
                r'MAE theo cell ($\times10^{-3}$) của PINN-lai ở 30\,\% nhãn sau khi hiệu chỉnh bằng bộ lọc Kalman theo ba '
                r'lịch đo dung lượng, tính trên các chu kỳ không được đo; cột áp chót dịch ước lượng theo sai số của lần đo '
                r'gần nhất. $^{*}$/$^{\dagger}$: tốt hơn/kém hơn không hiệu chỉnh có ý nghĩa (Wilcoxon ghép cặp theo cell, '
                r'$p < 0{,}05$).', 'tab:e25')
    rq = lambda sc: [x for x in S[S.sched == sc].ratio]
    worse_dau = sorted({r.ds for r in S[S.sched == 'dau'].itertuples() if r.p < 0.05 and r.kalman > r.goc})
    better_q25 = sorted({r.ds for r in S[S.sched == 'q25'].itertuples() if r.p < 0.05 and r.kalman < r.goc})
    nk = S[S.sched == 'q10']
    gan_better = sorted({r.ds for r in nk.itertuples() if r.p_vs_gan < 0.05 and r.gan_nhat < r.kalman})
    s25 = S[S.sched == 'q25']; kal_better25 = sorted({r.ds for r in s25.itertuples() if r.p_vs_gan < 0.05
                                                       and r.kalman < r.gan_nhat})
    fq = lambda ds, sc: F[(F.ds == ds) & (F.sched == sc)].iloc[0]
    return T(r"""
\section{Cập nhật khi có vài lần đo dung lượng}\label{sec:kalmanres}

Bảng~\ref{tab:e25} cho sai số của PINN-lai sau khi hiệu chỉnh theo Mục~\ref{sec:kalman_m}. Đo định kỳ mỗi $0{,}1L$
(khoảng <<N10>> lần trong đời cell) đưa sai số xuống còn <<R10>> lần sai số không hiệu chỉnh, tuỳ bộ và mức nhãn<<S10>>;
đo mỗi $0{,}25L$ còn <<R25>> lần, có ý nghĩa <<B25>>. Lợi ích lớn nhất ở <<TOP>>, những bộ mà sai số chủ yếu là độ lệch
mức của cả quỹ đạo, đúng như phân tích sai số ở Mục~\ref{sec:improve}.
Ngược lại, một lần đo duy nhất lúc nhận pin không giúp (tỉ số <<RD>>) và còn làm kém đi có ý nghĩa ở <<WD>>: dung lượng
của nhiều cell còn tăng trong vài chục chu kỳ đầu (giai đoạn kích hoạt), nên độ lệch đo ở chu kỳ đầu không đại diện cho
độ lệch về sau. Với lịch đo dày, cách đơn giản nhất là dịch ước lượng theo sai số của lần đo gần nhất, và cách này
<<GAN>>; bộ lọc có lợi khi đo thưa, ví dụ mỗi $0{,}25L$ nó tốt hơn có ý nghĩa trên <<K25>>, vì nó biết tin vào lần đo
bao nhiêu và trả độ lệch về mức chung khi lần đo đã xa.

<<TAB>>

Khoảng 90\,\% của bộ lọc hẹp hơn nhiều so với khoảng phù hợp cố định của Mục~\ref{sec:improve} (với đo mỗi $0{,}1L$ ở
30\,\% nhãn, độ rộng trung bình <<W1>> so với <<W2>>, đơn vị $10^{-3}$), nhưng độ phủ thực tế chỉ <<COV>>: phần trôi của
độ lệch giữa hai lần đo không hoàn toàn theo một quá trình Ornstein--Uhlenbeck chung cho mọi cell. Với dự báo, cộng độ
lệch hậu nghiệm tại gốc vào đường dự báo giảm sai số trên toàn bộ phần đời còn lại còn <<F10>> lần khi đo mỗi $0{,}1L$,
trong khi có đủ lịch sử đo cho <<FDO>> lần: một vài lần đo sửa được mức, nhưng hình dạng của quỹ đạo tương lai vẫn cần
lịch sử dung lượng đầy đủ hơn.
""", N10=rng([x for x in nk.n_meas], 0), R10=rng(rq('q10'), 2), R25=rng(rq('q25'), 2),
             B25=('trên ' + lst([x for x in DSX if x in better_q25])) if better_q25 else 'trên không bộ nào',
             S10=(', có ý nghĩa thống kê ở mọi trường hợp' if (nk.p < 0.05).all() and (nk.kalman < nk.goc).all() else
                  f', có ý nghĩa ở {int(((nk.p < 0.05) & (nk.kalman < nk.goc)).sum())}/{len(nk)} trường hợp'),
             TOP=lst(list(nk.groupby('ds').ratio.mean().sort_values().index[:3])),
             RD=rng(rq('dau'), 2), WD=lst([x for x in DSX if x in worse_dau]) or 'không bộ nào',
             GAN=(('tốt ngang hoặc hơn bộ lọc, hơn có ý nghĩa ở ' + lst([x for x in DSX if x in gan_better]) + ' (đo mỗi '
                   '$0{,}1L$)') if gan_better else 'không tốt hơn bộ lọc ở bộ nào'),
             K25=lst([x for x in DSX if x in kal_better25]) or 'không bộ nào', TAB=tab,
             W1=rng([1e3 * x for x in nk[nk.frac == 0.3].wid_kalman], 0),
             W2=rng([1e3 * x for x in nk[nk.frac == 0.3].wid_conf], 0),
             COV=rng([100 * x for x in nk.cov_kalman], 0) + r'\,\%',
             F10=rng([fq(ds, 'q10').ratio for ds in DSX], 2), FDO=rng([fq(ds, 'lich_su_do').ratio for ds in DSX], 2))
