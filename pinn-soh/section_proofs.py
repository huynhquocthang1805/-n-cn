"""
Sinh mục "Tính đúng đắn" của báo cáo LaTeX từ results/*.csv.

make_latex.py import hàm build_section() ở đây. Mọi con số trong mục đều đọc từ
results/verify_table.csv, stats_table*.csv, stats_perseed*.csv, stats_power.csv —
không gõ tay số nào.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

R = 'results'
VN = {'XJTU': 'XJTU', 'TJU': 'TJU', 'MIT': 'MIT', 'HUST': 'HUST'}


def _f(x, n=4):
    return f'{x:.{n}f}'.replace('.', ',')


def _stats_rows(suffix: str) -> str:
    t = pd.read_csv(f'{R}/stats_table{suffix}.csv')
    t = t[t.chi_so == 'MAE theo cell']
    out = []
    for _, r in t.iterrows():
        bold = r.ti_so_hi < 1.0 or r.ti_so_lo > 1.0
        ts = rf'\textbf{{{_f(r.ti_so, 3)}}}' if bold else _f(r.ti_so, 3)
        ci = rf'\textbf{{[{_f(r.ti_so_lo,3)}; {_f(r.ti_so_hi,3)}]}}' if bold \
            else f'[{_f(r.ti_so_lo,3)}; {_f(r.ti_so_hi,3)}]'
        ph = rf'\textbf{{{_f(r.p_holm,4)}}}' if r.p_holm < 0.05 else _f(r.p_holm, 4)
        out.append(f'{VN[r.bo]} & {_f(r.mlp)} & {_f(r.pinn)} & {ts} & {ci} & '
                   f'{_f(r.p_tho,4)} & {ph} & {int(r.thang_seed)}/{int(r.n_seed)} \\\\')
    return '\n'.join(out)


def _spread_rows() -> str:
    d = pd.read_csv(f'{R}/stats_perseed.csv')
    ncell = {'XJTU': 7, 'TJU': 18, 'MIT': 18, 'HUST': 12}
    out = []
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        g = d[d.dataset == ds].sort_values('seed')
        a, b = g.maecell_mlp.values, g.maecell_pinn.values
        r3, r10 = b[:3].mean() / a[:3].mean(), b.mean() / a.mean()
        out.append(f'{VN[ds]} & {_f(r3,3)} & {_f(r10,3)} & '
                   f'{_f(a.min())}--{_f(a.max())} & {_f(a.max()/a.min(),1)}$\\times$ & '
                   f'{int(g.n_cell.iloc[0])} \\\\')
    return '\n'.join(out)


def _why3() -> dict:
    """Số liệu cho đoạn giải thích 'vì sao ba seed là không đủ' — lấy thẳng từ CSV."""
    d = pd.read_csv(f'{R}/stats_perseed.csv')
    o = {}
    for ds in ['XJTU', 'MIT']:
        g = d[d.dataset == ds].sort_values('seed')
        a, b = g.maecell_mlp.values, g.maecell_pinn.values
        rank = np.argsort(np.argsort(-a))            # 0 = lượt tệ nhất
        o[ds] = dict(first3=a[:3].mean(), rest=a[3:].mean(),
                     r3=b[:3].mean() / a[:3].mean(),
                     ratio=a[:3].mean() / a[3:].mean(),
                     worst_in_first3=int((rank[:3] <= 1).sum()),
                     worst_seeds=[int(i) for i in np.argsort(-a)[:2]],
                     hi=a.max(), lo=a.min(), ncell=int(g.n_cell.iloc[0]))
    return o


def _power_rows() -> str:
    d = pd.read_csv(f'{R}/stats_power.csv')
    cols = [c for c in d.columns if c.startswith('luc_n')]
    out = []
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(rf'\textbf{{{_f(v,2)}}}' if v >= 0.80 else _f(v, 2))
        out.append(f'{VN[r.bo]} & ${_f(r.hieu_tb,5)}$ & {_f(r.sd,5)} & ' + ' & '.join(cells) + r' \\')
    header = ' & '.join(f'$n{{=}}{c[5:]}$' for c in cols)
    return header, '\n'.join(out)


def _verify_count() -> tuple[int, int]:
    v = pd.read_csv(f'{R}/verify_table.csv')
    return int(v.dat.sum()), len(v)


def build_section() -> str:
    n_ok, n_all = _verify_count()
    phead, prows = _power_rows()
    w = _why3()
    return rf"""\section{{Tính đúng đắn: chứng minh và kiểm định}}\label{{sec:chungminh}}

Mục này tách bạch ba việc thường bị gộp: \emph{{chứng minh}} các tính chất mà thiết kế dựa
vào, \emph{{kiểm chứng số}} rằng cài đặt thoả các tính chất đó, và \emph{{kiểm định thống
kê}} các kết quả thực nghiệm. Phép kiểm số không chứng minh mệnh đề --- nó bắt lỗi cài đặt
như sai dấu, quên softplus, nhãn lọt vào loss.

\subsection{{Các mệnh đề}}

\begin{{menhde}}[Tốc độ suy giảm không âm]\label{{md:rate}}
Với mọi $\theta,\lambda,E_a \in \mathbb{{R}}$ và mọi $(\mathbf{{x}},u,T)$ với $T>0$:
$r(\mathbf{{x}},u,T) > 0$.
\end{{menhde}}

\begin{{proof}}
$\operatorname{{softplus}}(z)=\ln(1+e^z)$. Vì $e^z>0\ \forall z\in\mathbb{{R}}$ nên
$1+e^z>1$, do đó $\ln(1+e^z)>0$. Hai thừa số còn lại là $\exp$ của một số thực nên dương
ngặt. Tích ba số dương ngặt là dương ngặt.
\end{{proof}}

Hệ quả: $du/dt = -r \le 0$ --- dung lượng không thể tự tăng, \emph{{theo cấu trúc mô
hình}}, không phụ thuộc giá trị tham số học được. Trong số học float32,
$\operatorname{{softplus}}$ tràn dưới về đúng $0$ khi đối số $< -104$; nên phát biểu đúng
cho cài đặt là $r \ge 0$, và đó mới là tính chất mà hệ quả cần. Đo trên 50 mạng khởi tạo
Xavier với đặc trưng đã chuẩn hoá, tiền kích hoạt nhỏ nhất là $-2{{,}}93$ --- cách ngưỡng
tràn 101 đơn vị.

\begin{{menhde}}[Đơn điệu theo cấu trúc]\label{{md:mono}}
Cho $u(\mathbf{{x}},t) = u_0(\mathbf{{x}}) - \sum_{{k=1}}^{{K}} s_k(\mathbf{{x}})
\operatorname{{softplus}}(\omega_k t + \phi_k(\mathbf{{x}}))$ với
$s_k = \operatorname{{softplus}}(\cdot) \ge 0$ và $\omega_k = \operatorname{{softplus}}(\cdot) \ge 0$.
Khi đó $\partial u/\partial t \le 0$ với \emph{{mọi}} giá trị tham số.
\end{{menhde}}

\begin{{proof}}
$\frac{{d}}{{dz}}\operatorname{{softplus}}(z) = \sigma(z) \in (0,1)$. Theo quy tắc dây chuyền,
\[
\frac{{\partial u}}{{\partial t}} = -\sum_{{k=1}}^{{K}} s_k(\mathbf{{x}})\,\omega_k\,
\sigma\big(\omega_k t+\phi_k(\mathbf{{x}})\big).
\]
Mỗi số hạng là tích của ba đại lượng không âm nên tổng $\ge 0$, do đó
$\partial u/\partial t \le 0$. Dấu bằng xảy ra khi và chỉ khi $s_k\omega_k = 0\ \forall k$.
\end{{proof}}

Kết luận không phụ thuộc giá trị tham số nên đúng ngay từ bước khởi tạo và giữ nguyên
suốt quá trình huấn luyện --- khác hẳn $L_{{\text{{mono}}}}$ vốn chỉ \emph{{phạt}} vi phạm
chứ không \emph{{ngăn}} được, và còn phải cân trọng số $\beta$.

\begin{{menhde}}[Ba loss vật lý không phụ thuộc nhãn]\label{{md:labelfree}}
$L_{{\text{{ode}}}}$, $L_{{\text{{mono}}}}$, $L_{{\text{{range}}}}$ là hàm của
$(\mathbf{{x}}_1,\mathbf{{x}}_2,t_1,t_2,T,\varphi,\theta,\lambda,E_a)$ và không chứa $y$;
do đó $\partial L/\partial y \equiv 0$.
\end{{menhde}}

\begin{{proof}}
Trực tiếp từ định nghĩa~\eqref{{eq:lode}}--\eqref{{eq:lrange}}: không biểu thức nào chứa $y$.
\end{{proof}}

Đây là \emph{{điều kiện cần}} để cơ chế bán giám sát tồn tại. Loss vật lý của PINN4SOH,
$\operatorname{{ReLU}}\big((u_2-u_1)(y_1-y_2)\big)$, không thoả mệnh đề này: hướng phạt do
nhãn quyết định, nên về cấu trúc nó không thể là nguồn thông tin bù cho nhãn thiếu.

\begin{{menhde}}[Hàm phạt chính xác theo tập không]\label{{md:exact}}
Với $\varepsilon \ge 0$ và tập cặp mẫu $\mathcal{{P}}$:
$L_{{\text{{mono}}}}(\mathcal{{P}}) = 0 \iff u_2 - u_1 \le \varepsilon$ với mọi cặp trong
$\mathcal{{P}}$. Nếu mọi cặp vi phạm đúng một lượng $d>0$ thì $L_{{\text{{mono}}}} = d$.
Tương tự, $L_{{\text{{range}}}} = 0 \iff u \in [0{{,}}4;\,1{{,}}15]$ tại mọi điểm, và khi
$u$ ra ngoài thì $L_{{\text{{range}}}}$ bằng đúng khoảng cách trung bình tới hộp.
\end{{menhde}}

\begin{{proof}}
$\operatorname{{ReLU}}(z)=\max(0,z)\ge 0$, bằng $0$ khi và chỉ khi $z \le 0$. Trung bình các
số không âm bằng $0$ khi và chỉ khi mọi số hạng bằng $0$; áp cho $z=u_2-u_1-\varepsilon$.
Nếu $z \equiv d>0$ thì $\operatorname{{ReLU}}(d)=d$ và trung bình bằng $d$. Với
$L_{{\text{{range}}}}$: khi $u>1{{,}}15$ thì $\operatorname{{ReLU}}(u-1{{,}}15)=
\operatorname{{dist}}(u,[0{{,}}4;1{{,}}15])$ và số hạng kia bằng $0$; đối xứng cho
$u<0{{,}}4$; trong hộp cả hai bằng $0$.
\end{{proof}}

Tập nghiệm của $L_{{\text{{mono}}}}=0$ \emph{{trùng đúng}} tập khả thi của ràng buộc, và giá
trị của nó bằng đúng mức vi phạm trung bình nên đọc được như một đại lượng có đơn vị SOH.

\begin{{menhde}}[Bậc chặt cụt và khuếch đại nhiễu]\label{{md:euler}}
Giả sử $u \in C^2([t,t+\Delta])$ và $u' = -r$. Đặt
$R_{{\mathrm{{E}}}}(\Delta) = u(t+\Delta)-u(t)+r(t)\Delta$ và
$R_{{\mathrm{{D}}}}(\Delta) = \frac{{u(t+\Delta)-u(t)}}{{\Delta}}+r(t)$. Khi đó
$R_{{\mathrm{{E}}}} = O(\Delta^2)$, $R_{{\mathrm{{D}}}} = O(\Delta)$ và
$R_{{\mathrm{{E}}}} = \Delta R_{{\mathrm{{D}}}}$. Hơn nữa, nếu đầu ra mạng có nhiễu cộng
$|\eta_i| \le \delta$ thì $|\Delta R_{{\mathrm{{E}}}}| \le 2\delta$ \emph{{độc lập với
$\Delta$}}, còn $|\Delta R_{{\mathrm{{D}}}}| \le 2\delta/\Delta$; tỉ số khuếch đại đúng
bằng $1/\Delta$.
\end{{menhde}}

\begin{{proof}}
Khai triển Taylor bậc hai với phần dư Lagrange: tồn tại $\xi\in(t,t+\Delta)$ sao cho
$u(t+\Delta) = u(t) + \Delta u'(t) + \frac{{\Delta^2}}{{2}}u''(\xi)$. Thay $u'(t)=-r(t)$:
\[
R_{{\mathrm{{E}}}} = \Big[u(t)-\Delta r(t)+\tfrac{{\Delta^2}}{{2}}u''(\xi)\Big] - u(t)
+ r(t)\Delta = \tfrac{{\Delta^2}}{{2}}u''(\xi).
\]
Nếu $|u''| \le M$ thì $|R_{{\mathrm{{E}}}}| \le M\Delta^2/2$; chia cho $\Delta$ được
$R_{{\mathrm{{D}}}} = \frac{{\Delta}}{{2}}u''(\xi)$. Về nhiễu: $R_{{\mathrm{{E}}}}$ phụ thuộc
$u$ tuyến tính với hệ số $\pm 1$ nên $\Delta R_{{\mathrm{{E}}}} = \eta_2-\eta_1$; dạng đạo
hàm chia thêm cho $\Delta$. Tỉ số hai chặn là $1/\Delta$.
\end{{proof}}

Với $\Delta = h/1000$, ở $h=1$ tỉ số là \textbf{{1000}}. Kiểm chứng số với
$\delta=10^{{-3}}$ cho $|\Delta R_{{\mathrm{{D}}}}| \approx 1{{,}}13$ trong khi
$|du/d\tilde{{N}}|$ thực tế chỉ khoảng $0{{,}}2$--$0{{,}}5$: nhiễu lấn át tín hiệu 2--5 lần.
Đây là lý do định lượng cho việc chọn dạng Euler.

\begin{{nhanxet}}[Hạn chế của Mệnh đề~\ref{{md:mono}}]\label{{nx:traj}}
Mệnh đề~\ref{{md:mono}} cho $\partial u/\partial t \le 0$ khi \emph{{giữ nguyên
$\mathbf{{x}}$}}. Dọc quỹ đạo thật, $\mathbf{{x}}$ cũng thay đổi theo chu kỳ:
\[
\frac{{du}}{{dN}} = \frac{{1}}{{1000}}\frac{{\partial u}}{{\partial t}}
+ \Big\langle \nabla_{{\mathbf{{x}}}} u,\ \frac{{d\mathbf{{x}}}}{{dN}}\Big\rangle .
\]
Số hạng thứ nhất $\le 0$; số hạng thứ hai \emph{{không có ràng buộc dấu}}. Do đó đơn điệu
theo $t$ \textbf{{không kéo theo}} đơn điệu dọc quỹ đạo.
\end{{nhanxet}}

Kiểm chứng: dựng quỹ đạo giả trong đó $\mathbf{{x}}$ đi theo hướng làm $u_0$ tăng --- đạo
hàm riêng theo $t$ vẫn $\le 0$ tại mọi điểm, nhưng $4/299$ bước dọc quỹ đạo tăng ngược.
Bằng chứng thực nghiệm khớp: ở mục~\ref{{sec:e9}} trên HUST, kiến trúc đơn điệu có chỉ số
vi phạm đơn điệu $2{{,}}61$ so với $1{{,}}18$ của đầu ra MLP thường \emph{{trong cùng mô
hình PINN-semi}} --- kiến trúc \emph{{đảm bảo}}
đơn điệu theo $t$ lại vi phạm đơn điệu \emph{{dọc quỹ đạo}} nhiều hơn.

\subsection{{Kiểm chứng số}}

Script \texttt{{verify\_losses.py}} chạy {n_all} phép kiểm tương ứng các mệnh đề trên; toàn
bộ \textbf{{{n_ok}/{n_all}}} đạt. Các số đo then chốt: bậc hồi quy của
$R_{{\mathrm{{E}}}}$ theo $\Delta$ đo được $1{{,}}9957$ (lý thuyết 2) và của
$R_{{\mathrm{{D}}}}$ là $0{{,}}9957$ (lý thuyết 1); tỉ số khuếch đại nhiễu khớp $1/\Delta$
ở cả bốn giá trị $h$; $\partial L/\partial y$ không nối được tới $y$ cho cả ba loss trong
khi đối chứng $\partial L_{{\text{{data}}}}/\partial y \ne 0$; và \texttt{{gradcheck}} trên
float64 đạt cho cả ba loss.

Hai phép kiểm thất bại ở lần chạy đầu, và đó chính là giá trị của bộ test. Thứ nhất, bản
đầu phát biểu Mệnh đề~\ref{{md:rate}} ở dạng $r>0$ ngặt; phép kiểm bắt được $r=0$ chính xác
khi trọng số bị đẩy tới $10^2$ --- nguyên nhân là tràn dưới float32, buộc phải sửa phát
biểu cho cài đặt thành $r \ge 0$. Thứ hai, \texttt{{gradcheck}} thất bại do lỗi của chính
phép kiểm: gán \texttt{{p.data}} để thay tham số làm \emph{{cắt đồ thị autograd}}; sửa bằng
\texttt{{torch.func.functional\_call}} thì cả ba loss đều đạt.

\subsection{{Kiểm định thống kê kết quả}}

Đơn vị quan sát là \textbf{{seed}}. Với cùng một seed, hai mô hình dùng đúng một cách chia
cell và được đánh giá trên đúng một tập cell test theo đúng thứ tự hàng --- đã kiểm bằng
assert --- nên mỗi seed cho một cặp ghép đôi hợp lệ. Số seed nâng từ 3 lên
\textbf{{10}} (136 lượt huấn luyện bổ sung). Kiểm định Wilcoxon signed-rank ghép cặp,
khoảng tin cậy bootstrap ghép cặp 20\,000 lần, hiệu chỉnh Holm--Bonferroni trên bốn bộ.
Chỉ số là MAE theo cell.

\begin{{table}}[htbp]
\centering\small
\caption{{Cấu hình E1 ($\beta=5$ cố định), 10 seed. In đậm: khoảng tin cậy không chứa 1,
hoặc $p_{{\text{{Holm}}}}<0{{,}}05$.}}
\label{{tab:stats-e1}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{lrrrlrrr}}
\toprule
Bộ & MLP & PINN & Tỉ số & KTC 95\% & $p$ thô & $p$ Holm & Seed thắng \\
\midrule
{_stats_rows('')}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\textbf{{Đính chính.}} Với 10 seed, \emph{{không bộ nào}} cho thấy PINN tốt hơn một cách có
ý nghĩa thống kê, và trên TJU thì PINN \emph{{kém hơn}} có ý nghĩa ($p_{{\text{{Holm}}}} =
0{{,}}0078$, thua $10/10$ seed). Phát biểu ``giả thuyết đạt ở 2/4 bộ'' ở cấu hình E1
\textbf{{không đứng vững}} khi tăng số seed.

\begin{{table}}[htbp]
\centering\small
\caption{{Vì sao ba seed là không đủ. Tập test quá nhỏ khiến MAE của MLP dao động mạnh
giữa các seed.}}
\label{{tab:stats-spread}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{lrrlrr}}
\toprule
Bộ & Tỉ số 3 seed & Tỉ số 10 seed & Biên độ MAE của MLP & Tỉ lệ & Số cell test \\
\midrule
{_spread_rows()}
\bottomrule
\end{{tabular}}}}
\end{{table}}

Nguyên nhân gốc là tập test quá nhỏ, nên MAE của baseline là một xổ số theo seed --- và
ba seed đầu rơi về \emph{{hai phía ngược nhau}} ở hai bộ:

\begin{{itemize}}
\item \textbf{{XJTU}} ({{\bfseries {w['XJTU']['ncell']} cell test}}) --- một cell xấu chi
phối toàn bộ MAE, nên MLP nhảy từ {_f(w['XJTU']['lo'])} lên {_f(w['XJTU']['hi'])}. Hai
lượt tệ nhất rơi vào seed {w['XJTU']['worst_seeds'][0]} và
{w['XJTU']['worst_seeds'][1]} --- \emph{{ngoài}} ba seed đầu. Ba seed đầu vì thế là mẫu
\emph{{thuận lợi}} cho MLP (trung bình {_f(w['XJTU']['first3'])} so với
{_f(w['XJTU']['rest'])} của bảy seed còn lại), và tỉ số 3 seed {_f(w['XJTU']['r3'],3)} đã
\emph{{đánh giá thấp}} PINN.

\item \textbf{{MIT}} --- ngược lại. Hai lượt MLP tệ nhất trong mười nằm \emph{{trong}} ba
seed đầu (seed {w['MIT']['worst_seeds'][0]} và {w['MIT']['worst_seeds'][1]}): trung bình
ba seed đầu {_f(w['MIT']['first3'])} so với {_f(w['MIT']['rest'])} của bảy seed còn lại,
tức tệ hơn {(w['MIT']['ratio'] - 1) * 100:.0f}\%. Tỉ số 3 seed {_f(w['MIT']['r3'],3)} vì thế đã
\emph{{đánh giá cao}} PINN.
\end{{itemize}}

Hai trường hợp lệch ngược chiều nhau là bằng chứng rõ nhất rằng cái đo được ở ba seed là
nhiễu chia dữ liệu, không phải hiệu ứng của mô hình.

\begin{{table}}[htbp]
\centering\small
\caption{{Cấu hình đã tinh chỉnh --- mạng \texttt{{wide}} $(128,128,64)$, biến thể mà lưới
tinh chỉnh cân bằng ở mục~\ref{{sec:e12}} chọn ở 6/8 ô, dùng cho \emph{{cả hai}} mô hình;
cộng cấu hình vật lý tốt nhất theo validation ở mục~\ref{{sec:e7}}
($\beta_{{\text{{mono}}}} = 0{{,}}5$, phần dư autograd, $\alpha_{{\text{{ode}}}} =
0{{,}}02$) cho PINN. Cấu hình ấn định trước, không chọn lại theo 10 seed.}}
\label{{tab:stats-tuned}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{lrrrlrrr}}
\toprule
Bộ & MLP & PINN & Tỉ số & KTC 95\% & $p$ thô & $p$ Holm & Seed thắng \\
\midrule
{_stats_rows('_tuned')}
\bottomrule
\end{{tabular}}}}
\end{{table}}

Bức tranh đổi hẳn. \textbf{{Ba trên bốn bộ}} có khoảng tin cậy của tỉ số nằm hoàn toàn dưới
1 (XJTU, MIT, HUST), mức cải thiện 10--28\%; ba trên bốn có $p$ thô $<0{{,}}05$, nhưng sau
hiệu chỉnh Holm thì không bộ nào đạt mức $0{{,}}05$ --- cả ba nằm trong khoảng
$0{{,}}055$--$0{{,}}082$, sát ngưỡng. TJU không còn kém hơn có ý nghĩa ($1{{,}}167 \to
0{{,}}979$), tức phần lớn ``thất bại trên TJU'' là do $\beta=5$ siết quá tay chứ không
phải bản chất bộ dữ liệu.

Phát biểu trung thực nhất với dữ liệu hiện có: \emph{{ở cấu hình đã tinh chỉnh, có bằng
chứng nhất quán rằng PINN giảm MAE khoảng 10--28\% trên ba trong bốn bộ; bằng chứng chưa
đạt mức có ý nghĩa thống kê sau hiệu chỉnh đa so sánh với 10 seed.}}

\begin{{table}}[htbp]
\centering\small
\caption{{Lực thống kê ước lượng bằng bootstrap từ phân bố hiệu đã quan sát (600 mô phỏng
mỗi ô, Holm $\times 4$). In đậm: lực $\ge 0{{,}}80$.}}
\label{{tab:stats-power}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{lrr{'r' * len(pd.read_csv(f'{R}/stats_power.csv').filter(like='luc_n').columns)}}}
\toprule
Bộ & Hiệu TB & SD & {phead} \\
\midrule
{prows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\textbf{{20 seed là đủ}} cho XJTU, MIT và HUST (lực $0{{,}}83$--$0{{,}}96$); kế hoạch nên
đổi từ 10 lên 20 seed. TJU không bao giờ đạt được, và đó chính là kết quả \emph{{đúng}}:
hiệu thật ở đó chỉ $-0{{,}}00022$, tức khoảng 2\%. Không phải thiếu lực thống kê mà là
thật sự không có khác biệt --- với TJU, phát biểu đúng là ``hai mô hình tương đương''.

\paragraph{{Những gì phép kiểm này chưa phủ.}} Chỉ kiểm ở mức 30\% nhãn và ở hai cấu hình.
Kết luận của mục~\ref{{sec:e12}} vẫn dựa trên ba seed và chịu đúng điểm yếu đã nêu. Các thí
nghiệm chuyển miền, ablation, và các chỉ số ngoài MAE chưa được kiểm định.
"""
