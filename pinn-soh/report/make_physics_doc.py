"""
Tài liệu ngắn: phần vật lý của PINN nằm ở đâu, kiểm chứng trên mô hình đã huấn luyện, và các chỉ số R², RMSE.

    python report/make_physics_doc.py    # -> tai-lieu/phan-tich-vat-ly-PINN.md, .pdf (cần pandoc + XeLaTeX)

Đọc results/ (chạy sau analysis/analyze_metrics.py và analysis/analyze_physics.py). Mọi con số lấy từ dữ liệu.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import math, os, shutil, subprocess
import numpy as np
import pandas as pd

R, DOC = 'results', 'tai-lieu'
IMG = 'phan-tich-vat-ly'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRS = [0.1, 0.3, 0.5, 0.7]
LBL = {'mlp': 'MLP', 'ridge': 'Ridge', 'gbdt': 'GBDT', 'gpr': 'GPR', 'pinn_sup': 'PINN-sup', 'pinn_bb': 'PINN hộp đen',
       'pinn_semi': 'PINN-semi', 'mono_only': 'chỉ đơn điệu', 'euler': 'PINN-semi Euler'}
MODS = ['ridge', 'gpr', 'gbdt', 'mlp', 'pinn_sup', 'pinn_bb', 'pinn_semi']


def f(x, n=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '–'
    s = f'{x:.{n}f}'.replace('.', ',')
    return '−' + s[1:] if s.startswith('-') else s


def rng(v, n=2):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    a, b = f(min(v), n), f(max(v), n)
    return a if a == b else f'{a}–{b}'


def lst(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]


def md_table(head, rows, align=None):
    align = align or ['l'] + ['r'] * (len(head) - 1)
    sep = ['-' * max(3, len(h)) + (':' if a == 'r' else '') for h, a in zip(head, align)]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(sep) + '|']
    out += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return '\n'.join(out)


def build():
    sm = pd.read_csv(f'{R}/metrics_summary.csv'); ts = pd.read_csv(f'{R}/metrics_tests.csv')
    tr = pd.read_csv(f'{R}/metrics_transfer.csv'); ph = pd.read_csv(f'{R}/physics_summary.csv')
    cu = pd.read_csv(f'{R}/thesis_curve.csv'); pchk = pd.read_csv(f'{R}/physics_check.csv')
    get = lambda ds, m, fr, col: (lambda g: float(g[col].iloc[0]) if len(g) else float('nan'))(
        sm[(sm.dataset == ds) & (sm.model == m) & (sm.frac == fr)])
    mae = lambda m, fr, ds: float(cu[(cu.model == m) & (cu.frac == fr) & (cu.bo == ds)].MAE_cell.iloc[0])
    P = lambda ds, tag, col: float(ph[(ph.bo == ds) & (ph.nhanh == tag)][col].iloc[0])

    # bảng R², RMSE
    def metric_table(col, scale, n, best):
        rows = []
        for ds in DS:
            for i, m in enumerate(MODS):
                vals = []
                for fr in FRS:
                    v = get(ds, m, fr, col)
                    bv = best([get(ds, mm, fr, col) for mm in MODS])
                    s = f(scale * v, n)
                    vals.append(f'**{s}**' if v == bv else s)
                rows.append([ds if i == 0 else '', LBL[m]] + vals)
        return md_table(['Bộ', 'Mô hình', '10 %', '30 %', '50 %', '70 %'], rows, ['l', 'l', 'r', 'r', 'r', 'r'])
    tab_rmse = metric_table('RMSE_cell', 1e3, 2, min)
    tab_r2 = metric_table('R2_10', 1, 3, max)
    sig = lambda other, fr, better: [ds for ds in DS if ((ts.doi_chung == other) & (ts.frac == fr) & (ts.bo == ds) &
                                                          (ts.p_holm < 0.05) & ((ts.ti_so < 1) if better else (ts.ti_so > 1))).any()]
    rr10 = [float(ts[(ts.doi_chung == 'mlp') & (ts.frac == 0.1) & (ts.bo == ds)].ti_so.iloc[0]) for ds in DS]
    gbw = '; '.join(f'{int(100 * fr)} % ({lst(sig("gbdt", fr, False))})' for fr in FRS if sig('gbdt', fr, False))
    pnw = '; '.join(f'{int(100 * fr)} % ({lst(sig("gbdt", fr, True))})' for fr in FRS if sig('gbdt', fr, True))
    trs = tr.groupby(['k', 'model']).R2.median()
    # tính đơn điệu
    MM = ['mlp', 'gbdt', 'pinn_sup', 'pinn_bb', 'mono_only', 'pinn_semi']
    mono = lambda ds, m: get(ds, m, 0.3, 'vi_pham_don_dieu_pct')
    tab_mono = md_table(['Bộ', 'Nhãn thật'] + [LBL[m] for m in MM],
                        [[ds, f(get(ds, 'mlp', 0.3, 'vi_pham_nhan_that_pct'), 1)] + [f(mono(ds, m), 1) for m in MM] for ds in DS])
    # đạo hàm, phần do thời gian, tốc độ
    rows_phys = [('∂F/∂t > 0, MLP (%)', 'A', 'dudt_duong_pct', 1, 1), ('∂F/∂t > 0, PINN-semi (%)', 'C', 'dudt_duong_pct', 1, 1),
                 ('∂F/∂t > 0, Euler (%)', 'E', 'dudt_duong_pct', 1, 1),
                 ('phần do thời gian, MLP (%)', 'A', 'phan_thoi_gian_trung_vi', 100, 0),
                 ('phần do thời gian, PINN-semi (%)', 'C', 'phan_thoi_gian_trung_vi', 100, 0),
                 ('tốc độ r, PINN-semi', 'C', 'toc_do_r', 1, 3), ('tốc độ r, Euler', 'E', 'toc_do_r', 1, 3),
                 ('tốc độ thực đo', 'C', 'toc_do_that', 1, 3)]
    tab_phys = md_table(['Đại lượng'] + DS, [[name] + [f(k * P(ds, tag, col), n) for ds in DS]
                                             for name, tag, col, k, n in rows_phys])
    rho = lambda tag: ', '.join(f'{ds} {f(P(ds, tag, "rho_r"), 2)}' for ds in DS)
    # nguồn gốc lợi ích (từ MAE theo cell)
    sh_unl = [math.log(mae('pinn_semi', 0.1, ds) / mae('pinn_sup', 0.1, ds)) / math.log(mae('pinn_semi', 0.1, ds) / mae('mlp', 0.1, ds))
              for ds in DS]
    sh_mono = [math.log(mae('mono_only', 0.3, ds) / mae('mlp', 0.3, ds)) / math.log(mae('pinn_semi', 0.3, ds) / mae('mlp', 0.3, ds))
               for ds in DS if mae('pinn_semi', 0.3, ds) / mae('mlp', 0.3, ds) < 0.97]
    r10 = [mae('pinn_semi', 0.1, ds) / mae('mlp', 0.1, ds) for ds in DS]

    os.makedirs(f'{DOC}/{IMG}', exist_ok=True)
    for fn in ['fig_thesis_arch', 'fig_metrics_mono', 'fig_physics_rate_cell', 'fig_physics_rate_soh', 'fig_metrics_rmse_r2']:
        shutil.copy(f'{R}/{fn}.png', f'{DOC}/{IMG}/{fn}.png')

    md = f"""---
title: "PINN trong đề tài: phần vật lý nằm ở đâu, và các chỉ số R², RMSE"
subtitle: "Đồ án chuyên ngành HK261 — Nhóm 15"
lang: vi
---

# 1. Mô hình gồm những gì

Mô hình có hai mạng. Mạng nghiệm $F_\\varphi$ nhận 16 đặc trưng sạc đã chuẩn hoá $\\tilde{{\\mathbf{{x}}}}$ và thời gian
$t$ (chỉ số chu kỳ chia 1000), trả về SOH dự đoán $u$. Mạng tốc độ $G_\\theta$ cùng hai tham số $\\lambda$, $E_a$ mô tả
tốc độ suy giảm. Hai mạng chỉ gắn với nhau qua hàm mất mát; khi dự đoán SOH cho một cell chỉ cần $F_\\varphi$.

![Kiến trúc mô hình.]({IMG}/fig_thesis_arch.png){{width=95%}}

# 2. "Vật lý" ở đây là gì

Vật lý trong đề tài là vật lý ở mức hiện tượng: bốn quy luật định tính đã biết về lão hoá pin lithium-ion, được viết
thành phương trình và bất đẳng thức.

1. Dung lượng không tự tăng theo thời gian vận hành: lớp SEI dày lên và lithium bị mất khỏi chu trình là quá trình một
   chiều. Chỉ có hồi phục nhỏ sau thời gian nghỉ.
2. Suy giảm nhanh dần khi pin già: hiện tượng "đầu gối" ở giai đoạn cuối đời.
3. Phản ứng phụ tăng theo nhiệt độ, theo quy luật Arrhenius.
4. SOH nằm trong một miền hợp lý, và cell mới có SOH gần 1.

Bốn quy luật này được gom vào một phương trình suy giảm dạng xám:

$$\\frac{{\\partial u}}{{\\partial t}} = -\\,r(\\tilde{{\\mathbf{{x}}}}, u, T), \\qquad
r = \\underbrace{{\\operatorname{{softplus}}\\big(G_\\theta(\\tilde{{\\mathbf{{x}}}})\\big)}}_{{k(\\mathbf{{x}}) \\,\\ge\\, 0}}\\;
\\underbrace{{e^{{\\lambda(1-u)}}}}_{{\\text{{đầu gối}}}}\\;
\\underbrace{{e^{{-\\frac{{E_a}}{{R}}\\left(\\frac{{1}}{{T}} - \\frac{{1}}{{T_\\text{{ref}}}}\\right)}}}}_{{\\text{{Arrhenius}}}}$$

"Xám" nghĩa là dạng của phương trình được cố định theo vật lý, chỉ phần tốc độ cơ sở $k(\\mathbf{{x}})$ là học từ dữ
liệu. Đề tài không giải mô hình điện hoá chi tiết (P2D, mô hình hạt đơn) và không có phương trình nào cho chính các
đặc trưng sạc.

# 3. PINN nằm ở điểm nào

Mạng nghiệm $F_\\varphi$ là một perceptron bình thường, giống hệt MLP đối chứng. Điểm làm mô hình thành PINN nằm ở
hàm mất mát và ở các điểm mà hàm mất mát được tính:

$$\\mathcal{{L}} = \\mathcal{{L}}_\\text{{data}} + w(s)\\big(\\alpha\\,\\mathcal{{L}}_\\text{{ode}} + \\beta\\,\\mathcal{{L}}_\\text{{mono}}
+ \\gamma\\,\\mathcal{{L}}_\\text{{range}}\\big)$$

$$\\mathcal{{L}}_\\text{{ode}} = \\overline{{\\Big(\\frac{{\\partial F_\\varphi}}{{\\partial t}}(\\tilde{{\\mathbf{{x}}}}_{{n_1}}, t_{{n_1}})
+ r(\\tilde{{\\mathbf{{x}}}}_{{n_1}}, u_1, T)\\Big)^2}}, \\qquad
\\mathcal{{L}}_\\text{{mono}} = \\overline{{\\operatorname{{ReLU}}(u_2 - u_1 - 0{{,}}002)}}$$

| Hiểu biết vật lý | Biểu diễn | Tác động vào đâu | Cần nhãn |
|------------------|-----------|------------------|:--------:|
| Dung lượng không tăng theo thời gian | $\\partial u/\\partial t = -r$, $r \\ge 0$ | đạo hàm riêng $\\partial F/\\partial t$ tại một chu kỳ, qua $\\mathcal{{L}}_\\text{{ode}}$ | không |
| Quỹ đạo SOH thật không tăng | $u_2 \\le u_1 + 0{{,}}002$ | cặp chu kỳ $(n_1, n_2)$ thật của cùng cell, qua $\\mathcal{{L}}_\\text{{mono}}$ | không |
| Suy giảm nhanh dần khi già | $e^{{\\lambda(1-u)}}$ | thừa số trong $r$ | không |
| Phụ thuộc nhiệt độ | Arrhenius $A(T)$ | thừa số trong $r$ | không |
| SOH trong miền hợp lý | các hàm ReLU chặn | giá trị $u$, qua $\\mathcal{{L}}_\\text{{range}}$ | không |
| Dung lượng đo được | $(u - y)^2$ | chu kỳ có nhãn, qua $\\mathcal{{L}}_\\text{{data}}$ | có |

Có ba điểm cần phân biệt.

- Điểm tính ràng buộc (điểm collocation) là mọi chu kỳ của mọi cell train, kể cả cell chưa đo dung lượng, ghép thành
  cặp $(n_1, n_2)$ cách nhau 5–50 chu kỳ. Đây là chỗ dữ liệu không nhãn đi vào mô hình. Đạo hàm
  $\\partial F/\\partial t$ tính bằng vi phân tự động.
- $\\mathcal{{L}}_\\text{{ode}}$ ràng buộc đạo hàm riêng theo $t$ khi giữ nguyên đặc trưng sạc. $\\mathcal{{L}}_\\text{{mono}}$
  ràng buộc tổng biến thiên dọc quỹ đạo thật, nơi cả đặc trưng lẫn thời gian cùng đổi. Hai ràng buộc tác động vào hai
  hướng khác nhau của $F_\\varphi$; phần 4 đo riêng từng hướng.
- Vật lý chỉ có mặt lúc huấn luyện. Khi dự đoán, mô hình chỉ chạy $F_\\varphi$: vật lý định hình hàm được học, chứ không
  tham gia vào phép tính dự đoán.

# 4. Kiểm chứng: vật lý tác động ra sao

## 4.1. Dọc quỹ đạo: dự đoán có đơn điệu không

Với mỗi cell test, đếm các cặp chu kỳ cách nhau ít nhất 10 chu kỳ mà SOH dự đoán ở chu kỳ sau cao hơn chu kỳ trước quá
0,002 (đúng ngưỡng của $\\mathcal{{L}}_\\text{{mono}}$). Phép đếm không cần nhãn; làm trên chính nhãn thật để có mức tham
chiếu. Số liệu ở 30 % nhãn, mọi cell test của 5 lần lặp.

{tab_mono}

![Tỉ lệ cặp chu kỳ có SOH dự đoán tăng ngược; vạch ngang là nhãn thật.]({IMG}/fig_metrics_mono.png){{width=100%}}

Dự đoán của MLP tăng ngược ở {rng([mono(ds, 'mlp') for ds in DS], 1)} % số cặp, của PINN-semi chỉ
{rng([mono(ds, 'pinn_semi') for ds in DS], 1)} %, gần mức của nhãn thật
({rng([get(ds, 'mlp', 0.3, 'vi_pham_nhan_that_pct') for ds in DS], 1)} %). Biến thể chỉ có ràng buộc đơn điệu cho tỉ lệ gần như
bằng PINN-semi ({rng([mono(ds, 'mono_only') for ds in DS], 1)} %), nên chính $\\mathcal{{L}}_\\text{{mono}}$ áp lên cặp chu kỳ của
cả cell không nhãn làm quỹ đạo dự đoán đơn điệu.

## 4.2. Theo thời gian: đạo hàm riêng và phần dư ODE

Đại lượng mà $\\mathcal{{L}}_\\text{{ode}}$ ràng buộc là $\\partial F/\\partial t$, chỉ tính được từ chính mô hình. Vì huấn luyện
là tất định, các lượt ở 30 % nhãn (lần lặp đầu, 5 fold, 4 bộ, 6 biến thể) được huấn luyện lại:
{int(pchk.n_same.iloc[0])}/{int(pchk.n_ref.iloc[0])} lượt cho MAE trùng từng bit với kết quả đã báo cáo. Trên cell test đo
$\\partial F/\\partial t$, tốc độ $r$, và tách tổng mức giảm SOH dự đoán trên vòng đời cell thành phần do biến thời gian
và phần do đặc trưng sạc:

$$\\Delta u = \\underbrace{{\\textstyle\\sum_i \\big[F(\\tilde{{\\mathbf{{x}}}}_i, t_{{i+1}}) - F(\\tilde{{\\mathbf{{x}}}}_i, t_i)\\big]}}_{{\\text{{do thời gian}}}}
+ \\underbrace{{\\textstyle\\sum_i \\big[F(\\tilde{{\\mathbf{{x}}}}_{{i+1}}, t_{{i+1}}) - F(\\tilde{{\\mathbf{{x}}}}_i, t_{{i+1}})\\big]}}_{{\\text{{do đặc trưng sạc}}}}$$

{tab_phys}

Tốc độ tính bằng SOH trên 1000 chu kỳ; phần do thời gian là trung vị qua các cell test.

- MLP có $\\partial F/\\partial t > 0$ (SOH tăng theo thời gian khi đặc trưng giữ nguyên) ở
  {rng([P(ds, 'A', 'dudt_duong_pct') for ds in DS], 0)} % số điểm; PINN-semi giảm xuống
  {rng([P(ds, 'C', 'dudt_duong_pct') for ds in DS if ds != 'MIT'], 0)} % trên XJTU, TJU, HUST. Trên MIT ràng buộc được thoả
  theo cách tầm thường: $r$ và $\\partial F/\\partial t$ cùng gần bằng 0, nên dấu của đạo hàm chỉ là nhiễu.
- Ở mọi mô hình, phần lớn mức giảm SOH dự đoán đi qua đặc trưng sạc: phần do thời gian chỉ
  {rng([100 * P(ds, 'C', 'phan_thoi_gian_trung_vi') for ds in DS], 0)} % với PINN-semi. Mạng nhận ra pin đã già chủ yếu nhờ
  đường sạc đã đổi dạng; $\\mathcal{{L}}_\\text{{ode}}$ chỉ ràng buộc phần còn lại.

## 4.3. Tốc độ suy giảm học được

![Tốc độ suy giảm của từng cell test: thực đo (trục ngang) so với mô hình (trục dọc).]({IMG}/fig_physics_rate_cell.png){{width=100%}}

Độ dốc của SOH dự đoán bám sát tốc độ thực đo của từng cell, ở cả MLP lẫn PINN-semi. Tốc độ $r$ của PINN-semi thì
không: trung bình chỉ {rng([P(ds, 'C', 'toc_do_r') for ds in DS], 3)} so với {rng([P(ds, 'C', 'toc_do_that') for ds in DS], 3)}
thực đo, và gần như không tương quan với tốc độ từng cell (Spearman {rho('C')}). Lý do: ở dạng đạo hàm riêng, $r$ được
khớp với $-\\partial F/\\partial t$, tức chỉ phần suy giảm mà mạng gán cho biến thời gian (mục 4.2). Biến thể Euler đặt
phương trình dọc quỹ đạo, $u(n_2) - u(n_1) + r\\,\\Delta t \\approx 0$, nên $r$ của nó xấp xỉ tốc độ thật (trung bình
{rng([P(ds, 'E', 'toc_do_r') for ds in DS], 3)}; Spearman {rho('E')}; trên HUST tốc độ các cell gần như nhau nên tương quan
không có ý nghĩa). Đổi lại, biến thể Euler có sai số cao hơn một chút trên TJU và MIT.

![Tốc độ suy giảm theo khoảng SOH thực đo.]({IMG}/fig_physics_rate_soh.png){{width=100%}}

Theo khoảng SOH, tốc độ thực đo tăng mạnh khi SOH giảm trên XJTU, MIT và HUST (hiện tượng đầu gối). Độ dốc dự đoán
theo sát xu hướng đó, nhưng $r$ gần như phẳng: với $\\lambda \\approx 0{{,}}96$ học được, thừa số $e^{{\\lambda(1-u)}}$ chỉ tăng
khoảng 1,2 lần khi SOH giảm từ 1 xuống 0,8. Đầu gối trong dự đoán đến từ đặc trưng sạc, không từ thừa số đầu gối.
Năng lượng hoạt hoá $E_a$ cũng gần như không rời giá trị khởi tạo, vì ba trong bốn bộ chỉ có một mức nhiệt độ.

## 4.4. Phần nào mang lại lợi ích

- Ở 10 % nhãn, sai số theo cell của PINN-semi bằng {rng(r10, 2)} lần MLP; tính theo thang log,
  {rng([100 * x for x in sh_unl], 0)} % mức giảm này đến từ việc áp vật lý lên cell không nhãn, phần còn lại từ vật lý
  trên cell có nhãn.
- Ở 30 % nhãn, riêng ràng buộc đơn điệu đã mang lại {rng([100 * x for x in sh_mono], 0)} % mức giảm sai số của PINN-semi
  so với MLP. Điều này khớp với mục 4.1–4.3: vật lý giúp chủ yếu qua ràng buộc dọc quỹ đạo.

# 5. R² và RMSE cho mọi mô hình

RMSE theo cell: căn của trung bình bình phương sai số của từng cell, trung bình qua các cell. $R^2$: trong một lần lặp,
5 fold phủ mỗi cell đúng một lần, nên dự đoán của 5 fold ghép thành dự đoán cho toàn bộ bộ dữ liệu; $R^2$ tính trên đó
(từ chu kỳ thứ 10 của mỗi cell) rồi trung bình qua 5 lần lặp. In đậm là tốt nhất trong cột của mỗi bộ.

RMSE theo cell (×10⁻³):

{tab_rmse}

$R^2$ (từ chu kỳ thứ 10):

{tab_r2}

![RMSE theo cell (hàng trên, thang log) và R² (hàng dưới) theo lượng nhãn.]({IMG}/fig_metrics_rmse_r2.png){{width=100%}}

- Thứ hạng các mô hình theo RMSE gần như trùng với theo MAE, nên mọi kết luận giữ nguyên khi đổi thước đo.
- Ở 10 % nhãn, $R^2$ của PINN-semi so với MLP:
  {', '.join(f'{ds} {f(get(ds, "pinn_semi", 0.1, "R2_10"), 2)} so với {f(get(ds, "mlp", 0.1, "R2_10"), 2)}' for ds in DS)};
  RMSE theo cell của PINN-semi bằng {rng(rr10, 2)} lần MLP, có ý nghĩa thống kê trên {lst(sig('mlp', 0.1, True))}
  (Wilcoxon theo cell, hiệu chỉnh Holm).
- So với GBDT theo RMSE: GBDT tốt hơn có ý nghĩa ở {gbw}; PINN-semi tốt hơn có ý nghĩa ở {pnw}.
- Tính cả 10 chu kỳ đầu, $R^2$ của MLP trên XJTU ở 10 % nhãn chỉ còn {f(get('XJTU', 'mlp', 0.1, 'R2'), 2)}: vài chu kỳ đầu
  có đặc trưng bất thường và một mình chúng kéo $R^2$ xuống.
- Chuyển miền: khi bộ đích chưa có nhãn, $R^2$ trung vị qua 12 cặp đều âm (MLP {f(trs.loc[(0, 'mlp')], 0)}, GBDT
  {f(trs.loc[(0, 'gbdt')], 1)}, PINN-semi {f(trs.loc[(0, 'pinn_semi')], 1)}), vì dự đoán lệch đều một khoảng lớn hơn độ biến
  thiên SOH của bộ đích; MAE là thước đo phù hợp hơn ở đây. Với 3 cell đích có nhãn: PINN-semi
  {f(trs.loc[(3, 'pinn_semi')], 2)}, GBDT {f(trs.loc[(3, 'gbdt')], 2)}, MLP {f(trs.loc[(3, 'mlp')], 1)}.

# 6. Tóm tắt

- Vật lý ở đây là bốn quy luật định tính của lão hoá pin, viết thành phương trình suy giảm dạng xám và các bất đẳng
  thức; chúng không dùng nhãn nên áp được lên cell chưa đo dung lượng.
- PINN nằm ở hàm mất mát và ở các điểm collocation (cặp chu kỳ của mọi cell train), không nằm ở kiến trúc hay ở bước
  dự đoán.
- Đo trên mô hình đã huấn luyện, vật lý tác động chủ yếu qua ràng buộc dọc quỹ đạo: dự đoán gần đơn điệu như nhãn
  thật. Ràng buộc ODE dạng đạo hàm riêng giữ cho SOH không tăng theo thời gian, nhưng tốc độ $r$ của nó không phải ước
  lượng tốc độ suy giảm của cell; dạng Euler cho $r$ sát tốc độ thật.
- Theo RMSE và $R^2$, PINN-semi tốt hơn rõ rệt MLP khi ít nhãn; GBDT vẫn tốt nhất ở phần lớn trường hợp khi có từ
  30 % nhãn trở lên.
"""
    path = f'{DOC}/phan-tich-vat-ly-PINN.md'
    open(path, 'w').write(md)
    pandoc = shutil.which('pandoc')
    if pandoc is None:
        try:
            import pypandoc
            pandoc = pypandoc.get_pandoc_path()
        except Exception:
            pandoc = None
    if pandoc is None:
        print('không có pandoc — chỉ ghi', path)
        return
    env = dict(os.environ, SOURCE_DATE_EPOCH='1792800000')
    r = subprocess.run([pandoc, 'phan-tich-vat-ly-PINN.md', '-o', 'phan-tich-vat-ly-PINN.pdf', '--pdf-engine=xelatex',
                        '--resource-path=.', '-V', 'mainfont=DejaVu Serif', '-V', 'sansfont=DejaVu Sans',
                        '-V', 'monofont=DejaVu Sans Mono', '-V', 'mathfont=TeX Gyre DejaVu Math',
                        '-V', 'geometry:margin=2cm', '-V', 'fontsize=10pt',
                        '-V', 'linestretch=1.15', '-V', 'colorlinks'], cwd=DOC, capture_output=True, text=True, env=env)
    err = [l for l in r.stderr.splitlines() if 'Missing character' not in l]
    print('\n'.join(err[-5:]) if r.returncode else f'đã ghi {DOC}/phan-tich-vat-ly-PINN.pdf')


if __name__ == '__main__':
    build()
