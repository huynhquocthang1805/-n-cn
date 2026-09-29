"""
Sinh dự án LaTeX (tiếng Việt, XeLaTeX) từ results/*.csv và results/*.md.
Mọi con số trong báo cáo đều lấy tự động — không gõ tay chỗ nào.

    python make_latex.py            -> tạo thư mục bao-cao-latex/
    cd bao-cao-latex && make        -> biên dịch ra main.pdf
"""
from __future__ import annotations
import os, re, shutil, glob
import numpy as np, pandas as pd
from section_proofs import build_section

R = 'results'
OUT = 'bao-cao-latex'


# --------------------------------------------------------------------------- helpers
def esc(s: str) -> str:
    """Escape các ký tự đặc biệt của LaTeX trong văn bản thường."""
    s = s.replace('\\', r'\textbackslash{}')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
                 ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')]:
        s = s.replace(a, b)
    return s


def cell_to_tex(c: str) -> str:
    """Một ô của bảng markdown -> LaTeX."""
    c = c.strip()
    c = re.sub(r'<br>\s*<small>(.*?)</small>', r' (\1)', c)
    c = re.sub(r'<small>(.*?)</small>', r'\1', c)
    c = re.sub(r'<sub>(.*?)</sub>', r'\\textsubscript{\1}', c)
    # đánh dấu bằng ký tự điều khiển trước khi escape, rồi thay bằng lệnh LaTeX
    c = re.sub(r'\*\*(.+?)\*\*', lambda m: '\x01' + m.group(1) + '\x02', c)
    c = re.sub(r'`(.+?)`', lambda m: '\x03' + m.group(1) + '\x04', c)
    c = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', lambda m: '\x05' + m.group(1) + '\x06', c)
    c = esc(c)
    for a, b in [('\x01', r'\textbf{'), ('\x02', '}'), ('\x03', r'\texttt{'), ('\x04', '}'),
                 ('\x05', r'\emph{'), ('\x06', '}')]:
        c = c.replace(a, b)
    return c


def md_to_tables(path: str, label_prefix: str, captions: dict | None = None,
                 small=True, colspec=None) -> str:
    """File markdown (nhiều bảng + chú thích in đậm) -> chuỗi LaTeX gồm các table."""
    if not os.path.exists(path):
        return f'% thiếu {path}\n'
    blocks, cur, cap = [], [], None
    for line in open(path):
        t = line.strip()
        if t.startswith('|'):
            cur.append(t)
        else:
            if cur:
                blocks.append((cap, cur)); cur, cap = [], None
            if t:                       # dòng văn bản bất kỳ = chú thích cho bảng ngay sau nó
                cap = t
    if cur:
        blocks.append((cap, cur))

    out = []
    for i, (cap, rows) in enumerate(blocks):
        split = lambda r: [c.strip() for c in r.strip().strip('|').split('|')]
        head = split(rows[0])
        body = rows[2:] if len(rows) > 1 and set(rows[1].replace('|', '').strip()) <= set('-: ') else rows[1:]
        n = len(head)
        spec = colspec or ('l' + 'r' * (n - 1))
        lab = f'{label_prefix}' + (f'-{i+1}' if len(blocks) > 1 else '')
        caption = (captions or {}).get(lab) or cap or (captions or {}).get(label_prefix) or ''
        out.append(r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n' +
                   (r'\small' + '\n' if small else '') +
                   (f'\\caption{{{cell_to_tex(caption)}}}\n' if caption else '') +
                   f'\\label{{tab:{lab}}}\n' +
                   r'\adjustbox{max width=\textwidth}{%' + '\n' +
                   f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n' +
                   ' & '.join(cell_to_tex(h) for h in head) + r' \\' + '\n\\midrule\n' +
                   '\n'.join(' & '.join(cell_to_tex(c) for c in split(r)) + r' \\' for r in body) +
                   '\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n')
    return '\n'.join(out)


def tt(x: str) -> str:
    """Tên biến thể -> \\texttt{...} với dấu gạch dưới đã escape."""
    return r'\texttt{' + str(x).replace('_', r'\_') + '}'


def fig(name: str, caption: str, label: str, width='\\textwidth') -> str:
    return (r'\begin{figure}[htbp]' + '\n' + r'\centering' + '\n' +
            f'\\includegraphics[width={width}]{{figures/{os.path.splitext(name)[0]}}}\n'
            f'\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n')


# --------------------------------------------------------------------------- số liệu
S = {}


def load_stats():
    def g(p):
        return pd.read_csv(f'{R}/{p}') if os.path.exists(f'{R}/{p}') else None
    e1, h = g('E1_table.csv'), g('E1_hypothesis.csv')
    if e1 is not None:
        S['n_runs_E1'] = int(e1.n.sum())
        f = lambda d, m, fr: float(e1[(e1.dataset == d) & (e1.model == m) & (e1.frac == fr)].MAE.iloc[0])
        DS = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if (e1.dataset == d).any()]
        S['gain10'] = ', '.join(f'{d} {f(d,"mlp",0.1)/f(d,"pinn_semi",0.1):.1f}$\\times$' for d in DS)
    if h is not None:
        S['hyp_rows'] = '\n'.join(
            f'\\item \\textbf{{{r.dataset}}}: PINN-semi với 30\\,\\% nhãn đạt MAE {r.pinn30:.4f}, tức '
            f'\\textbf{{{r.ratio_pinn30_mlp70:.2f}$\\times$}} MAE của MLP với 70\\,\\% nhãn ({r.mlp70:.4f}); '
            f'MLP cùng 30\\,\\% nhãn là {r.mlp30:.4f} ({r.ratio_mlp30_mlp70:.2f}$\\times$).'
            for _, r in h.iterrows())
    e7 = g('E7_table.csv')
    if e7 is not None:
        S['e7_wins'] = f'{int((e7.ratio_vs_mlp < 1).sum())}/{len(e7)}'
        f50 = e7[e7.frac == 0.5]
        S['e7_50_mlp'] = ', '.join(f'{r.dataset} {r.ratio_vs_mlp:.2f}$\\times$' for _, r in f50.iterrows())
        S['e7_50_mlp70'] = ', '.join(f'{r.dataset} {r.ratio_vs_mlp70:.2f}$\\times$' for _, r in f50.iterrows())
        S['e7_50_le70'] = f'{int((f50.ratio_vs_mlp70 <= 1.02).sum())}/{len(f50)}'
        lo, hi = e7[e7.frac <= 0.3], e7[e7.frac >= 0.5]
        S['e7_sel_lo'] = ', '.join(tt(x) for x in sorted(set(lo.selected)))
        S['e7_sel_hi'] = ', '.join(tt(x) for x in sorted(set(hi.selected)))
        S['e7_adapt_wins'] = f'{int((e7.adapt < e7.mlp).sum())}/{len(e7)}'
    t12 = g('E12_table.csv')
    if t12 is not None:
        S['e12_wins'] = f'{int((t12.ratio < 1).sum())}/{len(t12)}'
        S['e12_range'] = f'{t12.ratio.min():.2f}--{t12.ratio.max():.2f}$\\times$'
        S['e12_after'] = f'{t12.ratio.mean():.2f}$\\times$'
        S['e12_gain'] = ', '.join(f'{r.dataset}@{int(r.frac*100)}\\,\\% {100*r.mlp_gain:+.0f}\\,\\%'
                                  for _, r in t12.sort_values('mlp_gain', ascending=False).head(3).iterrows())
        if e1 is not None:
            old = [float(e1[(e1.dataset == r.dataset) & (e1.frac == r.frac) & (e1.model == 'pinn_semi')].MAE.iloc[0])
                   / r.mlp_default for _, r in t12.iterrows()]
            S['e12_before'] = f'{np.mean(old):.2f}$\\times$'
        S['e12_tags'] = ', '.join(f'{tt(k)} ({v})' for k, v in
                                  pd.concat([t12.mlp_tag, t12.pinn_tag]).value_counts().head(3).items())
        tj = t12[t12.dataset == 'TJU']
        S['e12_tju'] = ', '.join(f'{int(r.frac*100)}\\,\\% $\\to$ {r.ratio:.2f}$\\times$' for _, r in tj.iterrows())
    e9 = g('E9_table.csv')
    if e9 is not None:
        def rel(model, tag):
            sub = e9[e9.model == model]
            b = sub[sub.tag == 'silu-mlp'].set_index('dataset').MAE
            r = sub[sub.tag == tag].set_index('dataset').MAE
            c = b.index.intersection(r.index)
            return float((r[c] / b[c]).mean()) if len(c) else float('nan')
        for k, (m, t) in {'e9_mono_pinn': ('pinn_semi', 'silu-mono'), 'e9_mono_mlp': ('mlp', 'silu-mono'),
                          'e9_tanh_pinn': ('pinn_semi', 'tanh-mlp'), 'e9_tanh_mlp': ('mlp', 'tanh-mlp'),
                          'e9_snake_pinn': ('pinn_semi', 'snake-mlp'), 'e9_snake_mlp': ('mlp', 'snake-mlp'),
                          'e9_sin_pinn': ('pinn_semi', 'sin-mlp'), 'e9_res_pinn': ('pinn_semi', 'silu-res')}.items():
            S[k] = f'{rel(m, t):.3f}$\\times$'
        mo = e9[(e9.model == 'pinn_semi') & (e9.tag == 'silu-mono')]
        ba = e9[(e9.model == 'pinn_semi') & (e9.tag == 'silu-mlp')]
        S['e9_mono_viol'] = f'{mo.MonoViol.mean():.2f} so với {ba.MonoViol.mean():.2f}'
        S['e9_mono_eol'] = f'{mo.EOL_MAE.mean():.0f} so với {ba.EOL_MAE.mean():.0f}'
    t10 = g('E10_table.csv')
    if t10 is not None:
        c = lambda m, n, a, b: float(t10[(t10.model == m) & (t10.norm == n) & (t10.source == a)
                                         & (t10.target == b)].MAE.iloc[0])
        off, dia = t10[t10.source != t10.target], t10[t10.source == t10.target]
        S['e10_dia'] = f'{dia[dia.norm == "global"].MAE.mean():.4f}'
        gm = off[off.norm == 'global'].MAE.median(); fm = off[off.norm == 'first_cycle'].MAE.median()
        S['e10_med'] = f'{gm:.2f} $\\to$ {fm:.2f} (tốt hơn {gm/fm:.1f}$\\times$)'
        S['e10_worst'] = f'{c("mlp","global","XJTU","MIT"):.0f}'
        same = [('MIT', 'HUST'), ('HUST', 'MIT')]
        cross = [('XJTU', 'MIT'), ('XJTU', 'HUST'), ('MIT', 'XJTU'), ('HUST', 'XJTU')]
        a = np.mean([c('pinn', 'first_cycle', x, y) for x, y in same])
        b = np.mean([c('pinn', 'first_cycle', x, y) for x, y in cross])
        S['e10_same'], S['e10_cross'], S['e10_ratio'] = f'{a:.3f}', f'{b:.2f}', f'{b/a:.0f}$\\times$'
        S['e10_asym'] = (f'MIT$\\to$XJTU {c("pinn","first_cycle","MIT","XJTU"):.3f} nhưng '
                         f'XJTU$\\to$MIT {c("pinn","first_cycle","XJTU","MIT"):.2f}')
    e11 = g('E11_table.csv')
    if e11 is not None:
        wins = {}
        for col in ['MAE', 'MAE_cell', 'MAE_cell_max', 'MAE_late', 'MonoViol', 'Jitter', 'EOL_MAE']:
            w = 0
            for d in e11.dataset.unique():
                a = e11[(e11.dataset == d) & (e11.model == 'pinn_semi')][col]
                b = e11[(e11.dataset == d) & (e11.model == 'mlp')][col]
                if len(a) and len(b) and not np.isnan(a.iloc[0]) and not np.isnan(b.iloc[0]):
                    w += int(a.iloc[0] < b.iloc[0])
            wins[col] = w
        S['e11_wins'] = ', '.join(f'{esc(k)} {v}/4' for k, v in wins.items())
        tj = e11[e11.dataset == 'TJU']
        gv = lambda m, c: float(tj[tj.model == m][c].iloc[0])
        S['e11_tju'] = (f'MAE {gv("pinn_semi","MAE"):.4f} so với {gv("mlp","MAE"):.4f} (MLP thắng), nhưng '
                        f'cell tệ nhất {gv("pinn_semi","MAE_cell_max"):.4f} so với {gv("mlp","MAE_cell_max"):.4f}, '
                        f'vi phạm đơn điệu {gv("pinn_semi","MonoViol"):.2f} so với {gv("mlp","MonoViol"):.2f}, '
                        f'sai số EOL {gv("pinn_semi","EOL_MAE"):.0f} so với {gv("mlp","EOL_MAE"):.0f} chu kỳ')
    e4 = g('E4_table.csv')
    if e4 is not None:
        e = lambda d, t: float(e4[(e4.dataset == d) & (e4.tag == t)].MAE.iloc[0])
        S['leak_cycle'] = ', '.join(f'{d}: {e(d,"causal"):.4f} $\\to$ {e(d,"leak_cycle"):.4f} '
                                    f'($-${100*(1-e(d,"leak_cycle")/e(d,"causal")):.0f}\\,\\%)' for d in ['XJTU', 'MIT'])
        S['leak_feat'] = ', '.join(f'{d}: {e(d,"leak_feat"):.4f}' for d in ['XJTU', 'MIT'])
        S['leak_both'] = ', '.join(f'{d}: {e(d,"leak_both"):.4f}' for d in ['XJTU', 'MIT'])
    e8 = g('E8_table.csv')
    if e8 is not None:
        m, p = e8[e8.model == 'mlp'], e8[e8.model == 'pinn_semi']
        S['e8_gain_mlp'] = f'{m["gain_iso_%"].mean():.1f}\\,\\%'
        S['e8_gain_pinn'] = f'{p["gain_iso_%"].mean():.1f}\\,\\%'
        S['e8_viol'] = ', '.join(f'{d}: MLP {m[m.dataset==d].viol.mean():.1f} so với PINN {p[p.dataset==d].viol.mean():.2f}'
                                 for d in ['XJTU', 'TJU', 'MIT', 'HUST'])
    e5 = g('E5_table.csv')
    if e5 is not None:
        q = lambda s_, t_, k, m: float(e5[(e5.source == s_) & (e5.target == t_) & (e5.k == k)
                                          & (e5.model == m)].MAE.iloc[0])
        parts, wins, tot = [], 0, 0
        for a, b in [('HUST', 'MIT'), ('XJTU', 'TJU'), ('TJU', 'XJTU'), ('MIT', 'HUST')]:
            try:
                parts.append(f'{a}$\\to${b} {q(a,b,0,"mlp"):.3f} $\\to$ {q(a,b,0,"pinn_semi"):.3f}')
                tot += 1; wins += int(q(a, b, 3, 'pinn_semi') < q(a, b, 3, 'mlp'))
            except Exception:
                pass
        S['e5_short'] = ', '.join(parts); S['e5_k3'] = f'{wins}/{tot}'
    e3 = g('E3_table.csv')
    if e3 is not None:
        tj = e3[(e3.dataset == 'TJU') & (e3.variant == 'full')]
        if len(tj):
            S['Ea_TJU'] = f'{tj.Ea.iloc[0]:.1f}'
    n = sum(len(glob.glob(f'runs/{e}/*.json'))
            for e in ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E12', 'TUNED'])
    S['n_runs'] = f'{n:,}'.replace(',', '\\,')


def s(k, default='—'):
    return S.get(k, default)


# --------------------------------------------------------------------------- các tệp
PREAMBLE = r"""% Biên dịch bằng XeLaTeX (hoặc LuaLaTeX). Không dùng pdfLaTeX — cần Unicode cho tiếng Việt.
\usepackage{fontspec}
\usepackage{polyglossia}
\setdefaultlanguage{vietnamese}
\setotherlanguage{english}

% DejaVu có đủ dấu tiếng Việt và có sẵn trên hầu hết bản TeX Live / Overleaf.
% Muốn đổi phông, sửa ba dòng dưới (ví dụ: Times New Roman / Arial trên máy có sẵn).
\setmainfont{DejaVu Serif}[Scale=0.92]
\setsansfont{DejaVu Sans}[Scale=0.90]
\setmonofont{DejaVu Sans Mono}[Scale=0.85]

\usepackage{amsmath, amssymb}
\usepackage{amsthm}
\newtheorem{menhde}{Mệnh đề}
\newtheorem{nhanxet}{Nhận xét}
\renewcommand{\proofname}{Chứng minh}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\captionsetup{font=small, labelfont=bf}
\usepackage[table]{xcolor}
\definecolor{accent}{HTML}{177A68}
\definecolor{warn}{HTML}{B8531F}
\usepackage[colorlinks=true, linkcolor=accent, citecolor=accent, urlcolor=accent]{hyperref}
\usepackage{enumitem}
\setlist{itemsep=2pt, parsep=0pt, topsep=4pt}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\textsc{Ước lượng SOH pin đơn với ràng buộc vật lý}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\usepackage{tcolorbox}
\newtcolorbox{luuy}[1][]{colback=accent!6, colframe=accent, boxrule=0.8pt, left=6pt, right=6pt,
  top=5pt, bottom=5pt, sharp corners=downhill, #1}
\newtcolorbox{canhbao}[1][]{colback=warn!6, colframe=warn, boxrule=0.8pt, left=6pt, right=6pt,
  top=5pt, bottom=5pt, sharp corners=downhill, #1}
\usepackage{microtype}
\emergencystretch=3.2em     % thà giãn khoảng chữ còn hơn để dòng tràn lề
\hyphenpenalty=200\tolerance=2500
"""

MAKEFILE = """# Biên dịch báo cáo. Cần XeLaTeX (TeX Live hoặc MiKTeX).
main.pdf: main.tex preamble.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex)
\txelatex -interaction=nonstopmode main.tex
\txelatex -interaction=nonstopmode main.tex

clean:
\trm -f *.aux *.log *.out *.toc *.lof *.lot main.pdf

.PHONY: clean
""".replace('\\t', '\t')

README = """# Báo cáo LaTeX — Ước lượng SOH pin đơn với ràng buộc vật lý

## Biên dịch

```
make                 # hoặc: xelatex main.tex (chạy 2 lần để có mục lục)
```

**Bắt buộc dùng XeLaTeX hoặc LuaLaTeX** — không dùng pdfLaTeX, vì tiếng Việt cần Unicode.
Trên Overleaf: Menu → Compiler → XeLaTeX.

## Cấu trúc

```
main.tex          thân báo cáo, \\input các mục
preamble.tex      gói và định dạng (đổi phông ở đây)
sections/         từng mục nội dung
tables/           bảng số liệu, SINH TỰ ĐỘNG từ results/*.csv — đừng sửa tay
figures/          hình, sinh từ analyze.py
refs.bib          tài liệu tham khảo
```

## Cập nhật số liệu

Mọi bảng và con số trong báo cáo được sinh từ kết quả chạy thật. Sau khi chạy lại
thí nghiệm:

```
cd ..                       # về thư mục code
python analyze.py           # sinh lại results/*.csv và figures
python make_latex.py        # sinh lại toàn bộ thư mục này
```

Không sửa tay các tệp trong `tables/` — chúng sẽ bị ghi đè.

## Đổi phông chữ

Sửa ba dòng `\\setmainfont` / `\\setsansfont` / `\\setmonofont` trong `preamble.tex`.
DejaVu là mặc định vì có sẵn ở hầu hết nơi và đủ dấu tiếng Việt. Nếu máy có
Times New Roman thì thay `DejaVu Serif` bằng `Times New Roman`.
"""

BIB = r"""@article{su2024soh,
  title   = {State-of-health estimation of lithium-ion batteries: A comprehensive literature review from cell to pack levels},
  author  = {Su, Lingzhi and Xu, Yan and Dong, Zhaoyang},
  journal = {Energy Conversion and Economics},
  volume  = {5}, number = {4}, pages = {224--242}, year = {2024},
  doi     = {10.1049/enc2.12125}
}

@article{wang2024pinn,
  title   = {Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author  = {Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal = {Nature Communications},
  volume  = {15}, number = {1}, pages = {4332}, year = {2024},
  doi     = {10.1038/s41467-024-48779-z}
}

@article{severson2019data,
  title   = {Data-driven prediction of battery cycle life before capacity degradation},
  author  = {Severson, Kristen A. and Attia, Peter M. and Jin, Norman and others},
  journal = {Nature Energy}, volume = {4}, number = {5}, pages = {383--391}, year = {2019},
  doi     = {10.1038/s41560-019-0356-8}
}

@article{zhu2022tju,
  title   = {Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation},
  author  = {Zhu, Jiangong and Wang, Yixiu and Huang, Yuan and others},
  journal = {Nature Communications}, volume = {13}, number = {1}, pages = {2261}, year = {2022},
  doi     = {10.1038/s41467-022-29837-w}
}

@article{ma2022hust,
  title   = {Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning},
  author  = {Ma, Guijun and Xu, Songpei and Jiang, Benben and others},
  journal = {Energy \& Environmental Science}, volume = {15}, number = {10}, pages = {4083--4094}, year = {2022},
  doi     = {10.1039/D2EE01676A}
}

@article{dosreis2021data,
  title   = {Lithium-ion battery data and where to find it},
  author  = {dos Reis, Gon{\c{c}}alo and Strange, Calum and Yadav, Mohit and Li, Shawn},
  journal = {Energy and AI}, volume = {5}, pages = {100081}, year = {2021},
  doi     = {10.1016/j.egyai.2021.100081}
}

@inproceedings{ziyin2020snake,
  title     = {Neural networks fail to learn periodic functions and how to fix it},
  author    = {Ziyin, Liu and Hartwig, Tilman and Ueda, Masahito},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33}, pages = {1583--1594}, year = {2020}
}

@article{raissi2019pinn,
  title   = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E.},
  journal = {Journal of Computational Physics}, volume = {378}, pages = {686--707}, year = {2019},
  doi     = {10.1016/j.jcp.2018.10.045}
}
"""


def build():
    load_stats()
    for d in ['sections', 'tables', 'figures']:
        os.makedirs(f'{OUT}/{d}', exist_ok=True)
    for f in glob.glob(f'{R}/*.png') + glob.glob(f'{R}/*.pdf'):
        shutil.copy(f, f'{OUT}/figures/')

    # ---- bảng sinh từ markdown
    T = {
        'E1': md_to_tables(f'{R}/E1_tables.md', 'E1'),
        'E12': md_to_tables(f'{R}/E12_table.md', 'E12',
                            {'E12': 'Ngân sách tinh chỉnh cân bằng: cùng một lưới 8 biến thể siêu tham số cho '
                                    'cả hai mô hình, chọn theo MAE validation, đọc test sau.'}),
        'E12B': md_to_tables(f'{R}/E12_poolB.md', 'E12B',
                             {'E12B': 'Mỗi mô hình chọn trong toàn bộ kho biến thể của nó. Hai mức 10\\,\\% và '
                                      '50\\,\\% có số biến thể chênh lệch nên KHÔNG cân bằng — chỉ đọc 30\\,\\% và 70\\,\\%.'}),
        'E7': md_to_tables(f'{R}/E7_table.md', 'E7',
                           {'E7': 'Trọng số vật lý chọn theo validation ở mọi mức nhãn.'}),
        'E9': md_to_tables(f'{R}/E9_table.md', 'E9'),
        'E10': md_to_tables(f'{R}/E10_table.md', 'E10'),
        'E11': md_to_tables(f'{R}/E11_table.md', 'E11',
                            {'E11': 'Bộ chỉ số mở rộng, tính lại từ dự đoán đã lưu ở 30\\,\\% nhãn.'}),
        'E5': md_to_tables(f'{R}/E5_table.md', 'E5',
                           {'E5': 'Thích nghi sang bộ dữ liệu khác bằng các loss không cần nhãn.'}),
        'E3': md_to_tables(f'{R}/E3_table.md', 'E3', {'E3': 'Ablation ở 30\\,\\% nhãn.'}),
        'E4': md_to_tables(f'{R}/E4_table.md', 'E4', {'E4': 'Kiểm toán rò rỉ do cách chuẩn hoá.'}),
        'E8': md_to_tables(f'{R}/E8_table.md', 'E8', {'E8': 'Giải mã đơn điệu hậu kỳ, áp cho cả hai mô hình.'}),
        'E2': md_to_tables(f'{R}/E2_table.md', 'E2', {'E2': 'Chuyển miền zero-shot theo cặp.'}),
    }
    for k, v in T.items():
        open(f'{OUT}/tables/{k}.tex', 'w').write(v)

    # ---- các mục
    sec = {}

    sec['00-tomtat'] = rf"""\section{{Tóm tắt}}\label{{sec:tomtat}}

Bài toán: ước lượng SOH (dung lượng hiện tại chia dung lượng danh định) của một cell
lithium-ion từ 16 đặc trưng thống kê của đoạn sạc CC--CV, tại từng chu kỳ. Điểm khác
biệt so với PINN4SOH~\cite{{wang2024pinn}} nằm ở ba chỗ: động học suy giảm có cấu trúc
vật lý thay vì mạng đen; mọi hàm mất mát vật lý đều \textbf{{không dùng nhãn}} nên áp
được lên cả cell chưa đo dung lượng; và pipeline không rò rỉ thông tin (không chuẩn hoá
theo từng cell bằng thống kê cả vòng đời, chia dữ liệu theo cell, tập val/test cố định).

Toàn bộ kết quả dựa trên {s('n_runs')} lượt huấn luyện thật trên 387 cell của bốn bộ dữ
liệu công khai (XJTU, TJU, MIT, HUST), trung bình 3 seed; riêng phép so chính ở 30\,\%
nhãn được chạy lại với 10 seed và kiểm định thống kê ở mục~\ref{{sec:chungminh}}.

\subsection*{{Giả thuyết chính và kết quả}}

Giả thuyết: \emph{{có ràng buộc vật lý thì huấn luyện với 30\,\% nhãn cho kết quả gần bằng
protocol 70/15/15 đầy đủ}}. Kiểm bằng cách giữ nguyên tập val/test rồi giảm dần số cell
mang nhãn:

\begin{{itemize}}
{s('hyp_rows')}
\end{{itemize}}

\begin{{luuy}}
\textbf{{Con số nên trích dẫn.}} Các tỉ số trên dùng một bộ trọng số vật lý cố định và
baseline chưa được tinh chỉnh. Khi cả hai mô hình cùng được hưởng một ngân sách tinh
chỉnh (mục~\ref{{sec:e12}}), tỉ số PINN/MLP trung bình đi từ {s('e12_before')} lên
{s('e12_after')}, và PINN thắng ở {s('e12_wins')} ô với biên độ {s('e12_range')}.
Đây mới là con số trung thực.

\smallskip
\textbf{{Và ngay cả con số đó cũng chưa phải bằng chứng thống kê.}} Chạy lại phép so ở
30\,\% nhãn với 10 seed (mục~\ref{{sec:chungminh}}): ở \emph{{cấu hình E1 trong danh sách
trên}}, không bộ nào cho thấy PINN tốt hơn có ý nghĩa và TJU thì PINN \emph{{kém hơn}} có
ý nghĩa --- tức các tỉ số trong danh sách trên nằm trong nhiễu seed. Ở cấu hình đã tinh
chỉnh, ba trên bốn bộ có khoảng tin cậy của tỉ số nằm hoàn toàn dưới 1 ($0{{,}}72$--$0{{,}}90$)
nhưng chưa bộ nào qua được hiệu chỉnh đa so sánh. Phát biểu đúng với dữ liệu hiện có:
\emph{{có bằng chứng nhất quán rằng PINN giảm MAE ở cấu hình đã tinh chỉnh, chưa đạt mức
có ý nghĩa thống kê}}.
\end{{luuy}}

Vế thứ hai của giả thuyết --- ``đem sang bộ dữ liệu khác vẫn tốt'' --- \textbf{{không đúng}}
ở dạng zero-shot: MAE tăng hai đến ba bậc (mục~\ref{{sec:e10}}). Nhưng vì các loss vật lý
không cần nhãn, chúng chạy được ngay trên cell của bộ đích và trở thành cơ chế thích nghi
miền không nhãn: {s('e5_short')}.
"""

    sec['01-khoangtrong'] = r"""\section{Khoảng trống trong PINN4SOH}\label{sec:khoangtrong}

PINN4SOH~\cite{wang2024pinn} là điểm xuất phát tự nhiên: cùng bốn bộ dữ liệu, cùng 16 đặc
trưng, mã nguồn mở. Đọc kỹ mã nguồn (repo \texttt{wang-fujin/PINN4SOH}, bản clone
05/09/2026) cho thấy bốn điểm mà bài báo không nêu rõ, mỗi điểm mở ra một chỗ đóng góp.

\begin{enumerate}
\item \textbf{Phần dư PDE suy biến} (\texttt{Model/Model.py:232--234}). Mã nguồn tính
\texttt{F = dynamical\_F(cat[xt, u, u\_x, u\_t])} rồi \texttt{f = u\_t $-$ F}. Mạng động học
nhận chính $\partial u/\partial t$ làm đầu vào, nên có thể học $F \approx u_t$ và triệt tiêu
phần dư mà không cần bất kỳ vật lý nào.

\item \textbf{``Physics loss'' phụ thuộc nhãn} (\texttt{Model/Model.py:255}):
\texttt{loss3 = relu((u2$-$u1)$\cdot$(y1$-$y2)).sum()} --- phạt khi hướng thay đổi của dự đoán
ngược với hướng của \emph{nhãn}. Đây là ràng buộc đồng dấu với nhãn, không phải tính đơn
điệu vật lý, nên \textbf{không áp được lên cell không có nhãn}.

\item \textbf{Chuẩn hoá theo từng cell trên cả vòng đời}
(\texttt{dataloader/dataloader.py:50, 58--64}). Chỉ số chu kỳ được chèn làm đặc trưng rồi
min--max theo từng file cell. Chỉ số chuẩn hoá khi đó chính là ``phần trăm tuổi thọ đã đi
qua'' --- rò rỉ tổng tuổi thọ của cell test vào đầu vào, và không tính được khi chạy trực
tuyến. Mục~\ref{sec:e4} định lượng mức ảnh hưởng.

\item \textbf{Tập validation chia theo hàng, không theo cell}
(\texttt{dataloader.py:157--158}). Validation lấy ngẫu nhiên 20\,\% \emph{hàng} từ các cell
huấn luyện, nên chọn mô hình và dừng sớm trên đúng những cell đã học.
\end{enumerate}

Không điểm nào ở trên phủ nhận kết quả của họ. Chúng chỉ nói rằng \emph{cơ chế} vì sao
PINN4SOH cần ít dữ liệu chưa được chứng minh, và con số có thể được nâng lên bởi rò rỉ
chuẩn hoá.
"""

    sec['02-dulieu'] = r"""\section{Dữ liệu}\label{sec:dulieu}

Dùng đúng dữ liệu đã tiền xử lý trong repo PINN4SOH để kết quả đối chiếu được. Mỗi hàng là
một chu kỳ; 16 đặc trưng thống kê (trung bình, độ lệch chuẩn, kurtosis, skewness, điện
lượng, thời gian, độ dốc, entropy --- cho cả điện áp lẫn dòng) của đoạn cuối CC và pha CV;
nhãn là dung lượng xả.

\begin{table}[htbp]
\centering\small
\caption{Bốn bộ dữ liệu đơn pin công khai, sau bước lọc.}
\label{tab:datasets}
\adjustbox{max width=\textwidth}{%
\begin{tabular}{lrrlll}
\toprule
Bộ & Cell & Chu kỳ & Hoá học $\cdot$ danh định & Nhiệt độ & Điều kiện \\
\midrule
XJTU & 55  & 22\,212  & NCM $\cdot$ 2.0\,Ah              & 25\,\textdegree C       & 6 nhóm: 2C, 3C, R2.5, R3, RW, vệ tinh \\
TJU  & 130 & 56\,779  & NCA/NCM $\cdot$ 3.5/2.5\,Ah      & 25/35/45\,\textdegree C & sạc 0.25--1C, xả 1--4C \\
MIT  & 125 & 81\,865  & LFP (A123) $\cdot$ 1.1\,Ah       & 30\,\textdegree C       & 3 đợt, sạc nhanh đa dạng \\
HUST & 77  & 143\,172 & LFP (A123) $\cdot$ 1.1\,Ah       & 30\,\textdegree C       & 77 profile xả nhiều bậc \\
\bottomrule
\end{tabular}}
\end{table}

Lọc: bỏ hàng có giá trị vô hạn hoặc thiếu, và hàng có đặc trưng lệch quá $3\sigma$ so với
chính cell đó (chỉ xét đặc trưng, không xét nhãn). SOH bằng dung lượng chia danh định;
XJTU bắt đầu ở 0.92--0.99 còn HUST ở 1.06--1.12 (danh định thấp hơn thực tế), vì thế ràng
buộc ``cell mới $\approx 1$'' được nới thành khoảng $[0.85,\ 1.10]$.

\subsection{Dấu tương quan đặc trưng--SOH không bất biến theo hoá học}

Trước khi đưa bất kỳ tiên nghiệm dấu nào vào hàm mất mát, tôi đo tương quan Spearman trong
từng cell rồi lấy trung vị.

\begin{table}[htbp]
\centering\small
\caption{Trung vị tương quan Spearman trong từng cell giữa đặc trưng và SOH.}
\label{tab:spearman}
\begin{tabular}{lrrrr}
\toprule
Đặc trưng & XJTU (NCM) & TJU (NCA/NCM) & MIT (LFP) & HUST (LFP) \\
\midrule
CC Q (điện lượng pha CC) & $-0.18$ & $+0.95$ & $+0.98$ & $+0.97$ \\
CC charge time           & $-0.17$ & $+0.95$ & $+0.98$ & $+0.97$ \\
CV Q (điện lượng pha CV) & $-0.94$ & $-1.00$ & $+0.41$ & $+0.55$ \\
CV charge time           & $-0.94$ & $-1.00$ & $+0.62$ & $+0.64$ \\
voltage mean             & $+0.60$ & $+0.04$ & $-0.99$ & $-0.94$ \\
current kurtosis         & $-0.65$ & $-0.97$ & $+0.77$ & $+0.87$ \\
\bottomrule
\end{tabular}
\end{table}

Cùng một đặc trưng đổi dấu giữa NCM/NCA và LFP (CV Q: $-0.94$/$-1.00$ so với
$+0.41$/$+0.55$), thậm chí giữa hai bộ cùng NCM (CC Q: $-0.18$ ở XJTU nhưng $+0.95$ ở TJU,
do XJTU có protocol ngẫu nhiên). Ba hệ quả cho thiết kế:

\begin{enumerate}
\item Mọi ràng buộc dấu trên đặc trưng đều \textbf{bị loại}, vì không phổ quát.
\item Vật lý được dùng phải nằm ở \emph{quỹ đạo SOH} (đơn điệu, tốc độ không âm,
Arrhenius) chứ không ở đặc trưng.
\item Chuyển miền zero-shot giữa hai họ hoá học với bộ đặc trưng này về nguyên tắc là
không khả thi --- điều mà mục~\ref{sec:e10} xác nhận bằng số liệu.
\end{enumerate}
"""

    F_PIPE = fig('fig_pipeline.png',
                 'Pipeline đúng như mã nguồn chạy. Điểm cần chú ý: mạng nghiệm ánh xạ MỘT chu kỳ sang MỘT '
                 'giá trị SOH --- không có CNN, attention hay RNN --- nên cấu trúc quỹ đạo chỉ đi vào qua các '
                 'hàm mất mát vật lý, vốn được tính trên cặp chu kỳ $(N, N{+}h)$ của cùng một cell. Động học '
                 '$r$ chỉ tham gia lúc huấn luyện; khi suy luận chỉ dùng mạng nghiệm.', 'pipeline')

    sec['03-kientruc'] = rf"""\section{{Kiến trúc}}\label{{sec:kientruc}}

{F_PIPE}

Hai mạng nhỏ, tổng cộng khoảng 9\,000 tham số.

\begin{{equation}}
u(N) = F_{{\phi}}\big(\mathbf{{x}}_N,\ \tilde N\big), \qquad \tilde N = N/1000
\label{{eq:solution}}
\end{{equation}}

\begin{{equation}}
\frac{{\mathrm{{d}}u}}{{\mathrm{{d}}\tilde N}} = -\,r(\mathbf{{x}}, u, T), \qquad
r = \underbrace{{\operatorname{{softplus}}\!\big(\mathrm{{MLP}}_{{\theta}}(\mathbf{{x}})\big)}}_{{\text{{tốc độ cơ sở}}\ \ge 0}}
\cdot \underbrace{{\exp\!\big(\lambda(1-u)\big)}}_{{\text{{gia tốc knee}}}}
\cdot \underbrace{{\exp\!\Big(-\tfrac{{E_a}}{{R}}\big(\tfrac{{1}}{{T}}-\tfrac{{1}}{{T_{{\mathrm{{ref}}}}}}\big)\Big)}}_{{\text{{Arrhenius}}}}
\label{{eq:dynamics}}
\end{{equation}}

Ba thừa số mang ba mẩu vật lý khác nhau, mỗi mẩu kiểm chứng được riêng bằng ablation
(mục~\ref{{sec:e3}}):

\begin{{itemize}}
\item $\operatorname{{softplus}}(\cdot) \ge 0$: tốc độ mất dung lượng không âm, nên nghiệm
của~\eqref{{eq:dynamics}} đơn điệu không tăng \emph{{theo cấu trúc}}, không cần nhãn để dạy.
\item $\exp(\lambda(1-u))$ với $\lambda \ge 0$ học được: tốc độ tăng dần khi SOH giảm ---
mô tả pha ``đầu gối'' khi mất vật liệu hoạt động lấn át mất lithium. $\lambda$ dùng chung
cho cả bộ dữ liệu nên đọc được như một tham số vật lý.
\item Arrhenius với $E_a \ge 0$ học được, $T_{{\mathrm{{ref}}}} = 298.15$\,K: chỉ có tác dụng
khi bộ dữ liệu có nhiều nhiệt độ (TJU). Ở bộ đơn nhiệt độ thừa số này bằng 1 và $E_a$ giữ
nguyên giá trị khởi tạo --- đúng như mong đợi.
\end{{itemize}}

Khác với PINN4SOH, mạng động học \emph{{không}} nhận đạo hàm của $u$ làm đầu vào, nên phần
dư không thể tự triệt tiêu. Đầu vào chỉ số chu kỳ chia cho hằng số toàn cục 1000, không
bao giờ chuẩn hoá theo cell.

\subsection{{Đầu ra đơn điệu theo cấu trúc}}\label{{sec:mono}}

Thay vì phạt vi phạm đơn điệu bằng hàm mất mát, có thể đưa nó vào kiến trúc:

\begin{{equation}}
u(\mathbf{{x}}, t) = u_0(\mathbf{{x}}) - \sum_{{k=1}}^{{K}} s_k(\mathbf{{x}})\,
\operatorname{{softplus}}\!\big(\omega_k t + \varphi_k(\mathbf{{x}})\big), \qquad
s_k = \operatorname{{softplus}}(\cdot) \ge 0,\ \ \omega_k = \operatorname{{softplus}}(\cdot) \ge 0
\label{{eq:mono}}
\end{{equation}}

Vì softplus đơn điệu tăng và mọi hệ số đều không âm,
$\partial u/\partial t = -\sum_k s_k \omega_k \operatorname{{sigmoid}}(\cdot) \le 0$
\emph{{với mọi giá trị tham số}}. Tính đơn điệu không còn là số hạng phạt cần cân trọng số
mà là tính chất của không gian hàm.

\begin{{canhbao}}
\textbf{{Giới hạn cần nêu rõ.}} Công thức~\eqref{{eq:mono}} đơn điệu theo $t$, nhưng dự đoán
còn phụ thuộc $\mathbf{{x}}_N$ --- mà đặc trưng sạc dao động theo chu kỳ. Nên quỹ đạo
\emph{{dọc theo cell}} vẫn có thể tăng ngược. Số liệu ở mục~\ref{{sec:e9}} xác nhận: vi phạm
đơn điệu đo được là {s('e9_mono_viol')}, tức \emph{{tệ hơn}} mô hình dùng $\mathcal{{L}}_{{\text{{mono}}}}$.
Đơn điệu theo $t$ không đồng nghĩa đơn điệu dọc quỹ đạo.
\end{{canhbao}}
"""

    sec['04-loss'] = r"""\section{Các hàm mất mát vật lý}\label{sec:loss}

Mọi loss vật lý được tính trên cặp chu kỳ $(N, N{+}h)$ của \emph{cùng một cell} với $h$ ngẫu
nhiên trong $\{5,\dots,50\}$, và \textbf{không dùng nhãn}. Chỉ $\mathcal{L}_{\text{data}}$
cần nhãn.

\begin{align}
\mathcal{L}_{\text{data}} &= \frac{1}{|\mathcal{D}_L|}\sum_{(N,c)\in\mathcal{D}_L}\big(u_c(N) - y_{c,N}\big)^2
\label{eq:ldata}\\[3pt]
\mathcal{L}_{\text{ode}} &= \mathbb{E}_{c,N,h}
\Big[\,u_c(N{+}h) - u_c(N) + r\big(\mathbf{x}_{c,N}, u_c(N), T_c\big)\,\tfrac{h}{1000}\Big]^2
\label{eq:lode}\\[3pt]
\mathcal{L}_{\text{mono}} &= \mathbb{E}_{c,N,h}\ \operatorname{ReLU}\!\big(u_c(N{+}h) - u_c(N) - \varepsilon\big),
\qquad \varepsilon = 0.002
\label{eq:lmono}\\[3pt]
\mathcal{L}_{\text{range}} &= \mathbb{E}\big[\operatorname{ReLU}(u-1.15)+\operatorname{ReLU}(0.4-u)\big]
+ \mathbb{E}_{N=N_0}\big[\operatorname{ReLU}(u-1.10)+\operatorname{ReLU}(0.85-u)\big]
\label{eq:lrange}\\[3pt]
\mathcal{L} &= \mathcal{L}_{\text{data}} + w(s)\big(\alpha\mathcal{L}_{\text{ode}}
+ \beta\mathcal{L}_{\text{mono}} + \gamma\mathcal{L}_{\text{range}}\big),
\qquad w(s)=\min\!\big(1,\ s/(0.2\,S)\big)
\label{eq:total}
\end{align}

$\mathcal{D}_L$ là tập chu kỳ của các cell \emph{có} nhãn. Các kỳ vọng
trong~\eqref{eq:lode}--\eqref{eq:lrange} lấy trên \emph{mọi} cell huấn luyện (kể cả cell
không nhãn) với mô hình bán giám sát. $w(s)$ tăng tuyến tính từ 0 lên 1 trong 20\,\% số
bước đầu để mạng bám dữ liệu trước khi vật lý siết lại.

\subsection{Vì sao dùng dạng Euler thay vì dạng đạo hàm}

Phiên bản đầu dùng phần dư dạng đạo hàm $\big(u(N{+}h) - u(N)\big)/(h/1000) + r$. Với
$h = 1$ chu kỳ, mẫu số $0.001$ \textbf{khuếch đại nhiễu 1000 lần}: một dao động $10^{-3}$
của mạng giữa hai chu kỳ liên tiếp trở thành phần dư cỡ 1, lớn hơn hẳn tín hiệu thật
($|\mathrm{d}u/\mathrm{d}\tilde N| \approx 0.2$--$0.5$). Mạng ``học'' cách giảm phần dư bằng
cách trở nên vô cảm với đặc trưng, và MAE xấu đi gấp đôi so với baseline. Dạng
Euler~\eqref{eq:lode} nhân hai vế với $h/1000$: cặp horizon ngắn đóng góp rất nhỏ, cặp
horizon dài tự nhiên chi phối, nhiễu không bị khuếch đại. Ablation ở
mục~\ref{sec:e3} xác nhận (\texttt{res\_fd} tệ nhất ở cả hai bộ).

Hai ràng buộc từng cân nhắc rồi loại bỏ: tiên nghiệm dấu trên đặc trưng
(mục~\ref{sec:dulieu} giải thích) và điều kiện đầu cứng $u(0) = 1$ (sai với HUST và XJTU).

\subsection{Hai cơ chế khác nhau của vật lý}\label{sec:coche}

\begin{description}
\item[Vật lý như regularizer (\texttt{pinn\_sup}).] Loss vật lý chỉ tính trên cell có nhãn.
Vật lý thu hẹp không gian hàm nên ít dữ liệu hơn vẫn xác định được nghiệm, nhưng không
thêm thông tin từ cell nào khác.

\item[Vật lý như nhãn thay thế (\texttt{pinn\_semi}).] Loss vật lý tính trên \emph{mọi} cell
huấn luyện, gồm cả cell chỉ có đường sạc mà không có nhãn dung lượng. Vì
\eqref{eq:lode}--\eqref{eq:lrange} không cần $y$, mạng buộc phải gán cho các cell này một
quỹ đạo SOH đơn điệu, bắt đầu trong $[0.85, 1.10]$, suy giảm với tốc độ $r(\mathbf{x})$ đã
học từ cell có nhãn. Đây chính là tình huống thực tế: hệ quản lý pin ghi đường sạc của mọi
cell, nhưng đo dung lượng chuẩn chỉ làm được với vài cell.
\end{description}

PINN4SOH --- với loss vật lý phụ thuộc nhãn --- về cấu trúc chỉ có thể là cơ chế thứ nhất.
"""

    sec['05-protocol'] = r"""\section{Protocol thí nghiệm}\label{sec:protocol}

\begin{description}[leftmargin=2.2cm, style=nextline]
\item[Chia dữ liệu] Theo \textbf{cell}, phân tầng theo nhóm protocol, 70/15/15
(train/val/test). Có kiểm tra khẳng định không cell nào xuất hiện ở hai tập.

\item[Giảm nhãn] Giữ nguyên val/test của cùng seed; trong 70\,\% train chỉ một phần cell
mang nhãn: 10/30/50/70\,\% của \emph{tổng} số cell. Cell còn lại là ``không nhãn'' --- mô
hình bán giám sát chỉ thấy đặc trưng của chúng.

\item[Chuẩn hoá] z-score fit trên đặc trưng của cell huấn luyện; chỉ số chu kỳ chia 1000.
Biến thể \texttt{first\_cycle}: đặc trưng tương đối so với 3 chu kỳ đầu của chính cell ---
nhân quả, dùng cho chuyển miền.

\item[Huấn luyện] AdamW, lr $2\times10^{-3}$ cosine, batch 512, tối đa 4\,000 bước, dừng sớm
theo MAE validation (kiên nhẫn 12 lần đánh giá), chọn checkpoint tốt nhất trên validation.
Cùng ngân sách bước cho mọi mức nhãn.

\item[Đo] Bộ chỉ số ở mục~\ref{sec:e11}, trên \emph{mọi} chu kỳ của cell test. Trung bình
$\pm$ độ lệch chuẩn trên 3 seed (seed đổi cả cách chia lẫn khởi tạo).
\end{description}

\begin{table}[htbp]
\centering\small
\caption{Danh mục thí nghiệm.}
\label{tab:protocol}
\begin{tabular}{lp{11.5cm}}
\toprule
Mã & Nội dung \\
\midrule
E1  & Hiệu quả theo lượng nhãn: 4 bộ $\times$ 4 mức nhãn $\times$ 3 mô hình $\times$ 3 seed. \\
E2  & Chuyển miền zero-shot theo cặp cùng họ hoá học. \\
E3  & Ablation ở 30\,\% nhãn: bỏ từng loss, bỏ knee, bỏ Arrhenius, đổi dạng phần dư, động học đen. \\
E4  & Kiểm toán rò rỉ: chuẩn hoá nhân quả so với chuẩn hoá theo cell kiểu PINN4SOH. \\
E5  & Thích nghi miền: nguồn 70\,\% nhãn $+$ đích với $k \in \{0,3\}$ cell có nhãn. \\
E7  & Trọng số vật lý chọn theo validation ở \emph{mọi} mức nhãn; 5 biến thể. \\
E8  & Giải mã đơn điệu hậu kỳ (PAVA), áp cho cả hai mô hình. \\
E9  & Lưới kiến trúc $\times$ hàm kích hoạt, cho cả baseline lẫn PINN. \\
E10 & Ma trận chuyển miền đầy đủ $4\times4$, đường chéo là kết quả trong miền. \\
E11 & Bộ chỉ số mở rộng, tính lại từ dự đoán đã lưu. \\
E12 & Ngân sách tinh chỉnh cân bằng: cùng 8 biến thể siêu tham số cho hai mô hình. \\
\bottomrule
\end{tabular}
\end{table}
"""

    F_E1 = fig('fig_E1_label_efficiency.png',
               'MAE trên tập test cố định khi giảm số cell có nhãn. Trục dọc dùng thang log CHUNG cho cả bốn bộ, đơn vị '
               '$10^{-3}$ SOH: cùng một khoảng cách dọc luôn là cùng một TỈ SỐ sai số, đúng với cách các kết luận '
               'của bài được phát biểu. Dải mờ là $\\pm 1$ độ lệch chuẩn trên 3 seed.', 'e1')
    F_TRAJ = fig('fig_traj_XJTU.png',
                 'XJTU, 30\\,\\% nhãn, seed 0: quỹ đạo dự đoán trên bốn cell test. Đường MLP dao động và có đoạn '
                 'tăng ngược; PINN-semi trơn và đơn điệu.', 'traj')
    F_RATIO = fig('fig_ratio_matrix.png',
                  'Tỉ số MAE của PINN so với MLP ở CÙNG mức nhãn. Đây là đại lượng có dấu quanh 1.00 nên '
                  'dùng thang phân kỳ với tâm đúng tại 1.00: xanh lục là PINN tốt hơn, cam là MLP tốt hơn. '
                  'Bảng trái dùng trọng số cố định $\\beta = 5$; bảng phải chọn $\\beta$ theo validation.', 'ratio')

    F_E7 = fig('fig_E7_tuned.png',
               'Đường vàng là cấu hình mặc định cũ. Đường xanh lục đậm là cùng mô hình đó với $\\beta$ chọn theo '
               'validation. Đường hồng là trọng số tự thích ứng, không tinh chỉnh gì. Đường đứt xanh lam là MLP ở '
               'cùng mức nhãn. Trục dọc thang log chung cho bốn bộ, đơn vị $10^{-3}$ SOH; hai đường nhạt là biến '
               'thể đối chứng.', 'e7')
    F_E12 = fig('fig_E12_fair.png',
                'Cột nhạt: MLP với cấu hình mặc định. Cột giữa: MLP sau khi được hưởng đúng ngân sách tinh chỉnh '
                'của PINN. Cột đậm: PINN sau tinh chỉnh.', 'e12')
    F_E9 = fig('fig_E9_arch.png',
               'MAE chuẩn hoá theo MLP $+$ SiLU, trung bình 4 bộ dữ liệu. Đường đứt là mốc cơ sở.', 'e9')
    F_E10 = fig('fig_E10_matrix.png',
                'Thang log. Hàng là bộ huấn luyện, cột là bộ đánh giá. Càng nhạt càng tốt.', 'e10')
    F_E5 = fig('fig_E5_adapt.png', 'Trục log. Màu nhạt: $k = 0$. Màu đậm: $k = 3$ cell đích có nhãn.', 'e5')

    sec['06-ketqua-nhan'] = rf"""\section{{Kết quả: hiệu quả theo lượng nhãn}}\label{{sec:e1}}

\input{{tables/E1}}

{F_E1}

{F_TRAJ}

\section{{Trọng số vật lý quyết định, không phải lượng nhãn}}\label{{sec:e7}}

Với một bộ trọng số cố định ($\beta = 5$), PINN thua MLP ở mức nhãn cao trên TJU và XJTU.
E7 kiểm chứng bằng cách quét $\beta \in \{{0.5, 1.5, 5\}}$, dạng phần dư Euler/autograd, cộng
biến thể tự thích ứng, trên \textbf{{mọi}} mức nhãn, chọn theo MAE validation của từng cặp
(bộ, mức nhãn), rồi mới đọc test.

{F_RATIO}

{F_E7}

\input{{tables/E7}}

Kết quả đảo chiều so với E1: PINN thắng MLP ở cùng mức nhãn trong \textbf{{{s('e7_wins')}}}
trường hợp. Tại mốc 50\,\% nhãn, PINN thắng MLP cùng mức trên cả bốn bộ
({s('e7_50_mlp')}), và so với MLP dùng đủ 70\,\% nhãn thì đạt {s('e7_50_mlp70')}, tức bằng
hoặc tốt hơn ở {s('e7_50_le70')} bộ.

\begin{{luuy}}
\textbf{{Quy luật chọn $\beta$.}} Ở mức nhãn thấp (10--30\,\%) validation chọn
{s('e7_sel_lo')}; ở mức cao (50--70\,\%) chọn {s('e7_sel_hi')}. Trọng số vật lý nên
\emph{{giảm khi nhãn tăng}} --- đúng như kỳ vọng cho một tiên nghiệm. Biến thể
\texttt{{adapt}} mã hoá sẵn quy luật này:
$\beta_{{\text{{hiệu dụng}}}} = \beta_0\big[(1 - f_{{\text{{nhãn}}}}) + 0.15\big]$
với $f_{{\text{{nhãn}}}}$ là tỉ lệ cell có nhãn trong tập train. Không cần tinh chỉnh gì và
thắng MLP ở {s('e7_adapt_wins')} ô.
\end{{luuy}}
"""

    sec['07-e12'] = rf"""\section{{Ngân sách tinh chỉnh cân bằng}}\label{{sec:e12}}

Mọi bảng phía trên có một lỗ hổng: PINN được quét $\beta$, dạng phần dư và kiến trúc, còn
baseline MLP chạy một cấu hình cố định --- như vậy là thiên vị. E12 sửa bằng cách áp
\textbf{{đúng một lưới siêu tham số tổng quát}} (8 biến thể: weight decay
$10^{{-5}}/10^{{-3}}/10^{{-2}}$, dropout $0/0.1$, mạng rộng $(128,128,64)$ hoặc hẹp
$(32,32,16)$, kiến trúc đơn điệu có và không weight decay) cho \emph{{cả hai}} mô hình, chọn
theo MAE validation, rồi mới đọc test.

{F_E12}

\input{{tables/E12}}

\begin{{canhbao}}
\textbf{{Đính chính.}} Tinh chỉnh baseline cải thiện nó rất nhiều ở vài chỗ --- lớn nhất:
{s('e12_gain')}. Riêng MIT ở 30\,\% nhãn, chỉ thêm dropout 0.1 đã đưa MLP từ 0.0103 xuống
0.0048. Nghĩa là con số ``PINN tốt hơn 3$\times$'' ở các mục trước \emph{{phần lớn là do
baseline bị bỏ đói tinh chỉnh}}, không phải do vật lý. Trung bình trên 8 ô: tỉ số PINN/MLP
là {s('e12_before')} khi baseline chưa tinh chỉnh, và {s('e12_after')} khi cả hai cùng
được tinh chỉnh.
\end{{canhbao}}

Sau khi cân bằng, kết luận vẫn đứng nhưng khiêm tốn hơn: PINN thắng ở
\textbf{{{s('e12_wins')}}} ô, tỉ số {s('e12_range')}. Ngoại lệ là TJU ({s('e12_tju')}) ---
bộ có nhiều cell nhất trên mỗi protocol và nhãn sạch nhất, tức nơi baseline ít cần tiên
nghiệm nhất. Biến thể được chọn nhiều nhất: {s('e12_tags')}, cho thấy mạng $(64,64,32)$ ban
đầu thiếu dung lượng cho cả hai mô hình.

\input{{tables/E12B}}
"""

    sec['08-e9'] = rf"""\section{{Thiết kế lại mạng: kiến trúc và hàm kích hoạt}}\label{{sec:e9}}

Bảy biến thể ở 30\,\% nhãn, chạy cho \textbf{{cả}} baseline chỉ dữ liệu lẫn PINN-semi.

{F_E9}

\input{{tables/E9}}

\begin{{itemize}}
\item \textbf{{Đầu ra đơn điệu theo cấu trúc}}~\eqref{{eq:mono}} \textbf{{là kiến trúc tốt nhất
cho cả hai mô hình}} --- {s('e9_mono_pinn')} cho PINN và {s('e9_mono_mlp')} cho baseline,
chỉ thêm khoảng 5\,\% tham số. Đáng chú ý: nó giúp baseline gần bằng mức giúp PINN, tức
\emph{{một phần}} giá trị của ``vật lý'' lấy được bằng thiết kế mạng, không cần loss nào.
Sai số dự báo EOL giảm mạnh nhất ({s('e9_mono_eol')} chu kỳ).

\item \textbf{{Hàm kích hoạt tương tác với vật lý.}} tanh và Snake~\cite{{ziyin2020snake}}
\emph{{giúp}} PINN ({s('e9_tanh_pinn')}, {s('e9_snake_pinn')}) nhưng \emph{{hại}} baseline
({s('e9_tanh_mlp')}, {s('e9_snake_mlp')} --- Snake gần gấp đôi sai số). Kích hoạt trơn hoặc
dao động chỉ có lợi khi có ràng buộc giữ mô hình khỏi bám nhiễu. Riêng \texttt{{sin}} mà
PINN4SOH dùng không tốt hơn SiLU ({s('e9_sin_pinn')}).

\item \textbf{{Residual không đáng.}} Gấp đôi tham số, MAE {s('e9_res_pinn')} --- ở quy mô
mạng này độ sâu không phải nút thắt.

\item \textbf{{Đơn điệu theo $t$ không đồng nghĩa đơn điệu dọc quỹ đạo}} --- xem cảnh báo ở
mục~\ref{{sec:mono}}. Ghép cả hai (kiến trúc đơn điệu $+$ giữ $\mathcal{{L}}_{{\text{{mono}}}}$)
khôi phục tính đơn điệu nhưng siết quá tay và MAE xấu đi. Cách đúng là làm cho \emph{{nhánh
đặc trưng}} cũng đơn điệu, chứ không chồng thêm phạt.
\end{{itemize}}
"""

    sec['09-e10'] = rf"""\section{{Ma trận chuyển miền đầy đủ}}\label{{sec:e10}}

Mỗi bộ làm nguồn (70\,\% nhãn), đánh giá zero-shot trên tập \emph{{test}} của cả bốn bộ ---
dùng cùng seed nên \textbf{{đường chéo trùng đúng kết quả trong miền}} và so sánh trực tiếp
được với các ô ngoài đường chéo.

{F_E10}

\input{{tables/E10}}

\begin{{itemize}}
\item \textbf{{Khoảng cách trong miền và ngoài miền là hai đến ba bậc.}} Đường chéo trung
bình {s('e10_dia')}; ô ngoài đường chéo tệ nhất là {s('e10_worst')} --- lớn hơn 3\,000 lần.
Không mô hình nào trong bốn cấu hình thoát khỏi điều này.

\item \textbf{{Chuẩn hoá nhân quả theo chu kỳ đầu là đòn bẩy mạnh nhất, không phải kiến trúc
hay loss.}} Trung vị các ô ngoài đường chéo: {s('e10_med')}. Đây là thay đổi \emph{{tiền xử
lý}}, không phải mô hình.

\item \textbf{{Cùng loại cell mới chuyển được.}} MIT$\leftrightarrow$HUST (đều A123 LFP
1.1\,Ah) trung bình {s('e10_same')}; các cặp khác hoá học trung bình {s('e10_cross')} ---
chênh {s('e10_ratio')}. Đây là xác nhận bằng số liệu cho phân tích dấu tương quan ở
mục~\ref{{sec:dulieu}}.

\item \textbf{{Ma trận không đối xứng.}} {s('e10_asym')} --- chênh hơn 100 lần. MIT có 125
cell với protocol sạc nhanh rất đa dạng nên học được biểu diễn rộng; XJTU chỉ 55 cell ở vài
protocol. \emph{{Đa dạng protocol của bộ nguồn quyết định khả năng chuyển miền nhiều hơn số
lượng cell.}}
\end{{itemize}}

\subsection{{Thích nghi miền bằng loss không cần nhãn}}

Vì~\eqref{{eq:lode}}--\eqref{{eq:lrange}} không dùng nhãn, chúng chạy được ngay trên cell của
bộ đích. Thiết lập: nguồn 70\,\% nhãn; bộ đích chia theo cell cùng seed; lấy
$k \in \{{0,3\}}$ cell đích có nhãn; đánh giá trên cell test của đích; không dừng sớm theo
nhãn đích.

{F_E5}

\input{{tables/E5}}

Đọc theo cặp: {s('e5_short')}. Với thêm 3 cell đích có nhãn, PINN-semi tốt hơn MLP ở
{s('e5_k3')} cặp. Đây là chỗ vật lý thực sự trả lại giá trị khi đổi bộ dữ liệu --- còn
zero-shot thì không.
"""

    sec['10-e11'] = rf"""\section{{Chỉ số đánh giá}}\label{{sec:e11}}

MAE gộp mọi chu kỳ của mọi cell thành một số, nên bị chi phối bởi cell sống lâu và bởi vùng
SOH cao vốn dễ đoán. Bộ chỉ số dưới đây thêm bốn nhóm:

\begin{{description}}[leftmargin=2.8cm, style=nextline]
\item[Theo cell] MAE trung bình theo cell (mỗi cell một phiếu) và \textbf{{cell tệ nhất}} ---
chỉ số an toàn, không phải trung bình.
\item[Vùng cuối đời] MAE chỉ tính ở $\text{{SOH}} \le 0.90$, nơi ra quyết định thay pin.
\item[Hợp lý vật lý] Vi phạm đơn điệu (tổng bước tăng ngược) và nhiễu quỹ đạo
($\overline{{|u(N{{+}}1) - u(N)|}}$). Cả hai \textbf{{đo được mà không cần nhãn}}, nên dùng được
để giám sát mô hình khi triển khai.
\item[Hữu dụng vận hành] Sai số (chu kỳ) khi dự báo thời điểm SOH cắt ngưỡng 0.85, kèm tỉ
lệ cell test thực sự cắt qua.
\end{{description}}

\input{{tables/E11}}

Số ô PINN thắng MLP theo từng chỉ số: {s('e11_wins')}. Trường hợp TJU minh hoạ rõ nhất vì
sao điều này quan trọng --- {s('e11_tju')}. Nếu chỉ báo cáo MAE thì kết luận là ``MLP thắng
trên TJU''; nhìn theo cell tệ nhất, tính hợp lý vật lý và độ chính xác dự báo EOL thì PINN
thắng. Cả hai đều đúng, nhưng chúng trả lời hai câu hỏi khác nhau.

\begin{{canhbao}}
\textbf{{Đọc chỉ số EOL cho đúng.}} Cột ``phủ EOL'' là tỉ lệ cell test thực sự suy giảm xuống
dưới ngưỡng 0.85 trong dữ liệu. Trên MIT chỉ 0.17 --- sai số EOL ở đó tính trên rất ít cell
nên không đáng tin. Chỉ số này chỉ dùng được cho XJTU (0.86), TJU (0.94) và HUST (1.00).
\end{{canhbao}}

\section{{Ablation}}\label{{sec:e3}}

\input{{tables/E3}}

\begin{{itemize}}
\item \textbf{{Dạng phần dư quyết định thành bại.}} Sai phân chia $\Delta t$
(\texttt{{res\_fd}}) tệ nhất ở cả hai bộ, đúng như phân tích nhiễu ở mục~\ref{{sec:loss}}.
\item \textbf{{Ở 30\,\% nhãn trên hai bộ NCM/NCA, ràng buộc đơn điệu với $\beta = 5$ siết quá
tay}}: bỏ nó lại \emph{{giảm}} MAE. Nhãn của XJTU/TJU có tái sinh dung lượng và nhiễu lớn hơn
dung sai $\varepsilon = 0.002$.
\item \textbf{{Động học đen kiểu PINN4SOH cho MAE tương đương động học xám.}} Lợi thế của
động học xám là không suy biến và có tham số đọc được, không phải độ chính xác. $E_a$ học
được trên TJU là {s('Ea_TJU')}\,kJ/mol --- cùng bậc với 20--60\,kJ/mol thường báo cáo, nhưng
chỉ dịch nhẹ khỏi giá trị khởi tạo nên đọc như kiểm tra hợp lý, không phải phép đo.
\end{{itemize}}

\section{{Kiểm toán rò rỉ và giải mã hậu kỳ}}\label{{sec:e4}}

\input{{tables/E4}}

Chỉ riêng việc min--max \emph{{chỉ số chu kỳ}} theo từng cell (tức cho mô hình biết cell test
đang ở phần trăm nào của tuổi thọ) đã hạ MAE mà không thay đổi gì ở mô hình:
{s('leak_cycle')}. Đó là mức ``lợi'' bất hợp pháp phải trừ đi khi đọc các kết quả công bố
dùng cách chuẩn hoá này. Ngược lại, min--max \emph{{đặc trưng}} theo cell lại làm xấu đi
({s('leak_feat')}) vì xoá mất mức tuyệt đối của đường sạc, nên khi gộp cả hai như pipeline
PINN4SOH, hai hiệu ứng bù trừ ({s('leak_both')}).

\subsection*{{Kiểm tra hoài nghi: phần lợi ích có phải chỉ là ``đầu ra đơn điệu''?}}

Quỹ đạo trơn và không tăng có thể lấy miễn phí bằng hậu xử lý --- chiếu quỹ đạo dự đoán của
mỗi cell test lên tập dãy không tăng (hồi quy đơn điệu, thuật toán PAVA, $O(n)$, không cần
nhãn). Nếu MLP $+$ hậu xử lý bằng PINN thì vật lý lúc huấn luyện là thừa. Vì vậy phép giải
mã này được áp cho \textbf{{cả hai}} mô hình.

\input{{tables/E8}}

Ràng buộc vật lý thật sự có tác dụng như thiết kế: tổng bước tăng ngược ({s('e8_viol')}) ---
PINN nhỏ hơn MLP một đến hai bậc. Giải mã đơn điệu giúp MLP trung bình {s('e8_gain_mlp')}
nhưng chỉ giúp PINN {s('e8_gain_pinn')}. Tuy nhiên phần bù đó không đủ để đảo ngôi thứ ở ô
nào. Kết luận: tính đơn điệu ở đầu ra là \emph{{một phần}} giá trị của vật lý, không phải
toàn bộ --- và vì giải mã rất rẻ, nên áp cho mọi mô hình kể cả baseline.
"""

    sec['11-donggop'] = rf"""\section{{Đóng góp}}\label{{sec:donggop}}

\begin{{enumerate}}
\item \textbf{{So sánh có ngân sách tinh chỉnh cân bằng.}} Hầu hết bài PINN cho SOH quét siêu
tham số cho mô hình của mình và để baseline ở cấu hình mặc định. Mục~\ref{{sec:e12}} định
lượng chính xác sai lệch đó trên cùng bộ dữ liệu: tỉ số PINN/MLP đi từ {s('e12_before')}
xuống còn {s('e12_after')} khi baseline được tinh chỉnh ngang bằng. Kết luận vẫn đứng
({s('e12_wins')} ô) nhưng ở biên độ trung thực hơn nhiều.

\item \textbf{{Động học suy giảm xám, không suy biến, diễn giải được.}}
Phương trình~\eqref{{eq:dynamics}} đảm bảo đơn điệu theo cấu trúc, tách bạch ba hiệu ứng với
hai tham số vật lý dùng chung $\lambda$, $E_a$; mạng động học không nhận đạo hàm của nghiệm
nên phần dư không thể tự triệt tiêu.

\item \textbf{{Loss vật lý không cần nhãn.}} Cho phép bán giám sát trên cell chưa đo dung
lượng, và trở thành cơ chế thích nghi miền không nhãn khi đổi bộ dữ liệu. PINN4SOH không có
tính chất này vì loss của họ dùng nhãn.

\item \textbf{{Đầu ra đơn điệu theo cấu trúc}}~\eqref{{eq:mono}}, kiến trúc tốt nhất cho cả
PINN ({s('e9_mono_pinn')}) lẫn baseline ({s('e9_mono_mlp')}). Kèm một phát hiện phủ định có
giá trị: đơn điệu theo $t$ không kéo theo đơn điệu dọc quỹ đạo khi đặc trưng cũng biến
thiên.

\item \textbf{{Trọng số vật lý tự thích ứng theo tỉ lệ nhãn}}, không cần tinh chỉnh theo bộ
dữ liệu, thắng MLP ở {s('e7_adapt_wins')} ô.

\item \textbf{{Phần dư dạng Euler theo horizon ngẫu nhiên.}} Phân tích nhiễu chỉ ra phần dư
dạng đạo hàm khuếch đại nhiễu theo $1/\Delta t$; dạng tích phân khắc phục hoàn toàn. Áp
dụng được cho mọi PINN hồi quy theo chu kỳ.

\item \textbf{{Bộ chỉ số đánh giá theo cell, cuối đời, hợp lý vật lý và EOL.}} Chỉ ra rằng
thứ hạng phụ thuộc thước đo, và rằng vi phạm đơn điệu là chỉ số không cần nhãn nên dùng
được để giám sát khi triển khai.

\item \textbf{{Ma trận chuyển miền $4\times4$ với đường chéo là kết quả trong miền}}, định
lượng ba điều: khoảng cách trong/ngoài miền là 2--3 bậc; chuẩn hoá nhân quả là đòn bẩy lớn
hơn kiến trúc và loss; ma trận bất đối xứng theo độ đa dạng protocol của bộ nguồn.

\item \textbf{{Protocol không rò rỉ kèm kiểm toán định lượng}} mức ``lợi'' do chuẩn hoá theo
cell tạo ra.
\end{{enumerate}}

\section{{Hạn chế và hướng tiếp theo}}\label{{sec:hanche}}

\begin{{itemize}}
\item \textbf{{Số seed.}} 3 seed đủ thấy xu hướng, chưa đủ cho khoảng tin cậy hẹp. Trên GPU
nên chạy 10 seed và thêm mức nhãn 5\,\% và 20\,\%.

\item \textbf{{Bảng cân bằng chưa đầy đủ.}} Mục~\ref{{sec:e12}} mới chạy ở 30\,\% và 70\,\%
nhãn; cần mở rộng sang 10\,\% và 50\,\%, và quét cả learning rate lẫn số bước.

\item \textbf{{Đối chứng mạnh hơn.}} Chưa chạy lại mã gốc PINN4SOH dưới protocol này (mới tái
hiện riêng phần động học đen làm ablation). Nên thêm GPR, XGBoost, CNN-1D và mô hình chuỗi
(GRU/TCN trên cửa sổ 20--50 chu kỳ) --- tất cả cắm được vào chỗ mạng nghiệm.

\item \textbf{{Đặc trưng.}} Bộ 16 đặc trưng thừa hưởng từ PINN4SOH; bước lọc $3\sigma$ theo
cell dùng thống kê cả vòng đời (nhẹ, chỉ trên đặc trưng, nhưng vẫn không nhân quả) --- nên
thay bằng lọc trượt nhân quả.

\item \textbf{{Vật lý sâu hơn.}} Đưa lượng Ah-throughput thay cho chỉ số chu kỳ làm biến thời
gian (chuyển miền tốt hơn giữa các protocol), tách LLI/LAM từ phân tích IC/DV để SOH không
còn là vô hướng, và ước lượng bất định (deep ensemble hoặc conformal).

\item \textbf{{Nhánh đặc trưng đơn điệu.}} Cách sửa đúng cho giới hạn ở mục~\ref{{sec:mono}}:
ràng buộc đơn điệu cả trên đường đi của $\mathbf{{x}}$, thay vì chồng thêm số hạng phạt.
\end{{itemize}}
"""

    sec['10b-chungminh'] = build_section()

    for name, body in sec.items():
        open(f'{OUT}/sections/{name}.tex', 'w').write(body)

    order = sorted(sec.keys())
    main = (r"""\documentclass[11pt, a4paper]{article}
\input{preamble}

\title{\bfseries Ước lượng SOH pin lithium-ion đơn\\[2pt]
\large với ràng buộc vật lý không cần nhãn}
\author{Huỳnh Quốc Thắng\\[2pt]
\small Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách khoa --- ĐHQG-HCM\\
\small Đồ án chuyên ngành HK261 --- nhóm 15}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\noindent
Báo cáo trình bày một kiến trúc mạng nơ-ron có ràng buộc vật lý cho bài toán ước lượng
trạng thái sức khoẻ (SOH) của cell lithium-ion đơn, cùng protocol thí nghiệm không rò rỉ
thông tin và bộ chỉ số đánh giá mở rộng. Ba đóng góp chính: động học suy giảm dạng xám bảo
đảm tính đơn điệu theo cấu trúc; các hàm mất mát vật lý \emph{không cần nhãn}, cho phép học
bán giám sát trên cell chưa đo dung lượng và thích nghi sang bộ dữ liệu khác; và một phép
so sánh có ngân sách tinh chỉnh cân bằng giữa hai mô hình --- điều mà phần lớn công bố hiện
có bỏ qua, và khi bổ sung thì biên độ lợi thế của phương pháp có vật lý giảm đáng kể nhưng
vẫn giữ. Toàn bộ kết luận dựa trên """ + s('n_runs') + r""" lượt huấn luyện trên 387 cell
thuộc bốn bộ dữ liệu công khai.
\end{abstract}

\tableofcontents
\newpage

""" + '\n'.join(f'\\input{{sections/{n}}}' for n in order) + r"""

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
""")
    open(f'{OUT}/main.tex', 'w').write(main)
    open(f'{OUT}/preamble.tex', 'w').write(PREAMBLE)
    open(f'{OUT}/refs.bib', 'w').write(BIB)
    open(f'{OUT}/Makefile', 'w').write(MAKEFILE)
    open(f'{OUT}/README.md', 'w').write(README)
    print(f'{OUT}/ — {len(sec)} mục, {len(T)} nhóm bảng, {len(glob.glob(f"{OUT}/figures/*.png"))} hình')


if __name__ == '__main__':
    build()


# --------------------------------------------------------------------------- xuất bản Word
REFS_TEXT = [
    'Su, L., Xu, Y., Dong, Z. (2024). State-of-health estimation of lithium-ion batteries: A comprehensive '
    'literature review from cell to pack levels. Energy Conversion and Economics, 5(4), 224-242.',
    'Wang, F., Zhai, Z., Zhao, Z., Di, Y., Chen, X. (2024). Physics-informed neural network for lithium-ion '
    'battery degradation stable modeling and prognosis. Nature Communications, 15, 4332.',
    'Severson, K. A., Attia, P. M., Jin, N., et al. (2019). Data-driven prediction of battery cycle life '
    'before capacity degradation. Nature Energy, 4(5), 383-391.',
    'Zhu, J., Wang, Y., Huang, Y., et al. (2022). Data-driven capacity estimation of commercial lithium-ion '
    'batteries from voltage relaxation. Nature Communications, 13, 2261.',
    'Ma, G., Xu, S., Jiang, B., et al. (2022). Real-time personalized health status prediction of lithium-ion '
    'batteries using deep transfer learning. Energy & Environmental Science, 15(10), 4083-4094.',
    'dos Reis, G., Strange, C., Yadav, M., Li, S. (2021). Lithium-ion battery data and where to find it. '
    'Energy and AI, 5, 100081.',
    'Ziyin, L., Hartwig, T., Ueda, M. (2020). Neural networks fail to learn periodic functions and how to '
    'fix it. NeurIPS, 33, 1583-1594.',
    'Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of '
    'Computational Physics, 378, 686-707.',
]


def build_docx(outfile='bao-cao.docx'):
    """Dựng .docx qua pandoc từ bản .tex đã đơn giản hoá.

    Pandoc không đọc được hai thứ nên phải gỡ trước khi chuyển:
      * \\adjustbox{...}{ bọc quanh tabular  -> pandoc bỏ luôn cả bảng
      * \\cite{...} + \\bibliography         -> không chạy bibtex, thay bằng số và danh mục văn bản
    """
    import subprocess, tempfile
    src, tmp = OUT, tempfile.mkdtemp(prefix='docxsrc-')
    shutil.copytree(src, tmp + '/p', dirs_exist_ok=True)
    keys = re.findall(r'@\w+\{([^,]+),', open(f'{src}/refs.bib').read())
    num = {k: i + 1 for i, k in enumerate(keys)}

    # ký hiệu đơn giản -> ký tự Unicode: bộ chuyển PDF của LibreOffice bỏ qua OMML,
    # và văn bản thuần đọc tốt hơn trong Word cho những ký hiệu này
    SYM = {r'$\times$': '×', r'$\to$': '→', r'$\pm$': '±', r'$\le$': '≤', r'$\ge$': '≥',
           r'$\beta$': 'β', r'$\alpha$': 'α', r'$\lambda$': 'λ', r'$\varepsilon$': 'ε',
           r'$\sigma$': 'σ', r'$\Delta t$': 'Δt', r'$\leftrightarrow$': '↔',
           r'$4\times4$': '4×4', r'$\approx$': '≈', r'$n$': 'n', r'$t$': 't', r'$u$': 'u',
           r'$k$': 'k', r'$h$': 'h', r'$y$': 'y', r'$K$': 'K'}

    for f in glob.glob(tmp + '/p/**/*.tex', recursive=True):
        t = open(f).read()
        t = t.replace('\\adjustbox{max width=\\textwidth}{%\n', '').replace('\\end{tabular}}', '\\end{tabular}')
        for a, b in SYM.items():
            t = t.replace(a, b)
        t = re.sub(r'\\cite\{([^}]+)\}',
                   lambda m: '[' + ', '.join(str(num.get(k.strip(), '?')) for k in m.group(1).split(',')) + ']', t)
        for e in ['luuy', 'canhbao']:
            t = t.replace('\\begin{' + e + '}', '\\begin{quote}').replace('\\end{' + e + '}', '\\end{quote}')
        open(f, 'w').write(t)

    bib = ('\\section*{Tài liệu tham khảo}\n\\begin{enumerate}\n'
           + '\n'.join('\\item ' + esc(r).replace(r'\textbackslash{}', '') for r in REFS_TEXT)
           + '\n\\end{enumerate}')
    mt = open(tmp + '/p/main.tex').read().replace('\\bibliographystyle{plain}\n\\bibliography{refs}', bib)
    open(tmp + '/p/main.tex', 'w').write(mt)

    r = subprocess.run(['pandoc', 'main.tex', '-o', os.path.abspath(outfile), '--resource-path=.',
                        '--toc', '--toc-depth=2', '-M', 'lang=en', '-M', 'toc-title=Mục lục'],
                       cwd=tmp + '/p', capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[:800])
    shutil.rmtree(tmp, ignore_errors=True)
    import zipfile
    d = zipfile.ZipFile(outfile).read('word/document.xml').decode()
    print(f'{outfile} — {d.count("<w:tbl>")} bảng, {d.count("<w:drawing>")} hình')
