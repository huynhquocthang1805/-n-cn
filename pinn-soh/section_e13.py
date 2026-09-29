"""
Sinh mục "Thí nghiệm xác nhận E13" (kèm E14) từ results/E13_*.csv, E14_table.csv, verify_pipeline.csv.

    python section_e13.py        # -> tai-lieu/ket-qua-E13.md  (bản Markdown, đọc trên GitHub)
    make_latex.py import build_section() để chèn mục này vào báo cáo LaTeX.

Không gõ tay số nào: mọi con số và mọi câu kết luận có điều kiện đều tính từ CSV. Câu chữ kết luận
đi theo đúng quy tắc quyết định đã ấn định ở tai-lieu/de-cuong-E13.md.
"""
from __future__ import annotations
import os, subprocess
import numpy as np
import pandas as pd

R = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
CHEM = {'XJTU': 'NCM', 'TJU': 'NCA/NCM', 'MIT': 'LFP', 'HUST': 'LFP'}
ARM = {'A': 'MLP 30 %', 'B': 'PINN-sup 30 %', 'C': 'PINN-semi 30 %', 'D': 'chỉ $L_\\text{mono}$ 30 %',
       'E': 'PINN-semi Euler 30 %', 'F': 'MLP 70 %', 'G': 'PINN-semi 70 %', 'M10': 'MLP 10 %',
       'P10': 'PINN-semi 10 %', 'M50': 'MLP 50 %', 'P50': 'PINN-semi 50 %'}
PREREG_COMMIT = '543dbd0'


def f(x, n=3):
    """Số kiểu Việt: dấu phẩy thập phân."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return f'{x:.{n}f}'.replace('.', ',')


def pval(p):
    if not np.isfinite(p):
        return '—'
    return '< 0,0001' if p < 1e-4 else f(p, 4)


def pct(x, n=0):
    return f(100 * x, n) + ' %'


def v1_over(ver) -> str:
    """Tỉ lệ cặp v1 có khoảng cách chu kỳ > h, đọc từ verify_pipeline.csv."""
    import re
    if ver is None:
        return '—'
    r = ver[ver.phep_kiem.str.startswith('đối chứng v1: tỉ lệ cặp')]
    m = re.search(r'v1 ([0-9.]+) %', r.chi_tiet.iloc[0]) if len(r) else None
    return (m.group(1).replace('.', ',') + ' %') if m else '—'


def load():
    d = dict(tests=pd.read_csv(f'{R}/E13_tests.csv'), cur=pd.read_csv(f'{R}/E13_curve.csv'),
             sec=pd.read_csv(f'{R}/E13_secondary.csv'), fold=pd.read_csv(f'{R}/E13_perfold.csv'))
    d['ver'] = pd.read_csv(f'{R}/verify_pipeline.csv') if os.path.exists(f'{R}/verify_pipeline.csv') else None
    d['e14'] = pd.read_csv(f'{R}/E14_table.csv') if os.path.exists(f'{R}/E14_table.csv') else None
    d['traj'] = pd.read_csv(f'{R}/E13_traj_cells.csv') if os.path.exists(f'{R}/E13_traj_cells.csv') else None
    return d


def _row(t, key, ds):
    r = t[(t.so_sanh == key) & (t.bo == ds)]
    return None if r.empty else r.iloc[0]


# ─────────────────────────────────────────── chú thích hình: KHÔNG vẽ lên ảnh, đặt ở caption / chân slide
CAP_FOREST = ('Chấm = tỉ số MAE theo cell trung bình; vạch ngang = KTC 95 % bootstrap theo cell (H2: KTC 90 %). '
              'Xanh = tốt hơn có ý nghĩa sau Holm; cam = tệ hơn có ý nghĩa; xám = chưa đủ bằng chứng. '
              'Vạch dọc liền tại 1 = ngang nhau; vạch đứt tại 1,10 = biên không kém hơn của H2.')
CAP_CURVE = ('Dải = KTC 95 % bootstrap theo cell; vạch đứt ngang = MLP dùng 70 % nhãn (mốc của H2); '
             'trục dọc thang log. Kiểm định chéo 5 fold × 5 lần lặp, pipeline v2.')


def cap_cells(d) -> str:
    """Tỉ lệ cell mà PINN-semi có MAE thấp hơn MLP (30 % nhãn), từng bộ."""
    parts = []
    for ds in DS:
        r = _row(d['tests'], 'C vs A', ds)
        if r is not None:
            parts.append(f'{ds} {pct(r.cell_b_tot_hon)}')
    return ('Mỗi chấm là một cell (MAE trung bình 5 lần lặp); dưới đường chéo, màu xanh lục = PINN-semi sai số '
            'thấp hơn. Tỉ lệ cell PINN-semi thấp hơn: ' + ', '.join(parts) + '.')


def cap_traj(d) -> str:
    """Cell được vẽ quỹ đạo và MAE của từng nhánh trên cell đó."""
    t = d.get('traj')
    if t is None or t.empty:
        return 'Cell có MAE của MLP 30 % đúng trung vị mỗi bộ, lần lặp 0.'
    parts = []
    for _, r in t.iterrows():
        v = ' / '.join(f(1e3 * r[c], 1) if c in r and np.isfinite(r[c]) else '—' for c in ['mae_A', 'mae_C', 'mae_F'])
        parts.append(f'{r.bo} {r.cell.split("/")[-1]}: {v}')
    return ('Cell có MAE của MLP 30 % đúng trung vị mỗi bộ (quy tắc chọn không phụ thuộc PINN), lần lặp 0. '
            'MAE ×10⁻³ (MLP 30 % / PINN-semi 30 % / MLP 70 %) — ' + '; '.join(parts) + '.')


# ─────────────────────────────────────────────────────── các khối nội dung (Markdown trung lập)
def facts(d):
    """Các con số và phán quyết dùng chung cho Markdown, LaTeX và slide."""
    t, fl = d['tests'], d['fold']
    F = {}
    F['n_runs'] = int(len(fl))
    F['n_arms'] = int(fl.arm.nunique())
    F['repeats'] = int(fl.repeat.nunique()); F['folds'] = int(fl.fold.nunique())
    h1 = {ds: _row(t, 'C vs A', ds) for ds in DS}
    h2 = {ds: _row(t, 'C vs F', ds) for ds in DS}
    F['h1'] = h1; F['h2'] = h2
    F['h1_win'] = [ds for ds, r in h1.items() if r is not None and r.p_holm < 0.05 and r.trung_vi_hieu < 0]
    F['h1_lose'] = [ds for ds, r in h1.items() if r is not None and r.p_holm < 0.05 and r.trung_vi_hieu > 0]
    F['h1_nb_win'] = [ds for ds, r in h1.items() if r is not None and r.p_nb_holm < 0.05 and r.ti_so < 1]
    F['h2_ok'] = [ds for ds, r in h2.items() if r is not None and str(r.ket_luan).startswith('không kém')]
    F['h2_nb_ok'] = [ds for ds, r in h2.items() if r is not None and str(r.ket_luan_nb).startswith('không kém')]
    return F


def table_h1(d, F, key='C vs A'):
    rows = []
    for ds in DS:
        r = _row(d['tests'], key, ds)
        if r is None:
            continue
        rows.append([f'{ds} ({CHEM[ds]})', str(int(r.n_cell)), f(1e3 * r.mae_a, 2), f(1e3 * r.mae_b, 2),
                     f'{f(r.ti_so)} [{f(r.ci_lo)}; {f(r.ci_hi)}]', pct(r.cell_b_tot_hon),
                     pval(r.p_holm), pval(r.p_nb_holm), r.ket_luan])
    return rows


def table_curve(d):
    c = d['cur']
    rows = []
    for ds in DS:
        g = c[c.bo == ds]
        if g.empty:
            continue
        row = [ds]
        for fr in [0.1, 0.3, 0.5, 0.7]:
            m = g[(g.model == 'mlp') & (g.frac == fr)].MAE_cell
            p = g[(g.model == 'pinn_semi') & (g.frac == fr)].MAE_cell
            row.append(f'{f(1e3 * m.iloc[0], 2)} / {f(1e3 * p.iloc[0], 2)}' if len(m) and len(p) else '—')
        rows.append(row)
    return rows


def table_secondary_tests(d):
    t = d['tests']
    name = {'B vs A': 'PINN-sup / MLP (regularization)', 'C vs B': 'PINN-semi / PINN-sup (cell không nhãn)',
            'D vs C': 'chỉ L_mono / PINN-semi (ích lợi ODE)', 'E vs C': 'Euler dọc quỹ đạo / autograd',
            'G vs F': 'PINN-semi / MLP ở 70 %', 'P10 vs M10': 'PINN-semi / MLP ở 10 %',
            'P50 vs M50': 'PINN-semi / MLP ở 50 %'}
    rows = []
    for key, lab in name.items():
        cells = []
        for ds in DS:
            r = _row(t, key, ds)
            if r is None:
                cells.append('—'); continue
            mark = '*' if r.p_holm < 0.05 else ''
            cells.append(f'{f(r.ti_so)}{mark}')
        rows.append([lab] + cells)
    return rows


def table_metrics(d):
    s = d['sec']
    rows = []
    for ds in DS:
        for arm in ['A', 'C', 'F']:
            g = s[(s.bo == ds) & (s.nhanh == arm)]
            if g.empty:
                continue
            g = g.iloc[0]
            rows.append([ds, ARM[arm].replace('$L_\\text{mono}$', 'L_mono'), f(1e3 * g.MAE_cell, 2), f(1e3 * g.MAE_late, 2),
                         f(g.EOL_MAE, 1), f'{int(g.EOL_bo_sot)} / {int(g.EOL_bao_gia)}',
                         f(g.MonoViol100, 3), f(1e3 * g.Jitter, 2)])
    return rows


def table_e14(d):
    w = d['e14']
    if w is None:
        return []
    rows = []
    for _, r in w.sort_values(['cap', 'k']).iterrows():
        rows.append([r.cap, str(int(r.k)), f'{f(r.mlp, 4)} ± {f(r.sd_mlp, 4)}',
                     f'{f(r.pinn_semi, 4)} ± {f(r.sd_pinn, 4)}', f(r.ti_so), f'{int(r.seed_pinn_thang)}/{int(r.n_seed)}'])
    return rows


def conclusion_lines(d, F):
    """Kết luận theo đúng quy tắc quyết định ấn định trước."""
    L = []
    n = len(DS)
    win, lose, nbw = F['h1_win'], F['h1_lose'], F['h1_nb_win']
    if win:
        L.append(f'H1 (cùng 30 % nhãn): PINN-semi có MAE theo cell thấp hơn MLP, có ý nghĩa sau hiệu chỉnh Holm, '
                 f'trên {len(win)}/{n} bộ ({", ".join(win)}).')
    else:
        L.append('H1 (cùng 30 % nhãn): chưa bộ nào cho thấy PINN-semi tốt hơn MLP có ý nghĩa sau hiệu chỉnh Holm.')
    rest = [ds for ds in DS if ds not in win and ds not in lose]
    if rest:
        L.append(f'Trên {", ".join(rest)} chưa đủ bằng chứng theo hướng nào; p lớn KHÔNG có nghĩa là hai mô hình tương đương.')
    if lose:
        L.append(f'Trên {", ".join(lose)}, MLP tốt hơn PINN-semi có ý nghĩa.')
    if set(nbw) != set(win):
        L.append(f'Phép t hiệu chỉnh ở mức fold (bảo thủ hơn, tính cả biến thiên do huấn luyện lại) chỉ xác nhận '
                 f'{len(nbw)}/{n} bộ' + (f' ({", ".join(nbw)})' if nbw else '') +
                 ' — kết luận phụ thuộc đơn vị phân tích, nên phát biểu mạnh chỉ dành cho các bộ qua cả hai phép kiểm.')
    else:
        L.append('Phép t hiệu chỉnh ở mức fold cho cùng kết luận.')
    ok, nbok = F['h2_ok'], F['h2_nb_ok']
    if ok:
        L.append(f'H2 (PINN-semi 30 % so với MLP 70 %, biên 10 %): không kém hơn trên {len(ok)}/{n} bộ ({", ".join(ok)}).')
    else:
        L.append('H2 (PINN-semi 30 % so với MLP 70 %, biên 10 %): chưa chứng minh được không kém hơn trên bộ nào.')
    miss = [ds for ds in DS if ds not in ok]
    if miss and ok:
        L.append(f'Trên {", ".join(miss)} chưa chứng minh được tiết kiệm nhãn ở biên 10 %.')
    if ok and set(nbok) != set(ok):
        L.append(f'Với phép t mức fold, H2 chỉ đứng vững trên {len(nbok)}/{n} bộ'
                 + (f' ({", ".join(nbok)})' if nbok else '') + '.')
    return L


# ─────────────────────────────────────────────────────── Markdown
def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def build_markdown(d) -> str:
    F = facts(d)
    ver = d['ver']
    nver = f'{int(ver.dat.sum())}/{len(ver)}' if ver is not None else '—'
    tests = d['tests']
    md = []
    md.append('# Kết quả thí nghiệm xác nhận E13 (pipeline v2) và E14\n')
    md.append('_Sinh tự động bởi `section_e13.py` từ `results/E13_*.csv` — không có số nào gõ tay._\n')
    md.append('## 1. Vì sao cần E13\n')
    md.append('Bản rà soát ngày 29/09 chỉ ra năm điểm làm phép so MLP–PINN chưa công bằng hoặc chưa đúng '
              '(slide 16) và đề ra khung kết quả còn thiếu (slide 18). E13 sửa cả năm điểm rồi chạy lại phép so '
              'dưới một đề cương **ấn định trước**: `tai-lieu/de-cuong-E13.md`, commit '
              f'`{PREREG_COMMIT}`, đẩy lên git trước lượt huấn luyện đầu tiên.\n')
    md.append(md_table(['#', 'Vấn đề (v1)', 'Sửa trong v2', 'Kiểm chứng'], [
        ['1', 'Lọc 3σ dùng thống kê cả vòng đời cell', 'Cửa sổ 50 chu kỳ quá khứ, z bền vững > 5',
         'phá dữ liệu tương lai: 0 quyết định quá khứ đổi'],
        ['2', '`dropna()` loại hàng thiếu dung lượng', 'Chỉ loại khi đặc trưng không hữu hạn; nhãn NaN giữ lại',
         'CSV không cột capacity nạp đủ hàng'],
        ['3', 'MLP, PINN fit normaliser trên pool khác nhau', 'Mọi mô hình fit trên đặc trưng mọi cell train',
         'mu, sd trùng từng bit giữa 3 mô hình'],
        ['4', '`h` đếm theo hàng sau lọc', 'Cặp (N, N+h) theo chu kỳ gốc', f'v1: {v1_over(ver)} cặp có Δchu kỳ > h'],
        ['5', 'EOL theo chỉ số hàng, trộn quan sát bị kiểm duyệt', 'EOL theo chu kỳ, tách bỏ sót / báo giả',
         'khớp đáp án tính tay'],
    ]))
    md.append(f'\n`verify_pipeline.py`: **{nver} phép kiểm đạt**. Chế độ v1 vẫn là mặc định và tái lập **từng bit** '
              'dự đoán của các log cũ (đã kiểm trên XJTU và TJU, cấu hình TUNED seed 0).\n')

    md.append('## 2. Thiết kế\n')
    md.append(f'* Kiểm định chéo **{F["folds"]} fold theo cell × {F["repeats"]} lần lặp**, phân tầng theo batch: '
              'mỗi cell nằm trong test đúng một lần mỗi lần lặp → mỗi cell có đủ dự đoán test dưới mọi nhánh, '
              'ghép cặp theo cell được. Mỗi fold: 70 % train, 10 % validation, 20 % test.\n'
              '* Validation cần nhãn **ngoài** ngân sách 30 %; giống nhau cho mọi nhánh.\n'
              '* Cùng mạng (128-128-64, SiLU), cùng tối ưu, cùng dừng sớm theo validation. Khác nhau **chỉ ở hàm mất mát**.\n'
              f'* {F["n_arms"]} nhánh, **{F["n_runs"]} lượt huấn luyện**. Chỉ số chính: MAE theo cell, '
              'trung bình qua các lần lặp; đơn vị kiểm định: cell (Wilcoxon, Holm cho 4 bộ). Độ nhạy: phép t '
              'hiệu chỉnh Nadeau–Bengio ở mức fold.\n')

    md.append('## 3. H1 — PINN-semi so với MLP, cùng 30 % nhãn\n')
    md.append(md_table(['Bộ', 'Cell', 'MLP (×10⁻³)', 'PINN-semi (×10⁻³)', 'Tỉ số [KTC 95 %]', 'Cell PINN tốt hơn',
                        'p Holm (cell)', 'p Holm (fold, NB)', 'Kết luận'], table_h1(d, F)))
    md.append(f'\n![](ket-qua-E13/fig_E13_forest.png)\n\n_{CAP_FOREST}_\n')
    md.append(f'\n![](ket-qua-E13/fig_E13_cells.png)\n\n_{cap_cells(d)}_\n')

    md.append('## 4. H2 — tiết kiệm nhãn: PINN-semi 30 % so với MLP 70 %\n')
    rows = []
    for ds in DS:
        r = F['h2'][ds]
        if r is None:
            continue
        rows.append([ds, f(1e3 * r.mae_a, 2), f(1e3 * r.mae_b, 2), f'{f(r.ti_so)} [{f(r.ci_lo)}; {f(r.ci_hi)}]',
                     pval(r.p_holm), pval(r.p_nb_holm), r.ket_luan])
    md.append(md_table(['Bộ', 'MLP 70 % (×10⁻³)', 'PINN-semi 30 % (×10⁻³)', 'Tỉ số [KTC 90 %]', 'p Holm (cell)',
                        'p Holm (fold, NB)', 'Kết luận (δ = 10 %)'], rows))
    md.append('\nĐường cong theo lượng nhãn — MAE theo cell ×10⁻³, **MLP / PINN-semi**:\n')
    md.append(md_table(['Bộ', '10 %', '30 %', '50 %', '70 %'], table_curve(d)))
    md.append(f'\n![](ket-qua-E13/fig_E13_curve.png)\n\n_{CAP_CURVE}_\n')

    md.append('## 5. So sánh phụ (thăm dò) — tỉ số MAE theo cell, * = p Holm < 0,05\n')
    md.append(md_table(['So sánh'] + DS, table_secondary_tests(d)))

    md.append('\n## 6. Chỉ số phụ — sai số EOL, tính hợp lý của quỹ đạo\n')
    md.append(md_table(['Bộ', 'Nhánh', 'MAE cell ×10⁻³', 'MAE SOH ≤ 0,9 ×10⁻³', 'EOL MAE (chu kỳ)',
                        'bỏ sót / báo giả', 'vi phạm đơn điệu /100 ck', 'jitter ×10⁻³'], table_metrics(d)))
    md.append('\n_EOL tại ngưỡng SOH 0,85, tính trên chu kỳ gốc; sai số chỉ lấy trên các cặp (cell, lần lặp) '
              'mà cả nhãn lẫn dự đoán cùng cắt ngưỡng; "bỏ sót" = nhãn cắt nhưng dự đoán không cắt._\n')
    md.append(f'\n![](ket-qua-E13/fig_E13_traj.png)\n\n_{cap_traj(d)}_\n')

    if d['e14'] is not None:
        md.append('## 7. E14 — chuyển miền với cùng normaliser\n')
        md.append(md_table(['Nguồn → đích', 'k', 'MLP', 'PINN-semi', 'Tỉ số', 'seed PINN thắng'], table_e14(d)))
        md.append('\n_MAE theo cell trên tập test của bộ đích, trung bình ± SD qua 5 seed. Cả hai mô hình fit '
                  'normaliser trên cùng pool đặc trưng nguồn + đích; chỉ khác loss vật lý._\n')

    md.append('## 8. Kết luận theo quy tắc đã ấn định\n')
    md += [f'* {x}' for x in conclusion_lines(d, F)]
    md.append('')
    return '\n'.join(md) + '\n'


# ─────────────────────────────────────────────────────── LaTeX (cho make_latex.py)
def _tex_esc(s: str) -> str:
    return (s.replace('%', r'\%').replace('_', r'\_').replace('#', r'\#').replace('→', r'$\to$').replace('×10⁻³', r'$\times10^{-3}$')
             .replace('≤', r'$\le$').replace('δ', r'$\delta$').replace('±', r'$\pm$').replace('L_mono', r'$L_\text{mono}$'))


def _tex_table(header, rows, spec, caption, label):
    body = '\n'.join(' & '.join(_tex_esc(c) for c in r) + r' \\' for r in rows)
    head = ' & '.join(r'\textbf{' + _tex_esc(h) + '}' for h in header) + r' \\'
    return (r'\begin{table}[htbp]\centering\small' + '\n' + rf'\caption{{{caption}}}\label{{{label}}}' + '\n'
            + r'\begin{adjustbox}{max width=\textwidth}' + '\n' + rf'\begin{{tabular}}{{{spec}}}\toprule' + '\n'
            + head + r'\midrule' + '\n' + body + '\n' + r'\bottomrule\end{tabular}\end{adjustbox}\end{table}' + '\n')


def build_section() -> str:
    """Mục LaTeX; trả chuỗi rỗng nếu chưa có kết quả E13."""
    if not os.path.exists(f'{R}/E13_tests.csv'):
        return ''
    d = load(); F = facts(d)
    ver = d['ver']
    nver = f'{int(ver.dat.sum())}/{len(ver)}' if ver is not None else '—'
    h2rows = []
    for ds in DS:
        r = F['h2'][ds]
        if r is None:
            continue
        h2rows.append([ds, f(1e3 * r.mae_a, 2), f(1e3 * r.mae_b, 2), f'{f(r.ti_so)} [{f(r.ci_lo)}; {f(r.ci_hi)}]',
                       pval(r.p_holm), pval(r.p_nb_holm), r.ket_luan])
    concl = '\n'.join(rf'\item {_tex_esc(x)}' for x in conclusion_lines(d, F))
    s = rf"""\section{{Thí nghiệm xác nhận E13: pipeline v2 và đề cương ấn định trước}}\label{{sec:e13}}

Mục này trả lời hai câu hỏi nghiên cứu bằng một phép so \emph{{công bằng}}: cùng cách chia, cùng
normaliser, cùng lượng nhãn và cùng ngân sách chọn mô hình --- hai mô hình chỉ khác nhau ở hàm mất mát.
Năm điểm yếu của pipeline cũ (lọc ngoại lai dùng thống kê cả vòng đời, \texttt{{dropna}} trên nhãn,
normaliser khác nhau giữa MLP và PINN, horizon đếm theo hàng, EOL theo chỉ số hàng) được sửa trong
pipeline v2 và kiểm bằng {nver} phép kiểm của \texttt{{verify\_pipeline.py}}. Đề cương
(\texttt{{tai-lieu/de-cuong-E13.md}}) được commit (\texttt{{{PREREG_COMMIT}}}) trước lượt huấn luyện đầu tiên.

\paragraph{{Thiết kế.}} Kiểm định chéo {F['folds']} fold theo cell $\times$ {F['repeats']} lần lặp, phân tầng
theo batch; mỗi cell được test đúng một lần mỗi lần lặp nên ghép cặp theo cell được. {F['n_arms']} nhánh,
{F['n_runs']} lượt huấn luyện. Chỉ số chính: MAE theo cell trung bình qua các lần lặp; đơn vị kiểm định là
cell (Wilcoxon, Holm cho bốn bộ). Phép t hiệu chỉnh Nadeau--Bengio ở mức fold là phân tích độ nhạy.

""" + _tex_table(['Bộ', 'Cell', 'MLP (×10⁻³)', 'PINN-semi (×10⁻³)', 'Tỉ số [KTC 95 %]', 'Cell PINN tốt hơn',
                  'p Holm (cell)', 'p Holm (fold)', 'Kết luận'], table_h1(d, F), 'lrrrlrrrl',
                 'H1 --- PINN-semi so với MLP khi cùng 30\\,\\% cell có nhãn (MAE theo cell).', 'tab:e13h1') + \
        r"""
\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{figures/fig_E13_forest.pdf}
\caption{Tỉ số MAE theo cell cho các so sánh của E13. """ + _tex_esc(CAP_FOREST) + r"""}\label{fig:e13forest}\end{figure}

\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{figures/fig_E13_cells.pdf}
\caption{MAE từng cell, MLP so với PINN-semi ở 30\,\% nhãn. """ + _tex_esc(cap_cells(d)) + r"""}\label{fig:e13cells}\end{figure}

""" + _tex_table(['Bộ', 'MLP 70 % (×10⁻³)', 'PINN-semi 30 % (×10⁻³)', 'Tỉ số [KTC 90 %]', 'p Holm (cell)',
                  'p Holm (fold)', 'Kết luận (δ = 10 %)'], h2rows, 'lrrlrrl',
                 'H2 --- tiết kiệm nhãn: PINN-semi 30\\,\\% so với MLP 70\\,\\%, biên không kém hơn 10\\,\\%.',
                 'tab:e13h2') + r"""
\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{figures/fig_E13_curve.pdf}
\caption{MAE theo cell theo lượng nhãn. """ + _tex_esc(CAP_CURVE) + r"""}\label{fig:e13curve}\end{figure}

\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{figures/fig_E13_traj.pdf}
\caption{Quỹ đạo SOH trên cell test. """ + _tex_esc(cap_traj(d)) + r"""}\label{fig:e13traj}\end{figure}

""" + _tex_table(['So sánh'] + DS, table_secondary_tests(d), 'lrrrr',
                 'So sánh phụ (thăm dò): tỉ số MAE theo cell; * = p Holm < 0,05.', 'tab:e13phu') + \
        _tex_table(['Bộ', 'Nhánh', 'MAE cell ×10⁻³', 'MAE SOH ≤ 0,9 ×10⁻³', 'EOL MAE (chu kỳ)', 'bỏ sót / báo giả',
                    'vi phạm đơn điệu', 'jitter ×10⁻³'], table_metrics(d), 'llrrrrrr',
                   'Chỉ số phụ: EOL theo chu kỳ tại ngưỡng 0,85 và tính hợp lý của quỹ đạo.', 'tab:e13metrics') + \
        (_tex_table(['Nguồn → đích', 'k', 'MLP', 'PINN-semi', 'Tỉ số', 'seed PINN thắng'], table_e14(d), 'lrllrr',
                    'E14 --- chuyển miền khi cả hai mô hình dùng cùng normaliser.', 'tab:e14')
         if d['e14'] is not None else '') + rf"""
\paragraph{{Kết luận theo quy tắc đã ấn định.}}
\begin{{itemize}}
{concl}
\end{{itemize}}
"""
    return s


EXPORT = 'tai-lieu/ket-qua-E13'
EXPORT_FILES = ['E13_tests.csv', 'E13_tests.md', 'E13_curve.csv', 'E13_secondary.csv', 'E13_perfold.csv',
                'E13_physics_params.csv', 'E13_traj_cells.csv', 'E14_table.csv', 'verify_pipeline.csv',
                'fig_E13_forest.png', 'fig_E13_curve.png', 'fig_E13_cells.png', 'fig_E13_traj.png']


def export():
    """results/ không lên git; chép hình + bảng nhỏ sang thư mục có theo dõi để đọc được trên GitHub.
    E13_percell.csv (dấu vết kiểm toán, đủ để tính lại mọi số) được nén gzip."""
    import shutil, gzip
    os.makedirs(EXPORT, exist_ok=True)
    for fn in EXPORT_FILES:
        if os.path.exists(f'{R}/{fn}'):
            shutil.copy(f'{R}/{fn}', f'{EXPORT}/{fn}')
    if os.path.exists(f'{R}/E13_percell.csv'):
        with open(f'{R}/E13_percell.csv', 'rb') as a, gzip.open(f'{EXPORT}/E13_percell.csv.gz', 'wb') as b:
            shutil.copyfileobj(a, b)


if __name__ == '__main__':
    d = load()
    md = build_markdown(d)
    os.makedirs('tai-lieu', exist_ok=True)
    open('tai-lieu/ket-qua-E13.md', 'w').write(md)
    export()
    print(f'đã ghi tai-lieu/ket-qua-E13.md và {EXPORT}/')
    for x in conclusion_lines(d, facts(d)):
        print(' -', x)
