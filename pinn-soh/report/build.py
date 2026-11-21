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
  * dự báo (E22): runs/E22 và runs/E20 -> analysis/analyze_e22.py, analysis/figures_report.py; bản sao bảng ở
    report/generated/data/e22_*.csv, hình ở report/graphics/fig_e22_*.pdf;
  * dự báo có đặc trưng lịch sử, HiPINN (E33): runs/E33 -> analysis/analyze_e33.py, analysis/figures_report.py; bản sao
    bảng ở report/generated/data/e33_*.csv, hình ở report/graphics/fig_e33_*.pdf; bảng mô hình cập nhật có đặc trưng
    lịch sử và số của lần chạy trước lấy từ các tệp kết quả trong kho phát triển G-doan (biến môi trường GDOAN).
Các con số trong lời văn của Chương 5 (kết quả từ báo cáo đã nộp) chép từ lời văn của báo cáo đó, trừ mục dự báo
(sinh từ kết quả chạy lại E22) và mục HiPINN (sinh từ E33).
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


def write(name, text):
    os.makedirs(GEN, exist_ok=True)
    open(os.path.join(GEN, name), 'w').write(text.rstrip() + '\n')


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


def thesis_tables():
    write('bang_du_lieu.tex', thesis_block('tab:datastats'))
    write('bang_nhan.tex', thesis_block('tab:big'))
    write('bang_ti_so.tex', thesis_block('tab:cmp'))
    write('bang_nguon_loi.tex', thesis_block('tab:source'))
    write('bang_mae_chinh.tex', thesis_block('tab:allmae', [(
        r'mô hình cuối là PINN-lai dùng hay không dùng lịch sử theo validation (Mục~\ref{sec:histres}).',
        r'mô hình cuối là PINN-lai dùng hay không dùng lịch sử của cell theo validation (Mục~\ref{sec:dt_bo}); cả hai '
        r'là phân tích hậu kiểm.')]))
    write('bang_rmse.tex', thesis_block('tab:allrm'))
    write('bang_ngan_sach.tex', thesis_block('tab:e28'))
    write('bang_du_bao.tex', thesis_block('tab:e22soh') + '\n' + thesis_block('tab:e22rul'))


def history_table():
    """Đặc trưng có lịch sử (E29): bản rút gọn của bảng trong báo cáo đã nộp (PINN-semi và PINN-lai, có/không lịch sử)."""
    rows = table('tab:e29')
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Thêm lịch sử của cell vào đầu vào]{MAE theo cell ($\times10^{-3}$) trên test khi thêm lịch sử của '
           r'cell vào đầu vào, 3 lần lặp $\times$ 5 fold. Dạng lịch sử được chọn trên validation cho từng bộ và loại mô '
           r'hình; trên MIT không dạng nào giảm sai số validation đủ 2\,\% nên không có cột lịch sử. PINN-lai là bản có '
           r'trọng số ghép và cửa sổ lọc chọn trên validation. $^{*}$/$^{\dagger}$: thấp hơn/cao hơn có ý nghĩa '
           r'(Wilcoxon ghép cặp theo cell, $p < 0{,}05$) so với cột không lịch sử ngay trước.}\label{tab:hist}',
           r'\begin{tabular}{llrrrr}\toprule',
           r'Bộ & Nhãn & PINN-semi & + lịch sử & PINN-lai & + lịch sử \\\midrule']
    last = None
    for r in rows:
        cell = lambda x: '--' if x == '--' else (v(x) if isinstance(x, float) else
                                                 x.replace('*', '$^{*}$').replace('†', '$^{\\dagger}$').replace(',', '{,}'))
        lab = r[1].replace('%', r'\,\%')
        out.append(f'{r[0] if r[0] != last else ""} & {lab} & {cell(r[2])} & {cell(r[3])} & {cell(r[5])} & '
                   f'{cell(r[6])} \\\\')
        last = r[0]
    out += [r'\bottomrule\end{tabular}\end{table}']
    # hai hình của báo cáo đã nộp (E29, chỉ ba bộ có dạng lịch sử được chọn, không có NASA)
    out += ['', r'\begin{figure}[!htbp]\centering', r'\includegraphics[width=\textwidth]{graphics/fig_e29_pinn.pdf}',
            r'\caption[PINN-semi khi có và không có lịch sử của cell]{MAE theo cell trên test của PINN-semi không có (nét '
            r'đứt) và có lịch sử của cell (nét liền), trên ba bộ có dạng lịch sử được chọn; trên MIT không dạng nào được '
            r'chọn.}\label{fig:e29pinn}', r'\end{figure}', '',
            r'\begin{figure}[!htbp]\centering', r'\includegraphics[width=\textwidth]{graphics/fig_e29_lai.pdf}',
            r'\caption[PINN-lai khi có và không có lịch sử của cell]{MAE theo cell trên test của PINN-lai tinh chỉnh không '
            r'có (nét đứt) và có lịch sử của cell (nét liền), cạnh GBDT có hoặc không lịch sử chọn trên validation.}'
            r'\label{fig:e29lai}', r'\end{figure}']
    return '\n'.join(out)


# --------------------------------------------------------------------------- bảng thí nghiệm
EXPS = [(['E13'], 'Kiểm định chéo chính: MLP, PINN-semi ở 4 mức nhãn; PINN-sup và 2 biến thể cắt bỏ ở 30\\,\\%'),
        (['E15'], 'Kiểm toán rò rỉ: 4 cách chia và chuẩn hoá sai, MLP 70\\,\\% nhãn'),
        (['E16'], 'Ridge, GPR, GBDT ở 4 mức nhãn'),
        (['E17'], 'PINN-sup ở 10 và 50\\,\\%; PINN hộp đen ở 4 mức nhãn'),
        (['E20dev', 'E20'], 'PINN-lai: 9 hướng cải tiến chọn trên validation; 3 PINN-semi, GBDT, 3 lần lặp, 4 mức nhãn'),
        (['E30dev', 'E30'], 'CNN, LSTM, Transformer: chọn độ dài cửa sổ trên validation; 3 lần lặp, 4 mức nhãn'),
        (['E29dev', 'E29'], 'Đầu vào có lịch sử của cell: chọn dạng trên validation; PINN-semi, MLP, GBDT'),
        (['E28'], 'Ngân sách đo: đo thưa mọi cell train theo 4 lịch đo'),
        (['E32'], 'Mô hình cập nhật từ lần đo gần nhất (GBDT)'),
        (['E22', 'E23'], 'Dự báo SOH, dung lượng, RUL; bản có xét hồi phục dung lượng'),
        (['E33'], 'Dự báo có đặc trưng lịch sử: ba nhánh và HiPINN, 2 ngân sách nhãn, 3 lần lặp'),
        (['E27'], 'Mô hình của đồ án trên đúng 39 nhóm của PINN4SOH, 2 cách tiền xử lý, 3 hạt giống')]


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
    out.append(f'Chạy lại mã gốc PINN4SOH: 39 nhóm, bản công bố 10 lần và bản đã sửa 3 lần mỗi nhóm & {thou(n_rerun)} \\\\')
    tot += n_rerun
    if n_feat_jobs:
        out.append(f'Phân tích đặc trưng: {n_feat_jobs} fold $\\times$ {n_feat_fits // n_feat_jobs}--'
                   f'{-(-n_feat_fits // n_feat_jobs)} lần huấn luyện GBDT với các tập đầu vào khác nhau & {thou(n_feat_fits)} \\\\')
        tot += n_feat_fits
    out += [r'\midrule', f'Tổng & {thou(tot)} \\\\', r'\bottomrule\end{tabularx}\end{table}']
    write('bang_thi_nghiem.tex', '\n'.join(out))
    return tot


# --------------------------------------------------------------------------- kiểm toán rò rỉ
LEAK = [('row', 'chia theo chu kỳ'), ('minmax', 'min--max theo cell'), ('life3s', 'lọc dùng cả vòng đời'),
        ('normtest', 'chuẩn hoá gồm cả test')]


def leakage():
    path = keep_copy('results/thesis_leakage.csv', 'leakage.csv')
    T = pd.read_csv(path).set_index('bo').reindex(DS)
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Kiểm toán rò rỉ]{Kiểm toán rò rỉ: MAE theo cell của cùng một MLP (70\,\% nhãn, 2 lần lặp $\times$ 5 '
           r'fold) theo giao thức không rò rỉ, và tỉ số giữa MAE dưới mỗi cách làm sai với MAE đó (nhỏ hơn 1: sai số trông '
           r'thấp hơn thực tế). Cột cuối: chia theo chu kỳ rồi chép nhãn của chu kỳ train gần nhất trong cùng cell, không '
           r'dùng đặc trưng nào.}\label{tab:leak}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{lrrrrrr}\toprule',
           r'Bộ & MAE ($\times10^{-3}$) & ' + ' & '.join(l for _, l in LEAK) + r' & chép nhãn \\\midrule']
    for ds, r in T.iterrows():
        out.append(f'{ds} & {v(1e3 * r.chuan)} & ' + ' & '.join(v(r[f"ti_so_{s}"]) for s, _ in LEAK)
                   + f' & {v(r.ti_so_noi_suy_hang)} \\\\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    lo = lambda s: T[f'ti_so_{s}'].min(); hi = lambda s: T[f'ti_so_{s}'].max()
    m = T['ti_so_minmax']
    L = dict(row=f'{v(1 / hi("row"), 1)}--{v(1 / lo("row"), 1)}', copy=f'{v(lo("noi_suy_hang"))}--{v(hi("noi_suy_hang"))}',
             minmax=m, life=rng(T['ti_so_life3s']), normtest=rng(T['ti_so_normtest']),
             minmax_rng=f'{v(m.min())}--{v(m.max())}')
    text = (f'Bảng~\\ref{{tab:leak}} và Hình~\\ref{{fig:leak}} cho thấy mức độ của từng dạng rò rỉ khi chỉ đổi đúng một '
            f'yếu tố so với giao thức chuẩn. Chia theo chu kỳ làm sai số thấp đi {L["row"]} lần dù mô hình và dữ liệu '
            f'không đổi. Bộ “dự đoán” không dùng đặc trưng nào, chỉ chép nhãn của chu kỳ train gần nhất, có sai số '
            f'bằng {L["copy"]} lần mô hình thật được đánh giá đúng, đúng như Mệnh đề~\\ref{{md:rowleak}}. Chuẩn hoá '
            f'min--max theo cả vòng đời cell, như trong mã PINN4SOH, làm sai số thấp giả tạo trên XJTU ({v(m["XJTU"])}) '
            f'và HUST ({v(m["HUST"])}), nơi SOH cuối đời của mọi cell gần như bằng nhau nên vị trí trong vòng đời gần '
            f'như đủ để suy ra SOH; trên TJU ({v(m["TJU"])}) và MIT ({v(m["MIT"])}) nó làm sai số tăng vì xoá mất mức '
            f'tuyệt đối của đặc trưng. Dù theo chiều nào, con số đo được cũng không phản ánh sai số khi triển khai, vì '
            f'tuổi thọ của cell đang vận hành chưa biết trước. Lọc ngoại lai dùng cả vòng đời ({L["life"]}) và chuẩn hoá '
            f'gồm cả test ({L["normtest"]}) tác động nhỏ hơn nhưng vẫn là rò rỉ.')
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


def bench_table(S, T):
    mark = {}
    for r in T[(T.a == 'paper_pinn_lai') & (T.b == 'rerun_published')].itertuples():
        if r.p_holm < 0.05:
            mark[(r.dataset, r.mode)] = '*' if r.ratio < 1 else r'\dagger'
    cols = [('published_paper', None), ('rerun_published', 'sd'), ('rerun_current', 'sd'), ('paper_pinn_lai', 'mark'),
            ('paper_gbdt', None), ('strict_pinn_lai', None)]
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption[So với PINN4SOH trên đúng cách chia của bài]{MAE theo cell ($\times10^{-3}$) trên đúng các nhóm '
           r'của PINN4SOH, trung bình qua cell test, lần chạy và nhóm như bài. Chạy lại: mã gốc ở bản đã sinh số công bố '
           r'(siêu tham số theo nhật ký, 10 lần chạy mỗi nhóm) và ở bản hiện tại (loss đơn điệu đã sửa, trọng số mặc định '
           r'mới, 3 lần chạy); $\pm$ là độ lệch chuẩn qua lần chạy. Mô hình của đồ án: 3 hạt giống. $K$: số cell có nhãn '
           r'trong thí nghiệm ít mẫu. $^{*}$/$^{\dagger}$: PINN-lai với tiền xử lý của bài thấp hơn/cao hơn có ý nghĩa so '
           r'với mã gốc chạy lại bản công bố (Wilcoxon theo cell test, Holm qua 12 hàng).}\label{tab:bench}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{llrrrrrr}\toprule',
           r' & & \multicolumn{3}{c}{PINN4SOH} & \multicolumn{2}{c}{tiền xử lý của bài} & không rò rỉ \\'
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}',
           r'Bộ & Chế độ & công bố & chạy lại, bản công bố & chạy lại, bản đã sửa & PINN-lai & GBDT & PINN-lai \\\midrule']
    for key in ROWS:
        cells = []
        for col, extra in cols:
            if not _have(S, key, col):
                cells.append('--'); continue
            x = v(1e3 * S.loc[key, col])
            if extra == 'sd' and pd.notna(S.loc[key, f'{col}_sd']):
                x += f' $\\pm$ {v(1e3 * S.loc[key, col + "_sd"])}'
            if extra == 'mark' and key in mark:
                x += f'$^{{{mark[key]}}}$'
            cells.append(x)
        out.append(f'{key[0] if key[1] in ("full", "n1") else ""} & {setting(key)} & ' + ' & '.join(cells) + r' \\')
        if key in (('HUST', 'full'), ('XJTU', 'n4')):
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    return '\n'.join(out)


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
    s = []
    s.append(r'\paragraph{Cấu hình đã sinh ra số công bố.} Nhật ký huấn luyện ghi, ở mỗi epoch, giá trị trung bình của loss '
             r'đơn điệu. Trong bản mã tải lên cùng bài báo, loss này là $\sum \operatorname{ReLU}(y_2 - y_1)$, chỉ tính từ '
             r'nhãn của hai chu kỳ liền nhau. ' +
             (f'Ở cả {n_const} nhóm' if n_const == len(L) else f'Ở {n_const} trên {len(L)} nhóm') +
             ', giá trị ghi trong nhật ký là một hằng số qua mọi epoch của cả mười lần chạy, và ' +
             ('ở cả ' + str(n_eq) + ' nhóm' if n_eq == len(L) else f'ở {n_eq} trên {len(L)} nhóm') +
             r' hằng số đó bằng đúng biểu thức trên tính trên các hàng train mà lớp đọc dữ liệu của bài tạo ra từ dữ liệu '
             f'ở đây (ví dụ {v(ex["logged"][0], 6)} ở batch 2C của XJTU). Số hạng này vì vậy chỉ cộng thêm một hằng số và '
             r'không có gradient: số liệu công bố đến từ mô hình chỉ học với loss dữ liệu và loss PDE. Sự trùng khớp này '
             r'cũng cho thấy cách chia dựng lại và dữ liệu ở đây cho đúng các hàng train của mọi nhóm công bố. Hai bản sửa '
             r'sau đó đổi số hạng thành $\sum \operatorname{ReLU}(u_2 - u_1)$ rồi $\sum \operatorname{ReLU}\big((u_2 - u_1)'
             r'(y_1 - y_2)\big)$, và ghi chú mới nhất của kho mã cho biết trọng số $\alpha, \beta$ đã đổi sau khi sửa lỗi; '
             r'trọng số mặc định trong các script hiện tại không còn khớp với nhật ký.')
    s.append(r'\paragraph{Chạy lại.} Bảng~\ref{tab:bench} và Hình~\ref{fig:bench} cho kết quả. Bản đã sinh số công bố, '
             f'chạy {B["runs"]} lần mỗi nhóm với siêu tham số trong nhật ký, cho sai số bằng {rng(rep_full)} lần số '
             f'công bố ở bốn bộ đầy đủ và {rng(rep_few)} lần ở chế độ ít mẫu. So theo từng cell test, bản chạy lại cao '
             f'hơn số công bố có ý nghĩa ở {rep_sig[1]} trên 4 bộ đầy đủ. Độ lệch chuẩn giữa các lần chạy bằng '
             f'{rng([100 * x for x in sd_rel], 0)}\\,\\% giá trị trung bình, nên các lần chạy đơn lẻ chênh nhau cỡ khoảng '
             'cách trên. Các hàng train trùng khớp, nên phần chênh còn lại đến từ phần cứng, phiên bản thư viện và hạt '
             'giống; đồ án không dựng lại được môi trường gốc nên không biết yếu tố nào gây ra phần chênh có hệ thống. Vì '
             'vậy báo cáo nêu cả số công bố lẫn số chạy lại, và dùng số chạy lại cho kiểm định ghép cặp vì có dự đoán theo '
             'từng cell trong điều kiện đã biết. ' +
             f'Bản hiện tại của mã ({B["runs_cur"]} lần mỗi nhóm) cho sai số bằng {rng(cur_main)} lần bản công bố ở bốn '
             f'bộ đầy đủ và ở chế độ ít mẫu của HUST ('
             + (f'khác có ý nghĩa ở {cur_sig_main} trên {len(cur_main)}' if cur_sig_main else
                f'không trường hợp nào trong {len(cur_main)} khác có ý nghĩa') + '), nhưng '
             f'{rng(cur_xjtu)} lần ở chế độ ít mẫu của XJTU với $K \\ge 2$ (cao hơn có ý nghĩa ở {cur_sig_xjtu[1]} trên '
             f'{len(cur_xjtu)}), nơi script hiện tại vẫn đặt $\\beta = 10$ cho loss đơn điệu nay đã có tác dụng.')
    table_text = '\n\n'.join(s)
    return table_text, B


def _vn_ticks():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ','))


VN_TICKS = _vn_ticks()


def fig_benchmark(S):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, save
    apply_theme()
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.42)
    cols = [('published_paper', 'PINN4SOH, số công bố', '#9aa5ad', '--', 'o'),
            ('rerun_published', 'PINN4SOH chạy lại, bản công bố', '#eb6834', '-', 's'),
            ('rerun_current', 'PINN4SOH chạy lại, bản đã sửa', '#eda100', '-', 'D'),
            ('paper_pinn_lai', 'PINN-lai, tiền xử lý của bài', '#2a78d6', '-', 'o'),
            ('strict_pinn_lai', 'PINN-lai, không rò rỉ', '#0b3d73', '-', '^')]
    for i, ds in enumerate(DS):
        ax = fig.add_subplot(gs[0, i])
        key = (ds, 'full')
        for j, (c, lab, color, _, _) in enumerate(cols):
            x = 1e3 * S.loc[key, c]
            ax.bar(j, x, color=color, width=0.72)
            sd = S.loc[key, f'{c}_sd'] if c.startswith('rerun') else np.nan
            if pd.notna(sd):
                ax.errorbar(j, x, yerr=1e3 * sd, color='#1b2330', lw=0.8, capsize=2)
        ax.set_xticks([]); ax.grid(axis='y', color='#e7ebe9', lw=0.6); ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(VN_TICKS)
        ax.set_title(ds, loc='left', fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'MAE theo cell ($\times10^{-3}$)')
    for i, ds in enumerate(('XJTU', 'HUST')):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        K = [1, 2, 3, 4]
        for c, lab, color, ls, mk in cols:
            ax.plot(K, [1e3 * S.loc[(ds, f'n{k}'), c] for k in K], color=color, ls=ls, marker=mk, ms=3.4, lw=1.3)
        ax.set_yscale('log'); ax.set_xticks(K)
        lo, hi = ax.get_ylim()
        ticks = [t for t in (3, 5, 8, 10, 15, 20, 30, 50, 80) if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
        ax.grid(axis='y', color='#e7ebe9', lw=0.6)
        ax.set_xlabel('số cell có nhãn $K$' + (' mỗi batch' if ds == 'XJTU' else ''))
        ax.set_title(f'{ds}, ít mẫu', loc='left', fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'MAE theo cell ($\times10^{-3}$)')
    handles = [plt.Line2D([], [], color=c[2], ls=c[3], marker=c[4], ms=4, lw=1.3) for c in cols]
    fig.legend(handles, [c[1] for c in cols], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.18)
    save(fig, os.path.join(FIG, 'fig_benchmark'))
    plt.close(fig)


def bench_section(S, T, L):
    text, B = bench_parts(S, T, L)
    fig = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_benchmark.pdf}' '\n'
           r'\caption[So với PINN4SOH trên đúng cách chia của bài]{So với PINN4SOH trên đúng cách chia của bài. Trên: chế '
           r'độ đầy đủ, mỗi bộ năm cột (số công bố, hai bản mã gốc chạy lại với độ lệch chuẩn qua lần chạy, PINN-lai với '
           r'tiền xử lý của bài và với tiền xử lý không rò rỉ). Dưới: chế độ ít mẫu, MAE theo số cell có nhãn (thang '
           r'log).}\label{fig:bench}' '\n' r'\end{figure}')
    write('doan_chay_lai.tex', text + '\n\n' + bench_table(S, T) + '\n\n' + fig)
    return B


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


def features_parts(F, L_leak):
    R, D, A, P, K = F['relevance'], F['redundancy'], F['ablation'], F['permutation'], F['prune']
    fracs = sorted(A.frac.unique())
    # ---------------- liên quan và trùng lặp
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption[Tương quan hạng giữa từng đầu vào và SOH]{Hệ số Spearman $\rho$ giữa từng đầu vào và SOH: trung vị '
           r'qua các cell của $\rho$ tính trong từng cell (cột “trong cell”), và $\rho$ trên mọi chu kỳ của bộ (cột '
           r'“gộp”). Dấu âm: đầu vào giảm khi SOH tăng.}\label{tab:relevance}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'rr' * len(DS) + r'}\toprule',
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
    tab_rel = '\n'.join(out)
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
           r'\caption[Bỏ hoặc thêm nhóm đầu vào với GBDT]{Bỏ hoặc thêm nhóm đầu vào, GBDT trên đúng các fold không rò rỉ (3 lần '
           r'lặp $\times$ 5 fold). Hàng đầu: MAE theo cell ($\times10^{-3}$) với đủ 17 đầu vào. Các hàng sau: tỉ số MAE so '
           r'với đủ 17 đầu vào. $^{*}$/$^{\dagger}$: thấp hơn/cao hơn có ý nghĩa (Wilcoxon ghép cặp theo cell, Holm qua bốn '
           r'bộ).}\label{tab:ablation}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'rr' * len(DS) + r'}\toprule',
           ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{d}}}' for d in DS) + r' \\'
           + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))),
           'Tập đầu vào & ' + ' & '.join([' & '.join(f'{int(100 * f)}\\,\\%' for f in fracs)] * len(DS)) + r' \\\midrule',
           'đủ 17 đầu vào (MAE) & ' + ' & '.join(v(1e3 * Ai.loc[(d, f, 'all'), 'mae']) for d in DS for f in fracs) + r' \\',
           r'\addlinespace[2pt]']
    for var, lab in VARIANTS:
        out.append(lab + ' & ' + ' & '.join(_ratio_cell(Ai.loc[(d, f, var)]) for d in DS for f in fracs) + r' \\')
    if ('TJU', fracs[0], 'with_T') in Ai.index:
        out.append(r'thêm $1/T$ (chỉ TJU) & -- & -- & ' + ' & '.join(_ratio_cell(Ai.loc[('TJU', f, 'with_T')]) for f in fracs)
                   + ' & ' + ' & '.join(['--'] * 2 * (len(DS) - 2)) + r' \\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    tab_abl = '\n'.join(out)
    # hoán vị theo nhóm
    Pt = P[P.part == 'test'].set_index(['dataset', 'frac', 'feature'])
    out = [r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption[Độ quan trọng hoán vị trên test]{Độ quan trọng hoán vị trên test của GBDT đủ 17 đầu vào: tỉ số MAE theo '
           r'cell khi hoán vị cả một nhóm (hoặc một đầu vào) giữa các hàng test so với khi không hoán vị, trung bình 15 '
           r'fold. Bằng 1: mô hình không dùng thông tin của nhóm đó.}\label{tab:perm}',
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'rr' * len(DS) + r'}\toprule',
           ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{d}}}' for d in DS) + r' \\'
           + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))),
           'Hoán vị & ' + ' & '.join([' & '.join(f'{int(100 * f)}\\,\\%' for f in fracs)] * len(DS)) + r' \\\midrule']
    for g, lab in [('[V]', 'nhóm V (6)'), ('[I]', 'nhóm I (6)'), ('[QT]', 'nhóm QT (4)'), ('[t]', '$t$')]:
        out.append(lab + ' & ' + ' & '.join(v(Pt.loc[(d, f, g), 'mean']) for d in DS for f in fracs) + r' \\')
    out += [r'\bottomrule\end{tabular}\end{adjustbox}\end{table}']
    tab_perm = '\n'.join(out)
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
    t_txt = (f'Bỏ $t$ làm sai số tăng có ý nghĩa ở {len(up_t)} trên {n_cases} trường hợp, nhiều nhất '
             f'{v(max(r_("no_t")))} lần; ' + (
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
        'Bảng~\\ref{tab:ablation} và Hình~\\ref{fig:features} cho kết quả bỏ hoặc thêm nhóm ở 8 trường hợp (4 bộ $\\times$ 2 '
        'mức nhãn), Bảng~\\ref{tab:perm} cho độ quan trọng hoán vị của mô hình đủ 17 đầu vào.\n\n'
        '\\paragraph{Biến thời gian $t$.} ' + t_txt
        + f'Ngược lại, chỉ dùng $t$ mà bỏ hết đặc trưng sạc cho sai số gấp {rng(r_("t_only"))} lần: tuổi của cell một mình '
        'không đủ để biết SOH, vì các cell già đi với tốc độ rất khác nhau (tuổi thọ trong mỗi bộ chênh nhau nhiều lần, '
        f'Bảng~\\ref{{tab:datastats}}). Hoán vị $t$ giữa các hàng test làm sai số tăng {rng(perm("[t]"))} lần, nên mô hình '
        'thật sự dùng biến này.\n\n'
        '\\paragraph{Ba nhóm đặc trưng sạc.} Không có trường hợp nào bỏ một nhóm mà sai số giảm có ý nghĩa. Nhóm V (điện '
        'áp pha CC) và nhóm QT (điện lượng, thời gian) mang nhiều thông tin nhất: bỏ nhóm V làm sai số tăng có ý nghĩa ở '
        f'{nV} trên {n_cases} trường hợp, bỏ nhóm QT ở {nQT} trên {n_cases}, '
        + (f'có trường hợp gần gấp đôi ({v(mQT)}). ' if mQT >= 1.8 else f'nhiều nhất {v(mQT)} lần. ')
        + f'Bỏ nhóm I (dòng điện pha CV) chỉ làm sai số tăng có ý nghĩa ở {nI} trường hợp. Chỉ giữ nhóm QT và $t$, tức 5 '
        f'đầu vào, kém bộ đầy đủ có ý nghĩa ở {nQt} trên {n_cases} trường hợp. '
        + (f'Phép hoán vị cho cùng nhận xét: hoán vị nhóm QT làm sai số tăng tới {v(pQT)} lần, nhóm V tới {v(pV)} lần, '
           f'nhóm I chỉ tới {v(pI)} lần.' if same_order else
           f'Hoán vị nhóm QT làm sai số tăng tới {v(pQT)} lần, nhóm V tới {v(pV)} lần, nhóm I tới {v(pI)} lần.')
        + '\n\n'
        '\\paragraph{Chọn đặc trưng tự động.} Khi bỏ các đầu vào mà hoán vị trên validation làm sai số tăng dưới 1\\,\\%, '
        f'mỗi fold giữ lại trung vị {rng(kept.values, 0)} trên 17 đầu vào tuỳ bộ và mức nhãn; hay bị bỏ nhất là '
        + jv([NAMES_VI[f] for f, c in often]) + '. Trên test, cách chọn này cho sai số bằng '
        + f'{rng(r_("prune"))} lần bộ đầy đủ'
        + (', không khác có ý nghĩa ở trường hợp nào, tức bớt đầu vào không làm mô hình tốt lên.' if not (pr_up or pr_dn)
           else (', kém hơn có ý nghĩa ở ' + lst(pr_up) if pr_up else '')
           + (', tốt hơn có ý nghĩa ở ' + lst(pr_dn) if pr_dn else '') + '.')
        + '\n\n'
        '\\paragraph{Đọc kết quả.} '
        + ('Mọi tập rút gọn đều kém bộ đầy đủ ở ít nhất một trường hợp' if len(always_hurt) == len(reduced) else
           f'{len(always_hurt)} trên {len(reduced)} tập rút gọn kém bộ đầy đủ ở ít nhất một trường hợp')
        + ((', và các trường hợp có lợi (' + '; '.join(f'{labv[x]} ở {lst(k)}' for x, k in helps)
            + ') không lặp lại ở bộ khác. Chọn đầu vào riêng cho từng bộ dựa trên những kết quả này là chọn mô hình trên '
            'tập test, còn chọn riêng cho từng fold bằng validation thì không cải thiện. ')
           if helps and all(len({d for d, _ in k}) == 1 for _, k in helps) else
           ('. Các trường hợp có lợi: ' + '; '.join(f'{labv[x]} ở {lst(k)}' for x, k in helps) + '. ' if helps else
            ', và không tập rút gọn hay mở rộng nào tốt hơn có ý nghĩa ở trường hợp nào. '))
        + 'Vì vậy đồ án dùng chung bộ 17 đầu vào cho mọi bộ dữ liệu và mọi mô hình.')
    # ---------------- đã bỏ
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
           r'16 đặc trưng tương đối so với 3 chu kỳ đầu của cell & làm sai số validation của PINN-semi tăng; với GBDT tác '
           r'động trái chiều giữa các bộ & PINN-semi, validation: tỉ số 1{,}692 (HUST 10\,\% nhãn: gấp 3{,}4 lần); GBDT, '
           r'test: ' + rng(rel) + (', cao hơn có ý nghĩa ở ' + lst(sigrel) if sigrel else '')
           + (', thấp hơn có ý nghĩa ở ' + lst(relgood) if relgood else '') + r' \\',
           r'Đặc trưng có lịch sử của cell (trung bình trượt mũ, độ dốc) & chỉ có lợi ở một số bộ khi đủ nhãn & '
           r'Bảng~\ref{tab:hist}: PINN-lai 0{,}89--0{,}97 trên TJU, HUST từ 30\,\% nhãn; kém hơn trên XJTU (1{,}02--1{,}25) '
           r'và ở 10\,\% nhãn \\']
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
        'con số đẹp hơn. Đặc trưng tương đối bị loại ở bước chọn hướng cải tiến của PINN-lai vì làm sai số validation của '
        'PINN-semi tăng (tỉ số 1,692); phép chia cho mức ban đầu của cell khuếch đại các đặc trưng có giá trị ban đầu gần '
        '0. Với GBDT, tác động của chúng trái chiều giữa các bộ'
        + (' (thấp hơn có ý nghĩa ở ' + lst(relgood) + ', cao hơn có ý nghĩa ở ' + lst(sigrel) + ')' if relgood and sigrel
           else '') + ', nên không dùng làm mặc định cho mọi bộ. Đặc trưng có lịch sử (Bảng~\\ref{tab:hist}, '
        'Hình~\\ref{fig:e29pinn} và~\\ref{fig:e29lai}) '
        'giúp trên TJU và HUST khi có từ 30\\,\\% nhãn nhưng làm kém đi trên XJTU và ở 10\\,\\% nhãn, khi số chiều đầu vào '
        'tăng gấp đôi tới gấp bốn mà chỉ còn vài cell có nhãn; vì vậy mô hình chính không dùng chúng, và mô hình cuối chỉ '
        'dùng ở các trường hợp validation chọn (phân tích hậu kiểm). '
        + ('Nhiệt độ là hằng số trên ba bộ nên không mang thông tin; trên TJU, thêm $1/T$ vào GBDT cho tỉ số '
           + tT_txt.replace('$^{*}$', ' (có ý nghĩa)').replace('$^{\\dagger}$', ' (có ý nghĩa)')
           + ', tức thay đổi dưới 1\\,\\%, nên nhiệt độ chỉ được giữ trong thừa số Arrhenius của động học. '
           if has_T else '')
        + 'Đặc trưng từ lần xả và dung lượng các chu kỳ trước bị loại khỏi bài toán ước lượng vì đó chính là nhãn.')
    text_drop = text_drop + '\n\n' + tab_drop + '\n\n' + leak_table_cache[0] + '\n\n' + history_table()
    # ---------------- kết luận về đầu vào
    concl = (
        r'\begin{enumerate}' '\n'
        r'\item Dùng đủ 16 đặc trưng đoạn sạc và $t = n/1000$: cả 17 tính được ở mọi chu kỳ từ dữ liệu sạc; '
        f'{len(always_hurt)} trên {len(reduced)} tập rút gọn kém hơn có ý nghĩa ở ít nhất một trường hợp, không tập nào '
        f'tốt hơn ở mọi bộ, và chọn bớt đặc trưng trên validation cho tỉ số {rng(r_("prune"))}.' '\n'
        r'\item Không chuẩn hoá theo cả vòng đời cell, không lọc ngoại lai bằng tương lai của cell, không fit normaliser trên '
        r'cell test: đây là rò rỉ, và tác động của chúng lên sai số lên tới ' + L_leak['minmax_rng'] + r' lần tuỳ bộ.' '\n'
        r'\item Không dùng đặc trưng tương đối (tác động trái chiều giữa các bộ); đặc trưng có lịch sử chỉ dùng khi '
        r'validation chọn; nhiệt độ chỉ đi vào thừa số Arrhenius (thêm vào đầu vào thay đổi sai số dưới 1\,\%); không '
        r'dùng thông tin của lần xả khi ước lượng.' '\n'
        r'\end{enumerate}')
    return text_rel + '\n\n' + tab_rel, text_abl + '\n\n' + tab_abl + '\n\n' + tab_perm + '\n\n' + FIG_FEAT, text_drop, concl


leak_table_cache = ['']
FIG_FEAT = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_features.pdf}' '\n'
            r'\caption[Bỏ hoặc thêm nhóm đầu vào: tỉ số MAE so với đủ 17 đầu vào]{Tỉ số MAE theo cell so với GBDT đủ 17 đầu '
            r'vào, theo tập đầu vào và mức nhãn (thang log). Điểm đặc: khác có ý nghĩa (Wilcoxon theo cell, Holm qua bốn '
            r'bộ); điểm rỗng: không có ý nghĩa. Vạch đứt: bằng bộ đầy đủ.}\label{fig:features}' '\n' r'\end{figure}')


def fig_features(A):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, save
    apply_theme()
    fracs = sorted(A.frac.unique())
    colors = {fracs[0]: '#eb6834', fracs[-1]: '#2a78d6'}
    Ai = A.set_index(['dataset', 'frac', 'variant'])
    labels = [lab.replace('$t$', 't') for _, lab in VARIANTS]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.9), sharey=True)
    for ax, ds in zip(axes, DS):
        for j, f in enumerate(fracs):
            for i, (var, _) in enumerate(VARIANTS):
                r = Ai.loc[(ds, f, var)]
                y = i + (j - 0.5) * 0.28
                s = pd.notna(r.p_holm) and r.p_holm < 0.05
                ax.plot(r.ratio, y, marker='o', ms=4.2, color=colors[f], mfc=colors[f] if s else 'white', mew=1.1, ls='none')
        ax.axvline(1, color='#4a5563', ls='--', lw=0.8)
        ax.set_xscale('log')
        lo, hi = ax.get_xlim()
        span = np.log10(hi) - np.log10(lo)
        ticks = []
        for t in (1, 2, 5, 10, 1.5, 3, 1.25, 0.8, 20):      # số tròn trước; vạch chỉ giữ khi đủ cách các vạch đã có
            if lo <= t <= hi and all(abs(np.log10(t) - np.log10(u)) >= 0.18 * span for u in ticks):
                ticks.append(t)
        ticks.sort()
        ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}'.replace('.', ',') for t in ticks]); ax.minorticks_off()
        ax.grid(axis='x', color='#e7ebe9', lw=0.6)
        ax.set_title(ds, loc='left', fontweight='bold')
        ax.set_xlabel('tỉ số MAE')
    axes[0].set_yticks(range(len(VARIANTS))); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=colors[f], marker='o', ls='none', ms=4.5) for f in fracs]
    fig.legend(handles, [f'{int(100 * f)} % nhãn' for f in fracs], loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=0.8)
    save(fig, os.path.join(FIG, 'fig_features'))
    plt.close(fig)


# --------------------------------------------------------------------------- Chương 6: đối chiếu với bài báo đã công bố
def compare_chapter(S, T, B, L_leak, D22=None, E33=None):
    pub = {k: PAPER_MAE[k] for k in ROWS}
    out = [r'\begin{table}[!htbp]\centering\small',
           r'\caption[Đối chiếu với số liệu công bố của bài PINN4SOH]{MAE theo cell ($\times10^{-3}$) trên đúng cách chia '
           r'của bài PINN4SOH. Ba cột đầu: số công bố của bài cho PINN4SOH và hai mô hình đối chứng MLP, CNN của bài. Hai '
           r'cột sau: PINN-lai của đồ án với tiền xử lý của bài (cùng điều kiện với ba cột đầu) và với tiền xử lý không rò '
           r'rỉ. In đậm: thấp nhất trong bốn cột cùng điều kiện.}\label{tab:pub}',
           r'\begin{tabular}{llrrrrr}\toprule',
           r' & & \multicolumn{3}{c}{số công bố của bài} & \multicolumn{2}{c}{PINN-lai của đồ án} \\'
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
           r'Bộ & Chế độ & PINN4SOH & MLP & CNN & tiền xử lý của bài & không rò rỉ \\\midrule']
    beat = dict(PINN4SOH=[], MLP=[], CNN=[])
    for key in ROWS:
        vals = [pub[key]['PINN4SOH'], pub[key]['MLP'], pub[key]['CNN'], S.loc[key, 'paper_pinn_lai']]
        best = int(np.argmin(vals))
        cells = [(r'\textbf{%s}' if i == best else '%s') % v(1e3 * x) for i, x in enumerate(vals)]
        cells.append(v(1e3 * S.loc[key, 'strict_pinn_lai']))
        for m in beat:
            if S.loc[key, 'paper_pinn_lai'] < pub[key][m]:
                beat[m].append(key)
        out.append(f'{key[0] if key[1] in ("full", "n1") else ""} & {setting(key)} & ' + ' & '.join(cells) + r' \\')
        if key in (('HUST', 'full'), ('XJTU', 'n4')):
            out.append(r'\addlinespace')
    out += [r'\bottomrule\end{tabular}\end{table}']
    tab = '\n'.join(out)
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
             r'đúng các nhóm cell test của PINN4SOH (Mục~\ref{sec:rerun_m}) và cách tính sai số của bài; Bảng~\ref{tab:pub} '
             r'tập hợp các con số.' '\n\n' + tab)
    good = [
        rf'Cùng cách chia và cùng tiền xử lý với bài, PINN-lai có sai số thấp hơn mã gốc PINN4SOH chạy lại ở '
        rf'{B["beat_rerun"]} trên 12 chế độ, thấp hơn có ý nghĩa ở {B["w_full"] + B["w_few"]} (tỉ số {rng(B["hyb_full"])} '
        rf'ở chế độ đầy đủ, {rng(B["hyb_few"])} ở chế độ ít mẫu) và không chế độ nào cao hơn có ý nghĩa '
        rf'(Bảng~\ref{{tab:bench}}).',
        rf'So với số công bố của PINN4SOH, PINN-lai thấp hơn ở {len(beat["PINN4SOH"])} trên 12 chế độ (tỉ số '
        rf'{rng(B["hyb_pub_full"])} ở chế độ đầy đủ, {rng(B["hyb_pub_few"])} ở chế độ ít mẫu). Lợi thế lớn nhất ở chế độ '
        rf'ít mẫu, nơi cell train không nhãn đi vào loss vật lý.',
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
        rf'lần ở chế độ đầy đủ, cao hơn có ý nghĩa theo cell ở {B["rep_sig"][1]} trên 4 bộ (Mục~\ref{{sec:rerun}}).',
    ]
    if D22 is not None:                                     # dự báo: từ kết quả chạy lại E22
        _, t22, r22 = _e22_get(D22)
        fc_cov = pct_rng([t22(ds, sc, 'cov') for ds in DS for sc in ('do', 'uoc')])
        fc_rcov = pct_rng([r22(ds, sc, 'cov') for ds in DS for sc in ('do', 'uoc')])
        fc_in20 = ', '.join(f'{pct(r22(ds, "uoc", "in20_ket_hop"))} ({ds})' for ds in DS)
    else:                                                   # số của báo cáo đã nộp
        fc_cov, fc_rcov = r'78--84\,\%', r'62--84\,\%'
        fc_in20 = r'38\,\% (XJTU), 65\,\% (TJU), 48\,\% (MIT), 52\,\% (HUST)'
    internal = [
        r'GBDT đơn lẻ có sai số thấp hơn PINN-semi ở 14/16 trường hợp bộ $\times$ mức nhãn (10 có ý nghĩa); PINN-semi chỉ '
        r'vượt GBDT có ý nghĩa ở HUST 30\,\% nhãn (Mục~\ref{sec:labels}). PINN-lai cố định vẫn kém GBDT có ý nghĩa ở TJU 70\,\%.',
        r'Ở 70\,\% nhãn, PINN-semi kém MLP có ý nghĩa trên TJU (tỉ số 1{,}09); với kiểm định bảo thủ ở mức fold, lợi ích ở '
        r'30\,\% nhãn chỉ đứng vững trên MIT.',
        r'Khoảng dự đoán 90\,\% có độ phủ thực tế 84{,}9--92{,}6\,\% (trung bình 88{,}8\,\%), 14--25\,\% số cell có độ phủ '
        r'dưới 80\,\%; khoảng dự báo SOH 90\,\% chỉ phủ ' + fc_cov + ' và khoảng RUL ' + fc_rcov + '.',
        r'Tỉ lệ dự báo RUL nằm trong $\pm 20\,\%$ RUL thật chỉ là ' + fc_in20 + '.',
        r'Tham số vật lý không được dữ liệu xác định: $\lambda$ ở mức 0{,}96--0{,}97 trên mọi bộ (khởi tạo 1), $E_a$ dịch '
        r'không quá 1{,}1\,kJ/mol; tốc độ $r$ học được trung bình 0{,}001--0{,}063 trong khi tốc độ suy giảm thực đo là '
        r'0{,}137--0{,}356 SOH trên 1\,000 chu kỳ.',
        r'Bộ lọc xu thế không bảo đảm SOH dự đoán không tăng (đầu ra tăng ở 28--44\,\% số bước); ép không tăng làm sai số '
        r'trên XJTU tăng 1{,}76--1{,}94 lần vì dung lượng đo được của XJTU tăng ở đầu đời.',
    ]
    if E33 is not None and (min(E33['cov_h']) < 0.9 or min(E33['cov_rul']) < 0.9):
        internal.append(r'Với HiPINN (\Cref{sec:e33res}), khoảng SOH 90\,\% theo dự đoán phù hợp phủ '
                        + pct_rng(E33['cov_h']) + r' tuỳ bộ và thiết lập, khoảng RUL 90\,\% phủ '
                        + pct_rng(E33['cov_rul']) + r'; tỉ lệ dự báo RUL nằm trong $\pm 20\,\%$ RUL thật khi lịch sử là ước '
                        r'lượng là ' + pct_rng(E33['in20']) + r'. Ở 10\,\% đời cuối của ' + E33['worst']
                        + r', MAE của HiPINN vẫn là ' + v(E33['tail_h'], 1) + r' ($\times10^{-3}$).')
    item = lambda xs: '\\begin{enumerate}\n' + '\n'.join(r'\item ' + x for x in xs) + '\n\\end{enumerate}'
    s.append(r'\subsection{Các chỉ số tốt hơn các bài báo đã công bố}' '\n' + item(good))
    s.append(r'\subsection{Các chỉ số chưa tốt}' '\n' r'\paragraph{So với các bài báo đã công bố.}' '\n' + item(bad) + '\n'
             r'\paragraph{Các chỉ số của chính đồ án chưa đạt.}' '\n' + item(internal))
    s.append(r'\subsection{Nhận xét}' '\n'
             r'Trên đúng cách chia của PINN4SOH và cùng cách tính sai số, mô hình của đồ án có sai số thấp hơn số công bố ở '
             r'phần lớn các chế độ và thấp hơn mã gốc chạy lại ở mọi chế độ. Tuy vậy, trong điều kiện có rò rỉ của bài, một '
             r'GBDT cũng đạt kết quả tương tự, còn khi bỏ rò rỉ thì hơn hay kém tuỳ bộ dữ liệu. Vì thế, phần đóng góp mà đồ án '
             r'tin chắc hơn cả là giao thức đánh giá không rò rỉ, việc phát hiện số hạng không có gradient trong bản mã đã sinh '
             r'số công bố, và các so sánh ghép cặp có kiểm định trong cùng một giao thức.')
    write('doan_so_sanh.tex', '\n\n'.join(s))
    return dict(beat=beat, st_better=st_better, st_worse=st_worse, good=good, bad=bad)


# --------------------------------------------------------------------------- chạy lại các thí nghiệm của đồ án
def reproducibility(B, FC=None):
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
    if FC is not None and len(glob.glob('runs/E22/fc_*.npz')) >= E22_RUNS:
        rec['e22'] = dict(FC, runs_e22=len(glob.glob('runs/E22/fc_*.npz')),
                          runs_e20=len(glob.glob('runs/E20/*_f0.7_*.json')))
    log = os.path.join(DATA, 'kiem_tai_lap.txt')
    if os.path.exists(log):
        mm = re.search(r'(\d+)/(\d+) lượt giống từng bit', open(log).read())
        if mm:
            rec['sample'] = dict(same=int(mm.group(1)), n=int(mm.group(2)), text=open(log).read()[-2000:])
    os.makedirs(DATA, exist_ok=True)
    json.dump(rec, open(path, 'w'), ensure_ascii=False, indent=1)
    s = [r'Mọi lượt chạy dùng một luồng và hạt giống cố định, nên trên cùng loại CPU và cùng phiên bản thư viện, chạy lại '
         r'phải cho dự đoán trùng từng bit. Các kiểm tra đã làm:']
    items = []
    if 'sample' in rec:
        items.append(f'Hai mươi lượt mẫu phủ mọi loại thí nghiệm chính (MLP, PINN-semi, PINN-sup, PINN hộp đen, Ridge, GPR, '
                     f'GBDT, kiểm toán rò rỉ, chuyển miền), chạy lại vào thư mục tạm và so với giá trị lưu của lần chạy báo '
                     f'cáo đã nộp: {rec["sample"]["same"]} trên {rec["sample"]["n"]} lượt có MAE theo cell và mã băm của toàn '
                     'bộ dự đoán test trùng từng bit'
                     + ('.' if rec['sample']['same'] == rec['sample']['n'] else
                        '; lượt còn lại lệch ở mức làm tròn số thực.'))
    if 'leak_refs' in rec:
        items.append('Kiểm toán rò rỉ (160 lượt của bốn cách làm sai và 40 lượt tham chiếu theo cell) được chạy lại toàn bộ; '
                     f'{rec["leak_refs"]["same"]} trên {rec["leak_refs"]["n"]} lượt có giá trị tham chiếu lưu sẵn trùng từng '
                     'bit, và các tỉ số ở Bảng~\\ref{tab:leak} khớp với báo cáo đã nộp.')
    if 'e27' in rec:
        e = rec['e27']
        items.append(f'Mô hình của đồ án trên 39 nhóm của PINN4SOH ({thou(e["runs"])} lượt) được chạy lại toàn bộ và so với '
                     f'kết quả của lần chạy trước: {e["summary_same"]} trên {e["summary_n"]} giá trị trung bình (MAE, RMSE và '
                     'độ lệch chuẩn qua ba hạt giống của PINN-lai, GBDT và trung bình ba PINN-semi, với hai cách tiền xử lý, '
                     'trên 12 chế độ) trùng từng bit; '
                     + (f'MAE và RMSE theo cell của cả {thou(e["cells"])} dòng trùng ở 6 chữ số có nghĩa đã lưu.'
                        if e['cells_same'] else 'bảng MAE và RMSE theo cell không trùng hoàn toàn ở 6 chữ số có nghĩa đã lưu.'))
    if 'e22' in rec:
        e = rec['e22']
        items.append(f'Dự báo (Mục~\\ref{{sec:forecastres}}) được chạy lại toàn bộ: {thou(e["runs_e20"])} lượt ước lượng '
                     f'PINN-lai ở 70\\,\\% nhãn làm lịch sử, rồi {e["runs_e22"]} lượt dự báo; {thou(e["same"])} trên '
                     f'{thou(e["n"])} ô của ba bảng dự báo (Bảng~\\ref{{tab:fcset}}, \\ref{{tab:e22soh}} và '
                     '\\ref{tab:e22rul}) trùng với báo cáo đã nộp.')
    items.append('GBDT với đủ 17 đầu vào trong phân tích đặc trưng (Mục~\\ref{sec:dt_pp}) cho MAE theo cell trùng từng bit '
                 'với lượt GBDT tham chiếu của thí nghiệm chính (XJTU, 30\\,\\% nhãn, lần lặp 0, fold 0).')
    items.append(f'Mã gốc PINN4SOH: giá trị loss đơn điệu tính lại trên các hàng train mà lớp đọc dữ liệu của bài tạo ra '
                 f'khớp với nhật ký công bố ở cả {B["n_eq"]} nhóm, tức dữ liệu train trùng với lần chạy của bài '
                 '(Mục~\\ref{sec:rerun}); sai số chạy lại không trùng số công bố vì khác phần cứng và thư viện.')
    s.append('\\begin{enumerate}\n' + '\n'.join(r'\item ' + x for x in items) + '\n\\end{enumerate}')
    s.append('Khác CPU (ví dụ AVX-512 so với AVX2) hoặc khác phiên bản PyTorch có thể làm kết quả lệch ở mức làm tròn số '
             'thực; khi đó các kết luận không đổi nhưng chữ số cuối trong bảng có thể khác.')
    write('doan_tai_lap.tex', '\n\n'.join(s))
    return rec


# --------------------------------------------------------------------------- kết luận và tóm tắt
def conclusion(B, L_leak, C6, F_ok, n_total, E33=None):
    items = [
        r'Xây dựng đường ống dữ liệu cho bốn bộ dữ liệu công khai (387 cell, 305.802 chu kỳ sau lọc) với bộ lọc ngoại lai '
        r'nhân quả, và kiểm tra bằng số liệu vì sao dùng 16 đặc trưng sạc cùng chỉ số chu kỳ, vì sao bỏ các cách biến đổi '
        r'khác (Mục~\ref{ch:dactrung}).',
        rf'Thiết kế giao thức đánh giá không rò rỉ và đo tác động của các dạng rò rỉ: chia theo chu kỳ làm sai số thấp đi '
        rf'{L_leak["row"]} lần, chuẩn hoá theo cả vòng đời cell làm sai số đổi {L_leak["minmax_rng"]} lần tuỳ bộ.',
        r'Đề xuất PINN bán giám sát với động học xám luôn không âm: ở 10\,\% nhãn sai số bằng 0{,}39--0{,}90 lần MLP, tiết '
        r'kiệm 1{,}7--4{,}5 lần số nhãn; và mô hình kết hợp PINN-lai: 0{,}67--0{,}95 lần PINN-semi, thấp hơn GBDT ở 10/16 '
        r'trường hợp.',
        r'So sánh có kiểm định với chín mô hình đối chứng, gồm bốn mô hình học sâu của đề cương (MLP, CNN, LSTM, '
        r'Transformer), bằng MAE, RMSE, MAPE và $R^2$.',
        rf'Chạy lại mã gốc PINN4SOH trên đúng {B["n_groups"]} nhóm của bài ({thou(B["runs"] * B["n_groups"])} lần chạy bản '
        rf'công bố, {thou(B["runs_cur"] * B["n_groups"])} lần bản đã sửa), phát hiện loss đơn điệu của bản đã sinh số công bố '
        r'không có gradient, và so cùng điều kiện: PINN-lai thấp hơn mã gốc chạy lại ở '
        rf'{B["beat_rerun"]}/12 chế độ, thấp hơn số công bố ở {len(B["beat_pub"])}/12.',
        r'Đặt lại bài toán ít nhãn theo số lần đo dung lượng: đo thưa mọi cell với 1{,}2--6{,}0\,\% số chu kỳ cho sai số bằng '
        r'0{,}85--0{,}93 lần gắn nhãn đầy đủ cho 30\,\% số cell; mở rộng sang dự báo SOH, dung lượng còn lại và RUL.',
        *([e33_item(E33)] if E33 else []),
        rf'Tổ chức lại mã nguồn thành các khối dữ liệu, mô hình, thí nghiệm, phân tích, so sánh với PINN4SOH, bản thảo bài '
        rf'báo và báo cáo; {thou(n_total)} lượt huấn luyện có kết quả trong báo cáo, mọi bảng sinh tự động từ kết quả, chạy '
        r'lại cho kết quả trùng từng bit (Mục~\ref{sec:tailap}).',
    ]
    write('doan_ket_luan.tex', '\\begin{enumerate}\n' + '\n'.join(r'\item ' + x for x in items) + '\n\\end{enumerate}')
    tomtat = (
        r'Đồ án ước lượng trạng thái sức khoẻ (SOH) của cell lithium-ion từ 16 đặc trưng đoạn sạc khi chỉ một phần nhỏ cell '
        r'được đo dung lượng, trên bốn bộ dữ liệu công khai (387 cell). Phân tích đặc trưng bằng tương quan hạng, bỏ từng '
        r'nhóm đầu vào, độ quan trọng hoán vị và chọn trên validation cho thấy cả ba nhóm đặc trưng sạc và chỉ số chu kỳ '
        r'đều mang thông tin; chuẩn hoá theo cả vòng đời cell và lọc ngoại lai bằng tương lai của cell bị loại vì là rò rỉ. '
        rf'Kiểm toán cho thấy chia theo chu kỳ làm sai số thấp đi {L_leak["row"]} lần và chuẩn hoá theo cả vòng đời cell làm '
        rf'sai số đổi {L_leak["minmax_rng"]} lần tuỳ bộ. Trong giao thức không rò rỉ (kiểm định chéo 5 fold theo cell, lặp '
        r'lại, kiểm định ghép cặp), PINN bán giám sát có sai số bằng 0{,}39--0{,}90 lần MLP ở 10\,\% nhãn, nhưng GBDT vẫn tốt '
        r'hơn nó khi có từ 30\,\% nhãn; mô hình kết hợp PINN-lai có sai số bằng 0{,}67--0{,}95 lần PINN-semi và thấp hơn GBDT '
        rf'ở 10/16 trường hợp. Đồ án chạy lại mã gốc PINN4SOH trên đúng {B["n_groups"]} nhóm của bài và phát hiện loss đơn '
        r'điệu của bản đã sinh số công bố không có gradient; bản chạy lại có sai số bằng '
        rf'{rng(B["rep_full"])} lần số công bố. Cùng cách chia và tiền xử lý của bài, PINN-lai thấp hơn mã gốc chạy lại ở '
        rf'{B["beat_rerun"]}/12 chế độ và thấp hơn số công bố ở {len(B["beat_pub"])}/12, nhưng một GBDT cùng tiền xử lý đạt '
        r'kết quả tương tự; với tiền xử lý không rò rỉ, PINN-lai thấp hơn số công bố trên '
        + ' và '.join(k[0] for k in C6['st_better']) + ', cao hơn trên ' + ' và '.join(k[0] for k in C6['st_worse'])
        + r'. Đo thưa mọi cell với 1{,}2--6{,}0\,\% số chu kỳ cho sai số bằng 0{,}85--0{,}93 lần gắn nhãn đầy đủ cho 30\,\% '
        r'số cell.' + (e33_abstract(E33) if E33 else ''))
    return tomtat


def e33_item(E):
    """Một ý của phần kết luận về E33."""
    return (r'Đưa lịch sử của cell vào dự báo và xây dựng mô hình HiPINN gồm ba nhánh: GBDT lịch sử, PINN-dự báo có '
            r'hoặc không có lịch sử, và nhánh theo đời (\Cref{sec:e33pp,sec:e33res}). Khi có lịch sử đo dung lượng, '
            f'HiPINN có sai số bằng {rng(E["r_hk"])} lần mô hình kết hợp của E22 và thấp nhất trong bảy mô hình ở '
            f'{E["best_n"]}/12 trường hợp; đặc trưng lịch sử giúp GBDT ở mọi trường hợp ({rng(E["r_g"])} lần) nhưng giúp '
            f'PINN-dự báo không đều ({rng(E["r_p"])} lần). Phần cuối đời của {E["worst"]} vẫn là chỗ dự báo kém nhất.')


def e33_abstract(E):
    return (r' Đưa lịch sử của cell vào dự báo, mô hình HiPINN ghép GBDT có đặc trưng lịch sử, PINN-dự báo và một nhánh '
            rf'theo đời cell, có sai số dự báo SOH bằng {rng(E["r_hk"])} lần mô hình dự báo trước đó khi có lịch sử đo '
            r'dung lượng.')


# --------------------------------------------------------------------------- hình
# --------------------------------------------------------------------------- dự báo SOH, dung lượng còn lại, RUL (E22)
E22_FIGS = ['e22_traj', 'e22_cells', 'e22_origins', 'e22_parity', 'e22_horizon', 'e22_rul']
E22_CSV = ['horizon', 'traj', 'origin', 'tests', 'rul', 'rul_tests', 'lambda', 'setting', 'cells_pick',
           'origins_pick', 'parity']
E22_MODELS = [('giu', 'Giữ nguyên'), ('tuyen', 'Tuyến tính'), ('vatly', 'Vật lý theo cell'), ('gbdt', 'GBDT'),
              ('pinn', 'PINN-dự báo'), ('ket_hop', 'Kết hợp')]
E22_RUNS = 60                                    # 3 lần lặp × 4 bộ × 5 fold


def forecast_load():
    """Khi runs/E22 đủ 60 lượt (và có runs/E20 cho lịch sử ước lượng): tính lại bảng (analysis/analyze_e22.py), vẽ lại
    hình (analysis/figures_report.py) vào results/, chép hình sang report/graphics/. Bảng giữ bản sao ở generated/data/,
    nên không có runs/ vẫn dựng được báo cáo từ bản đã lưu."""
    fresh = len(glob.glob('runs/E22/fc_*.npz')) >= E22_RUNS and os.path.isdir('runs/E20')
    if fresh:
        import analyze_e22, figures_report as FR
        analyze_e22.main()
        for name in E22_FIGS:
            FR.FIGS[name]()
            shutil.copy(f'results/fig_{name}.pdf', os.path.join(FIG, f'fig_{name}.pdf'))
    D = {}
    for name in E22_CSV:
        p = keep_copy(f'results/e22_{name}.csv', f'e22_{name}.csv')
        if p is None:
            return None
        D[name] = pd.read_csv(p)
    return D


def m3(x, n=1):
    """SOH (hoặc sai số SOH) theo đơn vị 10⁻³."""
    return v(1e3 * x, n)


def _e22_get(D):
    hz, tj, ru = D['horizon'], D['traj'], D['rul']
    h = lambda ds, sc, hr, m: float(hz[(hz.ds == ds) & (hz.scen == sc) & np.isclose(hz.h_rel, hr)][m].iloc[0])
    t = lambda ds, sc, m: float(tj[(tj.ds == ds) & (tj.scen == sc)][m].iloc[0])
    r = lambda ds, sc, m: float(ru[(ru.ds == ds) & (ru.scen == sc)][m].iloc[0])
    return h, t, r


def _body(blk):
    return blk[blk.index('\\midrule') + len('\\midrule'):blk.index('\\bottomrule')]


def _rows(body):
    from thesis_tables import _cell
    return [[_cell(c) for c in raw.split('&')] for raw in body.split('\\\\') if raw.strip()]


def forecast_tables(D):
    """Ba bảng dự báo sinh lại từ kết quả chạy lại, đúng định dạng của báo cáo đã nộp (cùng tiêu đề, cùng cách làm tròn,
    in đậm số thấp nhất); trả số ô trùng với bảng của báo cáo đã nộp."""
    h, t, r = _e22_get(D)
    soh = []
    for ds in DS:
        for k, hr in enumerate((0.1, 0.25, 0.5)):
            vals = [h(ds, 'uoc', hr, f'e_{m}') for m, _ in E22_MODELS]
            cells = [(r'\textbf{' + m3(x) + '}') if x == min(vals) else m3(x) for x in vals]
            soh.append(' & '.join([ds if k == 0 else '', v(hr, 2 if hr == 0.25 else 1) + '$L$'] + cells
                                  + [m3(h(ds, 'do', hr, 'e_ket_hop')), v(h(ds, 'uoc', hr, 'mah'), 0)]) + r' \\')
    rul = []
    for ds in DS:
        vals = [r(ds, 'uoc', f'e_{m}') for m, _ in E22_MODELS[1:]]
        cells = [(r'\textbf{' + v(x, 0) + '}') if x == min(vals) else v(x, 0) for x in vals]
        rul.append(' & '.join([ds, v(r(ds, 'uoc', 'true'), 0)] + cells
                              + [v(r(ds, 'do', 'e_ket_hop'), 0), pct(r(ds, 'uoc', 'in20_ket_hop')),
                                 pct(r(ds, 'uoc', 'cov'))]) + r' \\')
    st = []
    for x in D['setting'].itertuples():
        nom = v(x.nom_min, 1) if x.nom_min == x.nom_max else f'{v(x.nom_min, 1)}--{v(x.nom_max, 1)}'
        st.append(' & '.join([x.ds, str(x.n_cell), str(int(x.L)), f'{x.h1}, {x.h2}, {x.h3}', v(x.tau, 2),
                              pct(x.reach), nom]) + r' \\')
    out, same, n = {}, 0, 0
    for label, rows in (('tab:e22soh', soh), ('tab:e22rul', rul), ('tab:fcset', st)):
        old = thesis_block(label)
        body = '\n' + '\n'.join(rows) + '\n'
        a, b = _rows(_body(old)), _rows(body)
        n += sum(len(x) for x in a)
        same += sum(x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb)) if len(a) == len(b) else 0
        i, j = old.index('\\midrule') + len('\\midrule'), old.index('\\bottomrule')
        out[label] = old[:i] + body + old[j:]
    nontop = lambda blk: blk.replace('\\begin{table}[!htbp]', '\\begin{table}[!hbp]', 1)   # không đẩy lên trên tiêu đề mục
    write('bang_du_bao.tex', nontop(out['tab:e22soh']) + '\n' + nontop(out['tab:e22rul']))
    write('bang_du_bao_tham_so.tex', out['tab:fcset'])
    return dict(n=n, same=same)


def forecast_text(D):
    """Lời văn Mục dự báo (Chương 5): mọi số lấy từ bảng kết quả E22 chạy lại."""
    h, t, r = _e22_get(D)
    T, RT, lam = D['tests'], D['rul_tests'], D['lambda']
    test = lambda tb, ds, vs: tb[(tb.ds == ds) & (tb.scen == 'uoc') & (tb.vs == vs)].iloc[0]
    sig = lambda tb, ds, vs, better: (test(tb, ds, vs).p_holm < 0.05) and \
        ((test(tb, ds, vs).better > test(tb, ds, vs).n / 2) == better)
    ratio = {vs: [float(test(T, ds, vs).ratio) for ds in DS] for vs in ('giu', 'tuyen', 'vatly', 'gbdt', 'pinn')}
    beats = all(sig(T, ds, vs, True) for ds in DS for vs in ('giu', 'tuyen', 'vatly'))
    join = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    where = lambda xs: 'cả bốn bộ' if len(xs) == 4 else (join(xs) if xs else 'không bộ nào')
    gb_better = [ds for ds in DS if sig(T, ds, 'gbdt', True)]; gb_worse = [ds for ds in DS if sig(T, ds, 'gbdt', False)]
    rul_gb = [r(ds, 'uoc', 'e_gbdt') / r(ds, 'uoc', 'e_ket_hop') for ds in DS]
    rul_gb_sig = [ds for ds in DS if sig(RT, ds, 'gbdt', True)]
    L = {x.ds: x for x in lam.itertuples()}
    pos = [ds for ds in DS if L[ds].lam_median > 0]; neg = [ds for ds in DS if L[ds].lam_median <= 0]
    s = [
        r'Bảng~\ref{tab:e22soh} và~\ref{tab:e22rul} cho sai số ở kịch bản cell test không được đo dung lượng lần nào, '
        r'lịch sử SOH tới gốc là ước lượng PINN-lai. Trên toàn bộ phần đời còn lại, sai số của mô hình kết hợp bằng '
        f'{rng(ratio["giu"])} lần dự báo giữ nguyên, {rng(ratio["tuyen"])} lần ngoại suy tuyến tính và '
        f'{rng(ratio["vatly"])} lần khớp vật lý theo cell'
        + (', khác biệt có ý nghĩa trên cả bốn bộ' if beats else '') + '. '
        f'Giữa các mô hình học từ nhiều cell khác biệt nhỏ hơn: so với GBDT, tỉ số là {rng(ratio["gbdt"])}'
        + (f', thấp hơn có ý nghĩa trên {where(gb_better)}' if gb_better else '')
        + (f', cao hơn có ý nghĩa trên {where(gb_worse)}' if gb_worse else '')
        + f'; so với PINN-dự báo, tỉ số là {rng(ratio["pinn"])}. Quy ra dung lượng, sai số của mô hình kết hợp ở tầm '
        f'$0{{,}}5L$ là {rng([h(ds, "uoc", 0.5, "mah") for ds in DS], 0)}\\,mAh. '
        r'Về RUL, GBDT không ngoại suy được ra ngoài miền tầm dự báo đã học nên cắt ngưỡng quá muộn; sai số RUL của nó '
        f'gấp {rng(rul_gb, 1)} lần mô hình kết hợp, và mô hình kết hợp tốt hơn GBDT có ý nghĩa trên {where(rul_gb_sig)}. '
        r'Dấu của $\lambda$ học được khớp hình dạng quỹ đạo: '
        + (f'trung vị dương ở {join(pos)} (tỉ lệ gốc có $\\lambda > 0$ là '
           f'{"; ".join(pct(L[ds].frac_pos) for ds in pos)}, các bộ có đầu gối)' if pos else '')
        + (f', còn ở {join(neg)} trung vị âm (suy giảm chậm dần)' if neg else '') + '.']
    return '\n'.join(s)


def forecast_figs_text(D):
    """Đoạn văn và hình so dự báo với SOH đo được (Chương 5); mọi số và mọi nhận xét định tính đều tính từ bảng E22."""
    h, t, r = _e22_get(D)
    pick, og, op, par = D['cells_pick'], D['origin'], D['origins_pick'], D['parity']
    HR = (0.1, 0.25, 0.5)
    join = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    where = lambda xs: 'cả bốn bộ' if len(xs) == 4 else join(xs)
    pk = lambda ds: pick[pick.ds == ds].sort_values('q')
    o = lambda ds, q, col='e_ket_hop': float(og[(og.ds == ds) & (og.scen == 'uoc') & (og.oq == q)][col].iloc[0])
    lst3 = lambda f, fmt: '; '.join(f'{ds} ' + ', '.join(fmt(f(ds, x)) for x in HR) for ds in DS)
    fig = lambda name, cap, short, label: (
        '\\begin{figure}[!htbp]\\centering\n\\includegraphics[width=\\textwidth]{graphics/' + name + '.pdf}\n'
        '\\caption[' + short + ']{' + cap + '}\\label{' + label + '}\n\\end{figure}')
    s = []
    # --- một cell và ba cell mỗi bộ
    s.append(
        r'\paragraph{Dự báo so với SOH đo được.} Các hình dưới đây dùng cùng kịch bản: cell test không được đo dung lượng, '
        r'lịch sử tới gốc là ước lượng, còn SOH đo được (chấm đen) chỉ dùng để chấm. Hình~\ref{fig:e22traj} cho một cell '
        r'mỗi bộ, cell có sai số ở trung vị, với dự báo của ba mô hình học từ nhiều cell. Hình~\ref{fig:e22cells} cho ba '
        r'cell mỗi bộ, chọn theo sai số của mô hình kết hợp ở phân vị 25, 50 và 75\,\% trong số cell test của bộ, để thấy '
        r'cả trường hợp dự báo tốt lẫn kém. Sai số trung bình trên phần đời còn lại của ba cell này là '
        + '; '.join(f'{ds} ' + ', '.join(m3(x) for x in pk(ds).mae) for ds in DS)
        + r' ($\times10^{-3}$), và tỉ lệ chu kỳ có SOH đo nằm trong khoảng 90\,\% là '
        + '; '.join(f'{ds} ' + ', '.join(pct(x) for x in pk(ds)['cov']) for ds in DS) + '.'
        + ''.join(f' Ở cell phân vị {int(x.q)}\\,\\% của {x.ds}, mức SOH ước lượng tại gốc đã '
                  f'{"cao" if x.bias0 > 0 else "thấp"} hơn số đo {m3(abs(x.bias0))}$\\times10^{{-3}}$; dự báo bắt đầu '
                  f'từ mức này nên lệch theo, và ' + ('không chu kỳ nào' if x.cov == 0 else f'chỉ {pct(x.cov)} số chu kỳ')
                  + ' có SOH đo nằm trong khoảng.'
                  for x in pick.itertuples() if x.cov < 0.5 and abs(x.bias0) >= 0.005))
    s.append(fig('fig_e22_traj',
                 r'Dự báo cho một cell test mỗi bộ (lần lặp 0, gốc ở 40\,\% đời, cell có sai số kết hợp ở trung vị): SOH '
                 r'đo được (chấm), SOH ước lượng tới gốc, dự báo của GBDT, PINN-dự báo và mô hình kết hợp kèm khoảng '
                 r'90\,\%; trục phải là dung lượng (Ah).', 'Dự báo cho một cell test mỗi bộ', 'fig:e22traj'))
    s.append(fig('fig_e22_cells',
                 r'Dự báo của mô hình kết hợp so với SOH đo được cho ba cell test mỗi bộ, cell có sai số ở phân vị 25, '
                 r'50 và 75\,\% của bộ (lần lặp 0, gốc ở 40\,\% đời, cell test không được đo dung lượng).',
                 'Dự báo so với SOH đo được, ba cell mỗi bộ', 'fig:e22cells'))
    # --- nhiều gốc cho cùng một cell
    dec = [ds for ds in DS if o(ds, 0) > o(ds, 1) > o(ds, 2)]; inc = [ds for ds in DS if o(ds, 0) < o(ds, 1) < o(ds, 2)]
    flat = [ds for ds in DS if ds not in dec + inc and max(o(ds, q) for q in (0, 1, 2)) < 1.1 * min(o(ds, q) for q in (0, 1, 2))]
    mixed = [ds for ds in DS if ds not in dec + inc + flat]
    out_fig = [ds for ds in DS if op[(op.ds == ds) & (op.oq == 0)].below.any()]
    floor_fig = [ds for ds in out_fig if op[(op.ds == ds) & (op.oq == 0)].floor.any()]
    txt = (r'\paragraph{Dự báo từ nhiều gốc.} Hình~\ref{fig:e22origins} giữ nguyên cell của Hình~\ref{fig:e22traj} và '
           r'dự báo lại từ gốc ở 20, 40 và 60\,\% đời. Gốc muộn hơn thì lịch sử dài hơn và phần cần ngoại suy ngắn hơn, '
           r'nhưng sai số cũng được chấm trên một đoạn khác, chỉ gồm phần đời còn lại sau gốc. Sai số trên phần đời còn '
           r'lại, trung bình qua mọi cell và lần lặp, ở gốc 20, 40 và 60\,\% là '
           + '; '.join(f'{ds} {m3(o(ds, 0))}, {m3(o(ds, 1))}, {m3(o(ds, 2))}' for ds in DS) + r' ($\times10^{-3}$)'
           + ': ' + ', '.join(f'{what} ở {where(xs)}' for what, xs in (('giảm dần theo gốc', dec), ('tăng dần', inc),
                                                                         ('gần như không đổi', flat),
                                                                         ('không theo một chiều', mixed)) if xs) + '. ')
    sing = (r'khi $\lambda > 0$, nghiệm đóng~\eqref{eq:fcdrop} có điểm kỳ dị tại $h = 1/(\lambda a)$, nên ở tầm đủ dài '
            r'PINN-dự báo rơi tới giới hạn dưới 0{,}40 của SOH và kéo mô hình kết hợp xuống theo')
    txt += (f'Từ gốc 20\\,\\%, đường dự báo của {join(floor_fig)} trong hình đi ra mép dưới: {sing}. ' if floor_fig
            else 'K' + sing[1:] + '. ')
    txt += (r'Trên mọi cell và lần lặp, điều này xảy ra ở '
            + ', '.join(pct_rng([o(ds, q, 'floor_pinn') for ds in DS]) + f' số quỹ đạo từ gốc {lab}'
                        for q, lab in ((0, r'20\,\%'), (1, r'40\,\%'), (2, r'60\,\%')))
            + ' (thấp nhất và cao nhất trong bốn bộ).')
    s.append(txt)
    s.append(fig('fig_e22_origins',
                 r'Dự báo của mô hình kết hợp cho cùng cell của Hình~\ref{fig:e22traj} từ ba gốc ở 20, 40 và 60\,\% đời '
                 r'(đường chấm dọc), so với SOH đo được; trục SOH giữ theo khoảng của số đo, đường đi ra mép dưới là dự '
                 r'báo thấp hơn khung.', 'Dự báo từ ba gốc cho cùng một cell', 'fig:e22origins'))
    # --- SOH dự báo so với đo ở ba tầm
    grow = [ds for ds in DS if h(ds, 'uoc', 0.1, 'e_ket_hop') < h(ds, 'uoc', 0.25, 'e_ket_hop') < h(ds, 'uoc', 0.5, 'e_ket_hop')]
    low = [ds for ds in DS if all(h(ds, 'uoc', x, 'd_ket_hop') < 0 for x in HR)]
    nb = par.groupby('ds')[['n', 'below']].sum()
    s.append(
        r'\paragraph{SOH dự báo so với SOH đo ở ba tầm.} Hình~\ref{fig:e22parity} đặt SOH dự báo cạnh SOH đo được ở tầm '
        r'$0{,}1L$, $0{,}25L$ và $0{,}5L$ cho mọi cell test và mọi gốc của lần lặp 0. '
        + (f'Sai số tăng theo tầm ở {where(grow)} (Bảng~\\ref{{tab:e22soh}}), nên các điểm toả rộng dần khỏi đường chéo. '
           if grow else '')
        + r'Sai số có dấu trung bình (dự báo trừ số đo) ở ba tầm là '
        + lst3(lambda ds, x: h(ds, 'uoc', x, 'd_ket_hop'), m3) + r' ($\times10^{-3}$)'
        + (f', tức trung bình dự báo thấp hơn số đo ở cả ba tầm trên {where(low)}' if low else '')
        + r'. Tỉ lệ số đo nằm trong khoảng 90\,\% ở ba tầm là ' + lst3(lambda ds, x: h(ds, 'uoc', x, 'cov'), pct) + '. '
        + 'Số điểm có dự báo thấp hơn khung hình (vẽ bằng tam giác ở mép dưới) là '
        + '; '.join(f'{ds} {int(nb.loc[ds, "below"])} trên {int(nb.loc[ds, "n"])}' for ds in DS) + '.')
    s.append(fig('fig_e22_parity',
                 r'SOH dự báo của mô hình kết hợp so với SOH đo được ở ba tầm dự báo (lần lặp 0, mọi cell test, gốc ở 20, '
                 r'40 và 60\,\% đời); đường chéo là dự báo đúng bằng số đo, tam giác ở mép dưới là dự báo thấp hơn khung.',
                 'SOH dự báo so với SOH đo ở ba tầm', 'fig:e22parity'))
    # --- sai số theo tầm và RUL
    own = ('giu', 'tuyen', 'vatly')
    short = [ds for ds in DS if min(h(ds, 'uoc', 0.1, f'e_{m}') for m in own) < h(ds, 'uoc', 0.1, 'e_ket_hop')]
    far = [ds for ds in DS if min(h(ds, 'uoc', 0.5, f'e_{m}') for m in own) > h(ds, 'uoc', 0.5, 'e_ket_hop')]
    s.append(
        r'\paragraph{Sai số theo tầm và RUL.} Hình~\ref{fig:e22horizon} cho sai số của sáu mô hình theo tầm dự báo. '
        + (f'Ở tầm $0{{,}}5L$, ba mô hình chỉ dựa vào lịch sử của chính cell đều kém mô hình kết hợp trên {where(far)}'
           if far else 'Ở tầm $0{,}5L$, khác biệt giữa các mô hình nhỏ')
        + (f'; ở tầm ngắn $0{{,}}1L$, ít nhất một trong ba mô hình đó tốt hơn mô hình kết hợp trên {where(short)} '
           r'(Bảng~\ref{tab:e22soh})' if short else '') + '. '
        r'Hình~\ref{fig:e22rul} so RUL dự báo với RUL thực tại gốc ở 25, 50 và 75\,\% của chu kỳ cuối đời thật; tỉ lệ '
        r'dự báo nằm trong $\pm20\,\%$ RUL thực là ' + '; '.join(f'{ds} {pct(r(ds, "uoc", "in20_ket_hop"))}' for ds in DS)
        + r', và độ phủ của khoảng RUL 90\,\% là ' + '; '.join(f'{ds} {pct(r(ds, "uoc", "cov"))}' for ds in DS)
        + r', thấp hơn mức đặt ra. Khoảng RUL được hiệu chỉnh trên ít cell validation (10\,\% số cell), trong khi sai '
          r'số của một cell tương quan mạnh giữa các tầm, nên phân vị lấy từ vài cell hẹp hơn phân bố sai số của cell mới.')
    s.append(fig('fig_e22_horizon',
                 r'MAE theo cell của sáu mô hình theo tầm dự báo $h/L$ ($L$: trung vị số chu kỳ của bộ), kịch bản lịch sử '
                 r'ước lượng, mọi lần lặp và mọi gốc.', 'Sai số dự báo theo tầm', 'fig:e22horizon'))
    s.append(fig('fig_e22_rul',
                 r'RUL dự báo của mô hình kết hợp so với RUL thực, trung bình qua ba lần lặp, tại gốc ở 25, 50 và 75\,\% '
                 r'của chu kỳ cuối đời thật; dải xám là sai lệch trong $\pm20\,\%$, $\tau$ là ngưỡng cuối đời của bộ. Dự báo '
                 r'không chạm ngưỡng trong $3L$ chu kỳ được ghi là $3L$.', 'RUL dự báo so với RUL thực', 'fig:e22rul'))
    return '\n\n'.join(s)


def pct_rng(vals, n=0):
    """Khoảng phần trăm: [0.02, 0.09] -> '2--9\\,\\%'."""
    lo, hi = v(100 * min(vals), n), v(100 * max(vals), n)
    return (lo if lo == hi else f'{lo}--{hi}') + r'\,\%'


# --------------------------------------------------------------------------- E33: dự báo có đặc trưng lịch sử (HiPINN)
E33_CSV = ['summary', 'tests', 'rul', 'rul_tests', 'choices', 'gdoan', 'gdoan_match', 'gdoan_anchor', 'traj_pick',
           'tail', 'cal']
E33_TAB = ['gbdt', 'pinn', 'ket_hop', 'gbdt_ls', 'pinn_ls', 'doi', 'hipinn']       # cột của bảng chính
E33_HEAD = {'gbdt': ('GBDT', 'theo tầm'), 'pinn': ('PINN-', 'dự báo'), 'ket_hop': ('Kết', 'hợp'),
            'gbdt_ls': ('GBDT', 'lịch sử'), 'pinn_ls': ('PINN có', 'lịch sử'), 'doi': ('Theo', 'đời'),
            'hipinn': ('', 'HiPINN')}
E33_TXT = {'giu': 'giữ nguyên', 'tuyen': 'ngoại suy tuyến tính', 'gbdt': 'GBDT theo tầm', 'gbdt_ls': 'GBDT lịch sử',
           'pinn': 'PINN-dự báo', 'pinn_ls': 'PINN-dự báo có lịch sử', 'ket_hop': 'mô hình kết hợp của E22',
           'ket_hop_ls': 'HiPINN không có nhánh theo đời', 'doi': 'nhánh theo đời', 'hipinn': 'HiPINN'}
E33_SET = [('lab30', 'do', r'30\,\% nhãn, lịch sử đo'), ('all', 'do', 'mọi cell train, lịch sử đo'),
           ('all', 'uoc', 'mọi cell train, lịch sử ước lượng')]
E33_RUNS = 120
E33_FIGS = ('e33_mae', 'e33_traj', 'e33_tail')


def e33_load():
    """Khi runs/E33 đủ 120 lượt: tính lại bảng (analysis/analyze_e33.py) và vẽ lại hình; bảng giữ bản sao ở
    generated/data/, nên không có runs/ vẫn dựng được."""
    fresh = len(glob.glob('runs/E33/fc_*.npz')) >= E33_RUNS
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
    if 'hipinn' not in D['summary'].columns:             # bản sao của lần chạy trước khi có HiPINN: chưa dựng mục này
        print('generated/data/e33_*.csv là bản cũ (chưa có HiPINN): chạy python run.py E33 rồi dựng lại')
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
        self.TL, self.TP = D['tail'], D['traj_pick']

    def val(self, b, sc, ds, m):
        S = self.S
        return float(S[(S.budget == b) & (S.scen == sc) & (S.ds == ds)][m].iloc[0])

    def rv(self, b, sc, ds, col):
        R = self.R
        return float(R[(R.budget == b) & (R.scen == sc) & (R.ds == ds)][col].iloc[0])

    def tail(self, b, sc, ds, pb, col):
        T = self.TL
        return float(T[(T.budget == b) & (T.scen == sc) & (T.ds == ds) & (T.pb == pb)][col].iloc[0])

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
            r'\caption[MAE của SOH dự báo, E33]{MAE theo cell ($\times10^{-3}$) của SOH dự báo trên mọi chu kỳ đo được '
            r'sau gốc ($0{,}2L$, $0{,}4L$, $0{,}6L$), trung bình qua 3 lần lặp; in đậm là thấp nhất mỗi hàng. '
            r'$^{*}$/$^{\dagger}$: HiPINN tốt hơn/kém hơn có ý nghĩa so với GBDT lịch sử theo kiểm định Wilcoxon ghép '
            r'cặp theo cell (Holm qua bốn bộ, $p<0{,}05$).}\label{tab:e33}' '\n'
            r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * len(E33_TAB) + r'}\toprule' '\n'
            r' & \multicolumn{3}{c}{mô hình của E22} & \multicolumn{4}{c}{có đặc trưng lịch sử} \\'
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-8}' '\n'
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
            rr.append(' & '.join([ds, v(E.rv(b, sc, ds, 'true'), 0)] + cells
                                 + [pct(E.rv(b, sc, ds, 'in20_hipinn')), pct(E.rv(b, sc, ds, 'cov_h'))]) + r' \\')
        rr.append(r'\addlinespace[3pt]')
    return (r'\begin{table}[!htbp]\centering\small' '\n'
            r'\caption[MAE của RUL dự báo, E33]{MAE của RUL (chu kỳ), trung bình qua các gốc ở 25, 50 và 75\,\% chu kỳ '
            r'cuối đời thật và qua 3 lần lặp; in đậm là thấp nhất. Hai cột cuối là của HiPINN: tỉ lệ dự báo nằm trong '
            r'$\pm20\,\%$ RUL thực và độ phủ của khoảng RUL 90\,\%.}\label{tab:e33rul}' '\n'
            r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * (3 + len(E33_TAB)) + r'}\toprule' '\n'
            r'Bộ & ' + head2('RUL', 'thực') + ' & ' + ' & '.join(head2(*E33_HEAD[m]) for m in E33_TAB)
            + ' & ' + head2('Trong', r'$\pm20\,\%$') + ' & ' + head2('Độ', 'phủ') + r' \\\midrule' '\n'
            + '\n'.join(rr[:-1]) + '\n' r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')


def e33_anchor(GA):
    """Phụ lục: mô hình cập nhật có đặc trưng lịch sử (kết quả lần chạy trước trong kho phát triển, không chạy lại)."""
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    tiers = [('q20', r'mỗi $0{,}2L$'), ('q10', r'mỗi $0{,}1L$'), ('q05', r'mỗi $0{,}05L$'), ('M2', 'mọi lần trước')]
    am = ['hold', 'linear', 'e32_style', 'history_delta', 'history_residual', 'selected']
    alab = [('Giữ', 'lần đo'), ('Tuyến', 'tính'), ('Kiểu', 'E32'), ('Lịch sử,', 'độ giảm'), ('Lịch sử,', 'phần dư'),
            ('Chọn', 'validation')]
    ga = lambda ds, t, m: float(GA[(GA.ds == ds) & (GA.tier == t)][m].iloc[0])
    ar = []
    for t, tl in tiers:
        ar.append(group_row('lịch đo ' + tl if t != 'M2' else 'biết mọi lần đo trước', 1 + len(am)))
        for ds in DS:
            vals = [ga(ds, t, m) for m in am]
            cells = [(r'\textbf{' + m3(x, 2) + '}') if x == min(vals) else m3(x, 2) for x in vals]
            r_ = GA[(GA.ds == ds) & (GA.tier == t)].iloc[0]
            cells[-1] += ((r'$^{*}$' if r_.wins_selected == 1 else r'$^{\dagger}$') if r_.p_selected < 0.05
                          else r'\phantom{$^{*}$}')
            ar.append(' & '.join([ds] + cells) + r' \\')
        ar.append(r'\addlinespace[3pt]')
    tab = (r'\begin{table}[!htbp]\centering\small' '\n'
           r'\caption[Mô hình cập nhật có đặc trưng lịch sử]{Cập nhật SOH ở các chu kỳ không đo từ lần đo dung lượng gần '
           r'nhất, theo lịch đo: MAE theo cell ($\times10^{-3}$), 30\,\% nhãn, 3 lần lặp $\times$ 5 fold, kết quả lần '
           r'chạy trước (không chạy lại). $^{*}$/$^{\dagger}$: cột cuối tốt hơn/kém hơn có ý nghĩa so với mô hình kiểu '
           r'E32 (Wilcoxon ghép cặp theo cell, $p<0{,}05$, không hiệu chỉnh).}\label{tab:anchor}' '\n'
           r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{l' + 'r' * len(am) + r'}\toprule' '\n'
           r'Bộ & ' + ' & '.join(head2(*x) for x in alab) + r' \\\midrule' '\n' + '\n'.join(ar[:-1]) + '\n'
           r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    low = {t: [ds for ds in DS if ga(ds, t, 'selected') < ga(ds, t, 'e32_style')] for t, _ in tiers}
    keep = len(low['q10']) < 3 and len(low['q20']) < 3
    assert len(low['q20']) <= 1 and len(low['q10']) <= 1, 'câu "khi đo thưa, lịch sử không giúp" không còn đúng'
    odd = [r_ for r_ in GA.itertuples() if r_.tier != 'M2' and r_.p_selected < 0.05 and r_.flip_selected > 0]
    tl = dict(tiers)
    text = (r'Đồ án cũng thử đặc trưng lịch sử cho bài toán cập nhật của \Cref{sec:mo_rong}: ước lượng SOH ở các chu kỳ '
            r'không đo từ lần đo dung lượng gần nhất, với $L$ tính trên cell train. Ngoài mô hình kiểu E32, có hai cách '
            r'dùng lịch sử (học độ giảm, hoặc học phần dư so với ngoại suy) và một mô hình chọn giữa chúng trên validation. '
            r'\Cref{tab:anchor} cho kết quả của lần chạy đó. Khi đo thưa, lịch sử không giúp: mô hình chọn trên validation '
            f'chỉ có MAE trung bình thấp hơn kiểu E32 ở {len(low["q20"])}/4 bộ với lịch đo mỗi $0{{,}}2L$ và '
            f'{len(low["q10"])}/4 bộ với lịch mỗi $0{{,}}1L$, và kém hơn có ý nghĩa trên TJU ở cả ba lịch đo thưa. Lịch sử '
            f'chỉ có lợi khi đo dày: {len(low["q05"])}/4 bộ'
            + (f' ({jv(low["q05"])})' if 0 < len(low['q05']) < 4 else '') + r' ở lịch mỗi $0{,}05L$ và '
            f'{len(low["M2"])}/4 bộ khi biết mọi lần đo trước, nhưng mức sau không phải là đo thưa.'
            + ''.join(f' Ở {r_.ds} với lịch {tl[r_.tier]}, mô hình chọn trên validation tốt hơn ở '
                      f'{int(r_.better_selected)}/{int(r_.n_cell)} cell nên kiểm định nghiêng về nó, dù MAE trung bình '
                      f'cao hơn vì {"một cell" if r_.flip_selected == 1 else "vài cell"} sai lớn.'
                      for r_ in odd if r_.wins_selected == 1)
            + (' Vì vậy đồ án giữ mô hình cập nhật kiểu E32.' if keep else ''))
    tju_bad = all(GA[(GA.ds == 'TJU') & (GA.tier == t)].iloc[0].p_selected < 0.05
                  and GA[(GA.ds == 'TJU') & (GA.tier == t)].iloc[0].wins_selected == 0 for t in ('q20', 'q10', 'q05'))
    assert tju_bad, 'câu "kém hơn có ý nghĩa trên TJU ở cả ba lịch đo thưa" không còn đúng'
    return text, tab


def e33_repro(E):
    """Phụ lục: GBDT lịch sử chạy lại trong E33 so với lần chạy trước (30 % nhãn, lịch sử đo)."""
    G, GM = E.D['gdoan'], E.D['gdoan_match']
    gd = lambda ds, m: float(G[G.ds == ds][m].iloc[0])
    gm = {r.ds: r for r in GM.itertuples()}
    ok_base = all(gm[ds].giu_equal == 1 and gm[ds].tuyen_equal == 1 for ds in DS)
    both = lambda m, g: '; '.join(f'{ds} {m3(E.val("lab30", "do", ds, m), 2)} (trước {m3(gd(ds, g), 2)})' for ds in DS)
    return (r'Ở ngân sách 30\,\% nhãn với lịch sử đo, E33 dùng đúng các cell có nhãn, cell test, gốc dự báo và cách '
            r'chọn trên validation của lần chạy trước của GBDT lịch sử'
            + (r'; hai mô hình giữ nguyên và ngoại suy tuyến tính cho sai số theo cell trùng nhau ở hai lần chạy'
               if ok_base else '')
            + r'. MAE ($\times10^{-3}$) của lần này, trong ngoặc là lần trước: GBDT lịch sử ' + both('gbdt_ls', 'selected')
            + '; GBDT theo tầm ' + both('gbdt', 'gbdt') + '. Tương quan giữa sai số theo cell của hai lần chạy là '
            + rng([gm[ds].corr_ls for ds in DS], 3) + ' với GBDT lịch sử và ' + rng([gm[ds].corr_gbdt for ds in DS], 3)
            + r' với GBDT theo tầm. Phần chênh đến từ phiên bản thư viện (scikit-learn 1.9 và NumPy 2.4 ở lần này, 1.8 và '
              r'2.3 ở lần trước); mọi so sánh trong báo cáo dùng số của lần chạy này, mọi mô hình trong cùng một môi trường.')


def e33_section(D):
    """Mục kết quả E33 (Chương 5) và Phụ lục A. Lời văn viết tay; số lấy từ bảng kết quả, nhận định định tính được
    kiểm bằng điều kiện trên số liệu (sai thì dừng dựng báo cáo thay vì in câu sai)."""
    E = E33Data(D)
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    where = lambda xs: 'cả bốn bộ' if len(xs) == 4 else (jv(xs) if xs else 'không bộ nào')
    do = [('lab30', 'do'), ('all', 'do')]
    uoc = [('all', 'uoc')]
    CH = E.CH

    def sig_txt(up, dn, good='thấp hơn', bad='cao hơn'):
        parts = ([f'{good} có ý nghĩa trên {where(up)}'] if up else []) + ([f'{bad} có ý nghĩa trên {where(dn)}'] if dn else [])
        return ', '.join(parts) if parts else 'không khác có ý nghĩa trên bộ nào'

    def both_budgets(x, y):
        """Các bộ x tốt hơn / kém hơn y có ý nghĩa ở cả hai ngân sách (lịch sử đo)."""
        u = [set(E.better(b, sc, x, y)[0]) for b, sc in do]
        d = [set(E.better(b, sc, x, y)[1]) for b, sc in do]
        keep = lambda ss: [ds for ds in DS if all(ds in s_ for s_ in ss)]
        any_ = lambda ss: [ds for ds in DS if any(ds in s_ for s_ in ss)]
        return keep(u), any_(d)

    # ---------------- bảng và mở đầu
    own = [E.val(b, sc, ds, m) / E.val(b, sc, ds, 'hipinn') for b, sc, _ in E33_SET for ds in DS for m in ('giu', 'tuyen')]
    p0 = (r'\Cref{tab:e33} cho MAE của SOH dự báo ở ba thiết lập: ngân sách 30\,\% nhãn hoặc mọi cell train, lịch sử SOH '
          r'của cell test là dung lượng đo hoặc ước lượng. Hai mô hình chỉ dựa vào chính cell là giữ nguyên và ngoại suy '
          f'tuyến tính có sai số gấp {rng(own, 1)} lần HiPINN nên không đưa vào bảng (có trong \\Cref{{fig:e33mae}}).')
    # ---------------- lịch sử trong từng nhánh
    r_g = E.ratios(do, 'gbdt_ls', 'gbdt'); ug, dg = both_budgets('gbdt_ls', 'gbdt')
    r_p = E.ratios(do, 'pinn_ls', 'pinn'); up_, dp_ = both_budgets('pinn_ls', 'pinn')
    share_pls = float((CH.pinn_branch == 'pinn_ls').mean())
    assert max(r_g) < 1, 'câu "lịch sử giúp GBDT ở mọi trường hợp" không còn đúng'
    p1 = (r'\paragraph{Thêm lịch sử vào từng nhánh.} Với GBDT, đặc trưng lịch sử giúp rõ rệt: khi lịch sử là dung lượng '
          f'đo, GBDT lịch sử có MAE bằng {rng(r_g)} lần GBDT theo tầm, thấp hơn có ý nghĩa trên {where(ug)} ở cả hai '
          'ngân sách. Cây quyết định dùng được trực tiếp những thông tin như độ dốc trên nhiều cửa sổ hay mức dao động '
          'gần đây để nhận ra cell đang suy giảm nhanh hay chậm. ')
    if max(r_p) < 1:
        p1 += (f'PINN-dự báo cũng tốt lên khi có lịch sử, MAE bằng {rng(r_p)} lần bản cũ. ')
    else:
        p1 += (f'Với PINN-dự báo, lợi ích không đều: bản có lịch sử có MAE bằng {rng(r_p)} lần bản cũ, '
               + sig_txt(up_, dp_) + '. Một cách giải thích: mạng chỉ chọn hai tham số $(a, \\lambda)$ của một đường '
               'suy giảm có dạng cố định, nên thông tin thêm từ lịch sử chủ yếu giúp ước lượng tốc độ hiện tại, điều mà '
               'hai độ dốc trong $\\mathbf{z}_k$ đã có một phần, trong khi số đầu vào tăng hơn gấp đôi. ')
    p1 += (f'Vì nhánh PINN của HiPINN được chọn trên validation, bản có lịch sử được dùng ở {pct(share_pls)} số lượt.')
    # ---------------- HiPINN
    best_n = sum(min(E33_TAB, key=lambda m: E.val(b, sc, ds, m)) == 'hipinn' for b, sc, _ in E33_SET for ds in DS)
    r_hk = E.ratios(do, 'hipinn', 'ket_hop'); uk, dk = both_budgets('hipinn', 'ket_hop')
    r_hl = E.ratios(do, 'hipinn', 'gbdt_ls')
    hl = {b: E.better(b, 'do', 'hipinn', 'gbdt_ls') for b, _ in do}
    r_hk_u = E.ratios(uoc, 'hipinn', 'ket_hop'); uk_u, dk_u = E.better('all', 'uoc', 'hipinn', 'ket_hop')
    r_h2 = E.ratios(do + uoc, 'hipinn', 'ket_hop_ls')
    lt_two = [E.val(b, sc, ds, 'hipinn') < E.val(b, sc, ds, 'ket_hop_ls') for b, sc, _ in E33_SET for ds in DS]
    p2 = (r'\paragraph{HiPINN.} '
          f'HiPINN có MAE thấp nhất trong bảy mô hình ở {best_n}/12 trường hợp (bộ $\\times$ thiết lập). Khi lịch sử là '
          f'dung lượng đo, MAE của nó bằng {rng(r_hk)} lần mô hình kết hợp của E22, thấp hơn có ý nghĩa trên {where(uk)} ở '
          f'cả hai ngân sách; so với GBDT lịch sử, tỉ số là {rng(r_hl)} '
          + '(' + '; '.join(f'{lab}: ' + sig_txt(*hl[b]) for (b, _), lab in zip(do, [r'30\,\% nhãn', 'mọi cell train']))
          + '). '
          f'Khi lịch sử là ước lượng, lợi thế nhỏ lại: HiPINN bằng {rng(r_hk_u)} lần mô hình kết hợp của E22, '
          + sig_txt(uk_u, dk_u) + ', vì khi đó sai số của mức SOH ước lượng tại gốc chiếm phần lớn sai số dự báo '
          '(\\Cref{sec:forecastres}). '
          f'Bỏ nhánh theo đời, MAE tăng ở {sum(lt_two)}/12 trường hợp (tỉ số HiPINN so với bản không có nhánh này '
          f'{rng(r_h2)}).')
    # ---------------- phần cuối đời
    tb = '0.9-1'
    tail = lambda ds, m: 1e3 * E.tail('all', 'do', ds, tb, m)
    bias = lambda ds, m: 1e3 * E.tail('all', 'do', ds, tb, f'b_{m}')
    worst = max(DS, key=lambda ds: tail(ds, 'gbdt_ls'))
    sgn = lambda x: ('+' if round(x, 1) > 0 else '') + v(x, 1)
    pbs = ['<0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1']
    curve = [1e3 * E.tail('all', 'do', worst, b_, 'gbdt_ls') for b_ in pbs]
    assert curve[-1] == max(curve) and curve[-1] > 1.5 * curve[0], 'sai số không tăng về cuối đời như lời văn'
    assert bias(worst, 'gbdt_ls') > 0, 'GBDT lịch sử không dự báo cao hơn thực tế ở cuối đời'
    assert abs(bias(worst, 'doi')) < abs(bias(worst, 'gbdt_ls')), 'nhánh theo đời không giảm độ lệch ở cuối đời'
    w3_worst = float(CH[CH.ds == worst].w_doi.mean())
    p3 = (r'\paragraph{Phần cuối đời.} \Cref{fig:e33tail} chia sai số theo phần đời của chính cell (mọi cell train, lịch '
          r'sử đo). Với các mô hình dự báo trực tiếp, sai số tăng dần về cuối đời và lớn nhất ở '
          f'{worst}: ở 10\\,\\% đời cuối, MAE của GBDT lịch sử là {v(tail(worst, "gbdt_ls"), 1)} và của PINN-dự báo là '
          f'{v(tail(worst, "pinn"), 1)} ($\\times10^{{-3}}$). GBDT lịch sử dự báo cao hơn thực tế ở đoạn này, độ lệch '
          f'trung bình {sgn(bias(worst, "gbdt_ls"))}: thời điểm gãy của từng cell khác nhau nên dự báo trực tiếp thành '
          'một đường trung bình thoải hơn thực tế. Nhánh theo đời đặt hình dạng suy giảm lên đời dự đoán của từng cell, '
          f'nên độ lệch ở đoạn này chỉ còn {sgn(bias(worst, "doi"))} và MAE là {v(tail(worst, "doi"), 1)}. ')
    doi_tot = E.val('all', 'do', worst, 'doi') / E.val('all', 'do', worst, 'gbdt_ls')
    p3 += ((f'Đổi lại, trên toàn bộ phần đời còn lại nhánh này có MAE bằng {v(doi_tot)} lần GBDT lịch sử, vì khi đời bị '
            'đoán sai thì điểm gãy cũng bị đặt sai chỗ. ') if doi_tot > 1 else
           f'Trên toàn bộ phần đời còn lại, nhánh này có MAE bằng {v(doi_tot)} lần GBDT lịch sử. ')
    p3 += (f'Ở {worst}, validation cho nhánh theo đời trọng số trung bình {v(w3_worst)} trong HiPINN, nên ở 10\\,\\% đời '
           f'cuối MAE của HiPINN là {v(tail(worst, "hipinn"), 1)} và độ lệch {sgn(bias(worst, "hipinn"))}: phần cuối đời '
           + ('vẫn là chỗ HiPINN dự báo kém nhất.' if tail(worst, 'hipinn') > 0.8 * tail(worst, 'gbdt_ls') else
              'tốt lên rõ so với GBDT lịch sử.'))
    # ---------------- trọng số và ví dụ
    wd = CH.w_doi; wl = CH.w_ls; wp = CH.w_pinn
    p4 = (r'\paragraph{Trọng số các nhánh.} Trên validation, GBDT lịch sử nhận trọng số trung bình '
          f'{v(wl.mean())}, nhánh PINN {v(wp.mean())} và nhánh theo đời {v(wd.mean())}; nhánh theo đời có mặt '
          f'($w_3 > 0$) ở {pct((wd > 0).mean())} số lượt, nhánh PINN ở {pct((wp > 0).mean())}. '
          r'\Cref{fig:e33traj} minh hoạ dự báo cho một cell mỗi bộ.')
    # ---------------- TJU: kiểm định và trung bình ngược chiều
    odd = []
    for b, sc, lab in E33_SET:
        for x, y in (('hipinn', 'gbdt_ls'), ('gbdt_ls', 'ket_hop')):
            for ds in DS:
                p, xw, ratio, r = E.row(b, sc, ds, x, y)
                if p < 0.05 and ((xw and ratio > 1) or (not xw and ratio < 1)):
                    odd.append((b, sc, ds, x, y, r))
    p5 = ''
    if odd:
        dss = sorted({o[2] for o in odd})
        tops = {o[5].top_cell for o in odd}
        p5 = (r'\paragraph{Trường hợp TJU.} ' if dss == ['TJU'] else r'\paragraph{Kiểm định và trung bình ngược chiều.} ')
        p5 += (f'Ở {jv(dss)}, có {len(odd)} phép so mà kiểm định ghép cặp và MAE trung bình chỉ hai hướng khác nhau: phần '
               'lớn cell đi theo hướng của kiểm định, còn trung bình bị kéo bởi vài cell sai rất lớn; bỏ '
               + rng([o[5].flip_k for o in odd], 0) + ' cell như vậy thì trung bình cũng đổi chiều. ')
        if len(tops) == 1:
            t = tops.pop(); r = odd[0][5]
            p5 += (f'Cell gây ra chênh lệch lớn nhất luôn là {tex_esc(t.split("/")[-1])} của {t.split("/")[0]}, nơi '
                   f'PINN-dự báo sai rất xa (MAE {v(1e3 * r.top_pinn, 0)} so với {v(1e3 * r.top_gbdt_ls, 0)} '
                   r'của GBDT lịch sử, $\times10^{-3}$), nên mô hình nào có nhánh PINN thì sai nhiều ở cell này.')
    # ---------------- RUL và khoảng
    rk = {sc: E.better('all', sc, 'hipinn', 'ket_hop', rul=True) for sc in ('do', 'uoc')}
    rr = {sc: [E.rv('all', sc, ds, 'e_hipinn') / E.rv('all', sc, ds, 'e_ket_hop') for ds in DS] for sc in ('do', 'uoc')}
    over = [ds for ds in DS if E.rv('all', 'do', ds, 's_gbdt_ls') > 0.8 * E.rv('all', 'do', ds, 'e_gbdt_ls')]
    cov_do = [E.val(b, sc, ds, 'cov_h') for b, sc in do for ds in DS]
    cov_u = [E.val('all', 'uoc', ds, 'cov_h') for ds in DS]
    p6 = (r'\paragraph{RUL và khoảng dự báo.} \Cref{tab:e33rul} ở \Cref{app:additional} cho MAE của RUL. Với mọi cell '
          f'train, RUL của HiPINN có MAE bằng {rng(rr["do"])} lần mô hình kết hợp của E22 khi lịch sử đo '
          f'({sig_txt(*rk["do"])}) và {rng(rr["uoc"])} lần khi lịch sử ước lượng ({sig_txt(*rk["uoc"])}). ')
    if over:
        r_hg = [E.rv('all', 'do', ds, 'e_hipinn') / E.rv('all', 'do', ds, 'e_gbdt_ls') for ds in over]
        p6 += (f'Riêng GBDT lịch sử dự báo RUL dài hơn thực tế trên {jv(over)} (sai số có dấu trung bình '
               + jv([f'{ds} +{v(E.rv("all", "do", ds, "s_gbdt_ls"), 0)}' for ds in over])
               + f' chu kỳ); trên các bộ đó, MAE của RUL của HiPINN bằng {rng(r_hg)} lần GBDT lịch sử. ')
    p6 += (f'Khoảng SOH 90\\,\\% của HiPINN phủ {pct_rng(cov_do)} khi lịch sử đo và {pct_rng(cov_u)} khi lịch sử '
           'ước lượng; khoảng này hiệu chỉnh trên lịch sử đo của cell validation nên hẹp hơn mức cần khi mức SOH tại gốc '
           'cũng là ước lượng.')
    # ---------------- chọn mô hình
    def verdict(b, sc, cand, get, rul=False):
        for a in cand:
            if all(sum(get(b, sc, ds, a) < get(b, sc, ds, c) for ds in DS) >= 3 and not E.better(b, sc, a, c, rul)[1]
                   for c in cand if c != a):
                return a
        return None
    cand = ['hipinn', 'gbdt_ls', 'ket_hop']
    soh = {(b, sc): verdict(b, sc, cand, E.val) for b, sc, _ in E33_SET}
    rul = {sc: verdict('all', sc, ['hipinn', 'ket_hop'], lambda b, s, ds, m: E.rv(b, s, ds, f'e_{m}'), True)
           for sc in ('do', 'uoc')}
    name = lambda m: E33_TXT[m] if m else None
    p7 = (r'\paragraph{Nên dùng mô hình nào.} Ba mô hình HiPINN, GBDT lịch sử và mô hình kết hợp của E22 được so theo '
          'cùng một quy tắc: mô hình được chọn phải có MAE thấp hơn từng mô hình còn lại trên ít nhất ba bộ và không kém '
          'có ý nghĩa trên bộ nào. Vì quy tắc được đặt sau khi đã có kết quả test, lựa chọn dưới đây là gợi ý cho lần '
          'dùng sau, chưa được kiểm chứng trên dữ liệu mới. ')
    lab = {k: l for *k_, l in [(b, sc, l) for b, sc, l in E33_SET] for k in [tuple(k_)]}
    chosen = [f'{lab[k]}: {name(m)}' for k, m in soh.items() if m]
    none = [lab[k] for k, m in soh.items() if not m]
    p7 += ('Với SOH, quy tắc chọn ' + '; '.join(chosen) + '. ' if chosen else '')
    p7 += (('Ở ' + jv(none) + ' không mô hình nào thoả. ') if none else '')
    p7 += ('Với RUL, ' + '; '.join(f'{"lịch sử đo" if sc == "do" else "lịch sử ước lượng"}: '
                                   + (name(m) if m else 'không mô hình nào thoả') for sc, m in rul.items()) + '.')
    figs = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_mae.pdf}' '\n'
            r'\caption[MAE của SOH dự báo trong E33]{MAE theo cell (trung bình qua 3 lần lặp, thang log) của các mô hình '
            r'ở ba thiết lập: ngân sách 30\,\% nhãn hoặc mọi cell train, lịch sử SOH đo hoặc ước lượng.}\label{fig:e33mae}'
            '\n' r'\end{figure}')
    fig_tail = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_tail.pdf}'
                '\n' r'\caption[Sai số dự báo theo phần đời của cell]{MAE ($\times10^{-3}$) của SOH dự báo theo phần đời '
                r'của chính cell (chu kỳ chia chu kỳ cuối ghi được), mọi cell train, lịch sử đo, trung bình theo cell. '
                r'Phần đời chỉ dùng để chia nhóm khi chấm, mô hình không biết nó.}\label{fig:e33tail}' '\n'
                r'\end{figure}')
    TP = E.TP
    fig_traj = (r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_e33_traj.pdf}'
                '\n' r'\caption[Dự báo của HiPINN so với SOH đo được]{Dự báo của HiPINN và ba nhánh so với SOH đo được '
                r'cho một cell test mỗi bộ: mọi cell train, lịch sử đo, lần lặp 0, gốc ở $0{,}4L$; trong các cell còn ít '
                r'nhất $' + v(float(TP.min_h.iloc[0]), 2) + r'L$ chu kỳ đo sau gốc, chọn cell có sai số ở trung vị. Trọng '
                r'số $(w_1, w_2, w_3)$ của lượt chứa cell đó: '
                + '; '.join(f'{r.ds} ({v(r.w_ls, 2)}; {v(r.w_pinn, 2)}; {v(r.w_doi, 2)})' for r in TP.itertuples())
                + r'. Khoảng 90\,\% theo dự đoán phù hợp.}\label{fig:e33traj}' '\n' r'\end{figure}')
    body = ('\\FloatBarrier\n\\subsection{Dự báo có đặc trưng lịch sử: HiPINN}\\label{sec:e33res}\n\n' + p0 + '\n\n'
            + e33_table(E) + '\n\n' + p1 + '\n\n' + p2 + '\n\n' + figs + '\n\n' + p3 + '\n\n' + fig_tail + '\n\n'
            + p4 + '\n\n' + fig_traj + '\n\n' + (p5 + '\n\n' if p5 else '') + p6 + '\n\n' + p7 + '\n')
    write('doan_e33.tex', body)
    # ---------------- Phụ lục A
    a_text, a_tab = e33_anchor(D['gdoan_anchor'])
    pa = (r'\section{Dự báo có đặc trưng lịch sử: RUL}' '\n\n' + e33_rul_table(E) + '\n\n'
          r'\section{Mô hình cập nhật có đặc trưng lịch sử}\label{sec:anchor}' '\n\n' + a_text + '\n\n' + a_tab + '\n\n'
          r'\section{GBDT lịch sử: lần chạy này và lần chạy trước}\label{sec:e33repro}' '\n\n' + e33_repro(E))
    write('phu_luc_a.tex', pa)
    return dict(soh=soh, rul=rul, name=name, r_hk=r_hk, r_hk_u=r_hk_u, r_hl=r_hl, r_g=r_g, r_p=r_p, best_n=best_n,
                cov_h=cov_do + cov_u, cov_rul=[E.rv('all', sc, ds, 'cov_h') for sc in ('do', 'uoc') for ds in DS],
                in20=[E.rv('all', 'uoc', ds, 'in20_hipinn') for ds in DS], worst=worst,
                tail_ls=tail(worst, 'gbdt_ls'), tail_h=tail(worst, 'hipinn'))


def demo_section():
    """Chương 4: công cụ dự báo cho một cell (tools/demo.py), HiPINN cho một cell test của mỗi bộ. Số và hình lấy từ
    demo/hipinn_bon_bo.* (lệnh `python run.py demo tat-ca`); bản sao ở generated/data/ và graphics/."""
    p = keep_copy('demo/hipinn_bon_bo.csv', 'demo_hipinn.csv')
    if os.path.exists('demo/hipinn_bon_bo.pdf'):
        shutil.copy('demo/hipinn_bon_bo.pdf', os.path.join(FIG, 'fig_demo_hipinn.pdf'))
    text = (r'\section{Công cụ dự báo cho một cell}\label{sec:congcu}' '\n\n'
            r'Công cụ dòng lệnh của đồ án gói các mô hình để dùng cho một cell bất kỳ, chỉ cần tệp đặc trưng sạc của cell '
            r'đó. Lệnh thứ nhất học PINN-semi ở 30\,\% nhãn cho một bộ dữ liệu, lệnh thứ hai ước lượng SOH ở từng chu kỳ, '
            r'lệnh thứ ba dự báo SOH, dung lượng và RUL từ một chu kỳ bất kỳ bằng HiPINN (\Cref{sec:e33pp}). Lần đầu dự báo '
            r'cho một bộ, HiPINN được học trên các cell có nhãn của cùng fold với mô hình ước lượng, chọn trọng số trên cell '
            r'validation rồi lưu lại; cell được dự báo '
            r'không tham gia khi học. Nếu cell không được đo dung lượng, lịch sử SOH lấy từ mô hình ước lượng và khoảng '
            r'90\,\% dùng phân vị hiệu chỉnh riêng cho trường hợp này.')
    if p is None or not os.path.exists(os.path.join(FIG, 'fig_demo_hipinn.pdf')):
        write('doan_cong_cu.tex', text)
        return
    Dm = pd.read_csv(p)
    jv = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' và ' + xs[-1]
    name = lambda cid: tex_esc(cid.split('/')[-1])
    text += ('\n\n' r'Lệnh thứ tư áp dụng HiPINN cho cả bốn bộ: với mỗi bộ, lấy cell test có số chu kỳ ở trung vị '
             r'và dự báo từ 40\,\% số chu kỳ của nó (\Cref{fig:demo4}). MAE trên phần đời còn lại là '
             + jv([f'{v(1e3 * r.mae, 1)} ở {r.bo} (cell {name(r.cell)})' for r in Dm.itertuples()])
             + r' ($\times10^{-3}$). Trọng số mà validation chọn khác nhau giữa các bộ: '
             + jv([f'{r.bo} ({v(r.w_gbdt_ls, 2)}; {v(r.w_pinn, 2)}; {v(r.w_doi, 2)})' for r in Dm.itertuples()])
             + r' cho GBDT lịch sử, nhánh PINN và nhánh theo đời. Mô hình của công cụ học trên 30\,\% cell có nhãn, ít '
               r'hơn thí nghiệm E33 với mọi cell train, nên đây là minh hoạ chứ không phải số đánh giá.' '\n\n'
             r'\begin{figure}[!htbp]\centering' '\n' r'\includegraphics[width=\textwidth]{graphics/fig_demo_hipinn.pdf}' '\n'
             r'\caption[HiPINN cho một cell test của mỗi bộ dữ liệu]{HiPINN áp dụng cho bốn bộ dữ liệu qua công cụ dòng '
             r'lệnh: với mỗi bộ, cell test có số chu kỳ ở trung vị, dự báo từ 40\,\% số chu kỳ của cell, lịch sử là dung '
             r'lượng đo; mô hình học trên 30\,\% cell có nhãn của fold dùng cho công cụ.}\label{fig:demo4}' '\n'
             r'\end{figure}')
    write('doan_cong_cu.tex', text)


def forecast_section():
    D = forecast_load()
    if D is None:
        write('bang_du_bao_tham_so.tex', thesis_block('tab:fcset'))
        write('doan_du_bao.tex', r'(Chưa có kết quả E22: chạy \texttt{python run.py E20}, \texttt{python run.py E22} rồi '
                                 r'dựng lại báo cáo.)')
        write('doan_du_bao_hinh.tex', '')
        return None
    rep = forecast_tables(D)
    write('doan_du_bao.tex', forecast_text(D))
    write('doan_du_bao_hinh.tex', forecast_figs_text(D))
    return dict(rep, D=D)


def figures():
    import diagrams
    os.makedirs(FIG, exist_ok=True)
    diagrams.arch('vi', os.path.join(FIG, 'fig_arch'))
    diagrams.pipeline(os.path.join(FIG, 'fig_pipeline'))
    diagrams.forecast_arch(os.path.join(FIG, 'fig_forecast_arch'))
    diagrams.hipinn_arch(os.path.join(FIG, 'fig_hipinn_arch'))
    for src, dst in [('fig_thesis_leakage', 'fig_leakage'), ('fig_thesis_curve_physics', 'fig_curve_physics'),
                     ('fig_e28_budget', 'fig_e28_budget'), ('fig_e29_pinn', 'fig_e29_pinn'), ('fig_e29_lai', 'fig_e29_lai')]:
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
    thesis_tables()
    leak_tab, leak_text, L_leak = leakage()
    leak_table_cache[0] = leak_tab
    write('doan_ro_ri.tex', leak_text)
    S, T, L = bench_load()
    B = bench_section(S, T, L)
    fig_benchmark(S)
    F = features_load()
    n_feat_jobs = n_feat_fits = 0
    if F is not None:
        rel, abl, drop, concl = features_parts(F, L_leak)
        fig_features(F['ablation'])
        jobs = F['prune'][['dataset', 'frac', 'repeat', 'fold']].drop_duplicates()
        n_feat_jobs = len(jobs)
        n_feat_fits = n_feat_jobs * 9 + int((jobs.dataset == 'TJU').sum())     # 9 tập đầu vào, thêm 1/T ở TJU
    else:
        rel = abl = drop = concl = r'(Chưa có kết quả phân tích đặc trưng: chạy \texttt{python analysis/features.py}.)'
    write('doan_lien_quan.tex', rel); write('doan_bo_nhom.tex', abl); write('doan_da_bo.tex', drop)
    write('doan_ket_luan_dt.tex', concl)
    n_rerun = int(B['runs'] * B['n_groups'] + B['runs_cur'] * B['n_groups'])
    n_total = experiments_table(n_rerun, n_feat_jobs, n_feat_fits)
    FC = forecast_section()
    demo_section()
    D33 = e33_load()
    E33 = None
    if D33 is not None:
        E33 = e33_section(D33)
    else:
        write('doan_e33.tex', r'(Chưa có kết quả E33 với HiPINN: chạy E33 rồi dựng lại báo cáo.)')
        write('phu_luc_a.tex', '')
    C6 = compare_chapter(S, T, B, L_leak, FC['D'] if FC else None, E33)
    reproducibility(B, {k: x for k, x in FC.items() if k != 'D'} if FC else None)
    tomtat = conclusion(B, L_leak, C6, F is not None, n_total, E33)
    write('so_lieu.tex', r'\newcommand{\TomTat}{' + tomtat + '}')
    figures()
    if not a.no_pdf:
        compile_pdf()


if __name__ == '__main__':
    main()
