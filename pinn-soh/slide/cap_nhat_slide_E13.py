"""
Cập nhật bộ slide báo cáo 29/09 với kết quả E13/E14 — giữ nguyên khuôn BK của bản gốc.

    python slide/cap_nhat_slide_E13.py <bản-gốc.pptx> <bản-mới.pptx>

Chạy từ thư mục gốc repo, SAU analyze_e13.py và verify_pipeline.py --csv. Mọi con số đọc từ results/
qua section_e13.facts() — cùng nguồn với báo cáo Markdown/LaTeX, không gõ tay số nào.

Thay đổi so với bản gốc:
  slide 1, 2      : phụ đề, bảng tiến độ
  slide 9–15      : gắn nhãn "(v1)" — kết quả trước khi sửa pipeline
  slide 16        : năm điểm rà soát → đã sửa, kèm phép kiểm
  slide 18        : khung kết quả → điền tỉ số E13
  chèn sau 18     : thiết kế E13, H1, forest plot, đường cong nhãn, H2, từng cell + quỹ đạo, chỉ số phụ, E14, demo
  slide 19, 20, 22: kế hoạch tiếp theo, kết luận, khả năng tái lập
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import pptx_tools as T
import section_e13 as S

R = 'results'
DS = S.DS
f, pval, pct = S.f, S.pval, S.pct
GREEN, ORANGE, INK = '0D6E56', 'B8531F', '163047'


def ratio_cell(r, star=True):
    if r is None:
        return '—'
    s = f(r.ti_so)
    if star and r.p_holm < 0.05:
        s += '*'
    return s


def build(src, dst):
    d = S.load(); F = S.facts(d)
    ver = d['ver']
    nver = f'{int(ver.dat.sum())}/{len(ver)}' if ver is not None else '—'
    tests = d['tests']
    n_e14 = 0 if d['e14'] is None else int(d['e14'].n_seed.sum() * 2)
    prs = Presentation(src)
    orig_n = len(prs.slides)
    TABLE_SRC, FRAME_SRC = 17, 8                                    # chỉ số 0-based trong bản gốc
    frame_shapes = [s for s in prs.slides[FRAME_SRC].shapes
                    if not s.has_chart and s.left < Inches(1) and Inches(1.3) < s.top < Inches(1.7)]

    def table_slide(title, header, rows, widths, font=13, row_h=0.42, msg=None, foot=None, notes=None, colors=None):
        s = T.dup_slide(prs, TABLE_SRC)
        T.set_text(T.title_shape(s), title)
        tb = [x for x in s.shapes if x.has_table][0]
        T.fill_table(tb, rows, header=header, col_widths_in=widths, font_pt=font, row_h_in=row_h, colors=colors)
        box = [x for x in s.shapes if x.has_text_frame and Inches(5.5) < x.top < Inches(6.5)]
        if box:
            if msg:
                T.set_text(box[0], msg)
                box[0].top = tb.top + tb.height + Inches(0.25)
            else:
                T.remove_shape(box[0])
        if foot is not None:
            T.set_text(T.footnote_shape(s), foot)
        if notes:
            T.set_notes(s, notes)
        return s

    def figure_slide(title, imgs, label=None, foot=None, notes=None, side_text=None):
        s = T.dup_slide(prs, TABLE_SRC)                  # dải tiêu đề navy như các slide nội dung
        T.set_text(T.title_shape(s), title)
        for x in [x for x in s.shapes if x.has_table or (x.has_text_frame and Inches(5.5) < x.top < Inches(6.5))]:
            T.remove_shape(x)
        frame = T.clone_shape(frame_shapes[0], s)
        frame.left, frame.top, frame.width, frame.height = Inches(0.57), Inches(1.35), Inches(12.19), Inches(5.15)
        top = Inches(1.45)
        if label and len(frame_shapes) > 1:
            lab = T.clone_shape(frame_shapes[1], s)
            T.set_text(lab, label)
            lab.left, lab.top = Inches(0.73), Inches(1.45)
            top = Inches(1.9)
        avail_w = Inches(11.9 if side_text is None else 8.3)
        h_total = Inches(6.4) - top
        each = h_total // len(imgs)
        for k, im in enumerate(imgs):
            T.add_picture_fit(s, im, Inches(0.73), top + k * each, avail_w, each - Inches(0.05))
        if side_text is not None:
            tb = s.shapes.add_textbox(Inches(9.25), Inches(2.2), Inches(3.3), Inches(3.8))
            tf = tb.text_frame; tf.word_wrap = True
            for i, (txt, bold, color) in enumerate(side_text):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                r = p.add_run(); r.text = txt
                r.font.size = Pt(16 if bold else 14); r.font.bold = bold; r.font.name = 'Arial'
                from pptx.dml.color import RGBColor
                r.font.color.rgb = RGBColor.from_string(color)
                p.space_after = Pt(8)
        if foot is not None:
            T.set_text(T.footnote_shape(s), foot)
        if notes:
            T.set_notes(s, notes)
        return s

    # ───────────────────────── slide 1: phụ đề
    s1 = prs.slides[0]
    sub = T.shapes_by_text(s1, 'Báo cáo tiến độ')[0]
    T.set_text(sub, 'Kết quả thí nghiệm xác nhận E13 · pipeline v2\n29/09/2026')

    # ───────────────────────── slide 2: tiến độ
    s2 = prs.slides[1]
    n_rows_v2 = None
    try:
        from pinnsoh.data import load_dataset
        n_rows_v2 = sum(len(c.soh) for ds in DS for c in load_dataset('PINN4SOH/data', ds, clean='causal'))
    except Exception:
        pass
    h1w, h2ok = F['h1_win'], F['h2_ok']
    T.fill_table([x for x in s2.shapes if x.has_table][0], [
        ['Bài toán', 'Ước lượng SOH khi ít cell có nhãn', 'Đã xác định'],
        ['Dữ liệu', f'387 cell · v2: {n_rows_v2:,} hàng sau lọc nhân quả'.replace(',', '.') if n_rows_v2 else '387 cell',
         'Đã đếm lại'],
        ['Pipeline', f'Sửa 5 điểm rà soát · {nver} phép kiểm đạt', 'Xong'],
        ['Thực nghiệm mới', f'E13: {F["n_runs"]:,} lượt (CV {F["folds"]}×{F["repeats"]})'.replace(',', '.')
         + (f' · E14: {n_e14} lượt' if n_e14 else ''), 'Đã chạy'],
        ['Kết luận', f'H1 đạt {len(h1w)}/4 bộ · H2 đạt {len(h2ok)}/4 bộ', 'Theo đề cương'],
    ])
    T.set_text(T.footnote_shape(s2), 'Mọi số trong các slide E13 đọc từ results/E13_*.csv, sinh từ '
               f'{F["n_runs"]} log huấn luyện mới. Đề cương chốt trước khi chạy (commit {S.PREREG_COMMIT}).')
    T.set_notes(s2, 'Tuần này: sửa pipeline, chốt đề cương, chạy E13 (kiểm định chéo theo cell) và E14. '
                    'Các slide 9–15 là kết quả v1 giữ lại để đối chiếu.')

    # ───────────────────────── slide 9–15: gắn nhãn v1
    for i in [8, 9, 10, 13, 14]:
        t = T.title_shape(prs.slides[i])
        T.set_text(t, t.text_frame.text + ' (v1)')

    # ───────────────────────── slide 16: đã sửa
    s16 = prs.slides[15]
    T.set_text(T.title_shape(s16), 'Mã nguồn: năm điểm đã sửa trong pipeline v2')
    T.fill_table([x for x in s16.shapes if x.has_table][0], [
        ['data.py: lọc 3σ', 'Dùng toàn vòng đời từng cell', 'Cửa sổ 50 chu kỳ quá khứ · phá tương lai: 0 thay đổi'],
        ['data.py: dropna()', 'Thiếu capacity làm mất hàng', 'Chỉ loại khi đặc trưng lỗi · CSV không nhãn nạp đủ'],
        ['train.py: normalizer', 'MLP và PINN fit trên pool khác', 'norm_pool=train · mu, sd trùng từng bit'],
        ['data.py: sample_pairs', 'h là số hàng sau lọc', f'Theo chu kỳ gốc · v1 lệch ở {S.v1_over(ver)} cặp'],
        ['metrics.py: EOL', 'Chỉ số hàng, chưa kiểm duyệt', 'Theo chu kỳ · tách bỏ sót / báo giả'],
    ], header=['Vị trí', 'Phát hiện', 'Đã sửa (v2) · kiểm chứng'], col_widths_in=[2.9, 3.6, 5.67], font_pt=15)
    T.set_text(T.footnote_shape(s16), f'verify_pipeline.py: {nver} phép kiểm đạt. v1 vẫn là mặc định và tái lập '
               'từng bit log cũ (đã kiểm trên XJTU, TJU).')
    T.set_notes(s16, 'Nguồn: pinnsoh/data.py (_clean_causal_features, sample_pairs_cycle, cv_split), train.py '
                     '(Config.v2, norm_pool), metrics.py (eol_cycle). Kết quả kiểm: results/verify_pipeline.csv.')

    # ───────────────────────── slide 18: khung kết quả đã điền
    s18 = prs.slides[17]
    T.set_text(T.title_shape(s18), 'Khung kết quả: đã chạy đủ trong E13')
    r = lambda k: ' / '.join(ratio_cell(S._row(tests, k, ds)) for ds in DS)
    T.fill_table([x for x in s18.shapes if x.has_table][0], [
        ['MLP, cùng normalizer', 'Mốc so sánh công bằng', '1 / 1 / 1 / 1 (mốc)'],
        ['PINN-sup', 'Tách tác dụng regularization', r('B vs A')],
        ['PINN-semi', 'Đo ích lợi cell không nhãn', r('C vs A')],
        ['Chỉ loss đơn điệu', 'Tách ích lợi của ODE', r('D vs A') if len(tests[tests.so_sanh == 'D vs A']) else
         ' / '.join(f(S._row(tests, 'D vs C', ds).ti_so) if S._row(tests, 'D vs C', ds) is not None else '—' for ds in DS)
         + ' (so PINN-semi)'],
        ['PINN 30% / MLP 70%', 'Kiểm chứng tiết kiệm nhãn', r('C vs F')],
    ], header=['Đối chứng sau sửa', 'Mục đích', 'Tỉ số MAE theo cell · XJTU / TJU / MIT / HUST'],
        col_widths_in=[3.1, 3.6, 5.47], font_pt=15)
    box = [x for x in s18.shapes if x.has_text_frame and Inches(5.5) < x.top < Inches(6.5)][0]
    T.set_text(box, 'Tỉ số < 1: sai số thấp hơn MLP 30 % (hàng cuối: so với MLP 70 %). * = p Holm < 0,05.')
    T.set_text(T.footnote_shape(s18), f'Kiểm định chéo {F["folds"]} fold × {F["repeats"]} lần lặp theo cell, '
               'cùng split, normalizer, nhãn và ngân sách lựa chọn. Chi tiết ở các slide sau.')

    # ───────────────────────── slide mới
    new = []
    new.append(table_slide('E13: thiết kế chốt trước khi chạy', ['Hạng mục', 'Đã ấn định trong đề cương'], [
        ['Chia dữ liệu', f'Kiểm định chéo {F["folds"]} fold theo cell × {F["repeats"]} lần lặp, phân tầng theo batch'],
        ['Mỗi fold', '70 % train · 10 % validation · 20 % test; mỗi cell test đúng 1 lần / lần lặp'],
        ['Nhánh', f'{F["n_arms"]} nhánh, cùng mạng 128-128-64 và tối ưu; chỉ khác hàm mất mát'],
        ['Chỉ số chính', 'MAE theo cell, trung bình qua các lần lặp'],
        ['H1', 'PINN-semi vs MLP, 30 %: Wilcoxon theo cell, Holm cho 4 bộ'],
        ['H2', 'PINN-semi 30 % vs MLP 70 %: không kém hơn, biên δ = 10 %'],
        ['Độ nhạy', 'Phép t hiệu chỉnh Nadeau–Bengio ở mức fold'],
    ], [2.6, 9.57], font=15, row_h=0.5, foot=f'Đề cương tai-lieu/de-cuong-E13.md, commit {S.PREREG_COMMIT} trước lượt '
                                              'huấn luyện đầu tiên. Validation cần nhãn ngoài ngân sách 30 %.',
        notes='Điểm khác so với E1–E12: đơn vị phân tích là cell, mỗi cell được test đúng một lần mỗi lần lặp, '
              'nên không còn chuyện các seed chồng lấp cell test.'))

    rows, colors = [], {}
    for i, ds in enumerate(DS):
        x = S._row(tests, 'C vs A', ds)
        if x is None:
            continue
        rows.append([f'{ds} ({S.CHEM[ds]})', str(int(x.n_cell)), f(1e3 * x.mae_a, 2), f(1e3 * x.mae_b, 2),
                     f'{f(x.ti_so)} [{f(x.ci_lo)}; {f(x.ci_hi)}]', pct(x.cell_b_tot_hon), pval(x.p_holm),
                     pval(x.p_nb_holm)])
        c = GREEN if (x.p_holm < 0.05 and x.trung_vi_hieu < 0) else (ORANGE if x.p_holm < 0.05 else None)
        if c:
            colors[(len(rows) - 1, 4)] = c
    lines = S.conclusion_lines(d, F)
    new.append(table_slide('H1: PINN-semi so với MLP, cùng 30% nhãn',
                           ['Bộ', 'Cell', 'MLP ×10⁻³', 'PINN ×10⁻³', 'Tỉ số [KTC 95 %]', 'Cell PINN tốt hơn',
                            'p Holm (cell)', 'p Holm (fold)'], rows, [2.0, 0.8, 1.35, 1.35, 2.5, 1.55, 1.3, 1.32],
                           font=14, row_h=0.5, msg=lines[0], colors=colors,
                           foot='MAE theo cell, trung bình 5 lần lặp. p (cell): Wilcoxon theo cell. '
                                'p (fold): t hiệu chỉnh Nadeau–Bengio. Holm cho 4 bộ.',
                           notes=' '.join(lines[:3])))
    new.append(figure_slide('Tỉ số MAE theo cell cho mọi so sánh', [f'{R}/fig_E13_forest.png'],
                            label='KHUNG KẾT QUẢ 3 · tỉ số và KTC 95 % theo cell',
                            foot='Xanh = tốt hơn có ý nghĩa sau Holm; xám = chưa đủ bằng chứng; vạch đứt 1,10 = biên H2.',
                            notes='Đọc theo hàng: H1 và H2 là hai giả thuyết chính; các hàng dưới là so sánh thăm dò. '
                                  + S.CAP_FOREST))
    new.append(figure_slide('Hiệu quả theo lượng nhãn', [f'{R}/fig_E13_curve.png'],
                            label='KHUNG KẾT QUẢ 4 · MAE theo cell ở 10 / 30 / 50 / 70 % nhãn',
                            foot='Dải = KTC 95 % bootstrap theo cell. Vạch đứt = MLP dùng 70 % nhãn (mốc của H2).',
                            notes=S.CAP_CURVE))
    rows2 = []
    for ds in DS:
        x = F['h2'][ds]
        if x is None:
            continue
        rows2.append([ds, f(1e3 * x.mae_a, 2), f(1e3 * x.mae_b, 2), f'{f(x.ti_so)} [{f(x.ci_lo)}; {f(x.ci_hi)}]',
                      pval(x.p_holm), x.ket_luan])
    h2line = [l for l in lines if l.startswith('H2')]
    new.append(table_slide('H2: PINN 30% có thay được MLP 70%?',
                           ['Bộ', 'MLP 70 % ×10⁻³', 'PINN 30 % ×10⁻³', 'Tỉ số [KTC 90 %]', 'p Holm', 'Kết luận'],
                           rows2, [1.5, 2.0, 2.0, 2.6, 1.3, 2.77], font=14, row_h=0.5,
                           msg=h2line[0] if h2line else None,
                           foot='Không kém hơn khi p Holm < 0,05 (Wilcoxon một phía trên e_PINN − 1,1·e_MLP) VÀ cận trên '
                                'KTC 90 % < 1,10.'))
    # hình không mang chữ chú thích: phần giải thích nằm ở nhãn khung và chân slide, số đọc từ CSV
    share = ', '.join(f'{ds} {pct(S._row(tests, "C vs A", ds).cell_b_tot_hon)}' for ds in DS
                      if S._row(tests, 'C vs A', ds) is not None)
    new.append(figure_slide('Từng cell: PINN-semi so với MLP, 30% nhãn', [f'{R}/fig_E13_cells.png'],
                            label='KHUNG KẾT QUẢ 5 · MAE từng cell, trung bình 5 lần lặp',
                            foot=f'Xanh lục (dưới đường chéo) = PINN-semi thấp hơn. Tỉ lệ cell: {share}.',
                            notes=S.cap_cells(d)))
    tr = d.get('traj')
    tr_foot = ('MAE ×10⁻³ (MLP 30 / PINN 30 / MLP 70): ' + ' · '.join(
        f'{r.bo} ' + ' / '.join(f(1e3 * r[c], 1) if c in r and np.isfinite(r[c]) else '—'
                                for c in ['mae_A', 'mae_C', 'mae_F']) for _, r in tr.iterrows())
               if tr is not None and not tr.empty else 'Quy tắc chọn cell không phụ thuộc PINN. Lần lặp 0.')
    new.append(figure_slide('Quỹ đạo SOH trên cell test', [f'{R}/fig_E13_traj.png'],
                            label='KHUNG KẾT QUẢ 6 · cell có MAE của MLP đúng trung vị mỗi bộ',
                            foot=tr_foot, notes=S.cap_traj(d)))
    sec = d['sec']; mrows = []
    for ds in DS:
        for arm in ['A', 'C']:
            g = sec[(sec.bo == ds) & (sec.nhanh == arm)]
            if g.empty:
                continue
            g = g.iloc[0]
            mrows.append([ds, 'MLP 30 %' if arm == 'A' else 'PINN-semi 30 %', f(1e3 * g.MAE_late, 2), f(g.EOL_MAE, 1),
                          f'{int(g.EOL_bo_sot)} / {int(g.EOL_bao_gia)}', f(g.MonoViol100, 3), f(1e3 * g.Jitter, 2)])
    new.append(table_slide('Chỉ số phụ: cuối đời và tính hợp lý của quỹ đạo',
                           ['Bộ', 'Mô hình', 'MAE SOH ≤ 0,9', 'EOL MAE (ck)', 'bỏ sót / báo giả', 'Vi phạm đơn điệu',
                            'Jitter ×10⁻³'], mrows, [1.2, 2.2, 1.75, 1.6, 1.9, 1.9, 1.62], font=13, row_h=0.4,
                           foot='EOL tại SOH 0,85 theo chu kỳ gốc, trên các lượt mà nhãn và dự đoán cùng cắt ngưỡng. '
                                'Vi phạm đơn điệu = tổng bước tăng ngược / 100 chu kỳ. MAE SOH ≤ 0,9 đơn vị ×10⁻³.'))
    if d['e14'] is not None:
        erows = S.table_e14(d)
        new.append(table_slide('E14: chuyển miền khi dùng cùng normalizer',
                               ['Nguồn → đích', 'k', 'MLP', 'PINN-semi', 'Tỉ số', 'Seed PINN thắng'], erows,
                               [2.6, 0.7, 2.6, 2.6, 1.4, 2.27], font=13, row_h=0.4,
                               foot='MAE theo cell trên tập test đích, TB ± SD qua 5 seed. Cả hai mô hình fit '
                                    'normalizer trên cùng pool đặc trưng nguồn + đích; chỉ khác loss vật lý.',
                               notes='E14 trả lời lưu ý ở slide 15: chênh lệch E5 có còn khi bỏ khác biệt normalizer.'))
    demo_png = sorted([p for p in os.listdir('demo') if p.endswith('_soh.png')]) if os.path.isdir('demo') else []
    if demo_png:
        new.append(figure_slide('Demo: ước lượng SOH cho cell chưa đo dung lượng', [os.path.join('demo', demo_png[0])],
                                side_text=[('Đầu vào', True, '20567B'),
                                           ('16 đặc trưng sạc CC–CV + chỉ số chu kỳ; KHÔNG cần cột capacity', False, INK),
                                           ('Suy luận', True, '20567B'),
                                           ('Nhân quả: chu kỳ N chỉ dùng dữ liệu đến N — chạy trực tuyến được', False, INK),
                                           ('Lệnh', True, '20567B'),
                                           ('python demo.py predict --model demo/XJTU.pt --csv <cell.csv> --an-nhan',
                                            False, INK)],
                                foot='Cell thuộc fold test của mô hình demo (chưa từng thấy khi huấn luyện). '
                                     'Có nhãn thì demo in thêm MAE để đối chiếu.'))

    # đưa các slide mới vào sau slide 18
    for k in range(len(new)):
        T.move_slide(prs, orig_n + k, 18 + k)

    # ───────────────────────── kế hoạch, kết luận, phụ lục (chỉ số đã dịch)
    off = len(new)
    s19 = prs.slides[18 + off]
    T.fill_table([x for x in s19.shapes if x.has_table][0], [
        ['Tuần 1', 'Viết chương kết quả từ E13 (mục 10c báo cáo LaTeX)', 'Báo cáo có số sinh tự động'],
        ['Tuần 1', 'Đối chứng mạnh hơn: GPR, XGBoost trên cùng CV', 'Bảng so sánh mở rộng'],
        ['Tuần 2', 'Ước lượng bất định (ensemble / conformal)', 'Khoảng tin cậy cho SOH'],
        ['Tuần 2', 'Hoàn thiện demo và slide bảo vệ', 'Demo 4 bộ + slide'],
    ], header=['Thời gian', 'Ưu tiên', 'Sản phẩm'])
    T.set_text([x for x in s19.shapes if x.has_text_frame and Inches(5.5) < x.top < Inches(6.5)][0],
               'Giữ nguyên giao thức E13 cho mọi đối chứng mới')
    if T.footnote_shape(s19) is not None:
        T.set_text(T.footnote_shape(s19), 'Kế hoạch đề xuất, không phải công việc đã hoàn thành.')
    T.set_notes(s19, 'Mọi đối chứng mới chạy trên đúng cv_split và pipeline v2 để so được trực tiếp với E13.')

    s20 = prs.slides[19 + off]
    body_zone = lambda x: x.has_text_frame and Inches(1.5) < x.top < Inches(6.3)
    heads = [x for x in s20.shapes if body_zone(x) and x.height < Inches(0.5)]
    bodies = [x for x in s20.shapes if body_zone(x) and x.height > Inches(0.8)]
    h2l = [l for l in lines if l.startswith('H2')]
    T.set_text(heads[0], 'Câu hỏi 1 · cùng lượng nhãn'); T.set_text(bodies[0], lines[0])
    T.set_text(heads[1], 'Câu hỏi 2 · tiết kiệm nhãn'); T.set_text(bodies[1], h2l[0] if h2l else '—')
    T.set_text(heads[2], 'Độ vững của kết luận')
    T.set_text(bodies[2], [l for l in lines if l.startswith('Phép t')][0] if any(l.startswith('Phép t') for l in lines)
               else lines[1])
    for b in bodies:
        T.set_font_size(b, 16)
    T.set_text(T.footnote_shape(s20), 'Kết luận đi theo quy tắc quyết định ấn định trước; p lớn không được hiểu là tương đương.')
    T.set_notes(s20, '\n'.join(lines))

    s22 = prs.slides[21 + off]
    bodies = [x for x in s22.shapes if body_zone(x) and x.height > Inches(0.5)]
    heads = [x for x in s22.shapes if body_zone(x) and x.height < Inches(0.5)]
    T.set_text(heads[1], 'Chạy lại kết quả v2')
    T.set_text(bodies[0], f'387 CSV. v1: 1.387 log. v2: {F["n_runs"]} log E13'
               + (f', {n_e14} log E14' if n_e14 else '') + ', đều lưu split và dự đoán test.')
    T.set_text(bodies[1], 'python experiments.py E13 · python analyze_e13.py · python section_e13.py · '
                          'python slide/cap_nhat_slide_E13.py')

    # đánh lại số trang
    for i, s in enumerate(prs.slides, start=1):
        ps = T.page_shape(s)
        if ps is not None:
            T.set_text(ps, f'{i:02d}')
    prs.save(dst)
    print(f'đã ghi {dst}: {len(prs.slides)} slide ({len(new)} slide mới)')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
