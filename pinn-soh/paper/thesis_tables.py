"""
Đọc các bảng số liệu của báo cáo đồ án (archive/thesis/do-an/main.tex) để bản thảo không phải gõ lại số nào.

Các bảng đó sinh từ results/*.csv của lần chạy công bố (commit 2d66f75); thư mục runs/ và results/ của lần chạy đó
không còn, nên bản thảo lấy số trực tiếp từ mã LaTeX của báo cáo. Khi chạy lại thí nghiệm, các bảng tương ứng trong
results/ là nguồn chính.

    from thesis_tables import table
    rows = table('tab:big')     # [['XJTU', 'Ridge', 18.22, 15.54, 14.81, 14.09], ...]
"""
from __future__ import annotations
import os, re

THESIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'archive', 'thesis', 'do-an',
                      'main.tex')


def _block(label: str) -> str:
    s = open(THESIS, encoding='utf-8').read()
    i = s.index(f'\\label{{{label}}}')
    a = s.rfind('\\begin{table}', 0, i)
    b = s.index('\\end{table}', i)
    return s[a:b]


def _cell(x: str):
    """Một ô: số (dấu phẩy thập phân, dấu chấm hàng nghìn, dấu trừ Unicode) -> float; còn lại -> chuỗi đã bỏ định dạng.
    Dấu * và † (có ý nghĩa thống kê) và ngoặc vuông của khoảng tin cậy được giữ trong chuỗi khi ô không phải một số."""
    t = x.strip()
    t = re.sub(r'\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', t)
    t = t.replace('{,}', ',').replace('\\,', '').replace('\\%', '%').replace('$', '')
    t = t.replace('^{*}', '*').replace('^{\\dagger}', '†').replace('−', '-').replace('--', '–')
    t = re.sub(r'\\addlinespace\[[^\]]*\]', '', t).strip()
    num = t.replace('.', '').replace(',', '.') if re.fullmatch(r'-?[\d.]+(,\d+)?', t) else None
    if num is not None:
        return float(num)
    return t


def table(label: str):
    """Các hàng của bảng (giữa \\midrule và \\bottomrule); ô đầu trống được điền bằng ô đầu của hàng trước."""
    blk = _block(label)
    body = blk[blk.index('\\midrule') + len('\\midrule'):blk.index('\\bottomrule')]
    rows, last = [], None
    for raw in body.split('\\\\'):
        raw = re.sub(r'\\addlinespace\[[^\]]*\]', '', raw).strip()
        if not raw or raw.startswith('\\cmidrule'):
            continue
        cells = [_cell(c) for c in raw.split('&')]
        if cells[0] == '' and last is not None:
            cells[0] = last
        last = cells[0]
        rows.append(cells)
    return rows


def ratio(cell: str):
    """'0,84 [0,76; 0,93]*' -> (0.84, 0.76, 0.93, True)."""
    m = re.match(r'\s*(-?[\d,]+)\s*\[(-?[\d,]+);\s*(-?[\d,]+)\]\s*(\*?)', cell)
    f = lambda v: float(v.replace(',', '.'))
    return f(m.group(1)), f(m.group(2)), f(m.group(3)), m.group(4) == '*'


if __name__ == '__main__':
    for lab in ('tab:datastats', 'tab:big', 'tab:cmp', 'tab:allmae', 'tab:allrm', 'tab:source', 'tab:e28', 'tab:e27'):
        rows = table(lab)
        print(lab, len(rows), rows[0], rows[-1])
