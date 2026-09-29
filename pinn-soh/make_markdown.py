"""
Xuất bản báo cáo (artifact "PINN-SOH Ít Nhãn") ra Markdown.

    python make_markdown.py            # sinh bao-cao-md/bao-cao.md + bao-cao-md/hinh/

Đọc thẳng report.html — nguồn duy nhất — nên nội dung luôn khớp với artifact.
Ảnh nhúng base64 được tách ra file PNG rời trong hinh/ thay vì để nguyên trong .md
(một file .md chứa 2,4 MB base64 thì không trình soạn thảo nào mở nổi).
"""
from __future__ import annotations
import base64, html as htmlmod, os, re, shutil

SRC = 'report.html'
OUT = 'bao-cao-md'
IMGDIR = 'hinh'

# thứ tự hình đúng như trong report.html — dùng để đặt tên file khi tách base64
FIG_NAMES = [
    'fig_E1_label_efficiency', 'fig_traj_XJTU', 'fig_traj_TJU', 'fig_E5_adapt',
    'fig_ratio_matrix', 'fig_E7_tuned', 'fig_E12_fair', 'fig_E9_arch', 'fig_E10_matrix',
]


# ───────────────────────────────────────────────────────── inline → markdown
def inline(s: str) -> str:
    """Đổi thẻ inline sang markdown. Giữ nguyên ký tự Unicode (≤, ×, β…)."""
    s = re.sub(r'<br\s*/?>', '  \n', s)
    s = re.sub(r'<code>(.*?)</code>', lambda m: '`' + strip(m.group(1)) + '`', s, flags=re.S)
    s = re.sub(r'<(b|strong)>(.*?)</\1>', lambda m: '**' + m.group(2).strip() + '**', s, flags=re.S)
    s = re.sub(r'<(i|em)>(.*?)</\2?>', lambda m: m.group(0), s)          # xử lý riêng bên dưới
    s = re.sub(r'<(i|em)>(.*?)</(?:i|em)>', lambda m: '*' + m.group(2).strip() + '*', s, flags=re.S)
    s = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>',
               lambda m: f'[{strip(m.group(2))}]({m.group(1)})', s, flags=re.S)
    # chỉ số trên/dưới: giữ dạng chữ vì Markdown thuần không có sub/sup
    s = re.sub(r'<sub>(.*?)</sub>', lambda m: '_' + strip(m.group(1)), s, flags=re.S)
    s = re.sub(r'<sup>(.*?)</sup>', lambda m: '^' + strip(m.group(1)), s, flags=re.S)
    s = re.sub(r'<span class="lab">(.*?)</span>', '', s, flags=re.S)      # nhãn công thức tách riêng
    s = re.sub(r'<[^>]+>', '', s)
    s = htmlmod.unescape(s).replace('\u00a0', ' ')
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()


def strip(s: str) -> str:
    return htmlmod.unescape(re.sub(r'<[^>]+>', '', s)).replace('\u00a0', ' ').strip()


def cell(s: str) -> str:
    """Ô bảng: markdown dùng | làm ranh giới nên phải thoát."""
    return inline(s).replace('|', '\\|').replace('\n', ' ')


# ───────────────────────────────────────────────────────── khối → markdown
def table(block: str) -> str:
    head = re.search(r'<thead>(.*?)</thead>', block, re.S)
    body = re.search(r'<tbody>(.*?)</tbody>', block, re.S)
    rows_src = body.group(1) if body else block
    out = []
    if head:
        hs = [cell(c) for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', head.group(1), re.S)]
        out.append('| ' + ' | '.join(hs) + ' |')
        out.append('|' + '|'.join(['---'] * len(hs)) + '|')
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', rows_src, re.S):
        cs = [cell(c) for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)]
        if not cs:
            continue
        if not out:                                   # bảng không có thead
            out.append('| ' + ' | '.join(cs) + ' |')
            out.append('|' + '|'.join(['---'] * len(cs)) + '|')
            continue
        out.append('| ' + ' | '.join(cs) + ' |')
    return '\n'.join(out)


def eq(block: str) -> str:
    lab = re.search(r'<span class="lab">(.*?)</span>', block, re.S)
    body = inline(re.sub(r'<span class="lab">.*?</span>', '', block, flags=re.S))
    tag = f'  — {strip(lab.group(1))}' if lab else ''
    return f'> **{body}**{tag}'


def callout(block: str) -> str:
    txt = '\n'.join(inline(p) for p in re.findall(r'<p>(.*?)</p>', block, re.S)) or inline(block)
    return '\n'.join('> ' + l if l else '>' for l in txt.split('\n'))


def cards(block: str) -> str:
    out = []
    for c in re.findall(r'<div class="card">(.*?)</div>', block, re.S):
        tag = re.search(r'<span class="tag[^"]*">(.*?)</span>', c, re.S)
        h4 = re.search(r'<h4>(.*?)</h4>', c, re.S)
        p = re.search(r'<p>(.*?)</p>', c, re.S)
        line = f'**{strip(h4.group(1))}**' if h4 else ''
        if tag:
            line += f'  (`{strip(tag.group(1))}`)'
        out.append(f'- {line}  \n  {inline(p.group(1)) if p else ""}')
    return '\n'.join(out)


def kv(block: str) -> str:
    parts = re.findall(r'<div>(.*?)</div>', block, re.S)
    rows = ['| | |', '|---|---|']
    for i in range(0, len(parts) - 1, 2):
        rows.append(f'| **{cell(parts[i])}** | {cell(parts[i + 1])} |')
    return '\n'.join(rows)




def _slug(t: str) -> str:
    """Neo kiểu GitHub: thường hoá, bỏ dấu câu, khoảng trắng thành gạch nối."""
    t = t.lower().replace('·', ' ')
    t = re.sub(r'[^\w\s-]', '', t, flags=re.U)
    return re.sub(r'\s+', '-', t.strip())


def _balanced(h: str, i: int) -> int:
    """Trả về chỉ số ngay sau </div> khớp với <div ...> bắt đầu tại i (có đếm lồng nhau)."""
    depth, j = 0, i
    for m in re.finditer(r'<div\b|</div>', h[i:]):
        j = i + m.end()
        depth += 1 if m.group(0) != '</div>' else -1
        if depth == 0:
            return j
    return j


def _extract_nested(body: str, store: dict) -> str:
    """Rút grid2/kv (khối có div lồng) ra khỏi luồng, thay bằng chỗ giữ chỗ."""
    for cls, fn in (('grid2', cards), ('kv', kv)):
        while True:
            m = re.search(r'<div class="' + cls + r'">', body)
            if not m:
                break
            end = _balanced(body, m.start())
            key = f'@@BLOCK{len(store)}@@'
            store[key] = fn(body[m.end():end - len('</div>')])
            body = body[:m.start()] + f'<p>{key}</p>' + body[end:]
    return body


def convert(src: str, outdir: str) -> str:
    h = open(src, encoding='utf-8').read()
    os.makedirs(f'{outdir}/{IMGDIR}', exist_ok=True)

    # ── phần đầu: tiêu đề + mô tả
    doc = []
    t = re.search(r'<h1>(.*?)</h1>', h, re.S)
    eyebrow = re.search(r'<div class="eyebrow">(.*?)</div>', h, re.S)
    lede = re.search(r'<p class="lede">(.*?)</p>', h, re.S)
    doc.append(f'# {strip(t.group(1))}\n' if t else '')
    if eyebrow:
        doc.append(f'*{strip(eyebrow.group(1))}*\n')
    if lede:
        doc.append(inline(lede.group(1)) + '\n')
    doc.append('---\n')

    body = h[h.index('<section'):]
    nested: dict[str, str] = {}
    body = _extract_nested(body, nested)

    # ── tách ảnh base64 ra file rời, thay bằng đường dẫn tương đối
    figs = list(re.finditer(r'src="data:image/png;base64,([^"]+)"', body))
    for i, m in enumerate(figs):
        name = FIG_NAMES[i] if i < len(FIG_NAMES) else f'hinh-{i + 1}'
        open(f'{outdir}/{IMGDIR}/{name}.png', 'wb').write(base64.b64decode(m.group(1)))
    def fig_path(i):
        return f'{IMGDIR}/{FIG_NAMES[i] if i < len(FIG_NAMES) else f"hinh-{i+1}"}.png'
    n = [0]
    body = re.sub(r'src="data:image/png;base64,[^"]+"',
                  lambda m: (lambda p: f'src="{p}"')(fig_path(n.__setitem__(0, n[0] + 1) or n[0] - 1)),
                  body)

    # ── sơ đồ SVG nội tuyến: lưu thành file .svg riêng
    svgs = re.findall(r'(<svg.*?</svg>)', body, re.S)
    for i, sv in enumerate(svgs):
        sv2 = re.sub(r'var\(--[a-z0-9-]+\)', lambda m: {
            'var(--ink)': '#1b2330', 'var(--ink-2)': '#4a5563', 'var(--line)': '#dfe3df',
            'var(--code-bg)': '#eef1ee', 'var(--accent)': '#177a68', 'var(--accent-soft)': '#e3f1ec',
            'var(--warn)': '#b8531f', 'var(--warn-soft)': '#f8ebe2', 'var(--surface)': '#ffffff',
        }.get(m.group(0), '#1b2330'), sv)
        open(f'{outdir}/{IMGDIR}/so-do-kien-truc-{i + 1}.svg', 'w', encoding='utf-8').write(
            '<?xml version="1.0" encoding="UTF-8"?>\n' + sv2)

    # ── duyệt tuần tự các khối cấp cao
    BLOCK = re.compile(
        r'<h2[^>]*>(?P<h2>.*?)</h2>'
        r'|<h3[^>]*>(?P<h3>.*?)</h3>'
        r'|<div class="tw">(?P<tw>.*?)</div>'
        r'|<figure>(?P<fig>.*?)</figure>'
        r'|<div class="svgwrap">(?P<svg>.*?)</div>\s*(?=<p|<div|<h|<section|</section)'
        r'|<div class="eq">(?P<eq>.*?)</div>'
        r'|<div class="callout(?P<cw>[^"]*)">(?P<callout>.*?)</div>\s*(?=<p|<div|<h|<section|</section|<ul|<ol)'
        r'|<pre>(?P<pre>.*?)</pre>'
        r'|<ul>(?P<ul>.*?)</ul>'
        r'|<ol>(?P<ol>.*?)</ol>'
        r'|<p>(?P<p>.*?)</p>',
        re.S)

    fi = [0]
    for m in BLOCK.finditer(body):
        g = m.groupdict()
        if g['h2'] is not None:
            txt = re.sub(r'<span class="num">(.*?)</span>', r'\1 · ', g['h2'], flags=re.S)
            doc.append(f'\n## {strip(txt)}\n')
        elif g['h3'] is not None:
            doc.append(f'\n### {strip(g["h3"])}\n')
        elif g['tw'] is not None:
            doc.append(table(g['tw']) + '\n')
        elif g['fig'] is not None:
            cap = re.search(r'<figcaption>(.*?)</figcaption>', g['fig'], re.S)
            src_ = re.search(r'src="([^"]+)"', g['fig'])
            alt = strip(cap.group(1))[:70] if cap else 'hình'
            doc.append(f'![{alt}]({src_.group(1) if src_ else ""})\n')
            if cap:
                doc.append(f'*Hình {fi[0] + 1}. {inline(cap.group(1))}*\n')
            fi[0] += 1
        elif g['svg'] is not None:
            doc.append(f'![Sơ đồ kiến trúc]({IMGDIR}/so-do-kien-truc-1.svg)\n')
            doc.append('*Sơ đồ kiến trúc: mạng nghiệm và động học xám.*\n')
        elif g['eq'] is not None:
            doc.append(eq(g['eq']) + '\n')
        elif g['callout'] is not None:
            doc.append(callout(g['callout']) + '\n')
        elif g['pre'] is not None:
            code = htmlmod.unescape(re.sub(r'</?code>', '', g['pre'])).strip('\n')
            lang = 'python' if re.search(r'^\s*(from|import)\s', code, re.M) else 'bash'
            doc.append(f'```{lang}\n' + code + '\n```\n')
        elif g['ul'] is not None or g['ol'] is not None:
            ordered = g['ol'] is not None
            items = re.findall(r'<li>(.*?)</li>', g['ul'] or g['ol'], re.S)
            doc.append('\n'.join(
                (f'{i + 1}. ' if ordered else '- ') + inline(it) for i, it in enumerate(items)) + '\n')
        elif g['p'] is not None:
            raw = g['p'].strip()
            if raw in nested:
                doc.append(nested[raw] + '\n')
                continue
            txt = inline(g['p'])
            if txt:
                doc.append(txt + '\n')

    foot = re.search(r'<footer>(.*?)</footer>', h, re.S)
    if foot:
        doc.append('\n---\n\n' + inline(foot.group(1)) + '\n')

    md = '\n'.join(doc)
    # chèn mục lục ngay sau phần mở đầu
    heads = re.findall(r'^## (.+)$', md, re.M)
    if heads:
        toc_md = '## Mục lục\n\n' + '\n'.join(
            f'{i + 1}. [{h}](#{_slug(h)})' for i, h in enumerate(heads)) + '\n'
        md = md.replace('---\n', '---\n\n' + toc_md + '\n---\n', 1)
    md = re.sub(r'\n{4,}', '\n\n\n', md)
    path = f'{outdir}/bao-cao.md'
    open(path, 'w', encoding='utf-8').write(md)
    return path


if __name__ == '__main__':
    shutil.rmtree(OUT, ignore_errors=True)
    p = convert(SRC, OUT)
    n_img = len(os.listdir(f'{OUT}/{IMGDIR}'))
    kb = os.path.getsize(p) / 1024
    print(f'{p} — {kb:.0f} KB, {n_img} tệp hình trong {OUT}/{IMGDIR}/')
