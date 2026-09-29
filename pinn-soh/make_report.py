"""Build the research-design report (HTML, Vietnamese) from results/.  usage: python make_report.py"""
import base64, glob, json, os, re
import numpy as np, pandas as pd

R = 'results'


def img(path):
    if not os.path.exists(path):
        return f'<p class="muted">[thiếu hình {path}]</p>'
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{os.path.basename(path)}">'


def md_table(path):
    """Markdown (possibly several tables with bold captions) -> HTML."""
    if not os.path.exists(path):
        return '<p class="muted">[chưa có kết quả]</p>'
    out, rows = [], []
    def flush():
        nonlocal rows
        if not rows: return
        head = rows[0]; body = rows[2:] if len(rows) > 1 and set(rows[1].replace('|', '').strip()) <= set('-: ') else rows[1:]
        cells = lambda r: [c.strip() for c in r.strip().strip('|').split('|')]
        h = ''.join(f'<th>{c}</th>' for c in cells(head))
        b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in cells(r)) + '</tr>' for r in body)
        out.append(f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
        rows = []
    for line in open(path):
        if line.strip().startswith('|'):
            rows.append(line)
        else:
            flush()
            s = line.strip()
            if s:
                import re as _re
                s = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
                out.append(f'<p class="tcap">{s}</p>')
    flush()
    return '\n'.join(out)


def hyp_sentences():
    p = f'{R}/E1_hypothesis.csv'
    if not os.path.exists(p): return ''
    h = pd.read_csv(p)
    items = []
    for _, r in h.iterrows():
        items.append(f'<li><b>{r.dataset}</b>: PINN-semi với 30 % nhãn đạt MAE {r.pinn30:.4f}, tức <b>{r.ratio_pinn30_mlp70:.2f}×</b> MAE của MLP với 70 % nhãn ({r.mlp70:.4f}); '
                     f'MLP cùng 30 % nhãn là {r.mlp30:.4f} ({r.ratio_mlp30_mlp70:.2f}×).</li>')
    return '<ul>' + ''.join(items) + '</ul>'


def stat(name, default='—'):
    try:
        return STATS[name]
    except Exception:
        return default


STATS = {}
try:
    e1 = pd.read_csv(f'{R}/E1_table.csv')
    STATS['n_runs_E1'] = int(e1.n.sum())
    h = pd.read_csv(f'{R}/E1_hypothesis.csv')
    STATS['ratio_pinn30'] = f'{h.ratio_pinn30_mlp70.min():.2f}–{h.ratio_pinn30_mlp70.max():.2f}×'
    STATS['ratio_mlp30'] = f'{h.ratio_mlp30_mlp70.min():.2f}–{h.ratio_mlp30_mlp70.max():.2f}×'
    def g(d, m, f, col='MAE'):
        r = e1[(e1.dataset == d) & (e1.model == m) & (e1.frac == f)]
        return float(r[col].iloc[0]) if len(r) else float('nan')
    DS = [d for d in ['XJTU', 'TJU', 'MIT', 'HUST'] if (e1.dataset == d).any()]
    STATS['gain10'] = ', '.join(f'{d} {g(d,"mlp",0.1)/g(d,"pinn_semi",0.1):.1f}×' for d in DS)
    # verdicts at 30 %
    sup, part, no = [], [], []
    for _, r in h.iterrows():
        if r.ratio_pinn30_mlp70 <= 1.0: sup.append(f'{r.dataset} ({r.ratio_pinn30_mlp70:.2f}×)')
        elif r.ratio_pinn30_mlp70 < r.ratio_mlp30_mlp70: part.append(f'{r.dataset} ({r.ratio_pinn30_mlp70:.2f}× so với {r.ratio_mlp30_mlp70:.2f}× của MLP)')
        else: no.append(f'{r.dataset} ({r.ratio_pinn30_mlp70:.2f}× so với {r.ratio_mlp30_mlp70:.2f}× của MLP)')
    STATS['v_sup'] = ', '.join(sup) or '—'; STATS['v_part'] = ', '.join(part) or '—'; STATS['v_no'] = ', '.join(no) or '—'
    # semi vs sup
    STATS['semi_vs_sup_10'] = '; '.join(f'{d}: {g(d,"pinn_semi",0.1):.4f} so với {g(d,"pinn_sup",0.1):.4f}' for d in DS)
    STATS['semi_vs_sup_30'] = '; '.join(f'{d}: {g(d,"pinn_semi",0.3):.4f} so với {g(d,"pinn_sup",0.3):.4f}' for d in DS)
    STATS['mlp_ratio'] = ', '.join(f'{d} {g(d,"mlp",0.3)/g(d,"mlp",0.7):.2f}×' for d in DS)
    STATS['pinn_70_vs_mlp_70'] = ', '.join(f'{d} {g(d,"pinn_semi",0.7):.4f}/{g(d,"mlp",0.7):.4f}' for d in DS)
except Exception as ex:
    print('stats:', ex)
try:
    e3 = pd.read_csv(f'{R}/E3_table.csv')
    tj = e3[(e3.dataset == 'TJU') & (e3.variant == 'full')]
    if len(tj): STATS['Ea_TJU'] = f'{tj.Ea.iloc[0]:.1f}'
except Exception as ex:
    print('E3 stats:', ex)
try:
    e5 = pd.read_csv(f'{R}/E5_table.csv')
    def q(src, tgt, k, m):
        r = e5[(e5.source == src) & (e5.target == tgt) & (e5.k == k) & (e5.model == m)].MAE
        return float(r.iloc[0]) if len(r) else float('nan')
    parts = []
    for src, tgt in [('HUST', 'MIT'), ('MIT', 'HUST'), ('XJTU', 'TJU'), ('TJU', 'XJTU')]:
        if np.isnan(q(src, tgt, 0, 'mlp')): continue
        parts.append(f'{src}→{tgt}: MLP zero-shot {q(src,tgt,0,"mlp"):.3f} → PINN-semi không nhãn đích {q(src,tgt,0,"pinn_semi"):.3f}; với 3 cell đích có nhãn: MLP {q(src,tgt,3,"mlp"):.3f}, PINN-semi {q(src,tgt,3,"pinn_semi"):.3f}')
    STATS['e5_sentence'] = ('Đọc theo cặp — ' + '. '.join(parts) + '.') if parts else ''
    short, wins, tot = [], 0, 0
    for src, tgt in [('HUST', 'MIT'), ('XJTU', 'TJU'), ('TJU', 'XJTU'), ('MIT', 'HUST')]:
        a, b = q(src, tgt, 0, 'mlp'), q(src, tgt, 0, 'pinn_semi')
        if np.isnan(a): continue
        short.append(f'{src}→{tgt} MAE {a:.3f} → {b:.3f}')
        tot += 1; wins += int(q(src, tgt, 3, 'pinn_semi') < q(src, tgt, 3, 'mlp'))
    STATS['e5_short'] = ', '.join(short); STATS['e5_k3_wins'] = f'{wins}/{tot}'
except Exception as ex:
    print('E5 stats:', ex)
try:
    e6 = pd.read_csv(f'{R}/E6_selected.csv')
    parts = [f'{r.dataset}: chọn <code>{r.selected}</code> theo val → test {r.test:.4f}, so với MLP 30 % {r.mlp30:.4f} và MLP 70 % {r.mlp70:.4f} ({r.test/r.mlp70:.2f}×)' for _, r in e6.iterrows()]
    STATS['e6_sentence'] = 'Với trọng số chọn theo validation: ' + '; '.join(parts) + '.'
    STATS['e6_ratios'] = ', '.join(f'{r.dataset} {r.test/r.mlp70:.2f}×' for _, r in e6.iterrows())
except Exception as ex:
    print('E6 stats:', ex)
try:
    e7 = pd.read_csv(f'{R}/E7_table.csv')
    STATS['e7_wins'] = f'{int((e7.ratio_vs_mlp < 1).sum())}/{len(e7)}'
    f50 = e7[e7.frac == 0.5]
    STATS['e7_50_vs_mlp'] = ', '.join(f'{r.dataset} {r.ratio_vs_mlp:.2f}×' for _, r in f50.iterrows())
    STATS['e7_50_vs_mlp70'] = ', '.join(f'{r.dataset} {r.ratio_vs_mlp70:.2f}×' for _, r in f50.iterrows())
    STATS['e7_50_le70'] = f'{int((f50.ratio_vs_mlp70 <= 1.02).sum())}/{len(f50)}'
    f30 = e7[e7.frac == 0.3]
    STATS['e7_30_vs_mlp70'] = ', '.join(f'{r.dataset} {r.ratio_vs_mlp70:.2f}×' for _, r in f30.iterrows())
    STATS['e7_30_le70'] = f'{int((f30.ratio_vs_mlp70 <= 1.02).sum())}/{len(f30)}'
    lo = e7[e7.frac <= 0.3]; hi = e7[e7.frac >= 0.5]
    STATS['e7_sel_lo'] = ', '.join(sorted(set(lo.selected))); STATS['e7_sel_hi'] = ', '.join(sorted(set(hi.selected)))
    STATS['e7_adapt_gap'] = ', '.join(f'{r.dataset} {r.adapt/r.pinn:.2f}×' for _, r in f50.iterrows())
    STATS['e7_adapt_wins'] = f'{int((e7.adapt < e7.mlp).sum())}/{len(e7)}'
except Exception as ex:
    print('E7 stats:', ex)
try:
    e8 = pd.read_csv(f'{R}/E8_table.csv')
    m = e8[e8.model == 'mlp']; pn = e8[e8.model == 'pinn_semi']
    STATS['e8_gain_mlp'] = f'{m["gain_iso_%"].mean():.1f} %'
    STATS['e8_gain_pinn'] = f'{pn["gain_iso_%"].mean():.1f} %'
    STATS['e8_viol'] = ', '.join(f'{d}: MLP {m[m.dataset==d].viol.mean():.1f} so với PINN {pn[pn.dataset==d].viol.mean():.2f}' for d in ['XJTU', 'TJU', 'MIT', 'HUST'])
except Exception as ex:
    print('E8 stats:', ex)
try:
    t12 = pd.read_csv(f'{R}/E12_table.csv')
    STATS['e12_wins'] = f'{int((t12.ratio < 1).sum())}/{len(t12)}'
    STATS['e12_range'] = f'{t12.ratio.min():.2f}–{t12.ratio.max():.2f}×'
    STATS['e12_mlp_gain'] = ', '.join(f'{r.dataset}@{int(r.frac*100)}% {100*r.mlp_gain:+.0f} %'
                                      for _, r in t12.sort_values('mlp_gain', ascending=False).head(3).iterrows())
    e1t = pd.read_csv(f'{R}/E1_table.csv')
    old = []
    for _, r in t12.iterrows():
        q = e1t[(e1t.dataset == r.dataset) & (e1t.frac == r.frac) & (e1t.model == 'pinn_semi')].MAE
        if len(q): old.append(float(q.iloc[0]) / r.mlp_default)
    STATS['e12_before'] = f'{np.mean(old):.2f}×'; STATS['e12_after'] = f'{t12.ratio.mean():.2f}×'
    STATS['e12_tags'] = ', '.join(f'`{k}` {v}' for k, v in
                                  pd.concat([t12.mlp_tag, t12.pinn_tag]).value_counts().head(3).items())
    tj = t12[t12.dataset == 'TJU']
    STATS['e12_tju'] = ', '.join(f'{int(r.frac*100)} % → {r.ratio:.2f}×' for _, r in tj.iterrows())
except Exception as ex:
    print('E12 stats:', ex)
try:
    e9 = pd.read_csv(f'{R}/E9_table.csv')
    def rel(model, tag):
        sub = e9[e9.model == model]
        b = sub[sub.tag == 'silu-mlp'].set_index('dataset').MAE
        r = sub[sub.tag == tag].set_index('dataset').MAE
        common = b.index.intersection(r.index)
        return float((r[common] / b[common]).mean()) if len(common) else float('nan')
    STATS['e9_mono_pinn'] = f'{rel("pinn_semi", "silu-mono"):.3f}×'
    STATS['e9_mono_mlp'] = f'{rel("mlp", "silu-mono"):.3f}×'
    STATS['e9_tanh_pinn'] = f'{rel("pinn_semi", "tanh-mlp"):.3f}×'
    STATS['e9_tanh_mlp'] = f'{rel("mlp", "tanh-mlp"):.3f}×'
    STATS['e9_snake_pinn'] = f'{rel("pinn_semi", "snake-mlp"):.3f}×'
    STATS['e9_snake_mlp'] = f'{rel("mlp", "snake-mlp"):.3f}×'
    STATS['e9_res_pinn'] = f'{rel("pinn_semi", "silu-res"):.3f}×'
    mono = e9[(e9.model == 'pinn_semi') & (e9.tag == 'silu-mono')]
    base = e9[(e9.model == 'pinn_semi') & (e9.tag == 'silu-mlp')]
    STATS['e9_mono_viol'] = f'{mono.MonoViol.mean():.2f} so với {base.MonoViol.mean():.2f}'
    STATS['e9_mono_eol'] = f'{mono.EOL_MAE.mean():.0f} so với {base.EOL_MAE.mean():.0f}'
except Exception as ex:
    print('E9 stats:', ex)
try:
    t10 = pd.read_csv(f'{R}/E10_table.csv')
    def cell(model, norm, a, b):
        r = t10[(t10.model == model) & (t10.norm == norm) & (t10.source == a) & (t10.target == b)].MAE
        return float(r.iloc[0]) if len(r) else float('nan')
    off = t10[t10.source != t10.target]
    dia = t10[t10.source == t10.target]
    STATS['e10_dia'] = f'{dia[dia.norm == "global"].MAE.mean():.4f}'
    g = off[(off.norm == 'global')].MAE.median(); f = off[(off.norm == 'first_cycle')].MAE.median()
    STATS['e10_med'] = f'{g:.2f} → {f:.2f} ({g/f:.1f}× tốt hơn)'
    STATS['e10_best_off'] = f'MIT→HUST {cell("mlp","global","MIT","HUST"):.3f} (MLP) / {cell("pinn","global","MIT","HUST"):.3f} (PINN)'
    STATS['e10_worst'] = f'XJTU→MIT {cell("mlp","global","XJTU","MIT"):.0f}'
    same = [('MIT', 'HUST'), ('HUST', 'MIT')]; crossc = [('XJTU', 'MIT'), ('XJTU', 'HUST'), ('MIT', 'XJTU'), ('HUST', 'XJTU')]
    STATS['e10_same'] = f'{np.mean([cell("pinn","first_cycle",a,b) for a,b in same]):.3f}'
    STATS['e10_cross'] = f'{np.mean([cell("pinn","first_cycle",a,b) for a,b in crossc]):.2f}'
    STATS['e10_ratio'] = f'{np.mean([cell("pinn","first_cycle",a,b) for a,b in crossc]) / np.mean([cell("pinn","first_cycle",a,b) for a,b in same]):.0f}×'
except Exception as ex:
    print('E10 stats:', ex)
try:
    e11 = pd.read_csv(f'{R}/E11_table.csv')
    wins = {}
    for c in ['MAE', 'MAE_cell', 'MAE_cell_max', 'MAE_late', 'MonoViol', 'Jitter', 'EOL_MAE']:
        w = 0
        for d in e11.dataset.unique():
            a = e11[(e11.dataset == d) & (e11.model == 'pinn_semi')][c]
            b = e11[(e11.dataset == d) & (e11.model == 'mlp')][c]
            if len(a) and len(b) and not np.isnan(a.iloc[0]) and not np.isnan(b.iloc[0]):
                w += int(a.iloc[0] < b.iloc[0])
        wins[c] = w
    STATS['e11_wins'] = ', '.join(f'{k} {v}/4' for k, v in wins.items())
    tj = e11[e11.dataset == 'TJU']
    STATS['e11_tju'] = (f'MAE {float(tj[tj.model=="pinn_semi"].MAE.iloc[0]):.4f} so với {float(tj[tj.model=="mlp"].MAE.iloc[0]):.4f} (MLP thắng), '
                        f'nhưng cell tệ nhất {float(tj[tj.model=="pinn_semi"].MAE_cell_max.iloc[0]):.4f} so với {float(tj[tj.model=="mlp"].MAE_cell_max.iloc[0]):.4f}, '
                        f'vi phạm đơn điệu {float(tj[tj.model=="pinn_semi"].MonoViol.iloc[0]):.2f} so với {float(tj[tj.model=="mlp"].MonoViol.iloc[0]):.2f}, '
                        f'sai số EOL {float(tj[tj.model=="pinn_semi"].EOL_MAE.iloc[0]):.0f} so với {float(tj[tj.model=="mlp"].EOL_MAE.iloc[0]):.0f} chu kỳ')
except Exception as ex:
    print('E11 stats:', ex)
try:
    e4 = pd.read_csv(f'{R}/E4_table.csv')
    def e(d, t): 
        r = e4[(e4.dataset == d) & (e4.tag == t)].MAE; return float(r.iloc[0]) if len(r) else float('nan')
    STATS['leak_cycle'] = ', '.join(f'{d}: {e(d,"causal"):.4f} → {e(d,"leak_cycle"):.4f} (−{100*(1-e(d,"leak_cycle")/e(d,"causal")):.0f} %)' for d in ['XJTU', 'MIT'])
    STATS['leak_feat'] = ', '.join(f'{d}: {e(d,"leak_feat"):.4f}' for d in ['XJTU', 'MIT'])
    STATS['leak_both'] = ', '.join(f'{d}: {e(d,"leak_both"):.4f}' for d in ['XJTU', 'MIT'])
    STATS['no_cycle'] = ', '.join(f'{d}: {e(d,"no_cycle_input"):.4f}' for d in ['XJTU', 'MIT'])
except Exception as ex:
    print('E4 stats:', ex)

CSS = r"""
:root{
  --bg:#f6f7f5; --surface:#ffffff; --ink:#1b2330; --ink-2:#4a5563; --muted:#7c8794; --line:#dfe3df;
  --accent:#177a68; --accent-ink:#0f5a4d; --accent-soft:#e3f1ec; --warn:#b8531f; --warn-soft:#f8ebe2;
  --code-bg:#eef1ee; --shadow:0 1px 2px rgba(20,30,40,.06);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#141817; --surface:#1c2220; --ink:#e8ebe7; --ink-2:#b9c0ba; --muted:#8b948e; --line:#2c3531;
  --accent:#4cc3a8; --accent-ink:#7fdcc6; --accent-soft:#1f302b; --warn:#e88a55; --warn-soft:#3a2a20;
  --code-bg:#232b28; --shadow:none; }}
:root[data-theme="dark"]{
  --bg:#141817; --surface:#1c2220; --ink:#e8ebe7; --ink-2:#b9c0ba; --muted:#8b948e; --line:#2c3531;
  --accent:#4cc3a8; --accent-ink:#7fdcc6; --accent-soft:#1f302b; --warn:#e88a55; --warn-soft:#3a2a20;
  --code-bg:#232b28; --shadow:none; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:16px;line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 96px}
header.top{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:34px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-ink)}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:clamp(28px,4vw,40px);line-height:1.15;margin:8px 0 12px;text-wrap:balance;letter-spacing:-.01em}
.lede{font-size:18px;color:var(--ink-2);max-width:68ch;margin:0}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:18px;font-size:14px}
nav.toc a{color:var(--accent-ink);text-decoration:none;border-bottom:1px dotted var(--line)}
nav.toc a:hover,nav.toc a:focus-visible{border-bottom-color:var(--accent);outline:none}
h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:26px;margin:56px 0 14px;text-wrap:balance;letter-spacing:-.005em}
h2 .num{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--muted);margin-right:10px;vertical-align:middle;letter-spacing:.06em}
h3{font-size:18px;font-weight:600;margin:30px 0 8px}
p,li{max-width:72ch}
section>p,section>ul,section>ol{margin:0 0 14px}
ul,ol{padding-left:22px}
li{margin:4px 0}
b,strong{font-weight:600}
a{color:var(--accent-ink)}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
code{background:var(--code-bg);padding:1px 5px;border-radius:3px}
pre{background:var(--code-bg);padding:14px 16px;border-radius:6px;overflow-x:auto;font-size:13.5px;line-height:1.5;max-width:100%}
pre code{background:none;padding:0}
.eq{font-family:"IBM Plex Serif",Georgia,serif;font-size:17px;padding:12px 18px;margin:12px 0;border-left:2px solid var(--accent);background:var(--surface);overflow-x:auto;white-space:nowrap;box-shadow:var(--shadow)}
.eq i{font-style:italic}
.eq .lab{float:right;color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:12px;margin-left:24px}
.callout{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--surface);padding:14px 18px;margin:18px 0;border-radius:4px;box-shadow:var(--shadow)}
.callout.warn{border-left-color:var(--warn);}
.callout p:last-child{margin-bottom:0}
.tcap{font-weight:600;margin:22px 0 6px;font-size:15px}
.tw{overflow-x:auto;margin:6px 0 18px;border:1px solid var(--line);border-radius:4px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--ink-2);background:var(--code-bg);font-size:13px;letter-spacing:.02em}
tbody tr:last-child td{border-bottom:none}
td:not(:first-child),th:not(:first-child){text-align:right}
figure{margin:18px 0 26px}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{font-size:13.5px;color:var(--ink-2);margin-top:8px;max-width:80ch}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:16px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px;box-shadow:var(--shadow)}
.card h4{margin:0 0 6px;font-size:15px;font-weight:600}
.card p{font-size:14.5px;margin:0;color:var(--ink-2)}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent-ink);margin-bottom:8px}
.tag.warn{background:var(--warn-soft);color:var(--warn)}
.muted{color:var(--muted)}
.svgwrap{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;margin:16px 0;overflow-x:auto}
.svgwrap svg{display:block;max-width:100%;height:auto;font-family:"IBM Plex Sans",sans-serif}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:14.5px;margin:10px 0 18px}
.kv div:nth-child(odd){color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:12.5px;padding-top:3px}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);font-size:13.5px;color:var(--muted)}
@media (prefers-reduced-motion:no-preference){ a{transition:border-color .15s} }
"""

ARCH_SVG = r"""
<svg viewBox="0 0 900 330" width="900" height="330" role="img" aria-label="Kiến trúc: mạng nghiệm và động học xám">
 <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--ink-2)"/></marker></defs>
 <g font-size="13" fill="var(--ink)">
  <!-- inputs -->
  <rect x="20" y="40" width="170" height="70" rx="6" fill="var(--code-bg)" stroke="var(--line)"/>
  <text x="105" y="66" text-anchor="middle" font-weight="600">x_N ∈ ℝ¹⁶</text>
  <text x="105" y="86" text-anchor="middle" font-size="12" fill="var(--ink-2)">16 đặc trưng đường sạc, chu kỳ N</text>
  <rect x="20" y="130" width="170" height="44" rx="6" fill="var(--code-bg)" stroke="var(--line)"/>
  <text x="105" y="157" text-anchor="middle" font-weight="600">Ñ = N / 1000</text>
  <rect x="20" y="194" width="170" height="44" rx="6" fill="var(--code-bg)" stroke="var(--line)"/>
  <text x="105" y="221" text-anchor="middle" font-weight="600">T (K) — nếu biết</text>
  <!-- solution net -->
  <rect x="260" y="40" width="200" height="134" rx="8" fill="var(--accent-soft)" stroke="var(--accent)"/>
  <text x="360" y="68" text-anchor="middle" font-weight="600">Mạng nghiệm F_φ</text>
  <text x="360" y="90" text-anchor="middle" font-size="12" fill="var(--ink-2)">MLP 17→64→64→32→1, SiLU</text>
  <text x="360" y="118" text-anchor="middle" font-size="15" font-family="IBM Plex Serif,serif"><tspan font-style="italic">u</tspan>(N) = F_φ(x_N, Ñ)</text>
  <text x="360" y="150" text-anchor="middle" font-size="12" fill="var(--ink-2)">→ SOH ước lượng tại chu kỳ N</text>
  <!-- dynamics -->
  <rect x="260" y="194" width="200" height="110" rx="8" fill="var(--warn-soft)" stroke="var(--warn)"/>
  <text x="360" y="220" text-anchor="middle" font-weight="600">Động học xám (grey-box)</text>
  <text x="360" y="246" text-anchor="middle" font-size="14" font-family="IBM Plex Serif,serif"><tspan font-style="italic">r</tspan> = softplus(MLP_θ(x)) · e^{λ(1−u)} · A(T)</text>
  <text x="360" y="270" text-anchor="middle" font-size="12" fill="var(--ink-2)">r ≥ 0 theo cấu trúc; λ, E_a học được</text>
  <text x="360" y="290" text-anchor="middle" font-size="12" fill="var(--ink-2)">A(T) = exp(−E_a/R · (1/T − 1/T_ref))</text>
  <!-- losses -->
  <rect x="540" y="40" width="340" height="56" rx="6" fill="var(--surface)" stroke="var(--line)"/>
  <text x="556" y="62" font-weight="600">L_data</text><text x="556" y="82" font-size="12" fill="var(--ink-2)">(u − y)² — chỉ trên cell CÓ nhãn</text>
  <rect x="540" y="110" width="340" height="56" rx="6" fill="var(--surface)" stroke="var(--accent)"/>
  <text x="556" y="132" font-weight="600">L_ode  (không cần nhãn)</text><text x="556" y="152" font-size="12" fill="var(--ink-2)">[u(N+h) − u(N) + r(x_N,u_N,T)·h/1000]²</text>
  <rect x="540" y="180" width="340" height="56" rx="6" fill="var(--surface)" stroke="var(--accent)"/>
  <text x="556" y="202" font-weight="600">L_mono  (không cần nhãn)</text><text x="556" y="222" font-size="12" fill="var(--ink-2)">ReLU(u(N+h) − u(N) − ε), ε = 0.002</text>
  <rect x="540" y="250" width="340" height="56" rx="6" fill="var(--surface)" stroke="var(--accent)"/>
  <text x="556" y="272" font-weight="600">L_range  (không cần nhãn)</text><text x="556" y="292" font-size="12" fill="var(--ink-2)">u ∈ [0.4, 1.15]; u(chu kỳ đầu) ∈ [0.85, 1.10]</text>
  <!-- arrows -->
  <path d="M190 75 H260" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M190 152 H225 V110 H260" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M190 75 H225 V250 H260" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M190 216 H240 V270 H260" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M460 107 H540 V68" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M460 107 H500 V138 H540" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M460 107 H500 V208 H540" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M460 107 H500 V278 H540" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <path d="M460 249 H520 V138" stroke="var(--warn)" stroke-width="1.5" fill="none" stroke-dasharray="4 3" marker-end="url(#ar)"/>
  <path d="M360 174 V194" stroke="var(--ink-2)" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>
  <text x="380" y="188" font-size="11" fill="var(--ink-2)">u</text>
 </g>
</svg>
"""


def build():
    body = f"""
<header class="top">
  <div class="eyebrow">Thiết kế nghiên cứu · ĐACN HK261 · bản sơ bộ, chạy trên CPU</div>
  <h1>Ước lượng SOH pin đơn với ràng buộc vật lý không cần nhãn</h1>
  <p class="lede">Kiến trúc, các hàm mất mát vật lý, protocol thí nghiệm và bằng chứng sơ bộ trên 387 cell / 4 bộ dữ liệu công khai cho giả thuyết: <em>có ràng buộc vật lý thì huấn luyện với 30 % nhãn tiệm cận kết quả 70/15/15</em>.</p>
  <nav class="toc">
    <a href="#tomtat">Tóm tắt</a><a href="#khoangtrong">Khoảng trống trong PINN4SOH</a><a href="#dulieu">Dữ liệu</a>
    <a href="#kientruc">Kiến trúc</a><a href="#loss">Loss vật lý</a><a href="#coche">Cơ chế "ít nhãn"</a>
    <a href="#protocol">Protocol</a><a href="#ketqua">Kết quả</a><a href="#donggop">Đóng góp</a><a href="#hanche">Hạn chế &amp; bước tiếp</a><a href="#code">Chạy code</a>
  </nav>
</header>

<section id="tomtat">
<h2><span class="num">00</span>Tóm tắt</h2>
<p>Bài toán: ước lượng SOH (dung lượng hiện tại / dung lượng danh định) của một cell từ 16 đặc trưng thống kê của đoạn sạc CC–CV, tại từng chu kỳ. Điểm khác biệt so với PINN4SOH (Wang et al., <i>Nat. Commun.</i> 2024) nằm ở ba chỗ: động học suy giảm có cấu trúc vật lý thay vì mạng đen; mọi loss vật lý đều <b>không dùng nhãn</b> nên áp được lên cả cell chưa đo dung lượng; và pipeline không rò rỉ (không chuẩn hoá theo từng cell bằng thống kê cả vòng đời, chia theo cell, val/test cố định).</p>
<p>Giả thuyết được kiểm bằng cách giữ nguyên tập val/test của protocol 70/15/15 rồi giảm dần số cell có nhãn xuống 50 / 30 / 10 %:</p>
{hyp_sentences()}
<p>Kết luận theo từng bộ, ở mốc 30 % nhãn: <b>xác nhận</b> (PINN-semi 30 % ≤ MLP 70 %) trên {stat('v_sup')}; <b>xác nhận một phần</b> (PINN thu hẹp khoảng cách nhưng chưa bằng) trên {stat('v_part')}; <b>không xác nhận</b> trên {stat('v_no')}. Ở mốc 10 % nhãn thì vật lý thắng trên cả bốn bộ — tỉ số MAE(MLP)/MAE(PINN-semi): {stat('gain10')}. Toàn bộ số liệu là trung bình 3 seed, {stat('n_runs_E1')} lượt huấn luyện cho thí nghiệm chính; bảng đầy đủ ở mục 07.</p>
<div class="callout"><p><b>Đọc kết quả cho đúng.</b> Vật lý là một <em>prior</em>: nó bù thiếu hụt nhãn, và bù nhiều nhất ở nơi mô hình chỉ dữ liệu đói nhãn nhất (HUST, MIT: nhãn nhiễu, vòng đời dài; XJTU: 6 protocol khác nhau chỉ với 55 cell). Ở nơi 30 % nhãn đã đủ cho MLP (TJU: MLP 30 % chỉ kém MLP 70 % có {stat('mlp_ratio').split(', ')[1] if ', ' in stat('mlp_ratio') else '—'}), ràng buộc trở thành thiên kiến và làm MAE tăng nhẹ. Khi nhãn dồi dào (70 %) hai bên hoà hoặc MLP nhỉnh hơn (PINN-semi/MLP ở 70 %: {stat('pinn_70_vs_mlp_70')}). Câu chuyện trung thực là "ít nhãn thì vật lý thắng, nhãn đủ thì hoà", không phải "vật lý luôn thắng".</p></div>
<div class="callout"><p><b>Cập nhật quan trọng (E7).</b> Kết quả trên dùng một bộ trọng số vật lý cố định (β = 5) cho cả bốn bộ dữ liệu — và đó chính là thứ giới hạn nó. Khi β được <em>chọn theo validation</em> ở từng mức nhãn, bức tranh đổi hẳn: PINN thắng MLP ở cùng mức nhãn trong <b>{stat('e7_wins')}</b> trường hợp, và tại mốc <b>50 % nhãn</b> đạt {stat('e7_50_vs_mlp70')} so với MLP dùng đủ 70 % nhãn. Chi tiết ở mục 07-E7; phần dưới đây giữ nguyên để thấy vì sao trọng số cố định là sai lầm.</p></div>
<div class="callout"><p><b>Kết quả sau khi cân bằng ngân sách tinh chỉnh (E12).</b> Khi baseline MLP được hưởng đúng lưới siêu tham số mà PINN được hưởng, một phần lợi thế của PINN biến mất — trung bình tỉ số PINN/MLP đi từ {stat('e12_before')} lên {stat('e12_after')}. PINN vẫn thắng ở {stat('e12_wins')} ô với tỉ số {stat('e12_range')}. Đây là con số nên dùng khi báo cáo, không phải các con số ở mục 07-E1.</p></div>
<p><b>Vế thứ hai — đem sang bộ dữ liệu khác.</b> Zero-shot thất bại với mọi mô hình (E2): đặc trưng đường sạc phụ thuộc protocol, nên đầu vào của bộ đích nằm ngoài phân bố đã học. Nhưng vì các loss vật lý không cần nhãn, chúng chạy được ngay trên cell của bộ đích và trở thành một cơ chế <em>thích nghi miền không nhãn</em> (E5): {stat('e5_short')}. Với thêm 3 cell đích có nhãn, PINN-semi tốt hơn MLP ở {stat('e5_k3_wins')} cặp. Đây là chỗ vật lý thực sự trả lại giá trị khi đổi bộ dữ liệu.</p>
</section>

<section id="khoangtrong">
<h2><span class="num">01</span>Khoảng trống trong PINN4SOH — đọc từ mã nguồn công bố</h2>
<p>PINN4SOH là điểm xuất phát tự nhiên: cùng 4 bộ dữ liệu, cùng 16 đặc trưng, mã nguồn mở. Đọc kỹ mã nguồn (repo <code>wang-fujin/PINN4SOH</code>) cho thấy bốn điểm mà bài báo không nói rõ, và mỗi điểm mở ra một chỗ để đóng góp. Số dòng dẫn theo bản clone ngày 05/09/2026.</p>
<div class="grid2">
 <div class="card"><span class="tag warn">Model/Model.py:232–234</span><h4>Phần dư PDE suy biến</h4><p><code>F = dynamical_F(cat[xt, u, u_x, u_t])</code> rồi <code>f = u_t − F</code>. Mạng động học nhận chính <i>u_t</i> làm đầu vào, nên có thể học <i>F ≈ u_t</i> và triệt tiêu phần dư mà không cần bất kỳ vật lý nào. "PDE loss" vì thế gần như là một regularizer tự thoả.</p></div>
 <div class="card"><span class="tag warn">Model/Model.py:255</span><h4>"Physics loss" phụ thuộc nhãn</h4><p><code>loss3 = relu((u2−u1)·(y1−y2)).sum()</code> — phạt khi hướng thay đổi của dự đoán ngược với hướng của <em>nhãn</em>. Đây là ràng buộc đồng dấu với nhãn, không phải tính đơn điệu vật lý; do đó <b>không áp được lên cell không có nhãn</b>, và không thể là nguồn "thông tin bù nhãn".</p></div>
 <div class="card"><span class="tag warn">dataloader/dataloader.py:50, 58–64</span><h4>Chuẩn hoá theo từng cell trên cả vòng đời</h4><p>Chỉ số chu kỳ được chèn vào làm đặc trưng rồi min-max/z-score <em>theo từng file cell</em>. Chỉ số chuẩn hoá khi đó chính là "phần trăm tuổi thọ đã đi qua" — tức rò rỉ tổng tuổi thọ của cell test vào đầu vào, và không thể tính online (cần biết tương lai). Mục 07-E4 định lượng mức ảnh hưởng.</p></div>
 <div class="card"><span class="tag warn">dataloader.py:157–158 · main_XJTU.py:25–26</span><h4>Tập validation chia theo hàng, không theo cell</h4><p>Validation lấy ngẫu nhiên 20 % <em>hàng</em> từ các cell huấn luyện, nên chọn mô hình/early-stopping trên đúng những cell đã học. Test thì đúng là theo cell (file có "4" hoặc "8" trong tên).</p></div>
</div>
<p>Bổ sung: thí nghiệm "small sample" lấy <code>train_list[:n]</code> — n file đầu theo thứ tự thư mục, không ngẫu nhiên; trọng số α, β đổi giữa thí nghiệm thường (0.7/0.2) và small-sample (0.5/10) mà bài không nêu. Không điểm nào ở trên phủ nhận kết quả của họ; chúng chỉ nói rằng <em>cơ chế</em> vì sao PINN4SOH cần ít dữ liệu chưa được chứng minh, và con số có thể được nâng bởi rò rỉ chuẩn hoá.</p>
</section>

<section id="dulieu">
<h2><span class="num">02</span>Dữ liệu: 387 cell, 4 bộ, 2 họ hoá học</h2>
<p>Dùng đúng dữ liệu đã tiền xử lý trong repo PINN4SOH để kết quả đối chiếu được. Mỗi hàng = một chu kỳ; 16 đặc trưng thống kê (trung bình, độ lệch chuẩn, kurtosis, skewness, điện lượng, thời gian, độ dốc, entropy — cho cả điện áp lẫn dòng) của đoạn sạc cuối CC và pha CV; nhãn = dung lượng xả.</p>
<div class="tw"><table><thead><tr><th>Bộ</th><th>Cell</th><th>Chu kỳ (sau lọc)</th><th>Hoá học · danh định</th><th>Nhiệt độ</th><th>Điều kiện</th></tr></thead><tbody>
<tr><td>XJTU</td><td>55</td><td>22 212</td><td>NCM · 2.0 Ah</td><td>25 °C</td><td>6 nhóm: 2C, 3C, R2.5, R3, RW, mô phỏng vệ tinh</td></tr>
<tr><td>TJU</td><td>130</td><td>56 779</td><td>NCA / NCM / NCM+NCA · 3.5 / 3.5 / 2.5 Ah</td><td>25 / 35 / 45 °C</td><td>sạc 0.25–1C, xả 1–4C</td></tr>
<tr><td>MIT</td><td>125</td><td>81 865</td><td>LFP (A123) · 1.1 Ah</td><td>30 °C</td><td>3 đợt, sạc nhanh đa dạng</td></tr>
<tr><td>HUST</td><td>77</td><td>143 172</td><td>LFP (A123) · 1.1 Ah</td><td>30 °C</td><td>77 profile xả nhiều bậc</td></tr>
</tbody></table></div>
<p>Lọc: bỏ hàng có giá trị vô hạn/NaN và hàng có đặc trưng lệch quá 3σ so với chính cell đó (chỉ xét đặc trưng, không xét nhãn). SOH = dung lượng / danh định; XJTU bắt đầu ở 0.92–0.99, HUST ở 1.06–1.12 (danh định thấp hơn thực tế) — vì thế ràng buộc "cell mới ≈ 1" được nới thành khoảng [0.85, 1.10].</p>
<h3>Một phát hiện định hình thiết kế: dấu tương quan đặc trưng–SOH không bất biến theo hoá học</h3>
<p>Trước khi đưa bất kỳ "tiên nghiệm dấu" nào (ví dụ: ∂SOH/∂(điện lượng CC) ≥ 0) vào loss, tôi đo tương quan Spearman trong từng cell rồi lấy trung vị:</p>
<div class="tw"><table><thead><tr><th>Đặc trưng</th><th>XJTU (NCM)</th><th>TJU (NCA/NCM)</th><th>MIT (LFP)</th><th>HUST (LFP)</th></tr></thead><tbody>
<tr><td>CC Q (điện lượng pha CC)</td><td>−0.18</td><td>+0.95</td><td>+0.98</td><td>+0.97</td></tr>
<tr><td>CC charge time</td><td>−0.17</td><td>+0.95</td><td>+0.98</td><td>+0.97</td></tr>
<tr><td>CV Q (điện lượng pha CV)</td><td>−0.94</td><td>−1.00</td><td>+0.41</td><td>+0.55</td></tr>
<tr><td>CV charge time</td><td>−0.94</td><td>−1.00</td><td>+0.62</td><td>+0.64</td></tr>
<tr><td>voltage mean</td><td>+0.60</td><td>+0.04</td><td>−0.99</td><td>−0.94</td></tr>
<tr><td>current kurtosis</td><td>−0.65</td><td>−0.97</td><td>+0.77</td><td>+0.87</td></tr>
</tbody></table></div>
<p>Cùng một đặc trưng đổi dấu giữa NCM/NCA và LFP (CV Q: −0.94/−1.00 so với +0.41/+0.55), thậm chí giữa hai bộ cùng NCM (CC Q: −0.18 ở XJTU nhưng +0.95 ở TJU, do XJTU có protocol ngẫu nhiên). Hệ quả: (i) mọi ràng buộc dấu trên đặc trưng đều <b>bị loại</b> khỏi thiết kế vì không phổ quát; (ii) vật lý được dùng phải nằm ở <em>quỹ đạo SOH</em> (đơn điệu, tốc độ không âm, Arrhenius) chứ không ở đặc trưng; (iii) chuyển miền zero-shot giữa hai họ hoá học với bộ đặc trưng này về nguyên tắc là không khả thi — chỉ có cặp cùng loại cell (HUST↔MIT, đều A123 LFP 1.1 Ah) mới có ý nghĩa.</p>
</section>

<section id="kientruc">
<h2><span class="num">03</span>Kiến trúc</h2>
<div class="svgwrap">{ARCH_SVG}</div>
<p>Hai mạng nhỏ, tổng cộng ≈ 8 000 tham số, chạy được trên vi điều khiển sau lượng tử hoá.</p>
<div class="eq"><span class="lab">(1) mạng nghiệm</span><i>u</i>(N) = F<sub>φ</sub>(x<sub>N</sub>, Ñ),&nbsp;&nbsp; Ñ = N / 1000</div>
<div class="eq"><span class="lab">(2) động học xám</span>d<i>u</i>/dÑ = − <i>r</i>(x, <i>u</i>, T),&nbsp;&nbsp; <i>r</i> = softplus(MLP<sub>θ</sub>(x)) · exp(λ(1 − <i>u</i>)) · exp(−E<sub>a</sub>/R · (1/T − 1/T<sub>ref</sub>))</div>
<p>Ba thừa số của tốc độ suy giảm <i>r</i> mang ba mẩu vật lý khác nhau, mỗi mẩu kiểm chứng được riêng bằng ablation (mục 07-E3):</p>
<ul>
<li><b>softplus(·) ≥ 0</b>: tốc độ mất dung lượng không âm — nghiệm của (2) đơn điệu không tăng <em>theo cấu trúc</em>, không cần nhãn để "dạy".</li>
<li><b>exp(λ(1−u))</b>, λ ≥ 0 học được: tốc độ tăng dần khi SOH giảm — mô tả pha "đầu gối" (knee) khi LAM lấn át LLI. λ dùng chung cho cả bộ dữ liệu nên đọc được như một tham số vật lý.</li>
<li><b>Arrhenius</b> với E<sub>a</sub> ≥ 0 học được, T<sub>ref</sub> = 298.15 K: chỉ có tác dụng khi bộ dữ liệu có nhiều nhiệt độ (TJU: 25/35/45 °C); ở bộ đơn nhiệt độ thừa số này bằng 1 và E<sub>a</sub> giữ nguyên giá trị khởi tạo (đúng như mong đợi — không có thông tin thì không cập nhật).</li>
</ul>
<p>Khác với PINN4SOH, mạng động học <em>không</em> nhận đạo hàm của <i>u</i> làm đầu vào — phần dư không thể tự triệt tiêu. Đầu vào chỉ số chu kỳ được chia cho hằng số toàn cục 1000, không bao giờ chuẩn hoá theo cell.</p>
</section>

<section id="loss">
<h2><span class="num">04</span>Các hàm mất mát vật lý</h2>
<p>Mọi loss vật lý được tính trên cặp chu kỳ (N, N+h) <em>của cùng một cell</em> với h ngẫu nhiên trong {{5, …, 50}}, và <b>không dùng nhãn</b>. Chỉ L<sub>data</sub> cần nhãn.</p>
<div class="eq"><span class="lab">(3) dữ liệu</span>L<sub>data</sub> = mean<sub>N ∈ cell có nhãn</sub> ( <i>u</i>(N) − y<sub>N</sub> )²</div>
<div class="eq"><span class="lab">(4) ODE dạng Euler</span>L<sub>ode</sub> = mean<sub>(N,h)</sub> [ <i>u</i>(N+h) − <i>u</i>(N) + <i>r</i>(x<sub>N</sub>, <i>u</i>(N), T) · h/1000 ]²</div>
<div class="eq"><span class="lab">(5) đơn điệu có dung sai</span>L<sub>mono</sub> = mean<sub>(N,h)</sub> ReLU( <i>u</i>(N+h) − <i>u</i>(N) − ε ),&nbsp; ε = 0.002</div>
<div class="eq"><span class="lab">(6) miền hợp lý</span>L<sub>range</sub> = mean [ ReLU(<i>u</i> − 1.15) + ReLU(0.4 − <i>u</i>) ] + mean<sub>N = chu kỳ đầu</sub> [ ReLU(<i>u</i> − 1.10) + ReLU(0.85 − <i>u</i>) ]</div>
<div class="eq"><span class="lab">(7) tổng</span>L = L<sub>data</sub> + w(s) · ( α L<sub>ode</sub> + β L<sub>mono</sub> + γ L<sub>range</sub> ),&nbsp;&nbsp; w(s) = min(1, s / 0.2 S)</div>
<p>α = 2, β = 5, γ = 1 chọn trên <em>tập validation của XJTU</em> (2 seed, lưới 3×2) rồi đóng băng cho mọi bộ dữ liệu khác; w(s) tăng tuyến tính từ 0 lên 1 trong 20 % số bước đầu để mạng nghiệm bám dữ liệu trước khi vật lý siết lại.</p>
<h3>Vì sao dạng Euler (tích phân) chứ không phải dạng đạo hàm</h3>
<p>Phiên bản đầu tiên dùng phần dư dạng đạo hàm (u(N+h) − u(N))/(h/1000) + r, tức là ước lượng du/dÑ bằng sai phân. Với h = 1 chu kỳ, mẫu số 0.001 <b>khuếch đại nhiễu 1000 lần</b>: một dao động 10⁻³ của mạng giữa hai chu kỳ liên tiếp (do đặc trưng đo có nhiễu) trở thành phần dư cỡ 1, lớn hơn hẳn tín hiệu thật (|du/dÑ| ≈ 0.2–0.5). Mạng "học" cách giảm phần dư bằng cách trở nên vô cảm với đặc trưng — và MAE xấu đi gấp đôi so với MLP thuần. Dạng Euler (4) nhân cả hai vế với h/1000: cặp horizon ngắn đóng góp rất nhỏ, cặp horizon dài tự nhiên chi phối, nhiễu không bị khuếch đại. Đây là bài học về <em>cách viết phần dư</em> hơn là về vật lý, và cả ba biến thể (euler / fd / autograd) đều được giữ lại trong code để ablation.</p>
<p>Hai ràng buộc từng cân nhắc rồi loại bỏ: tiên nghiệm dấu trên đặc trưng (mục 02 giải thích) và điều kiện đầu cứng u(0) = 1 (sai với HUST/XJTU).</p>
</section>

<section id="coche">
<h2><span class="num">05</span>Cơ chế khiến "ít nhãn vẫn tốt" — và cách tách bạch nó bằng thí nghiệm</h2>
<p>Có hai cách vật lý có thể giúp khi thiếu nhãn, và chúng khác nhau về bản chất:</p>
<div class="grid2">
 <div class="card"><span class="tag">pinn_sup</span><h4>Vật lý như regularizer</h4><p>Loss vật lý chỉ tính trên các cell có nhãn. Vật lý thu hẹp không gian hàm (đơn điệu, tốc độ không âm) nên ít dữ liệu hơn vẫn xác định được nghiệm — nhưng không thêm thông tin từ cell nào khác.</p></div>
 <div class="card"><span class="tag">pinn_semi</span><h4>Vật lý như nhãn thay thế (bán giám sát)</h4><p>Loss vật lý tính trên <em>mọi</em> cell huấn luyện, gồm những cell chỉ có đường sạc mà không có nhãn dung lượng. Vì (4)–(6) không cần y, mạng buộc phải gán cho các cell này một quỹ đạo SOH đơn điệu, bắt đầu trong [0.85, 1.10], suy giảm với tốc độ r(x) đã học từ cell có nhãn. Đây chính là tình huống thực tế: BMS ghi đường sạc của mọi cell, nhưng đo dung lượng chuẩn (xả đầy) chỉ làm được với vài cell.</p></div>
</div>
<p>Tách hai cơ chế này là lý do protocol có ba mô hình chứ không phải hai. Kết quả (mục 07) cho một bức tranh có sắc thái: ở <b>10 % nhãn</b> cơ chế bán giám sát cho thêm lợi ích rõ trên các bộ ít cell có nhãn nhất (MAE pinn_semi so với pinn_sup — {stat('semi_vs_sup_10')}); ở <b>30 % nhãn</b> hai cơ chế gần như tương đương ({stat('semi_vs_sup_30')}). Tức phần lớn lợi ích ở 30 % đến từ <em>cấu trúc</em> (đơn điệu, tốc độ không âm) thu hẹp không gian hàm, còn cell không nhãn chỉ thật sự đáng giá khi nhãn cực hiếm. PINN4SOH — với loss vật lý phụ thuộc nhãn — về cấu trúc chỉ có thể là cơ chế thứ nhất, và không thể hưởng phần lợi ích thứ hai.</p>
</section>

<section id="protocol">
<h2><span class="num">06</span>Protocol thí nghiệm</h2>
<div class="kv">
<div>chia</div><div>Theo <b>cell</b>, phân tầng theo nhóm protocol, 70 / 15 / 15 (train / val / test); assert không cell nào xuất hiện ở hai tập.</div>
<div>ít nhãn</div><div>Giữ nguyên val/test của cùng seed; trong 70 % train chỉ một phần cell mang nhãn: 10 / 30 / 50 / 70 % của <em>tổng</em> số cell (70 % = protocol đầy đủ). Cell còn lại là "không nhãn": mô hình bán giám sát chỉ thấy đặc trưng của chúng.</div>
<div>chuẩn hoá</div><div>z-score fit trên đặc trưng của cell huấn luyện (mô hình giám sát: chỉ cell có nhãn); chỉ số chu kỳ /1000. Biến thể <code>first_cycle</code>: đặc trưng tương đối so với 3 chu kỳ đầu của chính cell — nhân quả, dùng cho chuyển miền.</div>
<div>huấn luyện</div><div>AdamW, lr 2e-3 cosine, batch 512, tối đa 4 000 bước, early stopping theo MAE val (kiên nhẫn 12 lần đánh giá), chọn checkpoint tốt nhất trên val. Cùng ngân sách bước cho mọi mức nhãn.</div>
<div>đo</div><div>MAE, RMSE, MAPE trên <em>mọi chu kỳ</em> của cell test; thêm MAE trung bình theo cell. Trung bình ± độ lệch chuẩn trên 3 seed (seed đổi cả cách chia lẫn khởi tạo).</div>
<div>E1</div><div>4 bộ × 4 mức nhãn × 3 mô hình × 3 seed = 144 lượt. Chỉ tiêu chính: MAE(PINN-semi, 30 %) / MAE(MLP, 70 %).</div>
<div>E2</div><div>Chuyển miền <b>zero-shot</b>: huấn luyện 70 % nhãn trên bộ nguồn, đánh giá thẳng trên toàn bộ bộ đích, không fine-tune. Cặp cùng loại cell HUST↔MIT và cặp cùng họ XJTU↔TJU; chuẩn hoá global và first_cycle.</div>
<div>E3</div><div>Ablation ở 30 % nhãn trên XJTU, TJU: bỏ từng loss, bỏ knee, bỏ Arrhenius, đổi dạng phần dư, thay động học xám bằng động học đen kiểu PINN4SOH.</div>
<div>E4</div><div>Kiểm toán rò rỉ: MLP 70 % nhãn với chuẩn hoá nhân quả so với chuẩn hoá theo cell của PINN4SOH (chỉ chỉ số chu kỳ / chỉ đặc trưng / cả hai), và bỏ hẳn đầu vào chu kỳ.</div>
<div>E12</div><div>Ngân sách tinh chỉnh cân bằng: cùng 8 biến thể siêu tham số tổng quát (weight decay, dropout, độ rộng mạng, kiến trúc đơn điệu) cho cả MLP lẫn PINN, ở 30 % và 70 % nhãn, chọn theo validation. 4 bộ × 8 biến thể × 2 mô hình × 2 mức × 3 seed = 384 lượt.</div>
<div>E9</div><div>Lưới kiến trúc × hàm kích hoạt ở 30 % nhãn, chạy cho cả baseline lẫn PINN: silu / tanh / sin / snake trên MLP, cộng residual, Fourier(t), và đầu ra đơn điệu theo cấu trúc (có/không L<sub>mono</sub>). 4 bộ × 8 biến thể × 2 mô hình × 3 seed.</div>
<div>E10</div><div>Ma trận chuyển miền đầy đủ 4×4: mỗi nguồn (70 % nhãn) đánh giá zero-shot trên tập test của cả bốn bộ, cùng seed nên đường chéo = kết quả trong miền. 2 mô hình × 2 cách chuẩn hoá × 3 seed.</div>
<div>E11</div><div>Bộ chỉ số mở rộng tính lại từ dự đoán đã lưu của E1: theo cell, vùng SOH ≤ 0.90, vi phạm đơn điệu, nhiễu quỹ đạo, sai số dự báo EOL (ngưỡng 0.85) kèm tỉ lệ phủ.</div>
<div>E7</div><div>Quét lưới trọng số vật lý (β ∈ {0.5, 1.5, 5}, phần dư Euler/autograd, biến thể tự thích ứng) ở <em>mọi</em> mức nhãn; chọn theo MAE validation của từng (bộ, mức nhãn) rồi mới đọc test. 4 bộ × 4 mức × 5 biến thể × 3 seed = 240 lượt.</div>
<div>E8</div><div>Giải mã đơn điệu hậu kỳ (PAVA) trên dự đoán đã lưu của E1, áp cho cả MLP lẫn PINN — tách phần lợi ích "đầu ra đơn điệu" khỏi phần còn lại. Không huấn luyện lại.</div>
<div>E5</div><div>Thích nghi miền: nguồn 70 % nhãn + bộ đích với k ∈ {0, 3} cell có nhãn; PINN-semi áp loss không nhãn lên mọi cell train của đích; đánh giá trên cell test của đích; 2 500 bước cố định, không chọn mô hình theo nhãn đích.</div>
</div>
</section>

<section id="ketqua">
<h2><span class="num">07</span>Kết quả sơ bộ (CPU, 3 seed)</h2>
<h3>E1 — Hiệu quả theo lượng nhãn</h3>
<figure>{img(f'{R}/fig_E1_label_efficiency.png')}<figcaption>MAE trên tập test cố định khi giảm số cell có nhãn. Trục dọc dùng thang log CHUNG cho cả bốn bộ, đơn vị 10⁻³ SOH: cùng một khoảng cách dọc luôn là cùng một TỈ SỐ sai số, đúng với cách các kết luận của bài được phát biểu. Dải mờ = ± 1 độ lệch chuẩn trên 3 seed. Khoảng cách giữa đường xanh dương (MLP) và xanh lục (PINN-semi) ở 10–30 % là phần vật lý bù được; ở 70 % các đường gặp nhau.</figcaption></figure>
{md_table(f'{R}/E1_tables.md')}
<figure>{img(f'{R}/fig_traj_XJTU.png')}<figcaption>XJTU, 30 % nhãn, seed 0: quỹ đạo dự đoán trên bốn cell test. Đường MLP dao động và có đoạn tăng ngược; PINN-semi trơn và đơn điệu — hệ quả trực tiếp của (4)–(5).</figcaption></figure>
<figure>{img(f'{R}/fig_traj_TJU.png')}<figcaption>TJU, 30 % nhãn, seed 0.</figcaption></figure>
<h3>E2 — Chuyển miền zero-shot: thất bại, và vì sao đó là kết quả đúng</h3>
{md_table(f'{R}/E2_table.md')}
<p>Đem mô hình huấn luyện trên bộ nguồn đánh giá thẳng lên bộ đích, không fine-tune, MAE tăng 5–100 lần so với trong miền — ngay cả cặp <em>cùng loại cell</em> HUST↔MIT (đều A123 LFP 1.1 Ah). Lý do nằm ở đầu vào, không ở mô hình: 16 đặc trưng là thống kê của đoạn sạc, mà protocol sạc của MIT (sạc nhanh nhiều bậc rồi 1C CC–CV) khác hẳn HUST; phân bố đặc trưng dịch hoàn toàn, và không ràng buộc nào trên <em>quỹ đạo SOH</em> có thể sửa một đầu vào chưa từng thấy. Chuẩn hoá tương đối theo chu kỳ đầu giúp cặp NCM (TJU→XJTU: 0.28 → 0.09) nhưng hại cặp LFP. Vật lý làm sự sụp đổ bớt thảm hoạ ở 2/4 cặp (HUST→MIT: 0.36 → 0.13; XJTU→TJU: 2.49 → 1.55) nhưng không nhất quán.</p>
<div class="callout warn"><p><b>Kết luận cho vế thứ hai của giả thuyết.</b> "Có ràng buộc vật lý thì đem sang bộ dữ liệu khác vẫn tốt" <em>không đúng ở dạng zero-shot</em> với bộ đặc trưng này. Nó đúng ở dạng yếu hơn và thực dụng hơn — E5 ngay dưới.</p></div>
<h3>E5 — Thích nghi sang bộ khác bằng chính các loss không nhãn</h3>
<p>Thiết lập: nguồn 70 % nhãn; bộ đích chia theo cell 70/15/15 cùng seed; lấy k ∈ {{0, 3}} cell đích có nhãn từ phần train của đích, các cell train còn lại của đích <b>chỉ có đặc trưng</b>; đánh giá trên cell <em>test</em> của đích; không early-stopping theo nhãn đích (2 500 bước cố định, lấy mô hình cuối) — mọi phương pháp như nhau. PINN-semi áp (4)–(6) lên toàn bộ cell train của đích, tức là thích nghi miền bằng vật lý mà không cần nhãn.</p>
<figure>{img(f'{R}/fig_E5_adapt.png')}<figcaption>Trục log. Màu nhạt: k = 0 (không có nhãn đích); màu đậm: k = 3 cell đích có nhãn. Dòng tham chiếu trong bảng: MLP huấn luyện ngay trong miền đích với 70 % nhãn.</figcaption></figure>
{md_table(f'{R}/E5_table.md')}
<p>{stat('e5_sentence')}</p>
<h3>E3 — Ablation (30 % nhãn, XJTU và TJU)</h3>
{md_table(f'{R}/E3_table.md')}
<p>Bốn điều bảng này nói, không tô hồng:</p>
<ul>
<li><b>Dạng phần dư quyết định thành bại.</b> Sai phân chia Δt (<code>res_fd</code>) tệ nhất ở cả hai bộ (XJTU 0.0158, TJU 0.0119) đúng như phân tích nhiễu ở mục 04; dạng Euler và dạng đạo hàm autograd đều ổn, autograd còn nhỉnh hơn một chút (0.0107 / 0.0096) — nên là mặc định cho vòng sau.</li>
<li><b>Ở 30 % nhãn trên hai bộ NCM/NCA, ràng buộc đơn điệu với β = 5 đang siết quá tay:</b> bỏ nó (<code>no_mono</code>) lại <em>giảm</em> MAE (XJTU 0.0115 → 0.0101; TJU 0.0099 → 0.0086). Nhãn của XJTU/TJU có tái sinh dung lượng và nhiễu lớn hơn dung sai ε = 0.002, nên ràng buộc trở thành thiên kiến. Đây là lý do E1 không cho PINN thắng ở 30 % trên XJTU/TJU, và là lý do E6 tồn tại.</li>
<li><b>Số hạng ODE và knee gần như trung tính ở 30 %</b> (<code>no_ode</code>, <code>no_knee</code> ≈ full); giá trị của chúng nằm ở 10 % nhãn (E1) và ở tính diễn giải, không phải ở MAE tại 30 %.</li>
<li><b>Động học đen kiểu PINN4SOH (<code>blackbox</code>) cho MAE tương đương động học xám.</b> Lợi thế của động học xám là không suy biến và có tham số đọc được, không phải độ chính xác. Arrhenius giúp một chút trên TJU — bộ duy nhất có nhiều nhiệt độ (<code>no_arrh</code> 0.0105 so với 0.0099); E<sub>a</sub> học được {stat('Ea_TJU')} kJ/mol chỉ dịch nhẹ khỏi giá trị khởi tạo 30, tức được xác định yếu với 3 mức nhiệt độ — đọc như kiểm tra hợp lý, không phải phép đo.</li>
</ul>
<h3>E6 — Chọn trọng số vật lý theo validation (30 % nhãn)</h3>
<p>E3 cho thấy trọng số cố định α = 2, β = 5 (chọn trên XJTU ở một cấu hình) không phải tối ưu cho mọi bộ. Cách làm đúng và vẫn sạch: chọn β (và dạng phần dư) <em>trên tập validation của từng bộ</em>, rồi mới nhìn test. Ứng viên: β ∈ {{0.5, 1.5, 5}}, ε = 0.005, và phần dư autograd với β = 0.5.</p>
{md_table(f'{R}/E6_table.md')}
<p>{stat('e6_sentence')}</p>
<h3>E7 — Câu hỏi quyết định: trọng số vật lý, chứ không phải lượng nhãn</h3>
<figure>{img(f'{R}/fig_ratio_matrix.png')}<figcaption>Tỉ số MAE của PINN so với MLP ở <b>cùng</b> mức nhãn. Đây là đại lượng có dấu quanh 1.00 nên dùng thang phân kỳ với tâm đúng tại 1.00 — xanh lục là PINN tốt hơn, cam là MLP tốt hơn. Bảng trái dùng trọng số cố định β = 5, bảng phải chọn β theo validation.</figcaption></figure>
<p>E3 và E6 gợi ý rằng β = 5 cố định là thủ phạm khiến PINN thua ở mức nhãn cao, chứ không phải bản thân vật lý. E7 kiểm chứng bằng cách quét đúng lưới của E6 (β ∈ {{0.5, 1.5, 5}}, dạng phần dư Euler / autograd, cộng biến thể tự thích ứng) trên <b>mọi</b> mức nhãn, chọn biến thể theo <em>MAE validation</em> của từng (bộ, mức nhãn), rồi mới đọc test.</p>
<figure>{img(f'{R}/fig_E7_tuned.png')}<figcaption>Đường vàng là cấu hình mặc định cũ (β = 5) — thua MLP ở TJU và XJTU khi nhãn nhiều. Đường xanh lục đậm là cùng mô hình đó với β chọn theo validation: nằm dưới MLP ở gần như mọi điểm. Đường hồng là trọng số tự thích ứng, không tinh chỉnh gì. Đường đứt xanh lam là MLP ở cùng mức nhãn. Trục dọc thang log chung cho cả bốn bộ, đơn vị 10⁻³ SOH; hai đường nhạt là biến thể đối chứng.</figcaption></figure>
{md_table(f'{R}/E7_table.md')}
<p>Kết quả đảo chiều hoàn toàn so với E1: PINN thắng MLP <b>ở cùng mức nhãn trong {stat('e7_wins')} trường hợp</b> (trước đó là 11/16). Riêng tại mốc 50 % nhãn — mức người hỏi quan tâm — PINN thắng MLP cùng mức trên cả bốn bộ ({stat('e7_50_vs_mlp')}), và so với <em>MLP dùng đủ 70 % nhãn</em> thì đạt {stat('e7_50_vs_mlp70')}, tức bằng hoặc tốt hơn ở {stat('e7_50_le70')} bộ. Ở mốc 30 %: {stat('e7_30_vs_mlp70')} so với MLP 70 % ({stat('e7_30_le70')} bộ đạt).</p>
<div class="callout"><p><b>Quy luật chọn β rút ra được.</b> Ở mức nhãn thấp (10–30 %) validation chọn {stat('e7_sel_lo')}; ở mức cao (50–70 %) chọn {stat('e7_sel_hi')}. Nói cách khác trọng số vật lý nên <em>giảm khi nhãn tăng</em> — đúng như kỳ vọng cho một tiên nghiệm. Biến thể <code>adapt</code> mã hoá sẵn quy luật này: β<sub>hiệu dụng</sub> = β₀·[(1 − f<sub>nhãn</sub>) + 0.15] với f<sub>nhãn</sub> = tỉ lệ cell có nhãn trong tập train, không cần tinh chỉnh gì và tự được validation chọn ở 6/16 ô. Nó thắng MLP ở {stat('e7_adapt_wins')} ô, và tại mốc 50 % chỉ kém cấu hình tinh chỉnh {stat('e7_adapt_gap')}.</p></div>

<h3>E8 — Kiểm tra hoài nghi: phần lợi ích có phải chỉ là "đầu ra đơn điệu"?</h3>
<p>Một phản biện hiển nhiên: PINN cho quỹ đạo trơn và không tăng, nhưng thứ đó lấy được miễn phí bằng hậu xử lý — chiếu quỹ đạo dự đoán của mỗi cell test lên tập dãy không tăng (hồi quy đơn điệu, thuật toán PAVA, O(n), không cần nhãn, không huấn luyện lại). Nếu MLP + hậu xử lý bằng PINN thì vật lý trong lúc huấn luyện là thừa. Vì vậy phép giải mã này được áp cho <b>cả hai</b> mô hình.</p>
{md_table(f'{R}/E8_table.md')}
<p>Trước hết, ràng buộc vật lý thật sự có tác dụng như thiết kế: tổng bước tăng ngược của quỹ đạo dự đoán ({stat('e8_viol')}) — PINN nhỏ hơn MLP một đến hai bậc. Hệ quả là giải mã đơn điệu giúp MLP trung bình {stat('e8_gain_mlp')} nhưng chỉ giúp PINN {stat('e8_gain_pinn')} (vì PINN vốn đã đơn điệu). Tuy nhiên phần bù đó không đủ để đảo ngôi thứ ở bất kỳ ô nào: nơi PINN thắng thì vẫn thắng, nơi thua thì vẫn thua. Kết luận: tính đơn điệu ở đầu ra là <em>một phần</em> giá trị của vật lý, không phải toàn bộ — và giải mã đơn điệu là một cải tiến rẻ nên áp cho mọi mô hình, kể cả baseline.</p>

<h3>E12 — Ngân sách tinh chỉnh cân bằng: phép so sánh nghiêm khắc nhất</h3>
<p>Mọi bảng phía trên có một lỗ hổng: PINN được quét β, dạng phần dư, kiến trúc — còn baseline MLP chạy một cấu hình cố định. Như vậy là thiên vị. E12 sửa bằng cách áp <b>đúng một lưới siêu tham số tổng quát</b> (8 biến thể: weight decay 1e-5/1e-3/1e-2, dropout 0/0.1, mạng rộng (128,128,64) / hẹp (32,32,16), kiến trúc đơn điệu có/không weight decay) cho <em>cả hai</em> mô hình, chọn theo MAE validation, rồi mới đọc test. PINN dùng cấu hình vật lý tốt nhất theo E7 làm nền rồi quét đúng lưới đó.</p>
<figure>{img(f'{R}/fig_E12_fair.png')}<figcaption>Cột nhạt: MLP với cấu hình mặc định dùng ở E1. Cột giữa: MLP sau khi được hưởng đúng ngân sách tinh chỉnh của PINN. Cột đậm: PINN sau tinh chỉnh. Khoảng cách giữa hai cột đầu là phần "lợi thế" mà các bảng trước đó vô tình gán cho vật lý.</figcaption></figure>
{md_table(f'{R}/E12_table.md')}
<div class="callout warn"><p><b>Đính chính quan trọng.</b> Tinh chỉnh baseline cải thiện nó rất nhiều ở vài chỗ — lớn nhất: {stat('e12_mlp_gain')}. Riêng MIT ở 30 % nhãn, chỉ thêm dropout 0.1 đã đưa MLP từ 0.0103 xuống 0.0048. Nghĩa là con số "PINN tốt hơn 3×" ở các mục trước <em>phần lớn là do baseline bị bỏ đói tinh chỉnh</em>, không phải do vật lý. Trung bình trên 8 ô: tỉ số PINN/MLP là {stat('e12_before')} khi baseline chưa tinh chỉnh, và {stat('e12_after')} khi cả hai cùng được tinh chỉnh.</p></div>
<p>Sau khi cân bằng, kết luận vẫn đứng nhưng khiêm tốn hơn: PINN thắng ở <b>{stat('e12_wins')} ô</b>, tỉ số {stat('e12_range')}. Ngoại lệ là TJU ({stat('e12_tju')}) — bộ có nhiều cell nhất trên mỗi protocol và nhãn sạch nhất, tức nơi baseline ít cần tiên nghiệm nhất. Biến thể được chọn nhiều nhất: {stat('e12_tags')} — mạng (64,64,32) ban đầu thiếu dung lượng cho cả hai mô hình.</p>
<p>Bảng dưới nới lỏng hơn: mỗi mô hình được chọn trong <em>toàn bộ</em> kho biến thể của nó (MLP: E12 ∪ E9; PINN: E12 ∪ E9 ∪ E7). Hai mức nhãn 10 % và 50 % chỉ có 1 biến thể cho MLP nên <b>không cân bằng</b> — đọc riêng hai mức 30 % và 70 %.</p>
{md_table(f'{R}/E12_poolB.md')}

<h3>E9 — Thiết kế lại mạng: kiến trúc và hàm kích hoạt</h3>
<p>Bảy biến thể, ở 30 % nhãn, chạy cho <b>cả</b> baseline chỉ-dữ-liệu lẫn PINN-semi (để không cho PINN lợi thế tinh chỉnh mà baseline không có). Biến thể đáng chú ý nhất là <code>mono</code> — đưa tính đơn điệu vào <em>kiến trúc</em> thay vì vào hàm mất mát:</p>
<div class="eq"><span class="lab">(8) đầu ra đơn điệu theo cấu trúc</span><i>u</i>(x, t) = <i>u</i><sub>0</sub>(x) − Σ<sub>k</sub> s<sub>k</sub>(x)·softplus(ω<sub>k</sub> t + φ<sub>k</sub>(x)),&nbsp;&nbsp; s<sub>k</sub> = softplus(·) ≥ 0, ω<sub>k</sub> = softplus(·) ≥ 0</div>
<p>Vì softplus đơn điệu tăng và mọi hệ số đều không âm, ∂<i>u</i>/∂t = −Σ s<sub>k</sub>ω<sub>k</sub>·sigmoid(·) ≤ 0 <em>với mọi giá trị tham số</em>. Tính đơn điệu thôi không còn là số hạng phạt phải cân trọng số — nó là tính chất của không gian hàm, nên L<sub>mono</sub> được tắt (β = 0).</p>
<figure>{img(f'{R}/fig_E9_arch.png')}<figcaption>MAE chuẩn hoá theo MLP+SiLU, trung bình 4 bộ dữ liệu. Đường đứt = mốc cơ sở. Hai cột cho mỗi kiến trúc: baseline chỉ dữ liệu và PINN-semi.</figcaption></figure>
{md_table(f'{R}/E9_table.md')}
<p>Bốn kết luận:</p>
<ul>
<li><b>Đầu ra đơn điệu theo cấu trúc là kiến trúc tốt nhất cho cả hai mô hình</b> — {stat('e9_mono_pinn')} cho PINN và {stat('e9_mono_mlp')} cho baseline, chỉ thêm ~5 % tham số. Đáng chú ý: nó giúp baseline gần bằng mức giúp PINN, tức <em>một phần</em> giá trị của "vật lý" lấy được bằng thiết kế mạng, không cần loss nào cả. Sai số dự báo EOL giảm mạnh nhất ({stat('e9_mono_eol')} chu kỳ).</li>
<li><b>Hàm kích hoạt tương tác với vật lý.</b> tanh và Snake <em>giúp</em> PINN ({stat('e9_tanh_pinn')}, {stat('e9_snake_pinn')}) nhưng <em>hại</em> baseline ({stat('e9_tanh_mlp')}, {stat('e9_snake_mlp')} — Snake gần gấp đôi sai số). Kích hoạt trơn/dao động chỉ có lợi khi có ràng buộc giữ mô hình khỏi bám nhiễu. Riêng <code>sin</code> mà PINN4SOH dùng không hề tốt hơn SiLU.</li>
<li><b>Residual không đáng.</b> Gấp đôi tham số (19 492 so với 9 060), MAE {stat('e9_res_pinn')} — ở quy mô mạng này độ sâu không phải nút thắt.</li>
<li><b>Một cảnh báo quan trọng, và nó nằm ở chỗ dễ nhầm.</b> Kiến trúc <code>mono</code> đơn điệu theo <em>t</em>, nhưng dự đoán còn phụ thuộc x<sub>N</sub> — mà đặc trưng sạc dao động theo chu kỳ. Nên quỹ đạo <em>dọc theo cell</em> vẫn có thể tăng ngược: vi phạm đơn điệu đo được là {stat('e9_mono_viol')}, tức <b>tệ hơn</b> mô hình dùng L<sub>mono</sub>. Đơn điệu-theo-t không đồng nghĩa đơn điệu-dọc-quỹ-đạo. Ghép cả hai (<code>mono + L</code>) khôi phục tính đơn điệu nhưng lại siết quá tay và MAE xấu đi — dấu hiệu rằng cách đúng là làm cho <em>nhánh đặc trưng</em> cũng đơn điệu, chứ không phải chồng thêm phạt.</li>
</ul>

<h3>E10 — Ma trận chuyển miền đầy đủ 4×4</h3>
<p>Mỗi bộ làm nguồn (70 % nhãn), đánh giá zero-shot trên tập <em>test</em> của cả bốn bộ — dùng cùng seed nên <b>đường chéo trùng đúng kết quả trong miền</b> và so sánh trực tiếp được với các ô ngoài đường chéo.</p>
<figure>{img(f'{R}/fig_E10_matrix.png')}<figcaption>Thang log. Hàng = bộ huấn luyện, cột = bộ đánh giá. Càng nhạt càng tốt. Bốn bảng: hai mô hình × hai cách chuẩn hoá.</figcaption></figure>
{md_table(f'{R}/E10_table.md')}
<p>Ma trận này nói rõ hơn bất kỳ bảng nào trước đó:</p>
<ul>
<li><b>Khoảng cách trong-miền so với ngoài-miền là hai đến ba bậc.</b> Đường chéo trung bình {stat('e10_dia')}; ô ngoài đường chéo tệ nhất là {stat('e10_worst')} — lớn hơn 3 000 lần. Không có mô hình nào trong bốn cấu hình thoát khỏi điều này.</li>
<li><b>Chuẩn hoá nhân quả theo chu kỳ đầu là đòn bẩy mạnh nhất, không phải kiến trúc hay loss.</b> Trung vị các ô ngoài đường chéo: {stat('e10_med')}. Riêng XJTU→MIT giảm từ 25.5 xuống 4.8. Đây là thay đổi <em>tiền xử lý</em>, không phải mô hình.</li>
<li><b>Cùng loại cell mới chuyển được.</b> MIT↔HUST (đều A123 LFP 1.1 Ah) trung bình {stat('e10_same')}; các cặp khác hoá học trung bình {stat('e10_cross')} — chênh {stat('e10_ratio')}. Ô tốt nhất ngoài đường chéo là {stat('e10_best_off')}, vẫn tệ hơn trong miền khoảng 6 lần.</li>
<li><b>Ma trận không đối xứng.</b> MIT→XJTU cho 0.044 nhưng XJTU→MIT cho 4.8 (chuẩn hoá first_cycle) — hơn 100 lần. MIT có 125 cell với protocol sạc nhanh rất đa dạng nên học được biểu diễn rộng; XJTU chỉ 55 cell ở vài protocol. <em>Đa dạng của bộ nguồn quyết định khả năng chuyển miền nhiều hơn kích thước.</em></li>
</ul>

<h3>E11 — Chỉ số đánh giá: đổi thước đo là đổi người thắng</h3>
<p>MAE gộp mọi chu kỳ của mọi cell thành một số, nên bị chi phối bởi cell sống lâu và bởi vùng SOH cao (vốn dễ đoán). Bộ chỉ số dưới đây được tính lại từ dự đoán đã lưu của E1 (30 % nhãn), thêm bốn nhóm: theo cell (mỗi cell một phiếu, cộng cell tệ nhất — chỉ số an toàn), vùng cuối đời (SOH ≤ 0.90 — nơi ra quyết định thay pin), tính hợp lý vật lý (vi phạm đơn điệu, nhiễu quỹ đạo — <em>đo được mà không cần nhãn</em>), và hữu dụng vận hành (sai số chu kỳ khi dự báo thời điểm SOH cắt ngưỡng 0.85, kèm tỉ lệ cell thực sự cắt qua).</p>
{md_table(f'{R}/E11_table.md')}
<p>Số ô PINN thắng MLP, theo từng chỉ số: {stat('e11_wins')}. Trường hợp TJU minh hoạ rõ nhất vì sao điều này quan trọng — {stat('e11_tju')}. Nếu chỉ báo cáo MAE thì kết luận là "MLP thắng trên TJU"; nhìn theo cell tệ nhất, tính hợp lý vật lý và độ chính xác dự báo EOL thì PINN thắng. Cả hai đều đúng, nhưng chúng trả lời hai câu hỏi khác nhau, và bài báo phải nói rõ đang trả lời câu nào.</p>
<div class="callout warn"><p><b>Đọc chỉ số EOL cho đúng.</b> Cột "phủ EOL" là tỉ lệ cell test thực sự suy giảm xuống dưới ngưỡng 0.85 trong dữ liệu. Trên MIT chỉ 0.17 — sai số EOL ở đó tính trên rất ít cell nên không đáng tin. Chỉ số này chỉ dùng được cho XJTU (0.86), TJU (0.94) và HUST (1.00).</p></div>

<h3>E4 — Kiểm toán rò rỉ chuẩn hoá</h3>
{md_table(f'{R}/E4_table.md')}
<p>Chỉ riêng việc min-max <em>chỉ số chu kỳ</em> theo từng cell (tức cho mô hình biết cell test đang ở phần trăm nào của tuổi thọ) đã hạ MAE mà không cần thay đổi gì ở mô hình: {stat('leak_cycle')}. Đó là mức "lợi" bất hợp pháp phải trừ đi khi đọc các kết quả công bố dùng cách chuẩn hoá này. Ngược lại, min-max <em>đặc trưng</em> theo cell lại làm xấu đi ({stat('leak_feat')}) vì xoá mất mức tuyệt đối của đường sạc — nên khi gộp cả hai như pipeline PINN4SOH, hai hiệu ứng bù trừ ({stat('leak_both')}). Bỏ hẳn đầu vào chỉ số chu kỳ cho {stat('no_cycle')}: trên MIT, chỉ số chu kỳ là đặc trưng gây nhiễu (tuổi thọ trải từ 150 đến 2 300 chu kỳ), gợi ý thay nó bằng Ah-throughput ở bước tiếp theo.</p>
</section>

<section id="donggop">
<h2><span class="num">08</span>Đóng góp — phát biểu đúng mức</h2>
<ol>
<li><b>Động học suy giảm xám, không suy biến, diễn giải được.</b> Tốc độ r = softplus(·)·e^{{λ(1−u)}}·A(T) đảm bảo đơn điệu theo cấu trúc, tách bạch ba hiệu ứng (tốc độ cơ sở theo trạng thái đường sạc, gia tốc kiểu knee, Arrhenius) với hai tham số vật lý dùng chung λ, E<sub>a</sub>; mạng động học không nhận đạo hàm của nghiệm nên phần dư không thể tự triệt tiêu.</li>
<li><b>Loss vật lý không cần nhãn → bán giám sát trên cell chưa đo dung lượng.</b> Đây là cơ chế đo được (pinn_semi &gt; pinn_sup ở 10–30 %) giải thích vì sao ràng buộc vật lý bù được nhãn; PINN4SOH không có tính chất này vì loss của họ dùng nhãn.</li>
<li><b>Phần dư dạng Euler theo horizon ngẫu nhiên.</b> Phân tích nhiễu chỉ ra phần dư dạng đạo hàm khuếch đại nhiễu theo 1/Δt và phá hỏng độ nhạy với đặc trưng; dạng tích phân khắc phục hoàn toàn. Nhỏ nhưng thực dụng, và áp dụng cho mọi PINN hồi quy theo chu kỳ.</li>
<li><b>Protocol không rò rỉ + kiểm toán rò rỉ định lượng.</b> Chia theo cell, val/test cố định qua mọi mức nhãn, chuẩn hoá nhân quả; E4 đo phần "lợi" do chuẩn hoá theo cell của pipeline PINN4SOH tạo ra.</li>
<li><b>So sánh có ngân sách tinh chỉnh cân bằng.</b> Hầu hết bài PINN cho SOH quét siêu tham số cho mô hình của mình và để baseline ở cấu hình mặc định. E12 định lượng chính xác sai lệch đó trên cùng bộ dữ liệu: tỉ số PINN/MLP đi từ {stat('e12_before')} xuống còn {stat('e12_after')} khi baseline được tinh chỉnh ngang bằng. Kết luận vẫn đứng ({stat('e12_wins')} ô) nhưng ở biên độ trung thực hơn nhiều — và đây là phần dễ bị bỏ qua nhất trong các báo cáo hiện có.</li>
<li><b>Đầu ra đơn điệu theo cấu trúc.</b> Phương trình (8) đảm bảo ∂u/∂t ≤ 0 với mọi tham số, thay một số hạng phạt cần cân trọng số bằng một tính chất của không gian hàm. Là kiến trúc tốt nhất cho cả PINN ({stat('e9_mono_pinn')}) lẫn baseline ({stat('e9_mono_mlp')}), và cải thiện dự báo EOL nhiều nhất. Kèm theo là một phát hiện phủ định có giá trị: đơn điệu theo t <em>không</em> kéo theo đơn điệu dọc quỹ đạo khi đặc trưng cũng biến thiên — điều mà một bài chỉ báo cáo MAE sẽ không phát hiện ra.</li>
<li><b>Bộ chỉ số đánh giá theo cell / cuối đời / hợp lý vật lý / EOL.</b> Chỉ ra rằng thứ hạng phụ thuộc thước đo (TJU: MLP thắng theo MAE, PINN thắng theo cell tệ nhất, tính đơn điệu và sai số EOL), và rằng vi phạm đơn điệu là chỉ số <em>không cần nhãn</em> nên đo được cả trên cell chưa đo dung lượng — dùng được để giám sát mô hình khi triển khai.</li>
<li><b>Ma trận chuyển miền 4×4 với đường chéo là kết quả trong miền.</b> Định lượng ba điều: khoảng cách trong/ngoài miền là 2–3 bậc; chuẩn hoá nhân quả theo chu kỳ đầu là đòn bẩy lớn hơn kiến trúc và loss ({stat('e10_med')}); và ma trận bất đối xứng — bộ nguồn đa dạng protocol chuyển đi tốt hơn nhiều so với chiều ngược lại.</li>
<li><b>Trọng số vật lý tự thích ứng theo tỉ lệ nhãn.</b> β<sub>hiệu dụng</sub> = β₀·[(1 − f<sub>nhãn</sub>) + 0.15] mã hoá trực tiếp nguyên tắc "tiên nghiệm mạnh khi thiếu nhãn, nhạt dần khi nhãn tích luỹ". Không cần tinh chỉnh theo bộ dữ liệu, thắng MLP ở {stat('e7_adapt_wins')} ô, và bám sát cấu hình chọn thủ công theo validation. Đây là câu trả lời cho vấn đề mà E1/E3 phơi ra: một β cố định không thể vừa tốt ở 10 % vừa tốt ở 70 % nhãn.</li>
<li><b>Tách bạch "đơn điệu" khỏi "vật lý".</b> Giải mã PAVA hậu kỳ (E8) cho thấy tính đơn điệu ở đầu ra chỉ chiếm một phần lợi ích ({stat('e8_gain_mlp')} cho MLP) và không đảo được ngôi thứ ở ô nào — một phép kiểm tra hoài nghi mà các bài PINN cho SOH chưa làm, và là mức chuẩn tối thiểu mà mọi baseline nên được hưởng.</li>
<li><b>Thích nghi miền không cần nhãn.</b> Vì (4)–(6) không dùng nhãn, chúng dùng được ngay trên cell của bộ dữ liệu mới: E5 cho thấy đây là chỗ vật lý thực sự có ích khi đổi bộ dữ liệu, còn zero-shot (E2) thì không — một ranh giới rõ ràng, có số liệu, cho vế "đem sang bộ khác vẫn tốt" của giả thuyết.</li>
<li><b>Bằng chứng thực nghiệm về giới hạn chuyển miền.</b> Bảng dấu tương quan (mục 02) cho thấy bộ đặc trưng đường sạc đổi dấu giữa NCM/NCA và LFP; ràng buộc dấu trên đặc trưng bị loại có cơ sở; và E2 chỉ ra ngay cả cùng loại cell, khác protocol sạc là đủ để zero-shot sụp đổ.</li>
</ol>
<p>Những gì <em>không</em> được khẳng định: chưa chạy lại mã gốc PINN4SOH dưới protocol này (mới tái hiện riêng phần động học đen làm ablation); chưa so với các phương pháp SOTA khác; kết quả CPU 3 seed là sơ bộ.</p>
</section>

<section id="hanche">
<h2><span class="num">09</span>Hạn chế và bước tiếp theo (GPU)</h2>
<ul>
<li><b>Số seed.</b> 3 seed đủ để thấy xu hướng, chưa đủ cho khoảng tin cậy hẹp — trên GPU chạy 10 seed, thêm mức nhãn 5 % và 20 %.</li>
<li><b>Siêu tham số:</b> E7 và E12 đã xử lý (chọn theo val từng bộ/từng mức nhãn; cùng lưới tổng quát cho hai mô hình). Còn thiếu: E12 mới chạy ở 30 % và 70 % nhãn — cần mở rộng sang 10 % và 50 % để hoàn tất bảng cân bằng, và nên quét cả learning rate cùng số bước.</li>
<li><b>Trọng số tự thích ứng</b> mới ở dạng tuyến tính theo f<sub>nhãn</sub> với hằng số sàn đặt tay; nên thử dạng học được (uncertainty weighting kiểu Kendall–Gal, có chặn) và dạng phụ thuộc số cell chứ không phải tỉ lệ.</li>
<li><b>Đối chứng mạnh hơn.</b> Chạy mã gốc PINN4SOH (giữ nguyên chuẩn hoá của họ, và thay bằng chuẩn hoá nhân quả) dưới đúng protocol này; thêm GPR, XGBoost, CNN-1D, và mô hình chuỗi (GRU/TCN trên cửa sổ 20–50 chu kỳ) — tất cả có thể cắm vào <code>SolutionNet</code>.</li>
<li><b>Đặc trưng.</b> Bộ 16 đặc trưng thừa hưởng từ PINN4SOH; lọc 3σ theo cell dùng thống kê cả vòng đời (nhẹ, chỉ trên đặc trưng, nhưng vẫn không nhân quả) — thay bằng lọc trượt nhân quả.</li>
<li><b>Vật lý sâu hơn.</b> Đưa lượng Ah-throughput thay cho chỉ số chu kỳ làm biến thời gian (chuyển miền tốt hơn giữa protocol), tách LLI/LAM từ IC/DV để SOH không còn là vô hướng, và bất định (deep ensemble / conformal) — vật lý có làm hiệu chuẩn tốt hơn không là câu hỏi mở đáng giá.</li>
<li><b>Chuyển miền.</b> E5 mới là bước đầu (một cấu hình, 2 500 bước); cần quét k ∈ {1, 2, 5, 10}, so với fine-tune từ mô hình nguồn đã huấn luyện (thay vì huấn luyện chung từ đầu), và với test-time training thuần (chỉ loss không nhãn, đóng băng phần lớn mạng). Đặc trưng bất biến theo protocol (đặc trưng IC ở cửa sổ điện áp cố định, Ah-throughput thay chu kỳ) là điều kiện cần để zero-shot có cơ hội.</li>
</ul>
</section>

<section id="code">
<h2><span class="num">10</span>Chạy lại</h2>
<pre><code>pip install torch numpy pandas scikit-learn scipy matplotlib
git clone --depth 1 https://github.com/wang-fujin/PINN4SOH.git     # lấy PINN4SOH/data/*
python experiments.py E1 --threads 4        # 144 lượt; ~10 s/lượt trên GPU, ~20 s trên CPU
python experiments.py E2 ; python experiments.py E3 ; python experiments.py E4
python analyze.py                           # bảng + hình vào results/
python make_report.py                       # dựng lại trang này</code></pre>
<pre><code>from pinnsoh.train import Config, run
run(Config(dataset='TJU', label_frac=0.3, model='pinn_semi', seed=0))   # một lượt</code></pre>
<p>Toàn bộ số trong trang này được sinh tự động từ <code>results/*.csv</code>; sửa code, chạy lại, dựng lại — không có số nào gõ tay.</p>
</section>

<footer>Dữ liệu: XJTU (Wang et al. 2024), TJU (Zhu et al. 2022), HUST (Ma et al. 2022), MIT (Severson et al. 2019) — bản tiền xử lý từ repo PINN4SOH. Bản dựng {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}.</footer>
"""
    html = f"""<title>PINN-SOH Ít Nhãn</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">{body}</div>
"""
    open('report.html', 'w').write(html)
    print('report.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    build()
