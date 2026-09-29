"""
E8 — physics-consistent decoding (phân tích hậu kỳ, không huấn luyện lại).

Mạng cho ra u(N) độc lập từng chu kỳ. Nhưng SOH của MỘT cell là một quỹ đạo
đơn điệu không tăng. Vì vậy sau khi dự đoán, ta chiếu quỹ đạo dự đoán của từng
cell test lên tập các dãy không tăng — đúng bài toán hồi quy đơn điệu (PAVA,
O(n)), tức nghiệm bình phương tối tiểu:

    û = argmin_v  Σ (v_i − p_i)²   s.t.  v_1 ≥ v_2 ≥ … ≥ v_n

Đây là ràng buộc vật lý áp ở THỜI ĐIỂM SUY LUẬN, không cần nhãn, không cần
huấn luyện lại. Câu hỏi kiểm chứng: phần lợi ích của PINN có phải chỉ là
"đầu ra trơn và đơn điệu" — thứ có thể lấy miễn phí bằng hậu xử lý — hay không?
Vì vậy decoding được áp cho CẢ MLP lẫn PINN rồi so sánh.

Thêm biến thể `iso+ma`: trung bình trượt (window w) trước khi chiếu, để khử
nhiễu tần số cao mà PAVA một mình không xử lý.
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

OUT = 'results'


def pava_decreasing(p: np.ndarray) -> np.ndarray:
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
    return ir.fit_transform(np.arange(len(p)), p)


def moving_average(p: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(p) < w:
        return p
    k = np.ones(w) / w
    pad = w // 2
    q = np.pad(p, (pad, pad), mode='edge')
    return np.convolve(q, k, mode='valid')[:len(p)]


def decode_run(path: str, w_ma: int = 11) -> dict | None:
    r = json.load(open(path))
    if 'test_predictions' not in r:
        return None
    c = r['config']
    tp = r['test_predictions']
    y = np.asarray(tp['y'], dtype=float); p = np.asarray(tp['p'], dtype=float)
    cell = np.asarray(tp['cell'])
    out = {'raw': p.copy(), 'iso': p.copy(), 'ma': p.copy(), 'ma_iso': p.copy()}
    for k in np.unique(cell):                       # per cell, in cycle order
        m = cell == k
        seg = p[m]
        out['iso'][m] = pava_decreasing(seg)
        sm = moving_average(seg, w_ma)
        out['ma'][m] = sm
        out['ma_iso'][m] = pava_decreasing(sm)
    row = dict(dataset=c['dataset'], model=c['model'], frac=c['label_frac'], seed=c['seed'])
    for name, q in out.items():
        row[f'MAE_{name}'] = float(np.abs(q - y).mean())
        row[f'RMSE_{name}'] = float(np.sqrt(((q - y) ** 2).mean()))
    # how badly does the raw prediction violate monotonicity? (mean positive step per cell)
    v = []
    for k in np.unique(cell):
        seg = p[cell == k]
        d = np.diff(seg)
        v.append(float(d[d > 0].sum()))
    row['mono_violation'] = float(np.mean(v))
    return row


def main():
    rows = [d for d in (decode_run(f) for f in sorted(glob.glob('runs/E1/*.json'))) if d]
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/E8_raw.csv', index=False)
    agg = df.groupby(['dataset', 'model', 'frac']).agg(
        raw=('MAE_raw', 'mean'), iso=('MAE_iso', 'mean'), ma=('MAE_ma', 'mean'), ma_iso=('MAE_ma_iso', 'mean'),
        viol=('mono_violation', 'mean'), n=('seed', 'count')).reset_index()
    agg['gain_iso_%'] = 100 * (1 - agg.iso / agg.raw)
    agg['gain_ma_iso_%'] = 100 * (1 - agg.ma_iso / agg.raw)
    agg.to_csv(f'{OUT}/E8_table.csv', index=False)

    # markdown: best decoded MLP vs best decoded PINN, per dataset x fraction
    lines = ['| dataset | % nhãn | MLP thô | MLP + giải mã | PINN-semi thô | PINN-semi + giải mã | tốt nhất |',
             '|---|---|---|---|---|---|---|']
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        for fr in [0.1, 0.3, 0.5, 0.7]:
            def get(m, col):
                r = agg[(agg.dataset == ds) & (agg.model == m) & (agg.frac == fr)]
                return float(r[col].iloc[0]) if len(r) else np.nan
            mr, md = get('mlp', 'raw'), min(get('mlp', 'iso'), get('mlp', 'ma_iso'))
            pr, pd_ = get('pinn_semi', 'raw'), min(get('pinn_semi', 'iso'), get('pinn_semi', 'ma_iso'))
            if np.isnan(mr): continue
            best = 'PINN' if pd_ < md else 'MLP'
            lines.append(f'| {ds} | {int(fr*100)} % | {mr:.4f} | {md:.4f} | {pr:.4f} | {pd_:.4f} | **{best}** |')
    open(f'{OUT}/E8_table.md', 'w').write('\n'.join(lines))
    print(agg.round(4).to_string(index=False))
    print()
    print(open(f'{OUT}/E8_table.md').read())


if __name__ == '__main__':
    main()
