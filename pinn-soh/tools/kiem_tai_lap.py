"""
Kiểm tra máy này có tái lập đúng kết quả của báo cáo không: chạy lại 20 lượt mẫu (đủ mọi loại thí nghiệm
E13–E19, bộ XJTU, lần lặp 0, fold 0) vào thư mục tạm rồi so với giá trị đã công bố.

    python tools/kiem_tai_lap.py [--workers 4]        # ~5 phút CPU

So sánh MAE theo cell (đủ 17 chữ số) và mã băm của toàn bộ dự đoán test. Trên cùng phiên bản thư viện và
cùng loại CPU, mọi dòng phải "giống từng bit". Khác CPU (ví dụ AVX-512 so với AVX2, Apple Silicon) hoặc
khác phiên bản torch có thể lệch ở mức làm tròn số thực (~1e-6–1e-4 trên MAE); khi đó kết luận của báo cáo
không đổi nhưng các con số cuối trong bảng có thể khác ở chữ số thứ ba, thứ tư.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import argparse, copy, hashlib, json, os, tempfile

# (thí nghiệm, file kết quả gốc, MAE_cell, sha1 của dự đoán test) — trích từ runs/ của lần chạy công bố
REFERENCE = [
    ('E13', 'runs/E13/A_XJTU_mlp_f0.3_s0_global_global_cv0-0of5.json', 0.016669597686809864, '66684c6c3d3722b6'),
    ('E13', 'runs/E13/C_XJTU_pinn_semi_f0.3_s0_global_global_cv0-0of5.json', 0.017147821766788793, 'ede17fbc24285eb9'),
    ('E13', 'runs/E13/F_XJTU_mlp_f0.7_s0_global_global_cv0-0of5.json', 0.013636935291851478, '4703ece7aab73289'),
    ('E13', 'runs/E13/B_XJTU_pinn_sup_f0.3_s0_global_global_cv0-0of5.json', 0.01583427033372895, '8ab47ade882a935c'),
    ('E13', 'runs/E13/D_XJTU_pinn_semi_f0.3_s0_global_global_cv0-0of5.json', 0.017487423444370688, '284b49cc8fb47f6a'),
    ('E13', 'runs/E13/E_XJTU_pinn_semi_f0.3_s0_global_global_cv0-0of5.json', 0.013786960205684746, '0c546e2125df739b'),
    ('E13', 'runs/E13/G_XJTU_pinn_semi_f0.7_s0_global_global_cv0-0of5.json', 0.013507377846872693, 'c36b26c178d34703'),
    ('E17', 'runs/E17/B_XJTU_pinn_sup_f0.1_s0_global_global_cv0-0of5.json', 0.017075068681779035, 'c19cab7df9dbf6fa'),
    ('E17', 'runs/E17/H_XJTU_pinn_bb_f0.1_s0_global_global_cv0-0of5.json', 0.016652595091588715, '2ce6600b95109ee2'),
    ('E17', 'runs/E17/H_XJTU_pinn_bb_f0.3_s0_global_global_cv0-0of5.json', 0.015088877067618023, '3a3a2e79e4ba4b4b'),
    ('E15', 'runs/E15/row_XJTU_cv0-0.json', 0.007203264112831723, 'd6a62225987349a2'),
    ('E15', 'runs/E15/minmax_XJTU_cv0-0.json', 0.00958135805329788, '69fcc1d6159df123'),
    ('E15', 'runs/E15/life3s_XJTU_cv0-0.json', 0.011307267923460011, '2512d63c910d1e35'),
    ('E15', 'runs/E15/normtest_XJTU_cv0-0.json', 0.012908311081133621, 'e86a69e89dc02d93'),
    ('E16', 'runs/E16/ridge_XJTU_f0.3_cv0-0of5.json', 0.015626054176980694, '3f7f77789832737a'),
    ('E16', 'runs/E16/gbdt_XJTU_f0.3_cv0-0of5.json', 0.010007993938980327, '13c652eb15148f24'),
    ('E16', 'runs/E16/gpr_XJTU_f0.3_cv0-0of5.json', 0.013677598487468173, 'fe27ef23680f31e4'),
    ('E18', 'runs/E14/samenorm_XJTU_to_TJU_mlp_k3_s0_global.json', 0.08961263605146434, None),
    ('E18', 'runs/E14/samenorm_XJTU_to_TJU_pinn_semi_k3_s0_global.json', 0.027250138722303627, None),
    ('E19', 'runs/E14/samenorm_XJTU_to_TJU_gbdt_k3_s0_global.json', 0.028748334302731025, None),
]


def pred_hash(res):
    tp = res.get('test_predictions')
    return hashlib.sha1(json.dumps(tp['p']).encode()).hexdigest()[:16] if tp else None


def main():
    from pinnsoh import experiments as E
    from pinnsoh.runner import run_jobs
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    want = {out: (exp, mae, h) for exp, out, mae, h in REFERENCE}
    jobs = [j for exp in sorted({r[0] for r in REFERENCE}) for j in E.EXPERIMENTS[exp]() if j.out in want]
    assert len({j.out for j in jobs}) == len(REFERENCE), 'thiếu lượt mẫu'
    with tempfile.TemporaryDirectory() as tmp:
        moved = []
        for j in jobs:
            j2 = copy.deepcopy(j)
            j2.args['out_dir'] = tmp
            j2.out = os.path.join(tmp, os.path.basename(j.out))
            moved.append((j, j2))
        n_err = run_jobs([j2 for _, j2 in moved], workers=a.workers)
        ok = 0
        print(f'\n{"thí nghiệm":<10s} {"lượt":<52s} kết quả')
        for j, j2 in moved:
            exp, mae, h = want[j.out]
            if not os.path.exists(j2.out):
                print(f'{exp:<10s} {os.path.basename(j.out):<52s} LỖI (không có kết quả)')
                continue
            r = json.load(open(j2.out))
            got = (r.get('target_test') or r['test'])['MAE_cell']
            same = got == mae and (h is None or pred_hash(r) == h)
            ok += same
            msg = 'giống từng bit' if same else f'KHÁC: MAE_cell {got!r} so với {mae!r} (lệch {abs(got - mae):.2e})'
            print(f'{exp:<10s} {os.path.basename(j.out):<52s} {msg}')
    print(f'\n{ok}/{len(REFERENCE)} lượt giống từng bit với kết quả công bố'
          + ('' if ok == len(REFERENCE) else ' — xem ghi chú ở đầu file về khác biệt do CPU/phiên bản thư viện'))
    return 0 if ok == len(REFERENCE) and not n_err else 1


if __name__ == '__main__':
    raise SystemExit(main())
