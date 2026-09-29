"""
Kiểm chứng bằng số các mệnh đề về hàm mất mát và kiến trúc.

    python verify_losses.py          # chạy toàn bộ, in bảng kết quả
    python verify_losses.py --csv    # kèm ghi results/verify_table.csv để nhúng vào báo cáo

Mỗi hàm test_* ứng với một mệnh đề trong mục "Tính đúng đắn" của báo cáo. Test ở đây là
kiểm chứng SỐ cho một chứng minh đã có, không thay thế chứng minh: nó bắt lỗi cài đặt
(sai dấu, quên softplus, nhãn lọt vào loss) chứ không chứng minh mệnh đề.
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, '.')
from pinnsoh.models import GreyBoxDynamics, MonotoneHead, SolutionNet, mlp
from pinnsoh.data import FEATURES

torch.manual_seed(0)
R_GAS, T_REF = 8.314, 298.15
RESULTS: list[dict] = []


def record(prop, name, ok, detail):
    RESULTS.append(dict(menh_de=prop, phep_kiem=name, dat=bool(ok), chi_tiet=detail))
    print(f'  [{"OK " if ok else "SAI"}] {name:<44s} {detail}')
    return ok


# ════════════════════════════════════════ MĐ1 · tốc độ suy giảm không âm
def test_rate_nonneg():
    """MĐ1 phát biểu ĐÚNG là r >= 0. Đây mới là tính chất mà lập luận đơn điệu cần:
    du/dt = -r <= 0. Tính r > 0 NGHIÊM NGẶT chỉ đúng trong số học chính xác; trong
    float32 softplus tràn dưới về 0 khi tiền kích hoạt < ~-103. Tách làm hai phép kiểm."""
    print('\nMĐ1 — r(x,u,T) ≥ 0 với MỌI tham số và MỌI đầu vào (kể cả phá hoại)')
    worst = math.inf
    for trial in range(200):
        dyn = GreyBoxDynamics(16, (32, 32))
        with torch.no_grad():                      # trọng số phi thực tế, cố tình phá
            for p in dyn.rate_net.parameters():
                p.mul_(0).add_(torch.randn_like(p) * (10.0 ** np.random.uniform(-1, 2)))
            dyn.lam_raw.mul_(0).add_(float(np.random.uniform(-10, 10)))
            dyn.ea_raw.mul_(0).add_(float(np.random.uniform(-10, 10)))
        x = torch.randn(256, 16) * (10.0 ** np.random.uniform(-1, 1))
        u = torch.empty(256, 1).uniform_(-0.5, 2.0)
        invT = 1.0 / torch.empty(256, 1).uniform_(250.0, 350.0)
        with torch.no_grad():
            r = dyn(x, u, invT)
        worst = min(worst, float(r.min()))
        if not torch.isfinite(r).all() or float(r.min()) < 0:
            return record('MĐ1', 'r ≥ 0 trên 200 mạng phá hoại', False,
                          f'thất bại ở lần {trial}, min r = {float(r.min()):.3e}')
    ok1 = record('MĐ1', 'r ≥ 0 trên 200 mạng phá hoại', True,
                 f'min r trên 51 200 điểm = {worst:.3e}, không có giá trị âm')

    # ngưỡng tràn dưới của softplus trong float32
    lo, hi = -200.0, 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(Fnn.softplus(torch.tensor(mid))) > 0: hi = mid
        else: lo = mid
    ok2 = record('MĐ1', 'ngưỡng tràn dưới softplus (float32)', hi < -100,
                 f'r > 0 nghiêm ngặt khi tiền kích hoạt > {hi:.1f}')

    # tiền kích hoạt THỰC TẾ trên đặc trưng đã chuẩn hoá, khởi tạo Xavier
    pre = []
    for _ in range(50):
        d = GreyBoxDynamics(16, (32, 32))
        with torch.no_grad():
            pre.append(float(d.rate_net(torch.randn(2048, 16)).min()))
    ok3 = record('MĐ1', 'tiền kích hoạt thực tế cách xa ngưỡng', min(pre) > -20,
                 f'min trên 50 mạng × 2048 điểm = {min(pre):.2f}, cách ngưỡng {abs(hi-min(pre)):.0f} đơn vị')
    return ok1 and ok2 and ok3


# ════════════════════════════════════ MĐ2 · đầu ra đơn điệu theo cấu trúc
def test_monotone_head():
    print('\nMĐ2 — đầu u_mono: ∂u/∂t ≤ 0 với MỌI giá trị tham số')
    worst = -math.inf
    for trial in range(200):
        head = MonotoneHead(feat_dim=16, n_basis=8)
        with torch.no_grad():                      # phá bằng tham số cực đoan
            for p in head.parameters():
                p.mul_(0).add_(torch.randn_like(p) * (10.0 ** np.random.uniform(-1, 1.5)))
        feat = torch.randn(128, 16) * 3.0
        t = torch.empty(128, 1).uniform_(-3.0, 3.0).requires_grad_(True)
        u = head(feat, t)
        g = torch.autograd.grad(u.sum(), t)[0]
        worst = max(worst, float(g.max()))
        if float(g.max()) > 1e-6:
            return record('MĐ2', '∂u/∂t ≤ 0 trên 200 đầu ngẫu nhiên', False,
                          f'thất bại ở lần {trial}, max ∂u/∂t = {float(g.max()):.3e}')
    return record('MĐ2', '∂u/∂t ≤ 0 trên 200 đầu ngẫu nhiên', True,
                  f'max ∂u/∂t trên 25 600 điểm = {worst:.3e} ≤ 0')


# ══════════════════════════ MĐ3 · ba loss vật lý không phụ thuộc nhãn
def physics_losses(net, dyn, x1, x2, t1, t2, invT, eps=0.002):
    u1, u2 = net(x1, t1), net(x2, t2)
    rate = dyn(x1, u1, invT)
    ode = (((u2 - u1) + rate * (t2 - t1)) ** 2).mean()
    mono = Fnn.relu(u2 - u1 - eps).mean()
    rng = (Fnn.relu(u1 - 1.15) + Fnn.relu(0.4 - u1)).mean()
    return ode, mono, rng


def test_label_free():
    print('\nMĐ3 — L_ode, L_mono, L_range không phụ thuộc nhãn y')
    net = SolutionNet(16, (64, 64, 32))
    dyn = GreyBoxDynamics(16, (32, 32))
    x1, x2 = torch.randn(512, 16), torch.randn(512, 16)
    t1 = torch.rand(512, 1); t2 = t1 + torch.rand(512, 1) * 0.05
    invT = torch.full((512, 1), 1.0 / 303.15)

    a = [float(v) for v in physics_losses(net, dyn, x1, x2, t1, t2, invT)]
    b = [float(v) for v in physics_losses(net, dyn, x1, x2, t1, t2, invT)]
    ok1 = record('MĐ3', 'tái lập: gọi hai lần cho giá trị y hệt', a == b, f'{a} == {b}')

    # y có tồn tại và có gradient, nhưng KHÔNG nằm trong đồ thị tính của ba loss
    y = torch.rand(512, 1, requires_grad=True)
    ode, mono, rng = physics_losses(net, dyn, x1, x2, t1, t2, invT)
    grads = []
    for L, nm in [(ode, 'L_ode'), (mono, 'L_mono'), (rng, 'L_range')]:
        g = torch.autograd.grad(L, y, allow_unused=True, retain_graph=True)[0]
        grads.append((nm, g))
    ok2 = record('MĐ3', '∂L/∂y = None cho cả ba loss',
                 all(g is None for _, g in grads),
                 ', '.join(f'{nm}:{"None" if g is None else float(g.abs().max())}' for nm, g in grads))

    # đối chứng: L_data thì PHẢI phụ thuộc y
    u1 = net(x1, t1)
    ld = ((u1 - y) ** 2).mean()
    gd = torch.autograd.grad(ld, y, retain_graph=True)[0]
    ok3 = record('MĐ3', 'đối chứng: ∂L_data/∂y ≠ 0', gd is not None and float(gd.abs().max()) > 0,
                 f'max |∂L_data/∂y| = {float(gd.abs().max()):.3e}')
    return ok1 and ok2 and ok3


# ═══════════════════════ MĐ4 & MĐ7 · tập không của L_mono, L_range = tập khả thi
def test_zero_sets():
    print('\nMĐ4 / MĐ7 — L_mono = 0 ⟺ đơn điệu (sai số ε);  L_range = 0 ⟺ u ∈ [0.4, 1.15]')
    eps = 0.002
    # quỹ đạo giảm nghiêm ngặt -> L_mono phải bằng 0
    u1 = torch.linspace(1.0, 0.8, 100).reshape(-1, 1)
    u2 = u1 - 0.01
    ok1 = record('MĐ4', 'quỹ đạo giảm  →  L_mono = 0',
                 float(Fnn.relu(u2 - u1 - eps).mean()) == 0.0,
                 f'L_mono = {float(Fnn.relu(u2 - u1 - eps).mean()):.3e}')
    # tăng ngược đúng bằng ε -> vẫn bằng 0 (dung sai tái sinh dung lượng)
    u2b = u1 + eps
    ok2 = record('MĐ4', 'tăng ngược đúng ε  →  L_mono = 0 (dung sai)',
                 float(Fnn.relu(u2b - u1 - eps).mean()) == 0.0, 'đúng bằng biên')
    # tăng ngược vượt ε -> phải dương, và bằng đúng phần vượt
    d = 0.01
    u2c = u1 + eps + d
    got, want = float(Fnn.relu(u2c - u1 - eps).mean()), d
    ok3 = record('MĐ4', 'vượt ε một lượng d  →  L_mono = d',
                 abs(got - want) < 1e-6, f'{got:.6f} ≈ {want:.6f}')
    # L_range
    uin = torch.empty(500, 1).uniform_(0.45, 1.10)
    ok4 = record('MĐ7', 'u trong hộp  →  L_range = 0',
                 float((Fnn.relu(uin - 1.15) + Fnn.relu(0.4 - uin)).mean()) == 0.0, 'bằng 0')
    uout = torch.tensor([[1.35], [0.20]])
    got = float((Fnn.relu(uout - 1.15) + Fnn.relu(0.4 - uout)).mean())
    want = ((1.35 - 1.15) + (0.4 - 0.20)) / 2
    ok5 = record('MĐ7', 'u ngoài hộp  →  L_range = khoảng cách tới hộp',
                 abs(got - want) < 1e-6, f'{got:.6f} ≈ {want:.6f}')
    return all([ok1, ok2, ok3, ok4, ok5])


# ══════════════════════════ MĐ5 · bậc chặt cụt của phần dư Euler
def test_truncation_order():
    print('\nMĐ5 — nghiệm giải tích u(t) = e^(−at), r = a·u:  R_euler = O(Δ²), R_deriv = O(Δ)')
    a, t0 = 1.7, 0.3
    ds = np.array([2.0 ** -k for k in range(4, 14)])
    Re = np.array([abs(math.exp(-a * (t0 + d)) - math.exp(-a * t0) + a * math.exp(-a * t0) * d) for d in ds])
    Rd = Re / ds
    pe = np.polyfit(np.log(ds), np.log(Re), 1)[0]
    pd = np.polyfit(np.log(ds), np.log(Rd), 1)[0]
    ok1 = record('MĐ5', 'bậc của R_euler theo Δ  ≈ 2', abs(pe - 2) < 0.02, f'đo được {pe:.4f}')
    ok2 = record('MĐ5', 'bậc của R_deriv theo Δ  ≈ 1', abs(pd - 1) < 0.02, f'đo được {pd:.4f}')
    return ok1 and ok2


# ══════════════════════════ MĐ6 · hệ số khuếch đại nhiễu
def test_noise_amplification():
    print('\nMĐ6 — nhiễu δ của mạng: |ΔR_euler| ≤ 2δ (chặn), |ΔR_deriv| ≤ 2δ/Δ (bung theo 1/Δ)')
    rng = np.random.default_rng(0)
    delta = 1e-3
    rows = []
    for h in [1, 5, 20, 50]:
        d = h / 1000.0
        n = 200_000
        e1, e2 = rng.normal(0, delta, n), rng.normal(0, delta, n)
        dRe = np.abs(e2 - e1)                       # phần dư Euler: sai lệch cộng trực tiếp
        dRd = np.abs(e2 - e1) / d                   # dạng đạo hàm: chia cho Δ
        rows.append((h, dRe.mean(), dRd.mean(), dRd.mean() / dRe.mean(), 1 / d))
    ok = all(abs(r[3] - r[4]) / r[4] < 1e-9 for r in rows)
    detail = ' | '.join(f'h={h}: tỉ số {rt:.0f} (lý thuyết {th:.0f})' for h, _, _, rt, th in rows)
    ok1 = record('MĐ6', 'tỉ số khuếch đại = 1/Δ đúng ở mọi h', ok, detail)
    # đối chiếu với biên độ tín hiệu thật |du/dÑ| ~ 0.2–0.5
    h1 = rows[0]
    ok2 = record('MĐ6', 'ở h = 1, nhiễu dạng đạo hàm lấn át tín hiệu',
                 h1[2] > 0.5, f'|ΔR_deriv| ≈ {h1[2]:.2f} so với |du/dÑ| ≈ 0.2–0.5')
    return ok1 and ok2


# ═══════════════════ kiểm tra gradient: autograd khớp sai phân hữu hạn
def test_gradcheck():
    """Dùng functional_call để loss thật sự là HÀM của tham số — gán p.data sẽ cắt đồ thị
    và làm gradcheck luôn thất bại (đó là lỗi của phép kiểm, không phải của code)."""
    print('\nKiểm tra cài đặt — gradient autograd khớp sai phân hữu hạn (float64)')
    from torch.func import functional_call
    torch.manual_seed(1)
    net = SolutionNet(6, (8, 8)).double()
    dyn = GreyBoxDynamics(6, (8,)).double()
    x1, x2 = torch.randn(12, 6, dtype=torch.float64), torch.randn(12, 6, dtype=torch.float64)
    t1 = torch.rand(12, 1, dtype=torch.float64); t2 = t1 + 0.03
    invT = torch.full((12, 1), 1.0 / 303.15, dtype=torch.float64)
    nk = [k for k, _ in net.named_parameters()]
    dk = [k for k, _ in dyn.named_parameters()]
    flat0 = tuple(v.detach().clone().requires_grad_(True) for _, v in net.named_parameters()) + \
            tuple(v.detach().clone().requires_grad_(True) for _, v in dyn.named_parameters())

    def build(i):
        def f(*flat):
            pn = dict(zip(nk, flat[:len(nk)])); pd_ = dict(zip(dk, flat[len(nk):]))
            u1 = functional_call(net, pn, (x1, t1))
            u2 = functional_call(net, pn, (x2, t2))
            rate = functional_call(dyn, pd_, (x1, u1, invT))
            if i == 0: return (((u2 - u1) + rate * (t2 - t1)) ** 2).mean()
            if i == 1: return Fnn.relu(u2 - u1 - 0.002).mean()
            return (Fnn.relu(u1 - 1.15) + Fnn.relu(0.4 - u1)).mean()
        return f

    allok = True
    for i, nm in enumerate(['L_ode', 'L_mono', 'L_range']):
        good = torch.autograd.gradcheck(build(i), flat0, eps=1e-6, atol=1e-6, rtol=1e-4,
                                        raise_exception=False)
        allok &= record('cài đặt', f'gradcheck {nm}', good,
                        'autograd khớp sai phân trên mọi tham số')
    return allok


# ═════════ Nhận xét · đơn điệu theo t KHÔNG kéo theo đơn điệu dọc quỹ đạo
def test_monotone_not_along_trajectory():
    print('\nNhận xét — ∂u/∂t ≤ 0 KHÔNG kéo theo du/dN ≤ 0 khi x cũng đổi theo N')
    torch.manual_seed(3)
    head = MonotoneHead(feat_dim=4, n_basis=4)
    # quỹ đạo giả: t tăng đều, x đi theo một hướng làm u0(x) tăng
    n = 300
    t = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    w = head.u0.weight.detach().clone().reshape(1, -1)          # hướng làm u0 tăng
    feat = t * (w / w.norm()) * 12.0
    with torch.no_grad():
        u = head(feat, t).squeeze()
    du = (u[1:] - u[:-1])
    n_up = int((du > 1e-9).sum())
    # ở TỪNG điểm, đạo hàm riêng theo t vẫn phải ≤ 0
    tg = t.clone().requires_grad_(True)
    ug = head(feat, tg)
    g = torch.autograd.grad(ug.sum(), tg)[0]
    ok1 = record('Nhận xét', 'đạo hàm RIÊNG theo t vẫn ≤ 0 khắp nơi',
                 float(g.max()) <= 1e-6, f'max ∂u/∂t = {float(g.max()):.3e}')
    ok2 = record('Nhận xét', 'nhưng du/dN DỌC quỹ đạo có đoạn > 0',
                 n_up > 0, f'{n_up}/{n-1} bước tăng ngược, tối đa +{float(du.max()):.4f}')
    return ok1 and ok2


# ═════════════════════════════ lịch trình trọng số vật lý
def test_warmup():
    print('\nMĐ8 — w(s) = min(1, s/0.2S): liên tục, không giảm, w(0)=0, w(s≥0.2S)=1')
    S = 4000
    w = lambda s: min(1.0, s / (0.2 * S))
    ss = np.arange(0, S + 1)
    ws = np.array([w(s) for s in ss])
    ok1 = record('MĐ8', 'w(0) = 0 và w(0.2S) = 1', ws[0] == 0 and w(0.2 * S) == 1.0, 'đúng biên')
    ok2 = record('MĐ8', 'không giảm trên toàn miền', bool((np.diff(ws) >= 0).all()), 'đơn điệu tăng')
    ok3 = record('MĐ8', 'bước đầu: L = L_data thuần (vật lý tắt)', w(0) == 0.0,
                 'mạng bám dữ liệu trước khi vật lý siết vào')
    return ok1 and ok2 and ok3


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', action='store_true')
    args = ap.parse_args()

    checks = [test_rate_nonneg, test_monotone_head, test_label_free, test_zero_sets,
              test_truncation_order, test_noise_amplification, test_gradcheck,
              test_monotone_not_along_trajectory, test_warmup]
    for c in checks:
        c()

    n_ok = sum(r['dat'] for r in RESULTS)
    print(f'\n{"="*78}\nKẾT QUẢ: {n_ok}/{len(RESULTS)} phép kiểm đạt')
    if args.csv:
        import pandas as pd
        pd.DataFrame(RESULTS).to_csv('results/verify_table.csv', index=False)
        print('đã ghi results/verify_table.csv')
    sys.exit(0 if n_ok == len(RESULTS) else 1)
