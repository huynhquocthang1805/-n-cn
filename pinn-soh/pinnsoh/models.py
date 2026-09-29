"""
Mạng nghiệm (solution net) và động học suy giảm.

Mạng nghiệm  u = F_phi(x, t)     x: 16 đặc trưng sạc đã chuẩn hoá, t = chu kỳ/1000
Động học xám du/dt = -k(x)·g(u)·A(T)  với k >= 0, g = exp(lam(1-u)), A = Arrhenius

Hàm kích hoạt (`act`):
    silu   : mặc định, trơn, không bão hoà — tốt cho hồi quy
    tanh   : kinh điển cho PINN (đạo hàm bậc cao trơn), nhưng bão hoà
    sin    : PINN4SOH dùng; biểu diễn tốt tín hiệu tuần hoàn, dễ mất ổn định
    gelu   : như silu, đuôi khác
    snake  : x + sin^2(a x)/a — thiết kế cho tín hiệu có xu thế + dao động (Ziyin 2020),
             phù hợp SOH vì quỹ đạo = xu thế giảm + dao động tái sinh dung lượng

Kiến trúc (`arch`):
    mlp     : MLP thuần (cơ sở)
    res     : khối residual — gradient đi thẳng, huấn luyện ổn định hơn khi sâu
    fourier : mã hoá Fourier ngẫu nhiên cho t trước khi ghép với x — giúp mạng
              biểu diễn biến thiên theo chu kỳ mà không cần sâu
    mono    : ĐẦU RA ĐƠN ĐIỆU THEO CẤU TRÚC (đóng góp thiết kế):
                  u(x,t) = u0(x) - sum_k s_k(x) * softplus(w_k * t + phi_k(x)),  s_k, w_k >= 0
              => du/dt = -sum_k s_k w_k sigmoid(.) <= 0 với MỌI tham số.
              Tính đơn điệu không còn là một số hạng phạt cần cân trọng số mà là
              tính chất của không gian hàm; L_mono trở nên thừa (đặt beta = 0).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

R_GAS = 8.314
T_REF = 298.15


# --------------------------------------------------------------------------- activations
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Snake(nn.Module):
    """x + sin^2(a x)/a — giữ xu thế tuyến tính, thêm khả năng biểu diễn dao động."""
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))

    def forward(self, x):
        a = self.a.clamp_min(1e-3)
        return x + torch.sin(a * x) ** 2 / a


ACTS = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'sin': Sin, 'gelu': nn.GELU, 'snake': Snake}


def make_act(name: str) -> nn.Module:
    if name not in ACTS:
        raise ValueError(f'unknown activation {name}; choose from {list(ACTS)}')
    return ACTS[name]()


def _init(net: nn.Module, act: str):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if act in ('sin', 'snake'):
                nn.init.xavier_normal_(m.weight, gain=1.0)
            else:
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu') * 0.7)
            nn.init.zeros_(m.bias)


def mlp(in_dim, hidden, out_dim, act='silu', dropout=0.0):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), make_act(act)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    _init(net, act)
    return net


# --------------------------------------------------------------------------- backbones
class ResMLP(nn.Module):
    def __init__(self, in_dim, width, depth, out_dim, act='silu', dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.act = make_act(act)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), make_act(act),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(width, out_dim)
        _init(self, act)

    def forward(self, z):
        h = self.act(self.inp(z))
        for b in self.blocks:
            h = self.act(h + b(h))
            h = self.drop(h)
        return self.out(h)


class FourierT(nn.Module):
    """Mã hoá Fourier ngẫu nhiên (cố định) cho biến thời gian t."""
    def __init__(self, n_freq=8, scale=4.0):
        super().__init__()
        self.register_buffer('B', torch.randn(1, n_freq) * scale)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, t):
        z = 2 * math.pi * t @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=1)


class MonotoneHead(nn.Module):
    """u(x,t) = u0(x) - sum_k s_k(x)*softplus(w_k*t + phi_k(x)),  s_k >= 0, w_k >= 0.

    du/dt = -sum_k s_k * w_k * sigmoid(w_k t + phi_k) <= 0  --> đơn điệu không tăng theo
    cấu trúc, với mọi giá trị tham số. u0 bị chặn về [lo, hi] bằng sigmoid có tỉ lệ."""
    def __init__(self, feat_dim, n_basis=8, lo=0.40, hi=1.15):
        super().__init__()
        self.n = n_basis; self.lo, self.hi = lo, hi
        self.u0 = nn.Linear(feat_dim, 1)
        self.s = nn.Linear(feat_dim, n_basis)          # -> softplus -> biên độ >= 0
        self.phi = nn.Linear(feat_dim, n_basis)        # dịch pha phụ thuộc x
        self.w_raw = nn.Parameter(torch.linspace(-1.0, 2.0, n_basis))   # -> softplus -> nhịp >= 0
        nn.init.xavier_normal_(self.u0.weight); nn.init.zeros_(self.u0.bias)
        nn.init.xavier_normal_(self.s.weight); nn.init.constant_(self.s.bias, -2.0)
        nn.init.xavier_normal_(self.phi.weight); nn.init.zeros_(self.phi.bias)

    def forward(self, feat, t):
        u0 = self.lo + (self.hi - self.lo) * torch.sigmoid(self.u0(feat) + 2.0)   # khởi tạo gần 1.0
        s = Fnn.softplus(self.s(feat))                                            # (B, n) >= 0
        w = Fnn.softplus(self.w_raw).unsqueeze(0)                                 # (1, n) >= 0
        drop = (s * Fnn.softplus(w * t + self.phi(feat))).sum(dim=1, keepdim=True)
        return u0 - drop


class SolutionNet(nn.Module):
    """u = F(x, t) với backbone và hàm kích hoạt chọn được."""
    def __init__(self, n_feat=16, hidden=(64, 64, 32), dropout=0.0, use_cycle=True,
                 act='silu', arch='mlp', n_freq=8, n_basis=8):
        super().__init__()
        self.use_cycle, self.arch = use_cycle, arch
        self.fourier = FourierT(n_freq) if arch == 'fourier' else None
        t_dim = (self.fourier.out_dim if self.fourier is not None else 1) if use_cycle else 0
        in_dim = n_feat + t_dim
        if arch == 'res':
            self.net = ResMLP(in_dim, hidden[0], max(1, len(hidden) - 1), 1, act, dropout)
        elif arch == 'mono':
            # backbone chỉ ăn x; t đi qua đầu ra đơn điệu
            self.trunk = mlp(n_feat, hidden[:-1], hidden[-1], act, dropout)
            self.act = make_act(act)
            self.head = MonotoneHead(hidden[-1], n_basis)
            self.net = None
        else:                                     # 'mlp' hoặc 'fourier'
            self.net = mlp(in_dim, hidden, 1, act, dropout)

    def forward(self, x, t):
        if self.arch == 'mono':
            return self.head(self.act(self.trunk(x)), t)
        if not self.use_cycle:
            return self.net(x)
        tt = self.fourier(t) if self.fourier is not None else t
        return self.net(torch.cat([x, tt], dim=1))


# --------------------------------------------------------------------------- dynamics
class GreyBoxDynamics(nn.Module):
    """Tốc độ suy giảm không âm, có cấu trúc vật lý (đơn vị: trên 1000 chu kỳ)."""
    def __init__(self, n_feat=16, hidden=(32, 32), use_knee=True, use_arrhenius=True,
                 lam_init=1.0, ea_init_kj=30.0, act='silu'):
        super().__init__()
        self.rate_net = mlp(n_feat, hidden, 1, act)
        self.use_knee, self.use_arrhenius = use_knee, use_arrhenius
        self.lam_raw = nn.Parameter(torch.tensor(math.log(math.exp(lam_init) - 1.0)))
        self.ea_raw = nn.Parameter(torch.tensor(math.log(math.exp(ea_init_kj / 10.0) - 1.0)))

    @property
    def lam(self):
        return Fnn.softplus(self.lam_raw)

    @property
    def Ea(self):        # J / mol
        return 1e4 * Fnn.softplus(self.ea_raw)

    def forward(self, x, u, invT):
        k = Fnn.softplus(self.rate_net(x))
        if self.use_knee:
            k = k * torch.exp(self.lam * (1.0 - u).clamp(-0.5, 1.0))
        if self.use_arrhenius:
            k = k * torch.exp(-self.Ea / R_GAS * (invT - 1.0 / T_REF))
        return k


class BlackBoxDynamics(nn.Module):
    """Kiểu PINN4SOH: G(x, t, u, u_x, u_t) -> du/dt (dùng cho ablation)."""
    def __init__(self, n_feat=16, hidden=(60, 60, 60)):
        super().__init__()
        self.net = mlp(n_feat + 1 + 1 + n_feat + 1, hidden, 1, act='sin', dropout=0.2)

    def forward(self, x, t, u, u_x, u_t):
        return self.net(torch.cat([x, t, u, u_x, u_t], dim=1))


class SOHModel(nn.Module):
    def __init__(self, dyn: str = 'none', n_feat=16, hidden=(64, 64, 32), dropout=0.0,
                 use_cycle=True, use_knee=True, use_arrhenius=True, act='silu', arch='mlp'):
        super().__init__()
        self.dyn_kind = dyn
        self.u = SolutionNet(n_feat, hidden, dropout, use_cycle, act=act, arch=arch)
        if dyn == 'greybox':
            self.dyn = GreyBoxDynamics(n_feat, use_knee=use_knee, use_arrhenius=use_arrhenius, act=act)
        elif dyn == 'blackbox':
            self.dyn = BlackBoxDynamics(n_feat)
        else:
            self.dyn = None

    def forward(self, x, t):
        return self.u(x, t)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
