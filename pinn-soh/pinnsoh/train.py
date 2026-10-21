"""
Training / evaluation with physics-informed, label-free losses.

model kinds
  'mlp'        : data loss only (baseline)
  'pinn_sup'   : data + physics losses, physics evaluated on LABELLED cells only   (physics = regulariser)
  'pinn_semi'  : data + physics losses, physics evaluated on ALL training cells,
                 i.e. also on cells WITHOUT capacity labels                        (physics = label-free supervision)
  'pinn_bb'    : PINN4SOH-style black-box dynamics (ablation), semi-supervised like pinn_semi

Physics losses are evaluated on (N, N+h) pairs of the same cell with a random horizon
1 <= h <= h_max cycles.  Using a horizon (instead of only consecutive cycles) keeps the
finite-difference derivative  (u(N+h)-u(N))/(h/1000)  well above the network's own noise.

Pipeline v2 (Config.v2() bật cả năm): clean='causal', norm_pool='train' (MỌI mô hình dùng CÙNG một
normaliser fit trên đặc trưng của mọi cell train — hợp lệ vì đặc trưng không phải nhãn — nên khác biệt
giữa MLP và PINN chỉ còn là hàm mất mát), pair_by='cycle', nhãn tuỳ chọn, EOL theo chu kỳ.
split='cv' dùng kiểm định chéo K-fold theo cell (data.cv_split).
"""
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.autograd import grad

from .data import (Normalizer, PointSet, load_dataset, split_cells, choose_labelled, make_pointset, cv_split)
from .models import SOHModel
from .metrics import evaluate as eval_metrics


@dataclass
class Config:
    dataset: str = 'XJTU'
    label_frac: float = 0.7          # fraction of ALL cells that carry labels (0.7 == full 70/15/15)
    model: str = 'pinn_semi'         # mlp | pinn_sup | pinn_semi | pinn_bb
    seed: int = 0
    norm: str = 'global'             # global | first_cycle | per_cell_minmax
    cyc: str = 'global'              # global | per_cell_minmax   (cycle-index input)
    use_cycle: bool = True
    # physics-loss weights
    alpha_ode: float = 2.0
    beta_mono: float = 5.0
    gamma_range: float = 1.0
    eps_mono: float = 0.002
    h_min: int = 5                   # min / max horizon (cycles) for physics pairs
    h_max: int = 50
    warmup_frac: float = 0.2         # physics weights ramp 0 -> 1 over this fraction of max_steps
    adapt_w: bool = False            # scale physics weights by the UNLABELLED fraction of the training pool:
                                     #   w_adapt = (1 - f_lab) + floor,  f_lab = n_labelled / n_train_cells
                                     # the physics prior is a substitute for missing labels, so its weight
                                     # should fall as labels accumulate -- no per-fraction tuning needed.
    adapt_floor: float = 0.15
    residual: str = 'euler'          # euler | fd | autograd
                                     #  euler : (u(N+h)-u(N)) + rate*dt   (integrated form, horizon-weighted)
                                     #  fd    : (u(N+h)-u(N))/dt + rate   (derivative form)
    use_knee: bool = True
    use_arrhenius: bool = True
    # optimisation
    max_steps: int = 4000
    eval_every: int = 100
    patience: int = 12               # evaluations without val improvement
    batch: int = 512
    lr: float = 2e-3
    wd: float = 1e-5
    hidden: tuple = (64, 64, 32)
    dropout: float = 0.0
    act: str = 'silu'                # silu | tanh | sin | gelu | snake
    arch: str = 'mlp'                # mlp | res | fourier | mono  (mono = đơn điệu theo cấu trúc)
    # bookkeeping
    root: str = 'PINN4SOH/data'
    out_dir: str = 'runs'
    tag: str = ''
    save_predictions: bool = True
    # cross-dataset evaluation targets (evaluated with the SAME normaliser fitted on source train cells)
    transfer_to: tuple = ()
    transfer_test_only: bool = True  # đánh giá trên tập TEST của bộ đích (cùng seed) -> đường chéo của
                                     # ma trận chuyển miền trùng đúng kết quả trong miền
    # ---- pipeline v2 (mặc định = v1 để 1 387 log cũ tái lập được)
    clean: str = 'life3sigma'        # life3sigma (v1) | causal (v2: chỉ dùng quá khứ của cell)
    norm_pool: str = 'model'         # model (v1: semi -> mọi cell train, còn lại -> cell có nhãn) | train (v2: mọi mô hình)
    pair_by: str = 'row'             # row (v1: h đếm theo hàng sau lọc) | cycle (v2: h đếm theo chu kỳ gốc)
    split: str = 'seed'              # seed (70/15/15 gieo theo seed) | cv (K-fold theo cell, lặp lại)
    fold: int = 0
    n_folds: int = 5
    repeat: int = 0
    val_frac_cv: float = 0.10
    save_model: bool = False         # lưu trọng số + normaliser (<tên>.pt) để dùng cho demo
    # ---- cải tiến E20 (mặc định tắt -> mọi lượt cũ giữ nguyên từng bit)
    member: int = 0                  # thành viên ensemble: đổi hạt giống khởi tạo và lấy mẫu batch,
                                     # GIỮ NGUYÊN cách chia và tập cell có nhãn (chọn theo `seed`)
    alpha_euler: float = 0.0         # residual='autograd': thêm phần dư Euler dọc quỹ đạo thật với cùng r
    distill_w: float = 0.0           # > 0: GBDT học trên cell có nhãn làm mô hình thầy; dự đoán của nó trên
                                     # cell không nhãn là nhãn mềm cho mạng (học bán giám sát kiểu tự huấn luyện)
    save_val_predictions: bool = False
    range_first: tuple = (0.85, 1.10)  # miền SOH hợp lý ở chu kỳ đầu của cell; NASA (cell 4 °C có SOH đầu 0,5–0,8 so
                                     # với 2 Ah danh định) dùng (0,40; 1,10)

    def v2(self) -> 'Config':
        """Bật đủ năm sửa đổi của pipeline v2."""
        self.clean, self.norm_pool, self.pair_by = 'causal', 'train', 'cycle'
        return self


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


def metrics(y: np.ndarray, p: np.ndarray, cell_idx: Optional[np.ndarray] = None,
            cycle: Optional[np.ndarray] = None) -> Dict[str, float]:
    return eval_metrics(y, p, cell_idx, cycle=cycle)


_CACHE: Dict[tuple, list] = {}


def _load(cfg: Config, name: str):
    # chỉ truyền `clean` khi khác v1 -> các script cũ vá load_dataset(root, name) vẫn chạy.
    # v2 được nhớ đệm trong tiến trình (Cell không bị sửa ở đâu cả; transform luôn tạo mảng mới).
    if cfg.clean == 'life3sigma':
        return load_dataset(cfg.root, name)
    key = (cfg.root, name, cfg.clean)
    if key not in _CACHE:
        _CACHE[key] = load_dataset(cfg.root, name, clean=cfg.clean)
    return list(_CACHE[key])


def default_device() -> str:
    """'cpu' trừ khi đặt PINNSOH_DEVICE (ví dụ 'cuda'). Mọi kết quả trong báo cáo chạy trên CPU một luồng;
    GPU cho kết quả lệch ở mức làm tròn số thực, không tái lập từng bit."""
    return os.environ.get('PINNSOH_DEVICE', 'cpu')


class _Tensors:
    """Các cột của một PointSet dưới dạng tensor, dựng MỘT lần cho cả lượt huấn luyện (trên CPU là view
    không sao chép của mảng numpy). Lấy batch bằng index_select cho đúng các giá trị mà cách cũ
    torch.from_numpy(a[idx]) cho ra, nên kết quả không đổi."""
    __slots__ = ('X', 'T', 'Y', 'invT')

    def __init__(self, S: PointSet, dev):
        self.X, self.T, self.Y, self.invT = (torch.from_numpy(a).to(dev) for a in (S.X, S.T, S.Y, S.invT))


class Trainer:
    def __init__(self, cfg: Config, device=None):
        self.cfg, self.dev = cfg, device or default_device()
        self._tcache = {}
        set_seed(self.train_seed)

    @property
    def train_seed(self) -> int:
        return self.cfg.seed + 7919 * self.cfg.member

    def _t(self, S: PointSet) -> _Tensors:
        key = id(S)
        if key not in self._tcache:
            self._tcache[key] = (S, _Tensors(S, self.dev))     # giữ tham chiếu S để id không bị tái dùng
        return self._tcache[key][1]

    def _gather(self, S: PointSet, idx: np.ndarray):
        tt, ix = self._t(S), torch.from_numpy(idx).to(self.dev)
        return tt.X.index_select(0, ix), tt.T.index_select(0, ix), tt.Y.index_select(0, ix)

    # ---------------------------------------------------------------- data
    def prepare(self):
        cfg = self.cfg
        cells = _load(cfg, cfg.dataset)
        by_id = {c.cid: c for c in cells}
        if cfg.split == 'cv':
            split = cv_split(cells, cfg.fold, cfg.repeat, cfg.n_folds, cfg.val_frac_cv)
        elif cfg.split == 'samba':
            from .data import SAMBA_SPLIT
            split = {k: [f'NASA/{c}' for c in v] for k, v in SAMBA_SPLIT.items()}
        else:
            split = split_cells(cells, cfg.seed)
        lab_ids, unl_ids = choose_labelled(split['train'], cfg.label_frac, len(cells), cfg.seed, by_id)
        self.split = dict(split, labelled=lab_ids, unlabelled=unl_ids)
        tr_lab = [by_id[i] for i in lab_ids]; tr_unl = [by_id[i] for i in unl_ids]
        va = [by_id[i] for i in split['val']]; te = [by_id[i] for i in split['test']]
        semi = cfg.model in ('pinn_semi', 'pinn_bb')
        # Normaliser: fitted on features only (labels never used).
        #   v1 (norm_pool='model'): semi-supervised models use all training cells' features, purely
        #       supervised models only labelled cells -> MLP và PINN khác nhau CẢ ở normaliser.
        #   v2 (norm_pool='train'): mọi mô hình dùng cùng pool đặc trưng của mọi cell train.
        pool = tr_lab + tr_unl if (semi or cfg.norm_pool == 'train') else tr_lab
        self.norm = Normalizer(cfg.norm).fit(pool)
        lab_only = cfg.clean != 'life3sigma'          # v2: loss dữ liệu / val / test chỉ trên hàng có nhãn
        self.S_lab = make_pointset(tr_lab, self.norm, cfg.cyc, labelled_only=lab_only)
        self.S_unl = make_pointset(tr_unl, self.norm, cfg.cyc) if (semi and tr_unl) else None
        self.val = make_pointset(va, self.norm, cfg.cyc, labelled_only=lab_only)
        self.test = make_pointset(te, self.norm, cfg.cyc, labelled_only=lab_only)
        self.test_ids = [c.cid for c in te]
        self.val_ids = [c.cid for c in va]
        if cfg.distill_w > 0 and self.S_unl is not None:
            self.teacher_unl = self._teacher(tr_lab, va, self.S_unl)
        self.n_cells = dict(train_labelled=len(tr_lab), train_unlabelled=len(tr_unl), val=len(va), test=len(te),
                            all=len(cells))
        self.transfer = {}
        for tgt in cfg.transfer_to:
            tcells = _load(cfg, tgt)
            if cfg.transfer_test_only:
                keep = set(split_cells(tcells, cfg.seed)['test'])
                tcells = [c for c in tcells if c.cid in keep]
            self.transfer[tgt] = make_pointset(tcells, self.norm, cfg.cyc)
        return self

    def _teacher(self, lab_cells, val_cells, S_unl: PointSet) -> torch.Tensor:
        """GBDT (cùng cấu hình và cách chọn số cây theo validation như đối chứng E16) học trên cell có nhãn,
        trả dự đoán của nó cho mọi điểm của cell không nhãn. Không dùng nhãn nào ngoài cell có nhãn."""
        from .baselines import fit_predict
        lab_only = self.cfg.clean != 'life3sigma'
        Sl = make_pointset(lab_cells, self.norm, self.cfg.cyc, labelled_only=lab_only)
        Sv = make_pointset(val_cells, self.norm, self.cfg.cyc, labelled_only=lab_only)
        xy = lambda S: np.hstack([S.X, S.T])
        p, v, hp = fit_predict('gbdt', xy(Sl), Sl.Y.ravel(), xy(Sv), Sv.Y.ravel(), xy(S_unl), self.cfg.seed)
        self.teacher_info = dict(val_MAE=v, **hp)
        return torch.from_numpy(np.asarray(p, dtype=np.float32).reshape(-1, 1)).to(self.dev)

    def _pairs(self, S: PointSet, rng, batch):
        sampler = S.sample_pairs_cycle if self.cfg.pair_by == 'cycle' else S.sample_pairs
        i, j = sampler(rng, batch, self.cfg.h_min, self.cfg.h_max)
        tt = self._t(S)
        ii, jj = torch.from_numpy(i).to(self.dev), torch.from_numpy(j).to(self.dev)
        return dict(x1=tt.X.index_select(0, ii), x2=tt.X.index_select(0, jj), t1=tt.T.index_select(0, ii),
                    t2=tt.T.index_select(0, jj), y1=tt.Y.index_select(0, ii), y2=tt.Y.index_select(0, jj),
                    invT=tt.invT.index_select(0, ii), first=torch.from_numpy(i == S.start[i]).to(self.dev))

    # ---------------------------------------------------------------- losses
    def physics(self, model: SOHModel, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        x1, x2, t1, t2, invT = b['x1'], b['x2'], b['t1'], b['t2'], b['invT']
        out = {}
        if model.dyn_kind == 'blackbox':
            x1 = x1.clone().requires_grad_(True); t1 = t1.clone().requires_grad_(True)
            u1 = model.u(x1, t1)
            u_t = grad(u1.sum(), t1, create_graph=True)[0]
            u_x = grad(u1.sum(), x1, create_graph=True)[0]
            G = model.dyn(x1, t1, u1, u_x, u_t)
            out['ode'] = ((u_t - G) ** 2).mean()
            u2 = model.u(x2, t2)
        elif model.dyn_kind == 'greybox':
            if cfg.residual == 'autograd':
                t1 = t1.clone().requires_grad_(True)
                u1 = model.u(x1, t1)
                u_t = grad(u1.sum(), t1, create_graph=True)[0]
                u2 = model.u(x2, t2)
            else:                                   # finite difference along the real trajectory
                u1 = model.u(x1, t1); u2 = model.u(x2, t2)
                u_t = (u2 - u1) / (t2 - t1).clamp_min(1e-6)
            rate = model.dyn(x1, u1, invT)          # >= 0  (per 1000 cycles)
            if cfg.residual == 'euler':
                out['ode'] = (((u2 - u1) + rate * (t2 - t1)) ** 2).mean()
            else:
                out['ode'] = ((u_t + rate) ** 2).mean()
                if cfg.alpha_euler > 0:
                    out['euler'] = (((u2 - u1) + rate * (t2 - t1)) ** 2).mean()
        else:
            u1 = model.u(x1, t1); u2 = model.u(x2, t2)
        # label-free monotonicity with tolerance for regeneration
        out['mono'] = Fnn.relu(u2 - u1 - cfg.eps_mono).mean()
        # plausibility: fresh cell SOH in [0.85, 1.10]; global bounds
        first = b['first'].float().unsqueeze(1)
        lo, hi = cfg.range_first
        rng = Fnn.relu(u1 - 1.15) + Fnn.relu(0.4 - u1) + (Fnn.relu(u1 - hi) + Fnn.relu(lo - u1)) * first
        out['range'] = rng.mean()
        return out

    # ---------------------------------------------------------------- train
    def fit(self, verbose=True) -> Dict:
        cfg = self.cfg
        dyn = {'mlp': 'none', 'pinn_sup': 'greybox', 'pinn_semi': 'greybox', 'pinn_bb': 'blackbox'}[cfg.model]
        model = SOHModel(dyn=dyn, n_feat=self.S_lab.X.shape[1], hidden=cfg.hidden, dropout=cfg.dropout,
                         use_cycle=cfg.use_cycle,
                         use_knee=cfg.use_knee, use_arrhenius=cfg.use_arrhenius,
                         act=cfg.act, arch=cfg.arch).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, foreach=True)   # gộp phép tính, kết quả như cũ
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_steps, eta_min=cfg.lr * 0.05)
        rng = np.random.RandomState(self.train_seed)
        n_lab = len(self.S_lab)
        use_phys = cfg.model != 'mlp'
        nl, nu = self.n_cells['train_labelled'], self.n_cells['train_unlabelled']
        f_lab = nl / max(1, nl + nu)
        w_adapt = ((1.0 - f_lab) + cfg.adapt_floor) if cfg.adapt_w else 1.0
        best, best_state, bad, hist = math.inf, None, 0, []
        t0 = time.time()
        for step in range(1, cfg.max_steps + 1):
            model.train()
            # ----- data loss on labelled points
            idx = rng.randint(0, n_lab, size=min(cfg.batch, n_lab))
            xb, tb, yb = self._gather(self.S_lab, idx)
            loss_data = Fnn.mse_loss(model.u(xb, tb), yb)
            loss = loss_data
            logs = dict(data=loss_data.item())
            # ----- physics losses on (N, N+h) pairs
            if use_phys:
                w = w_adapt * min(1.0, step / max(1, cfg.warmup_frac * cfg.max_steps))
                bl = self._pairs(self.S_lab, rng, cfg.batch // 2 if self.S_unl is not None else cfg.batch)
                if self.S_unl is not None:
                    bu = self._pairs(self.S_unl, rng, cfg.batch // 2)
                    bl = {k: torch.cat([bl[k], bu[k]], 0) for k in bl}
                ph = self.physics(model, bl)
                loss = loss + w * (cfg.alpha_ode * ph.get('ode', torch.zeros(())) + cfg.beta_mono * ph['mono']
                                   + cfg.gamma_range * ph['range'])
                if 'euler' in ph:
                    loss = loss + w * cfg.alpha_euler * ph['euler']
                logs.update(ode=float(ph['ode'].detach()) if 'ode' in ph else 0.0,
                            mono=float(ph['mono'].detach()), range=float(ph['range'].detach()))
            if cfg.distill_w > 0 and getattr(self, 'teacher_unl', None) is not None:
                iu = rng.randint(0, len(self.S_unl), size=cfg.batch // 2)
                xu, tu, _ = self._gather(self.S_unl, iu)
                loss_kd = Fnn.mse_loss(model.u(xu, tu), self.teacher_unl.index_select(0, torch.from_numpy(iu).to(self.dev)))
                loss = loss + cfg.distill_w * loss_kd
                logs['kd'] = loss_kd.item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=True)
            opt.step(); sched.step()
            if step % cfg.eval_every == 0:
                vmae = self.val_mae(model, self.val)
                hist.append(dict(step=step, val_MAE=vmae, **logs))
                if vmae < best - 1e-6:
                    best, bad = vmae, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if verbose and step % (cfg.eval_every * 5) == 0:
                    print(f'  step {step:5d} val MAE {vmae:.4f} best {best:.4f} ' +
                          ' '.join(f'{k}={v:.2e}' for k, v in logs.items()), flush=True)
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        self.model = model
        res = dict(config=asdict(cfg), n_cells=self.n_cells, steps=step, train_time_s=time.time() - t0,
                   n_params=model.n_params(), val=self.evaluate(model, self.val),
                   test=self.evaluate(model, self.test), history=hist, split=self.split)
        res['w_adapt'] = w_adapt; res['f_lab'] = f_lab
        if dyn == 'greybox':
            res['physics_params'] = dict(lam=model.dyn.lam.item(), Ea_kJ_mol=model.dyn.Ea.item() / 1e3)
        for tgt, pts in self.transfer.items():
            res[f'transfer_{tgt}'] = self.evaluate(model, pts)
        if getattr(self, 'teacher_info', None):
            res['teacher'] = self.teacher_info
        if cfg.save_val_predictions:
            res['val_predictions'] = self.predict(model, self.val)
            res['val_predictions']['cell_ids'] = self.val_ids
        if cfg.save_predictions:
            res['test_predictions'] = self.predict(model, self.test)
            res['test_predictions']['cell_ids'] = getattr(self, 'test_ids', None)
        return res

    def save_checkpoint(self, path: str):
        """Trọng số mạng + thống kê normaliser + cấu hình: đủ để dự đoán SOH cho một cell mới."""
        torch.save(dict(state=self.model.state_dict(), mu=self.norm.mu, sd=self.norm.sd,
                        norm_mode=self.norm.mode, config=asdict(self.cfg)), path)

    @torch.no_grad()
    def _forward(self, model, S: PointSet) -> np.ndarray:
        model.eval()
        tt = self._t(S)
        return model.u(tt.X, tt.T).cpu().numpy()

    def val_mae(self, model, S: PointSet) -> float:
        """Chỉ MAE gộp — đủ cho dừng sớm, và tính đúng như metrics.evaluate (float64, bỏ nhãn không hữu
        hạn) nên quyết định dừng không đổi; bỏ phần tính chỉ số theo cell/EOL vốn chỉ cần ở cuối."""
        y = np.asarray(S.Y, dtype=float).ravel(); p = np.asarray(self._forward(model, S), dtype=float).ravel()
        keep = np.isfinite(y)
        if not keep.all():
            y, p = y[keep], p[keep]
        return float(np.abs(p - y).mean())

    @torch.no_grad()
    def predict(self, model, S: PointSet):
        p = self._forward(model, S)
        out = dict(y=S.Y.ravel().tolist(), p=p.ravel().tolist(), cell=S.cell.tolist())
        if S.cyc is not None:
            out['cycle'] = S.cyc.tolist()
        return out

    @torch.no_grad()
    def evaluate(self, model, S: PointSet) -> Dict[str, float]:
        p = self._forward(model, S)
        return metrics(S.Y, p, S.cell, S.cyc if self.cfg.clean != 'life3sigma' else None)


def run_name(cfg: Config) -> str:
    extra = '' if (cfg.act == 'silu' and cfg.arch == 'mlp') else f'_{cfg.act}-{cfg.arch}'
    if cfg.split == 'cv':
        extra += f'_cv{cfg.repeat}-{cfg.fold}of{cfg.n_folds}'
    if cfg.member:
        extra += f'_m{cfg.member}'
    return f'{cfg.tag + "_" if cfg.tag else ""}{cfg.dataset}_{cfg.model}_f{cfg.label_frac}_s{cfg.seed}_{cfg.norm}_{cfg.cyc}{extra}'


def write_json(res: Dict, path: str):
    """Ghi nguyên tử (file tạm rồi đổi tên): bị ngắt giữa chừng không để lại file dở dang mà lần chạy sau
    tưởng là đã xong."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f)
    os.replace(tmp, path)


def run(cfg: Config, verbose=True, skip_existing=True, log=True) -> Dict:
    path = os.path.join(cfg.out_dir, run_name(cfg) + '.json')
    if skip_existing and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    tr = Trainer(cfg).prepare()
    res = tr.fit(verbose=verbose)
    if cfg.save_model:
        os.makedirs(cfg.out_dir, exist_ok=True)
        tr.save_checkpoint(path[:-5] + '.pt')
    write_json(res, path)
    if not log:
        return res
    t = res['test']
    extra = ' '.join(f'{k[9:]}:MAE={v["MAE"]:.4f}' for k, v in res.items() if k.startswith('transfer_'))
    print(f'[{run_name(cfg)}] test MAE {t["MAE"]:.4f} RMSE {t["RMSE"]:.4f} MAPE {t["MAPE"]:.2f}% '
          f'(val {res["val"]["MAE"]:.4f}, steps {res["steps"]}, {res["train_time_s"]:.0f}s) {extra}', flush=True)
    return res
