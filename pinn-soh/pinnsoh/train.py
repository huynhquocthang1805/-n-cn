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
"""
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.autograd import grad

from .data import (Normalizer, PointSet, load_dataset, split_cells, choose_labelled, make_pointset)
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


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


def metrics(y: np.ndarray, p: np.ndarray, cell_idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    return eval_metrics(y, p, cell_idx)


class Trainer:
    def __init__(self, cfg: Config, device='cpu'):
        self.cfg, self.dev = cfg, device
        set_seed(cfg.seed)

    # ---------------------------------------------------------------- data
    def prepare(self):
        cfg = self.cfg
        cells = load_dataset(cfg.root, cfg.dataset)
        by_id = {c.cid: c for c in cells}
        split = split_cells(cells, cfg.seed)
        lab_ids, unl_ids = choose_labelled(split['train'], cfg.label_frac, len(cells), cfg.seed, by_id)
        self.split = dict(split, labelled=lab_ids, unlabelled=unl_ids)
        tr_lab = [by_id[i] for i in lab_ids]; tr_unl = [by_id[i] for i in unl_ids]
        va = [by_id[i] for i in split['val']]; te = [by_id[i] for i in split['test']]
        semi = cfg.model in ('pinn_semi', 'pinn_bb')
        # Normaliser: fitted on features only (labels never used). Semi-supervised models may use all
        # training cells' features; purely supervised models only see labelled cells.
        self.norm = Normalizer(cfg.norm).fit(tr_lab + tr_unl if semi else tr_lab)
        self.S_lab = make_pointset(tr_lab, self.norm, cfg.cyc)
        self.S_unl = make_pointset(tr_unl, self.norm, cfg.cyc) if (semi and tr_unl) else None
        self.val = make_pointset(va, self.norm, cfg.cyc)
        self.test = make_pointset(te, self.norm, cfg.cyc)
        self.n_cells = dict(train_labelled=len(tr_lab), train_unlabelled=len(tr_unl), val=len(va), test=len(te),
                            all=len(cells))
        self.transfer = {}
        for tgt in cfg.transfer_to:
            tcells = load_dataset(cfg.root, tgt)
            if cfg.transfer_test_only:
                keep = set(split_cells(tcells, cfg.seed)['test'])
                tcells = [c for c in tcells if c.cid in keep]
            self.transfer[tgt] = make_pointset(tcells, self.norm, cfg.cyc)
        return self

    def _pairs(self, S: PointSet, rng, batch):
        i, j = S.sample_pairs(rng, batch, self.cfg.h_min, self.cfg.h_max)
        f = lambda a, idx: torch.from_numpy(a[idx]).to(self.dev)
        return dict(x1=f(S.X, i), x2=f(S.X, j), t1=f(S.T, i), t2=f(S.T, j), y1=f(S.Y, i), y2=f(S.Y, j),
                    invT=f(S.invT, i), first=torch.from_numpy(i == S.start[i]).to(self.dev))

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
        else:
            u1 = model.u(x1, t1); u2 = model.u(x2, t2)
        # label-free monotonicity with tolerance for regeneration
        out['mono'] = Fnn.relu(u2 - u1 - cfg.eps_mono).mean()
        # plausibility: fresh cell SOH in [0.85, 1.10]; global bounds
        first = b['first'].float().unsqueeze(1)
        rng = Fnn.relu(u1 - 1.15) + Fnn.relu(0.4 - u1) + (Fnn.relu(u1 - 1.10) + Fnn.relu(0.85 - u1)) * first
        out['range'] = rng.mean()
        return out

    # ---------------------------------------------------------------- train
    def fit(self, verbose=True) -> Dict:
        cfg = self.cfg
        dyn = {'mlp': 'none', 'pinn_sup': 'greybox', 'pinn_semi': 'greybox', 'pinn_bb': 'blackbox'}[cfg.model]
        model = SOHModel(dyn=dyn, hidden=cfg.hidden, dropout=cfg.dropout, use_cycle=cfg.use_cycle,
                         use_knee=cfg.use_knee, use_arrhenius=cfg.use_arrhenius,
                         act=cfg.act, arch=cfg.arch).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_steps, eta_min=cfg.lr * 0.05)
        rng = np.random.RandomState(cfg.seed)
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
            xb = torch.from_numpy(self.S_lab.X[idx]).to(self.dev); tb = torch.from_numpy(self.S_lab.T[idx]).to(self.dev)
            yb = torch.from_numpy(self.S_lab.Y[idx]).to(self.dev)
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
                logs.update(ode=float(ph['ode'].detach()) if 'ode' in ph else 0.0,
                            mono=float(ph['mono'].detach()), range=float(ph['range'].detach()))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if step % cfg.eval_every == 0:
                vm = self.evaluate(model, self.val)
                hist.append(dict(step=step, val_MAE=vm['MAE'], **logs))
                if vm['MAE'] < best - 1e-6:
                    best, bad = vm['MAE'], 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if verbose and step % (cfg.eval_every * 5) == 0:
                    print(f'  step {step:5d} val MAE {vm["MAE"]:.4f} best {best:.4f} ' +
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
            res['physics_params'] = dict(lam=float(model.dyn.lam), Ea_kJ_mol=float(model.dyn.Ea) / 1e3)
        for tgt, pts in self.transfer.items():
            res[f'transfer_{tgt}'] = self.evaluate(model, pts)
        if cfg.save_predictions:
            res['test_predictions'] = self.predict(model, self.test)
        return res

    @torch.no_grad()
    def predict(self, model, S: PointSet):
        model.eval()
        p = model.u(torch.from_numpy(S.X).to(self.dev), torch.from_numpy(S.T).to(self.dev)).cpu().numpy()
        return dict(y=S.Y.ravel().tolist(), p=p.ravel().tolist(), cell=S.cell.tolist())

    @torch.no_grad()
    def evaluate(self, model, S: PointSet) -> Dict[str, float]:
        model.eval()
        p = model.u(torch.from_numpy(S.X).to(self.dev), torch.from_numpy(S.T).to(self.dev)).cpu().numpy()
        return metrics(S.Y, p, S.cell)


def run_name(cfg: Config) -> str:
    extra = '' if (cfg.act == 'silu' and cfg.arch == 'mlp') else f'_{cfg.act}-{cfg.arch}'
    return f'{cfg.tag + "_" if cfg.tag else ""}{cfg.dataset}_{cfg.model}_f{cfg.label_frac}_s{cfg.seed}_{cfg.norm}_{cfg.cyc}{extra}'


def run(cfg: Config, verbose=True, skip_existing=True) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, run_name(cfg) + '.json')
    if skip_existing and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    tr = Trainer(cfg).prepare()
    res = tr.fit(verbose=verbose)
    with open(path, 'w') as f:
        json.dump(res, f)
    t = res['test']
    extra = ' '.join(f'{k[9:]}:MAE={v["MAE"]:.4f}' for k, v in res.items() if k.startswith('transfer_'))
    print(f'[{run_name(cfg)}] test MAE {t["MAE"]:.4f} RMSE {t["RMSE"]:.4f} MAPE {t["MAPE"]:.2f}% '
          f'(val {res["val"]["MAE"]:.4f}, steps {res["steps"]}, {res["train_time_s"]:.0f}s) {extra}', flush=True)
    return res
