"""
Định nghĩa các thí nghiệm của báo cáo đồ án (pipeline v2) dưới dạng danh sách lượt chạy (Job).

    E13  kiểm định chéo 5 fold × 5 lần lặp theo cell: MLP, PINN-semi ở 4 mức nhãn; PINN-sup, 2 biến thể
         cắt bỏ ở 30 % nhãn (đề cương ấn định trước: tai-lieu/de-cuong-E13.md)          1 100 lượt
    E14  chuyển miền, 4 cặp cùng hoá học, MLP và PINN-semi cùng normaliser (tập con của E18)  80
    E15  kiểm toán rò rỉ: 4 cách chia/chuẩn hoá sai, MLP 70 % nhãn, 2 lần lặp                 160
    E16  Ridge, GPR, GBDT, cùng cách chia và cùng tập nhãn với E13                           1 200
    E17  PINN-sup ở 10 và 50 %, PINN hộp đen ở 4 mức nhãn, cùng giao thức E13                600
    E18  chuyển miền đủ 12 cặp nguồn–đích, MLP và PINN-semi, k ∈ {0, 3}, 5 hạt giống          240
    E19  GBDT trên đúng 12 cặp của E18                                                       120
    E20  PINN-lai: 3 PINN-semi + GBDT + bộ lọc quỹ đạo, 3 lần lặp (tai-lieu/de-cuong-E20.md)  1 440
    E21  huấn luyện một bộ, kiểm tra bộ khác (2 giao thức) + thích nghi, tự huấn luyện       168
    E22  dự báo SOH tương lai, dung lượng còn lại, RUL; cùng fold E20, dùng ước lượng E20      60

Mỗi Job biết file kết quả của nó (`out`), nên chạy lại sẽ bỏ qua lượt đã xong; tên file giữ nguyên như các
lần chạy trước để kết quả cũ dùng lại được. Cấu hình của E13 đã ấn định trước — KHÔNG sửa.
"""
from __future__ import annotations
import itertools, os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
SAME_CHEM = {'HUST': 'MIT', 'MIT': 'HUST', 'XJTU': 'TJU', 'TJU': 'XJTU'}

E13_NET = dict(hidden=(128, 128, 64))
E13_PHYS = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)
E13_FOLDS = 5
E13_ARMS = [   # (nhánh, mô hình, mức nhãn, cấu hình riêng)
    ('A', 'mlp', 0.3, {}),
    ('C', 'pinn_semi', 0.3, E13_PHYS),
    ('F', 'mlp', 0.7, {}),
    ('B', 'pinn_sup', 0.3, E13_PHYS),
    ('D', 'pinn_semi', 0.3, dict(E13_PHYS, alpha_ode=0.0, gamma_range=0.0)),       # chỉ đơn điệu
    ('E', 'pinn_semi', 0.3, dict(beta_mono=0.5, residual='euler', alpha_ode=2.0)),  # phần dư Euler
    ('G', 'pinn_semi', 0.7, E13_PHYS),
    ('L', 'mlp', 0.1, {}), ('L', 'pinn_semi', 0.1, E13_PHYS),
    ('L', 'mlp', 0.5, {}), ('L', 'pinn_semi', 0.5, E13_PHYS),
]
# ở 70 % nhãn mọi cell train đều có nhãn nên PINN-sup trùng từng bit với PINN-semi (nhánh G) — không chạy lại
E17_ARMS = [('B', 'pinn_sup', 0.1, E13_PHYS), ('B', 'pinn_sup', 0.5, E13_PHYS),
            ('H', 'pinn_bb', 0.1, E13_PHYS), ('H', 'pinn_bb', 0.3, E13_PHYS),
            ('H', 'pinn_bb', 0.5, E13_PHYS), ('H', 'pinn_bb', 0.7, E13_PHYS)]

# số bước trung bình tới lúc dừng sớm (đo trên E13/E17) × mili giây mỗi bước trên CPU 1 luồng của máy
# chạy thử -> chi phí tương đối để xếp lịch (lượt dài chạy trước) và ước lượng thời gian
MEAN_STEPS = {'mlp': 3600, 'pinn_sup': 3300, 'pinn_semi': 3330, 'pinn_bb': 3380}
MS_PER_STEP = {'mlp': 2.6, 'pinn_sup': 8.2, 'pinn_semi': 8.2, 'pinn_bb': 12.3}
BASELINE_S = {('ridge', 'XJTU'): 0.2, ('ridge', 'TJU'): 0.3, ('ridge', 'MIT'): 0.4, ('ridge', 'HUST'): 0.8,
              ('gbdt', 'XJTU'): 6, ('gbdt', 'TJU'): 11, ('gbdt', 'MIT'): 18, ('gbdt', 'HUST'): 20,
              ('gpr', 'XJTU'): 12, ('gpr', 'TJU'): 8, ('gpr', 'MIT'): 8, ('gpr', 'HUST'): 12}
TRANSFER_STEPS = 2500


@dataclass
class Job:
    kind: str                     # train | transfer | leak | baseline | baseline_transfer
    args: Dict
    out: str                      # file kết quả; có rồi thì bỏ qua
    cost: float                   # giây ước lượng trên CPU 1 luồng của máy chạy thử
    label: str = ''
    datasets: tuple = field(default_factory=tuple)

    def done(self) -> bool:
        return os.path.exists(self.out)


def _train_cost(model):
    return MEAN_STEPS[model] * MS_PER_STEP[model] / 1000 + 2.0


def _train_job(tag, model, frac, extra, r, ds, k, out):
    from .train import Config, run_name
    kw = dict(dataset=ds, label_frac=frac, model=model, seed=100 * r + k, split='cv', fold=k, repeat=r,
              n_folds=E13_FOLDS, tag=tag, out_dir=out, **E13_NET, **extra)
    return Job('train', kw, os.path.join(out, run_name(Config(**kw)) + '.json'), _train_cost(model),
               f'{tag} {ds:<4s} {model:<9s} f{frac} cv{r}-{k}', (ds,))


def E13(repeats=5, datasets=DATASETS, seeds=None, out='runs/E13') -> List[Job]:
    """seed = 100·lần_lặp + fold, giống nhau cho mọi nhánh -> so sánh ghép cặp theo cell."""
    return [_train_job(tag, m, f, extra, r, ds, k, out)
            for (tag, m, f, extra), r, ds, k in itertools.product(E13_ARMS, range(repeats), datasets, range(E13_FOLDS))]


def E17(repeats=5, datasets=DATASETS, seeds=None, out='runs/E17') -> List[Job]:
    return [_train_job(tag, m, f, extra, r, ds, k, out)
            for (tag, m, f, extra), r, ds, k in itertools.product(E17_ARMS, range(repeats), datasets, range(E13_FOLDS))]


def _transfer_job(src, tgt, model, k, s, out):
    kw = dict(dataset=src, target=tgt, model=model, k_target=k, seed=s, norm='global', norm_pool='train',
              tag='samenorm', out_dir=out, save_predictions=False)
    cost = TRANSFER_STEPS * MS_PER_STEP[model] / 1000 + 2.0
    return Job('transfer', kw, f'{out}/samenorm_{src}_to_{tgt}_{model}_k{k}_s{s}_global.json', cost,
               f'{src}->{tgt} {model} k{k} s{s}', (src, tgt))


def E18(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14') -> List[Job]:
    return [_transfer_job(src, tgt, m, k, s, out)
            for src, tgt, s, k, m in itertools.product(datasets, DATASETS, seeds, [0, 3], ['mlp', 'pinn_semi'])
            if src != tgt]


def E14(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14') -> List[Job]:
    """Bốn cặp cùng hoá học — tập con của E18, giữ lại để tái lập bảng E14 cũ."""
    return [_transfer_job(src, SAME_CHEM[src], m, k, s, out)
            for src, s, k, m in itertools.product(datasets, seeds, [0, 3], ['mlp', 'pinn_semi'])]


def E15(repeats=2, datasets=DATASETS, seeds=None, out='runs/E15') -> List[Job]:
    from .leakage import SETTINGS, leak_path
    return [Job('leak', dict(setting=st, ds=ds, r=r, k=k, out_dir=out), leak_path(st, ds, r, k, out),
                _train_cost('mlp'), f'{st} {ds} cv{r}-{k}', (ds,))
            for st, ds, r, k in itertools.product(SETTINGS, datasets, range(repeats), range(E13_FOLDS))]


def E16(repeats=5, datasets=DATASETS, seeds=None, out='runs/E16', models=('ridge', 'gbdt', 'gpr')) -> List[Job]:
    from .baselines import cv_path
    return [Job('baseline', dict(model=m, ds=ds, frac=f, r=r, k=k, out_dir=out), cv_path(m, ds, f, r, k, out),
                BASELINE_S[(m, ds)], f'{m} {ds} f{f} cv{r}-{k}', (ds,))
            for m, ds, f, r, k in itertools.product(models, datasets, FRACS, range(repeats), range(E13_FOLDS))]


def E19(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14', models=('gbdt',)) -> List[Job]:
    from .baselines import transfer_path
    return [Job('baseline_transfer', dict(model=m, src=src, tgt=tgt, k=k, seed=s, out_dir=out),
                transfer_path(m, src, tgt, k, s, out), 5 * BASELINE_S[(m, src)], f'{m} {src}->{tgt} k{k} s{s}',
                (src, tgt))
            for m, src, tgt, k, s in itertools.product(models, datasets, DATASETS, [0, 3], seeds) if src != tgt]


# ---------------------------------------------------------------- E20: cải tiến mô hình
# Giai đoạn phát triển: chỉ lần lặp 0, mức nhãn 10 % và 50 %, mọi quyết định dựa trên MAE theo cell của
# tập VALIDATION (dự đoán validation được lưu kèm). Tập test của các lượt này không được dùng để chọn.
E20_DEV_FRACS = [0.1, 0.5]
E20_DEV_ARMS = [   # (nhãn, mô hình, cấu hình riêng, các thành viên)
    ('P', 'pinn_semi', E13_PHYS, (0, 1, 2)),
    ('KD03', 'pinn_semi', dict(E13_PHYS, distill_w=0.3), (0,)),
    ('KD1', 'pinn_semi', dict(E13_PHYS, distill_w=1.0), (0,)),
    ('EU', 'pinn_semi', dict(E13_PHYS, alpha_euler=2.0), (0,)),
    ('M2', 'pinn_semi', dict(E13_PHYS, beta_mono=2.0), (0,)),
    ('REL', 'pinn_semi', dict(E13_PHYS, norm='both'), (0,)),
]


def _member_job(tag, model, frac, extra, r, ds, k, out, member):
    j = _train_job(tag, model, frac, dict(extra, member=member, save_val_predictions=True), r, ds, k, out)
    j.label += f' m{member}'
    return j


def E20dev(repeats=1, datasets=DATASETS, seeds=None, out='runs/E20dev') -> List[Job]:
    from .baselines import cv_path
    jobs = [_member_job(tag, m, f, extra, r, ds, k, out, mem)
            for (tag, m, extra, mems), f, r, ds, k in itertools.product(E20_DEV_ARMS, E20_DEV_FRACS, range(repeats),
                                                                          datasets, range(E13_FOLDS))
            for mem in mems]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True),
                 cv_path('gbdt', ds, f, r, k, out), BASELINE_S[('gbdt', ds)], f'gbdt {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(E20_DEV_FRACS, range(repeats), datasets, range(E13_FOLDS))]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True, norm='both'),
                 cv_path('gbdt', ds, f, r, k, out, 'both'), BASELINE_S[('gbdt', ds)], f'gbdt-both {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(E20_DEV_FRACS, range(repeats), datasets, range(E13_FOLDS))]
    return jobs


# Đánh giá cuối (chốt sau giai đoạn phát triển, xem tai-lieu/de-cuong-E20.md): 3 lần lặp × 5 fold × 4 mức nhãn.
#   P  PINN-semi E13, thành viên 0–2, lưu dự đoán validation (thành viên 0 trùng từng bit với E13)
#   M  MLP E13, thành viên 1–2 (thành viên 0 lấy từ E13) — nhánh đối chứng không vật lý của mô hình lai
#   gbdt  GBDT E16, lưu dự đoán validation (dự đoán test trùng E16)
E20_REPEATS = 3
E20_MEMBERS = (0, 1, 2)


def E20(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E20') -> List[Job]:
    from .baselines import cv_path
    rr = range(repeats)
    jobs = [_member_job('P', 'pinn_semi', f, E13_PHYS, r, ds, k, out, mem)
            for f, r, ds, k, mem in itertools.product(FRACS, rr, datasets, range(E13_FOLDS), E20_MEMBERS)]
    jobs += [_member_job('M', 'mlp', f, {}, r, ds, k, out, mem)
             for f, r, ds, k, mem in itertools.product(FRACS, rr, datasets, range(E13_FOLDS), E20_MEMBERS[1:])]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True),
                 cv_path('gbdt', ds, f, r, k, out), BASELINE_S[('gbdt', ds)], f'gbdt {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(FRACS, rr, datasets, range(E13_FOLDS))]
    return jobs


# E21: huấn luyện trên một bộ, kiểm tra trên các bộ khác (pinnsoh/crossds.py)
#   base   2 giao thức (70/30 và 70/15/15) × 4 bộ nguồn × 3 hạt giống × {MLP, GBDT, 3 PINN}
#   chain  giao thức 70/30, 12 cặp nguồn–đích × 2 hạt giống × 2 nửa bộ đích; vòng 0 thích nghi không nhãn,
#          vòng 1–2 tự huấn luyện bằng nhãn giả của mô hình kết hợp
E21_SEEDS = (0, 1, 2)
E21_CHAIN_SEEDS = (0, 1)
E21_ROUNDS = 2


def E21(repeats=None, datasets=DATASETS, seeds=None, out='runs/E21') -> List[Job]:
    from .crossds import base_path, chain_path, PROTOCOLS
    sd = E21_SEEDS if seeds is None else tuple(seeds)
    cs = E21_CHAIN_SEEDS if seeds is None else tuple(seeds)
    jobs = []
    for proto, src, s in itertools.product(PROTOCOLS, datasets, sd):
        for model, mem, cost in [('mlp', 0, 40), ('gbdt', 0, 30)] + [('pinn_semi', m, 90) for m in range(3)]:
            jobs.append(Job('cross_base', dict(proto=proto, src=src, seed=s, model=model, member=mem),
                            base_path(proto, src, s, model, mem), cost, f'{proto} {src} s{s} {model} m{mem}',
                            tuple(DATASETS)))
    for src, tgt, s, h in itertools.product(datasets, DATASETS, cs, (0, 1)):
        if src == tgt:
            continue
        jobs.append(Job('cross_chain', dict(src=src, tgt=tgt, seed=s, half=h, rounds=E21_ROUNDS),
                        chain_path(src, tgt, s, h, E21_ROUNDS, 'gbdt'), (E21_ROUNDS + 1) * 300,
                        f'chain {src}->{tgt} s{s} h{h}', (src, tgt)))
    return jobs


# E22: dự báo SOH tương lai, dung lượng còn lại và RUL (pinnsoh/forecast.py) — cùng fold với E13/E20, lần lặp 0–2,
#      lịch sử ước lượng lấy từ PINN-lai của E20 (mức nhãn 70 %)
E22_COST = {'XJTU': 60, 'TJU': 90, 'MIT': 120, 'HUST': 200}


def E22(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E22') -> List[Job]:
    from .forecast import out_path
    return [Job('forecast', dict(ds=ds, r=r, k=k, root=out), out_path(ds, r, k, out), E22_COST[ds],
                f'forecast {ds} cv{r}-{k}', (ds,))
            for r, ds, k in itertools.product(range(repeats), datasets, range(E13_FOLDS))]


EXPERIMENTS: Dict[str, Callable[..., List[Job]]] = dict(E13=E13, E14=E14, E15=E15, E16=E16, E17=E17, E18=E18, E19=E19,
                                                        E20dev=E20dev, E20=E20, E21=E21, E22=E22)
# thứ tự chạy cho `run.py experiments`: E14 nằm trong E18 nên không lặp lại
ORDER = ['E13', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20dev', 'E20', 'E21', 'E22']
