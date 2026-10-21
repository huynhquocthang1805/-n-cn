"""
Chạy song song một danh sách Job (pinnsoh.experiments), dùng chung cho mọi thí nghiệm.

  * bỏ qua lượt đã có file kết quả -> dừng giữa chừng (Ctrl+C, mất điện) rồi chạy lại là chạy tiếp;
    file kết quả được ghi nguyên tử nên không bao giờ còn lại file dở dang
  * mỗi tiến trình con dùng 1 luồng (torch, OpenMP, MKL): N tiến trình × 1 luồng nhanh hơn 1 tiến trình
    × N luồng với mạng nhỏ, và giữ kết quả giống từng bit với các lần chạy trước
  * lượt dài chạy trước (xếp theo chi phí ước lượng) để các tiến trình kết thúc gần cùng lúc
  * dữ liệu được đọc và lọc một lần ở tiến trình chính rồi lưu bộ đệm .cache/, tiến trình con chỉ nạp lại
  * một lượt lỗi không làm dừng cả loạt: lỗi được ghi lại và báo ở cuối
"""
from __future__ import annotations
import json, os, time, traceback
from collections import Counter
from typing import Callable, List, Optional

from .experiments import Job, MS_PER_STEP

BENCH_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache', 'bench.json')


def _execute(kind: str, args: dict):
    if kind == 'train':
        from .train import Config, run
        return run(Config(**args).v2(), verbose=False, log=False)['test'].get('MAE_cell')
    if kind == 'transfer':
        from .transfer import TransferConfig, run_transfer
        cfg = TransferConfig(**args)
        cfg.clean, cfg.pair_by = 'causal', 'cycle'                      # pipeline v2; norm_pool nằm trong args
        return run_transfer(cfg, log=False)['target_test'].get('MAE_cell')
    if kind == 'leak':
        from .leakage import run_one
        return run_one(**args)
    if kind == 'baseline':
        from .baselines import run_cv
        return run_cv(**args)
    if kind == 'baseline_transfer':
        from .baselines import run_transfer
        return run_transfer(**args)
    if kind == 'cross_base':
        from .crossds import run_base
        return run_base(**args)
    if kind == 'cross_chain':
        from .crossds import run_chain
        return run_chain(**args)
    if kind == 'forecast':
        from .forecast import run_fold
        return run_fold(**args)
    if kind == 'recovery':
        from .recovery import run_fold
        return run_fold(**args)
    if kind == 'baseline_samba':
        from .baselines import run_samba
        return run_samba(**args)
    raise ValueError(kind)


def _safe_execute(kind, args):
    t0 = time.time()
    try:
        return _execute(kind, args), time.time() - t0, None
    except Exception:                                                   # báo lỗi về tiến trình chính
        return None, time.time() - t0, traceback.format_exc()


def _init_worker(threads: int):
    import torch
    torch.set_num_threads(threads)


def _set_thread_env(threads: int):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = str(threads)


def speed_factor() -> float:
    """Tỉ số thời gian máy này / máy chạy thử, đo bằng `python run.py bench`; 1.0 nếu chưa đo."""
    try:
        return float(json.load(open(BENCH_FILE))['factor'])
    except Exception:
        return 1.0


def _fmt(sec: float) -> str:
    return f'{sec / 3600:.1f} giờ' if sec >= 5400 else f'{sec / 60:.0f} phút'


def plan(jobs: List[Job], workers: int, log: Callable = print) -> List[Job]:
    """Bỏ trùng và bỏ lượt đã xong, in tóm tắt và thời gian ước lượng; trả danh sách cần chạy."""
    uniq = {j.out: j for j in jobs}
    todo = [j for j in uniq.values() if not j.done()]
    f = speed_factor()
    cpu = sum(j.cost for j in todo) * f
    log(f'{len(uniq)} lượt, đã có {len(uniq) - len(todo)}, cần chạy {len(todo)}'
        + (f' — ước lượng {_fmt(cpu)} CPU, ~{_fmt(cpu / max(1, workers))} với {workers} tiến trình' if todo else ''))
    if todo:
        kinds = Counter(j.label.split()[0] if j.kind == 'train' else j.kind for j in todo)
        log('  ' + ', '.join(f'{k}: {v}' for k, v in sorted(kinds.items())))
    return sorted(todo, key=lambda j: -j.cost)


def prewarm(jobs: List[Job]):
    """Đọc + lọc dữ liệu một lần ở tiến trình chính, ghi bộ đệm .cache/ cho các tiến trình con."""
    from .data import load_dataset
    for ds in sorted({d for j in jobs for d in j.datasets}):
        load_dataset('PINN4SOH/data', ds, clean='causal')
    if any(j.kind == 'leak' and j.args.get('setting') == 'life3s' for j in jobs):
        for ds in sorted({d for j in jobs for d in j.datasets}):
            load_dataset('PINN4SOH/data', ds, clean='life3sigma')


def run_jobs(jobs: List[Job], workers: Optional[int] = None, threads: int = 1, dry_run: bool = False,
             log: Callable = print) -> int:
    """Chạy các lượt chưa xong; trả số lượt lỗi."""
    workers = max(1, workers or os.cpu_count() or 1)
    todo = plan(jobs, workers, log)
    if dry_run or not todo:
        return 0
    _set_thread_env(threads)
    prewarm(todo)
    t0, failed = time.time(), []

    def report(i, job, mae, dt, err):
        el = time.time() - t0
        if err:
            failed.append((job, err))
            log(f'[{i:5d}/{len(todo)}] LỖI {job.label}: {err.strip().splitlines()[-1]}')
            return
        m = f'MAE_cell {mae:.4f}' if mae is not None else ''
        log(f'[{i:5d}/{len(todo)}] {job.label:<34s} {m}  {dt:5.0f}s  (đã {_fmt(el)}, còn ~{_fmt(el / i * (len(todo) - i))})')

    if workers == 1:
        _init_worker(threads)
        for i, job in enumerate(todo, 1):
            report(i, job, *_safe_execute(job.kind, job.args))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=_init_worker, initargs=(threads,)) as ex:
            futs = {ex.submit(_safe_execute, j.kind, j.args): j for j in todo}
            try:
                for i, fu in enumerate(as_completed(futs), 1):
                    report(i, futs[fu], *fu.result())
            except KeyboardInterrupt:
                log('Dừng theo yêu cầu — các lượt đã xong vẫn được giữ; chạy lại lệnh để tiếp tục.')
                for fu in futs:
                    fu.cancel()
                raise
    log(f'xong sau {_fmt(time.time() - t0)}' + (f' — {len(failed)} lượt lỗi' if failed else ''))
    for job, err in failed[:5]:
        log(f'--- {job.label}\n{err}')
    return len(failed)


def bench(log: Callable = print) -> float:
    """Đo mili giây mỗi bước của MLP và PINN-semi trên máy này (1 luồng), so với máy chạy thử."""
    import torch
    from .train import Config, Trainer
    from .experiments import E13_NET, E13_PHYS
    torch.set_num_threads(1)
    ratios = []
    for model, extra in [('mlp', {}), ('pinn_semi', E13_PHYS)]:
        cfg = Config(dataset='XJTU', label_frac=0.3, model=model, split='cv', max_steps=300, patience=10_000,
                     save_predictions=False, **E13_NET, **extra).v2()
        tr = Trainer(cfg).prepare()
        cfg.max_steps = 30; tr.fit(verbose=False)                         # khởi động
        cfg.max_steps = 300
        t = time.time(); tr.fit(verbose=False); ms = (time.time() - t) / 300 * 1000
        ratios.append(ms / MS_PER_STEP[model])
        log(f'{model:<10s} {ms:5.1f} ms/bước (máy chạy thử: {MS_PER_STEP[model]} ms)')
    factor = sum(ratios) / len(ratios)
    os.makedirs(os.path.dirname(BENCH_FILE), exist_ok=True)
    json.dump(dict(factor=factor, threads=1), open(BENCH_FILE, 'w'))
    log(f'hệ số tốc độ {factor:.2f} (< 1: nhanh hơn máy chạy thử); đã lưu cho các lần ước lượng sau')
    return factor
