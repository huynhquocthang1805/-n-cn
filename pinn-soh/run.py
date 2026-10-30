"""
Một lệnh cho mọi bước. Chạy từ thư mục nào cũng được.

    python run.py check                       kiểm tra Python, thư viện, dữ liệu, XeLaTeX
    python run.py data                        tải dữ liệu (nếu chưa có) và dựng bộ nhớ đệm
    python run.py bench                       đo tốc độ máy này để ước lượng thời gian chạy
    python run.py E13 [tuỳ chọn]              một thí nghiệm (danh sách: pinnsoh/experiments.py)
    python run.py experiments [tuỳ chọn]      mọi thí nghiệm, theo thứ tự
    python run.py analyze                     bảng + hình từ runs/ -> results/ (lần đầu thêm ~40 phút CPU để
                                              huấn luyện lại 120 lượt cho phân tích phần vật lý)
    python run.py benchmark [--workers N]     chạy lại mã gốc PINN4SOH trên cách chia của bài (benchmark/) rồi lập bảng
                                              so sánh với số công bố và với E27 -> results/benchmark_*
    python run.py paper                       dựng bản thảo bài báo paper/main.pdf
    python run.py all [tuỳ chọn]              experiments -> analyze -> benchmark -> paper
    python run.py reproduce                   chạy lại 20 lượt mẫu, so từng bit với kết quả công bố
    python run.py demo train|predict ...      demo ước lượng SOH cho một cell (xem tools/demo.py)

Tuỳ chọn cho thí nghiệm:
    --workers N     số tiến trình song song (mặc định: số lõi CPU), mỗi tiến trình 1 luồng
    --quick         chạy thử nhanh: 1 lần lặp kiểm định chéo thay vì 5, 1 hạt giống thay vì 5 (~1/5 thời gian);
                    kết quả là TẬP CON của bản đầy đủ, chạy tiếp bản đầy đủ sau sẽ không phải chạy lại
    --datasets ...  chỉ chạy một số bộ, ví dụ --datasets XJTU HUST
    --dry-run       chỉ liệt kê số lượt còn thiếu và thời gian ước lượng, không chạy
    --device cuda   dùng GPU (nhanh hơn với 1 tiến trình; kết quả lệch ở mức làm tròn, không tái lập từng bit)

Dừng bất cứ lúc nào bằng Ctrl+C; chạy lại đúng lệnh đó sẽ tiếp tục từ lượt đang dở.
Báo cáo đồ án, slide và script giai đoạn đầu nằm ở archive/ (xem archive/README.md).
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ANALYSIS = [['analysis/analyze_e13.py'], ['analysis/analyze_predictions.py'], ['analysis/verify_pipeline.py', '--csv'],
            ['analysis/analyze_thesis.py'], ['analysis/analyze_metrics.py'], ['analysis/analyze_physics.py'],
            ['analysis/error_decomp.py'], ['analysis/dev_e20.py'], ['analysis/analyze_e20.py'],
            ['analysis/analyze_tuning.py'], ['analysis/analyze_e21.py'], ['analysis/analyze_e22.py'],
            ['analysis/analyze_e23.py'], ['analysis/analyze_e25.py'],
            ['analysis/analyze_e26.py'], ['analysis/analyze_e27.py'], ['analysis/analyze_e28.py'],
            ['analysis/analyze_mono.py'], ['analysis/dev_e29.py'], ['analysis/analyze_e29.py'],
            ['analysis/dev_e30.py'], ['analysis/analyze_e30.py'], ['analysis/analyze_e32.py'],
            ['analysis/analyze_overview.py'], ['analysis/figures_report.py'], ['analysis/diagrams.py']]


def sh(args):
    print(f'\n$ python {" ".join(args)}', flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    print(f'  ({time.time() - t:.0f} s, mã thoát {r.returncode})', flush=True)
    if r.returncode:
        sys.exit(r.returncode)


def cmd_check(_):
    import importlib
    print(f'Python {sys.version.split()[0]}', '(cần ≥ 3.10)' if sys.version_info < (3, 10) else '')
    for mod in ['numpy', 'pandas', 'scipy', 'torch', 'sklearn', 'matplotlib']:
        try:
            print(f'  {mod:<11s} {importlib.import_module(mod).__version__}')
        except ImportError:
            print(f'  {mod:<11s} THIẾU — pip install -r requirements.txt')
    from pinnsoh.data import resolve_root
    from tools.tai_du_lieu import count_csv, EXPECT
    have = count_csv(resolve_root())
    print(f'dữ liệu {resolve_root()}: {have}', 'đủ' if have == EXPECT else '— THIẾU, chạy: python run.py data')
    print('XeLaTeX:', shutil.which('xelatex') or 'không có (chỉ cần cho bước paper)')
    print(f'số lõi CPU: {os.cpu_count()}')


def cmd_data(_):
    sh(['tools/tai_du_lieu.py'])


def cmd_bench(_):
    from pinnsoh.runner import bench
    bench()


def experiment_jobs(names, a):
    from pinnsoh import experiments as E
    kw = dict(datasets=a.datasets)
    if a.quick:
        kw.update(repeats=1, seeds=(0,))
    return [j for n in names for j in E.EXPERIMENTS[n](**kw)]


def cmd_experiments(a, names=None):
    from pinnsoh import experiments as E
    from pinnsoh.runner import run_jobs
    if a.device:
        os.environ['PINNSOH_DEVICE'] = a.device
    names = names or E.ORDER
    n_err = 0
    for n in names:
        print(f'\n══ {n} ══', flush=True)
        n_err += run_jobs(experiment_jobs([n], a), workers=a.workers, dry_run=a.dry_run)
    if n_err:
        sys.exit(f'{n_err} lượt lỗi — xem nhật ký phía trên')


def cmd_analyze(_):
    for args in ANALYSIS:
        sh(args)


def benchmark_python():
    """Trình thông dịch cho mã gốc PINN4SOH: cần pandas < 3 (benchmark/rerun_pinn4soh.py); nếu môi trường hiện tại có
    pandas 3 thì dựng .cache/venv-pinn4soh (dùng chung các gói đã cài, chỉ thêm pandas 2.3.3)."""
    import pandas
    if int(pandas.__version__.split('.')[0]) < 3:
        return sys.executable
    venv = os.path.join(ROOT, '.cache', 'venv-pinn4soh')
    py = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(py):
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', venv], check=True)
        subprocess.run([py, '-m', 'pip', 'install', '-q', 'pandas==2.3.3'], check=True)
    return py


def cmd_benchmark(a):
    py = benchmark_python()
    workers = ['--workers', str(a.workers)] if a.workers else []
    for version in ('published', 'current'):
        args = [py, 'benchmark/rerun_pinn4soh.py', '--version', version, *workers]
        print(f'\n$ {" ".join(args[1:])}', flush=True)
        if subprocess.run(args, cwd=ROOT).returncode:
            sys.exit(1)
    sh(['benchmark/compare.py'])


def cmd_paper(_):
    sh(['paper/build.py'])


def cmd_all(a):
    cmd_experiments(a)
    if a.dry_run:
        return
    cmd_analyze(a)
    cmd_benchmark(a)
    cmd_paper(a)


def cmd_reproduce(a):
    sh(['tools/kiem_tai_lap.py'] + (['--workers', str(a.workers)] if a.workers else []))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':          # chuyển nguyên tham số cho tools/demo.py
        sh(['tools/demo.py', *sys.argv[2:]])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from pinnsoh.experiments import EXPERIMENTS
    ap.add_argument('command', choices=['check', 'data', 'bench', 'experiments', 'analyze', 'benchmark', 'paper', 'all',
                                        'reproduce', *EXPERIMENTS])
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--datasets', nargs='+', default=['XJTU', 'TJU', 'MIT', 'HUST'],
                    choices=['XJTU', 'TJU', 'MIT', 'HUST'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    if a.command[0] == 'E' and a.command[1:3].isdigit():
        cmd_experiments(a, [a.command])
    else:
        globals()[f'cmd_{a.command}'](a)


if __name__ == '__main__':
    main()
