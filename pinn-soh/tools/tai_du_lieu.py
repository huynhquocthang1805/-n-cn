"""
Tải bốn bộ dữ liệu (XJTU, TJU, MIT, HUST — 387 file CSV, ~94 MB) từ repo công khai wang-fujin/PINN4SOH,
đúng commit đã dùng để sinh mọi kết quả, rồi dựng bộ nhớ đệm .cache/.

    python tools/tai_du_lieu.py                    # tự tải (git nếu có, không thì file zip của GitHub)
    python tools/tai_du_lieu.py --zip PINN4SOH.zip # dùng file zip đã tải tay (nút Code → Download ZIP)

Kết quả: PINN4SOH/data/{XJTU,TJU,MIT,HUST} data/... Muốn để dữ liệu ở chỗ khác thì đặt biến môi trường
PINNSOH_DATA=<thư mục chứa bốn thư mục "... data">.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import argparse, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile

REPO = 'https://github.com/wang-fujin/PINN4SOH'
COMMIT = 'cc3cc5053caf38f04e0665f7f88cb109144d035e'
EXPECT = {'XJTU': 55, 'TJU': 130, 'MIT': 125, 'HUST': 77}


def count_csv(root):
    out = {}
    for ds in EXPECT:
        d = os.path.join(root, f'{ds} data')
        out[ds] = sum(f.endswith('.csv') for _, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
    return out


def via_git(tmp):
    if shutil.which('git') is None:
        raise RuntimeError('không có git')
    repo = os.path.join(tmp, 'PINN4SOH')
    subprocess.run(['git', 'clone', '--quiet', '--no-checkout', REPO + '.git', repo], check=True)
    subprocess.run(['git', '-C', repo, 'checkout', '--quiet', COMMIT, '--', 'data'], check=True)
    return os.path.join(repo, 'data')


def via_zip(tmp, zip_path=None):
    if zip_path is None:
        zip_path = os.path.join(tmp, 'PINN4SOH.zip')
        url = f'{REPO}/archive/{COMMIT}.zip'
        print(f'tải {url} ...', flush=True)
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        top = z.namelist()[0].split('/')[0]                  # PINN4SOH-<commit> hoặc PINN4SOH-main
        members = [m for m in z.namelist() if m.startswith(f'{top}/data/')]
        if not members:
            raise RuntimeError(f'{zip_path} không có thư mục data/')
        z.extractall(tmp, members)
    return os.path.join(tmp, top, 'data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', help='file zip của repo PINN4SOH đã tải sẵn')
    ap.add_argument('--dest', default='PINN4SOH/data')
    a = ap.parse_args()
    from pinnsoh.data import load_dataset, resolve_root
    dest = resolve_root(a.dest) if a.dest == 'PINN4SOH/data' else os.path.abspath(a.dest)
    have = count_csv(dest)
    if have != EXPECT:
        with tempfile.TemporaryDirectory() as tmp:
            src = None
            for how in ([lambda: via_zip(tmp, a.zip)] if a.zip else [lambda: via_git(tmp), lambda: via_zip(tmp)]):
                try:
                    src = how(); break
                except Exception as e:
                    print(f'  chưa được ({e}); thử cách khác', flush=True)
            if src is None:
                sys.exit(f'Không tải được. Tải tay {REPO} (Code → Download ZIP) rồi chạy: '
                         f'python tools/tai_du_lieu.py --zip <file.zip>')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.islink(dest):
                os.unlink(dest)
            elif os.path.isdir(dest):
                shutil.rmtree(dest)
            shutil.move(src, dest)
        have = count_csv(dest)
    print('số file CSV:', have, '(đủ)' if have == EXPECT else f'(THIẾU, cần {EXPECT})')
    if have != EXPECT:
        sys.exit(1)
    print('dựng bộ nhớ đệm .cache/ ...', flush=True)
    for ds in EXPECT:
        for clean in ('causal', 'life3sigma'):
            load_dataset(dest, ds, clean=clean)
    print('xong:', dest)


if __name__ == '__main__':
    main()
