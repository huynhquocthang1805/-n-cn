"""Cần bao nhiêu seed để kết luận vững? Ước lượng lực thống kê từ hiệu đã quan sát."""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(7)
d = pd.read_csv('results/stats_perseed_tuned.csv')
NS, REP = [10, 15, 20, 30, 50], 600
print('Lực thống kê (bootstrap từ 10 hiệu đã quan sát, MAE theo cell, Holm ×4)')
print(f'{"bộ":<6}{"hiệu TB":>11}{"SD":>10}   ' + ''.join(f'n={n:<8}' for n in NS))
out = []
for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
    g = d[d.dataset == ds]
    diff = (g.maecell_pinn - g.maecell_mlp).values
    row = []
    for n in NS:
        smp = rng.choice(diff, size=(REP, n), replace=True)
        p = np.array([stats.wilcoxon(s, alternative='two-sided',
                                     zero_method='wilcox', method='approx').pvalue for s in smp])
        row.append(float((np.minimum(1.0, 4 * p) < 0.05).mean()))
    out.append([ds, diff.mean(), diff.std(ddof=1)] + row)
    print(f'{ds:<6}{diff.mean():>+11.5f}{diff.std(ddof=1):>10.5f}   ' + ''.join(f'{v:<10.2f}' for v in row))
pd.DataFrame(out, columns=['bo', 'hieu_tb', 'sd'] + [f'luc_n{n}' for n in NS]).to_csv(
    'results/stats_power.csv', index=False)
print('\nđã ghi results/stats_power.csv')
