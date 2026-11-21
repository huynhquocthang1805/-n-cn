"""
Kiểm tra tính nhân quả của đặc trưng dự báo (ý từ tests/test_improvements.py của repo G-doan): đặc trưng tại hàng i
chỉ được dùng các hàng <= i, nên tính trên cả cell hay trên một đoạn đầu của cell phải cho cùng giá trị ở đoạn đầu đó.

    python -m unittest discover -s tests
"""
import os, sys, unittest
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinnsoh.forecast import origin_features, causal_slope          # noqa: E402
from pinnsoh.postproc import causal_trend                           # noqa: E402
from pinnsoh import forecast_hist as H                              # noqa: E402

PREFIX = [1, 2, 3, 4, 5, 6, 11, 40]


class Causal(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.cyc = np.arange(60) * 2.0 + 1.0
        self.x = rng.normal(size=(60, 16))
        self.soh = 1.0 - 0.001 * self.cyc + 0.004 * rng.normal(size=60)

    def test_origin_features_prefix(self):
        Z, lev = origin_features(self.soh, self.cyc, self.x, 298.15)
        for n in PREFIX:
            z, l = origin_features(self.soh[:n], self.cyc[:n], self.x[:n], 298.15)
            np.testing.assert_allclose(z, Z[:n], atol=1e-6, err_msg=f'n={n}')
            np.testing.assert_allclose(l, lev[:n], atol=1e-12, err_msg=f'n={n}')

    def test_future_soh_hidden(self):
        """Đổi SOH từ hàng 30 trở đi không được làm đổi đặc trưng của các hàng trước 30."""
        a = H.features(NS(soh=self.soh, cycle=self.cyc, temp_k=298.15), self.x)
        s2 = self.soh.copy(); s2[30:] = 0.5
        b = H.features(NS(soh=s2, cycle=self.cyc, temp_k=298.15), self.x)
        for u, w in zip(a, b):
            np.testing.assert_allclose(u[:30], w[:30], atol=1e-12)

    def test_history_features_prefix(self):
        full = H.features(NS(soh=self.soh, cycle=self.cyc, temp_k=298.15), self.x)
        for n in PREFIX:
            part = H.features(NS(soh=self.soh[:n], cycle=self.cyc[:n], temp_k=298.15), self.x[:n])
            for u, w in zip(full, part):
                np.testing.assert_allclose(u[:n], w, atol=1e-6, err_msg=f'n={n}')

    def test_explicit_history_equals_measured(self):
        """features(soh=None) và features(soh=SOH đo) phải trùng nhau (E33 dùng tham số này cho lịch sử ước lượng)."""
        c = NS(soh=self.soh, cycle=self.cyc, temp_k=298.15)
        for u, w in zip(H.features(c, self.x), H.features(c, self.x, soh=self.soh.copy())):
            np.testing.assert_array_equal(u, w)

    def test_trend_and_slope_prefix(self):
        lev = causal_trend(self.soh, self.cyc, 20)
        sl = causal_slope(self.soh, self.cyc, 30)
        for n in PREFIX:
            np.testing.assert_allclose(causal_trend(self.soh[:n], self.cyc[:n], 20), lev[:n], atol=1e-12)
            np.testing.assert_allclose(causal_slope(self.soh[:n], self.cyc[:n], 30), sl[:n], atol=1e-12)

    def test_macro_weights(self):
        """Mỗi cell có tổng trọng số như nhau, chia đều cho các gốc của nó (như G-doan)."""
        np.testing.assert_allclose(H.weights(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])), [.125, .125, .25, .5])


if __name__ == '__main__':
    unittest.main()
