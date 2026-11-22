"""
Kiểm thử cho phần gộp từ kho G-doan (theo tests/test_improvements.py của kho đó): bài toán cập nhật có lịch sử
(pinnsoh/update_hist.py, E34) và HiPINN (pinnsoh/forecast_combo.py). Đặc trưng tại chu kỳ n chỉ được dùng các lần đo
trước n; lịch đo và đặc trưng tính trên một đoạn đầu của cell phải trùng với khi tính trên cả cell.

    python -m unittest discover -s tests
"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinnsoh.update_hist import schedule, inputs, online, weights          # noqa: E402
from pinnsoh.postproc import causal_mono                                  # noqa: E402


class Stub:
    def predict(self, x):
        return np.full(len(x), .7)


class UpdateHistory(unittest.TestCase):
    def setUp(self):
        self.c = np.arange(60) * 2.
        self.z = np.random.default_rng(2).normal(size=(60, 16))
        self.y = 1 - self.c * .001
        self.o = np.where(np.arange(60) % 10 == 0, self.y, np.nan)

    def test_schedule_prefix(self):
        for q in [.2, .1, .05, None]:
            a, b = schedule(self.c, q, 100)
            for n in [2, 11, 21, 55]:
                x, y = schedule(self.c[:n], q, 100)
                np.testing.assert_array_equal(a[:n], x); np.testing.assert_array_equal(b[:n], y)

    def test_input_prefix(self):
        a = inputs(self.z, self.c, self.o, 100)
        for n in [2, 11, 21, 55]:
            b = inputs(self.z[:n], self.c[:n], self.o[:n], 100)
            for k in a:
                np.testing.assert_allclose(b[k], a[k][a['rows'] < n], atol=1e-8)

    def test_current_measurement_hidden(self):
        a = inputs(self.z, self.c, self.o, 100); o = self.o.copy(); o[20] = .4
        b = inputs(self.z, self.c, o, 100); ix = a['rows'] <= 20
        np.testing.assert_allclose(a['rich'][ix], b['rich'][ix])

    def test_future_hidden(self):
        a = inputs(self.z, self.c, self.o, 100); o = self.o.copy(); o[30:] = .4
        b = inputs(self.z, self.c, o, 100); ix = a['rows'] < 30
        np.testing.assert_allclose(a['rich'][ix], b['rich'][ix])

    def test_mark_after_last_cycle(self):
        """Mốc đo cuối vượt chu kỳ cuối chỉ vì sai số làm tròn (15 × 89,6 > 1344) không được làm lỗi chỉ số; như E25,
        E34 và E32, mốc đó bị bỏ."""
        from pinnsoh.update import schedule as s32
        cyc = np.arange(1345, dtype=float)
        m32, _ = s32(cyc, 0.05, 1792.0)
        m34, _ = schedule(cyc, 0.05, 1792.0)
        np.testing.assert_array_equal(m32, m34)

    def test_cold_start(self):
        self.assertEqual(len(inputs(self.z, self.c, np.full(60, np.nan), 100)['rows']), 0)

    def test_invalid_cycles(self):
        with self.assertRaises(ValueError):
            schedule([0, 1, 1], .1, 100)
        with self.assertRaises(ValueError):
            inputs(self.z, self.c[::-1], self.o, 100)

    def test_e32_equivalence(self):
        """Đặc trưng của GBDT-cập nhật trùng với đặc trưng của E32 (pinnsoh/update.py) trên cùng lịch đo."""
        from pinnsoh.update import pairs
        m, ev = schedule(self.c, .1, 100); d = inputs(self.z, self.c, np.where(m, self.y, np.nan), 100)
        f, t, r, a = pairs(self.z, self.c, self.y, 100, .1); ix = ev[d['rows']]
        np.testing.assert_allclose(f, d['basic'][ix]); np.testing.assert_array_equal(r, d['rows'][ix])

    def test_macro_weights(self):
        np.testing.assert_allclose(weights(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])), [.125, .125, .25, .5])

    def test_clip_before_blend(self):
        b = dict(L_train=100, configs=[dict(rich=False, residual=False)], models=[Stub()],
                 choices={'q10': dict(model_index=0, prior='hold', alpha=.5)})
        p = online(b, self.z, self.c, self.o); d = inputs(self.z, self.c, self.o, 100)
        np.testing.assert_allclose(p[d['rows']], .5 * 1.15 + .5 * d['hold'])

    def test_mono(self):
        self.assertTrue(np.all(np.diff(causal_mono(self.y + .03 * np.sin(self.c), self.c, 10, a=.3)) <= 1e-10))

    def test_invalid_mono_alpha(self):
        with self.assertRaises(ValueError):
            causal_mono(self.y, self.c, 10, a=-1)


class Const:
    """Mô hình giả: GBDT trả độ giảm cố định, PINN trả (a, λ) cố định, nhánh theo đời trả SOH cố định."""
    def __init__(self, v):
        self.v = v

    def predict(self, x):
        return np.full(len(x), self.v)

    def params(self, z):
        return np.full(len(z), self.v), np.zeros(len(z))


class HiPINNMedian(unittest.TestCase):
    def test_combo_median(self):
        """HiPINN trong Combo.predict là trung vị của HiGBDT, PINN-dự báo, nhánh theo đời; HiPINN-tb là trung bình;
        HiPINN-w dùng đúng trọng số wh."""
        from pinnsoh.forecast_combo import Combo
        n = 6; rng = np.random.default_rng(1)
        d = dict(z=rng.normal(size=(n, 23)), rich=rng.normal(size=(n, 40)), lev=np.full(n, .95),
                 h=np.linspace(10, 300, n), origin=np.full(n, 100.))
        cfg = dict(rich=False, residual=False)
        life = Const(.85); life.predict = lambda dd: np.full(len(dd['h']), .85)
        m = Combo(Const(.02), Const(.05), cfg, 1.0, [Const(.3)], w=1.0, life=life, wh=(.5, .25, .25))
        P = m.predict(d)
        B = np.column_stack([P['gbdt_ls'], P['pinn'], P['doi']])
        np.testing.assert_allclose(P['hipinn'], np.median(B, axis=1))
        np.testing.assert_allclose(P['tb3'], B.mean(axis=1))
        np.testing.assert_allclose(P['hipinn_w'], .5 * P['gbdt_ls'] + .25 * P['pinn'] + .25 * P['doi'])


if __name__ == '__main__':
    unittest.main()
