"""微分演算子 (S2MFD.tools) のテスト。

注意: drr1/dth1 の docstring は実装と逆である (2026-08-19 レビューで確認)。
実装の実際の挙動:
  - dir='up': dqq[i] = (qq[i] - qq[i-1]) / d   (後退差分、i = 1..N-1 に格納)
  - dir='dw': dqq[i] = (qq[i+1] - qq[i]) / d   (前進差分、i = 0..N-2 に格納)
diffusion() では 'up' でセル境界 i-1/2 の勾配を作り 'dw' で発散を取る
保存形ペアとして正しく使われている。本テストは実装の挙動を正として固定する。
"""
import numpy as np
import pytest

from S2MFD.tools import drr1, drr2, dth1, dth2


def _linear_r(nx=8, ny=5, a=2.0, b=1.0, d=0.1):
    r = np.arange(nx) * d
    qq = a * r[:, None] + b * np.ones((nx, ny))
    return qq, d


def _linear_th(nx=5, ny=8, a=2.0, b=1.0, d=0.1):
    t = np.arange(ny) * d
    qq = a * t[None, :] + b * np.ones((nx, ny))
    return qq, d


class TestDrr1:
    def test_up_is_backward_difference(self):
        qq, d = _linear_r()
        dqq = drr1(qq, np.full(qq.shape[0], d), 'up')
        # i=0 は 0 のまま、i>=1 に (qq[i]-qq[i-1])/d
        assert np.allclose(dqq[0, :], 0.0)
        assert np.allclose(dqq[1:, :], 2.0)

    def test_dw_is_forward_difference(self):
        qq, d = _linear_r()
        dqq = drr1(qq, np.full(qq.shape[0], d), 'dw')
        # i=N-1 は 0 のまま、i<=N-2 に (qq[i+1]-qq[i])/d
        assert np.allclose(dqq[-1, :], 0.0)
        assert np.allclose(dqq[:-1, :], 2.0)

    def test_up_dw_conservative_pair(self):
        """'up' → 'dw' の合成が2階中心差分(保存形)になること。"""
        nx, d = 32, 0.05
        r = np.arange(nx) * d
        qq = np.sin(r)[:, None] * np.ones((nx, 3))
        hh = np.full(nx, d)
        lap = drr1(drr1(qq, hh, 'up'), hh, 'dw')
        expected = (qq[2:, :] - 2 * qq[1:-1, :] + qq[:-2, :]) / d**2
        assert np.allclose(lap[1:-1, :], expected)


class TestDrr2:
    def test_exact_for_quadratic(self):
        nx, d = 16, 0.1
        r = np.arange(nx) * d
        qq = (3.0 * r**2 + 2.0 * r + 1.0)[:, None] * np.ones((nx, 4))
        dqq = drr2(qq, np.full(nx, 2.0*d))
        expected = (6.0 * r + 2.0)[1:-1]
        assert np.allclose(dqq[1:-1, :], expected[:, None])
        # 端は 0 のまま
        assert np.allclose(dqq[0, :], 0.0)
        assert np.allclose(dqq[-1, :], 0.0)

    def test_second_order_convergence(self):
        errors = []
        for nx in (32, 64):
            d = 1.0 / nx
            r = np.arange(nx) * d
            qq = np.sin(2 * np.pi * r)[:, None] * np.ones((nx, 2))
            dqq = drr2(qq, np.full(nx, 2.0*d))
            expected = 2 * np.pi * np.cos(2 * np.pi * r)
            errors.append(np.max(np.abs(dqq[1:-1, 0] - expected[1:-1])))
        # 格子を半分にすると誤差 ~1/4
        assert errors[1] < errors[0] / 3.0


class TestDth1:
    def test_up_is_backward_difference(self):
        qq, d = _linear_th()
        dqq = dth1(qq, d, 'up')
        assert np.allclose(dqq[:, 0], 0.0)
        assert np.allclose(dqq[:, 1:], 2.0)

    def test_dw_is_forward_difference(self):
        qq, d = _linear_th()
        dqq = dth1(qq, d, 'dw')
        assert np.allclose(dqq[:, -1], 0.0)
        assert np.allclose(dqq[:, :-1], 2.0)


class TestDth2:
    def test_exact_for_quadratic(self):
        ny, d = 16, 0.1
        t = np.arange(ny) * d
        qq = (t**2)[None, :] * np.ones((4, ny))
        dqq = dth2(qq, d)
        assert np.allclose(dqq[:, 1:-1], (2.0 * t)[None, 1:-1])
