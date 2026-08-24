"""``results_rempel/*/dynamo.npz`` の履歴列を名前で引けること。

``run_paris/dynamo7.py`` の ``HCOLS`` は 2026-08-23 に E_Omega / E_M /
放射層の磁束を末尾に足して 16 列から 22 列になった。このとき
``run_paris/table1.py`` が QKEYS の位置を ``h.shape[1] - len(qkeys)``
で当てにしていたため、新しい記録で 14 列目 (= E_B より後ろ) を
``Q_Lambda`` として読み、res144_* の Q_Lambda が 100 分の 1、
Q_nu^M が 1e12 倍という値を出した。**列を差し引きで当てない。**
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QKEYS = ['Q_Lambda', 'Q_nu_Omega', 'Q_C', 'Q_L_Omega',
         'Q_nu_M', 'Q_B', 'Q_L_M', 'Q_eta']
# dynamo7.py の HCOLS と同じ並び (22 列)
HCOLS_NEW = (['t', 'Bph_max', 'Br_surf', 'Bph_735eq', 'DR', 'dOm_pole',
              'dOm_lat60', 'E_B', 'Om1_pole_raw', 'Om1_lat60_raw'] + QKEYS
             + ['E_Omega', 'E_M', 'Phi_rad', 'absPhi_rad'])


def _load_script(name):
    path = os.path.join(ROOT, 'run_paris', f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'_rp_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hist(ncol):
    """列番号がそのまま値になる履歴 (nrow=4)。"""
    return np.tile(np.arange(ncol, dtype=float), (4, 1))


def _write(tmp_path, tag, hist, **extra):
    d = tmp_path / 'results_rempel' / tag
    d.mkdir(parents=True)
    np.savez(d / 'dynamo.npz', hist=hist, qkeys=np.array(QKEYS), **extra)


class TestTable1Columns:
    def test_new_layout_uses_hist_cols(self):
        """22 列の記録では QKEYS は 10-17。差し引きの 14 ではない。"""
        table1 = _load_script('table1')
        h = _hist(len(HCOLS_NEW))
        d = {'qkeys': np.array(QKEYS), 'hist_cols': np.array(HCOLS_NEW)}
        qc = table1.qcols(d, h)
        assert qc['Q_Lambda'][0] == 10.0
        assert qc['Q_eta'][0] == 17.0

    def test_old_layout_falls_back_to_offset(self):
        """hist_cols の無い 16 列の記録は 8 列目から QKEYS。"""
        table1 = _load_script('table1')
        h = _hist(16)
        qc = table1.qcols({'qkeys': np.array(QKEYS)}, h)
        assert qc['Q_Lambda'][0] == 8.0
        assert qc['Q_eta'][0] == 15.0


class TestBudgetCheckColumns:
    @pytest.mark.parametrize('with_cols', [True, False])
    def test_load_finds_q_lambda(self, tmp_path, monkeypatch, with_cols):
        """新旧どちらの記録でも Q_Lambda と E_B を正しく引くこと。

        古い記録に対する読み口は 2026-08-24 まで ``10 + i`` 固定で、
        16 列の記録では IndexError になっていた。
        """
        budget = _load_script('budget_check')
        ncol = len(HCOLS_NEW) if with_cols else 16
        extra = {'hist_cols': np.array(HCOLS_NEW)} if with_cols else {}
        _write(tmp_path, 'dummy', _hist(ncol), **extra)
        monkeypatch.chdir(tmp_path)
        cols = budget.load('dummy')
        assert cols['E_B'][0] == 7.0
        assert cols['Q_Lambda'][0] == (10.0 if with_cols else 8.0)


class TestCyclePeriodSplitsByTime:
    """``cycle_period`` の「後半」は時刻で切ること。

    延長ランをつなぐと区間ごとに出力間隔が違う (42 年に 2000 点 +
    18 年に 2000 点)。インデックスの中点で切ると t = 42 年から後ろだけを
    見ることになり、反転が 2 回しか入らず周期が nan になっていた。
    """

    def test_uneven_sampling_still_gives_the_period(self):
        table1 = _load_script('table1')
        # 0-42 年を 2000 点、42-60 年を 2000 点 (後半が 4.7 倍密)
        t = np.concatenate([np.linspace(0, 42, 2000),
                            np.linspace(42, 60, 2000)[1:]])
        b = np.sin(2*np.pi*t/18.0)          # 周期 18 年
        per, n = table1.cycle_period(t, b)
        assert n >= 2          # 反転 3 回 -> 周期 2 個
        assert abs(per - 18.0) < 0.2

    def test_uniform_sampling_unchanged(self):
        table1 = _load_script('table1')
        t = np.linspace(0, 60, 4000)
        b = np.sin(2*np.pi*t/18.0)
        per, n = table1.cycle_period(t, b)
        assert abs(per - 18.0) < 0.2


class TestDetrendKeepsTheWholeWindow:
    """``detrend`` は区間を削らないこと。

    2026-08-25 まで幅 18 年の移動平均を引いて両端を捨てていた。解析区間も
    20 年なので**残るのが 2 年ぶんだけ**になり、「末尾 20 年の最大」が
    「どこか 2 年間の最大」になっていた。これで表 1 のトーショナル振動が
    論文比 0.67-0.70 に見えていた (正しくは 0.95-1.04)。
    """

    def test_length_is_preserved(self):
        table1 = _load_script('table1')
        t = np.linspace(40.0, 60.0, 704)
        x = np.sin(2*np.pi*t/18.0)
        assert len(table1.detrend(t, x, 18.0)) == len(t)

    def test_linear_drift_is_removed_and_amplitude_kept(self):
        """ドリフトは消え、振幅は 1 割の偏りの内側で残る。

        20 年の区間に周期 18 年が 1.1 サイクルしか入らないので、1 次の
        当てはめが振動そのものを一部吸い、振幅を 1 割ほど過大に返す
        (3.0 -> 3.28)。全ランに同じだけ乗るので比較には効かないが、
        論文値との比を 1 割の精度で議論してはいけない。
        """
        table1 = _load_script('table1')
        t = np.linspace(40.0, 60.0, 704)
        x = 3.0*np.sin(2*np.pi*t/18.0) + 0.5*(t - 50.0) + 7.0
        r = table1.detrend(t, x, 18.0)
        assert 3.0 <= np.abs(r).max() < 3.45         # 振幅は残る (偏り +9%)
        assert abs(np.polyfit(t, r, 1)[0]) < 1e-9    # ドリフトは消える


class TestBphIsMeasuredAtTheRightRadius:
    """``max(B_phi)`` は r = 0.735 RSUN で測ること。

    Rempel (2006) 表 1 の注記: "The maximum of Omega - Omega_bar and B_r is
    evaluated at 0.985 R_sun and **the maximum of B_Phi at 0.735 R_sun**"。
    2026-08-25 まで ``hist[:,1]`` (全半径の最大) を使っており、実測で
    5-14 パーセント過大だった。
    """

    def test_uses_butter_not_the_global_max(self, tmp_path, monkeypatch):
        table1 = _load_script('table1')
        n = 400
        h = np.zeros((n, len(HCOLS_NEW)))
        h[:, 0] = np.linspace(0, 60, n)*3.156e7
        h[:, 1] = 9.99                      # 全半径の最大 (別の半径にある)
        h[:, 7] = 1.0e38                    # E_B (ゼロ割りを避けるため)
        for i, k in enumerate(QKEYS):       # Q_* もゼロ以外にしておく
            h[:, HCOLS_NEW.index(k)] = 1.0e31
        # butter は 0.735 RSUN の B_phi [G]。ピークは 2.0 T = 2e4 G
        bu = np.zeros((n, 8))
        bu[:, 3] = 2.0e4*np.sin(np.linspace(0, 6*np.pi, n))
        d = tmp_path/'results_rempel'/'dummy'
        d.mkdir(parents=True)
        np.savez(d/'dynamo.npz', hist=h, butter=bu, th=np.linspace(0.01, 1.55, 8),
                 qkeys=np.array(QKEYS), hist_cols=np.array(HCOLS_NEW))
        monkeypatch.chdir(tmp_path)
        r = table1.analyse('dummy')
        assert abs(r['Bph'] - 2.0) < 0.02       # 0.735R の値
        assert abs(r['Bph_anyr'] - 9.99) < 1e-6  # 参考値として残る
