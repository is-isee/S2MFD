import glob
import json
import os

import numpy as np
import S2MFD

# 再開時の整合性チェックで比較しないキー (再開時に変更してよいもの)
_RESUMABLE_KEYS = {
    'cont_flag', 'tend', 'dtout', 'datadir', 'parameter_file',
    'configfile', 'gridfile', 'setupfile', 'legendrefile',
}


class Simulation(S2MFD.Data):
    """
    Class for running the simulation inheriting from S2MFD.Data
    """
    def __init__(self, cfg, grid=None, setup=None, legendre=None):
        """
        Parameters
        ----------
        cfg : S2MFD.Cfg
            Configuration object
        grid : S2MFD.Grid, optional
            Grid object
        setup : S2MFD.Setup, optional
            Setup object
        legendre : S2MFD.Legendre, optional
            Legendre object
        """
        # 基本量が変更されている場合に派生パラメタへ反映する
        if hasattr(cfg, 'resolve'):
            cfg.resolve()

        if grid is None:
            grid = S2MFD.Grid(
                ix=cfg.ix, jx=cfg.jx, margin=cfg.margin
               ,rrmin=cfg.rrmin, rrmax=cfg.rrmax
               ,thmin=cfg.thmin, thmax=cfg.thmax
               ,stretch=getattr(cfg, 'grid_stretch', 0.0)
               ,stretch_center=getattr(cfg, 'grid_stretch_center',
                                       0.5*(cfg.rrmin + cfg.rrmax))
               ,stretch_width=getattr(cfg, 'grid_stretch_width', 0.0)
               )
        if setup is None:
            setup = S2MFD.Setup(cfg, grid)
        if legendre is None:
            legendre = S2MFD.Legendre(grid)
        super().__init__(cfg, grid, setup, legendre)

    def initialize_simulation(self):
        """
        Initialize the simulation by setting up the grid and setup objects.

        If ``cfg.cont_flag`` is True and the data directory already exists,
        the run is resumed from the saved grid/setup files. In that case the
        current configuration must be consistent with the saved one
        (except for keys such as ``tend``/``dtout``); otherwise a
        RuntimeError is raised.
        """
        # make data directory
        if not os.path.isdir(self.cfg.datadir):
            self.cfg.cont_flag = False
            os.makedirs(self.cfg.datadir, exist_ok=True)

        if self.cfg.cont_flag:
            self._check_resume_consistency()
            print(f"Resuming existing run in '{self.cfg.datadir}' "
                  "(set cfg.cont_flag = False to start a fresh run)")

        self.cfg.save()

        # create grid and setup
        if self.cfg.cont_flag:
            self.grid = S2MFD.Grid.load(
                os.path.join(self.cfg.datadir, self.cfg.gridfile))
            self.setup = S2MFD.Setup.load(
                os.path.join(self.cfg.datadir, self.cfg.setupfile))
            # Legendre は grid の純関数なので再構築する
            # (旧フォーマットの legendre.npz との互換性のため)
            self.legendre = S2MFD.Legendre(self.grid)
        else:
            self.grid.save(os.path.join(self.cfg.datadir, self.cfg.gridfile))
            self.setup.save(os.path.join(self.cfg.datadir, self.cfg.setupfile))
            self.legendre.save(os.path.join(self.cfg.datadir, self.cfg.legendrefile))

    def _check_resume_consistency(self):
        """再開時に既存の config.json と現在の設定の整合性を確認する。

        再開ランは grid/setup を既存の npz から読み込むため、それらに影響する
        パラメタが変わっていると「設定と実際の背景場が食い違う」状態になる。
        """
        configpath = os.path.join(self.cfg.datadir, self.cfg.configfile)
        if not os.path.exists(configpath):
            return
        with open(configpath) as f:
            saved = json.load(f)
        mismatches = []
        for key, saved_value in saved.items():
            if key in _RESUMABLE_KEYS or not hasattr(self.cfg, key):
                continue
            current = getattr(self.cfg, key)
            if not isinstance(current, (int, float, str, bool, list, dict, type(None))):
                continue
            if current != saved_value:
                mismatches.append(f"{key}: saved={saved_value!r} current={current!r}")
        if mismatches:
            raise RuntimeError(
                "Resume consistency check failed for "
                f"'{self.cfg.datadir}'. The existing run was created with a "
                "different configuration:\n  "
                + "\n  ".join(mismatches)
                + "\nUse a new datadir (or set cfg.cont_flag = False after "
                "removing the old data) to start a fresh run."
            )

    def cfl_condition(self):
        """
        Applies CFL condition
        """
        grid = self.grid
        setup = self.setup
        # CFL condition
        # NOTE: 拡散制限の min(dr, r*dθ)**2/(2η) 形は dr << r*dθ (現行の
        # 128x128 格子で約6倍差) を前提とした近似。セルが等方に近い格子に
        # 変更する場合は 1/(2η(1/dr² + 1/(r dθ)²)) 形への修正を検討すること。
        c_cfl = 0.8
        m = grid.margin
        rr = grid.rr[m:grid.ixg - m, None]
        cell = np.minimum(grid.drr[m:grid.ixg - m, None], rr * grid.dth)
        urr = setup.urr[m:grid.ixg - m, m:grid.jxg - m]
        uth = setup.uth[m:grid.ixg - m, m:grid.jxg - m]
        et = setup.et[m:grid.ixg - m, m:grid.jxg - m]
        dt_adv = c_cfl * cell / np.sqrt(urr**2 + uth**2 + 1.e-20)
        dt_dif = c_cfl * cell**2 / (2 * et + 1.e-20)
        self.dt = min(1.e10, dt_adv.min(), dt_dif.min())

    def save(self):
        """
        Saves data to file

        Notes
        -----
        nd, dt に加えて、その時点の uu0 (子午面流振幅), so0 (α効果振幅) も
        保存する。時間依存パラメタのランを事後解析で復元するため。
        """
        if getattr(self.cfg, 'verbose', True):
            print(f"{self.time/86400:7.1f} [day]; n={self.n:06d}; nd={self.nd:04d}")
        filename = self.get_data_file_path(self.nd)
        np.savez(file=filename,
                 Bph=self.Bph, Aph=self.Aph, time=self.time, n=self.n,
                 nd=self.nd, dt=self.dt,
                 uu0=float(self.cfg.uu0), so0=float(self.cfg.so0))

    def initial_condition(self):
        """
        Applies initial condition
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        # 初期条件
        if cfg.cont_flag:
            files = glob.glob(os.path.join(cfg.datadir, 'data.*.npz'))
            self.nd = max([int(f.split('.')[-2]) for f in files])
            self.data_load(self.nd)
        else:
            self.nd = 0
            self.n = 0
            self.time = 0.0
            self.Bph = np.zeros((grid.ixg, grid.jxg))
            self.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
            self.Aph[0:setup.ibase, :] = 0
            self.update_time_dependent_parameters()
            self.save()

    def update_time_dependent_parameters(self):
        """時間依存パラメタのフックを評価し、背景場を差分更新する。

        規約: cfg に呼び出し可能な属性 ``uu0_of_time(t)`` / ``so0_of_time(t)``
        (t はシミュレーション時刻 [s]、返り値は振幅) が定義されていれば、
        現在時刻で評価して cfg.uu0 / cfg.so0 を更新し、Setup の該当
        プロファイルのみ再構築する。流れが変わった場合は CFL も再計算する。
        """
        cfg = self.cfg
        flow_updated = False
        if callable(getattr(cfg, 'so0_of_time', None)):
            cfg.so0 = float(cfg.so0_of_time(self.time))
            self.setup.build_alpha(cfg, self.grid)
        if callable(getattr(cfg, 'uu0_of_time', None)):
            cfg.uu0 = float(cfg.uu0_of_time(self.time))
            self.setup.build_flow(cfg, self.grid)
            flow_updated = True
        if flow_updated and self.dt is not None:
            self.cfl_condition()

    def check_finite(self):
        """
        Raises RuntimeError if the fields contain NaN/Inf.

        発散したランを tend まで走らせないための早期検知。
        """
        if not (np.all(np.isfinite(self.Bph)) and np.all(np.isfinite(self.Aph))):
            raise RuntimeError(
                f"Non-finite field detected at t={self.time/86400:.1f} day "
                f"(n={self.n}). The run is numerically unstable "
                "(check CFL condition and parameter values)."
            )

    def tvd_runge_kutta(self):
        """
        Applies TVD Runge-Kutta method
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        legendre = self.legendre
        #### dynamo equation
        Bphm, Aphm = S2MFD.physics.time_marching(self.Bph, self.Aph, self.dt, cfg, grid, setup)
        Bphm, Aphm = S2MFD.physics.boundary_condition(Bphm, Aphm, cfg, grid, legendre)

        Bphn, Aphn = S2MFD.physics.time_marching(Bphm, Aphm, self.dt, cfg, grid, setup)
        Bphn, Aphn = S2MFD.physics.boundary_condition(Bphn, Aphn, cfg, grid, legendre)

        self.Bph = S2MFD.physics.rk_combine(self.Bph, Bphn)
        self.Aph = S2MFD.physics.rk_combine(self.Aph, Aphn)

    def main_loop(self):
        """
        Runs the main loop of the simulation

        Notes
        -----
        スナップショットは「積分 → 時刻更新 → 出力」の順で書かれるため、
        保存される場は time ラベルと一致する (2026-08-19 修正。それ以前は
        1ステップ前の場が保存されていた)。
        """
        # 実体は run_window (numba 側で出力時刻までまとめて積分する)
        self.run_window(self.cfg.tend)

    def run_window(self, t_end, on_output=None, save_output=True):
        """指定時刻まで積分する (パラメタ推定用の観測窓ループ)。

        main_loop と同じ構造だが、終了時刻を引数で受け、出力ステップ毎に
        コールバックを呼べる。時間依存パラメタのフック
        (cfg.uu0_of_time / cfg.so0_of_time) も出力ステップ毎に評価される。

        Parameters
        ----------
        t_end : float
            終了時刻 [s]
        on_output : callable, optional
            出力ステップ毎に on_output(self) の形で呼ばれる。
            黒点数時系列の記録などに使う。
        save_output : bool
            False にするとスナップショットを書かない (GA の個体評価など
            ディスク出力が不要な場合)。

        Returns
        -------
        S2MFD.Simulation
            self
        """
        cfg = self.cfg
        while self.time < t_end:
            # 次の出力時刻までを numba 側でまとめて進める
            t_next = (np.floor(self.time/cfg.dtout) + 1.0)*cfg.dtout
            t_stop = min(t_next, t_end)
            before = self.time
            S2MFD.physics.stepping.advance_to(self, t_stop)
            if self.time//cfg.dtout != before//cfg.dtout:
                self.nd += 1
                self.update_time_dependent_parameters()
                self.check_finite()
                if save_output:
                    self.save()
                if on_output is not None:
                    on_output(self)
        return self

    def set_field(self, Bph, Aph, time=0.0, n=0, nd=0):
        """磁場の状態を直接設定する (スピンアップ初期場の投入などに使う)。"""
        self.Bph = np.array(Bph, dtype=np.float64, copy=True)
        self.Aph = np.array(Aph, dtype=np.float64, copy=True)
        self.time = float(time)
        self.n = int(n)
        self.nd = int(nd)

    def spin_up(self, duration, until_minimum=True, proxy=None,
                max_extra=100 * 365 * 86400):
        """現在の状態から助走計算を行う (パラメタ推定の初期条件生成)。

        Shimizu & Hotta (2026) の手順:

        1. duration (既定は呼び出し側で 80 年を指定) だけ積分する
        2. until_minimum=True なら、黒点数プロキシが極小 (連続3点の中央が
           最小) になるまで積分を続ける

        スナップショットは出力しない。終了時の self.Bph/Aph が推定の
        初期状態になる。時刻・ステップ数のラベル合わせは呼び出し側で行う
        (set_field や time/nd への代入)。

        Parameters
        ----------
        duration : float
            最低限積分する時間 [s]
        until_minimum : bool
            True なら黒点数プロキシの極小まで継続する
        proxy : callable, optional
            proxy(sim) -> float。既定は tools.sunspot_proxy。
        max_extra : float
            極小検出の打ち切り時間 [s] (発振しない解での無限ループ防止)

        Returns
        -------
        S2MFD.Simulation
            self
        """
        proxy_is_default = proxy is None
        if proxy_is_default:
            def proxy(sim):
                return S2MFD.tools.sunspot_proxy(sim.Bph, sim.grid, sim.cfg)

        t_start = self.time
        sn_history = np.zeros(3)

        # 履歴用の最後3ステップを残して、numba 側でまとめて進める
        # (ステップ数は 1 ステップずつ回す実装と一致する)
        t_target = t_start + duration
        t_batch = t_target - 3.0*self.dt
        if t_batch > self.time:
            S2MFD.physics.stepping.advance_to(self, t_batch)

        recorded = 0
        while self.time < t_target or recorded < 3:
            self.tvd_runge_kutta()
            self.time += self.dt
            self.n += 1
            sn_history[0] = sn_history[1]
            sn_history[1] = sn_history[2]
            sn_history[2] = proxy(self)
            recorded += 1

        if until_minimum:
            t_limit = self.time + max_extra
            base, loca, gamma = self._proxy_indices()
            if proxy_is_default and base >= 0:
                # 極小検出まで numba 側で回す
                _, ok = S2MFD.physics.stepping.advance_until_minimum(
                    self, t_limit, sn_history, base, loca, gamma)
            else:
                ok = True
                while not (sn_history[1] < sn_history[0]
                           and sn_history[1] < sn_history[2]):
                    if self.time > t_limit:
                        ok = False
                        break
                    self.tvd_runge_kutta()
                    self.time += self.dt
                    self.n += 1
                    sn_history[0] = sn_history[1]
                    sn_history[1] = sn_history[2]
                    sn_history[2] = proxy(self)
            if not ok:
                raise RuntimeError(
                    "spin_up: no sunspot-number minimum detected within "
                    f"{max_extra/86400/365:.0f} years of extra integration."
                )

        self.check_finite()
        return self

    def _proxy_indices(self, r_frac=0.7, theta_deg=75.0, gamma=5.8653520852):
        """既定の黒点数プロキシが参照する格子インデックスを返す。"""
        base = 1 + int(np.argmin(abs(self.grid.rr - r_frac*self.cfg.RSUN)))
        loca = int(np.argmin(abs(self.grid.th - theta_deg/180*np.pi)))
        return base, loca, gamma
