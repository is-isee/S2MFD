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
        cell = np.minimum(grid.drr, rr * grid.dth)
        urr = setup.urr[m:grid.ixg - m, m:grid.jxg - m]
        uth = setup.uth[m:grid.ixg - m, m:grid.jxg - m]
        et = setup.et[m:grid.ixg - m, m:grid.jxg - m]
        dt_adv = c_cfl * cell / np.sqrt(urr**2 + uth**2 + 1.e-20)
        dt_dif = c_cfl * cell**2 / (2 * et + 1.e-20)
        self.dt = min(1.e10, dt_adv.min(), dt_dif.min())

    def save(self):
        """
        Saves data to file
        """
        print(f"{self.time/86400:7.1f} [day]; n={self.n:06d}; nd={self.nd:04d}")
        filename = self.get_data_file_path(self.nd)
        np.savez(file=filename,
                 Bph=self.Bph, Aph=self.Aph, time=self.time, n=self.n)

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
            self.save()

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

        self.Bph = 0.5*self.Bph + 0.5*Bphn
        self.Aph = 0.5*self.Aph + 0.5*Aphn

    def main_loop(self):
        """
        Runs the main loop of the simulation

        Notes
        -----
        スナップショットは「積分 → 時刻更新 → 出力」の順で書かれるため、
        保存される場は time ラベルと一致する (2026-08-19 修正。それ以前は
        1ステップ前の場が保存されていた)。
        """
        cfg = self.cfg

        while self.time < cfg.tend:
            self.tvd_runge_kutta()
            self.time += self.dt
            self.n += 1
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.nd += 1
                self.check_finite()
                self.save()
