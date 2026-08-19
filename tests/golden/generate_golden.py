"""ゴールデンデータ生成スクリプト。

使い方 (リポジトリルートで):
    .venv/bin/python tests/golden/generate_golden.py

物理を意図的に変更した場合のみ再生成し、変更内容を
doc/dev_records/worklog.md に記録すること。
"""
import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(TESTS_DIR, 'golden')
sys.path.insert(0, TESTS_DIR)

import S2MFD  # noqa: E402
from conftest import make_cfg, make_grid, run_short_simulation  # noqa: E402


def generate_setup_profiles():
    for name in ('defaults', 'alpha_omega', 'hotta10'):
        cfg = make_cfg(f'parameters/{name}.py', ix=32, jx=32)
        grid = make_grid(cfg)
        setup = S2MFD.Setup(cfg, grid)
        out = os.path.join(GOLDEN_DIR, f'setup_{name}.npz')
        np.savez(out,
                 om=setup.om, omrr=setup.omrr, omth=setup.omth,
                 et=setup.et, etrr=setup.etrr, so=setup.so,
                 urr=setup.urr, uth=setup.uth, ibase=setup.ibase)
        print(f'wrote {out}')


def generate_short_run():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(
            'parameters/alpha_omega.py',
            datadir=os.path.join(tmp, 'data'),
            ix=32, jx=32,
            tend=200 * 86400,
            dtout=40 * 86400,
        )
        sim = run_short_simulation(cfg)
        out = os.path.join(GOLDEN_DIR, 'short_run.npz')
        np.savez(out, Bph=sim.Bph, Aph=sim.Aph,
                 time=sim.time, n=sim.n, nd=sim.nd, dt=sim.dt)
        print(f'wrote {out} (n={sim.n}, nd={sim.nd}, '
              f'time={sim.time/86400:.1f} day)')


if __name__ == '__main__':
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    generate_setup_profiles()
    generate_short_run()
