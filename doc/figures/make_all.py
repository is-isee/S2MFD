"""ドキュメント用の図をまとめて作る.

::

    python doc/figures/make_all.py [出力先]

既定の出力先は ``doc/source/_static/figures``。**結果ファイル
(``results_rempel/``) が要る**ので、生成済みの PNG はリポジトリに
コミットしてある (CI で作り直さない)。

図の色の割り当ての方針は ``style.py`` の docstring を参照。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT = os.path.join(ROOT, 'doc', 'source', '_static', 'figures')

import matplotlib                                       # noqa: E402
matplotlib.use('Agg')

sys.path.insert(0, HERE)
import fig_profiles, fig_convergence, fig_butterfly     # noqa: E402
import fig_table1, fig_energy                           # noqa: E402


def main(outdir=DEFAULT):
    os.makedirs(outdir, exist_ok=True)
    for name, fn in (('reference model', fig_profiles.main),
                     ('convergence', fig_convergence.main),
                     ('butterfly', fig_butterfly.main),
                     ('Table 1', fig_table1.main),
                     ('energy budget', fig_energy.main)):
        try:
            fn(outdir)
        except Exception as e:                          # noqa: BLE001
            print(f'  !! {name}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
