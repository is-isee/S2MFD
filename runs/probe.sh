#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
pwd
export OMP_NUM_THREADS=${T:-4} NUMBA_NUM_THREADS=${T:-4} MPLBACKEND=Agg
/scr/a000/c0234hotta/venv-paris/bin/python - <<'PY'
import sys; sys.path.insert(0,'tests')
import S2MFD, numba, numpy
print("S2MFD ok, numba threads =", numba.get_num_threads(), "numpy", numpy.__version__)
from conftest import make_cfg, make_grid
print("conftest ok")
PY
