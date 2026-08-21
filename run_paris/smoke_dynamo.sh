#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=2 S2MFD_PARALLEL=2 S2MFD_PARFILE=parameters/rempel06_paper.py
/scr/a000/c0234hotta/venv-paris/bin/python -u run_paris/dynamo7.py 108 72 0 0.4 smoke_d full \
    magnetic_buoyancy=1 bseed=3000 init=results_rempel/paper_relax/state.npz
