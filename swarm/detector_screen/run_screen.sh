#!/bin/bash
# One model per process: a CUDA device-side assert poisons the whole process.
cd "$(dirname "$0")/../.."
PY=venv/Scripts/python.exe
for m in $($PY -c "import sys; sys.path.insert(0,'swarm/detector_screen'); import screen; print(' '.join(screen.CANDIDATES))"); do
  $PY swarm/detector_screen/screen.py --models "$m" >> swarm/detector_screen/screen_log.txt 2>&1 || \
  $PY -c "
import pandas as pd, sys, os
p='swarm/detector_screen/screen_results.csv'; d=pd.read_csv(p)
if sys.argv[1] not in set(d['model']):
    pd.concat([d, pd.DataFrame([{'model': sys.argv[1], 'error': 'process crashed (CUDA assert / vocab mismatch)'}])]).to_csv(p, index=False)
" "$m"
done
