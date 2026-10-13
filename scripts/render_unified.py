#!/usr/bin/env python3
"""Run from the repository root. Presets are in configs/unified.yaml."""
import os
import sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified.cli import main

if __name__ == '__main__':
    main()
