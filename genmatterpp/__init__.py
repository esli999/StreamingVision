"""Self-contained vendor of GenMatter++ at branch arijit/realtime-demo.

The vendored code uses ``from genmatter.*`` absolute imports throughout, so
this package's directory is added to ``sys.path`` on first import — that
makes the inner ``genmatter/`` package importable without modifying any
vendored file.  See ``VENDORED.md`` for the source SHA and policy.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
