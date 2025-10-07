#!/usr/bin/env python3
import os
import runpy
import sys

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "python", "scripts", "compare_with_tiktoken.py")

if not os.path.exists(SCRIPT):
    sys.stderr.write("Unable to locate python/scripts/compare_with_tiktoken.py\n")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(SCRIPT))
runpy.run_path(SCRIPT, run_name="__main__")
