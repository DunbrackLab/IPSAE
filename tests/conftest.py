# Ensure the src/ipsae package is importable (and takes precedence over the
# backwards-compatible ipsae.py wrapper script in the repository root) even
# when the package has not been installed.
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)
