"""Acceso al código original de code/surrogate_ELM para las pruebas de equivalencia.

Fuera de este repositorio (o sin sus dependencias: skelm, tensorflow) esas pruebas se saltan.
"""
import sys

import pytest

from zcatcmp import config

ORIGINAL_DIR = config.REPO / "code" / "surrogate_ELM"
if ORIGINAL_DIR.is_dir() and str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

try:
    import archive as original_archive      # noqa: E402
    import snsgaii as original_snsgaii      # noqa: E402
    import surrogate as original_surrogate  # noqa: E402
    AVAILABLE = True
except ImportError:
    original_archive = original_snsgaii = original_surrogate = None
    AVAILABLE = False

requires_original = pytest.mark.skipif(not AVAILABLE, reason="requiere el código original de code/surrogate_ELM")
