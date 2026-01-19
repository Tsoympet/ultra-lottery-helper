import os
import sys
import importlib.util
import pytest
import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 not installed; skipping desktop import"
)
def test_import_ulh_desktop_offscreen():
    """
    Smoke test for the desktop UI:
    - Forces QT_QPA_PLATFORM=offscreen so no real display is required.
    - Ensures ulh_desktop.py can be imported without errors.
    """
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.append("src")
    import ulh_desktop  # noqa: F401
