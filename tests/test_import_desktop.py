import ctypes.util
import os
import sys
import warnings
import importlib.util
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _ensure_egl_runtime():
    """
    Best-effort helper to make EGL runtime available for PySide6 imports in headless
    environments. When ULH_AUTO_INSTALL_EGL is set in CI it emits a warning if libEGL
    is missing so CI images can preinstall libegl1/libgl1 explicitly.
    """
    if ctypes.util.find_library("EGL"):
        return
    auto_install = os.environ.get("ULH_AUTO_INSTALL_EGL", "").lower() == "ci-sudo-ok"
    ci_context = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
    if not (auto_install and ci_context):
        return
    warnings.warn(
        "libEGL runtime missing; install libegl1/libgl1 in CI image or set ULH_AUTO_INSTALL_EGL=ci-sudo-ok only after preinstalling dependencies."
    )


@pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 not installed; skipping desktop import"
)
def test_import_ulh_desktop_offscreen():
    """
    Smoke test for the desktop UI:
    - Forces headless Qt platform so no real display is required.
    - Ensures ulh_desktop.py can be imported without errors.
    """
    _ensure_egl_runtime()
    if ctypes.util.find_library("EGL") is None:
        pytest.skip(
            "libEGL runtime missing; install libegl1/libgl1 or set ULH_AUTO_INSTALL_EGL=ci-sudo-ok in CI"
        )
    pytest.importorskip("PySide6.QtWidgets")
    sys.path.append("src")
    import ulh_desktop  # noqa: F401
