import ctypes.util
import os
import shutil
import subprocess
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
    environments. It only attempts installation when ULH_AUTO_INSTALL_EGL is set and may
    run privileged package installs; enable only in trusted CI environments with
    controlled sudo configuration.
    """
    if ctypes.util.find_library("EGL"):
        return
    auto_install = os.environ.get("ULH_AUTO_INSTALL_EGL")
    ci_context = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
    if not (auto_install and ci_context):
        return
    apt = shutil.which("apt-get")
    sudo = shutil.which("sudo")
    if not (apt and sudo):
        # Skip silently when package manager is unavailable to keep tests portable.
        return
    try:
        subprocess.run(
            [sudo, "-n", apt, "update"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        subprocess.run(
            [sudo, "-n", apt, "install", "-y", "libegl1", "libgl1"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        warnings.warn("Timed out while attempting to install libegl1/libgl1.")
        return
    if ctypes.util.find_library("EGL") is None:
        warnings.warn("libEGL installation attempt failed; ensure libegl1/libgl1 are installed.")


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
            "libEGL runtime missing; install libegl1/libgl1 or set ULH_AUTO_INSTALL_EGL=1"
        )
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
    sys.path.append("src")
    import ulh_desktop  # noqa: F401
