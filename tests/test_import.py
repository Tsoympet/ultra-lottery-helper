import pathlib
import runpy


def test_import():
    """
    Basic smoke test:
    - Verifies that src/ultra_lottery_helper.py exists.
    - Runs the script in an isolated namespace (not as __main__),
      so no side-effects like QApplication startup are triggered.
    """
    script = pathlib.Path("src") / "ultra_lottery_helper.py"
    assert script.exists(), f"Missing script: {script}"
    runpy.run_path(str(script), run_name="not_main")
