import pathlib
import runpy

def test_import():
    # Φορτώνει το src/ultra_lottery_helper.py χωρίς να απαιτείται το 'src' ως package στο PYTHONPATH
    script = pathlib.Path("src") / "ultra_lottery_helper.py"
    assert script.exists(), f"Missing script: {script}"
    # Τρέχει το module σε απομονωμένο namespace, χωρίς να εκκινήσει το __main__ guard
    runpy.run_path(str(script), run_name="not_main")
