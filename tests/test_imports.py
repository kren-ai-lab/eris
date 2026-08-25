import subprocess
import sys


def test_import():
    import eris  # noqa: F401


def test_core_and_learned_imports_do_not_load_gearnet_frameworks():
    script = """
import sys
import eris
import eris.core.learned

blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
