from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = ["cbow.py", "skipgram.py", "sgns.py"]


def run_script(script_name):
    script_path = SCRIPT_DIR / script_name
    print(f"Running {script_name}...")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPT_DIR,
        check=True,
    )
    print(f"Finished {script_name}.")


if __name__ == "__main__":
    for script in SCRIPTS:
        run_script(script)
    print("Generated cbow.vec, skipgram.vec, and sgns.vec.")
