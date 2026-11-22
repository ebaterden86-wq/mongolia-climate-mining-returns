"""Reproduce the archived estimates using a privately obtained workbook or raw files."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
MASTER = "Master_Dataset_2018_2025.xlsx"
RAW_FILES = (
    "0975.HK.csv",
    "STGO.TO.csv",
    "^HSI.csv",
    "^GSPTSE.csv",
    "UHG.csv",
    "ATO.csv",
    "CMO-Historical-Data-Monthly.xlsx",
)


def run_python(script, *args, cwd):
    subprocess.run([sys.executable, str(script), *(str(arg) for arg in args)], check=True, cwd=cwd)


def recreate_master(raw_dir, run_dir):
    """Stage only the seven expected inputs and the unmodified Mission 1 scripts."""
    files = [raw_dir / name for name in RAW_FILES]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Required private inputs are missing:\n" + "\n".join(missing))
    mission1_dir = run_dir / "mission1"
    raw_stage = mission1_dir / "Raw"
    raw_stage.mkdir(parents=True)
    for file in files:
        shutil.copyfile(file, raw_stage / file.name)
    shutil.copytree(CODE / "mission1", mission1_dir / "Code", ignore=shutil.ignore_patterns("__pycache__"))
    run_python(mission1_dir / "Code" / "run_all.py", cwd=mission1_dir)
    return mission1_dir / "Final" / MASTER


def reproduce(workbook, raw_dir, output_dir):
    if workbook is not None and not workbook.is_file():
        raise FileNotFoundError(f"Private workbook was not found: {workbook}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir)).resolve()
    print(f"Private replication outputs: {run_dir}", flush=True)
    if workbook is None:
        workbook = recreate_master(raw_dir, run_dir)

    mission2_dir = run_dir / "mission2"
    mission2_dir.mkdir()
    # Mission 3 fingerprints the exact Mission 2 script and both generated files.
    mission2_script = mission2_dir / "run_mission_2.py"
    shutil.copyfile(CODE / "mission2" / mission2_script.name, mission2_script)
    run_python(mission2_script, "--input", workbook, "--output-dir", mission2_dir, cwd=mission2_dir)

    mission3_dir = run_dir / "mission3"
    run_python(
        CODE / "mission3" / "mission3_analysis.py",
        "--workbook", workbook,
        "--mission2-dir", mission2_dir,
        "--output-dir", mission3_dir,
        cwd=run_dir,
    )
    print(f"PASS replication pipeline. Private results are in {run_dir}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--master", type=Path, help=f"Path to a private {MASTER}.")
    inputs.add_argument("--raw-dir", type=Path, help="Directory containing the seven private source files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication",
                        help="Parent for new private runs (default: repository replication/).")
    args = parser.parse_args()
    workbook = args.master.resolve() if args.master is not None else None
    raw_dir = args.raw_dir.resolve() if args.raw_dir is not None else None
    try:
        reproduce(workbook, raw_dir, args.output_dir.resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Replication failed: {error}\n")


if __name__ == "__main__":
    main()
