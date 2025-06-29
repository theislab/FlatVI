from pathlib import Path

ROOT = Path(__file__).parent.resolve()

PROJECT_FOLDER = ROOT / "project_folder"
EXPERIMENT_FOLDER = PROJECT_FOLDER / "experiments"
DATA_DIR = PROJECT_FOLDER / "data"
CKPT_FOLDER = PROJECT_FOLDER / "checkpoints"
