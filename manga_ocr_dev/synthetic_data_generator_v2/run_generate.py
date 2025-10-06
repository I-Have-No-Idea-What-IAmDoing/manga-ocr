"""
This script is a wrapper for the main run_generate.py script,
calling it with the `pictex` renderer.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import fire
from manga_ocr_dev.synthetic_data_generator.run_generate import run

if __name__ == "__main__":
    fire.Fire(lambda **kwargs: run(renderer='pictex', **kwargs))