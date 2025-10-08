"""Entry point for generating synthetic data using the PicTex renderer.

This script serves as a convenient wrapper around the main data generation
script located in `manga_ocr_dev.synthetic_data_generator.run_generate`.
It is specifically configured to invoke the generation process with the
`renderer` argument set to 'pictex', utilizing the `SyntheticDataGeneratorV2`
engine.

This allows for a clear separation of concerns, providing a dedicated entry
point for the V2 data generation pipeline while reusing the common argument
parsing and execution logic from the main script. All command-line arguments
passed to this script are forwarded to the underlying `run` function.
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