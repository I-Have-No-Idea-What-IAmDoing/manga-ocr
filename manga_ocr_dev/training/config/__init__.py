import yaml
from pathlib import Path
from manga_ocr_dev.training.config.schemas import AppConfig

def load_config(config_path: str) -> AppConfig:
    """Loads the YAML configuration file and validates it with Pydantic.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        AppConfig: A Pydantic object representing the validated configuration.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)

# Load the default config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
config = load_config(str(CONFIG_PATH))