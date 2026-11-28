import yaml
from pathlib import Path

from src.preprocessing import DataPreprocessor


def load_config():
    """Load project settings from config.yaml."""

    config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    """Prepare and initialise the machine learning pipeline."""

    config = load_config()

    preprocessor = DataPreprocessor(config["preprocessing"])

    print("Project configuration loaded successfully.")
    print("Target column:", config["data"]["target_column"])
    print("Preprocessing settings initialised.")

    # TODO 1: Import data using Person 2's data_ingestion.py
    # TODO 2: Apply preprocessing
    # TODO 3: Train models using Person 3's model_trainer.py
    # TODO 4: Evaluate models using Person 3's evaluator.py


if __name__ == "__main__":
    main()