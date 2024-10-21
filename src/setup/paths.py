import os 
from pathlib import Path 


PARENT_DIR = Path("_file_").parent.resolve()
IMAGES_DIR = PARENT_DIR/"images"

DATA_DIR = PARENT_DIR/"data"
RAW_DATA_DIR = DATA_DIR/"raw"

MODELS_DIR = PARENT_DIR/"models"
LOCAL_SAVE_DIR = MODELS_DIR/"locally_created"
COMET_SAVE_DIR = MODELS_DIR/"comet_downloads"

PARQUETS = RAW_DATA_DIR/"Parquets"

CLEANED_DATA = DATA_DIR/"cleaned"
TRANSFORMED_DATA = DATA_DIR/"transformed"
GEOGRAPHICAL_DATA = DATA_DIR/"geographical"

ROUNDING_INDEXER = GEOGRAPHICAL_DATA / "rounding_indexer"
MIXED_INDEXER = GEOGRAPHICAL_DATA / "mixed_indexer"

TIME_SERIES_DATA = TRANSFORMED_DATA/"time_series"
TRAINING_DATA = TRANSFORMED_DATA/"training_data"
INFERENCE_DATA = TRANSFORMED_DATA/"inference"


def make_fundamental_paths() -> None:
    for path in [
        DATA_DIR, CLEANED_DATA, RAW_DATA_DIR, PARQUETS, GEOGRAPHICAL_DATA, TRANSFORMED_DATA, TIME_SERIES_DATA, 
        IMAGES_DIR, TRAINING_DATA, INFERENCE_DATA, MODELS_DIR, LOCAL_SAVE_DIR, COMET_SAVE_DIR, ROUNDING_INDEXER,
        MIXED_INDEXER
    ]: 
        if not Path(path).exists():
            os.mkdir(path)


if __name__ == "__main__":
    os.chdir(PARENT_DIR)
    print(PARENT_DIR)
    