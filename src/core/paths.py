from pathlib import Path 
import os 

PARENT_DIR = Path("_file_").parent.resolve().parent
DATA_DIR = PARENT_DIR/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
MODELS_DIR = PARENT_DIR/"models"

DATA_PICKLES = RAW_DATA_DIR/"Pickles"
ORIGINAL_DATA_PICKLES = DATA_PICKLES/"With Original Variable Types"
ALTERED_DATA_PICKLES = DATA_PICKLES/"With Altered Variable Types"

CLEANED_DATA = DATA_DIR/"cleaned"
TRANSFORMED_DATA_DIR = DATA_DIR/"transformed"


if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)