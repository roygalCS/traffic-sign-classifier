import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

DATASETS = ["train", "val", "test"]
CLASS_NAMES = ["priority_road",  "give_way", "stop", "no_entry"]
CLASS_INDICES = [12, 13, 14, 17]

#ImageNet normalization stats
MEAN_NUMS = [0.485, 0.456, 0.406]
STD_NUMS = [0.229, 0.224, 0.225]

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")