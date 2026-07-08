import shutil
import urllib.request
import zipfile
from glob import glob

import numpy as np

from src.config import (
    RAW_DATA_DIR, DATA_DIR, DATASETS, CLASS_NAMES, CLASS_INDICES, RANDOM_SEED
)

DATASET_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"


def download_and_extract():
    """Downloads the GTSRB zip and extracts it into raw_data/, if not already done."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "GTSRB.zip"
    extracted_marker = RAW_DATA_DIR / "GTSRB"

    if extracted_marker.exists():
        print("Dataset already downloaded and extracted, skipping.")
        return

    print("Downloading dataset (this is a few hundred MB, may take a while)...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    zip_path.unlink() 


def build_folders():
    """Creates data/train/<class>, data/val/<class>, data/test/<class>."""
    for ds in DATASETS:
        for cls in CLASS_NAMES:
            (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)


def split_and_copy():
    """For each of our 4 chosen classes, shuffle its images and copy 80/10/10 into train/val/test."""
    np.random.seed(RANDOM_SEED)
    train_folders = sorted(glob(str(RAW_DATA_DIR / "GTSRB" / "Final_Training" / "Images" / "*")))

    for i, cls_index in enumerate(CLASS_INDICES):
        class_name = CLASS_NAMES[i]
        image_paths = np.array(glob(f"{train_folders[cls_index]}/*.ppm"))
        print(f"{class_name}: {len(image_paths)} images found")

        np.random.shuffle(image_paths)

        ds_split = np.split(
            image_paths,
            indices_or_sections=[int(0.8 * len(image_paths)), int(0.9 * len(image_paths))],
        )

        for ds, images in zip(DATASETS, ds_split):
            for img_path in images:
                shutil.copy(img_path, DATA_DIR / ds / class_name)


if __name__ == "__main__":
    download_and_extract()
    build_folders()
    split_and_copy()
    print("Data prep done. Check the data/ folder.")