import argparse
from src.prepare_data import download_and_extract, build_folders, split_and_copy
from src.dataset import get_data_loaders
from src.model import create_model
from src.train import train_model
from src.config import MODELS_DIR
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("Step 1/3: Preparing data...")
    download_and_extract()
    build_folders()
    split_and_copy()

    print("Step 2/3: Training...")
    data_loaders, dataset_sizes, class_names = get_data_loaders()
    model = create_model(len(class_names))
    model, history = train_model(model, data_loaders, dataset_sizes, num_epochs=args.epochs)

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "base_model.pt")
    print("Step 3/3: Done. Run `python -m src.evaluate` to see results.")


if __name__ == "__main__":
    main()