import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.config import DATA_DIR, DATASETS, MEAN_NUMS, STD_NUMS

TRANSFORMS = {
    "train" : T.compose([
        T.RandomResizedCrop(size=256),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN_NUMS, STD_NUMS),
    ]),
    "val": T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(MEAN_NUMS, STD_NUMS),
    ]),
    "test": T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(MEAN_NUMS, STD_NUMS),
    ]),
}


def get_data_loaders(batch_size=4, num_workers=4):
    "num_workers -> CPU cores to prepare images in parallel for GPU"

    image_datasets = {
        d: ImageFolder(str(DATA_DIR / d), TRANSFORMS[d]) for d in DATASETS
    }

    dataset_sizes = {
        d: len(image_datasets[d]) for d in DATASETS
    }

    class_names = image_datasets["train"].classes

    return data_loaders, dataset_sizes, class_names