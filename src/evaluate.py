import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import DEVICE, MODELS_DIR, OUTPUTS_DIR
from src.dataset import get_data_loaders
from src.model import create_model


def get_predictions(model, data_loader):
    model.eval()
    predictions, real_values = [], []

    with torch.no_grad():
        for inputs, labels, in data_loader:
            inputs = inputs.to(DEVICE) 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.numpy())

            return predictions, real_values


def show_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    data_loaders, dataset_sizes, class_names = get_data_loaders()
    model = create_model(len(class_names))
    model.load_state_dict(torch.load(MODELS_DIR / "base_model.pt", map_location=DEVICE))

    predictions, real_values = get_predictions(model, data_loaders["test"])

    print(classification_report(real_values, predictions, labels=list(range(len(class_names))), target_names=class_names))

    cm = confusion_matrix(real_values, predictions)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    show_confusion_matrix(cm, class_names, OUTPUTS_DIR / "confusion_matrix.png")