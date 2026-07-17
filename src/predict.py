import sys

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import DEVICE, MODELS_DIR, OUTPUTS_DIR
from src.dataset import TRANSFORMS
from src.model import create_model


def predict_proba(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img = TRANSFORMS["test"](img).unsqueeze(0)

    pred = model(img.to(DEVICE))
    pred = F.softmax(pred, dim=1)
    return pred.detach().cpu().numpy().flatten()


def show_prediction_confidence(prediction, class_names, save_path):
    pred_df = pd.DataFrame({"class_names": class_names, "values": prediction})
    plt.figure(figsize=(8,4))
    sns.barplot(x="values", y="class_names", data=pred_df, orient="h")
    plt.xlim([0,1])
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction chart saved to {save_path}")


if __name__ == "main":

    if len(sys.argv) <2:
        print("Usage: python -m src.predict <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    class_names = ["give_way", "no_entry", "priority_road", "stop"]

    model = create_model(len(class_names))
    model.load_state_dict(torch.load(MODELS.DIR / "base_model.pt", map_location=DEVICE))
    
    prediction = predict_proba(mode, image_path)

    for name, prob in zip(class_names, prediction):
        print(f" {name}: {prob:.4f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    show_prediction_confidence(prediction, class_names, OUTPUTS_DIR / "prediction_confidence.png")