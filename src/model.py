from torch import nn
from torchvision import models
from src.config import DEVICE

def create_model(num_classes):
    """loads pretrained ResNet-34 and replaces its final layer for our num classes"""
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)

    return model.to(DEVICE)