import copy
import time
from collections import defaultdict

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from src.config import DEVICE, MODELS_DIR
from src.dataset import get_data_loaders
from src.model import create_model


def train_model(model, data_loaders, dataset_sizes, num_epochs=3):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    citerion = nn.CrossEntropyLoss()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start = time.time()

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
