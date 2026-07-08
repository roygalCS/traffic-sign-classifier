import copy
import time
from collections import defaultdict

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from src.config import DEVICE, MODELS_DIR
from src.dataset import get_data_loaders
from src.model import create_model