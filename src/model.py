import torch
import torch.nn as nn
from torchvision.models import resnet50
from data_processing import full_dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

breeds = full_dataset.breeds

def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    model_ft = resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model(num_classes=len(breeds))
model = model.to(device)
