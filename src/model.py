import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from PIL import Image

dataset = ImageFolder("../Images")
print(len(dataset))
print(dataset.classes)

breeds = []
def rename(name):
    return ' '.join(' '.join(name.split("-")[1:]).split('_'))

for n in dataset.classes:
    breeds.append(rename(n))

print(breeds)

#data_transforms = {
#    'train': transforms.Compose([
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'val': transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456. 0.406], [0.229, 0.224, 0.225])
#    ])
#}

random_seed = 69
torch.manual_seed(random_seed)

test_pct = 0.1
test_size = int(len(dataset) * test_pct)
dataset_size = len(dataset) - test_size

val_pct = 0.1
val_size = int(dataset_size * val_pct)
train_size = dataset_size - val_size

print(train_size, val_size, test_size)

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
print(len(train_ds), len(val_ds), len(test_ds))



img, label = train_ds[6]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))


img, label = train_ds[256]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))

img, label = train_ds[90]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))

img, label = train_ds[335]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))


img, label = train_ds[1000]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))

img, label = train_ds[1500]
print(dataset.classes[label])
plt.imshow(img)
plt.show()
print(type(img))
