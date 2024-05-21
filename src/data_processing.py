import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from xml.etree.ElementTree import parse

class DogBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.breeds = []
        self._load_dataset()

    def _load_dataset(self):
        for breed_dir in os.listdir(self.root_dir):
            breed_path = os.path.join(self.root_dir, breed_dir)
            if os.path.isdir(breed_path):
                breed_name = ' '.join(breed_dir.split('-')[1:]).replace('_', ' ')
                if breed_name not in self.breeds:
                    self.breeds.append(breed_name)
                for image_name in os.listdir(breed_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append((os.path.join(breed_path, image_name), breed_name))
        
        self.breeds.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, breed = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.breeds.index(breed)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

root_dir = "../data/Images"

full_dataset = DogBreedDataset(root_dir, transform=train_transform)
test_pct = 0.1
val_pct = 0.1
test_size = int(len(full_dataset) * test_pct)
val_size = int((len(full_dataset) - test_size) * val_pct)
train_size = len(full_dataset) - test_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}

# Display some images and labels to check
def show_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
    for i in range(num_samples):
        img, label = dataset[i]
        img = transforms.ToPILImage()(img)
        axes[i].imshow(img)
        axes[i].set_title(full_dataset.breeds[label])
        axes[i].axis('off')
    plt.show()

show_samples(train_ds)
