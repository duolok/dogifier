import os
import torch
from torch.utils.data import Dataset
from PIL import IMAGE
from parse_annotations import parse_annotation

class DogBreedDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, breeds, transform=None):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.breeds = breeds
        self.transform = transform
        self.annotations = self.load_annotations()
    
    def load_annotations(self):
        annotations = []
        for annotation_file in os.listdir(self.annotations_dir):
            if annotation_file.endswith(".xml"):
                annotation_path = os.path.join(self.annotations_dir, annotation_file)
                filename, breed = parse_annotation(annotation_path)
                annotations.append((filename, breed))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, breed = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.breeds.index(breed)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

