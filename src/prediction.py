import torch
import os
from model import initialize_model
from data_processing import full_dataset
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

num_classes = len(full_dataset.breeds)
model = initialize_model(num_classes=num_classes)
model.load_state_dict(torch.load('dog_breed_classifier.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0) 
    return image

def predict_breed(image_path, model, device):
    image = load_image(image_path)
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    breed = full_dataset.breeds[preds.item()]
    return breed

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

#directory = '../../ziggy/'
#for path in get_file_paths(directory):
#    image_path = path
#    predicted_breed = predict_breed(image_path, model, device)
#    print(f'PATH: {path} | The predicted breed is: {predicted_breed}') 

path = '../../german_shephard.jpg'
image_path = path
predicted_breed = predict_breed(image_path, model, device)
print(f'PATH: {path} | The predicted breed is: {predicted_breed}') 
