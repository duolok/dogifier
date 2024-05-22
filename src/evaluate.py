import torch
from model import model, device
from data_processing import dataloaders

model.load_state_dict(torch.load('dog_breed_classifier.pth'))

def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader['test'].dataset)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

evaluate_model(model, dataloaders)
