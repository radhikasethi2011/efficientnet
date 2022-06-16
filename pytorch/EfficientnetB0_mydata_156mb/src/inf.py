import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots
from tqdm.auto import tqdm
import torch.nn as nn

DATA_PATH = '/content/drive/MyDrive/Brain_tumor_detection/brain_tumor_original_dataset/Testing'
IMAGE_SIZE = 224
device = 'cpu'
DEVICE = 'cpu'
# Class names.

model = build_model(pretrained=False, fine_tune=False, num_classes=4)
checkpoint = torch.load('/content/efficientnet/pytorch/EfficientnetB0_mydata_156mb/outputs/model.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

dataset_train, dataset_test, dataset_valid, dataset_classes = get_datasets()
train_loader, test_loader, valid_loader = get_data_loaders(dataset_train, dataset_test, dataset_valid)
criterion = nn.CrossEntropyLoss()


model.eval()
print('Testing')
valid_running_loss = 0.0
valid_running_correct = 0
counter = 0
with torch.no_grad():
    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        counter += 1
        
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()
    
# Loss and accuracy for the complete epoch.
epoch_loss = valid_running_loss / counter
epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))

print(epoch_acc)
  

