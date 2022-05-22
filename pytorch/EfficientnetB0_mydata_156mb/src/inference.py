import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots

DATA_PATH = '/content/Testing'
IMAGE_SIZE = 224
DEVICE = 'cpu'
# Class names.

dataset_train, dataset_test, dataset_valid, dataset_classes = get_datasets()
print(f"[INFO]: Number of testing images: {len(dataset_test)}")
print(f"[INFO]: Class names: {dataset_classes}\n")
# Load the training and validation data loaders.
train_loader, test_loader, valid_loader = get_data_loaders(dataset_train, dataset_test, dataset_valid)
dataiter = iter(test_loader)
images, labels = dataiter.next()
print("##############################")
print("image shape ",images.shape)
print("label shape ", labels.shape)
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=4)
checkpoint = torch.load('/content/efficientnet/pytorch/EfficientnetB0_mydata_156mb/outputs/model.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

# Get all the test image paths.
correct_count, all_count = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
  
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    img = images[i].view(1,3, 224, 224)
    #img = torch.unsqueeze()
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.cpu()[i]
    print("true label: ", true_label)
    print("pred label: ", pred_label)
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
