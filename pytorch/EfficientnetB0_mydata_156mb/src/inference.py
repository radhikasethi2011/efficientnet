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

model = build_model(pretrained=False, fine_tune=False, num_classes=4)
checkpoint = torch.load('/content/efficientnet/pytorch/EfficientnetB0_mydata_156mb/outputs/model.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

twod_path_list = []
class_names = ['Te-gl', 'Te-me', 'Te-no', 'Te-pi']
class_dict = {'Te-gl': 'glioma', 'Te-me': 'meningioma', 'Te-no': 'notumor', 'Te-pi' : 'pituitary'}
for count, i in enumerate(class_dict): 
  all_image_paths = glob.glob(f'/content/Testing/{class_dict[i]}/*')
  twod_path_list.append(all_image_paths)
  
all_image_path_list = [item for sublist in twod_path_list for item in sublist]

k = 0
ctr = 0
for image_path in all_image_path_list: 
  gt_class_name = class_dict[image_path.split(os.path.sep)[-1][:5]]
  image = cv2.imread(image_path)
  orig_image = image.copy()
  # Preprocess the image
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])
  image = transform(image)
  image = torch.unsqueeze(image, 0)
  image = image.to(DEVICE)
  
  # Forward pass throught the image.
  outputs = model(image)
  outputs = outputs.detach().numpy()
  pred_class_name = class_dict[class_names[np.argmax(outputs[0])]]
  print(f'K = {k}, GT = {gt_class_name}, Pred = {pred_class_name}')
  k+=1
  if(gt_class_name==pred_class_name): 
    ctr+=1
print("CTR: ", ctr)
print('acc: ', ctr/1311)


