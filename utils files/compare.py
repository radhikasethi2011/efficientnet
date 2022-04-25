import cv2
import argparse
from utils import (
    load_efficientnet_model, preprocess, 
    read_classes, run_through_model, load_resnet50_model,
    plot_time_vs_iter
)
# Construct the argumet parser to parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg', 
                    help='path to the input image')
parser.add_argument('-d', '--device', default='cpu', 
                    help='computation device to use', 
                    choices=['cpu', 'cuda'])
args = vars(parser.parse_args())

# Number of times to forward pass through the model.
N_RUNS = 500
# Set the computation device.
DEVICE = args['device']
# Load the ImageNet class names.
categories = read_classes()
# Initialize the image transforms.
transform = preprocess()
print(f"Computation device: {DEVICE}")

image = cv2.imread(args['input'])
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Apply transforms to the input image.
input_tensor = transform(image)
# Add the batch dimension.
input_batch = input_tensor.unsqueeze(0)
# Move the input tensor and model to the computation device.
input_batch = input_batch.to(DEVICE)
# Initialize the EfficientNetB0 model model.
model = load_efficientnet_model()
model.to(DEVICE)
print('Running through EfficinetNetB0 model.')
effcientnetb0_times = run_through_model(model, input_batch, N_RUNS)
# Initialize the ResNet50 model model.
model = load_resnet50_model()
model.to(DEVICE)
print('Running through ResNet50 model.')
resnet50_times = run_through_model(model, input_batch, N_RUNS)
plot_time_vs_iter(
    model_names=['EfficientNetB0', 'ResNet50'],
    time_lists=[effcientnetb0_times, resnet50_times],
    device=DEVICE
)