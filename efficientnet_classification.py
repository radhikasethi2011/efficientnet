import torch
import cv2 
import argparse
import time
from google.colab.patches import cv2_imshow
from utils import (
    load_efficientnet_model, preprocess, read_classes
)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg', 
                    help='path to the input image')
parser.add_argument('-d', '--device', default='cpu', 
                    help='computation device to use', 
                    choices=['cpu', 'gpu'])
args = vars(parser.parse_args())

# Set the computation device.
DEVICE = args['device']
# Initialize the model.
model = load_efficientnet_model()
# Load the ImageNet class names.
categories = read_classes()
# Initialize the image transforms.
transform = preprocess()
print(f"Computation device: {DEVICE}")

image = cv2.imread(args['input'])
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(DEVICE)
model.to(DEVICE)

with torch.no_grad():
    start_time = time.time()
    output = model(input_batch)
    end_time = time.time()
# Get the softmax probabilities.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# Check the top 5 categories that are predicted.
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    cv2.putText(image, f"{top5_prob[i].item()*100:.3f}%", (15, (i+1)*30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{categories[top5_catid[i]]}", (160, (i+1)*30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    print(categories[top5_catid[i]], top5_prob[i].item())
    print('yay')
cv2_imshow(image)
cv2.waitKey(0)
# Define the outfile file name.
save_name = "outputs/image_1.jpg"
cv2.imwrite(save_name, image)
print(f"Forward pass time: {(end_time-start_time):.3f} seconds")