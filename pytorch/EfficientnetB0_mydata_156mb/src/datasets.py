from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

TRAIN_DIR = '/content/drive/MyDrive/Brain_tumor_detection/gaussian'
TEST_DIR = '/content/drive/MyDrive/Brain_tumor_detection/brain_tumor_original_dataset/Testing'
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 32 
NUM_WORKERS = 4 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return train_transform
# testing transforms
def get_test_transform(IMAGE_SIZE):
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return test_transform


def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_SIZE))
    )
    dataset_test1 = datasets.ImageFolder(
        TEST_DIR, 
        transform=(get_test_transform(IMAGE_SIZE))
    )
    dataset_test, dataset_val = torch.utils.data.random_split(dataset_test1, [918, 393])
    return dataset_train, dataset_test, dataset_val, dataset_train.classes
    
def get_data_loaders(dataset_train, dataset_test, dataset_valid):
    """
    Prepares the training and testing and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_test: The testing dataset.
    :param dataset_valid: The validation dataset.
    Returns the training, testing and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader =  DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, test_loader, valid_loader 
