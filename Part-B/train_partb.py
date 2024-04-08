import numpy as np
import random
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandomHorizontalFlip, ToTensor, Compose, RandomRotation, Resize
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
import argparse

# Set random seed for reproducibility
torch.manual_seed(74)
random.seed(74)
np.random.seed(74)

# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("-dp","--dataset_path",default="/kaggle/input/nature-12k/inaturalist_12K/")
parser.add_argument("-e","--epochs",choices=[5,10,15,20,25,30],default=10)
parser.add_argument("-b","--batch_size",choices=[16,32,64],default=64)
parser.add_argument("-lr","--learning_rate",choices=[1e-3,1e-4],default=0.001)
parser.add_argument("-da","--data_aug",choices=[1,0],default=False)
parser.add_argument("-f","--freeze_k",default=-1)
# Function to get desired pre-trained model
def get_desired_model(model_name):
    if model_name.lower() == "googlenet":
        model = models.googlenet(pretrained=True)
    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)
    return model

# Function to freeze layers in the model
def freeze_layers(model, freeze_from_layer, freeze_to_layer):
    for idx, (name, param) in enumerate(model.named_children()):
        if idx >= freeze_from_layer and idx < freeze_to_layer:
            for param in param.parameters():
                param.requires_grad = False

# Function to print which layers require gradients
def print_requires_grad(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

# Function to train the model
def train_model(data_aug, model, train_dir, val_dir, num_epochs, batch_size, learning_rate, optimizer_name, img_size):
    # Define transforms for data augmentation and normalization
    if data_aug:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the datasets using ImageFolder
    val_dataset = ImageFolder(val_dir, transform=transform_val)
    train_dataset = ImageFolder(train_dir, transform=transform_train)


    labels = train_dataset.classes
    train_set, val_set = random_split(train_dataset, [8000, 1999])

    # Create data loaders for training and validation
    total_workers = 4
    test_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers = total_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,num_workers = total_workers, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = total_workers,shuffle=True)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'sgd':
        # print("SGD")
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, alpha = 0.99, eps = 1e-8)
        # print("RMSPROP")
    elif optimizer_name.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr = learning_rate, lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10)
        # print("ADAGRAD")
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-8)
        # print("ADAM")


    epoch = 0
    while epoch < num_epochs:
        model.train()  # Set model to training mode
        # count = 0
        running_loss, train_correct_p, train_total_p = 0.0, 0, 0
        val_loss,correct = 0.0,0
        total = 0

        for i, data in train_loader:
            inputs = i.to(device)
            labels = data.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            _, pred = torch.max(outputs.data, 1)
            train_total_p += labels.size(0)
            temp = (pred == labels).sum()
            train_correct_p += temp.item()


        # count += 1
#         print(count)

        # Validate the model after each epoch

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for i, data in val_loader:
                inputs = i.to(device) 
                labels = data.to(device)
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        # Print Train Statistics
        running_loss = running_loss / len(train_loader)
        acur_on_scale_1 = train_correct_p / train_total_p
        train_accuracy = acur_on_scale_1 * 100
        print(f'Epoch {epoch+1}, Train Loss: {running_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%')
        # Print validation statistics
        n_val = len(val_loader)
        temp2 = correct / total

        print('Epoch ',epoch + 1,' Validation Loss: ',round(val_loss/n_val,3),' Validation Accuracy: ',round(100 * temp2,2))

        epoch = epoch + 1
    print('Training finished')
    return model, 100 * temp2


# Define parameters and directories
model_name = "googlenet"
model = get_desired_model(model_name)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
freeze_from_layer = 0  # Start freezing from the first layer
freeze_to_layer = 8    # Freeze layers up to the kth layer
freeze_layers(model, freeze_from_layer, freeze_to_layer)
model = model.to(device)

num_epochs = 10
learning_rate = 0.001
batch_size = 250
img_size = 256
optimizer_name = "ADAM"
train_dir = '/kaggle/input/nature/inaturalist_12K/train'
val_dir = '/kaggle/input/nature/inaturalist_12K/val'

# Train the model
model, validation_accuracy = train_model(False, model, train_dir, val_dir, num_epochs, batch_size, learning_rate, optimizer_name, img_size)
