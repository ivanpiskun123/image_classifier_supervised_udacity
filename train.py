import argparse
import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import json
from datetime import datetime
torch.manual_seed(17)

parser = argparse.ArgumentParser(description='Command-line neural network training')
parser.add_argument('data_dir', type=str, help='Directory to the folder with training images')
parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, default='vgg19', help='Base model architecture (default is vgg19)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default is 0.01)')
parser.add_argument('--hidden_units', type=int, default=1024, help='Hidden units (default is 1024)')
parser.add_argument('--epoch', type=int, default=3,  help='Number of epochs to train on (default is 3)')
parser.add_argument('--gpu', type=bool, default=False, help='Enable training on gpu (default is False)')
my_namespace = parser.parse_args()

print(f'Parameters: \ndata_dir: {my_namespace.data_dir}\narch: {my_namespace.arch}\nlearning_rate: {my_namespace.learning_rate}\nepoch: {my_namespace.epoch}\ngpu:{my_namespace.gpu}\n\n')

data_dir = my_namespace.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for train images
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# A set of transforms with validation and test images
val_test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_folder = ImageFolder(train_dir, transform=train_transform)
valid_folder = ImageFolder(valid_dir, transform=val_test_transform)
test_folder = ImageFolder(test_dir, transform=val_test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = DataLoader(train_folder, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_folder, batch_size=32)
test_loader  = DataLoader(test_folder, batch_size=32)

print('TRAIN LOADER:', len(train_loader))
print('VALID LOADER:', len(valid_loader))
print('TEST LOADER: ', len(test_loader))

dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
data = {'train': train_folder, 'val': valid_folder, 'test': test_folder}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

CLASSES_NUM = len(cat_to_name.values())

pre_trained_models = {'vgg11': models.vgg11, 
                      'vgg13': models.vgg13, 
                      'vgg16': models.vgg16, 
                      'vgg19': models.vgg19}

model = pre_trained_models[my_namespace.arch](pretrained=True)
#model = models.vgg19(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, my_namespace.hidden_units)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(my_namespace.hidden_units, CLASSES_NUM)),
                          ('output', nn.LogSoftmax(dim=1))]))
    
model.classifier = classifier
print('Classifier: ', model.classifier)

model.class_to_idx = data['train'].class_to_idx

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = my_namespace.learning_rate)

# Parameters
epochs = my_namespace.epoch
print_every = 5

# Lists for saving results
train_loss_scores = []
test_loss_scores = []
accuracy_scores = []

device = torch.device('cuda:0' if my_namespace.gpu and torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(1, epochs + 1):
    print(f'\nEPOCH № {epoch} FROM {epochs} started...\n',
         f'====================\n\n')
    
    step = 0
    running_loss = 0
    accuracy = 0
    
    print(f'\n----- TRAINING STAGE -----\n')
    model.train()
    
    for images, labels in train_loader:
        step += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        #Forward
        output = model.forward(images)
        loss = criterion(output, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # training loss
        running_loss += loss.item()
        
        # accuracy
        ps = torch.exp(output)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        if step % print_every == 0:
            print(f'---------{step}---------\n')
            print(f'Training loss: {running_loss / step:.3f}\n')
            print(f'Accuracy: {accuracy / step:.3f}\n')
            train_loss_scores.append(round(running_loss / step, 3))
        
    step = 0
    running_loss = 0
    accuracy = 0
    
    print(f'\n----- VALIDATION STAGE -----\n')
    model.eval()
    
    for images, labels in valid_loader:
        step += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        #Only forward
        output = model.forward(images)
        loss = criterion(output, labels)

        # validation loss
        running_loss += loss.item()

        # accuracy
        ps = torch.exp(output)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        if step % print_every == 0:
            print(f'---------{step}---------\n')
            print(f'Validation loss: {running_loss / step:.3f}\n')
            print(f'Accuracy: {accuracy / step:.3f}\n')

            test_loss_scores.append(round(running_loss / step, 3))
            accuracy_scores.append(round(accuracy / step, 3))
print(f'TRAINING WITH {my_namespace.epoch} EPOCHS DONE!')
print(f' Minimum training loss value is: {min(train_loss_scores)}\n',
f'Minimum validation loss value is: {min(test_loss_scores)}\n',
f'Maximum accuracy is: {max(accuracy_scores)}')

checkpoint = {
    'base': my_namespace.arch,                                  # base pre-trained model
    'input_size': 25088,                              # input array size
    'hidden_units': my_namespace.hidden_units,
    'output_size': CLASSES_NUM,                       # output size (the same as classes amount)
    'classifier': classifier,                         # classifier object
    'class_to_idx': model.class_to_idx,               # classes and indexes dictionary
    'state_dict': model.state_dict(),                 # model state dict (the most important!)
    'optimizer': optimizer.state_dict()}              # model optimizer

checkpoint_loc = f'{my_namespace.save_dir}/checkpoint-{datetime.now()}.pth'
torch.save(checkpoint, checkpoint_loc)
print(f'Checkpoint saved to {checkpoint_loc}')