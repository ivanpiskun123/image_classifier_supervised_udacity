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
from PIL import Image
import matplotlib.pyplot as plt
import random, glob
from datetime import datetime
torch.manual_seed(17)

# The function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['base'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image) / 255
    
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - means) / std_dev
    norm_image = norm_image.transpose(2, 0, 1)

    return norm_image


def load_classes(json_path):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    cat_to_name = {int(key): value for key, value in cat_to_name.items()}
    return cat_to_name


def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    
    result = model.forward(image)
    probs = torch.exp(result).data
    
    top_ps, top_class= torch.topk(probs, topk)
    cat_to_name = load_classes(my_namespace.category_names)
    classes_dict = {value:cat_to_name[int(key)] for key, value in model.class_to_idx.items()}
    
    probs = [round(float(prob), 3) for prob in top_ps[0]]
    labels = [classes_dict[int(label)] for label in top_class[0]]
    return probs, labels

parser = argparse.ArgumentParser(description='Command-line neural network training')
parser.add_argument('image_dir', type=str, help='Directory to the image')
parser.add_argument('checkpoint', type=str, help='Trained model checkpoint')
parser.add_argument('--top_k', type=int, default=1, help='Top-k probable classes to output')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the file with custom classes')
parser.add_argument('--gpu', type=bool, default=False, help='Enable training on gpu (default is False)')
my_namespace = parser.parse_args()

model, optimizer = load_checkpoint(my_namespace.checkpoint)
if my_namespace.gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
probs, labels = predict(my_namespace.image_dir, model, my_namespace.top_k, device)

for i in range(my_namespace.top_k):
    print(f'Flower: {labels[i]}, probability: {probs[i]}')

