import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import sys
import os

CUR_DIR = os.getcwd()
Project_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = os.path.join(Project_DIR, "data")
CHECKPOINT_DIR = os.path.join(Project_DIR, "checkpoints")

sys.path.append(Project_DIR)

from src.lsh_layer import LSHLayer
from src.models.simple_mlp import SimpleMLP
from src.utils import train

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

train_dataset = MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
test_dataset = MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

print(f"size of train_dataset: {len(train_dataset)}")
print(f"size of test dataset: {len(test_dataset)}")

num_epochs = 20
hidden_size = 10000
num_classes = 10
model = SimpleMLP(28 * 28, hidden_size, num_classes, flatten_first=True)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model = train(model, train_loader, criterion, optimizer, test_loader, num_epochs=num_epochs)

# Save the model
savepath = f"{CHECKPOINT_DIR}/mnist_h{hidden_size}.pt"
torch.save(model.state_dict(), f"{savepath}")
print(f"Model saved to {savepath}")