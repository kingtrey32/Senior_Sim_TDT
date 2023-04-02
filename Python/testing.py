import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

from model import MyModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 100

# Data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Fashion MNIST dataset
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = MyModel()
model = model.to(device)

# Load the trained model checkpoint
model.load_state_dict(torch.load('fashion_mnist_cnn.ckpt'))

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
