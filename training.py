import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from model import MyModel

classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def output_label(label):
    return classes[label]


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Fashion MNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = MyModel()
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
train_loss = []  # Store the training loss values for each epoch
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track the total loss for the epoch
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()  # Add the batch loss to the epoch loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            # Get a test image and its corresponding label
            test_image, test_label = test_dataset[0]
            test_image = test_image.to(device)

            # Get the predicted label
            y_pred = model(test_image.unsqueeze(0))
            predicted_class = output_label(torch.argmax(y_pred))


            # Print the predicted class label
            print("Predicted class: ", predicted_class)
            
            
    epoch_loss /= len(train_loader)  # Calculate the average epoch loss
    train_loss.append(epoch_loss)  # Add the average epoch loss to the list 
    
    

# Save the model checkpoint
torch.save(model.state_dict(), 'fashion_mnist_cnn.ckpt')



plt.plot(range(num_epochs), train_loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()













