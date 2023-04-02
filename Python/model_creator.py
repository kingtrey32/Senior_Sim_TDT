# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

from itertools import chain

#for saving the model
from joblib import Parallel, delayed
import joblib











#provides a label, or Class, to the object in question
def output_label(label) :
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot" }
    
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))




batchSize = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize)


image, label = next(iter(train_set))


data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(data_loader))
images, labels = batch







#*************************************************The CNN model***********************
#We will need to change a few things about it to fit her criteria like leakyReLu etc.

class Fashion_Class_Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 5) #num_channels changed from 3 to 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#*************************************************************************************





model = Fashion_Class_Model()
model.to(device)

error_rate = nn.CrossEntropyLoss()

learning_rate = .01 #previous value = 0.01 acc=84.6
momentum_value = 0.92

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)







#***********************Training a network and Testing it***************************

num_epochs_to_train = 2 #each epoch adds about 600 iterations it seems
num_epochs = 0 #counter for metrics at the end of training

#lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
#accuracy_list = []

#lists for knowing classwise accuracy, used in Confusion Matrix
predictions_list = []
labels_list = []

for epoch in range(num_epochs_to_train) :
#****training the model**********************************************************************
    for images, labels in train_loader:
        #transferring images and laels to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        #forward pass
        outputs = model(train)
        loss = error_rate(outputs, labels)
        
        #initializing gradient at 0 for each batch
        optimizer.zero_grad()
        
        #backpropogation of error found
        loss.backward()
        
        #optimizing parameters given loss rate
        optimizer.step()
        
        num_epochs += 1
        
#******testing the model**************************************************************************
        if not (num_epochs % 50): #the same as "if num_wpochs % 50 == 0"
            total = 0
            correct = 0
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                
                test = Variable(images.view(100, 1, 28, 28))
                
                outputs = model(test)
                
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(num_epochs)
            #accuracy_list.append(accuracy)
        
        if not (num_epochs % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%" .format(num_epochs, loss.data, accuracy))
          

#visualizing the loss and accuracy with iterations
# plt.plot(iteration_list, loss_list)
# plt.xlabel("No. of iterations")
# plt.ylabel("Loss")
# plt.title("Iterations vs. Loss")
# plt.show()





#*************************************saving the trained model**************************************
filename = "trained_Model.pkl"
joblib.dump(model, filename)
#seems to work, file shows up in file explorer

#***************************************************************************************************


#*************************************loading a model*********************************************
#needed for the Android studio
#variable_name_for_loaded_model = joblib.load('filename_of_saved_model.pkl')

#model_from_joblib = joblib.load('filename.pkl') # i think the file needs to be in the same folder as the calling file
#***************************************************************************************************





#printing the Confusion Matrix
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

confusion_matrix(labels_l, predictions_l)
print("\nClassification report for CNN:\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))
print("Epochs {}" .format(num_epochs_to_train))





