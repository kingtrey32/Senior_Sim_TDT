# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:53:22 2023

@author: Ender
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
#from torch.utils.data import Dataset, DataLoader

from itertools import chain

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# tensor_cuda = torch.randn((3, 3), device=device)

# # Convert the tensor to numpy array
# numpy_array = tensor_cuda.cpu().numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

batchSize = 100

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize)

#***************************************
#need to load the saved model here 
#***************************************

labels_list = []
predictions_list = []
loss_list = []
iteration_list = []
accuracy = 0

    
def model_test(num_iterations, loss, model):

    
    if not (num_iterations % 600): #the same as "if num_wpochs % 50 == 0"

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
        iteration_list.append(num_iterations/600)
        #accuracy_list.append(accuracy)
        
        print("Epoch: {}, Loss: {}, Accuracy: {}%" .format(num_iterations/600, loss.data, accuracy))
    
    #if not (num_iterations % 600):
        #print("Iteration: {}, Loss: {}, Accuracy: {}%" .format(num_iterations, loss.data, accuracy))


def return_labelsList():
    return labels_list

def return_lossList():
    return loss_list

def return_iterationList():
    return iteration_list

def return_predictionsList():
    return predictions_list