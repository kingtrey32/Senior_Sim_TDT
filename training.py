# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:52:25 2023

@author: Ender
"""


import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
#from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

from itertools import chain

#for saving the model
#from joblib import Parallel, delayed
import joblib

from model import Fashion_Class_Model
#from model import output_label

from testing import model_test
from testing import return_predictionsList
from testing import return_labelsList
from testing import return_lossList
from testing import return_iterationList


# # Define a tensor on CUDA device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# tensor_cuda = torch.randn((3, 3), device=device)

# # Convert the tensor to numpy array
# numpy_array = tensor_cuda.cpu().numpy()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))




batchSize = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize)


image, label = next(iter(train_set))


data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(data_loader))
images, labels = batch




model = Fashion_Class_Model()
model.to(device)



error_rate = nn.CrossEntropyLoss()


learning_rate = .01 #previous value = 0.01 acc=84.6
momentum_value = 0.9


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)







#***********************Training a network and Testing it***************************

num_epochs_to_train = 10 #each epoch adds about 600 iterations it seems
num_iterations = 0 #counter for metrics at the end of training

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
        
        num_iterations += 1
        #save model here for export to testing file
        
#need to export this part to the testing file
#******testing the model**************************************************************************

        model_test(num_iterations, loss, model)

        
        
            #here is where I copied over to the testing.py file
        #if not (num_epochs % 50): #the same as "if num_wpochs % 50 == 0"
            
            
        

            #total = 0
            #correct = 0
            
            #for images, labels in test_loader:
                #images, labels = images.to(device), labels.to(device)
                #labels_list.append(labels)
                
                #test = Variable(images.view(100, 1, 28, 28))
                
                #outputs = model(test)
                
                #predictions = torch.max(outputs, 1)[1].to(device)
                #predictions_list.append(predictions)
                #correct += (predictions == labels).sum()
                
                #total += len(labels)
            
            #accuracy = correct * 100 / total
            #loss_list.append(loss.data)
            #iteration_list.append(num_epochs)
            ##accuracy_list.append(accuracy)
        
        #if not (num_epochs % 500):
            #print("Iteration: {}, Loss: {}, Accuracy: {}%" .format(num_epochs, loss.data, accuracy))
   


#need to convert thisto tensorboard maybe (she did say she ultimately didnt care and just wanted the metrics
#need epoch and not iteration
#need stats per class as well




loss_list = return_lossList()
iteration_list = return_iterationList()
#accuracy_list = []

#lists for knowing classwise accuracy, used in Confusion Matrix
predictions_list = return_predictionsList()
labels_list = return_labelsList()



#visualizing the loss and accuracy with iterations
plt.plot(iteration_list, loss_list)
plt.xlabel("No. of epochs")
plt.ylabel("Loss")
plt.title("Epochs vs. Loss")
plt.show()





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
print("Number of training iterations: {}\n" .format(num_iterations))

