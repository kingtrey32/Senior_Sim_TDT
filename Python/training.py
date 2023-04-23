#Authors: D. Joyner, T. Rana, T. Daniels

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from itertools import chain
from model import Fashion_Class_Model
from testing import model_test
from testing import return_predictionsList
from testing import return_labelsList
from testing import return_lossList
from testing import return_iterationList


#**************Initial variables for model instantiation and training*************
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))])
#transform = transforms.Compose([transforms.ToTensor()]) #original transform value

train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=transform)

batchSize = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize)
image, label = next(iter(train_set))
data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
batch = next(iter(data_loader))
images, labels = batch

model = Fashion_Class_Model()
model.to(device)

error_rate = nn.CrossEntropyLoss()
learning_rate = .001 
momentum_value = 0.9

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)

#***********************Training a network***************************

num_epochs_to_train = 80 #each epoch adds about 600 iterations it seems
num_iterations = 0 #counter for metrics at the end of training

#lists for visualization of loss and accuracy
loss_list = []
iteration_list = []

#lists for knowing classwise accuracy, used in Confusion Matrix
predictions_list = []
labels_list = []

for epoch in range(num_epochs_to_train) :  
    #overwrites the model object with the most recent weights if not on the first epoch
    if epoch > 1:
        model.load_state_dict(torch.load('C:/Users/Ender/.spyder-py3/fashion_mnist_cnn.ckpt'))
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        train = Variable(images.view(100, 3, 28, 28))
        labels = Variable(labels)
        
        #forward pass
        outputs = model(train)
        loss = error_rate(outputs, labels)
        
        #initializing gradient at 0
        optimizer.zero_grad()
        
        #backpropogation of error found
        loss.backward()
        
        #optimizing parameters given loss rate
        optimizer.step()
        
        num_iterations += 1
        
        #save model here for export to testing file
        torch.save(model.state_dict(),'C:/Users/Ender/.spyder-py3/fashion_mnist_cnn.ckpt')
        
        #calls the testing function
        model_test(num_iterations, loss)


#*************saves the final version here for export to Android*********
#uses a different method, for model deployment testing in Android Studio, did not work
#torch.save(model.state_dict(), 'C:/Users/Ender/.spyder-py3/trained_cnn.pt')

model.eval()
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted_3_Channel.pt')

#*************metrics list population section
#uses getters for testing.py to populate the lists here for metrics production
loss_list = return_lossList()
iteration_list = return_iterationList()

#lists for knowing classwise accuracy, used in Confusion Matrix
predictions_list = return_predictionsList()
labels_list = return_labelsList()

#********************************plot made here*********************************
plt.plot(iteration_list, loss_list)
plt.xlabel("No. of epochs")
plt.ylabel("Loss")
plt.title("Epochs vs. Loss")
plt.show()

#********************************printing the Confusion Matrix******************
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

confusion_matrix(labels_l, predictions_l)
print("\nClassification report for CNN:\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))

print("Number of training iterations: {}\n" .format(num_iterations))
