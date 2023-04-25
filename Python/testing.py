#Authors: D. Joyner, T. Rana, T. Daniels

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from model import Fashion_Class_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1))])
#transform = transforms.Compose([transforms.ToTensor()]) #original transform value

test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transform)

batchSize = 100
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize)

#*************************variables for metrics production******************
labels_list = []
predictions_list = []
loss_list = []
iteration_list = []
accuracy = 0

model = Fashion_Class_Model()
model.to(device)
    
#************************Testing function**************************
def model_test(num_iterations, loss):
    
    if not (num_iterations % 600): #the same as "if num_epochs % 600 == 0"
        #overwrites the model object with the most recent weights from our "checkpoint" file
        model.load_state_dict(torch.load('C:/Users/Ender/.spyder-py3/fashion_mnist_cnn.ckpt'))
        
        total = 0
        correct = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)
            
            test = Variable(images.view(100, 3, 28, 28))
            
            outputs = model(test)
            
            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()
            
            total += len(labels)
        
        accuracy = correct * 100 / total
        loss_list.append(loss.data)
        iteration_list.append(num_iterations/600)
        
        print("Epoch: {}, Loss: {}, Accuracy: {}%" .format(num_iterations/600, loss.data, accuracy))
    

#*********************getters for training.py**************************
def return_labelsList():
    return labels_list

def return_lossList():
    return loss_list

def return_iterationList():
    return iteration_list

def return_predictionsList():
    return predictions_list
