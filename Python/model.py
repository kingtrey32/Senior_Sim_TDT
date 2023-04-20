#model provided by professor A. Choudhury
#modified by D. Joyner, T. Rana, T. Daniels


import torch
import torch.nn as nn
import torch.nn.functional as F


#*************************************************The CNN model***********************
class Fashion_Class_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):   
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)        
        x = F.leaky_relu(self.fc1(x))
        
        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        return x