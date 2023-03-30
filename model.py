# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:11:45 2023

@author: Ender
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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



#provides a label, or Class, to the object in question
#not used here
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