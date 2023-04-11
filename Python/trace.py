import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model import Fashion_Class_Model

# Load the trained model
model = Fashion_Class_Model()
model.load_state_dict(torch.load('fashion_mnist_cnn.ckpt'))

# Convert the model to a traced form
example_input = torch.randn(1, 1, 28, 28)
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save('fashion_mnist_cnn_traced.pt')
