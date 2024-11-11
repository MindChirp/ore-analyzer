import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# for images, labels in train_loader:
#   print('Image batch dimensions:', images.shape)
#   print('Image label dimensions:' , labels.shape)

# use binary_cross_entropy_with_logits for multi-label classification

class ImageNetwork(nn.Module):
  def __init__(self):
    super(ImageNetwork, self).__init__()

    # Define the convolutional layers 
    # 1st layer: 3 input channels (RGB), 32 output channels, 3x3 square convolution
    # 2nd layer: 16 input channels, 32 output channels, 3x3 square convolution
    # 3rd layer: 500 input features, 8 output features (8 classes: coal_ore, diamond_ore, emerald_ore, gold_ore, iron_ore, lapis_ore, nether_quartz_ore, redstone_ore)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.fc1 = nn.Linear(in_features=64*56*56, out_features=500)
    self.fc2 = nn.Linear(in_features=500, out_features=8)

  def forward(self, x):
    x = self.pool(self.relu1(self.conv1(x)))
    x = self.pool(self.relu2(self.conv2(x)))
    x = x.view(-1, 64*56*56) # Flatten the tensor 
    x = F.relu(self.fc1(x)) 
    x = self.fc2(x)
    return x
  