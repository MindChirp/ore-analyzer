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
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.fc1 = nn.Linear(in_features=128*56*56, out_features=500)
    self.fc2 = nn.Linear(in_features=500, out_features=8)

    self._initialize_fc()

    
  def _initialize_fc(self):
    dummy_input = torch.zeros(1, 3, 350, 350)
    dummy_output = self._forward_conv(dummy_input)
    n_size = dummy_output.view(1, -1).size(1)
    self.fc1 = nn.Linear(n_size, 500)
    self.fc2 = nn.Linear(500, 8)

    
  def _forward_conv(self, x):
    x = self.pool(self.relu1(self.conv1(x)))
    x = self.pool(self.relu2(self.conv2(x)))
    return x

  def forward(self, x):
    x = self._forward_conv(x)
    x = x.view(x.size(0), -1) # Flatten the tensor 
    x = F.relu(self.fc1(x)) 
    x = self.fc2(x)
    return x
  