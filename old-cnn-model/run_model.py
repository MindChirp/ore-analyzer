from model import ImageNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision



# Define transforms for the training set. This is important because minimizing the image sizes helps speed up training.
transforms = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
])



model = ImageNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

test_dataset = torchvision.datasets.ImageFolder(root='test', transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
  for inputs, labels in test_loader:
    
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    
    # Print the results
    print(f'Predicted: {predicted.cpu().numpy()}')
    print(f'Actual:    {labels.cpu().numpy()}')