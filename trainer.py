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

# Load the training set
train_dataset = torchvision.datasets.ImageFolder(root='train', transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(train_loader.dataset)

model = ImageNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
      inputs, labels = data

      inputs, labels = inputs.to(device), labels.to(device)

      # Convert labels to one-hot encoding
      labels_one_hot = torch.zeros(labels.size(0), 8).to(device)
      labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

      # Zero the pareaameters gradients
      optimizer.zero_grad()
      
      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, labels_one_hot)

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      # Print statistics
      running_loss += loss.item()
      if i % 100 == 99:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss/100:.4f}')
        running_loss = 0.0


torch.save(model.state_dict(), 'model.pth')
print('Finished Training')
