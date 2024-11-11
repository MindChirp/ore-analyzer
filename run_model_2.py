from model import ImageNetwork
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

type_name = "diamond_ore"
directory_path = f'./test/{type_name}'

name_map = [
  "coal",
  "diamond",
  "emerald",
  "gold",
  "iron",
  "lapis",
  "nether_gold_ore",
  "redstone_ore"
]

# Load the trained model
model = ImageNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Iterate over all images in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        image_path = os.path.join(directory_path, filename)
        
        # Load the image
        image = Image.open(image_path)
        
        # Apply the transformations
        image = transform(image)
        
        # Add a batch dimension
        image = image.unsqueeze(0)
        
        # Move the image to the GPU if available
        image = image.to(device)
        
        # Run the image through the model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        # Print the prediction
        print(f'Image: {filename}, Predicted class: {name_map[predicted.item()]}')