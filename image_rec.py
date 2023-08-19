import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision as tv
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from PIL import Image




# Define the CNN architecture
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Flatten the tensor
        x = self.fc1(x)
        return x



# Define dataset and dataloader
class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.classes = sorted(os.listdir(data_folder))
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(data_folder, class_name)
            class_image_paths = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
            self.image_paths.extend(class_image_paths)
            self.labels.extend([label] * len(class_image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)  # Open the image with PIL

        if self.transform:
            image = self.transform(image)  # Apply the transformation

        return image, label



def train_save_model(n_iterations, data_folder = "./fig", model_path="model.pth"):

    # Set up data transformations
    transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # Create dataset and dataloader
    dataset = CustomDataset(data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the network and loss function
    num_classes = len(os.listdir(data_folder))
    net = Net(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(n_iterations):  # Adjust the number of epochs as needed
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

    print("Finished Training")

    # Save the trained model's state dictionary
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return



# Load the trained model and make predictions
def load_and_predict(image_path, model_path="model.pth"):
    net = Net(10)  # Assuming you know the number of classes
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode

    transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = net(image_tensor)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class = predicted.item()
    return predicted_class


if __name__ == "__main__":
    train_save_model(100)
    load_and_predict("./test/test1.png")