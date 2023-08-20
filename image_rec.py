## image_rec.py
## Authors: Kyle Sung and Graeme McGaughey
## Description: ML file for Image Recognizer ML Software



import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision as tv
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from PIL import Image

import auxiliary as aux



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



def train_save_model(n_iterations, data_folder = "./fig_train"):

    # Set up data transformations
    transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # Create dataset and dataloader
    dataset = CustomDataset(data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

    # Initialize the network and loss function
    num_classes = len(os.listdir(data_folder))
    net = Net(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.0012, momentum=0.9) #OG: lr = 0.001

    # Training loop
    for epoch in range(n_iterations):  # Adjust the number of epochs as needed
        try:
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

            if running_loss / len(dataloader) < 0.1:
                # break
                pass

        except:
            break

    print("Finished Training")

    # Save the trained model's state dictionary
    torch.save(net.state_dict(), get_model())
    print(f"Model saved to {get_model()}")
    
    return



# Load the trained model and make predictions

def load_and_predict(image_path):

    net = Net(10)  # Assuming you know the number of classes
    net.load_state_dict(torch.load(get_model()))
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






def main(do_train_model):

    if do_train_model:
        train_save_model(n_iterations = 1000)

    correct = 0

    string = ""

    for i in range(10):
        for j in ["", "a", "b", "c"]:
            guess = load_and_predict(f"./fig_test/test{i}{j}.png")

            string += f"Real: {i}, Guess: {guess}\n"

            correct += int(i == guess)
        
        string += "\n"

    string += f"{correct} / 40\n"

    print(string)

    with open(f"{get_model()[:-4]}.txt", 'w') as file:
        file.write(string)




def get_model():
    return "models/model20.pth" ############### NOTE: PUT MODEL NAME HERE




if __name__ == "__main__":

    import time ; start = time.time()


    main(do_train_model = True)
    
    aux.play_sound()
    

    end = time.time() ; print("Time:", end - start)
