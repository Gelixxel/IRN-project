import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from importData import load_galaxy_data
import numpy as np  

# Define a custom Dataset class
class GalaxyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return {"image": sample["image"], "label": sample["label"]}

# Galaxy Classifier model
class GalaxyClassifier(nn.Module):
    def __init__(self):
        super(GalaxyClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2)

        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.calculate_to_linear()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)  # Adjust this if input image size changes
        self.fc2 = nn.Linear(128, 10)

    def calculate_to_linear(self):
        # Run a dummy forward pass to calculate the size
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 64)
            x = self.convs(x)
            self._to_linear = x.view(x.size(0), -1).size(1)

    def convs(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flattening while keeping batch size
        print(f"Flattened size: {x.size(1)}")  # Debug print
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Preprocess data function
def preprocess_data(example):
    image = np.array(example["image"]).astype(np.float32) / 255.0  # Convert to NumPy array and normalize
    image = torch.tensor(image).permute(2, 0, 1)  # Convert to tensor and change dimensions to [C, H, W]
    label = torch.tensor(example["label"]).long()  # Ensure label is of type long
    return {"image": image, "label": label}

print("Loading Dataset...")
# Load the dataset from Hugging Face
dataset = load_galaxy_data()
print(dataset)

# Transform data
train_data = GalaxyDataset(dataset['train'], transform=preprocess_data)
validation_data = GalaxyDataset(dataset['test'], transform=preprocess_data)

print("DataLoad...")
# DataLoader
BATCH_SIZE = 100
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

# Model, criterion, optimizer
model = GalaxyClassifier()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 50

print("Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {i+1}/{len(train_loader)} | Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

print("Eval...")
# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in validation_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the network on the validation images: {accuracy:.2f}%")