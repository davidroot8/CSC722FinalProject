import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

model.to('cuda')
model.eval()

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Load the dataset

current_directory = os.getcwd()
pathOS = os.path.join(current_directory, 'SSD_ResNet_Pytorch', 'archive', 'imagenetMini')
train_dataset = torchvision.datasets.ImageNet(root=pathOS, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Create SSD model with ResNet backbone
#model = ssd300_resnet50(pretrained_backbone=True, num_classes=len(train_dataset.classes))

# Define the loss function (you may need to replace this with the appropriate loss function for your dataset)
loss_function = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = loss_function(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'ssd_model.pth')