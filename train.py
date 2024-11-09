import json
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import websockets
import asyncio
from cnn_model import SimpleCNN

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

# Define the transformations, dataset, and dataloader for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

# Initialize the model, criterion, and optimizer
model = SimpleCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Placeholder for logs
logs = {"train_loss": [], "train_accuracy": [], "test_accuracy": []}

async def train(epoch, websocket):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(trainloader), 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(trainloader)
    average_accuracy = 100 * correct / total
    logs['train_loss'].append(average_loss)
    logs['train_accuracy'].append(average_accuracy)

    # Send log to WebSocket
    await websocket.send(json.dumps({
        "epoch": epoch + 1,
        "train_loss": average_loss,
        "train_accuracy": average_accuracy
    }))
    print(f"Epoch {epoch + 1}: Loss = {average_loss:.3f}, Accuracy = {average_accuracy:.2f}%")

async def test(epoch, websocket):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logs['test_accuracy'].append(accuracy)

    # Send log to WebSocket
    await websocket.send(json.dumps({
        "epoch": epoch + 1,
        "test_accuracy": accuracy
    }))
    print(f"Test Accuracy: {accuracy:.2f}%")

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        for epoch in range(10):
            await train(epoch, websocket)
            await test(epoch, websocket)
            # Save logs to file
            with open("logs.json", "w") as f:
                json.dump(logs, f)
            # Simulate real-time update
            await asyncio.sleep(1)
        print("Finished Training")
        # Save model
        torch.save(model.state_dict(), "cnn.pth")

if __name__ == "__main__":
    asyncio.run(main())
