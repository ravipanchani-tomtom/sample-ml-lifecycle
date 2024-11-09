import json
import os
import time
from cnn_model import SimpleCNN, trainloader
import torch.optim as optim
import torch.nn as nn
import torch

# Model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Placeholder for logs
logs = {"train_loss": []}

def train(epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            logs['train_loss'].append(running_loss / 100)
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        # Save logs to file
        with open("logs.json", "w") as f:
            json.dump(logs, f)
        # Simulate real-time update
        time.sleep(1)
    print("Finished Training")

    # Save model
    torch.save(model.state_dict(), "cnn.pth")
