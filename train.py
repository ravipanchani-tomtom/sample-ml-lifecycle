from model import SimpleCNN
from data import get_data_loaders
import torch.optim as optim
import torch
import torch.nn as nn
import asyncio
import websockets
import json

async def send_metrics(uri, metrics):
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(metrics))

def train_model(epochs=10, batch_size=32, learning_rate=0.001, websocket_uri="ws://localhost:8000/ws"):
    trainloader, testloader, classes = get_data_loaders(batch_size=batch_size)
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()

            # Calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(trainloader)
        train_accuracy = 100 * train_correct / train_total

        # Evaluate on test data
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total

        metrics = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        
        # Send metrics to WebSocket server
        asyncio.run(send_metrics(websocket_uri, metrics))

    print('Finished Training')
    torch.save(net.state_dict(), './cifar10_cnn.pth')

if __name__ == "__main__":
    train_model()
