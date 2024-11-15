import json
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from fastapi import FastAPI, WebSocket, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from model import SimpleCNN
import asyncio
from pydantic import BaseModel

app = FastAPI()

class TrainingParams(BaseModel):
    learning_rate: float
    optimizer: str
    epochs: int

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

clients = []
training_task = None
stop_training = False

losses = []
accuracies = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "start_training":
                await start_training()
            elif data == "stop_training":
                handle_stop_training()
    except WebSocketDisconnect:
        clients.remove(websocket)

@app.post("/start_training")
async def start_training_endpoint(params: TrainingParams, background_tasks: BackgroundTasks):
    global optimizer
    if params.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    elif params.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
    background_tasks.add_task(start_training, params.epochs)

@app.post("/stop_training")
async def stop_training_endpoint():
    handle_stop_training()

@app.post("/predict")
async def predict_endpoint(request: Request):
    data = await request.json()
    image = torch.tensor(data['image']).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return {"prediction": predicted.item()}

async def start_training(epochs: int):
    global training_task, stop_training
    stop_training = False
    if training_task is None or training_task.done():
        training_task = asyncio.create_task(train_model(epochs))

def handle_stop_training():
    global stop_training
    stop_training = True

async def train_model(epochs: int):
    global stop_training
    model.to(device)  # Ensure the model is moved to the device
    for epoch in range(epochs):
        if stop_training:
            break
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for i, data in enumerate(trainloader):
            if stop_training:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * running_corrects / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        for client in clients:
            await client.send_json({'loss': losses, 'accuracy': accuracies})
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    if not stop_training:
        torch.save(model.state_dict(), "cnn.pth")
        for client in clients:
            await client.send_json({'message': 'training_complete'})

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")

    # Define the transformations, dataset, and dataloader for Fashion MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)