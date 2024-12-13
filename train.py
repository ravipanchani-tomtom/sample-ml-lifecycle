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
import random
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

class TrainingParams(BaseModel):
    learning_rate: float
    optimizer: str
    epochs: int
    resume: bool = False

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

clients = []
training_tasks = {}
stop_training_flags = {}

losses = {}
accuracies = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start_training"):
                _, model_id = data.split(":")
                await start_training(model_id)
            elif data.startswith("stop_training"):
                _, model_id = data.split(":")
                handle_stop_training(model_id)
    except WebSocketDisconnect:
        clients.remove(websocket)

@app.post("/start_training")
async def start_training_endpoint(params: TrainingParams, background_tasks: BackgroundTasks):
    model_id = str(time.time())
    model = SimpleCNN().to(device)
    if params.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    elif params.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
    background_tasks.add_task(start_training, model_id, model, optimizer, params.epochs, params.resume)
    return {"model_id": model_id}

@app.post("/stop_training")
async def stop_training_endpoint(request: Request):
    data = await request.json()
    model_id = data['model_id']
    handle_stop_training(model_id)

@app.post("/test_model")
async def test_model_endpoint(request: Request):
    data = await request.json()
    model_id = data['model_id']
    weights_type = data['weights_type']
    return await test_model(model_id, weights_type)

async def start_training(model_id: str, model: SimpleCNN, optimizer, epochs: int, resume: bool):
    global training_tasks, stop_training_flags
    stop_training_flags[model_id] = False
    if model_id not in training_tasks or training_tasks[model_id].done():
        training_tasks[model_id] = asyncio.create_task(train_model(model_id, model, optimizer, epochs, resume))

def handle_stop_training(model_id: str):
    global stop_training_flags
    stop_training_flags[model_id] = True

async def train_model(model_id: str, model: SimpleCNN, optimizer, epochs: int, resume: bool):
    global stop_training_flags
    best_accuracy = 0.0
    start_epoch = 0

    if resume and os.path.exists(f"{model_id}_checkpoint.pth"):
        checkpoint = torch.load(f"{model_id}_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

    for epoch in range(start_epoch, epochs):
        if stop_training_flags[model_id]:
            break
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for i, data in enumerate(trainloader):
            if stop_training_flags[model_id]:
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
        if model_id not in losses:
            losses[model_id] = []
            accuracies[model_id] = []
        losses[model_id].append(epoch_loss)
        accuracies[model_id].append(epoch_acc)
        for client in clients:
            await client.send_json({'model_id': model_id, 'loss': losses[model_id], 'accuracy': accuracies[model_id]})
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, f"{model_id}_checkpoint.pth")

    if not stop_training_flags[model_id]:
        torch.save(model.state_dict(), f"{model_id}_final.pth")
        for client in clients:
            await client.send_json({'model_id': model_id, 'message': 'training_complete'})

def unnormalize_image(image: torch.Tensor):
    image = ((image.numpy() * 0.5) + 0.5) * 255
    return image

def array_to_base64(img_array):
    pil_img = Image.fromarray(img_array.reshape(28,28)).convert('RGB')
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_byte = buffer.read()
    base64_str = base64.b64encode(img_byte)
    return base64_str

async def test_model(model_id: str, weights_type: str):
    model = SimpleCNN().to(device)
    if weights_type == "best":
        checkpoint = torch.load(f"{model_id}_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(f"{model_id}_final.pth"))
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    images, labels, preds = [], [], []
    indices = random.sample(range(len(testset)), 10)
    for idx in indices:
        image, label = testset[idx]
        images.append(image)
        labels.append(testset.classes[label])
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            preds.append(testset.classes[predicted.item()])
    response = {
        'images': [array_to_base64(unnormalize_image(image)) for image in images],
        'labels': labels,
        'predictions': preds
    }
    return response

@app.get("/experiments_summary")
async def experiments_summary():
    summary = []
    for model_id in losses.keys():
        best_train_acc = max(accuracies[model_id])
        best_test_acc = 0.0
        if os.path.exists(f"{model_id}_checkpoint.pth"):
            checkpoint = torch.load(f"{model_id}_checkpoint.pth")
            best_test_acc = checkpoint['best_accuracy']
        summary.append({
            'model_id': model_id,
            'best_train_accuracy': best_train_acc,
            'best_test_accuracy': best_test_acc
        })
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)