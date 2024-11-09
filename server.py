from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import json
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from jinja2 import Template
import random
from os.path import exists

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the training log server!"}

@app.get("/logs")
async def get_logs():
    with open("logs.json", "r") as f:
        logs = json.load(f)
    return JSONResponse(content=logs)

@app.get("/results")
async def get_results():
    from cnn_model import SimpleCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    if not exists("cnn.pth"):
        return JSONResponse(content={"error": "Model not trained yet!"})
    model.load_state_dict(torch.load())
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    images, labels, preds = [], [], []
    indices = random.sample(range(len(testset)), 10)

    for idx in indices:
        image, label = testset[idx]
        images.append(image.numpy().tolist())
        labels.append(testset.classes[label])
        image = torch.unsqueeze(image, 0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            preds.append(testset.classes[predicted.item()])

    response = {
        'images': images,
        'labels': labels,
        'predictions': preds
    }

    return JSONResponse(content=response)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

@app.get("/show_results", response_class=HTMLResponse)
async def show_results():
    content = """
    <html>
        <head>
            <title>Model Results</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Model Results on Randomly Selected Test Images</h1>
            <ul id="results-list"></ul>

            <h2>Training Logs</h2>
            <canvas id="trainingChart" width="600" height="400"></canvas>
            <script>
                // Fetch training logs
                fetch('/logs')
                    .then(response => response.json())
                    .then(data => {
                        const ctx = document.getElementById('trainingChart').getContext('2d');
                        const chart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: Array.from({length: data.train_loss.length}, (_, i) => i + 1),
                                datasets: [{
                                    label: 'Training Loss',
                                    borderColor: 'rgb(255, 99, 132)',
                                    data: data.train_loss
                                }, {
                                    label: 'Training Accuracy',
                                    borderColor: 'rgb(54, 162, 235)',
                                    data: data.train_accuracy,
                                    yAxisID: 'y1',
                                  }, {
                                    label: 'Test Accuracy',
                                    borderColor: 'rgb(75, 192, 192)',
                                    data: data.test_accuracy,
                                    yAxisID: 'y1',
                                }]
                            },
                            options: {
                                scales: {
                                    y: {
                                        type: 'linear',
                                        position: 'left',
                                    },
                                    y1: {
                                        type: 'linear',
                                        position: 'right',
                                        beginAtZero: true,
                                        min: 90,
                                        max: 100,
                                    }
                                }
                            }
                        });

                        // WebSocket setup for live updates
                        const ws = new WebSocket('ws://localhost:8000/ws');
                        ws.onmessage = function (event) {
                            const logEntry = JSON.parse(event.data);
                            
                            // Update chart with new data
                            chart.data.labels.push(logEntry.epoch);
                            chart.data.datasets[0].data.push(logEntry.train_loss);
                            chart.data.datasets[1].data.push(logEntry.train_accuracy);
                            if ('test_accuracy' in logEntry) {
                                chart.data.datasets[2].data.push(logEntry.test_accuracy);
                            }
                            chart.update();
                        };
                    });

                // Fetch and display test results
                fetch('/results')
                    .then(response => response.json())
                    .then(data => {
                        const resultsList = document.getElementById('results-list');
                        data.images.forEach((img, index) => {
                            const imgElement = document.createElement('img');
                            const imgArray = new Uint8Array(img).buffer;
                            const base64String = btoa(
                                new Uint8Array(imgArray).reduce(
                                    (data, byte) => data + String.fromCharCode(byte),
                                    ''
                                )
                            );
                            imgElement.src = `data:image/png;base64,${base64String}`;
                            imgElement.alt = 'CIFAR10 Image';
                            const li = document.createElement('li');
                            li.appendChild(imgElement);
                            const predText = document.createElement('p');
                            predText.textContent = `Prediction: ${data.predictions[index]}`;
                            li.appendChild(predText);
                            const labelText = document.createElement('p');
                            labelText.textContent = `Label: ${data.labels[index]}`;
                            li.appendChild(labelText);
                            resultsList.appendChild(li);
                        });
                    });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

