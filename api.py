from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from model import SimpleCNN
import torch

app = FastAPI()

class PredictRequest(BaseModel):
    images: list

@app.post("/predict/")
async def predict(request: PredictRequest):
    if not model:
        # Load the trained model
        model = SimpleCNN()
        model.load_state_dict(torch.load('./cifar10_cnn.pth'))
        model.eval()
    # Convert the request images to tensor and make prediction
    images_tensor = torch.tensor(request.images, dtype=torch.float32)
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs, 1)
    return {"predictions": predicted.numpy().tolist()}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <html>
    <head><title>Training Dashboard</title></head>
    <body>
        <h1>Training Dashboard</h1>
        <div id="metrics"></div>
        <script>
            const metricsDiv = document.getElementById('metrics');
            const ws = new WebSocket('ws://localhost:8000/ws');
            ws.onmessage = (event) => {
                const metrics = JSON.parse(event.data);
                const epoch = metrics.epoch;
                const train_loss = metrics.train_loss;
                const train_accuracy = metrics.train_accuracy;
                const test_accuracy = metrics.test_accuracy;
                
                metricsDiv.innerHTML += `<p>Epoch: ${epoch} | Train Loss: ${train_loss} | Train Accuracy: ${train_accuracy} | Test Accuracy: ${test_accuracy}</p>`;
            };

            ws.onopen = () => {
                console.log('WebSocket connection established');
            };

            ws.onclose = () => {
                console.log('WebSocket connection closed');
            };
        </script>
    </body>
    </html>
    """
    return html_content

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            await websocket.send_text(data)
        except WebSocketDisconnect:
            break
