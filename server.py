from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import json
import torch
from torchvision import transforms
from PIL import Image
import random

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
    from torchvision.datasets import CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn.pth"))
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
        images.append(image)
        labels.append(testset.classes[label])
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            preds.append(testset.classes[predicted.item()])

    response = {
        'images': [image.numpy().tolist() for image in images],
        'labels': labels,
        'predictions': preds
    }

    return JSONResponse(content=response)

@app.get("/show_results", response_class=HTMLResponse)
async def show_results():
    content = """
    <html>
        <head>
            <title>Model Results</title>
        </head>
        <body>
            <h1>Model Results on Randomly Selected Test Images</h1>
            <ul>
                {% for img, pred, label in images %}
                <li>
                    <img src="data:image/png;base64,{{ img }}" alt="CIFAR10 Image">
                    <p>Prediction: {{ pred }}</p>
                    <p>Label: {{ label }}</p>
                </li>
                {% endfor %}
            </ul>
        </body>
    </html>
    """
    response = await get_results()
    results = response.body

    from jinja2 import Template
    template = Template(content)
    html_content = template.render(images=results["images"], preds=results["predictions"], labels=results["labels"])

    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
