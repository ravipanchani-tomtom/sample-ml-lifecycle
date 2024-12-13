# Fashion MNIST Training with Live Metrics

This project demonstrates training a SimpleCNN model on the Fashion MNIST dataset with live training metrics displayed on a web UI. The application uses FastAPI for the backend and WebSocket for real-time communication.

## Features

- Train a SimpleCNN model on the Fashion MNIST dataset.
- Display live training metrics (loss and accuracy) on a web UI.
- Start and stop training from the web UI.
- Save the best model checkpoints during training.
- Resume training from the last checkpoint.
- Allow multiple training sessions in parallel.
- Test the model on random images from the test set and display the results.
- Provide a summary of experiments with the best training and test accuracies.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Torch
- Torchvision
- Tqdm
- Pillow
- Chart.js (for the frontend)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ravipanchani-tomtom/sample-ml-lifecycle.git
    cd sample-ml-lifecycle
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install fastapi uvicorn torch torchvision tqdm pillow
    ```

## Usage

1. Run the FastAPI application:
    ```sh
    uvicorn train:app --reload
    ```

2. Open your web browser and navigate to `http://localhost:8000`.

3. Use the "Start Training" and "Stop Training" buttons to control the training process. The training metrics (loss and accuracy) will be displayed on a single chart with two different colored lines.

4. After training, you can test the model on random images from the test set and view the results.

5. View the summary of experiments to see the best training and test accuracies for each model.

## File Structure

- `train.py`: The main script that sets up the FastAPI application, handles WebSocket connections, and manages the training process.
- `model.py`: Defines the SimpleCNN model.
- `templates/index.html`: The frontend HTML file that includes the Chart.js library and WebSocket logic.

## License

This project is licensed under the MIT License.