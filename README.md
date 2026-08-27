#  TRAIN.TUNE.COMPETE — Day 3: Architecture Tuning, Model Export & Docker Deployment

Welcome to the **Day 3 Self-Paced Guide** for the **TRAIN.TUNE.COMPETE** AI & ML Workshop, organized by the **Open Source Club at Saintgits College of Engineering**.

Over Days 1 and 2, we covered the theory of transfer learning, configured PyTorch data pipelines, and trained our first vision model. Today, we bridge the gap between AI research and software engineering: **taking a model out of a research notebook and packaging it into a production-ready microservice with FastAPI and Docker.**

---

## 📂 Repository File Breakdown

| File | Type | Description |
| :--- | :--- | :--- |
| **`day3.ipynb`** | Jupyter Notebook | The master notebook containing all step-by-step code, Markdown explanations, architecture comparisons, model training, API generation, and in-memory test runs. |
| **`app.py`** | Python Script | The standalone production **FastAPI** application that loads the trained model and serves predictions over HTTP endpoints (`/` and `/predict`). |
| **`Dockerfile`** | Container Spec | The blueprint to package our Python environment, lightweight CPU-only PyTorch, FastAPI server, and model weights into an isolated container image. |
| **`vision_model.pth`** | PyTorch Weights | The exported binary state dictionary (`state_dict`) containing the fine-tuned numerical parameters of our trained model. |
| **`README.md`** | Markdown Docs | Complete documentation, theoretical foundations, testing steps, and execution commands for this session. |

---

##  Part 1: Architecture Selection — Beyond ResNet-18

In Days 1 & 2, we used **ResNet-18**. In real-world software engineering, different hardware platforms require different backbone architectures:

* **ResNet-18 / ResNet-50:** General-purpose standard. Utilizes residual skip connections (y = F(x) + x) to allow gradients to flow backwards through deep layers without vanishing.
* **MobileNetV3:** Designed by Google specifically for edge devices, drones, and mobile applications. It replaces standard convolutions with depthwise separable convolutions, reducing computational floating-point operations (FLOPs) by nearly 80% with minimal loss in accuracy.
* **EfficientNet:** Uses compound scaling to balance model depth, width, and input resolution simultaneously for high leaderboard benchmark accuracy.

### Swapping to MobileNetV3 in PyTorch:
```python
import torch.nn as nn
from torchvision import models

# 1. Load pre-trained MobileNetV3
mobilenet = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.DEFAULT
)

# 2. Freeze all feature extraction layers
for param in mobilenet.parameters():
    param.requires_grad = False

# 3. In MobileNet, the final layer sits inside the 'classifier' container at index [3]
in_features = mobilenet.classifier[3].in_features
NUM_CLASSES = 3  # e.g., Healthy, Rust, Blight
mobilenet.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

print("Swapped MobileNet Head Successfully:")
print(mobilenet.classifier)
```

---

##  Part 2: Model Weight Serialization (.pth)

Never save an entire Python object using `pickle`, as changes in library versions will corrupt the model. Instead, save only the **learned parameter weights (the state dictionary)**:

```python
import torch

# Save only the learned parameters
torch.save(model.state_dict(), "vision_model.pth")
print("Model weights successfully saved to vision_model.pth!")
```

This creates a lightweight, portable binary file containing our fine-tuned weights.

---

##  Part 3: The Production Inference Server (app.py)

To allow mobile apps, web dashboards, or edge systems to interact with our model, we wrap it in a REST API using **FastAPI**.

```python
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(
    title="AI Vision Agent API",
    description="Production REST API for PyTorch Vision Models",
)

# 1. Target class labels
CLASSES = ["healthy", "rust", "blight"]

# 2. Re-instantiate architecture and load weights on CPU for lightweight inference
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("vision_model.pth", map_location="cpu"))
model.eval()

# 3. Standard inference preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


# Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "online", "model": "ResNet-18 Transfer Learning"}


# Image Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Transform to tensor shape: [1, 3, 224, 224]
    tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, 0)

    return {
        "prediction": CLASSES[pred_idx.item()],
        "confidence": f"{confidence.item() * 100:.2f}%",
    }
```

---

##  Part 4: In-Memory API Endpoint Testing

You can simulate real HTTP requests and multipart file uploads directly in Python using FastAPI's built-in `TestClient`:

```python
import json
import app as backend_service
from fastapi.testclient import TestClient

client = TestClient(backend_service.app)

# 1. Test GET / (Health Check)
health_res = client.get("/")
print("Health Check Response:", health_res.json())

# 2. Test POST /predict (Image Upload Simulation)
test_image_path = "data/val/rust/img_0.jpg"
with open(test_image_path, "rb") as f:
    predict_res = client.post(
        "/predict", files={"file": ("sample.jpg", f, "image/jpeg")}
    )

print("Prediction Status Code:", predict_res.status_code)
print("Prediction Response:\n", json.dumps(predict_res.json(), indent=2))
```

Expected JSON Output:
```json
{
  "prediction": "rust",
  "confidence": "96.42%"
}
```

---

##  Part 5: Containerization with Docker

Docker solves the classic *"works on my machine"* problem by packaging the Python runtime, CPU-only PyTorch, FastAPI server, and trained model weights into a reproducible, isolated image.

### The Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install lightweight CPU-only PyTorch and web server dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) \
    fastapi uvicorn pillow python-multipart httpx

# Copy application script and trained model weights
COPY app.py vision_model.pth ./

EXPOSE 8000

# Start production ASGI web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

##  Step-by-Step Local Execution Guide

### Option 1: Run the Complete Jupyter Notebook
Open **`Day3_Tuning_and_Deployment.ipynb`** directly in Google Colab or local Jupyter Notebook and execute the cells sequentially from top to bottom.

### Option 2: Run the FastAPI Server Directly
```bash
# 1. Install required packages
pip install torch torchvision fastapi uvicorn pillow python-multipart httpx

# 2. Start Uvicorn ASGI server
uvicorn app:app --reload --port 8000
```
Open **`http://localhost:8000/docs`** in your browser to test the interactive Swagger UI.

### Option 3: Build & Deploy via Docker
```bash
# 1. Build the container image
docker build -t vision-agent:v1 .

# 2. Run the container mapping port 8000
docker run -d -p 8000:8000 --name leaf-classifier vision-agent:v1

# 3. Test with cURL in your terminal
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_your_test_image.jpg"
```



*Open Source Club — Saintgits College of Engineering*
