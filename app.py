import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(
    title="AI Vision Classifier API",
    description="Production REST API for PyTorch Transfer Learning models",
)

CLASSES = ["healthy", "rust", "blight"]

# Initialize architecture and load exported weights on CPU
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("vision_model.pth", map_location="cpu"))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


@app.get("/")
def health_check():
  return {"status": "online", "model": "ResNet-18 Vision Agent"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  contents = await file.read()
  image = Image.open(io.BytesIO(contents)).convert("RGB")
  tensor = transform(image).unsqueeze(0)

  with torch.no_grad():
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    confidence, pred_idx = torch.max(probabilities, 0)

  return {
    "prediction": CLASSES[pred_idx.item()],
    "confidence": f"{confidence.item() * 100:.2f}%",
  }
