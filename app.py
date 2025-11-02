from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import io
import os

# ==============================
# Config
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_efficientnet_b3.pth")



if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ==============================
# FastAPI + CORS setup
# ==============================
app = FastAPI(title="Crop Disease Detection API")

# ✅ Allow mobile, local React, and production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- allow all origins for mobile access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Data Transform
# ==============================
test_transforms = Compose([
    Resize(300, 300),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==============================
# Model Definition
# ==============================
class ImprovedEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet-b3", num_classes=None, dropout=0.5):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.classifier(features)

# ==============================
# Load Model Checkpoint
# ==============================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
NUM_CLASSES = checkpoint['num_classes']
CLASS_NAMES = checkpoint['class_names']

model = ImprovedEfficientNet("efficientnet-b3", NUM_CLASSES).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==============================
# Prediction Function
# ==============================
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        transformed = test_transforms(image=image_np)['image']
        input_tensor = transformed.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        pred_index = output.argmax(dim=1).item()
        return CLASS_NAMES[pred_index]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

# ==============================
# Routes
# ==============================
@app.get("/")
async def root():
    return {"message": "Crop Disease Detection API is running. Use /predict to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Use JPEG or PNG.")
    
    image_bytes = await file.read()
    pred_class = predict_image(image_bytes)

    # 🟩 Modify predicted class string (replace underscores and capitalize words)
    formatted_class = pred_class.replace("_", " ").title()

    return JSONResponse(content={"predicted_class": formatted_class})
