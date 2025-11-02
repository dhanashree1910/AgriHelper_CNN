import sys
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os

# ==============================
# Config
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\Engineering\Second Year\EDI-1\model_Edi\model_Edi\checkpoints\best_efficientnet_b3.pth"  # Path to your saved model

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure the file is in the 'checkpoints' folder.")
    input("Press Enter to exit...")
    exit()

# ==============================
# Get image path from command-line
# ==============================
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    print("Usage: python model_test.py <path_to_image>")
    exit()

# Check if image file exists
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at {IMAGE_PATH}")
    exit()

# ==============================
# Data Transform
# ==============================
test_transforms = Compose([
    Resize(300, 300),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==============================
# Load Model
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

# Load checkpoint
print(f"Loading model from {MODEL_PATH}...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    NUM_CLASSES = checkpoint['num_classes']
    CLASS_NAMES = checkpoint['class_names']

    model = ImprovedEfficientNet("efficientnet-b3", NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==============================
# Prediction Function
# ==============================
def predict_image(image_path, model, transforms):
    try:
        original_image = np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        print(f"Error opening image file: {e}")
        return None

    image = transforms(image=original_image)['image']
    input_tensor = image.unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        
    pred_index = output.argmax(dim=1).item()
    return CLASS_NAMES[pred_index]

# ==============================
# Predict
# ==============================
print(f"\nAnalyzing image: {IMAGE_PATH}...")
pred_class = predict_image(IMAGE_PATH, model, test_transforms)

if pred_class:
    # Modify the string: replace underscore with space and capitalize first letter
    formatted_class = pred_class.replace("_", " ").capitalize()

    print("\n--- Prediction Result ---")
    print(f"🏷  Predicted Class: {formatted_class}")
    print("-------------------------")
