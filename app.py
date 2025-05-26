import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from PIL import Image
from fusion_model import FusionModel
from wavelet_utils import extract_wavelet_features
from transformers import ViTForImageClassification  # keep if still using binary ViT
import gdown

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Define Class Labels
BINARY_LABELS = ["ALL", "Healthy"]
MULTICLASS_LABELS = ["Benign", "Early", "Pre", "Pro"]

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Models from 'model' Folder
import gdown

model_paths = {
    "ViT_Binary": "vit_binary_all_healthy.pth",
    "ViT_Multiclass": "vit_resnet_fusion_model.pth",
    "MobileNet": "mobilenet_xgboost.pkl",
    "VGG": "vgg16_svm.pkl",
    "ShuffleNet": "shufflenet_rf.pkl"
}

drive_links = {
    "ViT_Binary": "https://drive.google.com/uc?id=1hcl-kEyuAw8KQf0-Kk8Cmi8XiVJABgTY",
    "ViT_Multiclass": "https://drive.google.com/uc?id=1r4Mjn7j_UStJpdzRiNUFFilinS7ofWqC",
    "MobileNet": "https://drive.google.com/uc?id=1SE6PikSqMckgei-VT0JXcTjOZp0-dzyq",
    "VGG": "https://drive.google.com/uc?id=1pbEAM1U0qcQOv5gHcQmUnosrbl9MeTqp",
    "ShuffleNet": "https://drive.google.com/uc?id=11atr-AqGJcWmI_aOu441Gkc5mY4qYiyS"
}

# Download models if not already present
for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"📥 Downloading {name} model...")
        gdown.download(drive_links[name], path, quiet=False)

# ✅ Load ViT Models (Binary & Multiclass)
models = {}

# ✅ Load Binary Model for Detection
try:
    binary_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=2
    )
    binary_model.load_state_dict(torch.load(model_paths["ViT_Binary"], map_location=device), strict=False)
    binary_model.to(device)
    binary_model.eval()
    models["ViT_Binary"] = binary_model
except Exception as e:
    print(f"⚠ Error loading ViT_Binary model: {e}")

# ✅ Load Multiclass Model for Classification
# ✅ Load LeukVision Fusion Model
try:
    leukvision_model = FusionModel(num_classes=4, prompt_dim=128, wavelet_dim=4096).to(device)
    leukvision_model.load_state_dict(torch.load(model_paths["ViT_Multiclass"], map_location=device))
    leukvision_model.eval()
    models["LeukVision"] = leukvision_model
except Exception as e:
    print(f"⚠ Error loading LeukVision model: {e}")

# ✅ Load Other Models (MobileNet, VGG, ShuffleNet)
for model_name in ["MobileNet", "VGG", "ShuffleNet"]:
    try:
        models[model_name] = joblib.load(model_paths[model_name])  # Load sklearn models
    except Exception as e:
        print(f"⚠ Error loading {model_name}: {e}")

# ✅ Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Detection Page Route
@app.route('/detect_page', methods=['GET'])
def detect_page():
    return render_template('detect_page.html')

# ✅ Leukemia Information Page
@app.route('/leukemia')
def leukemia():
    return render_template('leukemia.html')

# ✅ Model Comparison Page
@app.route('/models')
def models_page():
    return render_template('models.html')

# ✅ Function to Save and Convert Image
def save_uploaded_image(file):
    image_path = "static/uploaded_image.png"
    image = Image.open(file).convert("RGB")
    image.save(image_path, "PNG")
    return image_path

# ✅ Function to Preprocess Image for Model
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        return image
    except Exception:
        return None

# ✅ Binary Classification Route (Detect)
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = save_uploaded_image(file)
    image = preprocess_image(image_path)
    if image is None:
        return jsonify({'error': 'Image processing failed. Try another image.'})

    with torch.no_grad():
        output = models["ViT_Binary"](image).logits
        predicted_label = torch.argmax(output, dim=1).cpu().item()

    return jsonify({
        'prediction': BINARY_LABELS[predicted_label],
        'image_url': "/" + image_path
    })

# ✅ Multiclass Classification Route (Classify)
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = save_uploaded_image(file)
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract Wavelet Features
    wavelet_feat = extract_wavelet_features(image).unsqueeze(0).to(device)

    # Dummy label input for prompt embedding
    dummy_label = torch.tensor([0]).to(device)

    with torch.no_grad():
        output = models["LeukVision"](image_tensor, wavelet_feat, dummy_label)
        predicted_label = torch.argmax(output, dim=1).cpu().item()

    return jsonify({
        'prediction': MULTICLASS_LABELS[predicted_label],
        'image_url': "/" + image_path
    })

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
