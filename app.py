import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import gdown
from transformers import ViTForImageClassification
from fusion_model import FusionModel
from wavelet_utils import extract_wavelet_features

app = Flask(__name__)
CORS(app)  # ✅ Fix CORS issue

BINARY_LABELS = ["ALL", "Healthy"]
MULTICLASS_LABELS = ["Benign", "Early", "Pre", "Pro"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    "ViT_Binary": "vit_binary_all_healthy.pth",
    "ViT_Multiclass": "vit_resnet_fusion_model.pth"
}

drive_links = {
    "ViT_Binary": "https://drive.google.com/uc?id=1hcl-kEyuAw8KQf0-Kk8Cmi8XiVJABgTY",
    "ViT_Multiclass": "https://drive.google.com/uc?id=1r4Mjn7j_UStJpdzRiNUFFilinS7ofWqC"
}

for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"📥 Downloading {name} model...")
        gdown.download(drive_links[name], path, quiet=False)

models = {}

try:
    binary_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=2
    )
    binary_model.load_state_dict(torch.load(model_paths["ViT_Binary"], map_location=device), strict=False)
    binary_model.to(device).eval()
    models["ViT_Binary"] = binary_model
except Exception as e:
    print(f"⚠ Error loading ViT_Binary: {e}")

try:
    leukvision_model = FusionModel(num_classes=4, prompt_dim=128, wavelet_dim=4096).to(device)
    leukvision_model.load_state_dict(torch.load(model_paths["ViT_Multiclass"], map_location=device))
    leukvision_model.eval()
    models["LeukVision"] = leukvision_model
except Exception as e:
    print(f"⚠ Error loading LeukVision: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def save_uploaded_image(file):
    image_path = "static/uploaded_image.png"
    image = Image.open(file).convert("RGB")
    image.save(image_path)
    return image_path

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except:
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect_page')
def detect_page():
    return render_template("detect_page.html")

@app.route('/leukemia')
def leukemia():
    return render_template("leukemia.html")

@app.route('/models')
def models_page():
    return render_template("models.html")

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get("file")
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'})

    image_path = save_uploaded_image(file)
    image = preprocess_image(image_path)
    if image is None:
        return jsonify({'error': 'Image processing failed'})

    with torch.no_grad():
        logits = models["ViT_Binary"](image).logits
        prediction = torch.argmax(logits, dim=1).item()

    return jsonify({
        'prediction': BINARY_LABELS[prediction],
        'image_url': "/" + image_path
    })

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files.get("file")
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'})

    image_path = save_uploaded_image(file)
    img = Image.open(image_path).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(device)
    wavelet_feat = extract_wavelet_features(img).unsqueeze(0).to(device)
    dummy_label = torch.tensor([0]).to(device)

    with torch.no_grad():
        output = models["LeukVision"](image_tensor, wavelet_feat, dummy_label)
        prediction = torch.argmax(output, dim=1).item()

    return jsonify({
        'prediction': MULTICLASS_LABELS[prediction],
        'image_url': "/" + image_path
    })

if __name__ == '__main__':
    CORS(app)
