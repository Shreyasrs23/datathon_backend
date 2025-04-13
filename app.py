from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import os

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XceptionDualHead(nn.Module):
    def __init__(self):
        super(XceptionDualHead, self).__init__()
        self.base = timm.create_model('xception', pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(0.3)
        in_features = self.base.num_features
        self.fc_binary = nn.Linear(in_features, 1)
        self.fc_class = nn.Linear(in_features, 3)

    def forward(self, x):
        feats = self.base(x)
        x = self.dropout(feats)
        return self.fc_binary(x).squeeze(1), self.fc_class(x)

model = XceptionDualHead().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

label_map = {0: "Animal", 1: "Human", 2: "Vehicle"}

@app.route("/", methods=["GET"])
def home():
    return "🧠 Deepfake Detector API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out_bin, out_cls = model(tensor)
        real_or_fake = "Real" if torch.sigmoid(out_bin).item() > 0.5 else "Fake"
        category = label_map[out_cls.argmax(dim=1).item()]

    return jsonify({
        "real_or_fake": real_or_fake,
        "category": category
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host="0.0.0.0", port=port)
