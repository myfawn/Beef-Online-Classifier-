from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as transforms
import os

app = Flask(__name__, template_folder='templates')

classes = ["Choice", "Prime", "Select"]

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))

model.load_state_dict(torch.load("best_model_20250624.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_t)
            pred = torch.argmax(output, 1).item()
            prediction = f"Predicted class: {classes[pred]}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
