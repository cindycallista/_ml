import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"  
response = requests.get(url)
img = Image.open(BytesIO(response.content))

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

with torch.no_grad():
    out = model(batch_t)

LABELS_URL = "https://github.com/cindycallista/_ml/blob/main/mid/img.jpg"
labels_response = requests.get(LABELS_URL)
labels = labels_response.text.strip().split("\n")

_, index = torch.max(out, 1)
predicted_label = labels[index[0]]
print(f"Predicted label: {predicted_label}")
