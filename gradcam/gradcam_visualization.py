import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

model = models.resnet50()
model.fc = torch.nn.Linear(2048, 1)
model.load_state_dict(torch.load("models/cnn_price_model.pth", map_location=device))
model.eval().to(device)


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# GRADCAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self):
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].cpu().detach().numpy()
        cam = cv2.resize(cam, (224,224))
        cam = cam / cam.max()
        return cam


# RUN ON ONE IMAGE

img_path = "sat_images/0.png"

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

target_layer = model.layer4
gradcam = GradCAM(model, target_layer)

output = model(input_tensor)
output.backward()

cam = gradcam.generate()


# VISUALIZE

img_np = np.array(img.resize((224,224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM: Influential Regions")
plt.show()
