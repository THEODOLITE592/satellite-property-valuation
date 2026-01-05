import os
import torch
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# CONFIG

CSV_PATH = "Data/train(1)(train(1)).csv"
IMAGE_DIR = "sat_images"
MODEL_PATH = "models/cnn_price_model.pth"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# TRANSFORMS
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================
# DATASET
# ============================
class HouseImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.valid_indices = []

        # Pre-check images
        for idx in range(len(df)):
            img_path = f"{IMAGE_DIR}/{idx}.png"
            try:
                Image.open(img_path).verify()
                self.valid_indices.append(idx)
            except:
                pass

        print(f"Valid images: {len(self.valid_indices)} / {len(df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        img_path = f"{IMAGE_DIR}/{idx}.png"

        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        price = self.df.iloc[idx]["price"]
        return img, torch.tensor(price, dtype=torch.float32)


# ============================
# LOAD DATA
# ============================
df = pd.read_csv(CSV_PATH)
df = df.reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = HouseImageDataset(train_df)
val_dataset = HouseImageDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ============================
# MODEL
# ============================
cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
cnn.fc = nn.Linear(2048, 1)  # regression head
cnn = cnn.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# ============================
# TRAIN
# ============================
for epoch in range(EPOCHS):
    cnn.train()
    total_loss = 0

    for imgs, prices in train_loader:
        imgs, prices = imgs.to(device), prices.to(device)

        optimizer.zero_grad()
        preds = cnn(imgs).squeeze()
        loss = criterion(preds, prices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(cnn.state_dict(), MODEL_PATH)
print("✅ CNN model saved:", MODEL_PATH)
