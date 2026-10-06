import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
# Make sure this points to the PyTorch model you trained earlier!
BASE_PATH = r"C:\Users\khaiw\Downloads\IDSC"
MODEL_PATH = r"C:\Users\khaiw\Downloads\IDSC\efficientnet_glaucoma.pth"
CSV_PATH = os.path.join(BASE_PATH, "Labels.csv")
IMAGE_FOLDER = os.path.join(BASE_PATH, "Images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Define Dataset ---
class TestDataset(Dataset):
    """PyTorch Dataset to efficiently load images from your CSV"""

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the image path
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["Image Name"])
        image = Image.open(img_path).convert("RGB")

        # 2. THE FIX: Read from the 'Label' column and convert 'GON+' to 1, everything else to 0
        label_text = self.df.iloc[idx]["Label"]
        label = 1 if label_text == 'GON+' else 0

        # 3. Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# --- 2. Image Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Load Trained Model ---
print("🚀 Loading PyTorch model...")
# (Assuming you are using the EfficientNet model we built earlier)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- 4. Setup DataLoader ---
test_dataset = TestDataset(csv_file=CSV_PATH, img_dir=IMAGE_FOLDER, transform=transform)
# DataLoader processes images in batches (e.g., 16 at a time) instead of 1 by 1, making it much faster
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 5. Generate Predictions ---
y_true = []
y_pred = []

print(f"⏳ Running inference on {len(test_dataset)} images...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        # Get model predictions
        outputs = model(images)
        _, predicted_classes = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes.cpu().numpy())

# --- 6. Print Metrics ---
cm = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Glaucoma']))

# Optional: Make the confusion matrix look pretty
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Glaucoma'],
            yticklabels=['Normal', 'Glaucoma'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()