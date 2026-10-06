import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===============================
# SETTINGS
# ===============================
BASE_PATH = r"C:\Users\khaiw\Downloads\IDSC"
MODEL_PATH = r"C:\Users\khaiw\Downloads\IDSC\efficientnet_glaucoma.pth"
CSV_PATH = os.path.join(BASE_PATH, "Labels.csv")
IMAGE_FOLDER = os.path.join(BASE_PATH, "Images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 1. LOAD TRAINED MODEL
# ===============================
print("🚀 Loading PyTorch model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ===============================
# 2. PREPARE DATASET & DATALOADER
# ===============================
class VisualizationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.valid_data = []

        # Check if images exist to avoid crashes
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row["Image Name"])
            if os.path.exists(img_path):
                # Dynamically handle either label format
                if "label_numeric" in row:
                    label = row["label_numeric"]
                else:
                    label = 1 if row.get("Label", "") == 'GON+' else 0

                self.valid_data.append((row["Image Name"], label))
            else:
                print(f"Missing image: {img_path}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_name, label = self.valid_data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # We return img_name so we know exactly which picture to plot later
        return image, label, img_name


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = VisualizationDataset(CSV_PATH, IMAGE_FOLDER, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Total test images loaded: {len(test_dataset)}")

# ===============================
# 3. GENERATE PREDICTIONS
# ===============================
all_probs = []
all_preds = []
all_labels = []
valid_image_names = []

print("⏳ Running inference...")
with torch.no_grad():
    for images, labels, img_names in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)

        # PyTorch outputs raw scores. We need Softmax to turn them into % probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Extract the probability specifically for Class 1 (Glaucoma)
        glaucoma_probs = probs[:, 1].cpu().numpy()
        _, predicted_classes = torch.max(outputs, 1)

        all_probs.extend(glaucoma_probs)
        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.numpy())
        valid_image_names.extend(img_names)

# Convert lists to numpy arrays for easier math later
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ===============================
# 4. LABEL MAPPING & INTERPRETATION
# ===============================
label_map = {1: "Glaucoma", 0: "Normal"}


def interpret_probability(prob):
    if prob >= 0.95:
        return "Very confident glaucoma"
    elif prob >= 0.70:
        return "Moderate confidence"
    elif prob >= 0.52:
        return "Uncertain prediction"
    else:
        return "Low probability of glaucoma"


# Helper function to plot any image by its index
def plot_image(idx):
    image_path = os.path.join(IMAGE_FOLDER, valid_image_names[idx])
    img = Image.open(image_path)  # Open original un-normalized image for display

    actual = all_labels[idx]
    predicted = all_preds[idx]
    probability = all_probs[idx]

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(
        f"Actual: {label_map[actual]} | Predicted: {label_map[predicted]}\n"
        f"Prob: {probability:.2f} | {interpret_probability(probability)}"
    )
    plt.axis("off")
    plt.show()


# ===============================
# 5. DISPLAY FIRST 10 PREDICTIONS
# ===============================
print("\nDisplaying first 10 predictions...")
num_display = min(10, len(valid_image_names))
for i in range(num_display):
    plot_image(i)

# ===============================
# 6. IDENTIFY INCORRECT PREDICTIONS
# ===============================
incorrect_indices = np.where(all_preds != all_labels)[0]
print(f"\nNumber of incorrect predictions: {len(incorrect_indices)}")

# ===============================
# 7. VISUALIZE FIRST 5 INCORRECT PREDICTIONS
# ===============================
print("\nDisplaying first 5 incorrect predictions...")
num_incorrect_display = min(5, len(incorrect_indices))
for idx in incorrect_indices[:num_incorrect_display]:
    plot_image(idx)

# ===============================
# 8. SHOW 5 RANDOM PREDICTIONS
# ===============================
print("\nDisplaying 5 random predictions...")
random_count = min(5, len(valid_image_names))
random_indices = random.sample(range(len(valid_image_names)), random_count)
for idx in random_indices:
    plot_image(idx)