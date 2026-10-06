import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import os
import copy
import time

# Import your custom dataset logic
from utils import HYGDDataset, get_transforms

# --- SETTINGS ---
BASE_PATH = r"C:\Users\khaiw\Downloads\IDSC"
DATA_DIR = os.path.join(BASE_PATH, "Images")
CSV_PATH = os.path.join(BASE_PATH, "Labels.csv")
SAVE_MODEL_PATH = "efficientnet_glaucoma.pth"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10  # Changed to 10!
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_history(history):
    """Generates a graph of loss and accuracy over the 10 epochs."""
    os.makedirs('results', exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/training_history.png')
    print("📈 Training history graph saved to results/training_history.png")
    plt.close()


def train_model():
    print(f"🖥️ Using device: {DEVICE}")

    # 1. Load and Split Data
    df = pd.read_csv(CSV_PATH).dropna(axis=1, how='all')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient']))

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_tf, val_tf = get_transforms()

    train_ds = HYGDDataset(train_df, DATA_DIR, train_tf)
    val_ds = HYGDDataset(val_df, DATA_DIR, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📊 Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    # 2. Initialize Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(DEVICE)

    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop setup
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    since = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for imgs, labels, _ in dataloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()  # Convert to standard Python float

            # Record metrics for the graph
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\n🎉 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")

    # Load best weights and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"💾 Model saved to: {SAVE_MODEL_PATH}")

    # Plot the results!
    plot_history(history)


if __name__ == "__main__":
    train_model()