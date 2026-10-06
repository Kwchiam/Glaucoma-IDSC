import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class HYGDDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'Image Name']
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = 1 if self.dataframe.loc[idx, 'Label'] == 'GON+' else 0
        # Quality Score weighting (Series 2 insight)
        weight = torch.tensor(self.dataframe.loc[idx, 'Quality Score'] / 10.0, dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long), weight


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def plot_bootcamp_results(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Weighted Loss')
    plt.title('Series 6: Training Progress')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Val Accuracy', color='green')
    plt.title('Series 7: Model Performance')
    plt.legend()
    plt.savefig('bootcamp_results.png')
    plt.show()