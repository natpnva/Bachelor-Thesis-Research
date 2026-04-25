import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# -------------------------------------------------------------------------------------------------------------------------------------

# CONFIGURATION

PARQUET_PATH = "/home/coder/project/gzd5_matched.parquet"
IMAGE_ROOT = "/tmp/images_flat/images_flat"
IMAGE_COL = "image_path"
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

TARGET_COLS = [
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "how-rounded_round_fraction",
    "how-rounded_in-between_fraction",
    "how-rounded_cigar-shaped_fraction",
    "disk-edge-on_yes_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "has-spiral-arms_yes_fraction",
]

NUM_TARGETS = len(TARGET_COLS)

# -------------------------------------------------------------------------------------------------------------------------------------

# DATASET

class GalaxyDataset(Dataset):
    def __init__(self, df, image_root, image_col, target_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.image_col = image_col
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row[self.image_col])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return image, targets

def split_dataset(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

# -------------------------------------------------------------------------------------------------------------------------------------

# TRANSFORM

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(180),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------------------------------------------------------------------------------------------------------------------------

# MODEL

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class GalaxyCNN(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_targets),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.head(x)

# -------------------------------------------------------------------------------------------------------------------------------------

# CLASSIFIER

def classify_galaxy(row, t_main=0.5, t_rare=0.35, min_confidence=0.2):
    smooth = row['smooth-or-featured_smooth_fraction']
    featured = row['smooth-or-featured_featured-or-disk_fraction']
    artifact = row['smooth-or-featured_artifact_fraction']
    edge_on = row['disk-edge-on_yes_fraction']
    spiral = row['has-spiral-arms_yes_fraction']
    bar_total = row['bar_strong_fraction'] + row['bar_weak_fraction']

    top_two = sorted([smooth, featured, artifact], reverse=True)
    if top_two[0] - top_two[1] < min_confidence:
        return 'other'

    if artifact > t_main:
        return 'other'
    if smooth > t_main:
        return 'elliptical'
    if featured > t_main:
        if edge_on > t_main:
            return 'spiral'
        if bar_total > t_main:
            return 'barred spiral'
        if spiral > t_main:
            return 'spiral'
        return 'other'
    return 'other'

# -------------------------------------------------------------------------------------------------------------------------------------

# EVALUATION

def evaluate_labels(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            preds = model(images).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pred_df = pd.DataFrame(all_preds, columns=TARGET_COLS)
    target_df = pd.DataFrame(all_targets, columns=TARGET_COLS)

    pred_labels = pred_df.apply(classify_galaxy, axis=1)
    target_labels = target_df.apply(classify_galaxy, axis=1)

    accuracy = (pred_labels == target_labels).mean()

    class_accuracy = {}
    for cls in target_labels.unique():
        mask = target_labels == cls
        if mask.sum() > 0:
            class_accuracy[cls] = (pred_labels[mask] == cls).mean()

    pred_dist = pred_labels.value_counts(normalize=True)
    target_dist = target_labels.value_counts(normalize=True)

    type_order = ['elliptical', 'spiral', 'barred spiral', 'other']
    cm = confusion_matrix(target_labels, pred_labels, labels=type_order)
    report = classification_report(target_labels, pred_labels, labels=type_order)
    f1 = f1_score(target_labels, pred_labels, labels=type_order, average='weighted')
    precision = precision_score(target_labels, pred_labels, labels=type_order, average='weighted')
    recall = recall_score(target_labels, pred_labels, labels=type_order, average='weighted')

    return {
        "Overall accuracy": accuracy,
        "Accuracy per class": class_accuracy,
        "Predicted distribution": pred_dist,
        "True distribution": target_dist,
        "Confusion matrix": cm,
        "Classification report": report,
        "F1 score": f1,
        "Precision": precision,
        "Recall": recall,
    }

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted label',
        ylabel='True label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.savefig('/home/coder/project/confusion_matrix_test.png')
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------

# MAIN

def main():
    print(f"\nLoading data from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {len(df)}")
    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda x: os.path.basename(x))

    _, _, test_df = split_dataset(df)

    test_ds = GalaxyDataset(test_df, IMAGE_ROOT, IMAGE_COL, TARGET_COLS, val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    print(f"Test: {len(test_df)}")

    model = GalaxyCNN(NUM_TARGETS).to(DEVICE)
    model.load_state_dict(torch.load("best_galaxy_cnn.pt", map_location=DEVICE))

    metrics = evaluate_labels(model, test_loader)

    print(f"\nFinal accuracy: {metrics['Overall accuracy']:.2%}")
    print(f"F1 score: {metrics['F1 score']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print("\nPer-class accuracy:")
    for cls, acc in metrics["Accuracy per class"].items():
        print(f"  {cls}: {acc:.2%}")
    print("\nClassification report:")
    print(metrics["Classification report"])
    print("\nTrue distribution:")
    print(metrics["True distribution"])
    print("\nPredicted distribution:")
    print(metrics["Predicted distribution"])

    plot_confusion_matrix(metrics["Confusion matrix"], ['elliptical', 'spiral', 'barred spiral', 'other'])


if __name__ == "__main__":
    main()