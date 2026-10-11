"""
Train the VGG16-based brain tumor classifier (PyTorch port of the reference notebook).

Expected data layout (Kaggle "Brain Tumor MRI Dataset"):
    data/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    data/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

Usage:
    python train.py                      # defaults: 5 epochs, batch 20, lr 1e-4
    python train.py --epochs 10 --batch-size 32

Outputs:
    models/model.pt          trained weights + class names
    outputs/training_history.png
    outputs/confusion_matrix.png
    outputs/roc_curve.png
    outputs/classification_report.txt
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless; plots are saved to files
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import IMAGE_SIZE, TumorClassifier, get_device, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="data/Training")
    p.add_argument("--test-dir", default="data/Testing")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/model.pt")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(args):
    # Original notebook: resize to 128x128, random brightness/contrast in [0.8, 1.2], scale to [0, 1].
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),  # -> float in [0, 1]
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(args.test_dir, transform=test_tf)
    assert train_ds.classes == test_ds.classes, "Train/test class folders differ"

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers)
    return train_dl, test_dl, train_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        # live running loss / accuracy on the progress bar
        bar.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return running_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    """Returns (true labels, predicted probabilities) over the whole loader."""
    model.eval()
    ys, probs = [], []
    for x, y in tqdm(loader, desc="  eval", leave=False):
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def plot_history(history, out_dir):
    epochs = range(1, len(history["acc"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.grid(True)
    plt.plot(epochs, history["acc"], ".g-", linewidth=2, label="Train accuracy")
    plt.plot(epochs, history["test_acc"], ".b-", linewidth=2, label="Test accuracy")
    plt.plot(epochs, history["loss"], ".r-", linewidth=2, label="Train loss")
    plt.title("Model Training History")
    plt.xlabel("epoch")
    plt.xticks(list(epochs))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc(y_true, probs, class_names, out_dir):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n))
    plt.figure(figsize=(10, 8))
    for i in range(n):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_dl, test_dl, class_names = build_loaders(args)
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dl.dataset)}  Test images: {len(test_dl.dataset)}")

    model = TumorClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()  # == sparse_categorical_crossentropy on logits
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"loss": [], "acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch, args.epochs)
        history["loss"].append(loss)
        history["acc"].append(acc)
        # quick test-set check after every epoch so you can watch generalisation
        y_true, probs = predict_all(model, test_dl, device)
        test_acc = (probs.argmax(1) == y_true).mean()
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}  train_acc={acc:.4f}  test_acc={test_acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_checkpoint(model, class_names, args.out)
    print(f"Saved model to {args.out}")

    # ---- Evaluation on the test set ----
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    y_true, probs = predict_all(model, test_dl, device)
    y_pred = probs.argmax(1)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_history(history, out_dir)
    plot_confusion(y_true, y_pred, class_names, out_dir)
    plot_roc(y_true, probs, class_names, out_dir)
    print(f"Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
