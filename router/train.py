"""Training script for the RouterGraph packet classifier."""

import argparse
import glob
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from core.constants import CLASS_NAMES
from router.model import RouterGraph
from router.dataset import PacketDataset, HybridDataset


def train(
    samples_per_class=50000,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    val_split=0.1,
    save_path="models/router_graph.pth",
    pcap_files=None,
    real_oversample=5,
):
    device = torch.device("cpu")  # Training on CPU is fine for this tiny model

    if pcap_files:
        print(f"Building hybrid dataset with {len(pcap_files)} pcap file(s)...")
        dataset = HybridDataset(
            pcap_files=pcap_files,
            samples_per_class=samples_per_class,
            real_oversample=real_oversample,
        )
    else:
        print("Generating synthetic dataset...")
        dataset = PacketDataset(samples_per_class=samples_per_class)

    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Dataset: {train_size} train, {val_size} val")
    print(f"Classes: {CLASS_NAMES}")

    model = RouterGraph().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_x.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct / total_samples

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RouterGraph classifier")
    parser.add_argument("--pcap", nargs="+", help="Real pcap files for hybrid training")
    parser.add_argument("--pcap-dir", help="Directory of .pcap files")
    parser.add_argument("--oversample", type=int, default=5,
                        help="Oversample factor for real data (default: 5)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=50000)
    args = parser.parse_args()

    pcap_files = []
    if args.pcap:
        pcap_files.extend(args.pcap)
    if args.pcap_dir:
        pcap_files.extend(glob.glob(os.path.join(args.pcap_dir, "*.pcap")))

    train(
        pcap_files=pcap_files or None,
        real_oversample=args.oversample,
        epochs=args.epochs,
        samples_per_class=args.samples_per_class,
    )
