"""U-Net egitim betigi.

Sentetik veri uzerinde varsayilan olarak calisir (gercek veri seti henuz
olmasa da pipeline'in uctan uca dogru calistigini gosterir). Gercek TUBITAK
veri setiniz hazir oldugunda tek yapmaniz gereken --data-root vermek;
kod otomatik olarak FolderSegmentationDataset'e gecer (bkz. data/dataset.py).

Kullanim:
    python train.py                          # sentetik veriyle hizli deneme
    python train.py --epochs 30 --batch-size 8
    python train.py --data-root path/to/veri  # gercek veri klasoru ile
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import FolderSegmentationDataset, SyntheticCellDataset
from model.unet import UNet
from utils.metrics import BCEDiceLoss, dice_score, iou_score
from utils.transforms import get_train_transform, get_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net segmentasyon egitimi")
    parser.add_argument("--data-root", type=str, default=None,
                         help="Gercek veri klasoru (images/ ve masks/ alt klasorleri ile). "
                              "Verilmezse sentetik veri kullanilir.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32,
                         help="U-Net ilk seviye kanal sayisi (CPU'da hizli denemek icin 32; "
                              "GPU + gercek veriyle 64 klasik degerdir)")
    parser.add_argument("--num-samples", type=int, default=240,
                         help="Sentetik veri kullanilirken toplam ornek sayisi")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_datasets(args: argparse.Namespace):
    train_tf = get_train_transform(args.img_size)
    val_tf = get_val_transform(args.img_size)

    if args.data_root is not None:
        full_train = FolderSegmentationDataset(args.data_root, transform=train_tf)
        full_val = FolderSegmentationDataset(args.data_root, transform=val_tf)
        n_val = int(len(full_train) * args.val_split)
        n_train = len(full_train) - n_val
        indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.seed))
        train_indices, val_indices = indices[:n_train], indices[n_train:]
        train_ds = torch.utils.data.Subset(full_train, train_indices.tolist())
        val_ds = torch.utils.data.Subset(full_val, val_indices.tolist())
    else:
        n_val = int(args.num_samples * args.val_split)
        n_train = args.num_samples - n_val
        # Train/val icin AYNI seed tabanini kullanan ama farkli idx araligina
        # dusen iki SyntheticCellDataset ornegi: veri sizintisi (data leakage)
        # olmadan, birbirinden bagimsiz iki bolum elde edilir.
        train_ds = SyntheticCellDataset(num_samples=n_train, img_size=args.img_size,
                                         transform=train_tf, base_seed=args.seed)
        val_ds = SyntheticCellDataset(num_samples=n_val, img_size=args.img_size,
                                       transform=val_tf, base_seed=args.seed + 100_000)

    return train_ds, val_ds


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_iou, total_dice, n_batches = 0.0, 0.0, 0.0, 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score(logits, masks).item()
        total_dice += dice_score(logits, masks).item()
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


def save_prediction_sample(model, val_ds, device, out_path: str):
    """Gorsel dogrulama: bir val ornegi icin girdi / gercek maske / tahmin."""
    model.eval()
    image, mask = val_ds[0]
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()

    img_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Girdi goruntu")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Gercek maske (ground truth)")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Model tahmini")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    train_ds, val_ds = build_datasets(args)
    print(f"Egitim ornegi: {len(train_ds)} | Dogrulama ornegi: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayisi: {n_params:,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_iou, val_dice = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        dt = time.time() - t0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} | "
            f"{dt:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = os.path.join(args.checkpoint_dir, "best_unet.pt")
            torch.save({"model_state": model.state_dict(), "args": vars(args), "val_dice": val_dice}, ckpt_path)

    sample_path = os.path.join(args.checkpoint_dir, "sample_prediction.png")
    save_prediction_sample(model, val_ds, device, sample_path)
    print(f"En iyi dogrulama Dice skoru: {best_val_dice:.4f}")
    print(f"Ornek tahmin gorseli kaydedildi: {sample_path}")
    print(f"En iyi model kaydedildi: {os.path.join(args.checkpoint_dir, 'best_unet.pt')}")


if __name__ == "__main__":
    main()
