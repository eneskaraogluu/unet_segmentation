"""2. Ay: CNN ve Transformer segmentasyon modellerinin 5-fold cross validation ile kiyaslanmasi.

Modeller (--models):
    unet_basic : 1. ayda sifirdan yazdigimiz U-Net (on-egitimsiz, referans)
    unet       : U-Net        + ResNet34 encoder (ImageNet on-egitimli)   [CNN]
    unetpp     : U-Net++      + ResNet34 encoder                          [CNN]
    linknet    : LinkNet      + ResNet34 encoder                          [CNN]
    vit        : DPT          + ViT-Small/16 encoder                      [Transformer]
    swin       : UPerNet      + Swin-Tiny encoder                         [Transformer]

Loss fonksiyonlari (--loss): bce, dice, focal, bce_dice, focal_dice

Veri: --images klasorundeki PNG veya NIfTI goruntuler + --masks klasorundeki NIfTI
maskeler. Ham / optimum odakli (net) / bulanik gruplar ayni maskeleri paylasir; her
grup icin --data-name farkli verilerek ayri ayri kosturulur.

Cross validation: goruntuler 5 parcaya (fold) bolunur; her fold'da model 4 parca ile
egitilip kalan 1 parcada test edilir. Boylece her goruntu tam olarak bir kez test
goruntusu olur ve skorlar 5 fold'un ortalamasi +- standart sapmasi olarak raporlanir.
Fold'lar --seed ile sabittir: tum modeller/loss'lar/veri gruplari AYNI bolunmeyi kullanir.

Her fold bittiginde sonuc <out>/results.csv dosyasina eklenir. Colab oturumu koparsa
ayni komut tekrar calistirildiginda bitmis fold'lar atlanir, kalan yerden devam edilir.

Kullanim:
    python benchmark.py --images SEG/train --masks SEG/maske --data-name ham \
        --models unet unetpp linknet vit swin --loss bce_dice --out sonuclar
    python benchmark.py --summary --out sonuclar
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)  # ImageNet istatistikleri (on-egitimli encoder'lar bunu bekler)
STD = (0.229, 0.224, 0.225)
MODEL_TYPES = {"unet_basic": "CNN", "unet": "CNN", "unetpp": "CNN", "linknet": "CNN",
               "vit": "Transformer", "swin": "Transformer"}
RESULT_FIELDS = ["data", "model", "type", "loss", "fold", "n_train", "n_test", "epochs",
                 "dice", "iou", "precision", "recall", "specificity", "accuracy",
                 "params_m", "train_min"]


# ---------------------------------------------------------------- veri
def strip_ext(filename: str) -> str:
    return filename[:-7] if filename.lower().endswith(".nii.gz") else os.path.splitext(filename)[0]


def read_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)  # (H, W[, C]); H/W PNG ile ayni eksen sirasinda


def load_rgb(path: str) -> np.ndarray:
    if path.lower().endswith((".nii", ".nii.gz")):
        arr = read_nifti(path)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.stack([arr] * 3, axis=2) if arr.ndim == 2 else np.ascontiguousarray(arr)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_mask(path: str) -> np.ndarray:
    if path.lower().endswith((".nii", ".nii.gz")):
        arr = read_nifti(path)
        if arr.ndim == 3:
            arr = arr.max(axis=2)  # cogu maske 3 kanalli (etiket ortada), bazilari tek kanalli
        return (arr > 0).astype(np.uint8)
    return (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def list_files(folder: str, exts) -> dict:
    return {strip_ext(f): os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)}


def load_dataset(images_dir, masks_dir, scale, cache_path, ids=None):
    """Tum goruntu+maskeleri kucultup bellege alir; ikinci calistirmada cache'ten okur."""
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        names, images, masks = list(data["names"]), data["images"], data["masks"]
    else:
        imgs = list_files(images_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nii", ".nii.gz"))
        msks = list_files(masks_dir, (".nii", ".nii.gz", ".png"))
        names = sorted(set(imgs) & set(msks), key=lambda s: int(s) if s.isdigit() else s)
        images, masks = [], []
        for i, n in enumerate(names):
            img, msk = load_rgb(imgs[n]), load_mask(msks[n])
            if img.shape[:2] != msk.shape:
                raise ValueError(f"{n}: goruntu {img.shape[:2]} / maske {msk.shape} boyut uyusmazligi")
            size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
            images.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(msk, size, interpolation=cv2.INTER_NEAREST))
            print(f"\r  veri okunuyor {i + 1}/{len(names)}", end="", flush=True)
        print()
        images, masks = np.stack(images), np.stack(masks)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, names=np.array(names), images=images, masks=masks)
    if ids is not None:
        keep = [i for i, n in enumerate(names) if n in set(ids)]
        missing = sorted(set(ids) - set(names))
        if missing:
            raise ValueError(f"Bu veri grubunda olmayan goruntuler: {missing}")
        names, images, masks = [names[i] for i in keep], images[keep], masks[keep]
    return names, images, masks


class CropDataset(Dataset):
    def __init__(self, images, masks, img_size, crops_per_image):
        self.images, self.masks, self.k = images, masks, crops_per_image
        self.tf = A.Compose([
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            # Bilincli olarak blur/noise augmentation YOK: ham ve optimum odakli veri
            # gruplarinin farkini olcuyoruz, augmentation bu farki maskelememeli.
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.images) * self.k

    def __getitem__(self, i):
        out = self.tf(image=self.images[i % len(self.images)], mask=self.masks[i % len(self.images)])
        return out["image"], out["mask"].float().unsqueeze(0)


# ---------------------------------------------------------------- modeller
def build_model(name: str, img_size: int) -> nn.Module:
    import segmentation_models_pytorch as smp
    if name == "unet_basic":
        from model.unet import UNet
        return UNet(in_channels=3, out_channels=1, base_channels=32)
    if name == "unet":
        return smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
    if name == "unetpp":
        return smp.UnetPlusPlus("resnet34", encoder_weights="imagenet", classes=1)
    if name == "linknet":
        return smp.Linknet("resnet34", encoder_weights="imagenet", classes=1)
    if name == "vit":
        return smp.DPT("tu-vit_small_patch16_224.augreg_in21k_ft_in1k", encoder_weights="imagenet",
                       classes=1, dynamic_img_size=True)
    if name == "swin":
        return smp.UPerNet("tu-swin_tiny_patch4_window7_224", encoder_weights="imagenet",
                           classes=1, img_size=img_size)
    raise ValueError(f"Bilinmeyen model: {name}")


# ---------------------------------------------------------------- loss
class Loss(nn.Module):
    def __init__(self, kind: str, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        if kind not in {"bce", "dice", "focal", "bce_dice", "focal_dice"}:
            raise ValueError(f"Bilinmeyen loss: {kind}")
        self.kind, self.alpha, self.gamma = kind, alpha, gamma

    @staticmethod
    def dice(logits, t, eps=1.0):
        p = torch.sigmoid(logits)
        inter = (p * t).sum(dim=(1, 2, 3))
        return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps)).mean()

    def focal(self, logits, t):
        # Focal Loss (Lin ve ark., 2017): kolay pikselleri (1-p_t)^gamma ile bastirir,
        # modeli zor piksellere (hucre sinirlari, ust uste binen hucreler) odaklar.
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

    def forward(self, logits, t):
        logits = logits.float()
        if self.kind == "bce":
            return F.binary_cross_entropy_with_logits(logits, t)
        if self.kind == "dice":
            return self.dice(logits, t)
        if self.kind == "focal":
            return self.focal(logits, t)
        if self.kind == "bce_dice":
            return F.binary_cross_entropy_with_logits(logits, t) + self.dice(logits, t)
        return self.focal(logits, t) + self.dice(logits, t)


# ---------------------------------------------------------------- degerlendirme
def _starts(length, tile):
    if length <= tile:
        return [0]
    stride = tile * 3 // 4
    s = list(range(0, length - tile, stride))
    return s + [length - tile]


@torch.no_grad()
def predict_full(model, image_rgb, tile, device):
    """Tam goruntu tahmini: ust uste binen tile x tile pencerelerle (sliding window).
    Swin sabit girdi boyutu istedigi icin tum modeller ayni sekilde degerlendirilir."""
    x = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    h, w = x.shape[2:]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        x = F.pad(x, [0, pw, 0, ph], mode="reflect")
    H, W = x.shape[2:]
    acc = torch.zeros(1, 1, H, W, device=device)
    cnt = torch.zeros_like(acc)
    for y0 in _starts(H, tile):
        for x0 in _starts(W, tile):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x[:, :, y0:y0 + tile, x0:x0 + tile])
            acc[:, :, y0:y0 + tile, x0:x0 + tile] += out.float()
            cnt[:, :, y0:y0 + tile, x0:x0 + tile] += 1
    return (torch.sigmoid(acc / cnt)[0, 0, :h, :w] > 0.5).cpu().numpy()


def evaluate(model, images, masks, tile, device):
    model.eval()
    tp = fp = fn = tn = 0
    dices, ious = [], []
    for img, gt in zip(images, masks):
        pred, gt = predict_full(model, img, tile, device), gt.astype(bool)
        a, b, c = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
        d = (~pred & ~gt).sum()
        tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
        dices.append(2 * a / max(2 * a + b + c, 1))
        ious.append(a / max(a + b + c, 1))
    metrics = {
        "dice": float(np.mean(dices)),  # goruntu basina ortalama
        "iou": float(np.mean(ious)),
        "precision": tp / max(tp + fp, 1),  # tum test pikselleri uzerinden
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
    }
    return metrics, [float(d) for d in dices]


# ---------------------------------------------------------------- egitim
def make_folds(n, k, seed):
    perm = np.random.RandomState(seed).permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def train_one(args, model_name, train_imgs, train_msks, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(model_name, args.img_size).to(device)
    loader = DataLoader(CropDataset(train_imgs, train_msks, args.img_size, args.crops_per_image),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    lr = args.lr if model_name != "unet_basic" else args.lr * 10  # on-egitimsiz model daha yuksek lr ister
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.epochs * len(loader),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = Loss(args.loss)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        history.append(total / len(loader))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"    epoch {epoch:3d}/{args.epochs} loss={history[-1]:.4f}", flush=True)
    return model, history


def done_runs(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["data"], r["model"], r["loss"], int(r["fold"])) for r in csv.DictReader(f)}


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    cache = os.path.join(args.out, "cache", f"{args.data_name}_s{args.scale}.npz")
    names, images, masks = load_dataset(args.images, args.masks, args.scale, cache, ids)
    folds = make_folds(len(names), args.folds, args.seed)
    print(f"Cihaz: {device} | veri: {args.data_name} ({len(names)} goruntu) | loss: {args.loss}")
    for i, f in enumerate(folds):
        print(f"  fold {i}: test = {[names[j] for j in f]}")

    results_csv = os.path.join(args.out, "results.csv")
    fold_ids = args.fold_list if args.fold_list else range(args.folds)
    for model_name in args.models:
        for k in fold_ids:
            key = (args.data_name, model_name, args.loss, k)
            if key in done_runs(results_csv):
                print(f"[atla] {key} zaten tamamlanmis")
                continue
            test_idx = folds[k]
            train_idx = np.setdiff1d(np.arange(len(names)), test_idx)
            print(f"\n=== {args.data_name} | {model_name} | {args.loss} | fold {k + 1}/{args.folds} "
                  f"(egitim {len(train_idx)}, test {len(test_idx)}) ===", flush=True)
            t0 = time.time()
            model, history = train_one(args, model_name, images[train_idx], masks[train_idx],
                                       device, seed=args.seed * 100 + k)
            train_min = (time.time() - t0) / 60
            metrics, per_image = evaluate(model, images[test_idx], masks[test_idx], args.img_size, device)
            print("    TEST: " + " ".join(f"{m}={v:.4f}" for m, v in metrics.items()) + f" | {train_min:.1f} dk")

            row = {"data": args.data_name, "model": model_name, "type": MODEL_TYPES[model_name],
                   "loss": args.loss, "fold": k, "n_train": len(train_idx), "n_test": len(test_idx),
                   "epochs": args.epochs, **{m: round(v, 5) for m, v in metrics.items()},
                   "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                   "train_min": round(train_min, 2)}
            new = not os.path.exists(results_csv)
            with open(results_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            log_dir = os.path.join(args.out, "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"{args.data_name}_{model_name}_{args.loss}_fold{k}.json"), "w") as f:
                json.dump({"train_loss": history, "test": metrics,
                           "per_image_dice": dict(zip([names[j] for j in test_idx], per_image))}, f)
            if args.save_models:
                torch.save(model.state_dict(), os.path.join(args.out, f"{args.data_name}_{model_name}_{args.loss}_fold{k}.pt"))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def summary(out_dir):
    import pandas as pd
    df = pd.read_csv(os.path.join(out_dir, "results.csv"))
    metrics = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
    g = df.groupby(["data", "type", "model", "loss"])
    table = g[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["folds"] = g.size()
    table["params_m"] = g["params_m"].first()
    table["train_min_per_fold"] = g["train_min"].mean().round(1)
    table = table.reset_index().sort_values(["data", "dice_mean"], ascending=[True, False])
    table.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    pd.set_option("display.width", 200)
    view = table[["data", "type", "model", "loss", "folds", "params_m", "train_min_per_fold"]].copy()
    for m in metrics:
        view[m] = [f"{a:.4f} ± {b:.4f}" if b == b else f"{a:.4f}" for a, b in zip(table[f"{m}_mean"], table[f"{m}_std"])]
    print(view.to_string(index=False))
    return table


def main():
    p = argparse.ArgumentParser(description="CNN vs Transformer segmentasyon kiyaslamasi (k-fold CV)")
    p.add_argument("--images", help="Goruntu klasoru (PNG veya NIfTI)")
    p.add_argument("--masks", help="Maske klasoru (NIfTI)")
    p.add_argument("--data-name", default="ham", help="Sonuc tablosundaki veri grubu adi (ham, net, bulanik ...)")
    p.add_argument("--ids", default=None, help="Virgulle ayrilmis goruntu adlari; tum veri gruplari ayni goruntuleri kullansin diye")
    p.add_argument("--models", nargs="+", default=["unet", "unetpp", "linknet", "vit", "swin"], choices=list(MODEL_TYPES))
    p.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "focal", "bce_dice", "focal_dice"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-list", type=int, nargs="+", default=None, help="Sadece bu fold'lari kostur (0'dan baslar)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--crops-per-image", type=int, default=4, help="Her epoch'ta goruntu basina rastgele kirpma sayisi")
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--summary", action="store_true", help="Sadece results.csv'den ozet tablo uret")
    args = p.parse_args()
    if args.summary:
        summary(args.out)
    else:
        if not (args.images and args.masks):
            p.error("--images ve --masks gerekli")
        run(args)


if __name__ == "__main__":
    main()
