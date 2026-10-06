"""Egitilmis U-Net ile bir klasordeki goruntuler uzerinde tahmin yapar.

Her goruntu icin cikti klasorune:
    <ad>_mask.png     -> tahmin edilen ikili maske (0/255)
    <ad>_overlay.jpg  -> goruntu + tahmin (yesil) ust uste
kaydedilir. --masks verilirse (NIfTI veya PNG maske klasoru) her goruntu
icin Dice/IoU hesaplanir ve ortalamasi yazdirilir.

Goruntuler egitimdeki gibi --scale (varsayilan 0.25) ile kucultulur ve
kirpilmadan, TAM goruntu olarak modele verilir (U-Net tam konvolusyonel
oldugu icin egitimdeki kirpma boyutundan bagimsiz calisir).

Kullanim:
    python predict.py --checkpoint checkpoints_kan/best_unet.pt --images SEG/test_net --out tahminler
    python predict.py ... --masks SEG/maske   # maske varsa skor da hesaplar
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.unet import UNet
from utils.transforms import MEAN, STD


def load_mask(path: str) -> np.ndarray:
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(path).dataobj)
        if arr.ndim == 3:
            arr = arr.max(axis=2)
        return (arr > 0).astype(np.uint8)
    return (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nii", ".nii.gz")


def strip_ext(filename: str) -> str:
    return filename[:-7] if filename.lower().endswith(".nii.gz") else os.path.splitext(filename)[0]


def load_image(path: str) -> np.ndarray:
    """BGR uint8 goruntu dondurur. NIfTI goruntuler (H, W, 3) RGB olarak saklanir."""
    if path.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(path).dataobj)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(arr[:, :, ::-1])
    return cv2.imread(path)


def find_mask(masks_dir: str, name: str):
    for ext in (".nii", ".nii.gz", ".png"):
        path = os.path.join(masks_dir, name + ext)
        if os.path.exists(path):
            return path
    return None


@torch.no_grad()
def predict(model, gray: np.ndarray, device) -> np.ndarray:
    x = (gray.astype(np.float32) / 255.0 - MEAN[0]) / STD[0]
    x = torch.from_numpy(x)[None, None].to(device)
    # U-Net 4 kez 2x kucultur: boyutlari 16'nin katina tamamla, sonra geri kirp.
    h, w = gray.shape
    pad_h, pad_w = (-h) % 16, (-w) % 16
    x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
    probs = torch.sigmoid(model(x))[0, 0, :h, :w]
    return (probs > 0.5).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="U-Net tahmin")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True, help="Goruntu klasoru")
    parser.add_argument("--out", required=True, help="Cikti klasoru")
    parser.add_argument("--masks", default=None, help="(Opsiyonel) gercek maske klasoru -> Dice/IoU")
    parser.add_argument("--scale", type=float, default=0.25, help="Egitimdeki prepare_data --scale ile ayni olmali")
    parser.add_argument("--val-only-from", default=None,
                        help="Egitim goruntu klasoru (ornegin SEG/train). Verilirse train.py'deki train/val "
                             "bolunmesi ayni seed ile yeniden uretilir ve SADECE dogrulama goruntuleri islenir "
                             "(egitimde gorulen goruntuler skoru sisirmesin diye).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet(in_channels=1, out_channels=1, base_channels=ckpt["args"]["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model yuklendi (egitimdeki en iyi val Dice: {ckpt['val_dice']:.4f}) | Cihaz: {device}")

    os.makedirs(args.out, exist_ok=True)
    # Alt klasorler de taranir (ornegin test_net/images/...).
    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(args.images)
        for f in names
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        found = sorted({os.path.splitext(f)[1] or f for _, _, names in os.walk(args.images) for f in names})
        print(f"UYARI: {args.images} icinde goruntu bulunamadi. Bulunan dosya turleri: {found or 'klasor bos'}")
        return

    if args.val_only_from:
        # train.py/build_datasets ile birebir ayni bolunme: sirali dosya listesi + ayni seed.
        train_names = sorted(f for f in os.listdir(args.val_only_from))
        n_val = int(len(train_names) * ckpt["args"]["val_split"])
        perm = torch.randperm(len(train_names), generator=torch.Generator().manual_seed(ckpt["args"]["seed"]))
        val_names = {strip_ext(train_names[i]) for i in perm[len(train_names) - n_val:].tolist()}
        files = [p for p in files if strip_ext(os.path.basename(p)) in val_names]
        print(f"Sadece dogrulama goruntuleri: {sorted(val_names, key=lambda s: int(s) if s.isdigit() else s)}")

    scores = []
    for path in files:
        f = os.path.basename(path)
        name = strip_ext(f)
        image = load_image(path)
        if args.scale != 1.0:
            size = (round(image.shape[1] * args.scale), round(image.shape[0] * args.scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        pred = predict(model, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), device)

        cv2.imwrite(os.path.join(args.out, name + "_mask.png"), pred * 255)
        overlay = image.copy()
        overlay[pred > 0] = (0, 255, 0)
        cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"), cv2.addWeighted(image, 0.6, overlay, 0.4, 0))

        line = f"{f}"
        mask_path = find_mask(args.masks, name) if args.masks else None
        if mask_path:
            gt = load_mask(mask_path)
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            inter = float((pred & gt).sum())
            total = float(pred.sum() + gt.sum())
            dice = (2 * inter + 1e-6) / (total + 1e-6)
            iou = (inter + 1e-6) / (total - inter + 1e-6)
            scores.append((dice, iou))
            line += f"  dice={dice:.4f} iou={iou:.4f}"
        print(line)

    print(f"{len(files)} goruntu islendi -> {args.out}")
    if scores:
        d, i = np.mean(scores, axis=0)
        print(f"Ortalama Dice: {d:.4f} | Ortalama IoU: {i:.4f} ({len(scores)} maskeli goruntu)")


if __name__ == "__main__":
    main()
