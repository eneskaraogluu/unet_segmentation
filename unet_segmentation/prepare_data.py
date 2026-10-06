"""Kan yaymasi veri setini train.py'nin bekledigi klasor yapisina donusturur.

Girdi (Drive'daki SEG klasoru):
    train/7.png    -> 3840x2160 RGB mikroskopi goruntusu
    maske/7.nii    -> ayni boyutta NIfTI maske (0 = arka plan, 1 = kan hucresi)

Cikti (FolderSegmentationDataset'in bekledigi yapi):
    out/images/7.png
    out/masks/7.png   (0/255 ikili maske)

Notlar:
  - Maskelerin cogu 3 kanalli (etiket ortadaki kanalda), bazilari (7, 9, 32)
    tek kanalli. Kanal ekseninde max alarak ikisi de ayni sekilde okunur.
  - 4K goruntuler hem egitimi yavaslatir hem de 256'lik rastgele kirpmalar
    goruntunun cok kucuk bir kismini gorur. Bu yuzden --scale (varsayilan
    0.25 -> 960x540) ile kucultulur. Maske NEAREST ile kucultulur ki 0/1
    disinda ara degerler olusmasin.

Kullanim:
    python prepare_data.py --images SEG/train --masks SEG/maske --out veri
"""
import argparse
import os

import cv2
import nibabel as nib
import numpy as np


def load_mask(path: str) -> np.ndarray:
    # nibabel (H, W, C) dondurur; H/W eksenleri PNG ile birebir ayni.
    arr = np.asarray(nib.load(path).dataobj)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return (arr > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="PNG + NIfTI maske -> images/ + masks/ PNG")
    parser.add_argument("--images", required=True, help="Goruntu klasoru (ornegin SEG/train)")
    parser.add_argument("--masks", required=True, help="NIfTI maske klasoru (ornegin SEG/maske)")
    parser.add_argument("--out", required=True, help="Cikti klasoru")
    parser.add_argument("--scale", type=float, default=0.25, help="Yeniden boyutlandirma orani")
    args = parser.parse_args()

    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    names = sorted(
        (os.path.splitext(f)[0] for f in os.listdir(args.images) if f.lower().endswith(".png")),
        key=lambda s: int(s) if s.isdigit() else s,
    )

    n_ok, skipped = 0, []
    for name in names:
        mask_path = os.path.join(args.masks, name + ".nii")
        if not os.path.exists(mask_path):
            mask_path += ".gz"
        if not os.path.exists(mask_path):
            skipped.append(name)
            continue

        image = cv2.imread(os.path.join(args.images, name + ".png"))
        mask = load_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"{name}: goruntu {image.shape[:2]} ile maske {mask.shape} boyutu uyusmuyor")

        if args.scale != 1.0:
            size = (round(image.shape[1] * args.scale), round(image.shape[0] * args.scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(out_images, name + ".png"), image)
        cv2.imwrite(os.path.join(out_masks, name + ".png"), mask * 255)
        n_ok += 1

    print(f"Hazirlanan ornek: {n_ok} -> {args.out}")
    if skipped:
        print(f"Maskesi olmadigi icin atlananlar: {skipped}")


if __name__ == "__main__":
    main()
