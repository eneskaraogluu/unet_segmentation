"""Segmentasyon veri setleri.

Iki Dataset sinifi var:

1) SyntheticCellDataset:
   Gercek biyomedikal veri gelene kadar boru hattini (pipeline) test etmek
   icin, mikroskopi goruntulerine benzeyen sentetik hucre goruntuleri ve
   ikili (binary) maskeler ureten bir Dataset. Her __getitem__ cagrisinda,
   idx'e bagli sabit bir tohum (seed) ile "ayni" temel goruntuyu, sonra
   ustune farkli bir rastgelelikle augmentation uygular. Boylece:
     - Ayni index her zaman ayni temel hucre yerlesimini verir (val/test
       icin tutarlilik saglar).
     - Augmentation her epoch'ta farkli sonuc uretir (egitim cesitliligi).

2) FolderSegmentationDataset:
   Gercek veri setiniz hazir oldugunda kullanacaginiz, disk uzerindeki
   goruntu/maske klasor ciftlerini okuyan genel amacli Dataset.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset


def _generate_synthetic_sample(
    idx: int,
    img_size: int,
    min_cells: int,
    max_cells: int,
    base_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Tek bir sentetik (goruntu, maske) cifti uretir.

    Goruntu: HxW, uint8, tek kanal (gri seviye mikroskopi goruntusu gibi).
    Maske:   HxW, uint8, {0,1} degerli ikili maske (1 = hucre pikseli).
    """
    rng = np.random.RandomState(base_seed + idx)

    # Arka plan: hafif dokulu, dusuk yogunluklu gurultu (mikroskopi
    # goruntulerindeki zemine benzer).
    background_level = rng.randint(20, 45)
    image = np.full((img_size, img_size), background_level, dtype=np.float32)
    image += rng.normal(0, 4, size=(img_size, img_size))

    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    num_cells = rng.randint(min_cells, max_cells + 1)
    for _ in range(num_cells):
        cx = rng.randint(int(img_size * 0.1), int(img_size * 0.9))
        cy = rng.randint(int(img_size * 0.1), int(img_size * 0.9))
        axis_a = rng.randint(8, max(9, img_size // 12))
        axis_b = rng.randint(8, max(9, img_size // 12))
        angle = rng.randint(0, 180)
        intensity = rng.randint(150, 230)

        cell_layer = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.ellipse(
            cell_layer,
            (cx, cy),
            (axis_a, axis_b),
            angle,
            0,
            360,
            color=255,
            thickness=-1,
        )

        # Hucre govdesini goruntuye ekle (yumusak kenar icin blur).
        cell_float = cell_layer.astype(np.float32) * (intensity / 255.0)
        cell_float = cv2.GaussianBlur(cell_float, (5, 5), 1.2)
        image = np.maximum(image, cell_float)

        mask[cell_layer > 0] = 1

    # Hafif optik bulaniklik (defocus) ve sensor gurultusu ekle.
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    image += rng.normal(0, 3, size=(img_size, img_size))
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, mask


class SyntheticCellDataset(Dataset):
    """Sentetik mikroskopi hucre goruntuleri + ikili segmentasyon maskeleri.

    Gercek veri seti hazir olana kadar DataLoader + augmentation + U-Net
    boru hattinin uctan uca calistigini dogrulamak icin kullanilir.
    """

    def __init__(
        self,
        num_samples: int = 200,
        img_size: int = 256,
        min_cells: int = 5,
        max_cells: int = 15,
        transform: Optional[Callable] = None,
        base_seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.img_size = img_size
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.transform = transform
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        image, mask = _generate_synthetic_sample(
            idx, self.img_size, self.min_cells, self.max_cells, self.base_seed
        )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.float().unsqueeze(0) if hasattr(mask, "float") else mask


class FolderSegmentationDataset(Dataset):
    """Disk uzerinde goruntu/maske klasor ciftlerinden okuyan genel Dataset.

    Beklenen klasor yapisi:
        root/
          images/
            img_001.png
            img_002.png
            ...
          masks/
            img_001.png   <- ayni dosya adinda, ikili (0/255) maske
            img_002.png
            ...

    Gercek TUBITAK veri setiniz hazir oldugunda SyntheticCellDataset'in
    yerine bunu kullanmaniz yeterli; DataLoader, augmentation ve egitim
    dongusunde hicbir degisiklik gerekmez.
    """

    def __init__(
        self,
        root: str,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = os.path.join(root, images_subdir)
        self.masks_dir = os.path.join(root, masks_subdir)
        self.filenames = sorted(os.listdir(self.images_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.float().unsqueeze(0) if hasattr(mask, "float") else mask
