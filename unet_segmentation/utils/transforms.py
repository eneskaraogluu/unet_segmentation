"""Albumentations tabanli veri artirma (data augmentation) pipeline'lari.

Neden albumentations?
  torchvision.transforms goruntu ve maskeyi BAGIMSIZ olarak dondurur; yani
  goruntuye rastgele bir rotasyon uygularsaniz, ayni rastgele rotasyonun
  maskeye de uygulandigindan emin olmak icin elle senkronize etmeniz
  gerekir. albumentations, image= ve mask= parametrelerini AYNI rastgele
  durumla (random state) birlikte donusturur, bu yuzden segmentasyon
  gorevlerinde fiilen standart tercih budur.

Egitim (train) ve dogrulama (val) icin AYRI pipeline kullanilir:
  - train_transform: modeli daha genellenebilir (robust) yapmak icin
    gecirisken/geometrik/fotometrik bozulmalar icerir.
  - val_transform: sadece yeniden boyutlandirma + normalizasyon yapar;
    dogrulama/test skorlarinin rastgelelige bagli olmamasi gerekir.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Tek kanal (gri seviye) goruntuler icin normalizasyon degerleri.
# [0, 255] -> [0, 1] -> (x - 0.5) / 0.5 ile [-1, 1] araligina cekilir.
MEAN = (0.5,)
STD = (0.5,)


def get_train_transform(img_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.RandomCrop(height=img_size, width=img_size),
            # --- Geometrik augmentation: mikroskop altinda hucreler her
            # yonelimde/olcekte gorulebilir, model yon/olcek degisimine
            # duyarsiz (invariant) olmali.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),
            # --- Elastik/optik deformasyon: orijinal U-Net makalesinde
            # (Ronneberger ve ark., 2015) biyomedikal goruntuler icin
            # ozellikle onerilen, doku benzeri esneme simule eden augmentation.
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(p=0.2),
            # --- Fotometrik augmentation: farkli mikroskop/aydinlatma
            # kosullarina karsi dayaniklilik.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # --- Normalizasyon + tensore cevirme (her zaman pipeline'in sonunda).
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )
