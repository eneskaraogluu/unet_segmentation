"""Segmentasyon metrikleri: IoU (Jaccard) ve Dice (F1) katsayisi.

Ikisi de tahmin edilen maske ile gercek (ground truth) maske arasindaki
ortusmeyi olcer, ama farkli agirliklandirirlar:

    IoU  = |A ∩ B| / |A ∪ B|
    Dice = 2*|A ∩ B| / (|A| + |B|)   (== F1 skoru, ikili maskeler icin)

Dice, IoU'ya gore ortusmeye biraz daha "affedici" agirlik verir; ikisi de
matematiksel olarak birbirinden turetilebilir (Dice = 2*IoU / (1+IoU)) ama
literaturde ikisinin de raporlanmasi standarttir, bu yuzden burada ikisini
de hesapliyoruz.

Tum fonksiyonlar `logits` (sigmoid oncesi ham model ciktisi) alir ve
kendi icinde sigmoid + esikleme (thresholding) uygular; boylece egitim
dongusunde dogrudan model ciktisini gecebilirsiniz.
"""
import torch


@torch.no_grad()
def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denom + eps)
    return dice.mean()


class DiceLoss(torch.nn.Module):
    """1 - Dice katsayisi, turevlenebilir (differentiable) kayip olarak.

    Neden gerekli: metrikteki esikleme (threshold) islemi turevlenemez,
    egitimde dogrudan kullanilamaz. DiceLoss, esiklemeden once olasiliklar
    (sigmoid ciktisi) uzerinden "yumusak" (soft) bir Dice hesaplar, boylece
    geri yayilim (backprop) ile optimize edilebilir.

    Segmentasyonda tek basina BCE kullanmak, hucre/arka plan piksel
    sayisi dengesiz oldugunda (mikroskopi goruntulerinde hucreler genelde
    goruntunun kucuk bir kismini kaplar) modeli "hep arka plan de" onyargisina
    itebilir. Dice, sinif dengesizligine karsi dogal olarak daha dayaniklidir
    ve genelde BCE ile birlikte (BCE + Dice) kullanilir.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(torch.nn.Module):
    """BCEWithLogitsLoss + DiceLoss toplami. Segmentasyonda yaygin bir standart."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
