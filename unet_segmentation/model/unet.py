"""Klasik U-Net mimarisi (Ronneberger ve ark., 2015) - sifirdan.

Yapi ozeti:
                input
                  |
     [Encoder]  DoubleConv -> skip1 --------------------+
                  | MaxPool                              |
                DoubleConv -> skip2 ----------------+     |
                  | MaxPool                          |     |
                DoubleConv -> skip3 ------------+    |     |
                  | MaxPool                      |    |     |
                DoubleConv -> skip4 --------+    |    |     |
                  | MaxPool                  |    |    |     |
     [Bottleneck] DoubleConv                |    |    |     |
                  | UpConv (transpose)       |    |    |     |
     [Decoder]   concat(skip4) -> DoubleConv-+    |    |     |
                  | UpConv                        |    |     |
                concat(skip3) -> DoubleConv -------+    |     |
                  | UpConv                              |     |
                concat(skip2) -> DoubleConv --------------+     |
                  | UpConv                                     |
                concat(skip1) -> DoubleConv ----------------------+
                  |
              1x1 Conv -> cikti (logits)

Encoder her asamada uzamsal cozunurlugu yariya indirip kanal sayisini
ikiye katlar (ne'yi ogrenir). Decoder bunu tersine cevirirken, skip
connection'lar encoder'daki INCE UZAMSAL DETAYI (nerede oldugunu)
decoder'a tasir - bu sayede piksel-seviyesinde kesin sinirlar elde edilir.
Skip connection olmasaydi, bottleneck'teki asiri sikistirilmis temsilden
(representation) ince kenarlari geri kazanmak neredeyse imkansiz olurdu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2

    U-Net'in her seviyesindeki temel yapi tasi. Iki ardisik 3x3 konvolusyon,
    tek bir 3x3'ten daha genis bir etkin alici alan (receptive field) saglar
    ve daha derin/dogrusal-olmayan bir temsil ogrenmeyi mumkun kilar.
    BatchNorm, egitimi stabilize edip daha yuksek ogrenme oranlariyla daha
    hizli yakinsamayi saglar (orijinal 2015 makalesinde yoktu, ama gunumuz
    standart pratiginde neredeyse her zaman eklenir).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder adimi: MaxPool (2x kucult) -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Decoder adimi: transpose conv ile 2x buyut -> skip ile birlestir -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels, alt seviyeden gelen kanal sayisi; upconv bunu
        # yariya indirir, sonra skip connection ile concat edilince
        # tekrar in_channels'a ulasilir.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Giris boyutu 2'nin kuvveti olmayabilir; pool/upsample sonrasi
        # boyut uyumsuzlugu olursa skip'e gore hizala (padding).
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Klasik U-Net.

    Args:
        in_channels: giris goruntusu kanal sayisi (gri seviye mikroskopi
            icin 1, RGB icin 3).
        out_channels: cikti kanal sayisi. Ikili segmentasyon icin 1
            (sigmoid + BCE/Dice ile kullanilir). Coklu sinif segmentasyon
            icin sinif sayisi (softmax + CrossEntropy ile kullanilir).
        base_channels: ilk seviyedeki kanal sayisi; sonraki her seviyede
            ikiye katlanir (base, 2*base, 4*base, 8*base, 16*base).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)   # skip 1 (tam cozunurluk)
        x2 = self.down1(x1)    # skip 2 (1/2)
        x3 = self.down2(x2)    # skip 3 (1/4)
        x4 = self.down3(x3)    # skip 4 (1/8)
        x5 = self.down4(x4)    # bottleneck (1/16)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)  # logits (sigmoid/softmax kayip fonksiyonunda uygulanir)


if __name__ == "__main__":
    # Hizli boyut kontrolu: rastgele bir tensor ile ileri gecis (forward pass).
    model = UNet(in_channels=1, out_channels=1, base_channels=32)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"input:  {tuple(dummy.shape)}")
    print(f"output: {tuple(out.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametre sayisi: {n_params:,}")
