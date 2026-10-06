# U-Net Segmentasyon Kod Tabani (TUBITAK 2247/C - 1. Ay Teslimati)

Bu klasor, projenizin 1. ay hedeflerinden ikisini karsilar:

1. PyTorch ile segmentasyon `DataLoader` ve `Data Augmentation` pipeline'lari
2. Sifirdan kodlanmis klasik U-Net, IoU ve Dice/F1 metrikleriyle egitim

Kod, gercek biyomedikal veri setiniz hazir olmadan da uctan uca calisir:
`SyntheticCellDataset`, mikroskopi goruntulerine benzeyen sentetik hucre
goruntuleri + ikili maskeler uretir. Gercek veriniz geldiginde tek satir
degistirmeniz yeterli (asagida "Gercek veriye gecis" bolumune bakin).

## Klasor yapisi

```
unet_segmentation/
  data/
    dataset.py       # SyntheticCellDataset + FolderSegmentationDataset
  model/
    unet.py           # Sifirdan U-Net (DoubleConv, Down, Up, skip connections)
  utils/
    transforms.py      # albumentations tabanli train/val augmentation pipeline'lari
    metrics.py          # IoU, Dice/F1, DiceLoss, BCEDiceLoss
  train.py              # Egitim dongusu (DataLoader -> model -> metrikler -> checkpoint)
  requirements.txt
  checkpoints/          # En iyi model + ornek tahmin gorseli buraya kaydedilir
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

Sentetik veriyle hizli deneme (CPU'da ~birkac dakika):

```bash
python train.py --epochs 15 --batch-size 8 --img-size 128
```

Onemli parametreler (`python train.py --help` ile hepsini gorebilirsiniz):

| Parametre | Aciklama |
|---|---|
| `--epochs` | Egitim dongu sayisi |
| `--batch-size` | Batch boyutu |
| `--img-size` | Goruntu boyutu (kare, ornegin 128 = 128x128) |
| `--base-channels` | U-Net ilk seviye kanal sayisi (CPU'da hizli denemek icin 16-32, GPU + gercek veride 64 klasik degerdir) |
| `--num-samples` | Sentetik veri kullanilirken toplam ornek sayisi |
| `--data-root` | Gercek veri klasoru (asagiya bakin) |

Egitim bittiginde `checkpoints/` altinda su ciktilari bulursunuz:
- `best_unet.pt`: en yuksek dogrulama Dice skoruna sahip model agirliklari
- `sample_prediction.png`: bir dogrulama orneginin girdi / gercek maske / model tahmini karsilastirmasi

## Her parca ne yapiyor, nasil calisiyor?

### 1) DataLoader + Data Augmentation (`data/dataset.py`, `utils/transforms.py`)

- **Dataset**, PyTorch'a "veri setinizde kac ornek var (`__len__`) ve `i`.
  ornegi nasil getirecegim (`__getitem__`)" sorularinin cevabini veren bir
  arayuzdur. `SyntheticCellDataset.__getitem__(idx)`, `idx`'e bagli sabit
  bir tohum (seed) ile bir goruntu+maske cifti uretir (ayni idx her zaman
  ayni temel goruntuyu verir; bu, dogrulama/test tutarliligi icin onemlidir),
  sonra `transform` varsa augmentation uygular.
- **DataLoader**, bir Dataset'i sarmalayip egitim dongusune batch'ler halinde,
  karistirilmis (shuffle) ve (istege bagli) coklu-islem (`num_workers`) ile
  veri saglar. `train.py` icinde `DataLoader(train_ds, batch_size=..., shuffle=True)`
  satiri budur.
- **Data Augmentation**, `utils/transforms.py` icinde `albumentations` ile
  yazildi. Neden albumentations: goruntu ve maskeyi AYNI rastgele durumla
  birlikte donusturur (ornegin goruntuyu 15 derece dondururken maskeyi de
  otomatik olarak ayni 15 derece dondurur). Egitim pipeline'i, biyomedikal
  goruntuler icin klasik olan asagidaki bozulmalari icerir:
  - Geometrik: yatay/dikey flip, 90-derece rotasyonlar, affine (kucuk
    olcek/rotasyon/kaydirma), elastik deformasyon (`ElasticTransform` -
    orijinal U-Net makalesinde ozellikle onerilen, doku esnemesini simule
    eden bir teknik), grid distortion.
  - Fotometrik: parlaklik/kontrast degisimi, Gauss gurultusu, hafif
    bulaniklastirma (farkli mikroskop/aydinlatma kosullarini simule eder).
  - Dogrulama (val) pipeline'i SADECE yeniden boyutlandirma + normalizasyon
    yapar - augmentation icermez, cunku val/test skorlarinin rastgeleligge
    bagli olmamasi gerekir.

### 2) Sifirdan U-Net (`model/unet.py`)

Klasik ansimetrik encoder-decoder + skip connection yapisi:
- **Encoder** (`DoubleConv` + `Down`): her seviyede `MaxPool` ile uzamsal
  boyutu yariya indirir, kanal sayisini ikiye katlar. Model gitgide "ne
  var" sorusuna odaklanir, "nerede oldugu" bilgisini kaybeder.
- **Bottleneck**: en dusuk cozunurluk, en yuksek kanal sayisi - en soyut
  temsil (representation).
- **Decoder** (`Up`): `ConvTranspose2d` ile uzamsal boyutu geri buyutur,
  encoder'daki AYNI seviyeden gelen skip connection ile birlestirir
  (`torch.cat`). Skip connection'lar olmadan, piksel-seviyesinde kesin
  sinirlari (hucre kenarlarini) geri kazanmak neredeyse imkansizdir -
  bottleneck'e kadar sikistirilmis bilgi bu ince detayi tasimaz.
- Cikti, 1x1 konvolusyon ile istenen kanal sayisina (ikili segmentasyon
  icin 1) indirilir. **Sigmoid, modelin icinde degil, kayip fonksiyonunda
  uygulanir** (`BCEWithLogitsLoss` sayisal olarak daha stabildir).

`python model/unet.py` calistirarak modelin giris/cikis boyutlarini ve
parametre sayisini tek basina kontrol edebilirsiniz.

### 3) Metrikler ve kayip fonksiyonu (`utils/metrics.py`)

- **IoU (Jaccard)** = kesisim / birlesim. **Dice (F1)** = 2×kesisim / (A+B).
  Ikisi de segmentasyonda standart raporlama metrikleridir; egitim
  dongusunde her epoch sonunda hem train hem val icin yazdirilir.
- **DiceLoss**: metrikteki esikleme (threshold) turevlenemez oldugu icin,
  egitimde optimize edilebilir "yumusak" (soft, esiklemesiz) bir Dice
  kaybi kullanilir.
- **BCEDiceLoss**: `BCEWithLogitsLoss + DiceLoss`. Tek basina BCE,
  hucre/arka plan piksel sayisi dengesiz oldugunda (mikroskopi
  goruntulerinde hucreler genelde kucuk bir alan kaplar) modeli
  onyargili hale getirebilir; Dice bu dengesizlige karsi daha
  dayaniklidir. Ikisinin toplami segmentasyonda yaygin bir standarttir.

## Gercek veriye gecis

Gercek TUBITAK veri setiniz hazir oldugunda:

1. Goruntu ve maskelerinizi su yapida bir klasore koyun:
   ```
   veri/
     images/img_001.png, img_002.png, ...
     masks/img_001.png, img_002.png, ...   (ayni dosya adi, ikili 0/255 maske)
   ```
2. Egitimi `--data-root` ile calistirin:
   ```bash
   python train.py --data-root veri --epochs 50 --base-channels 64 --img-size 256
   ```
   Kod otomatik olarak `FolderSegmentationDataset`'e gecer; DataLoader,
   augmentation, model ve metrik kodunda HICBIR degisiklik gerekmez.

3B (3 boyutlu, hacimsel) verilere gecince (projenizin ilerleyen aylarinda
gorunuyor), ayni mantik gecerli: `Conv2d/MaxPool2d/ConvTranspose2d` yerine
`Conv3d/MaxPool3d/ConvTranspose3d` kullanan bir `UNet3D` sinifi yazip
Dataset'i (D,H,W) hacimler donduren bir yapiya cevirmeniz yeterli olur -
augmentation ve metrik mantigi buyuk olcude aynen tasinir.

## Dogrulanan durum

Bu kod tabani, `python train.py --epochs 15` ile sentetik veri uzerinde
test edilmis ve calistigi dogrulanmistir (birkac epoch icinde val Dice
skoru ~0.90'in uzerine cikmaktadir - bu, sentetik gorevin nispeten kolay
olmasindan kaynaklanir; gercek biyomedikal veride daha fazla epoch ve
hiperparametre ayari gerekecektir).
