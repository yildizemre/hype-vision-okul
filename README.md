# Hype Vision · Okul Giriş

Okul kapısındaki IP kameradan öğrencileri yüz tanıma ile tanır, giriş/çıkış saatini kaydeder ve veliye WhatsApp ile bildirir. Tamamen okulun kendi sunucusunda (on-prem) çalışır; fotoğraf ve yüz verisi dışarı çıkmaz.

## Gereksinimler

- Python **3.10 – 3.12**
- NVIDIA GPU önerilir (RTX 3050 ve üstü). Sadece sürücü gerekir (550+); CUDA/cuDNN pip ile gelir.
- GPU yoksa CPU ile de çalışır (`requirements-cpu.txt`), pilot için yeterli.

## Kurulum — Windows

`BASLAT.bat` dosyasına çift tıkla. İlk seferde ortamı kurar, GPU varsa GPU paketlerini seçer, modelleri indirir (~2 GB) ve http://localhost:8000 adresini açar.

## Kurulum — Ubuntu 22.04 / 24.04

```bash
sudo apt install -y git
git clone https://github.com/yildizemre/hype-vision-okul.git /opt/hypevision
cd /opt/hypevision
chmod +x baslat.sh
./baslat.sh
```

Açılışta otomatik başlaması için `hypevision.service` dosyasındaki talimatları uygula (systemd).

Elle kurulum:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt      # GPU yoksa: requirements-cpu.txt
./venv/bin/python app.py
```

İlk giriş: `hype` / `hype` — girişten sonra **Ayarlar → Şifremi değiştir**.

## Kayıt fotoğrafı rehberi (doğruluk için en önemli kısım)

- Öğrenci başına **3–5 fotoğraf**: en az 1 telefon fotoğrafı (önden, iyi ışık) + 2–3 kare **kapı kamerasından** ("Yüz Kaydet → RTSP'den görüntü al").
- Gözlük kullanan öğrencinin hem gözlüklü hem gözlüksüz fotoğrafı olsun. Güneş gözlüğü / maske tanımayı bozar.
- Fotoğraflar 1 yılda bir yenilenmeli (çocuk yüzü hızlı değişir). "Onay Bekleyen" sayfasından yapılan onaylar galeriyi kendiliğinden günceller.
- Eğitim sonrası "En benzer iki kişi" değeri 0.40'a yaklaşıyorsa (ikizler, kardeşler) bu öğrencilere ekstra foto ekle.

## Özellikler

- Çoklu kamera (giriş / çıkış), tek GPU kuyruğu, kamera koptu uyarısı
- Kişi takibi + kare oylaması ile "GELDİ" kaydı; karar verilen kişide tanıma atlanır
- Ders saati / geç geldi, "henüz görülmeyenler" uyarısı, tatil takvimi
- Excel + fotoğraf zip ile toplu öğrenci kaydı
- Onay bekleyen (emin olunamayan) ve bilinmeyen yüzler; onaylanan foto galeriye eklenir
- Öğrenci / beyaz liste (personel) / kara liste (alarm + yöneticiye mesaj)
- Yoklama, elle "geldi/çıkış", öğretmen teyidi ile "gelmedi" bildirimi
- Raporlar (aylık devam, geç kalanlar, sınıf karşılaştırması), Excel/PDF
- Kullanıcı rolleri: yönetici, müdür yardımcısı, sınıf öğretmeni (sadece kendi sınıfı)
- WhatsApp: Whapi.cloud veya Meta Cloud API, mesaj geçmişi, test mesajı, toplu duyuru, veli sorgu botu
- KVKK: saklama süreleri, mezun öğrenci için kalıcı silme, işlem kaydı, canlı görüntü kaydedilmez
- Tek dosya yedek indir / geri yükle

## Modeller

- Yüz tespiti: [YuNet](https://github.com/opencv/opencv_zoo) (MIT)
- Yüz tanıma: [AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) (Apache 2.0, ticari kullanıma uygun)

`data/` (öğrenci verisi) ve `models/` git'e eklenmez.
