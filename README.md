# Hype Vision · Okul Giriş

Okul kapısındaki IP kameradan öğrencileri yüz tanıma ile tanır, giriş saatini kaydeder ve veliye WhatsApp ile bildirir. Tamamen okulun kendi sunucusunda (on-prem) çalışır; fotoğraf ve yüz verisi dışarı çıkmaz.

## Kurulum (Windows)

Python 3.12 kurulu olmalı. `BASLAT.bat` dosyasını çift tıkla:
ortamı kurar, modelleri indirir (ilk seferde ~2 GB), sunucuyu başlatır ve http://localhost:8000 adresini açar.

İlk giriş: `hype` / `hype` — girişten sonra Ayarlar → Şifremi değiştir.

## Özellikler

- Canlı RTSP izleme, kişi takibi + kare oylaması ile "GELDİ" kaydı
- Onay bekleyen (emin olunamayan) ve bilinmeyen yüzler; onaylanan foto galeriye eklenir
- Öğrenci / beyaz liste (personel) / kara liste (alarm + yöneticiye mesaj)
- Yoklama, elle "geldi", öğretmen teyidi ile "gelmedi" bildirimi, CSV dışa aktarma
- Kullanıcı rolleri: yönetici, müdür yardımcısı, sınıf öğretmeni (sadece kendi sınıfı)
- WhatsApp: Whapi.cloud veya Meta Cloud API (onaylı şablon), mesaj geçmişi, test mesajı, webhook
- KVKK: saklama süreleri, mezun öğrenci için kalıcı silme, canlı görüntü kaydedilmez
- Tek dosya yedek indir / geri yükle

## Modeller

- Yüz tespiti: [YuNet](https://github.com/opencv/opencv_zoo) (MIT)
- Yüz tanıma: [AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) (Apache 2.0, ticari kullanıma uygun)

`data/` (öğrenci verisi) ve `models/` git'e eklenmez.
