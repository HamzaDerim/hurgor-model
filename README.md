# 🚀 HürGör AI: Teknofest 2026 Havacılıkta Yapay Zeka Sistemi

![Teknofest](https://img.shields.io/badge/Teknofest-2026-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO11](https://img.shields.io/badge/Model-YOLO11-yellow)
![Status](https://img.shields.io/badge/Status-Flight_Ready-success)

HürGör, **Teknofest 2026 Havacılıkta Yapay Zeka Yarışması** için geliştirilmiş, uç cihazlarda (edge computing) gerçek zamanlı çalışabilen, otonom hedef tespiti ve pozisyon kestirimi yapabilen ileri düzey bir İHA yapay zeka boru hattıdır (pipeline).

Proje; klasik bilgisayarlı görü yaklaşımlarının ötesine geçerek **ileri matematiksel modellemeler**, **GPS-Denied (GPS'siz) ortamlarda otonom navigasyon** ve **asenkron iş parçacıklı (multi-threaded) yazılım mimarisi** kullanılarak inşa edilmiştir.

---

## 🧠 1. Yapay Zeka ve Görüntü İşleme (YOLO11)
Sistem, gerçek zamanlı hedef tespiti (İnsan, Taşıt, UAP, UAI) için sınıfının en yenisi olan **YOLO11** mimarisini kullanmaktadır.
* **Edge Optimizasyonu:** Model; TensorRT/ONNX formatında derlenerek düşük donanımlı cihazlarda maksimum FPS (Gecikme süresi < 300ms) verecek şekilde optimize edilmiştir.
* **Sıfır Gürültü (Zero-Noise):** Hatalı tespitleri (false-positive) elemek için YOLO11 çıktıları matematiksel filtrelerden geçirilmektedir.

## 📐 2. İleri Matematiksel Modelleme
HürGör'ün "beyni", basit 2D piksel okumalarının ötesinde akademik düzeyde matematiksel temellere dayanır:
* **SE(3) Manifoldları ve Diferansiyel Geometri:** Kamera matrisi (K) ile Dünya koordinatları arasındaki hizalama ve otonom İHA'nın pozisyon kestirimi SE(3) üzerinde modellenerek kamera sapmaları (drift) minimize edilmiştir.
* **3D IoU (Hacimsel Kestirim):** 2B bounding-box verisi kullanılarak referans nesnelerin uzaydaki hacimleri çıkartılmış ve hedefe olan milimetrik mesafeler hesaplanmıştır.
* **Topolojik Veri Analizi (TDA):** Görüntüdeki parazitleri topolojik kararlılıkla ayırmak için "persistence homology" yöntemleri sisteme entegre edilmiştir.
* **Bilgi Damıtma (Knowledge Distillation):** Karmaşık öğretmen modelinden damıtılan bilgi, HürGör'ün uç cihazda koşan hafif öğrenci modeline aktarılmış, böylece hem hız hem de yüksek doğruluk elde edilmiştir.

## 🛰️ 3. GPS-Denied Navigasyon & Dead Reckoning
Yarışma senaryolarındaki en zorlu test olan elektronik harp / GPS sinyal kesintisi (Jamming) durumları için HürGör **tam otonom navigasyon** yeteneğine sahiptir:
* **Origin-Lock (Orjin Kilidi):** GPS sinyali kaybolduğu an (`None` düştüğünde) sistem son geçerli konumu `(x=0, y=0, z=0)` referans noktası olarak kilitler.
* **Optical Flow & Dead Reckoning:** GPS verisi olmadan, piksellerin optik akışı (optical flow) hesaplanarak İHA'nın göreceli hızı ($v$) ve yönelimi bulunur. $x_{new} = x_{prev} + (v \times \Delta t)$ diferansiyel formülüyle "İçsel Kestirim" (Dead Reckoning) çalıştırılır.
* **Graceful Degradation:** Sistem, GPS'siz modda sapmanın (drift) çok arttığını hissederse ağır işlemleri otomatik olarak kapatıp "hayatta kalma" moduna geçer.

## 🗺️ 4. Trajectory Mapping (Otonom Haritalama)
GPS-Denied modunda uçan sistem, kendi içsel kestirimiyle yolunu bulurken aynı zamanda arkasında "Breadcrumb" (Ekmek kırıntısı) bırakır. 
* 2250 karelik (frame) görev uçuşu boyunca hesaplanan her $X, Y, Z$ koordinatı kaydedilir.
* Oturum sonunda sistem, uçuşun rotasını çıkaran `trajectory_map.json` dosyasını otomatik olarak oluşturur. 

## ⚙️ 5. Yazılım Mimarisi (Multi-Threaded Pipeline)
Yarışma sunucusu (P0 protokolü) ile mükemmel senkronizasyon sağlayan, kilitlenmeyen "State Machine" tasarımı:
* **Producer-Worker-Consumer:** Veri çekme, AI tahmini ve sonuç gönderme işlemleri 3 ayrı asenkron thread üzerinde koşar.
* **State Machine (Durum Makinesi):** - `[CALIBRATION_PHASE]`: İlk 450 karede sistem sunucuya tahmin atmaz (Registration Packet atar), sadece referans noktalarıyla kendi kamera matrisini kalibre eder.
  - `[INFERENCE_PHASE]`: 451. kareden itibaren otomatik otonom tahmin moduna geçer.
* **Exponential Backoff:** Ağ kopmalarında 1sn, 2sn, 4sn, 8sn bekleme algoritmaları ile bağlantı direncini maksimumda tutar.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install -r requirements.txt
