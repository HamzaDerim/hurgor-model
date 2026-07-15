import os
from ultralytics import YOLO

def main():
    # Mutlak yolları belirle
    proje_dizini = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_yolu = os.path.join(proje_dizini, "dataset", "data.yaml")
    models_dizini = os.path.join(proje_dizini, "models")
    
    # Models klasörü yoksa oluştur
    os.makedirs(models_dizini, exist_ok=True)
    
    print("🚀 HürGör Otonom Eğitim Motoru Başlatılıyor...")
    print(f"📁 Kullanılan YAML Yolu: {yaml_yolu}")
    print(f"💾 Çıktılar şuraya kaydedilecek: {models_dizini}/hurgor_v1")
    
    # YOLO11 modelini indir ve yükle (Empty model klasörü sorununu çözer)
    model = YOLO("yolo11s.pt")
    
    # Eğitimi başlat (Windows için kurşun geçirmez ayarlar)
    model.train(
        data=yaml_yolu,
        epochs=100,
        imgsz=640,
        batch=8,                # RTX 3060 için güvenli VRAM sınırı
        workers=0,              # WINDOWS KRİTİK: Multiprocessing donmasını engeller
        project=models_dizini,  # Çıktıları doğrudan 'models' klasörüne atar
        name="hurgor_v1",       # Modelin kayıt ismi
        device="0"              # Sadece NVIDIA GPU kullanımına zorla
    )

if __name__ == '__main__':
    main()
