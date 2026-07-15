import sys
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
# Modüllerin bulunabilmesi için ana dizini sisteme ekle
proje_dizini = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proje_dizini)


# Proje kök dizinini sisteme tanıtıyoruz ki 'src' klasörünü bulabilsin
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Eski 'scripts.vision' yerine, yeni evimiz olan 'src.hurgor.vision'dan çağırıyoruz
from src.hurgor.vision import HurgorVision, VisualOdometry, TargetTracker

def main():
    print("🎬 HürGör Video İşleme: Takip (Tracking) ve Hız Kestirimi Başlatılıyor...")
    
    vision = HurgorVision()
    vo = VisualOdometry()
    
    # Telemetri yükleme (Varsa)
    telemetry_file = os.path.join(proje_dizini, "THYZ_2026_Ornek_Veri_1_translation.csv")
    vision.load_telemetry(telemetry_file)

    model_path = os.path.join(proje_dizini, "models", "hurgor_v1", "weights", "best.pt")
    model = YOLO(model_path)

    # Video Girdi ve Çıktı Ayarları
    video_input = os.path.join(proje_dizini, "ornek_video.mp4") # İŞLENECEK VİDEO DOSYASI
    video_output = os.path.join(proje_dizini, "runs", "detect", "hurgor_cikis_videosu.mp4")
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print(f"❌ HATA: '{video_input}' bulunamadı! Lütfen proje ana dizinine 'ornek_video.mp4' adında bir video ekleyin.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 
    
    # Hız Kestirim Motorunu (Tracker) başlatıyoruz
    tracker = TargetTracker(fps=fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    
    results_list = []
    frame_id = 0

    print(f"📹 Video işleniyor: Çözünürlük: {width}x{height}, FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        results = model.predict(source=frame, conf=0.5, verbose=False)
        current_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2
                
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id]
                conf = float(box.conf[0].cpu().numpy())
                
                # 3B Diferansiyel Koordinat Çıkarımı
                rel_X, rel_Y, rel_Z = vision.get_3d_coordinates(u, v, frame_id, z_depth_estimate=25.0)
                
                current_detections.append({
                    'class': cls_name,
                    'conf': conf,
                    'rel_x': rel_X,
                    'rel_y': rel_Y,
                    'bbox': (x1, y1, x2, y2)
                })
        
        # 1. Odometri Güncellemesi
        uav_global_x, uav_global_y = vo.update_position(current_detections)
        
        # 2. Hedef Takibi (ID) ve Hız (km/h) Hesaplama
        tracked_detections = tracker.update(current_detections, frame_id, uav_global_x, uav_global_y)
        
        # GPS-Denied durumu için varsayılan başlangıç referansı (Örn: AFSÜ/Afyon bölgesi koordinatları)
        base_lat = vision.telemetry_data.get(frame_id, {}).get('lat', 38.7507)
        base_lon = vision.telemetry_data.get(frame_id, {}).get('lon', 30.5567)
        
        # İHA'nın anlık WGS84 GPS koordinatları
        uav_lat, uav_lon = vision.add_meters_to_gps(base_lat, base_lon, uav_global_x, uav_global_y)
        
        for det in tracked_detections:
            target_lat, target_lon = vision.add_meters_to_gps(uav_lat, uav_lon, det['rel_x'], det['rel_y'])
            
            # Verileri Rapor İçin Listeye Ekle
            results_list.append({
                "Frame_ID": frame_id,
                "Target_ID": f"{det['class']}_{det['track_id']}",
                "Class": det['class'],
                "Confidence": round(det['conf'], 2),
                "Speed_kmh": round(det['speed_kmh'], 1),
                "Rel_X_m": round(det['rel_x'], 2),
                "Rel_Y_m": round(det['rel_y'], 2),
                "UAV_Lat": round(uav_lat, 7),
                "UAV_Lon": round(uav_lon, 7),
                "Target_Lat": round(target_lat, 7),
                "Target_Lon": round(target_lon, 7)
            })
            
            # Kutuları ve ID/HIZ bilgisini videoya çiz!
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID:{det['track_id']} {det['class']} [{det['speed_kmh']:.1f} km/h]"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # İHA'nın Anlık Konumunu (HUD) Ekrana Bas
        cv2.putText(frame, f"GPS-DENIED VO POS -> X:{uav_global_x:.1f} Y:{uav_global_y:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"LAT: {uav_lat:.6f} LON: {uav_lon:.6f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)
        
        if frame_id % 30 == 0:
            print(f"⏳ İşlenen Kare: {frame_id} (Yaklaşık {frame_id//fps} saniye tamamlandı)")

    cap.release()
    out.release()
    
    if results_list:
        df = pd.DataFrame(results_list)
        # Raporu videonun ismiyle kaydet (ornek_video_koordinat_raporu.csv)
        video_adi = os.path.splitext(os.path.basename(video_input))[0]
        csv_out = os.path.join(proje_dizini, f"{video_adi}_koordinat_raporu.csv")
        
        df.to_csv(csv_out, index=False)
        print(f"\n✅ İŞLEM TAMAMLANDI!")
        print(f"🎥 İzlenebilir Çıktı Videosu: {video_output}")
        print(f"📊 Güncel Takip & Hız Raporu: {csv_out}")

if __name__ == '__main__':
    main()