from __future__ import annotations

import json
import sys
import random
from pathlib import Path

import cv2
import numpy as np

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Sadece "dataset" klasörü içindeki GERÇEK görüntülerde arıyoruz.
    dataset_path = repo_root / "dataset"
    image_candidates = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.JPG"))

    if not image_candidates:
        print("[ERROR] Test resmi bulunamadı. Lütfen dataset/ klasörünün dolu olduğundan emin olun.")
        return 1

    # =========================================================
    # YENİLİK: Her çalıştırmada havuzdan rastgele bir resim seç!
    # =========================================================
    image_path = random.choice(image_candidates)
    print(f"[INFO] Test resmi olarak RASTGELE şu dosya seçildi:\n{image_path}\n")

    img_array = np.fromfile(str(image_path), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        print(f"[ERROR] Görüntü decode edilemedi: {image_path}")
        return 1

    try:
        from src.hurgor.vision import HurgorVision
    except Exception as exc:
        print("[ERROR] import basarisiz: from src.hurgor.vision import HurgorVision")
        print(f"[ERROR] detay: {type(exc).__name__}: {exc}")
        return 1

    print("[INFO] HurgorVision ONNX Motoru Başlatılıyor...")
    vision = HurgorVision(model_path="models/hurgor_final.onnx")

    print("[INFO] Çıkarım (Inference) yapılıyor...\n")
    detections = vision.run_inference(frame, frame_id=1)

    print("==================================================")
    print("      TEKNOFEST 2026 - HÜRGÖR TESPİT RAPORU       ")
    print(" Beklenen Sınıflar: person, car, uap, uai, goal   ")
    print("==================================================\n")
    
    if not detections:
        print("[WARN] Tespit yok. Model bu rastgele resimde hedef sınıflardan birini bulamadı.")
        return 0

    results_json = []
    for i, det in enumerate(detections, start=1):
        cls = det.get("class")
        bbox = det.get("bbox")
        rel_x = det.get("rel_x")
        rel_y = det.get("rel_y")
        conf = det.get("conf", 0.0)

        print(f"[{i}] SINIF: {cls.upper(): <8} | GÜVEN: %{conf*100:.1f} | BBOX: {[round(c, 1) for c in bbox]}")
        print(f"    -> 3D Göreceli Metrik Konum (X, Y): ({rel_x:.2f}m, {rel_y:.2f}m)\n")
        
        results_json.append({
            "class_name": cls,
            "bounding_box": [round(c, 1) for c in bbox],
            "position_meters": {"x": round(rel_x, 2), "y": round(rel_y, 2)},
            "confidence": round(conf, 3)
        })

    print("=== ŞARTNAME İÇİN ÖRNEK JSON (API) ÇIKTISI ===")
    print(json.dumps(results_json, indent=4))

    return 0

if __name__ == "__main__":
    sys.exit(main())