from __future__ import annotations

import argparse
import json
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


class HurgorInference:
    """Gerçek zamanlı çıkarım motoru."""

    def __init__(self, config_path: str = "config.yaml", model_path: str | None = None) -> None:
        self.config = self._load_config(config_path)
        self.model_path = model_path or self._default_model_path()
        self.model = self.load_model(self.model_path)
        self.conf_threshold = self.config["inference"]["conf_threshold"]
        self.iou_threshold = self.config["inference"]["iou_threshold"]
        self.device = self.config["inference"]["device"]

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _default_model_path(self) -> str:
        return str((self.config["paths"]["models"] + "/best.pt").replace("\\", "/"))

    def load_model(self, model_path: str) -> YOLO:
        """YOLO modelini yükler."""
        return YOLO(model_path)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü çıkarım için hazırlar."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Görüntüden tespit sonuçlarını temiz JSON benzeri yapıda döndürür."""
        prepared = self.preprocess_image(image)
        results = self.model(prepared, conf=self.conf_threshold, iou=self.iou_threshold, stream=False)
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().tolist())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = result.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].cpu().item())
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detections

    def predict_json(self, image: np.ndarray) -> str:
        """Sonuçları JSON string olarak döndürür."""
        return json.dumps(self.predict(image), indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HürGör çıkarım motoru")
    parser.add_argument("--config", default="config.yaml", help="Yapılandırma YAML dosyası")
    parser.add_argument("--model", default=None, help="Model yolu")
    parser.add_argument("--image", default=None, help="Görüntü yolu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = HurgorInference(config_path=args.config, model_path=args.model)
    if args.image is None:
        print("Lütfen --image ile bir görüntü yolu belirtin.")
        return
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {args.image}")
    print(inference.predict_json(image))


if __name__ == "__main__":
    main()
