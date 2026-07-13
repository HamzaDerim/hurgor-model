from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


class YOLO11TrainingPipeline:
    """YOLO11 eğitim pipeline'ı için modüler sınıf."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.data = self.config["paths"]["dataset"]
        self.model_name = self.config["training"]["model_name"]
        self.model = self.load_model()

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def load_model(self) -> YOLO:
        """YOLO11s modelini yükler."""
        return YOLO(self.model_name)

    def train(self) -> None:
        """Eğitim sürecini başlatır."""
        training_cfg = self.config["training"]
        self.model.train(
            data=self.data,
            epochs=training_cfg["epochs"],
            imgsz=training_cfg["imgsz"],
            batch=training_cfg["batch"],
            device=self.config["inference"]["device"],
            project="runs/train",
            name="hurgor_yolo11",
            exist_ok=True,
            pretrained=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HürGör YOLO11 eğitim betiği")
    parser.add_argument("--config", default="config.yaml", help="Yapılandırma YAML dosyası")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = YOLO11TrainingPipeline(config_path=args.config)
    pipeline.train()


if __name__ == "__main__":
    main()
