from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


class ModelExporter:
    """YOLO modelini TensorRT/engine formatına dönüştürür."""

    def __init__(self, model_path: str, output_path: str | None = None) -> None:
        self.model_path = model_path
        self.output_path = output_path or str(Path(model_path).with_suffix(".engine"))

    def export(self, half: bool = True, int8: bool = False, workspace: int = 4096) -> None:
        """TensorRT optimizasyonu için export işlemini başlatır."""
        model = YOLO(self.model_path)
        model.export(
            format="engine",
            imgsz=640,
            half=half,
            int8=int8,
            workspace=workspace,
            simplify=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HürGör YOLO model export betiği")
    parser.add_argument("--model", default="runs/train/hurgor_yolo11/weights/best.pt", help="Eğitilmiş model yolu")
    parser.add_argument("--output", default=None, help="Çıktı engine yolu")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--int8", action="store_true", help="INT8 export")
    parser.add_argument("--workspace", type=int, default=4096, help="TensorRT workspace MB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = ModelExporter(model_path=args.model, output_path=args.output)
    exporter.export(half=args.half, int8=args.int8, workspace=args.workspace)


if __name__ == "__main__":
    main()
