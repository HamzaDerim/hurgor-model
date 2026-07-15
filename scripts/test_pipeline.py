from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
import sys

import httpx
import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _start_mock_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    from src.hurgor.mock_server import create_app

    host = "127.0.0.1"
    port = 8765
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread, f"http://{host}:{port}"


async def _wait_for_server(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        for _ in range(60):
            try:
                response = await client.get("/api/status")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError("mock server did not become ready in time")


async def _run_system_test(base_url: str) -> dict[str, object]:
    from src.hurgor.client import CompetitionAPI
    from src.hurgor.config import ClientSettings
    from src.hurgor.inference import PipelineInferenceEngine

    settings = ClientSettings(
        base_url=base_url,
        api_contract="local",
        frame_endpoint="/api/frames/next",
        prediction_endpoint="/api/predictions",
        progress_endpoint="/api/status",
        sla_seconds=2.0,
        inference_timeout_seconds=5.0,
        inference_startup_timeout_seconds=10.0,
        yolo_onnx_path="models/hurgor_final.onnx",
        reference_images_dir="assets/references",
        user_url=f"{base_url}/users/1/",
    )

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        api = CompetitionAPI(settings, client)
        frame = await api.fetch_frame()
        image_bytes = await api.fetch_image(frame.image_url)
        engine = PipelineInferenceEngine.from_settings(settings)
        outcome = engine.infer_timed(frame, image_bytes, settings.user_url)
        await api.submit(outcome.prediction)
        progress = await api.fetch_progress()

    return {
        "frame_url": frame.url,
        "image_url": frame.image_url,
        "detected_objects": len(outcome.prediction.detected_objects),
        "detected_translations": len(outcome.prediction.detected_translations),
        "prediction": outcome.prediction.model_dump(mode="json"),
        "progress": progress,
        "timings_ms": outcome.timings_ms,
    }


def main() -> int:
    server, thread, base_url = _start_mock_server()
    try:
        asyncio.run(_wait_for_server(base_url))
        result = asyncio.run(_run_system_test(base_url))
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
