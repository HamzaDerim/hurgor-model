from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=env_path)


def _build_auth_header(team_name: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{team_name}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


async def check_connection() -> int:
    _load_env()
    url = os.getenv("EVALUATION_SERVER_URL")
    team = os.getenv("TEAM_NAME")
    password = os.getenv("PASSWORD")

    if not url:
        print("BAGLANTI HATASI: EVALUATION_SERVER_URL bulunamadi.")
        return 2
    if not team or not password:
        print("BAGLANTI HATASI: TEAM_NAME veya PASSWORD bulunamadi.")
        return 2

    headers = _build_auth_header(team, password)
    print(f"--- Sunucuya baglaniliyor: {url} ---")

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=5.0)
            print(f"Sunucudan Yanit: {response.status_code}")

            # 200/401 means server is reachable; any <500 still confirms network path.
            if response.status_code in {200, 401} or response.status_code < 500:
                print("BAGLANTI BASARILI: Sunucu ayakta ve yanit veriyor.")
                return 0

            print("SUNUCU HATASI: Sunucu yanit veriyor ama 5xx donuyor.")
            return 1
    except httpx.TimeoutException as exc:
        print(f"BAGLANTI HATASI: Zaman asimi olustu. Hata: {exc}")
    except httpx.ConnectError as exc:
        print(f"BAGLANTI HATASI: Sunucuya baglanilamadi. Hata: {exc}")
    except httpx.HTTPError as exc:
        print(f"BAGLANTI HATASI: HTTP istemci hatasi. Hata: {exc}")
    except Exception as exc:  # Last-resort guard for unexpected runtime failures.
        print(f"BAGLANTI HATASI: Beklenmeyen hata. Hata: {exc}")

    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(check_connection()))
