"""Check the running patched localhost API, including concurrent PNG requests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import httpx
from PIL import Image


def main():
    output = Path("docs/audit/HTTP_POST_FIX.json")
    with httpx.Client(base_url="http://127.0.0.1:8766", timeout=60) as client:
        report = {"health": client.get("/health").json(),
                  "environment": client.get("/environment").json(),
                  "docs_status": client.get("/docs").status_code}
        response = client.post("/sessions", json={"seed": 123, "config": {"horizon": 6}})
        response.raise_for_status()
        identifier = response.json()["id"]
        try:
            client.post(f"/sessions/{identifier}/step", json={"expert": True, "repeat": 6}).raise_for_status()
            saved = client.post(f"/sessions/{identifier}/save")
            saved.raise_for_status()
            episode = saved.json()["id"]
            report["replay"] = client.get(f"/episodes/{episode}/replay").json()
            assert report["replay"]["matched"]
            urls = [f"/sessions/{identifier}/frame.png", f"/episodes/{episode}/frame.png"]
            def render(url):
                frame = client.get(url)
                frame.raise_for_status()
                with Image.open(io.BytesIO(frame.content)) as image:
                    assert image.size == (640, 480)
                    image.verify()
                return {"path": url, "status": frame.status_code, "bytes": len(frame.content),
                        "sha256": hashlib.sha256(frame.content).hexdigest()}
            with ThreadPoolExecutor(max_workers=2) as pool:
                report["concurrent_render_requests"] = list(pool.map(render, urls))
            invalid = client.post("/jobs/evaluate", json={"seeds": [-1]})
            assert invalid.status_code == 422
            report["invalid_seed_status"] = invalid.status_code
        finally:
            client.delete(f"/sessions/{identifier}").raise_for_status()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"HTTP, replay and concurrent PNG checks passed: {output}")


if __name__ == "__main__":
    main()
