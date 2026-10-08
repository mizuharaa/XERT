"""Exercise the real HTTP backend and save every measured outcome, including failures.

Run the API first, then .venv/Scripts/python.exe scripts/reference_experiment.py.
This is a bounded CPU experiment, not a claim of benchmark convergence.
"""
import argparse
import json
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("docs/REFERENCE_EXPERIMENT.json"))
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--baseline-epochs", type=int, default=80)
    parser.add_argument("--act-epochs", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages from the output report")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        def request(method, url, **kwargs):
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        def run(kind, body):
            job = request("POST", f"/jobs/{kind}", json=body)
            print(kind, job["id"], flush=True)
            deadline = time.monotonic() + 1200
            previous = None
            while time.monotonic() < deadline:
                status = request("GET", f"/jobs/{job['id']}")
                progress = status.get("progress")
                if progress and progress != previous:
                    print(json.dumps(progress), flush=True)
                    previous = progress
                if status["state"] == "completed":
                    return status["result"]
                if status["state"] in {"failed", "cancelled", "interrupted"}:
                    raise RuntimeError(json.dumps(status))
                time.sleep(.5)
            request("POST", f"/jobs/{job['id']}/cancel")
            raise TimeoutError("Experiment job exceeded its 20-minute limit; cancellation requested")

        started = time.time()
        result = {"started_at": started, "health": request("GET", "/health"),
                  "environment": request("GET", "/environment"),
                  "scope": "Small CPU reference experiment through real HTTP API; all outcomes retained"}
        if args.resume:
            previous = json.loads(args.output.read_text("utf-8"))
            if previous["environment"] != result["environment"]:
                raise ValueError("Cannot resume against a changed simulator")
            result = previous
            result.setdefault("resumed_at", []).append(started)

        def save():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

        if "collection" not in result:
            result["collection"] = run("collect", {"seeds": list(range(100, 100 + args.episodes))})
        elif result["collection"]["episodes"] != args.episodes:
            raise ValueError("Resume episode count differs from the saved collection")
        save()
        models = []
        for model_type, epochs in [("baseline_mlp", args.baseline_epochs), ("act", args.act_epochs)]:
            if model_type in result:
                saved = result[model_type]
                if saved["metrics"]["config"]["epochs"] != epochs:
                    raise ValueError("Resume epoch count differs from saved training")
                request("GET", f"/models/{saved['id']}")
                models.append(saved["id"])
                continue
            result[model_type] = run("train", {"episode_ids": result["collection"]["episode_ids"],
                                               "config": {"model_type": model_type, "epochs": epochs,
                                                          "hidden_dim": 64, "batch_size": 64, "chunk_size": 8}})
            models.append(result[model_type]["id"])
            save()
        result["benchmark"] = run("benchmark", {"seeds": [20000, 20001, 20002], "model_ids": models,
                                                 "conditions": [{"friction": 1.0}, {"friction": .6},
                                                                {"friction": 1.0, "delay_steps": 2}]})
        first = result["collection"]["episode_ids"][0]
        result["replay"] = request("GET", f"/episodes/{first}/replay")
        result["elapsed_seconds"] = time.time() - result["started_at"]
        result["last_session_elapsed_seconds"] = time.time() - started
        save()
        print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
