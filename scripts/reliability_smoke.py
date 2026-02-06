#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.synapse_brain import SynapseBrain


def _require_ok(name: str, payload: dict):
    if payload.get("error"):
        raise RuntimeError(f"{name} failed: {payload['error']}")


def main() -> int:
    brain = SynapseBrain()
    try:
        checks = brain.preflight_check()
        if not checks.get("ok"):
            print(json.dumps({"stage": "preflight", "checks": checks}, ensure_ascii=True, indent=2))
            return 2

        with tempfile.TemporaryDirectory() as td:
            doc_path = Path(td) / "smoke_doc.txt"
            doc_path.write_text(
                """
                The catalyst achieved 92.3% conversion at 350 K with 0.5 bar oxygen partial pressure.
                Repeated trials (n=5) showed standard deviation 1.1% and stability over 48 hours.
                Compared with baseline catalyst, activity improved by 37% under identical flow conditions.
                """.strip(),
                encoding="utf-8",
            )

            upload = brain.upload(str(doc_path))
            _require_ok("upload", upload)

            ask = brain.ask("What quantitative results are reported in the uploaded study?")
            _require_ok("ask", ask)

            explore = brain.explore("catalyst stability mechanisms", depth=1)
            _require_ok("explore", explore)

            trace = brain.trace("catalyst conversion")
            if not trace.get("found"):
                raise RuntimeError(f"trace failed: {trace.get('message', 'no memories found')}")

        events = []
        run_log = Path(brain.run_log_path)
        if run_log.exists():
            for line in run_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))

        event_types = {e.get("event_type") for e in events}
        required = {"upload", "ask", "explore"}
        missing = sorted(required - event_types)
        if missing:
            raise RuntimeError(f"run log missing events: {missing}")

        summary = {
            "ok": True,
            "run_log": str(run_log),
            "event_types": sorted(event_types),
            "memories": brain.get_stats().get("memory", {}).get("total_memcells", 0),
        }
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return 0
    finally:
        brain.close()


if __name__ == "__main__":
    sys.exit(main())
