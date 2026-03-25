"""Human-readable viewer for GraphRAG JSONL logs.

Usage examples (PowerShell):
    python ./utils/log_trace_viewer.py --trace-id <TRACE_ID>
    python ./utils/log_trace_viewer.py --file output/logs/debug.log --tail 100
    python ./utils/log_trace_viewer.py --stage final_answer --event prompt_input --tail 50
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any


DEFAULT_LOG_FILE = os.path.join("output", "logs", "app.log")


def _safe_parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
        if isinstance(value, dict):
            return value
        return None
    except Exception:
        return None


def _fmt_ts(raw: str | None) -> str:
    if not raw:
        return "-"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw


def _trim_text(value: Any, max_len: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [TRUNCATED]"


def _load_events(path: str, tail: int | None = None) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if tail and tail > 0:
        lines = lines[-tail:]

    out: list[dict[str, Any]] = []
    for ln in lines:
        item = _safe_parse_json_line(ln)
        if item:
            out.append(item)
    return out


def _filter_events(
    events: list[dict[str, Any]],
    trace_id: str | None,
    stage: str | None,
    event_name: str | None,
) -> list[dict[str, Any]]:
    out = []
    for e in events:
        if trace_id and str(e.get("trace_id", "")) != trace_id:
            continue
        if stage and str(e.get("stage", "")) != stage:
            continue
        if event_name and str(e.get("event", "")) != event_name:
            continue
        out.append(e)
    return out


def _print_grouped(events: list[dict[str, Any]], max_payload_chars: int, show_payload: bool) -> None:
    if not events:
        print("No matching log events found.")
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        grouped[str(e.get("trace_id", "NO_TRACE"))].append(e)

    for trace_id, group in grouped.items():
        print("=" * 88)
        print(f"TRACE ID: {trace_id}")
        print(f"EVENT COUNT: {len(group)}")
        print("=" * 88)

        def _sort_key(x: dict[str, Any]) -> str:
            return str(x.get("timestamp", ""))

        for idx, e in enumerate(sorted(group, key=_sort_key), 1):
            ts = _fmt_ts(e.get("timestamp"))
            lvl = e.get("level", "-")
            route = e.get("route", "-")
            stage = e.get("stage", "-")
            evn = e.get("event", "-")
            dur = e.get("duration_ms", "-")
            msg = e.get("message", "")

            print(f"\n[{idx}] {ts} | {lvl} | route={route} | stage={stage} | event={evn} | ms={dur}")
            print(f"message: {msg}")

            if show_payload and "payload" in e:
                payload_text = _trim_text(e.get("payload"), max_payload_chars)
                print("payload:")
                print(payload_text)

            if "exception" in e:
                ex_text = _trim_text(e.get("exception"), max_payload_chars)
                print("exception:")
                print(ex_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GraphRAG JSONL logs into readable trace view.")
    parser.add_argument("--file", default=DEFAULT_LOG_FILE, help="Path to log file (default: output/logs/app.log)")
    parser.add_argument("--trace-id", default=None, help="Filter by trace_id")
    parser.add_argument("--stage", default=None, help="Filter by stage")
    parser.add_argument("--event", default=None, help="Filter by event")
    parser.add_argument("--tail", type=int, default=300, help="Read only last N lines (default: 300)")
    parser.add_argument(
        "--max-payload-chars",
        type=int,
        default=4000,
        help="Max characters per payload/exception output (default: 4000)",
    )
    parser.add_argument(
        "--hide-payload",
        action="store_true",
        help="Hide payload content and show only metadata timeline",
    )

    args = parser.parse_args()

    events = _load_events(args.file, args.tail)
    filtered = _filter_events(events, args.trace_id, args.stage, args.event)
    _print_grouped(filtered, args.max_payload_chars, show_payload=not args.hide_payload)


if __name__ == "__main__":
    main()
