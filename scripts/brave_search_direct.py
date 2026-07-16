"""Direct Brave Search API fallback when the Brave MCP server fails."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def main() -> int:
    configure_console()
    args = parse_args()
    key = load_api_key()
    params = {
        "q": args.query,
        "count": str(args.count),
        "country": args.country,
        "search_lang": args.search_lang,
    }
    if args.result_filter:
        params["result_filter"] = args.result_filter
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if args.raw:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_results(payload))
    return 0


def configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--country", default="ALL")
    parser.add_argument("--search-lang", default="en")
    parser.add_argument("--result-filter", default="web")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--raw", action="store_true")
    return parser.parse_args()


def load_api_key() -> str:
    value = os.environ.get("BRAVE_API_KEY")
    if value:
        return value
    config = Path.home() / ".codex" / "config.toml"
    if config.exists():
        text = config.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'BRAVE_API_KEY\s*=\s*"?([^"\n\r]+)', text)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    raise RuntimeError("BRAVE_API_KEY not found in environment or ~/.codex/config.toml")


def format_results(payload: dict[str, Any]) -> str:
    rows = payload.get("web", {}).get("results", [])
    if not rows:
        return json.dumps(payload, ensure_ascii=False, indent=2)[:4000]
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        title = row.get("title", "(untitled)")
        url = row.get("url", "")
        description = row.get("description", "")
        lines.append(f"{index}. {title}\n   {url}\n   {description}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
