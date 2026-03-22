from __future__ import annotations

import json
import sys

from openpkpd.model.symbolic_eta import prewarm_symbolic_caches


def main() -> int:
    try:
        warmed = prewarm_symbolic_caches()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {
        "count": len(warmed),
        "cache_dir": warmed[0]["cache_path"].rsplit("/", 1)[0] if warmed else None,
        "caches": warmed,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())