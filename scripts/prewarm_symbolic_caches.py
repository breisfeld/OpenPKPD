#!/usr/bin/env python3
from __future__ import annotations

import json

from openpkpd.model.symbolic_eta import prewarm_symbolic_caches


def main() -> int:
    warmed = prewarm_symbolic_caches()
    print(json.dumps(warmed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
