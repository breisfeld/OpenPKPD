"""Helpers for compact clickable file and URL labels in rich-text Qt widgets."""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREFERRED_PATH_ANCHORS = ("shared_data", "catalog", "examples", "src", "tests")


def compact_path_label(path: str | Path, *, max_parts: int = 4) -> str:
    """Return a shortened human-readable label for a filesystem path."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(_REPO_ROOT):
        parts = list(resolved.relative_to(_REPO_ROOT).parts)
    else:
        parts = [part for part in resolved.parts if part != resolved.anchor]

    for anchor in _PREFERRED_PATH_ANCHORS:
        if anchor in parts:
            parts = parts[parts.index(anchor) :]
            break

    if len(parts) <= max_parts:
        return "/".join(parts)
    return "…/" + "/".join(parts[-max_parts:])


def compact_url_label(url: str) -> str:
    """Return a shortened human-readable label for a URL."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}" if path else parsed.netloc


def file_link(path: str | Path, *, label: str | None = None) -> str:
    """Return an HTML link for a local file or folder."""
    resolved = Path(path).expanduser().resolve()
    text = label or compact_path_label(resolved)
    return f'<a href="{resolved.as_uri()}">{html.escape(text)}</a>'


def external_link(url: str, *, label: str | None = None) -> str:
    """Return an HTML link for an external URL."""
    text = label or compact_url_label(url)
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(text)}</a>'


def copy_link(value: str | Path, *, label: str = "Copy") -> str:
    """Return an HTML link that encodes clipboard text in a custom copy scheme."""
    text = str(value)
    encoded = quote(text, safe="")
    return f'<a href="copy:{encoded}">{html.escape(label)}</a>'


def decode_copy_target(href: str) -> str | None:
    """Return the decoded clipboard target for a custom copy link, if present."""
    if not href.startswith("copy:"):
        return None
    return unquote(href.partition(":")[2])
