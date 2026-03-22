"""Tests for rich-text link formatting helpers."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.widgets.link_formatting import copy_link, decode_copy_target


def test_copy_link_round_trips_filesystem_paths() -> None:
    path = Path("/tmp/examples/catalog/pk/oral/demo/README.md")

    href = copy_link(path, label="Copy path")

    assert 'href="copy:' in href
    assert "Copy path" in href
    encoded_target = href.split('href="', 1)[1].split('"', 1)[0]
    assert decode_copy_target(encoded_target) == str(path)


def test_copy_link_round_trips_urls() -> None:
    url = "https://github.com/NMautoverse/NMdata"

    href = copy_link(url, label="Copy URL")

    assert 'href="copy:' in href
    assert "Copy URL" in href
    encoded_target = href.split('href="', 1)[1].split('"', 1)[0]
    assert decode_copy_target(encoded_target) == url


def test_decode_copy_target_ignores_non_copy_links() -> None:
    assert decode_copy_target("https://example.com") is None
