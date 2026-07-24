from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_MARKER = "to" + "y"
TEXT_SUFFIXES = {
    ".cfg",
    ".cmd",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def test_repository_has_no_deprecated_placeholder_assets() -> None:
    roots = [
        ROOT / "configs",
        ROOT / "examples",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
    ]
    matches: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            if DEPRECATED_MARKER in str(relative).casefold():
                matches.append(str(relative))
                continue
            if path.suffix.casefold() not in TEXT_SUFFIXES:
                continue
            if DEPRECATED_MARKER in path.read_text(encoding="utf-8").casefold():
                matches.append(str(relative))
    assert matches == []
