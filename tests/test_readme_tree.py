"""Ensure the README file tree lists every source and test Python module."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _readme_tree() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Repository structure", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    match = re.search(r"```text\s*\n(.*?)\n```", section, flags=re.DOTALL)
    assert match is not None, "Repository structure section must contain a text tree"
    return match.group(1)


def _listed_python_files(tree: str) -> set[str]:
    listed: set[str] = set()
    directories: dict[int, list[str]] = {}
    branch_pattern = re.compile(r"^(?P<prefix>(?:│   |    )*)(?:├──|└──) (?P<label>.*)$")

    for line in tree.splitlines():
        match = branch_pattern.match(line)
        if match is None:
            continue

        depth = len(match.group("prefix")) // 4
        raw_label = match.group("label").strip()
        label = raw_label.split("  #", maxsplit=1)[0].strip()
        parent = directories.get(depth, [])

        if label.endswith("\\"):
            directory = label.rstrip("\\")
            directories[depth + 1] = [*parent, directory]
            if "package marker" in raw_label:
                listed.add("/".join([*parent, directory, "__init__.py"]))
            continue

        path = "/".join([*parent, label])
        if path.endswith(".py"):
            listed.add(path)

    return listed


def test_readme_tree_lists_all_source_and_test_python_files() -> None:
    required = {
        path.relative_to(ROOT).as_posix()
        for folder in (ROOT / "src" / "entdec", ROOT / "tests")
        for path in folder.rglob("*.py")
        if not any(
            part in {".git", ".venv", "__pycache__"} for part in path.relative_to(ROOT).parts
        )
    }

    listed = _listed_python_files(_readme_tree())
    existing = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if not any(
            part in {".git", ".venv", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache"}
            for part in path.relative_to(ROOT).parts
        )
    }

    assert required <= listed, f"Missing from README tree: {sorted(required - listed)}"
    assert listed <= existing, (
        f"README tree lists nonexistent Python files: {sorted(listed - existing)}"
    )
