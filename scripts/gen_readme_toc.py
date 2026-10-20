#!/usr/bin/env python3
"""Generate a dynamic documentation index for a repository README.

The index is built from the repository's **tracked** Markdown files (discovered
with ``git ls-files``, so it follows adds / removes / renames automatically) and
injected between HTML comment markers in the target README. Re-running the
script is idempotent: only the block between the markers is rewritten, so the
rest of the README is never touched.

Configuration is read from ``scripts/docs_manifest.json`` (every key optional):

    {
      "readme": "README.md",                     # README to update
      "section_title": "Documentation Index",    # heading of the generated block
      "marker_start": "<!-- docs-toc:start -->", # start marker (auto-added if missing)
      "marker_end": "<!-- docs-toc:end -->",     # end marker
      "include": ["*.md"],                       # git pathspecs to discover
      "exclude": [".git", "node_modules", ...],  # dir names or fnmatch globs
      "description_chars": 150,                  # truncate the extracted summary
      "root_label": "Repository root"            # label for files at the repo root
    }

Usage (from the repository root):

    python scripts/gen_readme_toc.py            # update README in place
    python scripts/gen_readme_toc.py --dry-run  # print the generated block only
    python scripts/gen_readme_toc.py --check     # exit 1 if README would change
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

# --- Defaults (overridable through the manifest) ---------------------------
DEFAULTS = {
    "readme": "README.md",
    "manifest": "scripts/docs_manifest.json",
    "section_title": "Documentation Index",
    "marker_start": "<!-- docs-toc:start -->",
    "marker_end": "<!-- docs-toc:end -->",
    "include": ["*.md"],
    "exclude": [
        ".git",
        ".github",
        "node_modules",
        ".venv",
        "venv",
        "site-packages",
        ".pytest_cache",
        "__pycache__",
    ],
    "description_chars": 150,
    "root_label": "Repository root",
}

# Frontmatter block (Jekyll/other) and ATX heading matchers.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*#*\s*$")
# Lines that never make a good one-line summary (badges, tables, HTML, lists...).
_SKIP_PREFIXES = ("#", "|", ">", "<", "[!", "![", "```", "-", "*", "+", "---", "===")
# Mermaid/diagram first words that are never a useful summary line.
_DIAGRAM_WORDS = (
    "flowchart", "graph", "sequencediagram", "statediagram",
    "classdiagram", "erdiagram", "journey", "gantt", "pie", "mindmap",
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Strip bold/italic/code markers. Underscore is intentionally NOT stripped so
# identifiers like ``ensure_ascii`` survive intact.
_MD_EMPHASIS_RE = re.compile(r"[*`]+")


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* until a ``.git`` entry (dir or file) is found."""
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def load_manifest(repo_root: Path) -> dict:
    """Load ``scripts/docs_manifest.json`` merged over :data:`DEFAULTS`."""
    cfg = dict(DEFAULTS)
    manifest_path = repo_root / cfg["manifest"]
    if manifest_path.exists():
        try:
            cfg.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:  # pragma: no cover - config error
            raise SystemExit(f"Invalid manifest {manifest_path}: {exc}") from exc
    return cfg


def is_excluded(rel: str, patterns: list[str]) -> bool:
    """True when *rel* matches a bare directory name or an fnmatch glob."""
    parts = rel.split("/")
    for pattern in patterns:
        # Bare name (e.g. "node_modules") prunes that directory at any depth.
        if "/" not in pattern and "*" not in pattern:
            if pattern in parts:
                return True
        if fnmatch.fnmatch(rel, pattern):
            return True
        # "docs/**" style should also match "docs/x.md"
        if pattern.endswith("/**") and rel.startswith(pattern[:-3].rstrip("/") + "/"):
            return True
    return False


def list_tracked_markdown(repo_root: Path, include: list[str], exclude: list[str]) -> list[str]:
    """Return tracked Markdown paths (POSIX, repo-relative), filtered by *exclude*."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", *include],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"`git ls-files` failed (run inside the repo): {exc}") from exc

    paths = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
    return sorted(p for p in paths if not is_excluded(p, exclude))


def _clean_summary(text: str) -> str:
    """Strip Markdown emphasis/links from a candidate summary line."""
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    return " ".join(text.split()).strip()


def parse_markdown(path: Path) -> tuple[str, str]:
    """Extract a title and a one-line summary from a Markdown file.

    Prefers YAML frontmatter ``title`` / ``description``; otherwise uses the
    first ATX heading and the first prose paragraph.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    title = summary = ""

    fm = _FRONTMATTER_RE.match(text)
    body = text
    if fm:
        body = text[fm.end():]
        for line in fm.group(1).splitlines():
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            if key.strip().lower() == "title" and value:
                title = value
            elif key.strip().lower() == "description" and value:
                summary = value

    if not title:
        for line in body.splitlines():
            heading = _HEADING_RE.match(line)
            if heading:
                title = _clean_summary(heading.group(1))
                break
    if not title:
        title = path.stem.replace("-", " ").replace("_", " ").strip()

    if not summary:
        in_fence = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                continue
            if in_fence or not stripped:
                continue
            if stripped.startswith(_SKIP_PREFIXES):
                continue
            if stripped.lower().startswith(_DIAGRAM_WORDS):
                continue
            summary = _clean_summary(stripped)
            if summary:
                break

    return title, summary


def build_block(repo_root: Path, cfg: dict, files: list[str]) -> str:
    """Render the Markdown documentation-index block (markers included)."""
    readme_rel = cfg["readme"].replace("\\", "/")
    limit = int(cfg["description_chars"])
    root_label = cfg["root_label"]

    # Group by top-level directory; "" means a file at the repository root.
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for rel in files:
        if rel == readme_rel:  # never list the README inside itself
            continue
        title, summary = parse_markdown(repo_root / rel)
        if len(summary) > limit:
            summary = summary[: limit - 1].rstrip() + "…"
        top = rel.split("/", 1)[0] if "/" in rel else ""
        groups.setdefault(top, []).append((rel, title, summary))

    total = sum(len(items) for items in groups.values())
    noun = "file" if total == 1 else "files"

    lines: list[str] = [cfg["marker_start"], f"## {cfg['section_title']}", ""]
    lines.append(
        f"> Auto-generated from this repository's tracked Markdown notes "
        f"(**{total}** {noun}). Refresh with `python scripts/gen_readme_toc.py` — "
        f"the `docs-toc` workflow does this automatically on push."
    )
    lines.append("")

    ordered = [""] + sorted(g for g in groups if g)
    for group in ordered:
        items = groups.get(group)
        if not items:
            continue
        if group == "":
            lines.append(f"### {root_label}")
        else:
            lines.append(f"### `{group}/`")
        lines.append("")
        for rel, title, summary in sorted(items):
            entry = f"- [{title}]({rel})"
            if summary:
                entry += f" — {summary}"
            lines.append(entry)
        lines.append("")

    lines.append(cfg["marker_end"])
    return "\n".join(lines)


def inject(readme_text: str, block: str, cfg: dict) -> str:
    """Replace the marker block in *readme_text* (or append it when absent)."""
    start, end = cfg["marker_start"], cfg["marker_end"]
    if start in readme_text and end in readme_text:
        pre = readme_text[: readme_text.index(start)]
        post = readme_text[readme_text.index(end) + len(end):]
        return pre + block + post
    return readme_text.rstrip() + "\n\n" + block + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the block, do not write")
    parser.add_argument("--check", action="store_true", help="exit 1 if the README is stale")
    parser.add_argument("--root", default=None, help="repository root (default: autodetect)")
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve() if args.root else _find_repo_root(Path(__file__).resolve())
    cfg = load_manifest(repo_root)
    files = list_tracked_markdown(repo_root, cfg["include"], cfg["exclude"])
    block = build_block(repo_root, cfg, files)

    if args.dry_run:
        print(block)
        return 0

    readme_path = repo_root / cfg["readme"]
    original = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated = inject(original, block, cfg)

    if args.check:
        if original != updated:
            print(f"{cfg['readme']} is out of date — run scripts/gen_readme_toc.py", file=sys.stderr)
            return 1
        return 0

    if original != updated:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"Updated {cfg['readme']} ({len(files)} tracked Markdown files).")
    else:
        print(f"{cfg['readme']} already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
