# -*- coding: utf-8 -*-
"""Sync the vault and the public papers repo to GitHub.

Run after any paper/digest analysis, or automatically at the end of
daily_digest.py. Order: vault commit -> pull --rebase -> push, then
regenerate the public papers repo, run the leak guard, and push it.

The leak guard scans generated files for vault-internal markers
(LIT-/ZSTTS-/AntiCloning- shorthand titles, wikilinks) and aborts the
papers push on any hit — a real leak reached the public repo once
(2026-07-06), so this fails closed.

Paths are resolved relative to this file so the script works on any
machine that clones the vault. The papers repo step is skipped quietly
on machines that do not have that repo checked out.

Usage: python sync_repos.py [--vault-only | --papers-only]
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

VAULT = Path(__file__).resolve().parents[1]
PAPERS_REPO = VAULT.parent / "audio-deepfake-detection-papers"
EXPORT_SCRIPT = Path(__file__).with_name("export_papers_repo.py")

# Vault-internal markers that must never appear in the public repo.
LEAK_RE = re.compile(r"\bLIT-|\bZSTTS-|\bAntiCloning-|\[\[")


def run(args: list, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def git_sync(repo: Path, message: str) -> str:
    """Commit local changes, rebase onto origin/main, push. Returns 'ok' or an error."""
    status = run(["git", "status", "--porcelain"], repo)
    if status.returncode != 0:
        return f"not a git repo? {status.stderr.strip()}"
    if status.stdout.strip():
        run(["git", "add", "-A"], repo)
        commit = run(["git", "commit", "-m", message], repo)
        if commit.returncode != 0:
            return f"commit failed: {commit.stderr.strip()}"
    pull = run(["git", "pull", "--rebase", "origin", "main"], repo)
    if pull.returncode != 0:
        run(["git", "rebase", "--abort"], repo)
        return f"pull --rebase failed (resolve manually): {pull.stderr.strip()}"
    push = run(["git", "push", "origin", "main"], repo)
    if push.returncode != 0:
        return f"push failed: {push.stderr.strip()}"
    return "ok"


def leak_guard() -> list:
    """Scan generated public files for vault-internal markers."""
    hits = []
    files = [PAPERS_REPO / "README.md", *sorted((PAPERS_REPO / "topics").glob("*.md"))]
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in LEAK_RE.finditer(text):
            ctx = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            hits.append(f"{path.name}: ...{ctx}...")
    return hits


def sync_vault(today: str) -> bool:
    result = git_sync(VAULT, f"sync: vault update {today}")
    print(f"vault: {result}")
    return result == "ok"


def sync_papers(today: str) -> bool:
    if not (PAPERS_REPO / ".git").exists():
        print(f"papers: repo not found at {PAPERS_REPO}, skipped")
        return True
    export = run([sys.executable, str(EXPORT_SCRIPT)], VAULT)
    if export.stdout.strip():
        print(export.stdout.strip())
    if export.returncode != 0:
        print(f"papers: export failed: {export.stderr.strip()}")
        return False
    hits = leak_guard()
    if hits:
        print("papers: LEAK GUARD TRIPPED — push aborted. Review before publishing:")
        for h in hits[:10]:
            print("  " + h)
        return False
    result = git_sync(PAPERS_REPO, f"update: regenerate paper list ({today})")
    print(f"papers: {result}")
    return result == "ok"


def main() -> int:
    argv = sys.argv[1:]
    today = date.today().isoformat()
    ok = True
    if "--papers-only" not in argv:
        ok = sync_vault(today) and ok
    if "--vault-only" not in argv:
        ok = sync_papers(today) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
