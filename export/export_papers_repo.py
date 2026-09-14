# -*- coding: utf-8 -*-
"""Export vault paper metadata to the public GitHub repo.

Reads Corpus/, Sources/, Literature/ frontmatter and generates
README.md + topics/*.md in the sibling repo directory. Metadata only:
title, authors, year, venue, citations, links, topic mapping.
Personal analysis and note bodies are never exported.

Read-only with respect to the vault. No external dependencies.
"""
import os
import re
import shutil
import ssl
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path

# Resolved relative to this file so the script works on any machine
# that clones the vault; the papers repo is expected as a sibling dir.
VAULT = Path(os.environ.get("RESEARCH_VAULT", str(Path(__file__).resolve().parents[1])))
REPO = VAULT.parent / "audio-deepfake-detection-papers"
# Literature is intentionally NOT exported: its H1 titles are vault-internal
# shorthand with strategy annotations, not real paper titles.
SOURCE_DIRS = ["Corpus", "Sources"]
EXCLUDE_TAG_PREFIX = "research/anti-cloning"
# Vault-internal note-title conventions — never real paper titles.
INTERNAL_TITLE_RE = re.compile(r"^(LIT-|ZSTTS-|AntiCloning-|Source:|arxiv-\d)")

REPO_URL = "https://github.com/woongja/audio-deepfake-detection-papers"
LATEST_COUNT = 30

# 논문별 요약 페이지(doc62 §6.3): 로컬 Gemma 요약(_fulltext/_sum-*.md)이 있는 논문만
# summaries/arxiv-<id>.md 생성 → 테이블 제목이 이 페이지로 링크(요약 없으면 arxiv 직행).
# ⚠ 요약은 검증 안 된 LLM 생성물(신뢰 3티어 🔴) → 페이지·README에 배너 필수.
FULLTEXT_DIR = VAULT / "Sources" / "_fulltext"
SUMMARIES_DIR = REPO / "summaries"
_FT_ARXIV_RE = re.compile(r'^arxiv:\s*"?(\d{4}\.\d{4,5}(?:v\d+)?)"?', re.MULTILINE)
_SUM_BANNER = ("> ⚠ AI-generated summary (local LLM, **Korean**, unverified) — "
               "auto-generated from the paper, not fact-checked. Verify against the source.")

# (tag prefix, topic file stem, page title) — first match wins per tag,
# one paper may appear on several pages.
TOPIC_MAP = [
    ("research/add/arch/ssl", "ssl-models", "SSL Front-Ends (wav2vec2 / WavLM / HuBERT)"),
    ("research/add/arch/layer-selection", "ssl-models", "SSL Front-Ends (wav2vec2 / WavLM / HuBERT)"),
    ("research/add/arch/spectral", "spectral-frontends", "Spectral & Signal-Processing Front-Ends"),
    ("research/add/arch/encoder", "architectures", "Detection Architectures & Encoders"),
    ("research/add/arch/adapter", "adapters-lora-moe", "Adapters, LoRA & MoE"),
    ("research/add/arch/llm", "llm-based", "LLM-Based Detection"),
    ("research/add/explain", "xai-explainability", "Explainability & Attribution"),
    ("research/add/gen/noise-robust", "noise-robustness", "Noise & Real-World Robustness"),
    ("research/add/gen/ood", "generalization-ood", "Generalization & OOD"),
    ("research/add/gen/cross-dataset", "generalization-ood", "Generalization & OOD"),
    ("research/add/gen/codec", "codec-deepfake", "Neural Codec Deepfakes"),
    ("research/add/temporal", "temporal-localization", "Temporal Localization & Partial Spoof"),
    ("research/add/data", "datasets-benchmarks", "Datasets & Benchmarks"),
    ("research/add/train", "training-strategies", "Training Strategies"),
    ("research/add/adapt/transfer", "training-strategies", "Training Strategies"),
    ("research/add/cross-domain", "cross-domain-multimodal", "Cross-Domain & Multimodal"),
]
MISC = ("misc", "Other Topics")


def parse_note(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    fm = m.group(1)

    tags = re.findall(r"^\s+-\s+(research/\S+)", fm, re.MULTILINE)
    if any(t.startswith(EXCLUDE_TAG_PREFIX) for t in tags):
        return None
    # Notes without any research/add tag were judged off-topic
    # (speaker ID, attack-side, pure TTS) during reclassification.
    if not any(t.startswith("research/add/") for t in tags):
        return None

    def fm_value(field: str) -> str:
        vm = re.search(rf'^{field}:\s*"?([^"\n]+)"?\s*$', fm, re.MULTILINE)
        return vm.group(1).strip() if vm else ""

    title_m = re.search(r"^# (.+)$", text, re.MULTILINE)
    if not title_m:
        return None
    title = re.sub(r"\s+", " ", title_m.group(1)).strip()
    if INTERNAL_TITLE_RE.match(title):
        return None

    authors_m = re.search(r"^> \*\*저자:?\*\*\s*(.+)$", text, re.MULTILINE)
    first_author = ""
    if authors_m:
        names = [a.strip() for a in authors_m.group(1).split(",") if a.strip()]
        if names:
            first_author = names[0] + (" et al." if len(names) > 1 else "")

    # 노트에 기록된 제출일(YYYY-MM-DD). arxiv API가 429로 죽어도 정확 날짜를 잃지 않도록
    # latest_table의 폴백으로 사용(fetch_arxiv_dates 실패 시 월이 아니라 이 값).
    sub_m = re.search(r"^> \*\*제출일:?\*\*\s*(\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    submitted = sub_m.group(1) if sub_m else ""

    year = 0
    year_m = re.search(r"research/(?:meta/)?year/(\d{4})", fm)
    arxiv = fm_value("arxiv")
    if year_m:
        year = int(year_m.group(1))
    elif re.match(r"^\d{4}\.", arxiv):
        year = 2000 + int(arxiv[:2])
    if not year:
        return None

    citations = fm_value("citations")
    venue = fm_value("venue")
    if venue in ("?", "None", "null"):
        venue = ""
    return {
        "title": title,
        "author": first_author,
        "year": year,
        "venue": venue or ("arXiv" if arxiv else ""),
        "citations": int(citations) if citations.isdigit() else None,
        "arxiv": arxiv,
        "s2id": fm_value("s2id"),
        "submitted": submitted,
        "tags": tags,
    }


def dedup_key(p: dict) -> str:
    if p["arxiv"]:
        return f"arxiv:{p['arxiv']}"
    if p["s2id"]:
        return f"s2:{p['s2id']}"
    return "title:" + re.sub(r"[^a-z0-9]", "", p["title"].lower())


def collect() -> list:
    papers = {}
    skipped = 0
    for dirname in SOURCE_DIRS:
        for path in sorted((VAULT / dirname).glob("*.md")):
            try:
                paper = parse_note(path)
            except (OSError, ValueError) as e:
                print(f"warn: skip {path.name}: {e}")
                skipped += 1
                continue
            if paper is None:
                continue
            key = dedup_key(paper)
            if key not in papers:  # Corpus first — richest metadata wins
                papers[key] = paper
    if skipped:
        print(f"warn: {skipped} notes skipped on parse errors")
    return list(papers.values())


def link_of(p: dict) -> str:
    if p["arxiv"]:
        return f"https://arxiv.org/abs/{p['arxiv']}"
    if p["s2id"]:
        return f"https://www.semanticscholar.org/paper/{p['s2id']}"
    return ""


def _summary_index() -> dict:
    """_fulltext/*.md frontmatter arxiv → 전문노트 stem. {arxiv_id: stem}(버전 포함/제외 둘 다).
    stdlib only (export 무의존 유지). enrich `fulltext:` 필드에 안 걸어 Corpus·배치·앱
    on-demand로 만들어진 캐시까지 전부 포착."""
    idx = {}
    if not FULLTEXT_DIR.is_dir():
        return idx
    for p in FULLTEXT_DIR.glob("*.md"):
        if p.name.startswith("_"):
            continue
        m = _FT_ARXIV_RE.search(p.read_text("utf-8", "replace")[:600])
        if m:
            aid = m.group(1)
            idx.setdefault(aid, p.stem)
            idx.setdefault(aid.split("v")[0], p.stem)
    return idx


def _summary_sections(text: str) -> str:
    """Gemma 요약의 모든 섹션(한 줄 요약·문제 정의·제안 방법·실험·결과·한계)을 그대로 발췌.
    캐시 말미 로컬LLM 경고줄만 제거(export 배너로 대체). 헤딩 못 맞추면 전체 폴백."""
    text = re.sub(r"(?m)^>?\s*⚠.*로컬 LLM.*$", "", text).strip()
    parts = re.split(r"(?m)^#{2,3}\s*(.+?)\s*$", text)  # [pre, h1, b1, h2, b2, ...]
    it = iter(parts[1:])
    _skip = re.compile(r"핵심 사실|참고\s*문헌|참조|references|bibliography", re.I)
    picked = [(h.strip(), b.strip()) for h, b in zip(it, it)
              if not _skip.search(h)]                     # 추출 스캐폴딩·참고문헌은 공개 페이지서 제외
    if not picked:
        return text                                       # 폴백: 전체(빈 페이지 방지)
    return "\n\n".join(f"## {h}\n\n{b}" for h, b in picked)


def _slug(title: str, maxlen: int = 80) -> str:
    """논문 제목 → 파일명 슬러그. ascii 소문자/숫자/하이픈만. 빈 결과는 'paper'."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)   # 구두점·비ascii 제거
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:maxlen].strip("-") or "paper"


def summary_page(p: dict, idx: dict) -> str:
    """요약 캐시 있으면 summaries/<제목슬러그>.md 생성 후 파일명 반환. 없으면 ""."""
    aid = (p["arxiv"] or "").split("v")[0]
    stem = idx.get(aid) if aid else None
    if not stem:
        return ""
    cache = FULLTEXT_DIR / f"_sum-{stem}.md"
    if not cache.is_file():
        return ""
    body = _summary_sections(cache.read_text("utf-8", "replace"))
    if not body.strip():
        return ""
    lines = [
        f"# {p['title']}",
        "",
        _SUM_BANNER,
        "",
        f"**arXiv:** https://arxiv.org/abs/{aid}"
        + (f" · {p['author']}" if p["author"] else "")
        + (f" · {p['year']}" if p["year"] else ""),
        "",
        body,
        "",
        "---",
        f"_Part of [audio-deepfake-detection-papers]({REPO_URL}) · "
        "summary auto-generated by a local LLM, unverified._",
        "",
    ]
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{_slug(p['title'])}.md"
    if (SUMMARIES_DIR / name).exists():        # 같은 제목 슬러그 충돌 → id 접미로 유일화
        name = f"{_slug(p['title'])}-{aid}.md"
    (SUMMARIES_DIR / name).write_text("\n".join(lines), encoding="utf-8")
    return name


def md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def ym_of(arxiv: str) -> str:
    """arxiv id YYMM → YYYY-MM (업로드 연-월). 신형식(2007-04~)은 전부 20YY.
    1507.x→2015-07, 2608.x→2026-08. 구형식/미매칭은 "" (연도 폴백)."""
    m = re.match(r"^(\d{2})(\d{2})\.", arxiv or "")
    if not m:
        return ""
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return ""
    return f"20{yy:02d}-{mm:02d}"


_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_dates(ids: list) -> dict:
    """arxiv id_list API → {id: 'YYYY-MM-DD'} 제출일(정확 날짜). 1회 요청.
    네트워크/Norton 실패 시 {} 반환 → 호출측이 월(YYYY-MM)로 폴백(오프라인 내구)."""
    ids = [i for i in ids if i]
    if not ids:
        return {}
    url = ("http://export.arxiv.org/api/query?max_results=" + str(len(ids)) +
           "&id_list=" + ",".join(ids))
    req = urllib.request.Request(url, headers={"User-Agent": "add-papers-export/1.0"})
    try:
        try:
            raw = urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.URLError:
            raw = urllib.request.urlopen(req, timeout=30,
                                         context=ssl._create_unverified_context()).read()
        out = {}
        for e in ET.fromstring(raw).findall("a:entry", _ARXIV_NS):
            aid = (e.findtext("a:id", "", _ARXIV_NS) or "").rstrip("/").rsplit("/", 1)[-1].split("v")[0]
            pub = (e.findtext("a:published", "", _ARXIV_NS) or "")[:10]
            if aid and re.match(r"\d{4}-\d{2}-\d{2}", pub):
                out[aid] = pub
        return out
    except Exception:
        return {}


def latest_table(papers: list, dates: dict) -> list:
    """Latest 섹션 전용 — Date = arxiv 제출일(YYYY-MM-DD). 못 받으면 월(YYYY-MM) 폴백."""
    lines = ["| Date | Title | First Author | Summary | Citations |",
             "|---|---|---|---|---|"]
    for p in papers:
        url = link_of(p)
        title = f"[{md_escape(p['title'])}]({url})" if url else md_escape(p["title"])
        summary = f"[📝](summaries/{p['_page']})" if p.get("_page") else ""
        cit = str(p["citations"]) if p["citations"] is not None else ""
        aid = (p["arxiv"] or "").split("v")[0]
        # arxiv API(dates) → 노트 기록 제출일 → 월 폴백 → 연도. 429로 dates가 비어도 정확 날짜 유지.
        when = dates.get(aid) or p.get("submitted") or ym_of(p["arxiv"]) or str(p["year"])
        lines.append("| " + " | ".join([when, title, md_escape(p["author"]),
                                         summary, cit]) + " |")
    return lines


def table(papers: list, with_year: bool = True) -> list:
    head = "| Year | Title | First Author | Venue | Citations |"
    sep = "|---|---|---|---|---|"
    if not with_year:
        head, sep = head.replace("| Year ", "", 1), sep[4:]
    lines = [head, sep]
    for p in papers:
        url = link_of(p)
        title = f"[{md_escape(p['title'])}]({url})" if url else md_escape(p["title"])
        if p.get("_page"):
            title += f" · [📝 요약](../summaries/{p['_page']})"
        cit = str(p["citations"]) if p["citations"] is not None else ""
        cells = [title, md_escape(p["author"]), md_escape(p["venue"]), cit]
        if with_year:
            cells.insert(0, str(p["year"]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def topic_pages(papers: list) -> dict:
    pages = {}  # stem -> (title, [papers])
    for stem, page_title in {s: t for _, s, t in TOPIC_MAP}.items():
        pages[stem] = (page_title, [])
    pages[MISC[0]] = (MISC[1], [])
    for p in papers:
        stems = {stem for prefix, stem, _ in TOPIC_MAP
                 if any(t.startswith(prefix) for t in p["tags"])}
        if not stems:
            stems = {MISC[0]}
        for stem in stems:
            pages[stem][1].append(p)
    return pages


def write_repo(papers: list) -> None:
    (REPO / "topics").mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    papers.sort(key=lambda p: (-p["year"], -(p["citations"] or 0)))

    # 요약 페이지 전량 재생성(스테일 제거) — 캐시 있는 논문만 p["_page"] 세팅.
    shutil.rmtree(SUMMARIES_DIR, ignore_errors=True)
    sum_idx = _summary_index()
    n_sum = 0
    for p in papers:
        p["_page"] = summary_page(p, sum_idx)
        if p["_page"]:
            n_sum += 1
    print(f"summary pages: {n_sum} (요약 캐시 있는 논문만; 나머지는 arXiv 직행 링크)")

    pages = topic_pages(papers)
    for stem, (page_title, plist) in pages.items():
        lines = [
            f"# {page_title}",
            "",
            f"> {len(plist)} papers · part of [audio-deepfake-detection-papers]({REPO_URL}) · updated {today}",
            "",
            *table(plist),
            "",
        ]
        (REPO / "topics" / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")

    recent = sorted(
        [p for p in papers if p["arxiv"]],
        key=lambda p: p["arxiv"], reverse=True,
    )[:LATEST_COUNT]
    # 정확 제출일 조회(arxiv API 1회) → 실제 날짜순 재정렬(폴백: arxiv id 순)
    recent_dates = fetch_arxiv_dates([(p["arxiv"] or "").split("v")[0] for p in recent])
    recent.sort(key=lambda p: recent_dates.get((p["arxiv"] or "").split("v")[0], ym_of(p["arxiv"])),
                reverse=True)
    year_counts = Counter(p["year"] for p in papers)
    toc = [
        f"- [{title}](topics/{stem}.md) ({len(plist)})"
        for stem, (title, plist) in pages.items() if plist
    ]
    year_rows = [
        f"| {y} | {year_counts[y]} |" for y in sorted(year_counts, reverse=True)
    ]
    readme = [
        "# Audio Deepfake Detection Papers",
        "",
        f"![papers](https://img.shields.io/badge/papers-{len(papers)}-blue) "
        f"![updated](https://img.shields.io/badge/updated-{today.replace('-', '--')}-green)",
        "",
        "A curated list of audio deepfake detection (anti-spoofing) papers,",
        "organized by topic and year, with venues, citation counts, and arXiv links.",
        "Generated from a personally maintained research corpus.",
        "",
        "## Topics",
        "",
        *toc,
        "",
        f"## Latest {len(recent)} Papers",
        "",
        "_Date = arXiv submission date (falls back to YYYY-MM if unavailable)._",
        "",
        *latest_table(recent, recent_dates),
        "",
        "## Papers per Year",
        "",
        "| Year | Papers |",
        "|---|---|",
        *year_rows,
        "",
        "## About",
        "",
        "Citation counts come from Semantic Scholar at collection time and lag reality.",
        "Metadata is collected automatically and spot-checked, not exhaustively verified.",
        "Some papers link to a per-paper summary page (📝). Those summaries are "
        "**auto-generated by a local LLM in Korean and are not verified** — always "
        "check the original paper.",
        "Contributions and corrections are welcome via issues.",
        "",
        "License: [CC0 1.0](LICENSE)",
        "",
    ]
    (REPO / "README.md").write_text("\n".join(readme), encoding="utf-8")

    license_path = REPO / "LICENSE"
    if not license_path.exists():
        license_path.write_text(
            "CC0 1.0 Universal\n\nTo the extent possible under law, the author has waived all\n"
            "copyright and related or neighboring rights to this work.\n"
            "https://creativecommons.org/publicdomain/zero/1.0/\n",
            encoding="utf-8",
        )


def main():
    papers = collect()
    write_repo(papers)
    print(f"exported {len(papers)} papers -> {REPO}")


if __name__ == "__main__":
    main()
