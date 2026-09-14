# -*- coding: utf-8 -*-
"""Daily arxiv digest for the research vault.

Fetches recent eess.AS / cs.SD submissions plus ti/abs phrase-search hits
from other categories (cs.CV, cs.MM, cs.CR, ...), scores relevance against
ADD research topics, writes a digest note, and auto-creates Source notes
for high-relevance papers.

Run daily via Windows Task Scheduler. No external dependencies.
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

VAULT = Path(os.environ.get("RESEARCH_VAULT", str(Path(__file__).resolve().parents[1])))
QUEUE_DIR = VAULT / "Reading Queue"
SOURCES_DIR = VAULT / "Sources"
STATE_FILE = VAULT / "_scripts" / ".digest_state.json"

ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORIES = ["eess.AS", "cs.SD"]
# 카테고리 사각지대 보완: cs.CV/cs.MM/cs.CR 등에만 올라온 논문을 ti/abs 구문검색으로 포착
# (awesome-fake-audio-detection 리포 수집 키워드에서 채택, deception detection류는 제외)
PHRASE_QUERIES = [
    "audio deepfake",
    "deepfake speech detection",
    "fake audio detection",
    "speech antispoofing",
    "audio spoofing detection",
    "singfake",
]
LOOKBACK_DAYS = 4  # 주말+레이트리밋 갭 흡수 (seen_ids로 중복 방지되므로 안전)
MAX_RESULTS = 200
PHRASE_MAX_RESULTS = 50  # 구문검색은 히트 자체가 적어 소량이면 충분
AUTO_NOTE_THRESHOLD = 6  # score >= this -> auto-create Source note

# OpenAlex 무키 폴백(doc62): arxiv 429/timeout로 1차 수집이 빈손일 때만 발동하는 다중소스 백업.
# 무거운 deps 회피 위해 paper-search-mcp 패키지를 채택하지 않고 stdlib urllib로 최소 이식.
OPENALEX_API = "https://api.openalex.org/works"
POLITE_EMAIL = os.environ.get("OPENALEX_MAILTO", "")  # OpenAlex polite pool (optional)
_ARXIV_RE = re.compile(r"arxiv[.:/]*(?:abs/)?(\d{4}\.\d{4,5})", re.IGNORECASE)

# (pattern, score, suggested tags)
KEYWORD_RULES = [
    (r"deepfake|deep fake", 4, []),
    (r"anti-?spoof|spoof", 4, []),
    (r"asvspoof|add challenge|audio deepfake", 5, []),
    (r"fake (audio|speech|voice)|synthetic speech detection", 5, []),
    (r"audio forensic", 4, []),
    (r"voice conversion|text-to-speech|tts\b", 1, []),
    (r"speaker verification", 2, []),
    (r"countermeasure", 2, []),
    # topic tag mapping
    (r"nois(e|y)|reverb|real-?world|robust", 1, ["research/add/gen/noise-robust"]),
    (r"codec|neural audio codec", 1, ["research/add/gen/codec"]),
    (r"generaliz|out-of-distribution|ood|cross-dataset|unseen", 1, ["research/add/gen/ood"]),
    (r"large language model|llm|audio-?language", 1, ["research/add/arch/llm/audio"]),
    (r"\blora\b|low-rank|adapter|parameter-?efficient|peft", 2, ["research/add/arch/adapter/lora"]),
    (r"mixture[- ]of[- ]experts|moe\b", 2, ["research/add/arch/adapter/moe-lora"]),
    (r"self-?supervised|wav2vec|wavlm|hubert|ssl\b", 1, ["research/add/arch/ssl"]),
    (r"explain|interpret|attribution|saliency|xai", 1, ["research/add/explain/xai"]),
    (r"chain-?of-?thought|reasoning", 1, ["research/add/explain/cot"]),
    (r"localiz|partial(ly)? (fake|spoof)|frame-?level|boundary", 2, ["research/add/temporal/localization"]),
    (r"benchmark|dataset|corpus", 1, ["research/add/data/benchmark"]),
    (r"vocoder|source attribution|attack attribution", 1, ["research/add/explain/attribution"]),
    (r"singfake|singing voice (deepfake|spoof)|music deepfake|fake (music|song|singing)", 4, []),
]

# Hard requirement: paper must mention detection/spoof context, not just TTS.
GATE_RE = re.compile(
    r"deepfake|spoof|fake (audio|speech|voice|music|song|singing)|forensic|countermeasure"
    r"|synthetic speech detection|audio deepfake|asvspoof|liveness|singfake",
    re.IGNORECASE,
)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"seen_ids": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["seen_ids"] = state["seen_ids"][-3000:]
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _fetch_query(search_query: str, max_results: int) -> list:
    url = (
        f"{ARXIV_API}?search_query={search_query}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "vault-digest/1.0"})
    xml_data = None
    for attempt in range(4):  # 429/일시 오류 대비 백오프 재시도
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                xml_data = response.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(30 * (attempt + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            # 연결/읽기 타임아웃 등 전송 오류도 재시도(HTTPError 아니면 여기).
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    if xml_data is None:
        raise RuntimeError("arxiv fetch failed after retries")

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    papers = []
    for entry in root.findall("a:entry", ns):
        published = entry.findtext("a:published", "", ns)
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        if pub_dt < cutoff:
            continue
        raw_id = entry.findtext("a:id", "", ns)
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        papers.append({
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", entry.findtext("a:title", "", ns)).strip(),
            "abstract": re.sub(r"\s+", " ", entry.findtext("a:summary", "", ns)).strip(),
            "authors": [
                a.findtext("a:name", "", ns)
                for a in entry.findall("a:author", ns)
            ],
            "published": pub_dt.date().isoformat(),
            "link": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def _openalex_arxiv_id(item: dict) -> str:
    """OpenAlex work -> arxiv id(있으면). doi(10.48550/arXiv.XXXX)·모든 location url 스캔.
    arxiv로 해소 안 되면 "" — 폴백은 arxiv-키 레코드만 취해 하류 불변식(year_of·dedup·노트) 보존."""
    m = _ARXIV_RE.search(item.get("doi") or "")
    if m:
        return m.group(1)
    locs = list(item.get("locations") or [])
    if item.get("primary_location"):
        locs.insert(0, item["primary_location"])
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_RE.search(loc.get(key) or "")
            if m:
                return m.group(1)
    return ""


def _reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex 초록=inverted index -> 원문 복원(paper-search-mcp openalex.py 이식)."""
    if not inverted_index:
        return ""
    try:
        pos_word = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
        pos_word.sort(key=lambda x: x[0])
        return " ".join(w for _, w in pos_word)
    except Exception:
        return ""


def _fetch_openalex(max_results: int) -> list:
    """무키 OpenAlex 폴백: 최근(LOOKBACK_DAYS) ADD 구문 매칭 -> arxiv 해소분만 arxiv 스키마로 반환.
    arxiv Atom과 동일 dict shape({id,title,abstract,authors,published,link}). 실패=fail-soft([])."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    out, seen = [], set()
    ua = f"vault-digest/1.0 (mailto:{POLITE_EMAIL})" if POLITE_EMAIL else "vault-digest/1.0"
    for phrase in PHRASE_QUERIES:
        params = {
            "filter": f"from_publication_date:{cutoff},title_and_abstract.search:{phrase}",
            "per_page": min(max_results, 200),
        }
        if POLITE_EMAIL:
            params["mailto"] = POLITE_EMAIL
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{OPENALEX_API}?{query}",
            headers={"User-Agent": ua})
        data = None
        for attempt in range(4):  # OpenAlex도 연타 시 429 — 백오프 재시도
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                else:
                    break          # 한 구문 실패해도 다음 구문 진행(fail-soft)
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))
                else:
                    break
        if not data:
            print(f"warn: openalex fallback failed ({phrase})")
            time.sleep(3)
            continue
        for item in data.get("results", []):
            aid = _openalex_arxiv_id(item)
            title = item.get("title") or item.get("display_name")
            if not aid or aid in seen or not title:
                continue
            seen.add(aid)
            out.append({
                "id": aid,
                "title": re.sub(r"\s+", " ", title).strip(),
                "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),
                "authors": [a.get("author", {}).get("display_name", "")
                            for a in item.get("authorships", [])
                            if a.get("author", {}).get("display_name")],
                "published": item.get("publication_date", ""),
                "link": f"https://arxiv.org/abs/{aid}",
            })
        time.sleep(3)              # arxiv 권장 간격 준수(공통 페이싱)
    return out


def fetch_recent() -> list:
    """카테고리 스윕(정밀) + ti/abs 구문검색(카테고리 사각지대 커버) 2패스, id로 dedup.
    둘 다 빈손이면(arxiv 전면 장애) OpenAlex 무키 폴백(doc62)."""
    cat_query = "+OR+".join(f"cat:{c}" for c in CATEGORIES)
    # 1차 카테고리 스윕도 fail-soft: arxiv 429/일시장애로 죽어도 main()이 sync/push까지
    # 도달하게 함(digest 밀림 방지). HTTPError는 URLError 하위 → 재시도 소진분까지 포착.
    try:
        papers = _fetch_query(cat_query, MAX_RESULTS)
    except (OSError, RuntimeError) as e:  # URLError/HTTPError/TimeoutError 모두 OSError 하위
        print(f"warn: category sweep failed, degrading to phrase-only: {e}")
        papers = []
    seen_ids = {p["id"] for p in papers}

    for phrase in PHRASE_QUERIES:
        time.sleep(3)  # arxiv API 권장 간격
        quoted = urllib.parse.quote(f'"{phrase}"')
        phrase_query = f"ti:{quoted}+OR+abs:{quoted}"
        try:
            extra = _fetch_query(phrase_query, PHRASE_MAX_RESULTS)
        except (OSError, RuntimeError) as e:
            print(f"warn: phrase query failed ({phrase}): {e}")
            continue
        for p in extra:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                papers.append(p)

    if not papers:  # arxiv 전면 429/timeout → OpenAlex 무키 폴백(doc62)
        print("info: arxiv yielded 0 — trying OpenAlex fallback")
        try:
            papers = _fetch_openalex(PHRASE_MAX_RESULTS)
            print(f"info: openalex fallback recovered {len(papers)} arxiv-resolved papers")
        except Exception as e:
            print(f"warn: openalex fallback error: {type(e).__name__}: {e}")
    return papers


def score_paper(paper: dict):
    text = f"{paper['title']} {paper['abstract']}".lower()
    if not GATE_RE.search(text):
        return 0, []
    score = 0
    tags = []
    for pattern, points, rule_tags in KEYWORD_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            score += points
            for t in rule_tags:
                if t not in tags:
                    tags.append(t)
    return score, tags


def year_of(paper: dict) -> int:
    return int("20" + paper["id"][:2])


def write_source_note(paper: dict, tags: list, score: int) -> str:
    name = f"arxiv-{paper['id']}"
    path = SOURCES_DIR / f"{name}.md"
    if path.exists():
        return name
    all_tags = ["research/meta/type/source", f"research/meta/year/{year_of(paper)}"] + tags
    tag_lines = "\n".join(f"  - {t}" for t in all_tags)
    authors = ", ".join(paper["authors"])
    content = f"""---
date: {datetime.now().date().isoformat()}
tags:
{tag_lines}
arxiv: "{paper['id']}"
digest-score: {score}
status: unread
---

# {paper['title']}

> **arXiv:** [{paper['id']}]({paper['link']})
> **저자:** {authors}
> **제출일:** {paper['published']}
> **자동 수집:** Daily Digest (관련도 {score}점)

## Abstract

{paper['abstract']}

## 분석 메모

- [ ] 초록 검토
- [ ] 본문 분석 필요 여부 판단
- [ ] 관련 노트 연결
"""
    path.write_text(content, encoding="utf-8")
    return name


def write_digest(papers_scored: list, auto_notes: list) -> Path:
    today = datetime.now().date().isoformat()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUEUE_DIR / f"DIGEST-{today}.md"

    lines = [
        "---",
        f"date: {today}",
        "tags:",
        "  - research/meta/type/digest",
        f"papers: {len(papers_scored)}",
        "---",
        "",
        f"# 📥 Daily Digest — {today}",
        "",
        f"> eess.AS + cs.SD + 구문검색 최근 {LOOKBACK_DAYS}일 · 관련 논문 {len(papers_scored)}편"
        f" · 자동 노트 {len(auto_notes)}편",
        "",
    ]
    if not papers_scored:
        lines.append("관련 신규 논문 없음.")
    for paper, score, tags in papers_scored:
        marker = "⭐" if score >= AUTO_NOTE_THRESHOLD else "·"
        lines.append(f"## {marker} [{score}점] {paper['title']}")
        lines.append("")
        lines.append(f"- **arXiv:** [{paper['id']}]({paper['link']}) ({paper['published']})")
        lines.append(f"- **저자:** {', '.join(paper['authors'][:6])}")
        if tags:
            lines.append(f"- **추천 태그:** {' '.join('#' + t for t in tags)}")
        if score >= AUTO_NOTE_THRESHOLD:
            lines.append(f"- **노트:** [[arxiv-{paper['id']}]] (자동 생성)")
        abstract = paper["abstract"]
        snippet = abstract[:400] + ("..." if len(abstract) > 400 else "")
        lines.append(f"- {snippet}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    state = load_state()
    seen = set(state["seen_ids"])
    papers = fetch_recent()

    papers_scored = []
    for paper in papers:
        if paper["id"] in seen:
            continue
        score, tags = score_paper(paper)
        state["seen_ids"].append(paper["id"])
        if score >= 3:
            papers_scored.append((paper, score, tags))

    papers_scored.sort(key=lambda x: -x[1])
    auto_notes = [
        write_source_note(paper, tags, score)
        for paper, score, tags in papers_scored
        if score >= AUTO_NOTE_THRESHOLD
    ]
    digest_path = write_digest(papers_scored, auto_notes)
    save_state(state)
    print(f"digest: {digest_path} | papers: {len(papers_scored)} | auto-notes: {len(auto_notes)}")

    import subprocess
    import sys

    # 신규 자동 노트 논문 전문 다운로드 + 로컬 요약(doc62 "전문 다운로드 후 요약").
    # sync 앞에서 돌려 같은 날 vault 백업에 전문·요약 포함. arxiv 429나 Ollama 미가동이어도
    # digest/sync는 진행 — enrich는 부가 강화 단계(비치명). 오늘분 id는 fetch, --scan은 밀린
    # 요약(전문 있는 것)까지 소화. 스크립트는 miniconda(researchos_api deps) 필요 → 없으면 스킵.
    try:
        ids = [n[len("arxiv-"):] for n in auto_notes]
        enrich = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "summarize" / "enrich_new_papers.py"),
             *ids, "--scan"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800,
        )
        if enrich.stdout.strip():
            print(enrich.stdout.strip())
        if enrich.returncode != 0:
            print(f"enrich failed (non-fatal): {enrich.stderr.strip()[:500]}")
    except Exception as e:
        print(f"enrich skipped (non-fatal): {type(e).__name__}: {e}")

    # Push the vault and regenerate the public papers repo. The digest
    # result above must survive any sync failure, so this never raises.
    try:
        sync = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "export" / "sync_repos.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if sync.stdout.strip():
            print(sync.stdout.strip())
        if sync.returncode != 0:
            print(f"sync failed: {sync.stderr.strip()}")
    except Exception as e:
        print(f"sync skipped: {e}")


if __name__ == "__main__":
    main()
