# -*- coding: utf-8 -*-
"""ingest_papers — 논문 전문 md 파이프라인 (doc57, v1 standalone).

second brain RAG substrate: 논문 PDF → md(영어 원문, 번역 X) → vault `Sources/_fulltext/`.
그 폴더는 모든 KG/export 스캐너의 top-level glob("*.md") 밖(검증: doc57/58) → L-KG·R-KG·공개
export 미포착, index_vault.py(rglob)만 RAG 인덱싱. kg_type:none로 이중 격리.

⚠ frozen 사이드카에 markitdown(onnxruntime/magika) 번들 안 함 — 이 스크립트는 **miniconda
독립 실행**(doc57 §2.1 "사이드카 워처"에서 이탈, v2에서 사이드카가 subprocess로 트리거).

모드:
    python ingest_papers.py --smoke                 # 네트워크/claude 없이 로직 검증
    python ingest_papers.py --inbox                 # Inbox/papers/*.pdf 변환→저장→원본삭제
    python ingest_papers.py --fetch "AASIST anti-spoofing"   # OA 자동: 검색→다운→변환
    python ingest_papers.py --fetch-file queries.txt          # 줄마다 1 쿼리
못 받은(유료) 논문은 Sources/_fulltext/_unfetched.md에 리스트 → 사용자가 브라우저로 다운.
"""
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import quote

VAULT = Path(os.environ.get("RESEARCH_VAULT", str(Path(__file__).resolve().parents[1])))
OUT_DIR = VAULT / "Sources" / "_fulltext"       # 모든 스캐너 밖(하위폴더), RAG만 포착
INBOX = VAULT / "Inbox" / "papers"              # 전용 드롭 폴더(공유 Inbox 루트 안 건드림)
UNFETCHED = OUT_DIR / "_unfetched.md"           # 유료/실패 → 사용자 수동 다운로드 리스트
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
UA = f"ResearchOS-ingest/1.0 (mailto:{MAILTO})" if MAILTO else "ResearchOS-ingest/1.0"
MIN_BODY = 1000                                 # md 본문 최소 길이(미달=변환실패로 간주)
PACE = 2.0                                       # 요청 간 최소 간격(초) — arxiv/ar5iv 예의(429 방지)
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(v\d+)?")
_SLUG = re.compile(r"[^0-9A-Za-z가-힣]+")


def _log(msg):
    print(msg, flush=True)


def _slug(s, n=70):
    return (_SLUG.sub("-", s or "").strip("-") or "paper")[:n]


def _y(v):                                       # YAML 값 안전화(따옴표/개행 제거)
    return str(v or "").replace('"', "'").replace("\n", " ").replace("\r", " ").strip()[:300]


def _fetch_bytes(url, timeout=40, retries=3):
    """urllib GET → bytes. Norton MITM SSL 재서명 폴백 + 429 지수 백오프(rate-limit 방어)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    def _go(ctx=None):
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read()
    for attempt in range(retries):
        try:
            return _go()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:      # rate-limit → 대기 후 재시도
                wait = 15 * (attempt + 1)
                _log(f"    429 rate-limit, {wait}s 대기 후 재시도…")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                return _go(ssl._create_unverified_context())  # 로컬 AV MITM 전용 폴백
            raise
    raise urllib.error.URLError("retries exhausted")


def _is_pdf(b):                                  # 매직바이트: OA pdf_url이 랜딩페이지(HTML)인 경우 방어
    return bool(b) and b[:5] == b"%PDF-"


def _openalex(query, timeout=20):
    """query(또는 OpenAlex/arxiv id) → 메타+OA pdf url. 없으면 pdf_url=None."""
    q = query.strip()
    if re.fullmatch(r"10\.\d{4,9}/\S+", q):      # DOI 직접(works/doi: 포맷이 정답, path형은 429/오류)
        url = f"https://api.openalex.org/works/doi:{q}"
    elif re.fullmatch(r"W\d+", q):               # OpenAlex ID 직접
        url = f"https://api.openalex.org/works/{q}"
    elif re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", q):   # arXiv id
        url = f"https://api.openalex.org/works/https://arxiv.org/abs/{q}"
    else:
        url = "https://api.openalex.org/works?search=" + quote(q) + "&per_page=1"
    if MAILTO:                                    # OpenAlex polite pool(UA/mailto) — 선택
        url += ("&" if "?" in url else "?") + f"mailto={MAILTO}"
    try:
        raw = _fetch_bytes(url, timeout)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        return {"ok": False, "error": f"OpenAlex {type(e).__name__}", "query": q}
    w = data if "results" not in data else (data.get("results") or [None])[0]
    if not w:
        return {"ok": False, "error": "no result", "query": q}
    ids = w.get("ids") or {}
    arxiv = ""
    m = re.search(r"arxiv\.org/abs/([0-9.]+)", json.dumps(ids))
    if m:
        arxiv = m.group(1)
    best = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    pdf_url = best.get("pdf_url") or oa.get("oa_url")
    return {"ok": True, "query": q,
            "oaid": (w.get("id") or "").rsplit("/", 1)[-1],
            "title": w.get("display_name") or "(제목 없음)",
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "arxiv": arxiv, "year": w.get("publication_year"),
            "pdf_url": pdf_url,
            "landing": best.get("landing_page_url") or (w.get("primary_location") or {}).get("landing_page_url"),
            "publisher_url": (w.get("primary_location") or {}).get("landing_page_url")}


def _arxiv_search(query, timeout=20):
    """arxiv API로 title 검색 → {arxiv, title, pdf_url}. ADD는 arxiv 중심이라 OA 1순위.
    query가 arxiv id면 검색 없이 바로 구성."""
    import xml.etree.ElementTree as ET
    q = query.strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", q):     # arxiv id 직접
        aid = q
        url = f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1"
    else:
        url = ("http://export.arxiv.org/api/query?search_query="
               + quote(f'ti:"{q}"') + "&max_results=1")
    try:
        raw = _fetch_bytes(url, timeout)
    except Exception:
        return None
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        e = root.find("a:entry", ns)
        if e is None:
            return None
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1].strip()
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        if not aid:
            return None
        return {"arxiv": aid, "title": title,
                "pdf_url": f"https://arxiv.org/pdf/{aid}"}
    except Exception:
        return None


def _convert(pdf_path):
    """PDF → md 텍스트(markitdown). import는 함수 안(frozen 미번들, 독립 실행 전용)."""
    from markitdown import MarkItDown
    res = MarkItDown().convert(str(pdf_path))
    return res.text_content or ""


def _title_from_md(body):
    """ar5iv md 첫 '# ' 헤딩 = 논문 제목(arxiv API 안 쓰고 제목 확보)."""
    for line in (body or "").splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return ""


_DOI_STEM = re.compile(r"10\.\d{4,9}[_/]")      # DOI-슬러그 파일명 판정(예: 10.1109_...)


def _crossref_title(doi):
    """CrossRef /works/{doi} → 제목. 실패 시 ""(inbox 오프라인 동작 보존, P2-6 근본수정)."""
    try:
        raw = _fetch_bytes("https://api.crossref.org/works/" + quote(doi, safe="/.()"),
                           timeout=20, retries=2)
        t = (json.loads(raw.decode("utf-8", "replace"))
             .get("message", {}).get("title") or [""])[0]
        return " ".join(t.split())
    except Exception:
        return ""


def _try_ar5iv(arxiv_id, timeout=30):
    """arxiv → ar5iv HTML → md. PDF보다 공백·구조 보존 좋아 RAG 품질↑(arxiv 전용).
    실패/빈 결과 시 None → 호출측이 PDF로 폴백(사용자 PDF 결정은 baseline 유지)."""
    import tempfile
    from markitdown import MarkItDown
    aid = arxiv_id.split("v")[0] if re.match(r"\d{4}\.\d+v\d+", arxiv_id) else arxiv_id
    for base in (f"https://ar5iv.org/abs/{aid}",
                 f"https://ar5iv.labs.arxiv.org/html/{aid}"):
        try:
            raw = _fetch_bytes(base, timeout)
        except Exception:
            continue
        if b"<html" not in raw[:4000].lower() and b"<!doctype" not in raw[:200].lower():
            continue
        try:
            tp = Path(tempfile.gettempdir()) / f"ar5iv-{_slug(aid)}.html"
            tp.write_bytes(raw)
            txt = MarkItDown().convert(str(tp)).text_content or ""
            tp.unlink(missing_ok=True)
            # ar5iv 미렌더(최신 논문) → arxiv.org 초록페이지로 리다이렉트 = 네비 chrome junk.
            # 이걸 걸러야 RAG 오염 방지(→ None 반환 시 호출측 PDF 폴백).
            head = txt.lstrip()[:2500]
            if "Skip to main content" in head or "arxiv-logo" in head:
                continue
            if len(txt.strip()) >= MIN_BODY:
                return txt
        except Exception:
            continue
    return None


def _frontmatter(meta):
    L = ["---", "kg_type: none", "type: paper-fulltext",
         f'title: "{_y(meta.get("title"))}"']
    for k in ("oaid", "arxiv", "doi"):
        if meta.get(k):
            L.append(f'{k}: "{_y(meta[k])}"')
    if meta.get("src_url"):
        L.append(f'src_url: "{_y(meta["src_url"])}"')
    L.append(f"retrieved: {date.today().isoformat()}")
    L.append(f'source: {meta.get("source", "inbox")}')
    L.append("---")
    return "\n".join(L)


def _save(meta, body):
    """frontmatter+body 원자 저장. 이미 있으면 skip(멱등). 반환=rel path or None."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # stem을 title보다 앞에: inbox 제목이 실제 제목으로 바뀌어도 파일명 불변(재인제스트 멱등)
    name = meta.get("oaid") or (f"arxiv-{meta['arxiv']}" if meta.get("arxiv") else None) \
        or _slug(meta.get("stem") or meta.get("title") or "paper")
    p = OUT_DIR / f"{name}.md"
    if p.exists():
        _log(f"  skip(존재): {p.name}")
        return None
    tmp = p.with_suffix(".md.tmp")
    tmp.write_text(_frontmatter(meta) + "\n\n" + body.strip() + "\n", encoding="utf-8")
    tmp.replace(p)                               # 원자 저장
    return f"Sources/_fulltext/{p.name}"


def _convert_and_save(pdf_path, meta):
    """PDF → md → 저장 → (성공+본문 충분 시) 원본 PDF 삭제. 실패=PDF 유지+에러 반환."""
    try:
        body = _convert(pdf_path)
    except Exception as e:
        return {"ok": False, "error": f"convert {type(e).__name__}: {e}"}
    if len(body.strip()) < MIN_BODY:             # 변환 실패(스캔본/암호화 등) → PDF 보존
        return {"ok": False, "error": f"본문 {len(body.strip())}자 < {MIN_BODY}(변환 실패 추정)"}
    if meta.get("title") == meta.get("stem"):    # 제목이 파일명 그대로면 본문 헤딩 폴백
        t = _title_from_md(body)
        if t:
            meta = {**meta, "title": t}
    rel = _save(meta, body)
    if rel is None:
        return {"ok": True, "skipped": True}     # 이미 존재
    try:
        Path(pdf_path).unlink()                  # 사용자 firm: 변환 후 원본 삭제
    except OSError as e:
        _log(f"  ⚠ PDF 삭제 실패(무시): {e}")
    return {"ok": True, "rel": rel}


def run_inbox():
    """Inbox/papers/*.pdf → 변환·저장·삭제. 부분쓰기 방어(크기 안정화 후)."""
    INBOX.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(INBOX.glob("*.pdf"))
    if not pdfs:
        _log(f"Inbox 비어있음: {INBOX}")
        return
    done, fail = 0, 0
    for pdf in pdfs:
        s1 = pdf.stat().st_size
        time.sleep(1.2)
        if pdf.stat().st_size != s1:             # 다운로드 중(크기 변동) → 다음 회차로
            _log(f"  대기(쓰는 중): {pdf.name}")
            continue
        meta = {"title": pdf.stem, "stem": pdf.stem, "source": "inbox",
                "src_url": f"inbox:{pdf.name}"}
        if _DOI_STEM.match(pdf.stem):            # DOI-슬러그 파일명 → 실제 제목 조회(P2-6)
            doi = pdf.stem.replace("_", "/", 1)
            t = _crossref_title(doi)
            if t:
                meta.update(title=t, doi=doi)
        r = _convert_and_save(pdf, meta)
        if r.get("ok"):
            done += 1
            _log(f"  ✓ {pdf.name} → {r.get('rel') or '(이미 존재)'}")
        else:
            fail += 1
            _log(f"  ✗ {pdf.name}: {r.get('error')}")
    _log(f"Inbox 처리: {done} 성공 / {fail} 실패(PDF 보존)")


def run_fetch(queries, report=True):
    """각 query: OA 검색→pdf 다운(매직바이트 확인)→변환·저장·삭제. 실패=_unfetched 리스트.
    report=False면 UNFETCHED 파일을 건드리지 않고 실패 목록만 반환(enrich 등 프로그램 호출용 —
    유저가 수동으로 쌓은 '학교 라이센스로 받아주세요' 리스트를 덮어쓰지 않기 위함, doc62)."""
    INBOX.mkdir(parents=True, exist_ok=True)
    unfetched = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        time.sleep(PACE)                          # 요청 간 간격(429 방지)
        if _ARXIV_ID.fullmatch(q):                # arxiv id 직행: arxiv API(429 병목) 건너뜀
            m = {"ok": True, "arxiv": q, "title": "", "oaid": "", "doi": "",
                 "pdf_url": f"https://arxiv.org/pdf/{q}"}
        elif re.fullmatch(r"10\.\d{4,9}/\S+", q):  # DOI → OpenAlex OA 직행(arxiv 검색 낭비 skip)
            m = _openalex(q)
            if not m.get("ok"):
                _log(f"  ? {q}: {m.get('error')}")
                unfetched.append({"title": q, "why": m.get("error", "조회 실패"), "doi": q, "url": ""})
                continue
            if not m.get("pdf_url"):
                _log(f"  $ 유료(OA 없음): {m['title'][:60]}")
                unfetched.append({"title": m["title"], "why": "OA 없음(유료)",
                                  "url": m.get("publisher_url") or m.get("landing") or "",
                                  "doi": m.get("doi") or q})
                continue
        else:
            ax = _arxiv_search(q)                 # title 검색만 arxiv API 사용
            if ax:
                m = {"ok": True, **ax, "oaid": "", "doi": ""}
            else:
                m = _openalex(q)                  # 비-arxiv OA
                if not m.get("ok"):
                    _log(f"  ? {q}: {m.get('error')}")
                    unfetched.append({"title": q, "why": m.get("error", "조회 실패"), "url": ""})
                    continue
                if not m.get("pdf_url"):          # OA 없음 = 유료 → 수동 리스트
                    _log(f"  $ 유료(OA 없음): {m['title'][:60]}")
                    unfetched.append({"title": m["title"], "why": "OA 없음(유료)",
                                      "url": m.get("publisher_url") or m.get("landing") or "",
                                      "doi": m.get("doi")})
                    continue
        # arxiv면 ar5iv HTML 우선(공백 보존, RAG 품질↑) — PDF는 폴백(사용자 baseline).
        if m.get("arxiv"):
            body = _try_ar5iv(m["arxiv"])
            if body:
                title = m.get("title") or _title_from_md(body) or f"arXiv {m['arxiv']}"
                rel = _save({**m, "title": title, "source": "ar5iv",
                             "src_url": f"https://ar5iv.org/abs/{m['arxiv']}"}, body)
                _log(f"  ✓ (ar5iv) {title[:60]} → {rel or '(이미 존재)'}")
                continue
            _log(f"    ar5iv 실패 → PDF 폴백: {q}")
        try:
            raw = _fetch_bytes(m["pdf_url"])
        except Exception as e:
            unfetched.append({"title": m["title"], "why": f"다운로드 실패 {type(e).__name__}",
                              "url": m["pdf_url"]})
            continue
        if not _is_pdf(raw):                      # landing HTML 등 → 유지 안 함, 리스트로
            _log(f"  ✗ PDF 아님(랜딩페이지 추정): {m['title'][:60]}")
            unfetched.append({"title": m["title"], "why": "pdf_url이 PDF 아님",
                              "url": m.get("landing") or m["pdf_url"]})
            continue
        pdf = INBOX / f"{m.get('oaid') or _slug(m['title'])}.pdf"
        pdf.write_bytes(raw)
        meta = {**m, "source": "oa", "src_url": m["pdf_url"]}
        r = _convert_and_save(pdf, meta)
        if r.get("ok"):
            _log(f"  ✓ {m['title'][:60]} → {r.get('rel') or '(이미 존재)'}")
        else:
            _log(f"  ✗ 변환 실패: {m['title'][:60]}: {r.get('error')}")
            unfetched.append({"title": m["title"], "why": r.get("error"),
                              "url": m.get("landing") or ""})
    if report:
        _write_unfetched(unfetched)
    return unfetched


def _write_unfetched(items):
    if not items:
        UNFETCHED.unlink(missing_ok=True)         # 재실행서 다 받았으면 스테일 리스트 제거
        _log("못 받은 논문 없음.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["---", "kg_type: none", "type: paper-unfetched",
             f"updated: {date.today().isoformat()}", "---",
             "# 못 받은 논문 (학교 라이센스로 브라우저에서 다운 → Inbox/papers/ 에 넣어주세요)", ""]
    for it in items:
        u = it.get("url") or ""
        doi = f" · doi:{it['doi']}" if it.get("doi") else ""
        lines.append(f"- **{it['title']}** — {it.get('why', '')}{doi}"
                     + (f"\n  - {u}" if u else ""))
    UNFETCHED.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"못 받은 {len(items)}편 → {UNFETCHED}")


def _smoke():
    checks = [
        ("OUT_DIR = Sources/_fulltext(스캐너 밖)", OUT_DIR.name == "_fulltext" and OUT_DIR.parent.name == "Sources"),
        ("INBOX = Inbox/papers(전용, 공유루트 아님)", INBOX.parent.name == "Inbox" and INBOX.name == "papers"),
        ("frontmatter kg_type none", "kg_type: none" in _frontmatter({"title": "T"})),
        ("frontmatter type paper-fulltext", "type: paper-fulltext" in _frontmatter({"title": "T"})),
        ("frontmatter research/add 태그 없음(export 이중잠금)", "research/add" not in _frontmatter({"title": "T", "oaid": "W1"})),
        ("magic byte: %PDF 통과", _is_pdf(b"%PDF-1.7\n...")),
        ("magic byte: HTML 거부", not _is_pdf(b"<!doctype html>")),
        ("slug 한글보존·안전", _slug("잡음 강건 ADD!!") == "잡음-강건-ADD"),
        ("MIN_BODY 게이트 존재", MIN_BODY >= 1000),
        ("_y 개행·따옴표 제거", _y('a\nb"c') == "a bc" or _y('a\nb"c') == "a b'c"),
    ]
    ok = sum(1 for _, v in checks if v)
    for n, v in checks:
        print(f"  [{'OK' if v else 'FAIL'}] {n}")
    print(f"\ningest_papers smoke: {ok}/{len(checks)}")
    return 0 if ok == len(checks) else 1


def main(argv):
    if "--smoke" in argv or not argv:
        return _smoke()
    if "--inbox" in argv:
        run_inbox()
        return 0
    if "--fetch" in argv:
        i = argv.index("--fetch")
        run_fetch([argv[i + 1]] if len(argv) > i + 1 else [])
        return 0
    if "--fetch-file" in argv:
        i = argv.index("--fetch-file")
        qs = Path(argv[i + 1]).read_text(encoding="utf-8").splitlines()
        run_fetch(qs)
        return 0
    print("usage: --smoke | --inbox | --fetch \"<query>\" | --fetch-file <path>")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
