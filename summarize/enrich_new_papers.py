# -*- coding: utf-8 -*-
"""신규 digest 논문 전문 다운로드 후 요약 (doc62 후속 — "전문 다운로드 후 요약").

Daily Digest가 자동 노트(score >= AUTO_NOTE_THRESHOLD)를 만든 논문에 대해:
  1. OA/arxiv 전문 다운로드 (ingest_papers.run_fetch) → Sources/_fulltext/arxiv-<id>.md
  2. 로컬 Gemma 구조화 요약 (researchos_api.h_paper_summary) → _fulltext/_sum-arxiv-<id>.md 캐시
  3. digest Source 노트(Sources/arxiv-<id>.md)에 요약 임베드 링크 + frontmatter 플래그 주입

두 단계는 독립적으로 fail-soft (한쪽 실패가 다른쪽·digest·sync를 막지 않음):
  - OA 미제공/다운 실패 → 전문 없음 → 요약 스킵(마커 안 남김 → 다음 실행 재시도)
  - Ollama 미가동/요약 실패 → summary-status 미기록(pending) → 다음 실행/앱 on-demand로 이월

⚠ 격리(researchos_api.py 주석): Sources/*.md 는 RAG(index_vault)+KG 추출 스캔 대상.
   LLM 생성 요약 본문을 여기 인라인하면 격리 우회 → L-KG 소스·RAG 컨텍스트 오염.
   따라서 노트에는 `![[_sum-arxiv-<id>]]` 임베드 링크 한 줄만 넣는다(_ 접두 = 인덱서 제외 유지).

용법:
    python enrich_new_papers.py 2609.11763 2609.11404   # 명시 id: 전문 없으면 다운+요약
    python enrich_new_papers.py --scan                    # digest 노트 중 미요약+전문有만 요약(재fetch X)
    python enrich_new_papers.py 2609.11763 --scan         # daily_digest 배선용(오늘분 fetch + 밀린분 소화)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import researchos_local as LOC          # noqa: E402
import paper_summary as A               # noqa: E402  (_SUM_PROMPT·h_paper_summary 재사용)
import ingest_papers as ING            # noqa: E402  (run_fetch 재사용)

VAULT = A.VAULT
SOURCES = VAULT / "Sources"
FULLTEXT = SOURCES / "_fulltext"

_SECTION_HEAD = "## 🤖 자동 정리 (로컬 LLM · 검토필요 🔴)"
_ARXIV_FM = re.compile(r'^arxiv:\s*"?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)"?', re.MULTILINE)


def _read_note(path: Path):
    """frontmatter(양끝 --- 사이 원문)와 body 분리. frontmatter 없으면 (None, 전체)."""
    text = path.read_text("utf-8", "replace")
    if text.startswith("---"):
        parts = text.split("---", 2)     # ['', fm, body]
        if len(parts) == 3:
            return parts[1], parts[2]
    return None, text


def _fm_has_done(fm: str) -> bool:
    return bool(fm) and bool(re.search(r"^summary-status:\s*done\s*$", fm, re.MULTILINE))


def _fm_set(fm: str, key: str, value: str) -> str:
    """frontmatter 원문에서 key 라인 교체 또는 말미 추가(원문 개행 보존)."""
    line = f"{key}: {value}"
    pat = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    if pat.search(fm):
        return pat.sub(line, fm)
    return fm.rstrip("\n") + "\n" + line + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """tmp+replace 원자 저장 (daily_digest timeout이 도중에 죽여도 노트 잘림 방지)."""
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _fulltext_index() -> dict:
    """_fulltext/*.md 를 1회 훑어 {arxiv_id: path}. frontmatter arxiv 로 매칭(stem 추측 금지).
    OpenAlex 경로로 들어온 논문은 stem이 W<id>라 파일명만으론 못 찾음 → arxiv 필드가 정본."""
    idx = {}
    if not FULLTEXT.is_dir():
        return idx
    for p in FULLTEXT.glob("*.md"):
        if p.name.startswith("_"):
            continue
        fm, _ = _read_note(p)
        if not fm:
            continue
        m = _ARXIV_FM.search(fm)
        if m:
            idx.setdefault(m.group(1), p)
            idx.setdefault(m.group(1).split("v")[0], p)   # 버전 무시 매칭도 등록
    return idx


def _scan_ids() -> list:
    """digest-score 있고 아직 요약 안 된(summary-status != done) Source 노트 id."""
    ids = []
    for p in sorted(SOURCES.glob("arxiv-*.md")):
        fm, _ = _read_note(p)
        if fm and "digest-score:" in fm and not _fm_has_done(fm):
            ids.append(p.stem[len("arxiv-"):])
    return ids


def _inject(note: Path, sum_stem: str) -> bool:
    """Source 노트에 요약 임베드 링크 섹션 + frontmatter 플래그 주입(원자). done이면 재주입 안 함."""
    fm, body = _read_note(note)
    if fm is None:
        return False                      # frontmatter 없는 노트는 건드리지 않음
    fm = _fm_set(fm, "fulltext", f'"{sum_stem}"')
    fm = _fm_set(fm, "summary-status", "done")
    if _SECTION_HEAD not in body:
        body = body.rstrip() + (
            f"\n\n{_SECTION_HEAD}\n\n"
            f"![[_sum-{sum_stem}]]\n"      # Obsidian 임베드: 유저 눈엔 전문, 인덱서엔 링크 문자열만
            f"> 로컬 Gemma 자동 정리(검증 안 됨). 원문 전문: [[{sum_stem}]]\n"
        )
    _atomic_write(note, f"---{fm}---{body if body.endswith(chr(10)) else body + chr(10)}")
    return True


def main(argv) -> int:
    explicit = [a for a in argv if not a.startswith("--")]
    do_scan = "--scan" in argv

    # 1) fetch: 명시 id 중 전문이 아직 없는 것만 (scan분은 재fetch 안 함 — OA 없는 논문 매일 두드리기 방지)
    idx = _fulltext_index()
    need_fetch = [i for i in explicit if i not in idx]
    if need_fetch:
        print(f"전문 다운로드 시도: {len(need_fetch)}편 → {need_fetch}")
        try:
            unf = ING.run_fetch(need_fetch, report=False)   # UNFETCHED 파일 안 건드림
            if unf:
                print(f"  전문 미취득 {len(unf)}편(OA 미제공/실패) — 요약은 스킵, 다음 실행 재시도")
        except Exception as e:
            print(f"  ⚠ fetch 오류(비치명): {type(e).__name__}: {e}")
        idx = _fulltext_index()           # fetch 결과 반영해 재빌드

    # 2) summarize: (명시 ∪ scan) 중 노트 존재 & 미완료 & 전문 존재
    targets = list(dict.fromkeys(explicit + (_scan_ids() if do_scan else [])))
    if not targets:
        print("대상 논문 없음.")
        return 0

    ok = ft_miss = pending = skipped = 0
    for aid in targets:
        note = SOURCES / f"arxiv-{aid}.md"
        if not note.exists():
            continue
        fm, _ = _read_note(note)
        if _fm_has_done(fm):
            skipped += 1
            continue
        ft = idx.get(aid) or idx.get(aid.split("v")[0])
        if ft is None:
            ft_miss += 1
            print(f"  · 전문 없음(요약 보류): {aid}")
            continue
        rel = f"Sources/_fulltext/{ft.name}"
        r = A.h_paper_summary({"path": rel})
        if not r.get("ok"):
            pending += 1
            print(f"  · 요약 보류(pending): {aid}: {r.get('error')}")
            continue
        if _inject(note, ft.stem):
            ok += 1
            print(f"  ✓ 요약 주입: {aid} ← {ft.name}"
                  f"{' (캐시)' if r.get('cached') else ''}")

    print(f"\nenrich 완료: {ok} 요약 · {ft_miss} 전문없음 · {pending} 보류 · {skipped} 기완료")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
