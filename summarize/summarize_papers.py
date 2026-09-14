# -*- coding: utf-8 -*-
"""논문 정리본 일괄 생성 — 로컬 Gemma, 전량 사전 캐시 (doc61 후속).

Sources/_fulltext/*.md 전체를 순회하며 _sum-<stem>.md 캐시가 없는 것만 생성.
재개 가능(캐시 존재=스킵) — 중간에 끊고 다시 돌려도 됨. claude 비용 0.

⚠ 1090편 × 30~60초 ≈ 9~18시간 — 밤에 걸어두는 용도. Ollama 실행 필수.
   앱에서는 논문 열 때 on-demand 생성되므로 이 스크립트는 선택사항.

    python summarize_papers.py            # 전량(캐시 스킵)
    python summarize_papers.py --limit 20 # 최신 20편만
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import researchos_local as LOC                   # noqa: E402
import paper_summary as A                         # noqa: E402  (_SUM_PROMPT·_untrusted 재사용)

FULLTEXT = A.VAULT / "Sources" / "_fulltext"


def main():
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if not LOC.available():
        print("Ollama 미가동 — 종료")
        return 1
    papers = [p for p in FULLTEXT.glob("*.md") if not p.name.startswith("_")]
    papers.sort(key=lambda p: p.stat().st_mtime, reverse=True)   # 최신 우선
    todo = [p for p in papers if not p.with_name(f"_sum-{p.stem}.md").is_file()]
    if limit:
        todo = todo[:limit]
    print(f"전체 {len(papers)}편 · 정리본 미생성 {len(todo)}편 시작 (편당 30~60초)")
    ok = fail = 0
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        r = A.h_paper_summary({"path": f"Sources/_fulltext/{p.name}"})
        if r.get("ok"):
            ok += 1
            print(f"  ✓ [{i}/{len(todo)}] {p.stem[:50]} ({time.time() - t0:.0f}s)")
        else:
            fail += 1
            print(f"  ✗ [{i}/{len(todo)}] {p.stem[:50]}: {r.get('error')}")
    print(f"\n완료: {ok} 성공 / {fail} 실패")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
