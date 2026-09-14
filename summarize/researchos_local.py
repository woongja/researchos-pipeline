# -*- coding: utf-8 -*-
"""ResearchOS Local LLM — Ollama 래퍼 (doc63, 싼 정형작업 티어).

티어 분리(advisor): claude=깊은 판단(팀·novelty·컨셉), **local=싼 정형작업(회의록 정리·요약·
태그 추출)**. 비용·지연·오프라인 이득(품질은 claude급 아니고 '충분히 좋음'). 상주 금지—
keep_alive 짧게 해 유휴 시 언로드(RTX4060 8GB=사용자 학습 VRAM 보호).

    python researchos_local.py --smoke     # Ollama 없이 로직 검증
    python researchos_local.py --live "요약해줘: ..."
"""
import json
import os
import sys
import urllib.request
import urllib.error

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
LOCAL_MODEL = os.environ.get("RESEARCHOS_LOCAL_MODEL", "gemma3n:e4b")
KEEP_ALIVE = os.environ.get("RESEARCHOS_LOCAL_KEEPALIVE", "5m")   # 유휴 후 언로드(학습 VRAM 보호)


def available(timeout: int = 3, model: str = None) -> bool:
    """Ollama 서버 살아있나 + 모델 있나. 없으면 호출측이 claude 폴백."""
    want = model or LOCAL_MODEL
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=timeout) as r:
            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        return any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
    except Exception:
        return False


def run_local(prompt: str, model: str = None, timeout: int = 240) -> dict:
    """Ollama /api/generate → {ok, text} 또는 {ok:False, error}. 실패=호출측 claude 폴백."""
    body = json.dumps({
        "model": model or LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"ollama http {e.code}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}"}
    text = (data.get("response") or "").strip()
    return {"ok": True, "text": text, "model": model or LOCAL_MODEL} if text \
        else {"ok": False, "error": "빈 응답"}


def _smoke() -> int:
    checks = [
        ("OLLAMA host 기본", OLLAMA.endswith(":11434")),
        ("모델 env 기본 gemma3n", "gemma3n" in LOCAL_MODEL),
        ("keep_alive 유휴언로드", KEEP_ALIVE not in ("", "-1")),
        ("available 함수 존재", callable(available)),
        ("run_local 함수 존재", callable(run_local)),
    ]
    ok = sum(1 for _, v in checks if v)
    for n, v in checks:
        print(f"  [{'OK' if v else 'FAIL'}] {n}")
    print(f"\nresearchos_local smoke: {ok}/{len(checks)} · available={available()}")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if "--live" in args:
        i = args.index("--live")
        print(json.dumps(run_local(args[i + 1] if len(args) > i + 1 else "hi"),
                         ensure_ascii=False, indent=2))
        sys.exit(0)
    sys.exit(_smoke())
