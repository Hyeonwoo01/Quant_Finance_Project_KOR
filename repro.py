"""재현 정보: 결과가 어떤 데이터·코드로 계산됐는지.

무료 데이터(Yahoo 등)는 과거 값이 나중에 고쳐지기도 해서, 같은 화면이라도 받은 시점에 따라 숫자가 달라질 수 있다.
데이터 지문(기간·길이·값의 해시)과 코드 커밋을 함께 남기면, 결과가 달라졌을 때 데이터가 바뀐 건지 코드가 바뀐 건지 가릴 수 있다.
"""
import hashlib
import subprocess
from pathlib import Path

import numpy as np


def fingerprint(series):
    """기간 · 길이 · #값 해시 8자리. 값은 소수 6자리로 반올림해 해시한다 (표현 차이로 바뀌지 않게)."""
    if series is None or len(series) == 0:
        return "데이터 없음"
    v = np.round(np.asarray(series, float), 6)
    h = hashlib.sha1(v.tobytes() + f"{series.index[0]}{series.index[-1]}".encode()).hexdigest()[:8]
    return f"{series.index[0].date()} ~ {series.index[-1].date()} · {len(series):,}일 · #{h}"


def code_version():
    """지금 돌고 있는 코드의 커밋 (git 저장소에서 실행할 때). 모르면 '알 수 없음'."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or "알 수 없음"
    except Exception:
        return "알 수 없음"
