"""한국 주식 매매 비용: 팔 때 내는 증권거래세(연도별 세율) + 수수료·체결 비용.

증권거래세는 매도 금액에만 붙는다. 코스피는 '증권거래세 + 농어촌특별세 0.15%', 코스닥은 증권거래세만인데,
아래 표는 두 시장 모두 '팔 때 내는 총 세율'이다. 세율은 해마다 바뀌어 왔으므로 거래한 날짜의 세율을 쓴다.
ETF·ETN은 증권거래세가 없고, 지수(^KS11 등)는 직접 사고팔 수 없어 세금 없이 계산한다.

세율이 바뀌면 SELL_TAX 표에 한 줄만 추가하면 된다. 2026년 줄은 2025년 세법 개정(거래세를 2023년 수준으로 환원) 기준이다.
"""
import re

import numpy as np
import pandas as pd

# (시행일, 코스피 총 세율, 코스닥 총 세율) — 매도 금액 대비
SELL_TAX = [
    ("2000-01-01", 0.0030, 0.0030),
    ("2019-05-30", 0.0025, 0.0025),
    ("2021-01-01", 0.0023, 0.0023),
    ("2023-01-01", 0.0020, 0.0020),
    ("2024-01-01", 0.0018, 0.0018),
    ("2025-01-01", 0.0015, 0.0015),
    ("2026-01-01", 0.0020, 0.0020),
]
KIND_LABELS = {"KOSPI": "코스피 주식", "KOSDAQ": "코스닥 주식", "ETF": "ETF (거래세 없음)", "INDEX": "지수 (직접 매매 불가, 세금 없이 계산)"}
_ETF_BRANDS = re.compile(r"^(KODEX|TIGER|KBSTAR|RISE|ACE|ARIRANG|KOSEF|HANARO|SOL|PLUS|TIMEFOLIO|KIWOOM|WOORI|1Q|BNK|"
                         r"FOCUS|TREX|HK|UNICORN|마이티|에셋플러스|파워|히어로즈|VITA|DAISHIN343)\b", re.I)


def tax_kind(ticker, name=None, sec_type=None):
    """세금 종류: 'KOSPI' | 'KOSDAQ' | 'ETF' | 'INDEX'. sec_type: 네이버 종목 구분('stock'/'etf')을 알면 우선 쓴다."""
    t = str(ticker).upper()
    if t.startswith("^"):
        return "INDEX"
    if sec_type == "etf" or (name and _ETF_BRANDS.match(str(name).strip())):
        return "ETF"
    return "KOSDAQ" if t.endswith(".KQ") else "KOSPI"


def sell_tax_rate(dates, kind):
    """날짜별 매도 세율 Series (소수). ETF·지수는 0."""
    idx = pd.DatetimeIndex(dates)
    if kind not in ("KOSPI", "KOSDAQ"):
        return pd.Series(0.0, index=idx)
    col = 1 if kind == "KOSPI" else 2
    starts = pd.DatetimeIndex([r[0] for r in SELL_TAX])
    rates = np.array([r[col] for r in SELL_TAX])
    pos = starts.searchsorted(idx, side="right") - 1
    return pd.Series(np.where(pos >= 0, rates[np.maximum(pos, 0)], rates[0]), index=idx)


def trade_cost(pos, cost_bps=0.0, sell_tax=0.0, lag=0):
    """보유 비중 변화 → 날짜별 비용 (소수, 수익률에서 뺄 값). pos의 각 행은 '판단한 날'.

    사고팔 때 각각 cost_bps(수수료·체결 비용), 줄인 비중에는 매도 세율을 더 낸다.
    sell_tax: 숫자 또는 날짜별 Series (sell_tax_rate, 거래일 전체 날짜). lag: 판단 → 실제 체결까지 거래일 수 —
    세율은 실제로 파는 날(판단 + lag 거래일)의 것을 쓴다 (세율이 바뀌는 1월 1일 앞뒤에서 차이가 난다)."""
    d = pos.diff().fillna(pos.iloc[0])
    if isinstance(sell_tax, pd.Series):
        tax = sell_tax.shift(-lag).ffill().reindex(pos.index).ffill().fillna(0.0)
    else:
        tax = sell_tax
    return d.abs() * cost_bps * 1e-4 + (-d).clip(lower=0) * tax
