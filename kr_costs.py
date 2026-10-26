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


# 가격제한폭: 하루에 전날 종가 대비 이만큼까지만 움직일 수 있다 (2015-06-15부터 ±30%, 그 전 ±15%)
PRICE_LIMIT = [("1998-12-07", 0.15), ("2015-06-15", 0.30)]
LIMIT_MARGIN = 0.005             # 호가 단위로 딱 맞지 않는 만큼 — 제한폭에서 0.5%p 이내로 끝나면 제한폭에 걸린 날로 본다


def limit_rate(dates):
    """날짜별 가격제한폭 (소수)."""
    idx = pd.DatetimeIndex(dates)
    starts = pd.DatetimeIndex([r[0] for r in PRICE_LIMIT])
    rates = np.array([r[1] for r in PRICE_LIMIT])
    return pd.Series(rates[np.maximum(starts.searchsorted(idx, side="right") - 1, 0)], index=idx)


def limit_hits(simple_returns):
    """날짜별 (하한가 근처로 끝남, 상한가 근처로 끝남) — simple_returns: 전날 종가 대비 등락률 (Series 또는 DataFrame).

    하한가로 끝난 날은 팔자 주문이 쌓여 종가에 팔 수 없고, 상한가로 끝난 날은 살 수 없다고 본다."""
    lim = limit_rate(simple_returns.index) - LIMIT_MARGIN
    if isinstance(simple_returns, pd.DataFrame):
        return simple_returns.le(-lim, axis=0), simple_returns.ge(lim, axis=0)
    return simple_returns <= -lim, simple_returns >= lim


def apply_price_limits(target, daily_log_returns, lag=1):
    """원하는 보유 비중(판단일별) → 가격제한폭 때문에 실제로 가질 수 있었던 비중.

    판단일 t의 주문은 t+lag일 종가에 체결된다. 그날 하한가 근처로 끝났으면 줄이는(파는) 주문이, 상한가 근처로 끝났으면
    늘리는(사는) 주문이 체결되지 않아 직전 비중을 유지하고 다음 날 다시 시도한다. 지수는 제한폭에 닿을 일이 없어 그대로.
    """
    move = np.expm1(daily_log_returns).shift(-lag).reindex(target.index)
    down, up = limit_hits(move.fillna(0.0))
    out, prev = np.empty(len(target)), 0.0
    for i, (w, dn, u) in enumerate(zip(target.values, down.values, up.values)):
        if (w < prev and dn) or (w > prev and u):
            w = prev
        out[i] = prev = w
    return pd.Series(out, index=target.index)
