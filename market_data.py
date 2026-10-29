"""yfinance로 실제 시장 데이터(주가·옵션 체인·금리·배당)를 가져온다."""
import io
import re
import time
import zipfile
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import kr_data
from quant_engine import RiskNeutralPricingEngine


def load_history(ticker, years):
    t = ticker.upper()
    if t in NAVER_FIRST:                 # Yahoo에 과거 데이터가 거의 없는 지수는 네이버를 먼저
        try:
            return naver_index_history(t, years)
        except Exception:
            pass
    h = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    close = h["Close"].dropna() if not h.empty else pd.Series(dtype=float)
    if t in NAVER_INDEX and len(close) < 0.3 * 245 * years:      # 한국 지수인데 Yahoo 기록이 너무 짧으면 네이버로
        try:
            return naver_index_history(t, years)
        except Exception:
            pass
    if close.empty:
        raise ValueError(f"'{ticker}'의 주가 데이터를 찾을 수 없습니다. 티커를 확인하세요 (예: AAPL, SPY, 005930.KS).")
    close.index = close.index.tz_localize(None)
    return close


def load_ohlc(ticker, years):
    """Yahoo 일봉 시가·고가·저가·종가 (배당·분할 수정). 장중 고가·저가로 하루 변동성을 잴 때 쓴다 (quant_engine.range_variance)."""
    h = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}'의 일봉(시가·고가·저가)을 찾을 수 없습니다.")
    h = h[["Open", "High", "Low", "Close"]].copy()
    h.index = h.index.tz_localize(None)
    return h


# 한국 지수의 네이버 차트 기호. 코스피200(^KS200)은 Yahoo가 최근 1개월 요청에 1일치만 줘서(2026-10 점검) 네이버를 먼저 쓴다.
NAVER_INDEX = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ", "^KS200": "KPI200"}
NAVER_FIRST = {"^KS200"}
NAVER_CHART = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count={}&requestType=0"


def parse_naver_chart(text):
    """네이버 차트 XML의 <item data="날짜|시가|고가|저가|종가|거래량"> → 종가 Series."""
    rows = [r.split("|") for r in re.findall(r'<item data="([^"]+)"', text)]
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"), name="Close")
    return s[s > 0].sort_index()


def naver_index_history(ticker, years):
    r = requests.get(NAVER_CHART.format(NAVER_INDEX[ticker.upper()], int(years * 260) + 30), headers=_UA, timeout=20)
    r.raise_for_status()
    s = parse_naver_chart(r.text)
    if s.empty:
        raise ValueError(f"네이버 차트에 '{ticker}' 데이터가 없습니다.")
    return s[s.index > s.index[-1] - pd.DateOffset(years=years)]


# 과거 내재변동성 대용: Cboe 변동성 지수 (30일 만기 옵션으로 계산, 지수 포인트 = 연율 %)
VOL_INDEX = {"SPY": "^VIX", "VOO": "^VIX", "IVV": "^VIX", "^GSPC": "^VIX",
             "QQQ": "^VXN", "^NDX": "^VXN", "IWM": "^RVX", "^RUT": "^RVX"}


CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def parse_cboe_csv(text):
    """Cboe 지수 일별 CSV(DATE, OPEN, HIGH, LOW, CLOSE — 날짜는 월/일/연) → 종가 Series."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(pd.to_numeric(df["CLOSE"], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"), name="Close").dropna()
    return s[s > 0].sort_index()


def load_vol_index(ticker, years):
    """이 종목에 맞는 변동성 지수의 과거 값(소수, 예: 0.18). 없으면 None.

    Yahoo에서 먼저 받고, 없으면(예: RVX는 Yahoo에서 빠짐) Cboe가 무료로 올려 두는 공식 일별 CSV로 받는다."""
    sym = VOL_INDEX.get(ticker.upper())
    if sym is None:
        return None
    try:
        return load_history(sym, years) / 100.0
    except Exception:
        pass
    try:
        raw = requests.get(CBOE_URL.format(sym.lstrip("^")), headers=_UA, timeout=20)
        raw.raise_for_status()
        s = parse_cboe_csv(raw.text)
        return s[s.index > s.index[-1] - pd.DateOffset(years=years)] / 100.0
    except Exception:
        return None


_UA = {"User-Agent": "Mozilla/5.0"}
_MARKETS = {"KOSPI": ("KS", "코스피"), "KOSDAQ": ("KQ", "코스닥")}


def kr_symbols(max_age_hours=24):
    """코스피·코스닥 전 종목(주식·ETF)의 이름 ↔ 티커 표. 네이버 증권에서 받아 하루 동안 캐시한다."""
    path = kr_data.CACHE / "kr_symbols.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_pickle(path)
    rows = []
    for market, (suffix, label) in _MARKETS.items():
        for page in range(1, 60):
            js = requests.get(f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize=100",
                              headers=_UA, timeout=10).json()
            if not js.get("stocks"):
                break
            rows += [{"name": s["stockName"], "ticker": f"{s['itemCode']}.{suffix}", "market": label,
                      "cap": float(s.get("marketValueRaw") or 0), "type": s.get("stockEndType")}
                     for s in js["stocks"] if s.get("stockEndType") in ("stock", "etf")]
    df = pd.DataFrame(rows).sort_values("cap", ascending=False).drop_duplicates("ticker").reset_index(drop=True)
    kr_data.CACHE.mkdir(exist_ok=True)
    df.to_pickle(path)
    return df


def kr_info(ticker):
    """한국 종목의 (이름, 시장, 구분 'stock'/'etf'). 목록에 없거나 받지 못하면 (None, None, None)."""
    try:
        t = kr_symbols()
        hit = t[t["ticker"] == ticker.upper()]
    except Exception:
        return None, None, None
    if hit.empty:
        return None, None, None
    r = hit.iloc[0]
    return r["name"], r["market"], r.get("type")


def _norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def search_ticker(query, limit=10):
    """종목명 또는 티커 → [(티커, 표시 이름), ...] (가장 그럴듯한 순).

    - '005930' / '005930.KS' / 'SPY' 같은 코드는 그대로 (한국 6자리 코드는 시장을 찾아 붙임)
    - 한글 이름은 코스피·코스닥 목록에서: 정확히 일치 → 앞부분 일치 → 포함, 같은 순위면 시가총액 큰 순
    - 영문 이름(소문자·공백 포함 등)은 Yahoo Finance 검색
    """
    q = str(query or "").strip()
    if not q:
        return []
    m = re.fullmatch(r"(\d{6})(?:\.(KS|KQ))?", q, flags=re.I)
    if m:
        code, sfx = m.group(1), (m.group(2) or "").upper()
        try:
            hit = kr_symbols().query("ticker.str.startswith(@code)", engine="python")
        except Exception:
            hit = pd.DataFrame()
        if not hit.empty and not sfx:
            return [(r.ticker, f"{r['name']} ({r.ticker} · {r.market})") for _, r in hit.iterrows()]
        ticker = f"{code}.{sfx or 'KS'}"
        name = hit["name"].iloc[0] if not hit.empty else ""
        return [(ticker, f"{name} ({ticker})" if name else ticker)]
    if q.isascii():
        # 'NAVER', 'LG', 'KODEX 200'처럼 영문 이름의 한국 종목: 이름이 정확히 같으면 먼저 보여줌
        try:
            table = kr_symbols()
            kr = table[table["name"].map(_norm) == _norm(q)]
        except Exception:
            kr = pd.DataFrame()
        kr_hits = [(r.ticker, f"{r['name']} ({r.ticker} · {r.market})") for _, r in kr.iterrows()]
        # 'AAPL', 'BRK-B', '^KS11'처럼 대문자 티커 모양이면 그대로 사용
        if re.fullmatch(r"[A-Z0-9^.\-=]{1,10}", q):
            return kr_hits + [(q, f"{q} (미국 등 해외 티커)" if kr_hits else q)]
        if kr_hits:
            return kr_hits
        try:
            quotes = yf.Search(q, max_results=limit).quotes
        except Exception:
            quotes = []
        out = [(x["symbol"], f"{x.get('shortname') or x.get('longname') or x['symbol']} ({x['symbol']}"
                f"{' · ' + x['exchDisp'] if x.get('exchDisp') else ''})")
               for x in quotes if x.get("symbol") and x.get("quoteType") in ("EQUITY", "ETF", "INDEX", None)]
        return out or [(q.upper(), q.upper())]
    try:
        table = kr_symbols()
    except Exception:
        return []
    nq, names = _norm(q), table["name"].map(_norm)
    rank = np.select([names == nq, names.str.startswith(nq), names.str.contains(nq, regex=False)], [0, 1, 2], 9)
    hit = table.assign(rank=rank)[rank < 9].sort_values(["rank", "cap"], ascending=[True, False]).head(limit)
    return [(r.ticker, f"{r['name']} ({r.ticker} · {r.market})") for _, r in hit.iterrows()]


def is_us_ticker(ticker):
    """미국 등 해외 티커인지. 한국 지수(^KS11 코스피, ^KQ11 코스닥, ^KS200)는 점이 없어도 한국으로 본다."""
    if ticker.upper().startswith(("^KS", "^KQ")):
        return False
    return "." not in ticker or ticker.endswith(".US")


def load_macro(ticker, years=10):
    """방향 예측용 시장 전체 지표. 실패하면 None.

    한국 종목: 기준 지수를 코스피/코스닥으로 바꾸고, 원/달러 환율과 코스피·코스닥 둘 다 추가한다.
    """
    us = is_us_ticker(ticker)
    market = "SPY" if us else ("^KQ11" if ticker.endswith(".KQ") else "^KS11")
    pairs = [("^VIX", "vix"), ("^VIX3M", "vix3m"), ("^TNX", "tnx"), ("^IRX", "irx"), ("HYG", "hyg"),
             ("IEF", "ief"), ("^SKEW", "skew"), ("^VVIX", "vvix"), (market, "market")]
    if not us:
        pairs += [("KRW=X", "usdkrw"), ("^KS11", "kospi"), ("^KQ11", "kosdaq")]
    try:
        d = yf.download(list(dict.fromkeys(s for s, _ in pairs)), period=f"{years}y", auto_adjust=True,
                        progress=False)["Close"]
        d = pd.DataFrame({name: d[sym] for sym, name in pairs})      # 같은 지수가 market/kospi 두 이름을 가질 수 있음
        d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
        d = d.ffill(limit=5).dropna()
    except Exception:
        return None
    if not us:      # 한국은행 ECOS 금리 (키가 있을 때만; 실패해도 나머지는 그대로 사용)
        try:
            rates = kr_data.ecos_rates(d.index[0])
            if rates is not None:
                d = d.join(rates.reindex(d.index.union(rates.index)).ffill(limit=5).reindex(d.index))
        except Exception:
            pass
    return d if len(d) > 500 else None


# 미국 주식 요인 수익률 (Kenneth French 교수 데이터 라이브러리, 무료·일간, 보통 1~2개월 늦게 갱신)
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FRENCH_FILES = ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily")


def parse_french_csv(text):
    """French 라이브러리 CSV(앞뒤 설명 줄 + 'YYYYMMDD, 값…' 표, 단위 %) → 소수 단위 DataFrame (첫 번째 표만)."""
    header, rows = None, []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if len(cells) > 1 and cells[0] == "" and any(re.search(r"[A-Za-z]", c) for c in cells[1:]):
                header = cells[1:]
            continue
        if re.fullmatch(r"\d{8}", cells[0]):
            rows.append([cells[0], *cells[1:len(header) + 1]])
        elif rows:
            break                                   # 첫 표가 끝남 (뒤는 저작권 줄이나 다른 표)
    if not rows:
        raise ValueError("French 데이터 형식을 읽지 못했습니다")
    df = pd.DataFrame(rows, columns=["date", *header])
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce").where(lambda x: x > -99.0) / 100.0
    return df.rename(columns=lambda c: c.strip())


def load_fama_french(max_age_hours=24):
    """파마-프렌치 5요인(Mkt-RF·SMB·HML·RMW·CMA) + 모멘텀(Mom) + 무위험 수익률(RF), 일간 소수 단위. 실패하면 None.

    하루 동안 파일로 캐시한다 (원본은 한 달에 한 번 정도 갱신)."""
    path = kr_data.CACHE / "fama_french_daily.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_pickle(path)
    try:
        parts = []
        for name in FRENCH_FILES:
            raw = requests.get(FRENCH_URL.format(name), headers=_UA, timeout=30)
            raw.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(raw.content)) as z:
                parts.append(parse_french_csv(z.read(z.namelist()[0]).decode("latin-1")))
        ff = parts[0].join(parts[1], how="inner").dropna()
    except Exception:
        return pd.read_pickle(path) if path.exists() else None      # 오래된 캐시라도 있으면 쓴다
    kr_data.CACHE.mkdir(exist_ok=True)
    ff.to_pickle(path)
    return ff


def risk_free_rate(ticker="", default=0.04):
    """종목 통화에 맞는 단기 무위험이자율과 출처. 반환: (금리(소수), 출처 이름).

    한국 종목(원화) → 한국은행 ECOS CD(91일), 그 외 → 미국 13주 국채(^IRX). 실패하면 기본값."""
    if ticker and not is_us_ticker(ticker):
        kr = kr_data.kr_short_rate()
        if kr is not None and np.isfinite(kr):
            return kr, "CD 91일 (ECOS)"
        return default, "기본값 (ECOS 조회 실패)"
    try:
        v = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        if len(v):
            return float(v.iloc[-1]) / 100.0, "미국 13주 국채 (^IRX)"
    except Exception:
        pass
    return default, "기본값 (^IRX 조회 실패)"


def dividend_yield(ticker, spot):
    """최근 1년 배당금 합 / 현재가 (연속 배당수익률 근사)."""
    try:
        d = yf.Ticker(ticker).dividends
        if d.empty:
            return 0.0
        d.index = d.index.tz_localize(None)
        last_year = d[d.index > pd.Timestamp.now() - pd.Timedelta(days=365)]
        return float(last_year.sum()) / spot
    except Exception:
        return 0.0


def _quote_price(row):
    bid, ask, last = row.get("bid", 0) or 0, row.get("ask", 0) or 0, row.get("lastPrice", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2, "mid"
    return (last, "last") if last > 0 else (np.nan, None)


def atm_implied_vol(ticker, spot, target_cal_days, r, q):
    """목표 기간에 가장 가까운 만기의 ATM 콜/풋 호가에서 IV를 역산한다.

    반환: dict(iv, expiry, cal_days, strike, call_px, put_px, source) 또는 옵션이 없으면 None.
    """
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
    except Exception:
        expiries = ()
    if not expiries:
        return None

    today = date.today()
    days = [(datetime.strptime(e, "%Y-%m-%d").date() - today).days for e in expiries]
    cands = [(abs(d - target_cal_days), e, d) for e, d in zip(expiries, days) if d >= 7]
    if not cands:
        return None
    _, expiry, cal_days = min(cands)

    chain = t.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    strike = float(calls.loc[(calls["strike"] - spot).abs().idxmin(), "strike"])
    T = cal_days / 365.0

    ivs, prices, sources = [], {}, set()
    for typ, df in (("call", calls), ("put", puts)):
        row = df[df["strike"] == strike]
        if row.empty:
            continue
        row = row.iloc[0].to_dict()
        px, src = _quote_price(row)
        prices[typ] = px
        if src:
            iv = RiskNeutralPricingEngine(spot, strike, T, r, typ, q).implied_volatility(px)
            if np.isfinite(iv):
                ivs.append(iv)
                sources.add(src)
        if not ivs and (row.get("impliedVolatility") or 0) > 0.01:   # 최후 수단: yfinance 제공 IV
            ivs.append(float(row["impliedVolatility"]))
            sources.add("yahoo")
    if not ivs:
        return None
    return {"iv": float(np.mean(ivs)), "expiry": expiry, "cal_days": cal_days, "strike": strike,
            "call_px": prices.get("call", np.nan), "put_px": prices.get("put", np.nan),
            "source": "/".join(sorted(sources))}
