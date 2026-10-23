"""대체 데이터: 실적 서프라이즈, 애널리스트 변경, 수급(외국인·기관), 옵션 시장 정보, 뉴스 감성.

두 종류로 나뉜다.
- 과거 기록이 있는 것 → 일별 특징으로 만들어 방향 모델에 넣고 워크포워드로 검증한다.
    실적 서프라이즈(yfinance), 애널리스트 등급·목표가 변경(yfinance, 미국 위주),
    외국인·기관 순매수(네이버 증권, 한국 종목)
- 현재 값만 무료로 구할 수 있는 것 → '현재 스냅샷'으로만 보여주고 모델에는 넣지 않는다 (검증 불가).
    종목 옵션 풋/콜 비율·IV 스큐, EPS 추정치 변경, 뉴스 감성

모든 일별 특징은 '그 날 장 마감 시점에 이미 공개된 정보'만 쓰도록 날짜를 맞춘다.
"""
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import kr_data
from quant_engine import RiskNeutralPricingEngine

UA = {"User-Agent": "Mozilla/5.0"}
CACHE = Path(__file__).parent / "cache"

ALT_FEATURES = {
    "eps_surprise": "최근 실적 서프라이즈", "eps_surprise_avg4": "최근 4분기 평균 서프라이즈",
    "pead": "실적 서프라이즈 × 경과 감쇠",
    "analyst_net": "애널리스트 순상향 건수(3개월)", "pt_change": "목표가 평균 변경률(3개월)",
    "frgn_flow_5": "외국인 순매수 강도(1주)", "frgn_flow_21": "외국인 순매수 강도(1개월)",
    "inst_flow_5": "기관 순매수 강도(1주)", "inst_flow_21": "기관 순매수 강도(1개월)",
    "frgn_hold_chg": "외국인 지분율 변화(1개월)",
    **kr_data.DART_FEATURES,
}


def krx_code(ticker):
    return ticker.split(".")[0] if ticker.endswith((".KS", ".KQ")) else None


def _effective_pos(index, stamps):
    """발표 시각 → 그 정보를 처음 쓸 수 있는 거래일 위치 (발표일 '다음' 거래일, 보수적으로)."""
    days = pd.DatetimeIndex([pd.Timestamp(s.date()) for s in stamps])
    return np.searchsorted(index.values, days.values, side="right")


# ---------------- 과거 기록이 있는 특징 ----------------
def earnings_features(ticker, index, raw=None):
    """실적 서프라이즈(%) 계열. 데이터가 없으면 None. raw: 미리 받아둔 get_earnings_dates 결과."""
    try:
        e = yf.Ticker(ticker).get_earnings_dates(limit=60) if raw is None else raw
    except Exception:
        return None
    if e is None or e.empty or "Surprise(%)" not in e:
        return None
    e = e.dropna(subset=["Surprise(%)"]).sort_index()
    if len(e) < 4:
        return None
    surprise = np.clip(e["Surprise(%)"].astype(float).values, -100, 100) / 100
    pos = _effective_pos(index, e.index)
    keep = pos < len(index)
    ev = pd.DataFrame({"s": surprise[keep], "avg4": pd.Series(surprise).rolling(4, min_periods=1).mean().values[keep],
                       "pos": pos[keep]}).groupby("pos").last()
    out = pd.DataFrame(index=index)
    out["eps_surprise"] = pd.Series(ev["s"].values, index=index[ev.index]).reindex(index).ffill()
    out["eps_surprise_avg4"] = pd.Series(ev["avg4"].values, index=index[ev.index]).reindex(index).ffill()
    last_pos = pd.Series(np.where(np.isin(np.arange(len(index)), ev.index), np.arange(len(index)), np.nan)).ffill().values
    days_since = np.arange(len(index)) - last_pos
    out["pead"] = out["eps_surprise"].values * np.exp(-days_since / 63)    # 실적 후 표류(PEAD)는 수개월에 걸쳐 약해짐
    return out


def analyst_features(ticker, index, window=63):
    """애널리스트 등급 상·하향과 목표가 변경 (최근 window 거래일 합). 데이터가 없으면 None."""
    try:
        u = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        return None
    if u is None or u.empty or "Action" not in u:
        return None
    u = u.sort_index()
    pos = _effective_pos(index, u.index)
    keep = pos < len(index)
    u, pos = u[keep], pos[keep]
    n = len(index)
    up = np.bincount(pos[(u["Action"] == "up").values], minlength=n)
    down = np.bincount(pos[(u["Action"] == "down").values], minlength=n)
    cur, prior = u["currentPriceTarget"].astype(float).values, u["priorPriceTarget"].astype(float).values
    ok = (cur > 0) & (prior > 0) & (cur != prior)
    pt_sum = np.bincount(pos[ok], weights=np.log(cur[ok] / prior[ok]), minlength=n)
    pt_cnt = np.bincount(pos[ok], minlength=n)
    roll = lambda a: pd.Series(a, index=index).rolling(window, min_periods=1).sum()
    out = pd.DataFrame(index=index)
    out["analyst_net"] = roll(up) - roll(down)
    out["pt_change"] = (roll(pt_sum) / roll(pt_cnt).replace(0, np.nan)).fillna(0.0)
    return out


def _num(s):
    """'+1,234' / '46.35%' / '-' 같은 네이버 표기 → 숫자 (값 없음은 NaN)."""
    try:
        return float(str(s).replace(",", "").replace("%", "").replace("+", ""))
    except ValueError:
        return np.nan


def load_investor_flows(code, start, pause=0.15):
    """네이버 증권의 일별 외국인·기관 순매수량. 디스크에 캐시하고 새로운 날짜만 추가로 받는다."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"flows_{code}.pkl"
    cached = pd.read_pickle(path) if path.exists() else pd.DataFrame()
    have_until = cached.index.max() if len(cached) else None
    need_from = cached.index.min() if len(cached) else None

    rows, bizdate = [], None
    for _ in range(80):                                    # 60일 × 80 = 약 19년이 상한
        url = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60"
        if bizdate:
            url += f"&bizdate={bizdate}"
        data = requests.get(url, headers=UA, timeout=10).json()
        if not data:
            break
        for d in data:
            rows.append({"date": pd.Timestamp(d["bizdate"]), "frgn": _num(d["foreignerPureBuyQuant"]),
                         "inst": _num(d["organPureBuyQuant"]), "hold": _num(d["foreignerHoldRatio"]),
                         "volume": _num(d["accumulatedTradingVolume"])})
        oldest = rows[-1]["date"]
        # 캐시와 이어졌고 캐시가 이미 과거를 충분히 덮으면 중단
        if have_until is not None and oldest <= have_until and need_from <= pd.Timestamp(start):
            break
        if oldest <= pd.Timestamp(start):
            break
        bizdate = oldest.strftime("%Y%m%d")
        time.sleep(pause)

    new = pd.DataFrame(rows).drop_duplicates("date").set_index("date") if rows else pd.DataFrame()
    df = pd.concat([cached, new]) if len(cached) else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df):
        df.to_pickle(path)
    return df


def flow_features(ticker, index, raw=None):
    """raw: 미리 받아둔 load_investor_flows 결과."""
    code = krx_code(ticker)
    if not code:
        return None
    try:
        f = load_investor_flows(code, index[0]) if raw is None else raw
    except Exception:
        return None
    if f is None or f.empty:
        return None
    f = f.reindex(index)
    f[["frgn", "inst"]] = f[["frgn", "inst"]].fillna(0.0)     # '-'(거래정지 등) 날은 순매수 0
    f["hold"] = f["hold"].ffill()
    vol = f["volume"].replace(0, np.nan)
    out = pd.DataFrame(index=index)
    for k in (5, 21):      # 순매수량 ÷ 거래량: 종목 크기와 무관한 '매수 강도'
        out[f"frgn_flow_{k}"] = f["frgn"].rolling(k).sum() / vol.rolling(k).sum()
        out[f"inst_flow_{k}"] = f["inst"].rolling(k).sum() / vol.rolling(k).sum()
    out["frgn_hold_chg"] = f["hold"] - f["hold"].shift(21)
    return out


def build_alt_features(ticker, index):
    """방향 모델에 넣을 대체 데이터 특징들 + 각 데이터 출처의 사용 여부."""
    parts, status = [], {}
    sources = [("실적 서프라이즈", earnings_features), ("애널리스트 변경", analyst_features),
               ("외국인·기관 수급", flow_features)]
    code = krx_code(ticker)
    if code:
        if not kr_data.get_key("DART_API_KEY"):
            status["DART 공시"] = "키 없음"
        else:
            sources.append(("DART 공시", lambda t, idx: kr_data.dart_features(code, idx)))
    for name, fn in sources:
        try:
            df = fn(ticker, index)
        except Exception as e:
            df = None
            status[name] = f"오류: {e}"[:80]
        if df is not None and df.notna().any().any():
            parts.append(df)
            first = df.dropna().index.min()
            status[name] = f"사용 ({first.date() if first is not pd.NaT else '?'} 부터)"
        elif name not in status:
            status[name] = "데이터 없음"
    return (pd.concat(parts, axis=1) if parts else None), status


# ---------------- 현재 값만 있는 스냅샷 ----------------
def options_snapshot(ticker, spot, r=0.04, q=0.0):
    """풋/콜 비율(거래량·미결제약정)과 IV 스큐. 옵션이 없으면 None."""
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
    except Exception:
        return None
    if not expiries:
        return None
    today = date.today()
    exp_days = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in expiries]
    near = [(e, d) for e, d in exp_days if 5 <= d <= 70]
    if not near:
        return None
    cv = pv = coi = poi = 0.0
    chains = {}
    for e, d in near:
        ch = t.option_chain(e)
        chains[e] = (ch, d)
        cv += ch.calls["volume"].fillna(0).sum(); pv += ch.puts["volume"].fillna(0).sum()
        coi += ch.calls["openInterest"].fillna(0).sum(); poi += ch.puts["openInterest"].fillna(0).sum()

    e30, d30 = min(((e, d) for e, d in near), key=lambda x: abs(x[1] - 30))
    ch, _ = chains[e30]
    T = d30 / 365

    def iv_at(df, typ, k_target):
        row = df.loc[(df["strike"] - k_target).abs().idxmin()]
        bid, ask, last = row.get("bid", 0) or 0, row.get("ask", 0) or 0, row.get("lastPrice", 0) or 0
        px = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        iv = RiskNeutralPricingEngine(spot, float(row["strike"]), T, r, typ, q).implied_volatility(px) if px > 0 else np.nan
        if not np.isfinite(iv) and (row.get("impliedVolatility") or 0) > 0.01:
            iv = float(row["impliedVolatility"])
        return iv, float(row["strike"])

    put90, kp = iv_at(ch.puts, "put", 0.9 * spot)
    call110, kc = iv_at(ch.calls, "call", 1.1 * spot)
    atm, _ = iv_at(ch.calls, "call", spot)
    return {"pc_volume": pv / cv if cv > 0 else np.nan, "pc_oi": poi / coi if coi > 0 else np.nan,
            "put_volume": pv, "call_volume": cv, "expiry": e30, "days": d30,
            "iv_put90": put90, "iv_call110": call110, "iv_atm": atm, "k_put": kp, "k_call": kc,
            "skew": put90 - call110 if np.isfinite(put90) and np.isfinite(call110) else np.nan}


def estimate_revisions(ticker):
    """EPS 추정치의 최근 상·하향 (현재 시점 값만 제공됨). 없으면 None."""
    try:
        tr, rv = yf.Ticker(ticker).eps_trend, yf.Ticker(ticker).eps_revisions
    except Exception:
        return None
    if tr is None or tr.empty:
        return None
    out = pd.DataFrame(index=tr.index)
    out["현재 추정 EPS"] = tr["current"]
    for col, lab in (("30daysAgo", "30일 전 대비"), ("90daysAgo", "90일 전 대비")):
        if col in tr:
            out[lab] = tr["current"] / tr[col] - 1
    if rv is not None and not rv.empty:
        out["30일 상향 건수"] = rv.get("upLast30days")
        out["30일 하향 건수"] = rv.get("downLast30days")
    out.index = out.index.map({"0q": "이번 분기", "+1q": "다음 분기", "0y": "올해", "+1y": "내년"}.get).fillna(out.index.to_series())
    return out


NAVER_FIN_ROWS = ["매출액", "영업이익", "당기순이익", "영업이익률", "EPS", "PER", "PBR", "ROE", "부채비율", "주당배당금"]


def naver_financials(code, period="quarter"):
    """네이버 증권 재무 요약 (최근 실적 + 컨센서스 추정치). period: 'quarter' | 'annual'. 현재 시점 값만 제공."""
    d = requests.get(f"https://m.stock.naver.com/api/stock/{code}/finance/{period}", headers=UA, timeout=10).json()
    fi = d.get("financeInfo") or {}
    titles = fi.get("trTitleList") or []
    if not titles:
        return None
    cols = {t["key"]: t["title"].rstrip(".") + (" (E)" if t.get("isConsensus") == "Y" else "") for t in titles}
    data = {r["title"]: {cols[k]: _num(v.get("value")) for k, v in r["columns"].items() if k in cols}
            for r in fi.get("rowList", []) if r["title"] in NAVER_FIN_ROWS}
    df = pd.DataFrame(data).T
    return df.reindex(index=[r for r in NAVER_FIN_ROWS if r in df.index], columns=list(cols.values()))


# 금융 뉴스용 간단 감성 사전 (영어·한국어). 단어가 나오면 +1/−1, 제목 점수 = (긍정−부정)/(긍정+부정).
POS_EN = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "record", "upgrade",
          "upgraded", "raise", "raises", "raised", "strong", "growth", "outperform", "bullish", "gain", "gains", "top",
          "tops", "boost", "boosts", "rebound", "rebounds", "climb", "climbs", "high", "buy", "profit", "wins", "win"}
NEG_EN = {"miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump", "slumps", "downgrade",
          "downgraded", "cut", "cuts", "weak", "lawsuit", "probe", "recall", "bearish", "decline", "declines", "loss",
          "losses", "warn", "warns", "warning", "layoffs", "tumble", "tumbles", "sink", "sinks", "low", "sell",
          "selloff", "concern", "concerns", "risk", "fears", "crash", "slides", "slide"}
POS_KO = ["상승", "급등", "호실적", "최대", "상향", "매수", "수주", "성장", "개선", "돌파", "신고가", "강세", "반등", "흑자",
          "호재", "기대", "순매수", "서프라이즈", "확대"]
NEG_KO = ["하락", "급락", "부진", "하향", "매도", "적자", "감소", "우려", "악화", "쇼크", "소송", "리스크", "신저가", "약세",
          "악재", "순매도", "둔화", "축소", "폭락", "경고"]


def headline_score(title):
    words = re.findall(r"[a-z]+", title.lower())
    p = sum(w in POS_EN for w in words) + sum(title.count(k) for k in POS_KO)
    n = sum(w in NEG_EN for w in words) + sum(title.count(k) for k in NEG_KO)
    return (p - n) / (p + n) if p + n else 0.0


def news_sentiment(ticker, max_items=60):
    """최근 헤드라인과 감성 점수. 한국 종목은 네이버, 그 외는 Google News RSS(개인용 피드)."""
    items = []
    code = krx_code(ticker)
    try:
        if code:
            data = []
            for page in (1, 2, 3):
                data += requests.get(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=20&page={page}",
                                     headers=UA, timeout=10).json()
            for grp in data:
                for it in grp.get("items", []):
                    items.append({"time": datetime.strptime(it["datetime"], "%Y%m%d%H%M"),
                                  "title": it.get("titleFull") or it["title"], "source": it.get("officeName", ""),
                                  "link": it.get("mobileNewsUrl", "")})
        else:
            try:
                hit = [x for x in yf.Search(ticker, max_results=3).quotes if x.get("symbol") == ticker]
                name = (hit[0].get("shortname") or hit[0].get("longname")) if hit else ticker
                is_etf = bool(hit) and hit[0].get("quoteType") == "ETF"
            except Exception:
                name, is_etf = ticker, False
            # 회사명 앞부분으로 검색 (예: 'Apple Inc.' → 'Apple'), 최근 2주, 옵션 시세 페이지 제외
            short = re.sub(r"\b(Inc|Corp|Corporation|Co|Ltd|plc|Class [A-Z]|ETF.*|Trust)\b\.?", "", name).strip(" ,.")
            alt_q = f'"{ticker} ETF"' if is_etf else f"{ticker} stock"     # 'SPY' 같은 일반 단어 오검색 방지
            q = requests.utils.quote(f'"{short}" OR {alt_q} when:14d')
            xml = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                               headers=UA, timeout=10).text
            for it in ET.fromstring(xml).iter("item"):
                title = it.findtext("title") or ""
                if re.search(r"[A-Z]{1,5}\d{6}[CP]\d{8}", title):      # 옵션 계약 시세 페이지
                    continue
                src = it.findtext("source") or ""
                if src and title.endswith(" - " + src):
                    title = title[: -len(src) - 3]
                items.append({"time": parsedate_to_datetime(it.findtext("pubDate")).replace(tzinfo=None),
                              "title": title, "source": src, "link": it.findtext("link") or ""})
    except Exception:
        return None
    if not items:
        return None
    df = pd.DataFrame(items).sort_values("time", ascending=False).head(max_items).reset_index(drop=True)
    df["score"] = df["title"].map(headline_score)
    return df
