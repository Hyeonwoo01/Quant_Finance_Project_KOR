"""코스피 대형주 종목 고르기: 여러 종목을 서로 비교해 '앞으로 상대적으로 더 오를' 종목을 고른다.

한 종목의 방향(오를까/내릴까)은 거의 예측이 안 됐다. 대신 같은 날 여러 종목을 비교하면
- 표본이 수백 배 많아지고 (날짜 × 종목)
- 시장 전체 등락(예측 불가)이 상쇄되며
- 모멘텀·수급·자사주 같은 신호가 원래 '다른 종목 대비' 효과로 알려져 있다.

종목 범위 (생존 편향 완화): 현재 상장된 코스피 보통주 '전체'의 주가를 받고, 각 과거 시점에는 '그때의
63일 평균 거래대금' 상위 200개만 쓴다. '지금 큰 종목'으로 과거를 고르는 편향은 사라지고, 남는 건
그 사이 상장폐지된 종목이 빠지는 편향뿐이다 (무료 데이터로는 폐지 종목 주가를 구할 수 없음).
수급·실적·공시·재무는 '한 번이라도 200위 안에 들었던 종목'만 받는다.

명령줄:  python universe.py update     (처음엔 수십 분, 이후엔 새 데이터만)
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import alt_data as ad
import kr_data
import repro
from kr_costs import limit_hits, sell_tax_rate
from quant_engine import deflated_sharpe, fit_vol_model, hac_tstat, max_drawdown, ols, overlap_tstat, var_es

UDIR = kr_data.CACHE / "universe"
UA = {"User-Agent": "Mozilla/5.0"}
YEARS = 6
TOP_N = 200

FEATURES = {
    "mom_12_1": ("12-1개월 모멘텀", +1), "mom_1m": ("1개월 수익률(단기 반전)", -1),
    "mom_1w": ("1주 수익률", -1), "vol_63": ("3개월 변동성(저변동성)", -1),
    "max_21": ("1개월 최대 일간 상승(로또 효과)", -1), "ma200_gap": ("200일선 이격도", +1),
    "liq": ("거래대금 규모", 0),
    "frgn_flow_21": ("외국인 순매수 강도(1개월)", +1), "inst_flow_21": ("기관 순매수 강도(1개월)", +1),
    "frgn_flow_5": ("외국인 순매수 강도(1주)", +1), "inst_flow_5": ("기관 순매수 강도(1주)", +1),
    "frgn_hold_chg": ("외국인 지분율 변화(1개월)", +1),
    "pead": ("실적 서프라이즈 × 감쇠", +1),
    "dart_return": ("자사주 매입·소각 공시(3개월)", +1), "dart_dilution": ("증자·CB·BW 공시(3개월)", -1),
    "dart_contract": ("공급계약 공시(3개월)", +1),
    "bm": ("장부가/시총 (PBR 역수, 가치)", +1), "ep": ("순이익/시총 (PER 역수, 가치)", +1),
    "roe": ("ROE (수익성)", +1),
}
MODEL_LABELS = {"composite": "고정 가중 합산 (학계 알려진 방향, 학습 없음)", "ridge": "릿지 회귀",
                "gbm": "그래디언트 부스팅", "momentum": "모멘텀 단일 (기준)", "lowvol": "저변동성 단일 (기준)",
                "value": "가치(B/M) 단일 (기준)", "auto": "⭐ 자동 선택"}
# 비교만 하는 단순 기준 전략 (특징 하나, 알려진 방향) — 자동 선택 후보에 넣지 않는다 (후보를 늘리면 1등 착시가 커짐)
BASELINES = {"lowvol": ("vol_63", -1), "value": ("bm", +1)}


# ---------------- 데이터 받기 ----------------
def fetch_candidates(n=None):
    """네이버 증권 코스피 시가총액 순위의 보통주 전체(또는 상위 n개). ETF·ETN·우선주 제외.

    shares = 현재 시가총액 ÷ 현재가 (가치 지표의 시가총액 계산용), cap_rank = 현재 시총 순위."""
    out, page = [], 1
    while (n is None or len(out) < n) and page <= 40:
        js = requests.get(f"https://m.stock.naver.com/api/stocks/marketValue/KOSPI?page={page}&pageSize=100",
                          headers=UA, timeout=10).json()
        if not js.get("stocks"):
            break
        for s in js["stocks"]:
            if (s["stockEndType"] == "stock" and s["itemCode"].endswith("0")
                    and not re.search(r"\d?우[A-Z]?$|우\(", s["stockName"])):
                cap, px = float(s.get("marketValueRaw") or 0), float(s.get("closePriceRaw") or 0)
                out.append((s["itemCode"], s["stockName"], cap / px if px > 0 else np.nan))
        page += 1
    df = pd.DataFrame(out[:n] if n else out, columns=["code", "name", "shares"])
    df["cap_rank"] = np.arange(1, len(df) + 1)
    return df


def point_in_time_universe(close_raw, volume, top_n=TOP_N, value=None):
    """각 날짜의 '그때' 63일 평균 거래대금 상위 top_n (1년 이상 거래 이력 필수). 미래 정보 없음.
    value: KRX의 그날 실제 거래대금(원). 있으면 그 값을, 없는 날·종목은 종가 × 거래량을 쓴다."""
    traded = close_raw * volume if value is None else value.combine_first(close_raw * volume)
    liq = np.log(traded.replace(0, np.nan).rolling(63, min_periods=40).mean())
    eligible = close_raw.notna() & close_raw.shift(252).notna() & liq.notna()
    return liq.where(eligible).rank(axis=1, ascending=False) <= top_n, liq


def _parallel(fn, items, workers, progress, label):
    res, done = {}, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            it = futs[f]
            try:
                res[it] = f.result()
            except Exception:
                res[it] = None
            done += 1
            if progress and (done % max(1, len(items) // 20) == 0 or done == len(items)):
                progress(done / len(items), f"{label} {done}/{len(items)}")
    return res


def update_data(years=YEARS, workers=4, progress=print, limit=None):
    """코스피 전 종목 주가 → 한 번이라도 200위 안에 든 종목만 수급·실적·공시·재무를 받아 캐시에 저장.
    종목별 캐시라 두 번째부터는 새 데이터만 받는다. limit: 시총 상위 limit개만 (빠른 점검용)."""
    UDIR.mkdir(parents=True, exist_ok=True)
    say = (lambda p, m: progress(p, m)) if progress is not print else (lambda p, m: print(f"[{p:4.0%}] {m}", flush=True))
    say(0, "코스피 전 종목 목록" if limit is None else f"코스피 시총 상위 {limit}종목 목록 (빠른 점검)")
    cands = fetch_candidates(limit)
    cands.to_pickle(UDIR / "candidates.pkl")
    tickers = [c + ".KS" for c in cands["code"]]

    say(0, f"주가 {len(tickers)}종목 (한 번에)")
    px = yf.download(tickers, period=f"{years}y", auto_adjust=False, progress=False, threads=True)
    ren = lambda df: df.rename(columns=lambda t: t[:6])
    close, close_raw, volume = ren(px["Adj Close"]), ren(px["Close"]), ren(px["Volume"])
    idx = close.index.tz_localize(None) if close.index.tz is not None else close.index
    for df in (close, close_raw, volume):
        df.index = idx
    # 수익률은 배당·분할 조정 가격, 거래대금·시가총액은 실제 가격(분할만 조정)으로 계산
    pd.to_pickle({"close": close, "close_raw": close_raw, "volume": volume}, UDIR / "prices.pkl")
    start = close.index[0]

    univ, _ = point_in_time_universe(close_raw, volume)
    ever = set(univ.columns[univ.any()])
    codes = sorted((ever | set(cands.loc[cands["cap_rank"] <= 300, "code"])) & set(close.columns))
    say(0, f"한 번이라도 200위 안에 든 종목 {len(ever)}개 (+현재 시총 상위 300) → 상세 데이터 {len(codes)}종목")
    _parallel(lambda c: ad.load_investor_flows(c, start, pause=0.1), codes, workers,
              lambda p, m: say(p, m), "외국인·기관 수급")

    def earn(c):
        try:
            return yf.Ticker(c + ".KS").get_earnings_dates(limit=40)
        except Exception:
            return None
    earnings = _parallel(earn, codes, workers, lambda p, m: say(p, m), "실적 서프라이즈")
    pd.to_pickle(earnings, UDIR / "earnings.pkl")

    if kr_data.get_key("DART_API_KEY"):
        kr_data.dart_corp_code("005930")       # 고유번호 목록을 먼저 한 번만 받음
        for kind in ("B", "I"):                 # 주요사항보고(자사주·증자), 거래소공시(공급계약)
            _parallel(lambda c, k=kind: kr_data.dart_disclosures(c, start, kind=k), codes, workers,
                      lambda p, m: say(p, m), f"DART 공시({kind})")
    if kr_data.get_key("KRX_API_KEY"):      # 상장폐지 종목·그날의 시가총액 (빠른 점검은 최근 60거래일만 받아 해석만 확인)
        try:
            update_krx(close.index if limit is None else close.index[-60:], workers, say)
        except Exception as e:
            say(1, f"⚠️ KRX 일별매매정보 실패 (Yahoo 주가만으로 계산): {type(e).__name__}: {str(e)[:150]}")
    _write_meta(len(codes), len(cands), years)     # 재무제표가 실패해도 나머지 데이터로 쓸 수 있게 먼저 기록
    if kr_data.get_key("DART_API_KEY"):
        try:
            update_financials(codes, start, say)
        except Exception as e:
            say(1, f"⚠️ DART 재무제표 실패 (나머지는 저장됨, 나중에 'python universe.py financials'): {e}")
    say(1, "완료")


KRX_PATH = UDIR / "krx.pkl"
_PREFERRED = re.compile(r"\d?우[A-Z]?$|우\(")       # 우선주 이름 (fetch_candidates와 같은 규칙)


def update_krx(dates, workers=4, say=lambda p, m: print(f"[{p:4.0%}] {m}", flush=True)):
    """KRX 유가증권 일별매매정보를 날짜별로 받아(이미 받은 날짜는 캐시) 날짜 × 종목 표로 저장한다.

    그날 상장돼 있던 전 종목이라 나중에 상장폐지된 종목도 들어 있고, 그날의 시가총액·상장주식수·거래대금이 있다."""
    frames = _parallel(lambda d: kr_data.krx_stock_day(d, "STK"), list(dates), workers, say, "KRX 일별 전 종목")
    ok = {d: f for d, f in frames.items() if f is not None and len(f)}
    failed = sum(f is None for f in frames.values())
    if not ok:
        raise RuntimeError(f"KRX 일별매매정보를 하나도 받지 못함 (실패 {failed}일)")
    long = pd.concat([f.assign(date=pd.Timestamp(d)) for d, f in ok.items()], ignore_index=True)
    long = long.drop_duplicates(["date", "code"], keep="last")
    wide = {k: long.pivot(index="date", columns="code", values=k).sort_index()
            for k in ("close", "chg_pct", "volume", "value", "cap", "shares")}
    wide["names"] = long.sort_values("date").groupby("code")["name"].last()
    pd.to_pickle(wide, KRX_PATH)
    say(1, f"KRX 일별 전 종목: {len(ok)}일 · {long['code'].nunique()}종목 (받지 못한 날 {failed}일)")
    return wide


def load_krx(path=None):
    p = Path(path) if path else KRX_PATH
    return pd.read_pickle(p) if p.exists() else None


def merge_krx(close, close_raw, volume, krx):
    """Yahoo 주가(지금 상장된 종목) + KRX 일별매매정보(그날 상장돼 있던 전 종목) → 상장폐지 종목까지 포함한 표.

    - 지금 상장된 종목: 수익률은 Yahoo 수정주가(배당 포함) 그대로.
    - Yahoo에 없는 보통주(대부분 상장폐지): KRX 등락률(분할·증자를 반영한 기준가 대비)로 수정주가를 만들고, 마지막 거래일
      뒤로는 그 가격에 멈춘다 → 정리매매 마지막 가격까지의 손익을 반영하고 그 뒤는 현금(수익 0)으로 본다.
      원래 주가(close_raw)는 멈추지 않으므로 폐지 뒤엔 종목 범위에 들지 않는다.
    반환: (close, close_raw, volume, 더한 종목 이름).
    """
    idx = close.index
    names = krx["names"]
    extra = [c for c in krx["close"].columns
             if str(c).endswith("0") and not _PREFERRED.search(str(names.get(c, ""))) and c not in close.columns]
    kc = krx["close"].reindex(index=idx, columns=extra)
    growth = (1 + krx["chg_pct"].reindex(index=idx, columns=extra) / 100).where(kc.notna())
    adj = growth.cumprod().ffill()                     # 거래 없는 날·폐지 뒤 = 직전 가격 유지 (상장 전은 비어 있음)
    vol = krx["volume"].reindex(index=idx, columns=extra)
    return (pd.concat([close, adj], axis=1), pd.concat([close_raw, kc], axis=1), pd.concat([volume, vol], axis=1),
            names.reindex(extra))


def update_financials(codes=None, start=None, say=lambda p, m: print(f"[{p:4.0%}] {m}")):
    """DART 재무제표만 받기. 보고서별 캐시라 중간에 끊겨도 받은 만큼은 남는다."""
    if codes is None:
        pr = pd.read_pickle(UDIR / "prices.pkl")
        univ, _ = point_in_time_universe(pr.get("close_raw", pr["close"]), pr["volume"])
        cands = pd.read_pickle(UDIR / "candidates.pkl")
        codes = sorted((set(univ.columns[univ.any()]) | set(cands.loc[cands["cap_rank"] <= 300, "code"]))
                       & set(pr["close"].columns))
        start = pr["close"].index[0]
    # 첫 해의 TTM(최근 4분기) 계산을 위해 한 해 더 앞부터
    fin = kr_data.dart_financials(codes, range(start.year - 1, pd.Timestamp.today().year + 1), progress=say)
    fin.to_pickle(UDIR / "financials.pkl")
    meta = data_info() or {}
    meta["updated"] = pd.Timestamp.now().isoformat(timespec="minutes")     # 앱 캐시가 새로 계산되도록
    (UDIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _write_meta(n, n_all, years):
    (UDIR / "meta.json").write_text(json.dumps({"updated": pd.Timestamp.now().isoformat(timespec="minutes"),
                                                "n": n, "n_all": n_all, "years": years},
                                               ensure_ascii=False), encoding="utf-8")


def data_info():
    p = UDIR / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------- 특징 패널 ----------------
def build_panel(top_n=TOP_N, cap_limit=None, end=None):
    """(날짜, 종목) 특징 패널. 반환: dict(feat=특징별 wide DataFrame, close, universe=bool wide, names).

    cap_limit: 비교용. 주면 '현재 시총 상위 cap_limit개' 안에서만 고른다 (= 예전 방식, 생존 편향 있음).
    end: 이 날짜까지만 쓴다 (장중에 받은 '오늘' 가격을 빼고 계산할 때)."""
    cands = pd.read_pickle(UDIR / "candidates.pkl")
    pr = pd.read_pickle(UDIR / "prices.pkl")
    close, volume = pr["close"], pr["volume"]
    close_raw = pr.get("close_raw", close)
    if end is not None:
        close, close_raw, volume = close.loc[:end], close_raw.loc[:end], volume.loc[:end]
    keep = close.columns[close.notna().sum() > 252]      # 252일 이력이 없으면 어느 날에도 범위에 못 들어감 (point_in_time_universe)
    if cap_limit:
        keep = keep.intersection(cands.loc[cands["cap_rank"] <= cap_limit, "code"])
    close, close_raw, volume = close[keep], close_raw.reindex(columns=keep), volume.reindex(columns=keep)
    idx = close.index
    value = krx_cap = None
    n_delisted = 0
    names = cands.set_index("code")["name"]
    krx = load_krx()
    if krx is not None and not cap_limit:            # 상장폐지 종목까지 포함 + 그날의 실제 거래대금·시가총액
        close, close_raw, volume, extra_names = merge_krx(close, close_raw, volume, krx)
        keep = close.columns[close.notna().sum() > 252]
        close, close_raw, volume = close[keep], close_raw[keep], volume[keep]
        value = krx["value"].reindex(index=idx, columns=close.columns)
        krx_cap = krx["cap"].reindex(index=idx, columns=close.columns)
        names = pd.concat([names, extra_names[~extra_names.index.isin(names.index)]])
        added = [c for c in extra_names.index if c in keep]
        n_delisted = int(sum(close_raw[c].last_valid_index() < idx[-1] for c in added))    # 마지막 날 전에 거래가 끝남
    # 거래가 없는 날(거래정지 등)은 직전 가격 유지 → 들고 있던 종목의 수익률이 결측으로 조용히 빠지지 않게 한다.
    # 범위(universe)는 원래 주가(close_raw)로 정하므로 거래가 없는 날의 종목은 새로 고르지 않는다
    close = close.ffill()
    lc = np.log(close)
    r = lc.diff()
    universe, liq = point_in_time_universe(close_raw, volume, top_n, value)

    F = {
        "mom_12_1": lc.shift(21) - lc.shift(252),
        "mom_1m": lc - lc.shift(21),
        "mom_1w": lc - lc.shift(5),
        "vol_63": r.rolling(63).std(),
        "max_21": r.rolling(21).max(),
        "ma200_gap": lc - lc.rolling(200).mean(),
        "liq": liq,
    }

    def from_per_stock(fn):
        cols = {}
        for c in close.columns:
            try:
                df = fn(c)
            except Exception:
                df = None
            if df is not None:
                cols[c] = df
        return cols

    def flow(c):     # 캐시에 없는 종목은 건너뜀 (패널 만들 땐 네트워크를 쓰지 않는다)
        raw = _read_cache(f"flows_{c}.pkl")
        return ad.flow_features(c + ".KS", idx, raw=raw) if raw is not None else None
    flows = from_per_stock(flow)
    for k in ("frgn_flow_21", "inst_flow_21", "frgn_flow_5", "inst_flow_5", "frgn_hold_chg"):
        F[k] = pd.DataFrame({c: d[k] for c, d in flows.items()}, index=idx).reindex(columns=close.columns)

    earnings = pd.read_pickle(UDIR / "earnings.pkl") if (UDIR / "earnings.pkl").exists() else {}
    earn = from_per_stock(lambda c: ad.earnings_features(c + ".KS", idx, raw=earnings.get(c))
                          if earnings.get(c) is not None else None)
    F["pead"] = pd.DataFrame({c: d["pead"] for c, d in earn.items()}, index=idx).reindex(columns=close.columns)

    def dart(c):
        parts = [_read_cache(f"dart_{c}_{k}.pkl") for k in ("B", "I")]
        parts = [p for p in parts if p is not None]
        return kr_data.dart_features(c, idx, df=pd.concat(parts)) if parts else None
    dd = from_per_stock(dart)
    for k in ("dart_return", "dart_dilution", "dart_contract"):
        F[k] = pd.DataFrame({c: d[k] for c, d in dd.items()}, index=idx).reindex(columns=close.columns)

    fin = pd.read_pickle(UDIR / "financials.pkl") if (UDIR / "financials.pkl").exists() else None
    # 시가총액: KRX의 그날 실제 값(그날의 상장주식수 반영), 없는 날·종목만 '그날 주가 × 현재 상장주식수' 근사
    cap = close_raw * cands.set_index("code")["shares"].reindex(close.columns)
    if krx_cap is not None:
        cap = krx_cap.combine_first(cap)
    F.update(kr_data.value_features(fin, idx, close_raw, cands.set_index("code")["shares"], mcap=cap))

    coverage = {k: float(v.where(universe).notna().sum().sum() / max(universe.sum().sum(), 1)) for k, v in F.items()}
    return {"feat": F, "close": close, "universe": universe, "names": names,
            "coverage": coverage, "cap": cap, "close_raw": close_raw, "volume": volume,
            "krx": krx is not None and not cap_limit, "n_delisted": n_delisted}


def _read_cache(name):
    p = kr_data.CACHE / name
    return pd.read_pickle(p) if p.exists() else None


def cross_rank(df, universe):
    """특징용: 날짜별 종목 간 백분위 순위 − 0.5 (범위 밖·결측은 0 = 중간값). 극단값 영향이 없어진다."""
    x = df.where(universe)
    return (x.rank(axis=1, pct=True) - 0.5).fillna(0.0).where(universe)


def target_rank(fwd, universe):
    """정답용: 미래 수익률이 실제로 있는 종목끼리만 매긴 순위 − 0.5. 결측은 NaN으로 남겨 학습에서 뺀다.
    (특징처럼 중간값 0으로 채우면 '결과를 모르는 종목'이 '평균만큼 오른 종목'으로 학습된다.)"""
    return fwd.where(universe).rank(axis=1, pct=True) - 0.5


# ---------------- 모델 ----------------
def _stack(R, dates, universe, cols):
    """R: 특징별 순위 wide → (date, code) × 특징 long 행렬."""
    u = universe.loc[dates].stack()
    u = u[u]
    X = np.column_stack([R[c].loc[dates].stack().reindex(u.index).values for c in cols])
    return u.index, X


def composite_score(R, date, universe, cols):
    w = {c: FEATURES[c][1] for c in cols if FEATURES[c][1] != 0}
    s = sum(R[c].loc[date] * sgn for c, sgn in w.items()) / len(w)
    return s.where(universe.loc[date])


def walk_forward(panel, h=21, train_years=2, sample_every=5, refit_models=("ridge", "gbm"), progress=None,
                 exec_lag=1):
    """매 h일 리밸런싱. 각 시점 t에선 결과가 확정된(날짜 ≤ t−h−exec_lag) 데이터로만 학습·선택한다.

    exec_lag: t일 종가까지 보고 고른 뒤 실제로 사고파는 시점까지의 거래일 수.
              1(기본) = 다음 날 종가에 체결 → 성적은 t+1일 종가부터 h일 수익률로 매긴다. 0 = t일 종가에 체결(낙관적).
    학습 목표도 같은 '체결 이후 h일 수익률'의 순위다 — 판단일 t → t+1 종가처럼 실제로는 얻을 수 없는 하루를 배우지 않도록.
    """
    F, close, universe = panel["feat"], panel["close"], panel["universe"]
    cols = [c for c in FEATURES if panel["coverage"].get(c, 0) > 0.3]       # 데이터가 너무 빈 신호는 제외
    R = {c: cross_rank(F[c], universe) for c in cols}
    fwd_exec = np.log(close.shift(-h - exec_lag) / close.shift(-exec_lag))   # 실제 체결 이후 h일 수익률
    Y = target_rank(fwd_exec, universe)                                      # 학습 목표 = 채점 대상: 체결 이후 수익률 '순위'
    idx = close.index
    first = universe.any(axis=1).values.argmax() + 252 * train_years
    rebal = list(range(first, len(idx), h))
    if rebal and rebal[-1] != len(idx) - 1:
        rebal.append(len(idx) - 1)                                            # 마지막 = 오늘의 추천

    scores = {m: {} for m in ("composite", "momentum", *BASELINES, *refit_models)}
    for i, t in enumerate(rebal):
        d = idx[t]
        u = universe.loc[d]
        scores["composite"][d] = composite_score(R, d, universe, cols)
        scores["momentum"][d] = R["mom_12_1"].loc[d].where(u)
        for m, (c, sgn) in BASELINES.items():
            if c in R:
                scores[m][d] = sgn * R[c].loc[d].where(u)
        tr_dates = idx[universe.any(axis=1).values.argmax():t - h - exec_lag + 1][::sample_every]   # 결과가 t까지 확정된 날만
        if len(tr_dates) > 50 and refit_models:
            keys, X = _stack(R, tr_dates, universe, cols)
            y = Y.stack().reindex(keys).values
            ok = ~np.isnan(y)
            Xn = np.column_stack([R[c].loc[d].values for c in cols])
            live = u.values
            if "ridge" in refit_models:
                m = Ridge(alpha=100.0).fit(X[ok], y[ok])
                scores["ridge"][d] = pd.Series(np.where(live, m.predict(np.nan_to_num(Xn)), np.nan), index=u.index)
            if "gbm" in refit_models:
                m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                  min_samples_leaf=500, l2_regularization=1.0,
                                                  random_state=0).fit(X[ok], y[ok])
                scores["gbm"][d] = pd.Series(np.where(live, m.predict(np.nan_to_num(Xn)), np.nan), index=u.index)
        if progress:
            progress((i + 1) / len(rebal), f"리밸런싱 {i + 1}/{len(rebal)}")

    rdates = [idx[t] for t in rebal]
    # 모든 모델을 같은 리밸런싱 날짜에 맞춤 (학습 데이터가 부족했던 초기엔 점수 없음 = 전체 보유로 취급)
    scores = {m: pd.DataFrame(v).T.reindex(index=rdates, columns=universe.columns) for m, v in scores.items() if v}
    fwd_r = fwd_exec.loc[rdates]
    ic = {m: s.apply(lambda row: _ic(row, fwd_r.loc[row.name]), axis=1) for m, s in scores.items()}

    # 자동 선택: t에서 이미 결과가 확정된 과거 리밸런싱(t−h−exec_lag 이전)의 IC 평균(최근 12회)이 가장 높은 모델.
    # 이력이 6회 미만이면 학습 없는 고정 가중 모델, 모두 0 이하이면 '고르지 않음'(전체 동일가중 = 점수 없음).
    auto, chosen = {}, {}
    for j, d in enumerate(rdates):
        past = [rdates[k] for k in range(j) if idx.get_loc(rdates[k]) <= idx.get_loc(d) - h - exec_lag][-12:]
        if len(past) < 6:
            pick = "composite"
        else:
            # 학습 모델은 초기엔 점수가 없을 수 있음 → 과거 6회 이상 채점된 모델만 후보
            means = {m: ic[m].reindex(past).mean() for m in scores
                     if m not in BASELINES and ic[m].reindex(past).notna().sum() >= 6}
            pick = max(means, key=means.get) if means else "composite"
            pick = pick if means.get(pick, 1) > 0 else None
        chosen[d] = pick or "none"
        auto[d] = scores[pick].loc[d] if pick else pd.Series(np.nan, index=universe.columns)
    scores["auto"] = pd.DataFrame(auto).T
    ic["auto"] = scores["auto"].apply(lambda row: _ic(row, fwd_r.loc[row.name]), axis=1)
    return {"scores": scores, "ic": ic, "chosen": pd.Series(chosen), "rebal": rdates, "fwd": fwd_r,
            "cols": cols, "R": R, "h": h, "universe": universe, "exec_lag": exec_lag,
            "blocked": _blocked_orders(panel, rdates, exec_lag),
            "vol": F["vol_63"].reindex(rdates) if "vol_63" in F else None}           # 위험 반영 비중용 (그날까지의 63일 변동성)


def _blocked_orders(panel, rdates, exec_lag):
    """리밸런싱(판단일)별로 체결일에 팔 수 없는/살 수 없는 종목 표.

    체결일 종가가 하한가 근처면 팔 수 없고 상한가 근처면 살 수 없으며, 거래량이 0인 날(거래정지)은 둘 다 안 된다.
    원래 주가(close_raw)가 없는 패널(합성 데이터 등)이면 None = 제약 없음."""
    raw = panel.get("close_raw")
    if raw is None:
        return None
    idx, cols = panel["close"].index, panel["universe"].columns
    raw = raw.reindex(index=idx, columns=cols)
    down, up = limit_hits(raw / raw.shift(1) - 1)
    vol = panel.get("volume")
    halt = (vol.reindex(index=idx, columns=cols) == 0) & raw.notna() if vol is not None else False
    ex = idx[np.minimum(idx.get_indexer(rdates) + exec_lag, len(idx) - 1)]
    return {"sell": (down | halt).loc[ex].set_axis(rdates), "buy": (up | halt).loc[ex].set_axis(rdates)}


def _ic(score, fwd):
    ok = score.notna() & fwd.notna()
    if ok.sum() < 20 or score[ok].nunique() < 2:
        return np.nan
    return spearmanr(score[ok], fwd[ok])[0]


# ---------------- 평가 ----------------
def evaluate(wf, cost_bps=15.0, q=5, sell_tax=False, weighting="equal", models=None):
    """각 모델: 상위 1/q 포트폴리오 vs 전체 동일가중, IC, 회전율·비용 반영.

    cost_bps: 사고팔 때 각각 내는 수수료·체결 비용. sell_tax=True면 교체로 판 비중에 그날의 코스피 증권거래세를 더 뺀다.
    weighting: 'equal'(동일 비중) 또는 'inv_vol'(63일 변동성 역수 비중, 종목당 동일 비중의 2배 상한 — 위험을 비중에 반영).
    회전율 = ½ Σ|비중 변화| (판 비중 = 산 비중). models: 계산할 모델만 (기본: 전부).
    """
    h, fwd, univ = wf["h"], wf["fwd"], wf["universe"]
    rows, curves = {}, {}
    known = fwd.notna().sum(axis=1) > 20                 # 결과가 나온 리밸런싱만 채점
    dates = [d for d in wf["rebal"] if known.loc[d]]
    ret = np.exp(fwd.loc[dates]) - 1
    uni = ret.where(univ.loc[dates]).mean(axis=1)
    ix = np.minimum(univ.index.get_indexer(dates) + wf.get("exec_lag", 0), len(univ.index) - 1)
    exec_dates = univ.index[ix]                                              # 실제로 사고파는 날 (세율은 이날 기준)
    for m, S in wf["scores"].items():
        if models is not None and m not in models:
            continue
        hold, n_blocked = _holdings(wf, m, q)
        top_r, bot_r, turn, w_prev = [], [], [], None
        for d in dates:
            w = _weights(hold[d], d, wf, weighting)
            r = ret.loc[d, w.index]
            ok = r.notna()                                                   # (결측이면 있는 종목끼리 비중을 다시 맞춤)
            top_r.append(float((r[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else 0.0)   # 아무것도 못 샀으면 현금
            s = S.loc[d].dropna()
            bot_r.append(uni.loc[d] if len(s) < 20 else ret.loc[d, list(s.nsmallest(len(s) // q).index)].mean())
            turn.append(1.0 if w_prev is None else 0.5 * float(w.sub(w_prev, fill_value=0.0).abs().sum()))
            w_prev = w
        top = pd.Series(top_r, index=dates)
        turn_s = pd.Series(turn, index=dates)
        net = top - turn_s * 2 * cost_bps * 1e-4                            # 판 만큼 사고 → 양방향 비용
        if sell_tax:                                                         # 첫 매수 때는 판 것이 없음
            net -= turn_s.where(turn_s.index != dates[0], 0.0) * sell_tax_rate(exec_dates, "KOSPI").values
        ex = net - uni
        icm = wf["ic"][m].loc[dates].dropna()
        per_year = 252 / h
        rows[m] = {"모델": MODEL_LABELS[m], "평균 IC": icm.mean(),
                   "IC t값": hac_tstat(icm.values),                     # 자기상관 보정 (Newey-West)
                   "IC>0 비율": (icm > 0).mean(), "상위20% 초과수익(연, 비용후)": ex.mean() * per_year,
                   "초과수익 t값": hac_tstat(ex.values),
                   "상위−하위 스프레드(연)": (top - pd.Series(bot_r, index=dates)).mean() * per_year,
                   "최대 낙폭(비용후)": max_drawdown((1 + net).cumprod()),
                   "평균 회전율": np.mean(turn[1:]) if len(turn) > 1 else np.nan, "리밸런싱 수": len(dates),
                   "체결 막힌 주문": n_blocked}
        curves[m] = pd.DataFrame({"상위 20% (비용 후)": (1 + net).cumprod(), "하위 20%": (1 + pd.Series(bot_r, index=dates)).cumprod()})
    curves["_universe"] = (1 + uni).cumprod()
    return pd.DataFrame(rows).T, curves


def _weights(members, d, wf, weighting="equal", cap_mult=2.0):
    """보유 종목 → 비중 (합 1). 'inv_vol': 그날까지의 63일 변동성 역수에 비례, 종목당 동일 비중의 cap_mult배 상한.
    변동성이 없는 종목은 나머지의 중앙값을 쓰고, 변동성 자료가 아예 없으면 동일 비중."""
    if not members:
        return pd.Series(dtype=float)
    eq = pd.Series(1.0 / len(members), index=list(members))
    vol = wf.get("vol")
    if weighting != "inv_vol" or vol is None:
        return eq
    v = vol.loc[d].reindex(eq.index)
    inv = (1.0 / v).where(v > 0)
    if inv.notna().sum() == 0:
        return eq
    inv = inv.fillna(inv.median())
    cap = cap_mult / len(inv)                            # cap_mult ≥ 1이면 항상 가능 (상한 × 종목 수 ≥ 1)
    w, fixed = inv / inv.sum(), pd.Series(False, index=inv.index)
    while ((w > cap + 1e-12) & ~fixed).any():            # 상한에 닿은 종목은 상한으로 묶고, 남은 몫을 나머지에 비례 배분
        fixed |= w > cap + 1e-12
        w[fixed] = cap
        w[~fixed] = inv[~fixed] / inv[~fixed].sum() * (1 - cap * fixed.sum())
    return w


def cost_sensitivity(wf, costs=(0.0, 5.0, 10.0, 20.0, 40.0), sell_tax=False, model="auto", q=5):
    """수수료·체결 비용(슬리피지 포함, bp, 매수·매도 각각)을 바꿔 다시 계산한 초과수익 — 신호가 비용에 얼마나 약한지.
    반환: (비용별 표, 손익분기 비용 bp = 초과수익이 0이 되는 비용; 비용 0에서도 0 이하면 0)."""
    rows = {}
    for c in costs:
        t = evaluate(wf, c, q, sell_tax=sell_tax, models=[model])[0].loc[model]
        rows[f"{c:g}bp"] = {"초과수익(연, 비용후)": t["상위20% 초과수익(연, 비용후)"], "t값": t["초과수익 t값"],
                            "최대 낙폭(비용후)": t["최대 낙폭(비용후)"]}
    tab = pd.DataFrame(rows).T
    e0, e1 = tab.iloc[0, 0], tab.iloc[-1, 0]
    slope = (e0 - e1) / (costs[-1] - costs[0]) if costs[-1] > costs[0] else np.nan        # 비용 1bp당 줄어드는 연 초과수익
    breakeven = float(max(e0, 0.0) / slope) if np.isfinite(slope) and slope > 0 else np.nan
    return tab, breakeven


RECENT_DAYS = 252          # '최근 1년(이례적 구간)을 뺀' 점검에서 빼는 거래일 수 (EXPERIMENTS.md 사전 등록)


def drop_recent(wf, curves, model="auto", days=RECENT_DAYS):
    """마지막 days거래일이 없었다고 보고 다시 잰 성적: 보유 기간(체결 지연 포함)이 그 전에 끝난 리밸런싱만 쓴다.
    워크포워드라 그 전의 점수는 이후 데이터와 무관하다 → 마지막 1년을 지우고 다시 돌린 것과 같다. 보고용."""
    idx = wf["universe"].index
    cut = len(idx) - 1 - days
    if cut <= 0:
        return None
    lag = wf.get("exec_lag", 0)
    ex = (_period_returns(curves[model]["상위 20% (비용 후)"]) - _period_returns(curves["_universe"])).dropna()
    keep = [d for d in ex.index if idx.get_loc(d) + wf["h"] + lag <= cut]
    if len(keep) < 3:
        return None
    x, ic = ex.reindex(keep), wf["ic"][model].reindex(keep).dropna()
    return {"끝": idx[cut], "리밸런싱 수": len(keep), "초과수익(연)": float(x.mean() * 252 / wf["h"]),
            "초과수익 t값": hac_tstat(x.values), "평균 IC": float(ic.mean()) if len(ic) else np.nan,
            "IC t값": hac_tstat(ic.values)}


def market_breakdown(curves, model="auto"):
    """리밸런싱 기간을 그 기간 종목 범위 평균 수익률의 3분위(하락·횡보·상승)로 나눠, 구간별 상위 20%(비용 후)의 초과수익.
    특정 시장에서만 벌었는지(예: 상승장에서만) 본다."""
    uni = _period_returns(curves["_universe"])
    top = _period_returns(curves[model]["상위 20% (비용 후)"])
    ex = (top - uni).dropna()
    uni = uni.reindex(ex.index)
    if len(ex) < 9:
        return None
    lab = pd.qcut(uni.rank(method="first"), 3, labels=["하락", "횡보", "상승"])
    g = pd.DataFrame({"구간": lab, "시장": uni, "초과": ex}).groupby("구간", observed=False)
    return pd.DataFrame({"기간 수": g.size(), "시장 평균(기간)": g["시장"].mean(), "초과수익 평균(기간)": g["초과"].mean(),
                         "초과 > 0 비율": g["초과"].apply(lambda x: (x > 0).mean())})


# ---------------- 알려진 요인으로 설명되는가 ----------------
# 같은 종목 범위에서 직접 만든 요인 (상위 20% − 하위 20%, 동일가중, 리밸런싱 날짜의 순위로 나눔)
FACTOR_PROXIES = {"mom_12_1": ("모멘텀 (12-1개월)", +1), "vol_63": ("저변동성", -1),
                  "liq": ("소형주 (거래대금 작음)", -1), "bm": ("가치 (PBR 낮음)", +1)}


def _period_returns(curve):
    """누적 곡선(시작=1 기준) → 기간 수익률."""
    return curve / curve.shift(1).fillna(1.0) - 1


def factor_returns(wf, dates, q=5):
    """리밸런싱 날짜별 요인 수익률 (그날 순위로 나눈 상위 1/q − 하위 1/q, 체결 기준 h일 수익률)."""
    ret, univ = np.exp(wf["fwd"]) - 1, wf["universe"]
    out = {}
    for c, (label, sign) in FACTOR_PROXIES.items():
        if c not in wf["R"]:
            continue
        vals = []
        for d in dates:
            s = wf["R"][c].loc[d].where(univ.loc[d]).dropna()
            s = s[ret.loc[d].reindex(s.index).notna()]
            k = len(s) // q
            vals.append(sign * (ret.loc[d, s.nlargest(k).index].mean() - ret.loc[d, s.nsmallest(k).index].mean())
                        if k >= 5 else np.nan)
        out[label] = vals
    return pd.DataFrame(out, index=dates)


def factor_exposure(wf, curves, model="auto", q=5):
    """상위 20%의 초과수익(비용 후 − 전체 평균)을 시장·알려진 요인에 회귀 → 요인으로 설명되지 않는 부분(알파).

    알파가 0 근처면 '새로운 신호'가 아니라 이미 알려진 효과(모멘텀·저변동성 등)에 기울어진 결과라는 뜻이다.
    표본은 리밸런싱 횟수(겹치지 않는 h일 수익률)라 작다 — t값을 꼭 함께 본다.
    """
    cv = curves[model]
    net, uni = _period_returns(cv["상위 20% (비용 후)"]), _period_returns(curves["_universe"])
    F = factor_returns(wf, list(cv.index), q)
    X = pd.concat([uni.rename("시장 (전체 평균)"), F], axis=1)
    o = ols(net - uni, X)
    per_year = 252 / wf["h"]
    tab = pd.DataFrame({"노출(계수)": o["coef"], "t값": o["t"]}).drop(index="알파")
    tab["요인 자체 성과(연)"] = X.mean() * per_year
    return {"alpha": o["coef"]["알파"] * per_year, "alpha_t": o["t"]["알파"], "table": tab, "r2": o["r2"], "n": o["n"]}


def selection_check(curves, h):
    """모델 비교표에서 1등을 고르면 생기는 착시 보정 (Deflated Sharpe): 모델별 기간 초과수익(비용 후 − 전체 평균)의 샤프로."""
    uni = _period_returns(curves["_universe"])
    ex = {m: _period_returns(cv["상위 20% (비용 후)"]) - uni for m, cv in curves.items() if m != "_universe"}
    sr = {m: x.mean() / x.std() if x.std() > 0 else np.nan for m, x in ex.items()}
    valid = {m: v for m, v in sr.items() if np.isfinite(v)}
    if not valid:
        return None
    best = max(valid, key=valid.get)
    res = deflated_sharpe(ex[best].values, list(valid.values()))
    if res is None:
        return None
    k = np.sqrt(252 / h)
    return {**res, "best": best, "sr_ann": res["sr"] * k, "sr0_ann": res["sr0"] * k}


# ---------------- 운용 규모와 위험 ----------------
def _holdings(wf, model="auto", q=5):
    """리밸런싱 날짜별 실제 보유 종목: 점수 상위 1/q, 점수가 없으면(자동 선택이 '고르지 않음') 전체 종목.
    체결일에 하한가·거래정지면 못 판 종목은 계속 보유하고, 상한가·거래정지면 못 산 종목은 빼고 산다 (wf['blocked']).
    반환: ({날짜: 종목 목록}, 막힌 주문 수). evaluate·capacity가 같은 규칙을 쓴다."""
    univ, blocked = wf["universe"], wf.get("blocked")
    known = wf["fwd"].notna().sum(axis=1) > 20
    out, prev, n_blocked = {}, None, 0
    for d in [d for d in wf["rebal"] if known.loc[d]]:
        s = wf["scores"][model].loc[d].dropna()
        if len(s) < 20:
            members = set(univ.columns[univ.loc[d]])
        else:
            members = set(s.nlargest(len(s) // q).index)
            if blocked is not None:
                held = prev or set()
                keep = {c for c in held - members if blocked["sell"].at[d, c]}
                skip = {c for c in members - held if blocked["buy"].at[d, c]}
                members, n_blocked = (members - skip) | keep, n_blocked + len(keep) + len(skip)
        out[d] = sorted(members)
        prev = members
    return out, n_blocked


def capacity(wf, panel, curves, model="auto", capitals=(1e7, 1e8, 1e9, 1e10, 1e11), impact=1.0, q=5):
    """운용 규모(원)별 시장 충격 비용 — 주문이 그 종목 하루 거래대금에서 차지하는 비율이 클수록 불리하게 체결된다.

    제곱근 법칙: 한 번 사거나 팔 때 추가 비용 ≈ impact × 하루 변동성 × √(주문 금액 / 하루 거래대금).
    리밸런싱마다 새로 사거나 판 종목만 비용을 낸다 (종목당 금액 = 자본 / 보유 종목 수, 거래대금은 그날 기준 63일 평균).
    반환: 자본별 평균 주문 비율, 추가 비용(연), 비용 반영 초과수익(연)과 t값 — 수수료·세금은 curves에 이미 반영된 것에 더한다.
    """
    hold, _ = _holdings(wf, model, q)
    dates = list(hold)
    adv = np.exp(panel["feat"]["liq"])
    vol = np.log(panel["close"]).diff().rolling(63, min_periods=40).std()
    cv = curves[model]["상위 20% (비용 후)"].reindex(dates)
    uni = curves["_universe"].reindex(dates)
    ex = (cv / cv.shift(1).fillna(1.0) - 1) - (uni / uni.shift(1).fillna(1.0) - 1)
    per_year = 252 / wf["h"]
    rows = {}
    for cap in capitals:
        cost, part, prev = [], [], None
        for d in dates:
            now = set(hold[d])
            buys, sells = now - (prev or set()), (prev or set()) - now
            c = 0.0
            for names, n in ((buys, len(now)), (sells, len(prev) if prev else 1)):
                for code in names:
                    a, v = adv.at[d, code], vol.at[d, code]
                    if not (np.isfinite(a) and np.isfinite(v) and a > 0):
                        continue
                    share = cap / n / a                          # 주문 금액 / 하루 거래대금
                    c += impact * v * np.sqrt(share) / n         # 포트폴리오 대비 비용
                    part.append(share)
            cost.append(c)
            prev = now
        net = ex - pd.Series(cost, index=dates)
        rows[f"{cap / 1e8:,.0f}억 원" if cap >= 1e8 else f"{cap / 1e4:,.0f}만 원"] = {"주문 / 하루 거래대금 (중앙값)": float(np.median(part)) if part else np.nan,
                                "시장 충격 비용(연)": float(np.mean(cost) * per_year),
                                "초과수익(연, 모든 비용 후)": float(net.mean() * per_year),
                                "t값": hac_tstat(net.values)}
    return pd.DataFrame(rows).T


def current_basket(wf, q=5):
    """지금(마지막 리밸런싱) 추천 상위 1/q 종목 — 앱에 보이는 순위와 같은 점수(자동 선택, 없으면 고정 가중)."""
    d = wf["rebal"][-1]
    s = wf["scores"]["auto"].loc[d].dropna()
    if wf["chosen"].loc[d] == "none" or s.empty:
        s = wf["scores"]["composite"].loc[d].dropna()
    return list(s.nlargest(max(len(s) // q, 1)).index)


def basket_risk(panel, codes, lookback=756, n_stress=5):
    """지금 고른 종목들(동일 비중, 매일 비중 유지)을 지난 lookback일 동안 그대로 들고 있었다면 — 위험 지표와 시장 급락일 손실.

    반환: dict(var95·es95·var99 (하루, 과거 실제), garch_var95 (지금 변동성 기준 내일), worst_day, worst_month, beta,
              stress = 전체 종목 평균이 가장 많이 빠진 날들의 시장·포트폴리오 수익률)."""
    r = np.exp(np.log(panel["close"]).diff().iloc[-lookback:]) - 1
    uni = panel["universe"].reindex(r.index).fillna(False)
    port = r[codes].mean(axis=1).dropna()
    mkt = r.where(uni).mean(axis=1).reindex(port.index)
    lp = np.log1p(port)
    v95, e95 = var_es(lp.values, 0.95)
    v99, _ = var_es(lp.values, 0.99)
    g = fit_vol_model(lp.values)
    sims = g.simulate(1, 20000, seed=0)[:, 0]
    m21 = np.expm1(lp.rolling(21).sum())
    b = ols(port, mkt.rename("시장").to_frame())
    worst = mkt.nsmallest(n_stress).index
    stress = pd.DataFrame({"전체 종목 평균": mkt[worst], "지금 고른 종목": port[worst]}).sort_index()
    stress.index = stress.index.strftime("%Y-%m-%d")
    stress.index.name = "날짜"
    return {"var95": v95, "es95": e95, "var99": v99, "garch_var95": var_es(sims, 0.95)[0], "model": g.name,
            "worst_day": (port.idxmin(), float(port.min())), "worst_month": (m21.idxmin(), float(m21.min())),
            "beta": float(b["coef"]["시장"]), "n": len(codes), "days": len(port), "stress": stress}


def current_picks(panel, wf, k=20):
    """오늘(마지막 리밸런싱) 자동 선택 모델의 점수 상·하위 종목 + 주요 신호 값(순위)."""
    d = wf["rebal"][-1]
    pick = wf["chosen"].loc[d]
    s = wf["scores"]["auto"].loc[d].dropna()
    if pick == "none" or s.empty:
        s = wf["scores"]["composite"].loc[d].dropna()
    tbl = pd.DataFrame({"종목명": panel["names"].reindex(s.index), "점수": s})
    for c in wf["cols"]:
        tbl[FEATURES[c][0]] = wf["R"][c].loc[d].reindex(s.index) + 0.5           # 0~1 백분위 (1 = 가장 높음)
    tbl = tbl.sort_values("점수", ascending=False)
    tbl.insert(0, "순위", range(1, len(tbl) + 1))
    return d, pick, tbl.head(k), tbl.tail(k).iloc[::-1]


# ---------------- 한국판 요인 (파마-프렌치 방식의 근사) ----------------
# 한국 시장 전용 무료 일간 요인 데이터가 없어, 같은 종목 범위(그때의 거래대금 상위 200)로 직접 만든다.
# 열 이름은 French 데이터와 같게 두어 quant_engine.timing_attribution에 그대로 넣는다 (투자 요인 CMA는 자산 데이터가 없어 뺌).
KR_FACTORS = {"Mkt-RF": "시장", "SMB": "소형주", "HML": "가치", "RMW": "수익성", "Mom": "모멘텀"}
KR_FACTOR_SORTS = {"SMB": ("cap", -1), "HML": ("bm", +1), "RMW": ("roe", +1), "Mom": ("mom_12_1", +1)}


def kr_factors(panel, rf=None, every=21, q=0.3):
    """한국판 일간 요인 수익률 (소수). 반환 열: Mkt-RF, SMB, HML, RMW, Mom, RF (데이터가 없는 요인은 빠짐 — 예: DART 키가 없으면 가치·수익성).

    시장: 그 전날 종목 범위의 시가총액 가중 수익률 − 무위험 수익률(rf: 날짜별 '하루' 수익률, 없으면 0).
    나머지: every거래일마다 그날 값으로 종목을 줄 세워 상위 q − 하위 q (동일 비중)를 다음 리밸런싱까지 들고 있는 수익률.
    소형주 = 시가총액 하위 − 상위, 가치 = 장부가/시총 상위 − 하위, 수익성 = ROE 상위 − 하위, 모멘텀 = 12-1개월 수익률 상위 − 하위.
    줄 세우는 날까지의 정보만 쓰고(재무는 공시 다음 거래일부터), 그다음 날부터의 수익률을 번다 → 미래 정보 없음.
    """
    close, univ, idx = panel["close"], panel["universe"], panel["close"].index
    r = close.pct_change(fill_method=None)
    cap = panel.get("cap")
    if cap is not None:
        w = cap.shift(1).where(univ.shift(1, fill_value=False))
        mkt = (w * r).sum(axis=1, min_count=1) / w.where(r.notna()).sum(axis=1, min_count=1)
    else:                                                     # 시가총액이 없으면 동일 비중
        mkt = r.where(univ.shift(1, fill_value=False)).mean(axis=1)
    rf = (rf.reindex(idx).ffill().fillna(0.0) if rf is not None else pd.Series(0.0, index=idx))
    out = pd.DataFrame({"Mkt-RF": mkt - rf}, index=idx)
    src = {"cap": cap, **panel["feat"]}
    first = int(univ.any(axis=1).values.argmax())
    rebal = list(range(first, len(idx) - 1, every))
    for name, (col, sign) in KR_FACTOR_SORTS.items():
        x = src.get(col)
        if x is None:
            continue
        f = pd.Series(np.nan, index=idx)
        for j, t in enumerate(rebal):
            end = rebal[j + 1] if j + 1 < len(rebal) else len(idx) - 1
            s = x.iloc[t].where(univ.iloc[t].values).dropna()
            k = int(len(s) * q)
            if k < 5:
                continue
            hi, lo = s.nlargest(k).index, s.nsmallest(k).index
            long, short = (hi, lo) if sign > 0 else (lo, hi)
            seg = r.iloc[t + 1:end + 1]
            f.iloc[t + 1:end + 1] = (seg[long].mean(axis=1) - seg[short].mean(axis=1)).values
        if f.notna().any():                                   # 데이터가 없는 요인은 열째로 뺀다 (회귀 전체가 비지 않도록)
            out[name] = f
    out["RF"] = rf
    return out.dropna(subset=["Mkt-RF"]).reindex(columns=[c for c in [*KR_FACTORS, "RF"] if c in out])


def cd91_daily(start, end=None):
    """CD 91일 금리(한국은행 ECOS, 연 %) → 날짜별 하루 무위험 수익률 (소수). 키가 없거나 실패하면 None."""
    try:
        s = kr_data.ecos_series("817Y002", "010502000", pd.Timestamp(start), pd.Timestamp(end or pd.Timestamp.today()))
    except Exception:
        return None
    return None if s is None or s.empty else s / 100.0 / 252


KR_FACTORS_PATH = Path(__file__).parent / "ranking" / "kr_factors.csv"


def load_kr_factors(path=KR_FACTORS_PATH):
    """매주 계산해 둔 한국판 일간 요인 (없으면 None)."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def kr_factors_stamp(path=KR_FACTORS_PATH):
    return Path(path).stat().st_mtime if Path(path).exists() else None


# ---------------- 매주 미리 계산 (GitHub Actions) ----------------
RANK_DIR = Path(__file__).parent / "ranking"     # 계산 결과만 저장한다 (원본 시세·수급·재무는 올리지 않음)
RESULTS_PATH = RANK_DIR / "results.pkl"
PICKS_PATH = RANK_DIR / "picks.csv"
WEEKLY_COST = 5.0                   # 앱 기본 '수수료·체결 비용(bp)'과 같게
WEEKLY_HOLDS = (63, 21)             # 앱의 '종목 교체 주기' 선택지
RECORD_HOLD, RECORD_LAG = 63, 1     # 실전 기록에 남기는 설정 (앱 기본값)
SCORE_HORIZONS = (21, 63)           # 기록한 추천을 채점하는 기간 (거래일)
KR_CLOSE_DONE = 15 * 60 + 30 + 40   # 한국 장 마감 15:30 + 여유 40분 (forecast_log와 같은 기준)
PICK_COLUMNS = ["date", "model", "n", "codes", "names",
                *[f"{k}_{h}" for h in SCORE_HORIZONS for k in ("basket", "market")], "version"]


def _optional(fn, *args, **kwargs):
    """부가 분석은 실패해도 나머지를 막지 않는다 (데이터 열이 빠진 경우 등) → None."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def results(panel, wf, cost_bps=WEEKLY_COST, sell_tax=False):
    """앱 '코스피 종목 고르기' 탭이 보여 주는 결과 묶음. 계산 결과만 들어 있고 원본 시세는 없다."""
    table, curves = evaluate(wf, cost_bps, sell_tax=sell_tax)
    d, pick, top, bottom = current_picks(panel, wf)
    rw = evaluate(wf, cost_bps, sell_tax=sell_tax, weighting="inv_vol", models=["auto"])[0]     # 위험 반영 비중
    ch = wf["chosen"]
    return {"table": table, "curves": curves, "ic": wf["ic"]["auto"], "chosen": wf["chosen"], "date": d,
            "pick": pick, "top": top, "bottom": bottom, "coverage": panel["coverage"], "cols": wf["cols"],
            "n_univ": int(panel["universe"].iloc[-1].sum()), "factors": factor_exposure(wf, curves),
            "selection": selection_check(curves, wf["h"]), "capacity": _optional(capacity, wf, panel, curves),
            "risk": _optional(lambda: basket_risk(panel, current_basket(wf))),
            "krx": bool(panel.get("krx")), "n_delisted": int(panel.get("n_delisted", 0)),
            "risk_weighted": rw.loc["auto"] if "auto" in rw.index else None,
            "cost_sens": _optional(cost_sensitivity, wf, sell_tax=sell_tax),
            "market": _optional(market_breakdown, curves),
            "ex_recent": _optional(drop_recent, wf, curves),
            "switches": int((ch != ch.shift()).iloc[1:].sum()) if len(ch) > 1 else 0}


def finished_end(index, now=None):
    """마지막 날이 '오늘'이고 한국 장이 아직 마감 전(여유 포함)이면 그 전 거래일, 아니면 마지막 날.
    장중에 수동으로 돌려도 움직이는 중인 가격으로 추천을 기록하지 않도록."""
    local = (now or pd.Timestamp.now(tz="UTC")).tz_convert("Asia/Seoul")
    if len(index) > 1 and index[-1].date() == local.date() and local.hour * 60 + local.minute < KR_CLOSE_DONE:
        return index[-2]
    return index[-1]


def read_picks(path=PICKS_PATH):
    """매주 미리 적어 둔 추천 기록 (없으면 빈 표)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.read_csv(path, dtype={"date": str, "model": str, "codes": str, "names": str, "version": str})


def record_picks(log, wf, panel, version=""):
    """이번 추천(앱에 보이는 순위의 상위 20%)을 기록에 추가. 같은 날짜가 이미 있으면 그대로 둔다 (미리 적어 둔 기록은 고치지 않음)."""
    d = wf["rebal"][-1]
    if (log["date"].astype(str) == d.date().isoformat()).any():
        return log
    codes = current_basket(wf)
    row = pd.DataFrame([{"date": d.date().isoformat(), "model": wf["chosen"].loc[d], "n": len(codes),
                         "codes": " ".join(codes), "names": ", ".join(panel["names"].reindex(codes).fillna("").astype(str)),
                         "version": version}]).reindex(columns=PICK_COLUMNS)
    return row if log.empty else pd.concat([log, row], ignore_index=True)


def score_picks(log, panel, lag=RECORD_LAG, horizons=SCORE_HORIZONS):
    """기록한 추천의 실제 성적: 기록 다음 거래일 종가에 사서 h거래일 뒤까지(비용 전),
    바구니(동일 비중) vs 그날 종목 범위 전체(동일 비중). 기간이 아직 안 끝난 기록은 비워 둔다."""
    close, univ, idx = panel["close"], panel["universe"], panel["close"].index
    out = log.copy()
    for col in (f"{k}_{h}" for h in horizons for k in ("basket", "market")):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for k, r in out.iterrows():
        i = idx.searchsorted(pd.Timestamp(r["date"]), side="right") - 1
        if i < 0:
            continue
        codes = [c for c in str(r["codes"]).split() if c in close.columns]
        members = univ.columns[univ.iloc[i].values]
        for h in horizons:
            if i + lag + h >= len(idx):
                continue
            g = close.iloc[i + lag + h] / close.iloc[i + lag] - 1
            out.at[k, f"basket_{h}"] = float(g.reindex(codes).mean())
            out.at[k, f"market_{h}"] = float(g.reindex(members).mean())
    return out


def picks_summary(log, horizons=SCORE_HORIZONS, every=5):
    """채점 기간별 채점 수·평균 초과수익·전체 평균보다 나은 비율·t값.
    기록이 매주(약 every거래일 간격)라 h거래일 성적이 서로 겹친다 → 겹침 보정 t값 (Hansen-Hodrick)."""
    rows = {}
    for h in horizons:
        ex = (pd.to_numeric(log[f"basket_{h}"], errors="coerce") - pd.to_numeric(log[f"market_{h}"], errors="coerce")).dropna()
        rows[f"{h}거래일"] = {"채점 수": float(len(ex)), "평균 초과수익": ex.mean() if len(ex) else np.nan,
                          "전체 평균보다 나은 비율": (ex > 0).mean() if len(ex) else np.nan,
                          "t값 (겹침 보정)": overlap_tstat(ex.values, int(np.ceil(h / every))) if len(ex) > 2 else np.nan}
    return pd.DataFrame(rows).T


def load_weekly(path=RESULTS_PATH):
    """GitHub Actions가 매주 계산해 둔 결과 (없으면 None)."""
    return pd.read_pickle(path) if Path(path).exists() else None


def weekly_stamp(path=RESULTS_PATH):
    """결과 파일이 바뀌면 달라지는 값 (앱 캐시 키)."""
    return Path(path).stat().st_mtime if Path(path).exists() else None


def weekly(update=True, panel=None, out_dir=RANK_DIR, now=None, refit_models=("ridge", "gbm"), progress=print, smoke=False,
           rf=None):
    """GitHub Actions가 매주 돌리는 일 (python universe.py weekly).

    ① 데이터 갱신 ② 앱 설정 조합(교체 주기 × 체결 시점 × 증권거래세)마다 탭에 보일 결과를 계산해 results.pkl로 저장
    ③ 이번 추천(앱 기본 설정)을 picks.csv에 미리 적고, 기간이 지난 추천을 채점. 원본 시세·수급·재무는 저장하지 않는다.
    ④ 한국판 일간 요인(시장·소형주·가치·수익성·모멘텀)을 kr_factors.csv로 저장 — 방향 예측·시장 국면의 수익 분해에 쓴다.
    panel: 테스트용 (주면 데이터를 받지 않고 그 패널로 계산). rf: 테스트용 무위험 수익률 (없으면 ECOS CD 91일).
    smoke: 코드가 바뀐 푸시에서 쓰는 빠른 점검 — 시총 상위 60종목·4년치로 전체 과정을 실제 데이터로 한 번 돌려 본다."""
    say = (lambda p, m: progress(p, m)) if progress is not print else (lambda p, m: print(f"[{p:4.0%}] {m}", flush=True))
    if panel is None:
        if update:
            update_data(years=4 if smoke else YEARS, progress=progress, limit=60 if smoke else None)
        prices = pd.read_pickle(UDIR / "prices.pkl")
        panel = build_panel(end=finished_end(prices["close"].index, now))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = os.environ.get("GITHUB_SHA", "")[:7] or repro.code_version()
    combos = [(h, lag) for h in WEEKLY_HOLDS for lag in (1, 0)]
    res, rec = {}, None
    for j, (h, lag) in enumerate(combos):
        say(j / len(combos), f"계산: 교체 주기 {h}거래일, 체결 {'다음 날' if lag else '그날'} 종가")
        wf = walk_forward(panel, h, refit_models=refit_models, exec_lag=lag)
        for tax in (True, False):
            res[(h, lag, tax)] = results(panel, wf, WEEKLY_COST, tax)
        if (h, lag) == (RECORD_HOLD, RECORD_LAG):
            rec = wf
    pd.to_pickle({"updated": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
                  "data_end": panel["close"].index[-1], "cost_bps": WEEKLY_COST, "results": res,
                  "meta": data_info(), "version": version}, out_dir / RESULTS_PATH.name)
    log = score_picks(record_picks(read_picks(out_dir / PICKS_PATH.name), rec, panel, version), panel)
    log.to_csv(out_dir / PICKS_PATH.name, index=False)
    idx = panel["close"].index
    try:                                      # 부가 결과 — 실패해도 위의 종목 고르기 결과는 그대로 커밋되도록
        kf = kr_factors(panel, rf=cd91_daily(idx[0], idx[-1]) if rf is None else rf)
        kf.round(7).to_csv(out_dir / KR_FACTORS_PATH.name, index_label="date")  # 계산 결과(요인 수익률)만 저장
        say(1, f"한국판 요인: {', '.join(KR_FACTORS[c] for c in kf.columns if c in KR_FACTORS)} · {len(kf)}일")
    except Exception as e:
        say(1, f"⚠️ 한국판 요인 계산 실패 (다른 결과는 저장됨): {type(e).__name__}: {str(e)[:200]}")
    say(1, f"완료: 설정 {len(res)}개, 데이터 {panel['close'].index[-1].date()}까지, 추천 기록 {len(log)}줄")
    return res, log


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    if cmd == "update":
        update_data()
    elif cmd == "financials":          # DART 재무제표만 (차단 등으로 실패했을 때)
        update_financials()
    elif cmd == "weekly":              # GitHub Actions (.github/workflows/ranking.yml). --smoke: 빠른 점검
        weekly(smoke="--smoke" in sys.argv)
    else:
        print("사용법: python universe.py update | financials | weekly")
    print(f"총 {time.time() - t0:.0f}초")
