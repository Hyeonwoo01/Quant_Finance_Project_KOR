"""앞으로의 예측 기록 (실전 검증): 매일 한국 장 마감 후 그날의 예측을 적어 두고, 결과가 나오면 채점한다.

백테스트는 결과를 이미 아는 과거로 시험하므로 가정·모델을 나중에 고를 여지가 남는다. 여기서는 예측을 먼저 기록하고
결과는 나중에 보므로 그럴 수가 없다 = 진짜 표본 밖 검증. 기록은 덮어쓰지 않고 추가만 한다.

기록은 GitHub Actions(.github/workflows/forecast-log.yml)가 평일마다 돌려 forecast_log/log.csv에 커밋한다.
기록할 종목은 forecast_log/tickers.txt (한 줄에 하나).

명령줄:  python forecast_log.py record   (오늘 예측을 기록에 추가)
         python forecast_log.py check    (저장하지 않고 몇 종목만 계산해 출력 — 데이터 연결 점검)
"""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import alt_data as ad
import direction as dr
import market_data as md
import regime as rg
import repro
from quant_engine import fit_vol_model, ols, overlap_tstat, var_es, var_exception_tests

LOG_DIR = Path(__file__).parent / "forecast_log"
LOG_PATH = LOG_DIR / "log.csv"
TICKERS_PATH = LOG_DIR / "tickers.txt"
H = 21                    # 모든 예측의 기간 (거래일, ≈ 1개월)
GARCH_YEARS = 3           # 앱 기본 '학습 데이터' 설정과 같게
# 시장 심리 지표 (앱 '시장 심리' 탭과 같은 값). 과거 기록을 무료로 구할 수 없어 지금 적어 두어야 나중에 검증할 수 있다.
SENTIMENT = {"pc_volume": "풋/콜 거래량 비율", "pc_oi": "풋/콜 미결제약정 비율", "skew": "옵션 스큐 (풋 90% − 콜 110% IV)",
             "eps_rev_30d": "올해 EPS 전망 30일 변화", "news_7d": "뉴스 감성 (최근 7일 평균)"}
COLUMNS = ["asof", "ticker", "close", "h", "p_up", "p_base", "model", "vol_garch", "lo95", "lo68", "hi68", "hi95",
           "var95_1d", "var99_1d", "garch_ok", "vol_model", "iv_atm", "iv_days", "vol_index", "p_high",
           *SENTIMENT, "news_n", "data_fp", "version", "logged_at"]


# 거래소별 (시간대, 장 마감 시각 + 여유 분). 그 전에 돌리면 오늘 봉은 아직 움직이는 중이라 쓰지 않는다.
_SESSIONS = {"KR": ("Asia/Seoul", 15 * 60 + 30 + 40), "US": ("America/New_York", 16 * 60 + 30)}


def drop_unfinished(close, ticker, now=None):
    """마지막 봉이 '오늘'이고 그 거래소가 아직 마감 전(여유 포함)이면 빼고 돌려준다.

    수동 실행이나 예약 시각이 장중에 걸려도, 끝나지 않은 가격으로 예측을 기록하지 않도록."""
    tz, cutoff = _SESSIONS["US" if md.is_us_ticker(ticker) else "KR"]          # ^KS11 같은 한국 지수는 한국 장
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(tz))
    if close.index[-1].date() == local.date() and local.hour * 60 + local.minute < cutoff:
        return close.iloc[:-1]
    return close


def tickers(path=TICKERS_PATH):
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [s.split("#")[0].strip() for s in lines if s.split("#")[0].strip()]


def read_log(path=LOG_PATH):
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(path, dtype={"ticker": str, "model": str, "version": str})
    log["asof"] = pd.to_datetime(log["asof"])
    return log


def predict(ticker, h=H, have=frozenset()):
    """앱 기본 설정과 같은 계산으로 오늘(마지막 종가 기준)의 예측 한 줄. 그 날짜가 이미 기록돼 있으면(휴장일 등) skip 표시만."""
    close = drop_unfinished(md.load_history(ticker, 10), ticker)
    asof, spot = close.index[-1], float(close.iloc[-1])
    row = {"asof": asof.date().isoformat(), "ticker": ticker, "close": spot, "h": h, "data_fp": repro.fingerprint(close)}
    if (row["asof"], ticker) in have:
        return {**row, "skip": True}

    # 방향: 앱 get_direction과 같은 입력(최대 10년 종가 + 거시 + 대체 데이터)과 같은 자동 선택
    macro = md.load_macro(ticker)
    alt, _ = ad.build_alt_features(ticker, close.index)
    wf, _ = dr.walk_forward(close, h, macro, alt=alt, ticker=ticker)
    p_up, pick, now_all, _, _ = dr.predict_now(close, h, wf, macro, alt=alt, ticker=ticker)
    row.update(p_up=p_up, p_base=now_all.get("base", np.nan), model=pick)

    # 변동성·가격 범위·하루 VaR: 앱 '주가 예측 범위' 탭과 같은 t분포 GARCH 시뮬레이션 (학습 3년, 대칭/GJR 중 AIC로)
    c3 = close[close.index > asof - pd.DateOffset(years=GARCH_YEARS)]
    g = fit_vol_model(np.log(c3).diff().dropna().values, "auto")        # 앱 기본 '자동 (AIC)'과 같게
    sims = g.simulate(h)
    q = spot * np.exp(np.quantile(sims[:, -1], [0.025, 0.16, 0.84, 0.975]))
    row.update(vol_garch=g.forecast_vol(h), lo95=q[0], lo68=q[1], hi68=q[2], hi95=q[3],
               var95_1d=var_es(sims[:, 0], 0.95)[0], var99_1d=var_es(sims[:, 0], 0.99)[0], garch_ok=bool(g.converged),
               vol_model=g.name)

    # 시장 기대 변동성: 지금 옵션의 ATM IV(약 1개월 만기)와, 있으면 변동성 지수(SPY → VIX 등)
    try:
        r, _ = md.risk_free_rate(ticker)
        q = md.dividend_yield(ticker, spot)
        opt = md.atm_implied_vol(ticker, spot, round(h * 365 / 252), r, q)
    except Exception:
        r, q, opt = 0.04, 0.0, None
    row.update(iv_atm=opt["iv"] if opt else np.nan, iv_days=opt["cal_days"] if opt else np.nan)
    vi = md.load_vol_index(ticker, 1)
    row["vol_index"] = float(vi.loc[:asof].iloc[-1]) if vi is not None and len(vi.loc[:asof]) else np.nan

    row.update(sentiment(ticker, spot, r, q))

    # 시장 국면: 앱과 같이 최대 10년치 전체로 추정한 지금의 불안 국면 확률
    rr = np.log(close).diff().dropna().values
    row["p_high"] = float(rg.RegimeSwitchingModel().fit(rr).filter(rr)[-1])
    row["version"] = os.environ.get("GITHUB_SHA", "")[:7]
    row["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return row


def sentiment(ticker, spot, r, q):
    """앱 '시장 심리' 탭과 같은 지표. 받지 못한 것은 결측(한국 종목은 옵션·EPS 전망이 보통 없음)."""
    out = dict.fromkeys([*SENTIMENT, "news_n"], np.nan)
    try:
        o = ad.options_snapshot(ticker, spot, r, q)
        if o:
            out.update(pc_volume=o["pc_volume"], pc_oi=o["pc_oi"], skew=o["skew"])
    except Exception:
        pass
    try:
        rv = ad.estimate_revisions(ticker)
        if rv is not None and "30일 전 대비" in rv and "올해" in rv.index:
            out["eps_rev_30d"] = float(rv.loc["올해", "30일 전 대비"])
    except Exception:
        pass
    try:
        nw = ad.news_sentiment(ticker)
        if nw is not None and len(nw):
            recent = nw[nw["time"] >= nw["time"].max() - pd.Timedelta(days=7)]
            out.update(news_7d=float(recent["score"].mean()), news_n=len(recent))
    except Exception:
        pass
    return out


def _predict_retry(ticker, tries=3, wait=30, have=frozenset()):
    for k in range(tries):
        try:
            return predict(ticker, have=have)
        except Exception as e:                       # 무료 데이터는 가끔 막히거나 비어 온다 → 잠시 뒤 다시
            print(f"{ticker}: 실패 ({k + 1}/{tries}) {type(e).__name__}: {e}", flush=True)
            if k + 1 < tries:
                time.sleep(wait)
    return None


def record(path=LOG_PATH, names=None):
    """오늘 예측을 기록에 추가 (같은 날짜·종목이 이미 있으면 건너뜀 — 먼저 적은 예측을 고치지 않는다)."""
    log = read_log(path)
    have = set(zip(pd.to_datetime(log["asof"]).dt.strftime("%Y-%m-%d"), log["ticker"])) if len(log) else set()
    new, failed = [], []
    for t in names or tickers():
        row = _predict_retry(t, have=have)
        if row is None:
            failed.append(t)
        elif row.get("skip"):
            print(f"{t}: {row['asof']} 기록이 이미 있음 (휴장일 등) — 건너뜀", flush=True)
        else:
            new.append(row)
            print(f"{t}: {row['asof']} 상승 확률 {row['p_up']:.1%} ({row['model']}), GARCH {row['vol_garch']:.1%}", flush=True)
    if new:
        out = pd.concat([log, pd.DataFrame(new)], ignore_index=True)[COLUMNS] if len(log) else pd.DataFrame(new)[COLUMNS]
        out["asof"] = pd.to_datetime(out["asof"]).dt.strftime("%Y-%m-%d")
        Path(path).parent.mkdir(exist_ok=True)
        out.to_csv(path, index=False, float_format="%.8g")
    return new, failed


def score(log, close):
    """한 종목의 기록을 실제 종가로 채점. close: 그 종목의 (수정)종가 — 기록 기간과 그 뒤를 포함해야 한다.

    가격 범위는 '기록한 날 종가 대비 비율'로 바꿔 비교한다 (배당 등으로 과거 수정주가가 바뀌어도 맞도록).
    반환: dict(rows = 기록별 채점 표, summary = 지표). 아직 결과가 안 나온 기록은 채점에서 빠진다.
    """
    log = log.sort_values("asof")
    px, idx = close.values, close.index
    lr = np.diff(np.log(px))
    rows = []
    for _, r in log.iterrows():
        i = idx.searchsorted(r["asof"], side="right") - 1          # 기록한 날(또는 그 직전 거래일)
        if i < 0:
            continue
        h = int(r["h"])
        out = {"asof": r["asof"], "p_up": r["p_up"], "p_base": r["p_base"], "model": r["model"], "p_high": r["p_high"],
               **{c: r.get(c, np.nan) for c in SENTIMENT}}
        if i + 1 < len(px):
            r1 = px[i + 1] / px[i] - 1
            out.update(ret_1d=r1, exc95=float(r1 < -r["var95_1d"]), exc99=float(r1 < -r["var99_1d"]))
        if i + h < len(px):
            g = px[i + h] / px[i]
            # 참/거짓은 숫자(1/0)로 둔다 — 결측과 섞인 object 열의 평균은 pandas에서 틀리게 나온다
            out.update(done=1.0, up=float(g > 1), ret_h=g - 1, rv=np.sqrt(np.mean(lr[i:i + h] ** 2) * 252),
                       vol_garch=r["vol_garch"], iv_atm=r["iv_atm"], vol_index=r["vol_index"],
                       in68=float(r["lo68"] / r["close"] <= g <= r["hi68"] / r["close"]),
                       in95=float(r["lo95"] / r["close"] <= g <= r["hi95"] / r["close"]))
        if i + 1 + h < len(px):           # 지표 채점용: 기록 다음 거래일 종가부터 (기록 뒤 밤사이 뉴스로 생긴 갭은 뺌)
            out.update(ret_next=px[i + 1 + h] / px[i + 1] - 1, rv_next=np.sqrt(np.mean(lr[i + 1:i + 1 + h] ** 2) * 252))
        rows.append(out)
    rows = pd.DataFrame(rows)
    s = {"기록 수": len(log), "첫 기록일": log["asof"].min(), "마지막 기록일": log["asof"].max()}
    if rows.empty:
        return {"rows": rows, "summary": s}
    if "exc95" in rows:
        d1 = rows.dropna(subset=["ret_1d"])
        s["하루 VaR 95%"] = var_exception_tests(d1["exc95"].values == 1, 0.05)
        s["하루 VaR 99%"] = var_exception_tests(d1["exc99"].values == 1, 0.01)
    done = rows[rows["done"] == 1] if "done" in rows else rows.iloc[:0]
    s["채점 수"] = len(done)
    if len(done):
        h = int(log["h"].iloc[0])
        up, p, b = done["up"].values, done["p_up"].values, done["p_base"].values
        bs, bb = (p - up) ** 2, (b - up) ** 2
        s.update({"적중률": float(np.mean((p > 0.5) == up)), "기본 확률 적중률": float(np.mean((b > 0.5) == up)),
                  "확률 정확도 개선": float(1 - bs.mean() / bb.mean()) if bb.mean() > 0 else np.nan,
                  "개선 z": overlap_tstat(bb - bs, h), "68% 범위 적중": float(done["in68"].mean()),
                  "95% 범위 적중": float(done["in95"].mean())})
        rmse = lambda c: float(np.sqrt(np.nanmean((done[c] - done["rv"]) ** 2))) if done[c].notna().any() else np.nan
        s.update({"GARCH 오차": rmse("vol_garch"), "옵션 IV 오차": rmse("iv_atm"), "변동성 지수 오차": rmse("vol_index"),
                  "불안 국면일 때 실제 변동성": float(done.loc[done["p_high"] >= 0.5, "rv"].mean()),
                  "안정 국면일 때 실제 변동성": float(done.loc[done["p_high"] < 0.5, "rv"].mean())})
    s["시장 심리 지표"] = sentiment_check(rows, int(log["h"].iloc[0]))
    return {"rows": rows, "summary": s}


def sentiment_check(rows, h, min_n=10):
    """시장 심리 지표가 '다음 거래일부터 h일' 수익률·실제 변동성과 관련이 있는지 (기록이 매일 겹치므로 겹침 보정 t값)."""
    out = {}
    for c, label in SENTIMENT.items():
        d = rows[[c, "ret_next", "rv_next"]].dropna() if {c, "ret_next"} <= set(rows) else pd.DataFrame()
        row = {"표본": len(d)}
        if len(d) >= min_n and d[c].std() > 0:
            for y, name in (("ret_next", "수익률"), ("rv_next", "변동성")):
                o = ols(d[y], d[[c]], hac_lags=h - 1)
                row[f"{name}과 상관"] = float(np.corrcoef(d[c], d[y])[0, 1])
                row[f"{name} t값"] = float(o["t"][c])
        out[label] = row
    cols = ["표본", "수익률과 상관", "수익률 t값", "변동성과 상관", "변동성 t값"]
    return pd.DataFrame(out).T.reindex(columns=cols).astype(float)          # 숫자 열로 (object 열의 결측은 화면에 None으로 보임)


def check():
    """데이터 연결 점검 (GitHub Actions에서 코드가 바뀔 때마다): 지수·주식·ETF 예측, 코스피200(네이버 차트), KRX 전 종목 시세 접근을 확인."""
    ok = True
    for t in ("^KS11", "005930.KS", "069500.KS"):
        row = _predict_retry(t, tries=2, wait=20)
        ok &= row is not None
        print(t, row, flush=True)
    try:                                       # 코스피200은 Yahoo에 기록이 거의 없어 네이버 차트에서 받는다
        k2 = md.load_history("^KS200", 1)
        print(f"코스피200: {len(k2)}일, {k2.index[0].date()} ~ {k2.index[-1].date()}, 마지막 {k2.iloc[-1]:.2f}", flush=True)
        ok &= len(k2) > 200
    except Exception as e:
        print(f"코스피200: 받지 못함 ({type(e).__name__}: {str(e)[:200]})", flush=True)
        ok = False
    import kr_data
    print("API 키 점검:", key_check(), flush=True)
    try:
        snap = kr_data.krx_daily_all(pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None) - pd.offsets.BDay(1))
        print(f"KRX 전 종목 시세: {len(snap)}종목, 열 {list(snap.columns)[:12]}", flush=True)
        print(snap.head(3).to_string(), flush=True)
    except Exception as e:                     # 아직 필수 아님 (상장폐지 포함 종목 범위를 만들 때 쓸 예정) → 실패해도 점검은 통과
        print(f"KRX 전 종목 시세: 받지 못함 ({type(e).__name__}: {str(e)[:200]})", flush=True)
    return ok


def key_check():
    """API 키가 있는지, 실제로 한 번 불러지는지 (키 값은 출력하지 않는다). 키가 주소에 들어가는 요청의 오류는 종류만 적는다."""
    import requests
    import kr_data
    out, end = {}, pd.Timestamp.today()
    if kr_data.get_key("DART_API_KEY"):
        try:
            out["DART"] = f"정상 (삼성전자 고유번호 {kr_data.dart_corp_code('005930')})"
        except RuntimeError as e:
            out["DART"] = f"오류: {str(e)[:120]}"
        except Exception as e:
            out["DART"] = f"오류: {type(e).__name__}"
    else:
        out["DART"] = "키 없음"
    if kr_data.get_key("ECOS_API_KEY"):
        try:
            s = kr_data.ecos_series("817Y002", kr_data.ECOS_RATES["kr3y"], end - pd.Timedelta(days=14), end)
            out["ECOS"] = f"정상 (국고채 3년 {s.index[-1].date()} {s.iloc[-1]:.3f}%)"
        except RuntimeError as e:
            out["ECOS"] = f"오류: {str(e)[:120]}"
        except Exception as e:
            out["ECOS"] = f"오류: {type(e).__name__}"
    else:
        out["ECOS"] = "키 없음"
    key = kr_data.get_key("KRX_API_KEY")
    if key:                                   # KRX Open API는 키를 머리말(AUTH_KEY)로 보낸다
        day = (end - pd.offsets.BDay(1)).strftime("%Y%m%d")
        for name, path in (("파생상품지수(VKOSPI)", "idx/drvprod_dd_trd"), ("옵션 일별", "drv/opt_bydd_trd"),
                           ("유가증권 일별", "sto/stk_bydd_trd")):
            try:
                r = requests.get(f"https://data-dbg.krx.co.kr/svc/apis/{path}", params={"basDd": day},
                                 headers={"AUTH_KEY": key}, timeout=30)
                rows = r.json().get("OutBlock_1") if r.ok else None
                out[f"KRX {name}"] = f"정상 ({len(rows)}행)" if rows else f"{r.status_code} {r.text[:100]!r}"
            except Exception as e:
                out[f"KRX {name}"] = f"오류: {type(e).__name__}"
    else:
        out["KRX"] = "키 없음"
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "record":
        new, failed = record()
        print(f"추가 {len(new)}줄, 실패 {len(failed)}종목 {failed}")
        sys.exit(1 if failed and not new else 0)      # 전부 실패했을 때만 실패로 표시 (알림 메일)
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    else:
        print("사용법: python forecast_log.py record | check")
