"""하루 VaR가 코스피 지수에서 위험을 작게 잡는 이유 진단 (보고만 — 후보 수정의 성적은 계산하지 않는다).

지금 기본 모델(자동: GARCH·GJR 중 AIC, t분포, 750일 적합·21일마다 재적합)의 검증 구간을 쪼개 본다.
  1. 연도별 예외 비율과 예측·실현 변동성
  2. 표준화 수익률 z = r / σ의 분포 — 크기(σ가 작음) 문제인지, 모양(왼쪽 꼬리) 문제인지. 위쪽 꼬리도 함께 센다
  3. 예측 변동성 수준별 예외 — 잠잠할 때 터지는지
  4. 최근 실현 변동성 ÷ 예측 변동성 수준별 예외 — 예측이 실제 변동성을 늦게 따라가는지
  5. 예외 몰림, 실현 분산 대 예측 분산 회귀(Mincer–Zarnowitz: 기울기 1, 절편 0이 이상적)
  6. 재적합마다 고른 모델·자유도·지속성, 데이터 이상(수익률이 정확히 0인 날)
VKOSPI(KRX 파생상품지수)는 검증 기간에 받을 수 있는지만 확인한다.
후보 수정은 이 진단을 본 뒤, 그 성적을 보기 전에 EXPERIMENTS.md에 사전 등록한다.

두 번째 진단(cluster, 2026-10-29): VKOSPI 섞기를 채택한 뒤에도 코스피 지수 VaR 검정이 기각되는 이유 — 예외가 몰려 나오는가.
지금 기본(코스피 200 종목은 VKOSPI 섞기)의 예외를 쪼개 본다: 수준별 Kupiec·Christoffersen와 전이 확률, 몰린 구간,
전날 표준화 수익률·요일·전날 VKOSPI 변화별 예외, 예외 다음 날 σ가 얼마나 따라 올랐는지. 후보의 성적은 계산하지 않는다.

실제 시세는 GitHub Actions(var-study.yml)에서 받는다: python var_study.py [sigma | cluster]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import forecast_log as fl
import kr_data
import market_data as md
from quant_engine import (VAR_IV_TICKERS, ewma_level, fit_vol_model, iv_blend_factor, range_blend_factor, range_variance, var_backtest,
                          var_exception_tests, var_rescale)
from validation import R1_MIN_VALID, VAR_START, _vkospi_prev

# 지수·지수 ETF와, 비교용으로 VaR 검정을 통과·기각한 개별 종목 몇 개
TICKERS = ["^KS11", "069500.KS", "229200.KS", "005930.KS", "000660.KS", "035420.KS", "모의 (모델이 맞는 경우)"]
REFIT = 21


def log_returns(ticker):
    if ticker == SIM:
        return simulated()
    close = fl.drop_unfinished(md.load_history(ticker, 10), ticker)
    return np.log(close).diff().dropna()


SIM = "모의 (모델이 맞는 경우)"


def simulated(n=2450, seed=1):
    """비교 기준: 실제로 GARCH(1,1)·t분포(자유도 6)에서 나온 수익률 — 모델이 맞을 때 진단 숫자가 어떻게 나오는지."""
    g = np.random.default_rng(seed)
    s2, r = np.empty(n), np.empty(n)
    s2[0] = 1e-4
    for t in range(n):
        if t:
            s2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * g.standard_t(6) / np.sqrt(6 / 4)
    return pd.Series(r, index=pd.bdate_range("2016-11-01", periods=n))


def _tercile(x, labels):
    return pd.qcut(x.rank(method="first"), 3, labels=labels)


def diagnose(r):
    """한 종목의 검증 구간 진단. 반환: dict(표·숫자들)."""
    rr = r.values
    start = max(min(VAR_START, len(rr) - 500), 250)        # validation.py와 같은 검증 구간
    _, d = var_backtest(rr, model="auto", start=start, detail=True, innov="t")
    d.index = r.index[start:]
    x, s = d["수익률"], d["변동성"]
    z = x / s
    h95, h99 = d["예외 95%"].astype(bool), d["예외 99%"].astype(bool)
    out = {"기간": f"{d.index[0].date()} ~ {d.index[-1].date()} · {len(d)}일"}

    g = d.groupby(d.index.year)
    out["연도별"] = pd.DataFrame({
        "일수": g.size(), "95% 예외": g["예외 95%"].mean(), "99% 예외": g["예외 99%"].mean(),
        "예측 변동성(연)": g["변동성"].apply(lambda v: np.sqrt((v ** 2).mean() * 252)),
        "실현 변동성(연)": g["수익률"].apply(lambda v: np.sqrt((v ** 2).mean() * 252))})

    q95, q99 = float((d["VaR 95%"] / s).mean()), float((d["VaR 99%"] / s).mean())
    out["z"] = {
        "z 평균": z.mean(), "z 표준편차": z.std(), "z 왜도": float(stats.skew(z)), "z 초과첨도": float(stats.kurtosis(z)),
        "z 1%": z.quantile(0.01), "z 5%": z.quantile(0.05), "z 95%": z.quantile(0.95), "z 99%": z.quantile(0.99),
        "모델 분위수 95% (t, 평균)": -q95, "모델 분위수 99% (t, 평균)": -q99,
        "아래쪽 95% 예외": h95.mean(), "위쪽 95% 넘음": float((x > d["VaR 95%"]).mean()),
        "아래쪽 99% 예외": h99.mean(), "위쪽 99% 넘음": float((x > d["VaR 99%"]).mean())}

    lv = _tercile(s, ["낮음", "중간", "높음"])
    out["예측 변동성 수준별"] = pd.DataFrame({"95% 예외": h95.groupby(lv, observed=False).mean(),
                                       "z 표준편차": z.groupby(lv, observed=False).std(),
                                       "예측 변동성(연)": (s * np.sqrt(252)).groupby(lv, observed=False).mean()})
    rv = np.sqrt((x ** 2).rolling(21).mean()).shift(1)       # 전날까지 21일 실현 변동성 (그날 수익률은 안 씀)
    ratio = (rv / s).dropna()
    lr = _tercile(ratio, ["실현 < 예측", "비슷", "실현 > 예측"])
    out["최근 실현 ÷ 예측 수준별"] = pd.DataFrame({"95% 예외": h95.reindex(ratio.index).groupby(lr, observed=False).mean(),
                                          "비율 평균": ratio.groupby(lr, observed=False).mean()})

    b, a = np.polyfit(s ** 2, x ** 2, 1)
    blk = np.arange(len(d)) // 21                             # 겹치지 않는 21일 묶음
    sb, xb = (s ** 2).groupby(blk).sum(), (x ** 2).groupby(blk).sum()
    b21, a21 = np.polyfit(sb, xb, 1)
    out["몰림·분산"] = {
        "전날 예외였을 때 95% 예외": float(h95[h95.shift(1, fill_value=False)].mean()) if h95.any() else np.nan,
        "실현 분산 ÷ 예측 분산 (전체)": float((x ** 2).mean() / (s ** 2).mean()),
        "MZ 기울기 (하루)": b, "MZ 절편 ÷ 평균 예측 분산 (하루)": a / float((s ** 2).mean()),
        "MZ 기울기 (21일 합)": b21, "MZ 절편 ÷ 평균 예측 분산 (21일 합)": a21 / float(sb.mean()),
        "수익률 0인 날": int((x == 0).sum()), "|수익률| > 8%인 날": int((x.abs() > 0.08).sum()),
        "수익률 0인 다음 날 95% 예외": float(h95[(x == 0).shift(1, fill_value=False)].mean()) if (x == 0).any() else np.nan}

    fits = []
    for b0 in range(start, len(rr), REFIT):
        m = fit_vol_model(rr[max(0, b0 - 750):b0], "auto")
        fits.append({"GJR": m.asym, "nu": m.nu, "지속성": m.persistence, "alpha": m.alpha, "gamma": m.gamma,
                     "장기 변동성(연)": np.sqrt(m.long_run_var * 252) if m.persistence < 1 else np.nan,
                     "적합 구간 표준화 잔차 표준편차": float(np.std(m.std_resid))})
    f = pd.DataFrame(fits)
    out["재적합"] = {"재적합 수": len(f), "GJR 비율": f["GJR"].mean(), "nu 중앙값": f["nu"].median(),
                  "nu 범위": f"{f['nu'].min():.1f} ~ {f['nu'].max():.1f}", "지속성 중앙값": f["지속성"].median(),
                  "alpha 중앙값": f["alpha"].median(), "gamma 중앙값": f["gamma"].median(),
                  "장기 변동성(연) 중앙값": f["장기 변동성(연)"].median(),
                  "적합 구간 표준화 잔차 표준편차 중앙값": f["적합 구간 표준화 잔차 표준편차"].median()}
    return out, d.index


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


CLUSTER_TICKERS = ["^KS11", "069500.KS", "229200.KS", "005930.KS", "000660.KS", SIM]


def current_var(ticker, r):
    """지금 기본 하루 VaR의 검증 구간 날짜별 표 (자동 GARCH·t분포, 코스피 200 종목은 전날 VKOSPI 섞기, 나머지는 장중 고가·저가
    섞기 — validation.py와 같게). 반환: (날짜별 표, 전날 VKOSPI 또는 None)."""
    rr = r.values
    start = max(min(VAR_START, len(rr) - 500), 250)
    tab, d = var_backtest(rr, model="auto", start=start, detail=True, innov="t")
    iv = None
    if ticker in VAR_IV_TICKERS:
        iv = _vkospi_prev(r.index[start:])
        if iv is not None:
            tab, d = var_rescale(tab, d, iv_blend_factor(d["변동성"], iv))
    else:
        try:
            rv = range_variance(md.load_ohlc(ticker, 10)).reindex(r.index)
            if rv.iloc[start:].notna().mean() >= R1_MIN_VALID:
                tab, d = var_rescale(tab, d, range_blend_factor(d["변동성"], ewma_level(rv).shift(1).iloc[start:].to_numpy()))
        except Exception:
            pass
    d.index = r.index[start:]
    return d, iv


def cluster_report(d, iv=None, gap=3):
    """예외 몰림 진단. gap: 예외 사이가 이 거래일 수 이하면 같은 묶음."""
    out = {}
    x, s = d["수익률"], d["변동성"]
    z = x / s
    for lv in (0.95, 0.99):
        h = d[f"예외 {lv:.0%}"].astype(bool).to_numpy()
        a, b = h[:-1], h[1:]
        n01, n11 = int(np.sum(~a & b)), int(np.sum(a & b))
        out[f"검정 {lv:.0%}"] = {**var_exception_tests(h, 1 - lv),
                               "π01 (전날 예외 아님 → 예외)": n01 / max(int(np.sum(~a)), 1),
                               "π11 (전날 예외 → 예외)": n11 / max(int(np.sum(a)), 1), "이틀 연속 예외": n11}
    h = d["예외 95%"].astype(bool)
    pos = np.flatnonzero(h.to_numpy())
    groups = np.split(pos, np.flatnonzero(np.diff(pos) > gap) + 1) if len(pos) else []
    out["몰린 구간 (95%, 예외 사이 3거래일 이하, 2번 이상)"] = pd.DataFrame(
        [{"시작": d.index[g[0]].date(), "끝": d.index[g[-1]].date(), "예외 수": len(g),
          "손실 ÷ VaR 평균": float((-x.iloc[g] / d["VaR 95%"].iloc[g]).mean()),
          "구간 첫날 예측 변동성(연)": float(s.iloc[g[0]] * np.sqrt(252))} for g in groups if len(g) >= 2])
    zp = z.shift(1)
    lab = pd.cut(zp, [-np.inf, -2, -1, 1, 2, np.inf], labels=["< −2", "−2 ~ −1", "−1 ~ 1", "1 ~ 2", "> 2"])
    out["전날 z별"] = pd.DataFrame({"일수": d.groupby(lab, observed=False).size(),
                                 "95% 예외": d["예외 95%"].groupby(lab, observed=False).mean(),
                                 "99% 예외": d["예외 99%"].groupby(lab, observed=False).mean(),
                                 "그날 z 표준편차": z.groupby(lab, observed=False).std()})
    wd = pd.Series(d.index.dayofweek, index=d.index).map(dict(enumerate("월화수목금토일")))
    out["요일별"] = pd.DataFrame({"일수": d.groupby(wd).size(), "95% 예외": d["예외 95%"].groupby(wd).mean(),
                               "z 표준편차": z.groupby(wd).std()}).reindex(list("월화수목금"))
    after = h.shift(1, fill_value=False)
    out["예외 다음 날"] = {"예외 다음 날 수": int(after.sum()),
                        "예측 σ 오른 배수 (중앙값)": float((s / s.shift(1))[after].median()),
                        "z 표준편차 (예외 다음 날)": float(z[after].std()), "z 표준편차 (그 밖)": float(z[~after].std()),
                        "95% 예외 비율 (예외 다음 날)": float(h[after].mean()),
                        "예외 다음 날 손실 쪽 / 이익 쪽 큰 움직임(|z|>2)": f"{int(((z < -2) & after).sum())} / {int(((z > 2) & after).sum())}"}
    if iv is not None:
        ivs = pd.Series(iv, index=d.index)
        dv = np.log(ivs).diff()                                   # 전날 VKOSPI의 하루 변화 (그날 아침에 이미 앎)
        ok = dv.notna()
        lv3 = _tercile(dv[ok], ["VKOSPI 내림", "비슷", "VKOSPI 오름"])
        out["전날 VKOSPI 변화별"] = pd.DataFrame({"일수": d[ok].groupby(lv3, observed=False).size(),
                                           "95% 예외": d.loc[ok, "예외 95%"].groupby(lv3, observed=False).mean(),
                                           "z 표준편차": z[ok].groupby(lv3, observed=False).std()})
    return out


def main_cluster():
    pd.set_option("display.width", 200)
    for t in CLUSTER_TICKERS:
        print(f"\n==================== {t} ====================", flush=True)
        try:
            r = log_returns(t)
            d, iv = current_var(t, r)
            out = cluster_report(d, iv)
        except Exception as e:
            print(f"실패: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        print(f"검증 기간: {d.index[0].date()} ~ {d.index[-1].date()} · {len(d)}일 · "
              f"{'VKOSPI 섞기' if iv is not None else '기본 (GARCH만)'}")
        for k, v in out.items():
            print(f"\n[{k}]")
            if isinstance(v, pd.DataFrame):
                print(v.to_string(float_format=lambda u: f"{u:.4f}") if len(v) else "  없음")
            else:
                for kk, vv in v.items():
                    print(f"  {kk}: {_fmt(vv)}")
        print(flush=True)


def main():
    pd.set_option("display.width", 200)
    idx = None
    for t in TICKERS:
        print(f"\n==================== {t} ====================", flush=True)
        try:
            out, dates = diagnose(log_returns(t))
        except Exception as e:
            print(f"실패: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        if t == "^KS11":
            idx = dates
        print("검증 기간:", out["기간"])
        for k in ("연도별", "예측 변동성 수준별", "최근 실현 ÷ 예측 수준별"):
            print(f"\n[{k}]")
            print(out[k].to_string(float_format=lambda v: f"{v:.4f}"))
        for k in ("z", "몰림·분산", "재적합"):
            print(f"\n[{k}]")
            for kk, v in out[k].items():
                print(f"  {kk}: {_fmt(v)}")
        print(flush=True)
    # VKOSPI: 받을 수 있는지만 (값으로 VaR를 계산하지 않는다)
    print("\n==================== VKOSPI 받기 점검 ====================", flush=True)
    if not kr_data.get_key("KRX_API_KEY"):
        print("KRX_API_KEY 없음 — 건너뜀")
        return
    if idx is None:
        print("코스피 검증 날짜가 없어 건너뜀")
        return
    try:
        sample = kr_data.krx_drv_index_day(idx[-1])
        print("파생상품지수 열:", list(sample.columns))
        name_col = next((c for c in ("IDX_NM", "ISU_NM") if c in sample), None)
        if name_col:
            print("지수 이름들:", list(sample[name_col].astype(str))[:40])
    except Exception as e:
        print(f"파생상품지수 받기 실패: {type(e).__name__}: {str(e)[:200]}")
        return
    v, names = kr_data.vkospi(idx)
    ok = v.dropna()
    print(f"VKOSPI로 잡힌 이름: {names}")
    print(f"검증 기간 {len(idx)}일 중 값이 있는 날 {len(ok)}일"
          + (f" ({ok.index[0].date()} ~ {ok.index[-1].date()})" if len(ok) else ""), flush=True)


if __name__ == "__main__":
    main_cluster() if (sys.argv[1:] or ["sigma"])[0] == "cluster" else main()
