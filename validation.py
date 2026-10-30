"""실제 데이터 검증 보고서: 계산이 맞는지(tests.py)와 별개로, 비용을 뺀 뒤에도 돈이 되는지를 실제 데이터로 잰다.

GitHub Actions(validation.yml)가 매주 돌려 validation/summary.csv와 validation/README.md로 커밋한다.
기록 종목(forecast_log/tickers.txt)마다 앱 기본 설정 그대로, 워크포워드(그 시점까지의 데이터로만 학습·선택)로:
- 방향 예측: 21거래일 방향을 21거래일 뒤 결과로 채점 (적중률, 확률 정확도, 겹침 보정 z)
- 방향 전략·시장 국면 전략: 판단 다음 날 종가 체결, 수수료·증권거래세(체결일 세율)를 뺀 뒤 단순 보유와 비교.
  방향 전략은 두 가지: 매일 다시 판단해 다음 하루를 들고 있는 일일 전략(앱 화면과 같음, 21일 예측과는 다른 질문)과
  매일 자본의 1/21을 그날 신호대로 사서 21거래일 들고 있는 분할 보유 전략(21일 예측을 그 기간 그대로 매매에 씀).
  일간 초과수익(전략 − 단순 보유)의 블록 부트스트랩 95% 신뢰구간, 가정 바꿔 점검 통과 수, 1등 착시 보정(Deflated Sharpe)
- 하루 VaR 95%·99% 예외 검정 (Kupiec·Christoffersen)
- 코스피 종목 고르기: 매주 계산 결과(ranking/)의 비용 후 초과수익·t값, 요인 알파, 실전 기록

명령: python validation.py          (기록 종목 전체, 결과 저장)
      python validation.py --smoke  (두 종목만, 코드가 바뀐 푸시에서 빠른 점검)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import alt_data as ad
import direction as dr
import forecast_log as fl
import kr_costs as kc
import market_data as md
import regime as rg
import repro
import universe as uv
from quant_engine import (VAR_INNOV, VAR_INNOVS, VAR_IV_TICKERS, after_break, block_bootstrap_ci, cond_recal_factor,
                          ewma_level, fit_vol_model, group_recal_factor, iv_blend_factor, range_blend_factor,
                          range_variance, recal_factor, var_backtest, var_rescale)

OUT_DIR = Path(__file__).parent / "validation"
H, COST_BPS, LAG = 21, 3.0, 1          # 앱 기본 설정 (방향 예측 기간, 수수료·체결 비용 bp, 다음 날 종가 체결)
BLOCK = 2 * H                          # 블록 부트스트랩 블록 길이 (거래일) — 포지션·변동성이 이어지는 기간보다 넉넉하게
VAR_START = 756                        # 하루 VaR 검증 시작 위치: 첫 3년은 학습에만 쓰고 그 뒤 전체를 검증
INDEX_NAMES = {"^KS11": "코스피 지수", "^KQ11": "코스닥 지수", "^KS200": "코스피 200 지수"}
# 하루 VaR σ 보정 두 후보 (EXPERIMENTS.md 사전 등록 2026-10-29): 같은 GARCH 예측에 날짜별 배수만 곱해 다시 채점
VAR_SCALES = {"recal": "크기 재보정 (C2)", "vkospi": "VKOSPI 섞기 (C1)",
              "vkospi2": "VKOSPI 하루 늦게 섞기 (C1L2)"}     # C1L2: 실전에서 받을 수 있는 전전날 값 (사전 등록 2026-10-30)
VKOSPI_TICKERS = VAR_IV_TICKERS          # 코스피 200을 따르는 종목만 VKOSPI 섞기 대상 (2026-10-29 판정으로 이 종목들의 기본)


def _ann(x):
    """일간 로그수익률 → (연 수익률, 샤프)."""
    m, s = x.mean() * 252, x.std() * np.sqrt(252)
    return float(np.expm1(m)), float(m / s) if s > 0 else 0.0


def _excess(daily):
    """일간 (전략 − 단순 보유) 로그수익률 → 연 초과수익과 블록 부트스트랩 95% 신뢰구간 (연 단위)."""
    ex = (daily["strat"] - daily["asset"]).values
    lo, hi = block_bootstrap_ci(ex, BLOCK)
    return float(np.nanmean(ex) * 252), lo * 252, hi * 252


def _quantile_loss(daily, lv):
    """날짜별 표(var_backtest detail)의 lv 수준 분위수 손실 합 (낮을수록 좋음)."""
    x, v = daily["수익률"].values, daily[f"VaR {lv:.0%}"].values
    return float(np.sum((x + v) * ((1 - lv) - (x < -v))))


def _var_fields(m, tab):
    """var_backtest 표 → 보고서 열 ([m] 예외 비율·검정·분위수 손실)."""
    return {f"[{m}] VaR 95% 예외": tab.loc["하루 VaR 95%", "예외 비율"], f"[{m}] VaR 99% 예외": tab.loc["하루 VaR 99%", "예외 비율"],
            f"[{m}] VaR 검정": "통과" if min(tab["종합 p값"]) > 0.05 else "기각",
            f"[{m}] ES 검정": "통과" if np.nanmin(tab["ES 검정 p값"].astype(float).fillna(1.0)) > 0.05 else "기각",
            f"[{m}] 분위수 손실": float(tab["분위수 손실(×10⁴)"].sum())}


def _halves(daily):
    n = len(daily) // 2
    return (("앞 절반", daily.iloc[:n]), ("뒤 절반", daily.iloc[n:]))


def _ql(daily):
    return sum(_quantile_loss(daily, lv) for lv in (0.95, 0.99))


def _vkospi_prev(dates, lag=1):
    """dates(수익률 날짜)마다 lag 거래일 전 종가의 VKOSPI (lag=1: 예측하는 날 전날 값 → 미래 정보 없음,
    lag=2: 그 전 거래일 값 — KRX가 그날 값을 다음 영업일 아침에 주므로 실전에서 받을 수 있는 것). 받지 못하면 None."""
    import kr_data
    if not kr_data.get_key("KRX_API_KEY"):
        return None
    vk, _ = kr_data.vkospi(dates)
    return vk.shift(lag).to_numpy()


def _without_recent(close, wf, r, tax, days):
    """마지막 days거래일이 없었다고 보고 방향·분할 보유·국면 전략을 다시 채점 (보고용). 워크포워드라 그 전의 예측은 그대로."""
    cut = close.index[-days - 1]
    c = close.loc[:cut]
    w = wf[wf.index <= cut].copy()
    w["fwd"] = dr.forward_log_return(c, H).reindex(w.index)          # 그 뒤의 결과는 모르는 것으로
    if w["fwd"].notna().sum() < 252:
        return {}
    rc = r.loc[:cut]
    ev = dr.evaluate(w, H, rc, COST_BPS, lag=LAG, sell_tax=tax)
    evh = dr.evaluate(w, H, rc, COST_BPS, lag=LAG, sell_tax=tax, hold=H)
    reg = rg.analyze(c, COST_BPS, lag=LAG, sell_tax=tax)
    out = {"[최근1년 제외] 끝": str(cut.date()), "[최근1년 제외] 방향 z": ev["z"], "[최근1년 제외] 확률 정확도": ev["skill"],
           "[최근1년 제외] 방향 예측력": "있음" if ev["skill"] > 0 and ev["z"] > 2 else "확인 안 됨"}
    for name, daily in (("방향", ev["daily"]), ("분할", evh["daily"]), ("국면", reg["daily"])):
        e, lo, hi = _excess(daily)
        out.update({f"[최근1년 제외] {name} 초과": e, f"[최근1년 제외] {name} 하한": lo, f"[최근1년 제외] {name} 상한": hi})
    return out


def validate(ticker):
    """한 종목의 검증 결과 한 줄 (dict)."""
    close = fl.drop_unfinished(md.load_history(ticker, 10), ticker)
    name, _, sec = md.kr_info(ticker) if ticker not in INDEX_NAMES else (INDEX_NAMES[ticker], None, None)
    kind = kc.tax_kind(ticker, name, sec)
    tax = kc.sell_tax_rate(close.index, kind)
    r = np.log(close).diff()
    row = {"종목": ticker, "이름": name or ticker, "세금 구분": kind, "데이터": repro.fingerprint(close)}

    # 방향 예측 (앱 get_direction과 같은 입력·같은 자동 선택)
    macro = md.load_macro(ticker)
    alt, alt_status = ad.build_alt_features(ticker, close.index)
    row["대체 데이터"] = ", ".join(f"{k} {v}" for k, v in alt_status.items())
    wf, _ = dr.walk_forward(close, H, macro, alt=alt, ticker=ticker)
    if wf.empty:
        raise ValueError(f"학습에 쓸 기간이 너무 짧음 (모든 특징이 있는 날 기준, 대체 데이터: {row['대체 데이터'] or '없음'})")
    ev = dr.evaluate(wf, H, r, COST_BPS, lag=LAG, sell_tax=tax)
    rb = dr.robustness(wf, H, r, COST_BPS, lag=LAG, sell_tax=tax)
    sel = rb["selection"]
    passes = int(rb["by_window"]["검증 통과"].sum() + rb["by_period"]["검증 통과"].sum())
    d_ret, d_sh = _ann(ev["daily"]["strat"])
    b_ret, b_sh = _ann(ev["daily"]["asset"])
    d_ex, d_lo, d_hi = _excess(ev["daily"])
    evh = dr.evaluate(wf, H, r, COST_BPS, lag=LAG, sell_tax=tax, hold=H)       # 21일 분할 보유
    # 확률 구간별 실제 수익 크기 (전 종목 합산용): 한 번 사서 팔 때 비용 = 수수료 양쪽 + 파는 날(체결 h일 뒤)의 세금
    sell_day_tax = tax.shift(-(LAG + H)).reindex(wf.index).ffill() if isinstance(tax, pd.Series) else tax
    row["_obs"] = pd.DataFrame({"p": wf["auto"].values, "fwd": wf["fwd"].values,
                                "c": 2 * COST_BPS * 1e-4 + np.asarray(sell_day_tax, float) * np.ones(len(wf))})
    h_ret, h_sh = _ann(evh["daily"]["strat"])
    h_ex, h_lo, h_hi = _excess(evh["daily"])
    row.update({
        "검증 기간": f"{wf.index[0].date()} ~ {wf.index[-1].date()}",
        "방향 적중률": ev["hit"], "항상 오른다 적중률": ev["base"], "방향 z": ev["z"], "확률 정확도 개선": ev["skill"],
        "방향 예측력": "있음" if ev["skill"] > 0 and ev["z"] > 2 else "확인 안 됨",
        "방향 전략 연 수익률": d_ret, "방향 전략 샤프": d_sh, "방향 전략 최대 낙폭": ev["mdd"],
        "단순 보유 연 수익률": b_ret, "단순 보유 샤프": b_sh, "단순 보유 최대 낙폭": ev["mdd_bh"],
        "방향 전략 보유 비중": ev["time_in_market"],
        "방향 초과수익(연)": d_ex, "방향 초과 95% 하한": d_lo, "방향 초과 95% 상한": d_hi,
        "가정 바꿔 통과(7개 중)": passes, "방향 DSR": sel["dsr"] if sel else np.nan,
        "방향 자동 선택 바뀐 횟수": int((wf["chosen"] != wf["chosen"].shift()).iloc[1:].sum()),
        "방향 전략 경제적 가치": "있음" if d_lo > 0 else "확인 안 됨",
        "분할 보유 연 수익률": h_ret, "분할 보유 샤프": h_sh, "분할 보유 최대 낙폭": evh["mdd"],
        "분할 보유 평균 비중": evh["time_in_market"],
        "분할 초과수익(연)": h_ex, "분할 초과 95% 하한": h_lo, "분할 초과 95% 상한": h_hi,
        "분할 보유 경제적 가치": "있음" if h_lo > 0 else "확인 안 됨",
    })

    # 시장 국면 ('불안 국면엔 현금', 앱 get_regime과 같은 설정)
    reg = rg.analyze(close, COST_BPS, lag=LAG, sell_tax=tax)
    g_ex, g_lo, g_hi = _excess(reg["daily"])
    t = reg["table"]
    row.update({
        "국면 전략 연 수익률": t.loc["불안 국면엔 현금", "연 수익률"], "국면 전략 샤프": t.loc["불안 국면엔 현금", "샤프"],
        "국면 전략 최대 낙폭": t.loc["불안 국면엔 현금", "최대 낙폭"], "국면 전략 보유 비중": reg["time_in_market"],
        "국면 초과수익(연)": g_ex, "국면 초과 95% 하한": g_lo, "국면 초과 95% 상한": g_hi,
        "국면 전략 경제적 가치": "있음" if g_lo > 0 else "확인 안 됨",
    })
    # 최근 1년(이례적 급등락 구간)을 뺀 재채점 — 보고만 (EXPERIMENTS.md 사전 등록)
    if len(close) > uv.RECENT_DAYS + 3 * 252:
        try:
            row.update(_without_recent(close, wf, r, tax, uv.RECENT_DAYS))
        except Exception as e:                       # 기간이 짧은 종목 등 → 이 점검만 건너뜀
            row["[최근1년 제외] 오류"] = f"{type(e).__name__}: {str(e)[:80]}"

    # 하루 VaR·ES 검정 (앱 기본: 자동 = AIC로 대칭/GJR 선택). 99% VaR는 500일에 예외가 5번 정도뿐이라 검정력이 약해
    # 첫 3년(학습)을 뺀 전체 기간을 쓴다. 앞·뒤 절반의 예외 비율로 시기에 따라 흔들리는지도 본다
    rd = r.dropna()
    rr = rd.values
    vstart = max(min(VAR_START, len(rr) - 500), 250)
    # 꼬리 분포 두 후보(t, FHS)를 같은 GARCH 적합으로 함께 계산 — 앞의 본 표는 지금 기본(VAR_INNOV), 비교 표는 둘 다 (EXPERIMENTS.md 사전 등록)
    both = var_backtest(rr, model="auto", start=vstart, detail=True, innov=tuple(VAR_INNOVS))
    vt, vd = both[VAR_INNOV]
    for m, (tab, _) in both.items():
        row.update(_var_fields(m, tab))
    row["분위수 손실 비율 (FHS ÷ t)"] = row["[fhs] 분위수 손실"] / row["[t] 분위수 손실"]
    # σ 보정 두 후보 (사전 등록 2026-10-29): 기본(VAR_INNOV)의 같은 예측에 날짜별 배수를 곱해 다시 채점
    factors = {"recal": recal_factor(vd["수익률"], vd["변동성"])}
    if ticker in VKOSPI_TICKERS:
        try:
            iv = _vkospi_prev(rd.index[vstart:])
            if iv is not None and np.isfinite(iv).any():
                factors["vkospi"] = iv_blend_factor(vd["변동성"], iv)
                row["[vkospi] VKOSPI 있는 날"] = f"{int(np.isfinite(iv).sum())}/{len(iv)}"
                # C1L2 (사전 등록 2026-10-30): 하루 더 지난 VKOSPI — 기본(t)과 비교, 보고만으로 C1과도 비교
                iv2 = _vkospi_prev(rd.index[vstart:], lag=2)
                factors["vkospi2"] = iv_blend_factor(vd["변동성"], iv2)
                row["[vkospi2] VKOSPI 있는 날"] = f"{int(np.isfinite(iv2).sum())}/{len(iv2)}"
        except Exception as e:
            row["[vkospi] 오류"] = f"{type(e).__name__}: {str(e)[:80]}"
    for m, f in factors.items():
        tab, d = var_rescale(vt, vd, f)
        row.update(_var_fields(m, tab))
        row[f"[{m}] 배수 중앙값"] = float(np.median(f))
        row[f"분위수 손실 비율 ({m} ÷ 기본)"] = _ql(d) / _ql(vd)
        for (part, dp), (_, bp) in zip(_halves(d), _halves(vd)):
            row[f"분위수 손실 비율 ({m} ÷ 기본) {part}"] = _ql(dp) / _ql(bp)
    # 채택된 보정 (2026-10-29 판정): 코스피 200 종목의 본 표는 VKOSPI를 섞은 VaR. 비교 열([t]·[recal]·[vkospi])은 위에서 이미 기록
    row["VaR σ 보정"] = "VKOSPI 섞기" if ticker in VKOSPI_TICKERS and "vkospi" in factors else "없음"
    if row["VaR σ 보정"] != "없음":
        vt, vd = var_rescale(vt, vd, factors["vkospi"])
    vdays = rd.index[vstart:]
    brk = after_break(vdays, rd.index[vstart - 1])

    def compare(m, tab, d, base, label, bt, bd):
        """후보 m의 표를 기준(base 열 이름, label 비율 이름)과 비교한 열."""
        for mm, (tt, dd) in ((base, (bt, bd)), (m, (tab, d))):
            row.update(_var_fields(mm, tt))
            row[f"[{mm}] 99% 독립성 p"] = float(tt.loc["하루 VaR 99%", "Christoffersen p값"])
            row[f"[{mm}] 95% 독립성 p"] = float(tt.loc["하루 VaR 95%", "Christoffersen p값"])
            row[f"[{mm}] 휴장 뒤 95% 예외"] = float(dd["예외 95%"].to_numpy()[brk].mean()) if brk.any() else np.nan
        row[f"분위수 손실 비율 ({m} ÷ {label})"] = _ql(d) / _ql(bd)
        for (part, dp), (_, bp) in zip(_halves(d), _halves(bd)):
            row[f"분위수 손실 비율 ({m} ÷ {label}) {part}"] = _ql(dp) / _ql(bp)

    # R1 (사전 등록 2026-10-29): 장중 고가·저가 변동성(밤사이 갭 + Rogers–Satchell, EWMA 0.94)을 반반 섞음 — R1 이전 기본([pre])과 비교.
    # 판정: 코스피 200 두 종목은 미채택, 나머지 종목은 채택 → 그 종목들의 본 표는 R1. 유효한 일봉이 80% 미만이면 GARCH만
    try:
        rv = range_variance(md.load_ohlc(ticker, 10)).reindex(rd.index)
        row["[r1] 유효한 날 비율"] = float(rv.reindex(vdays).notna().mean())
        if row["[r1] 유효한 날 비율"] >= R1_MIN_VALID:
            rt, rdd = var_rescale(vt, vd, range_blend_factor(vd["변동성"], ewma_level(rv).shift(1).reindex(vdays).to_numpy()))
            compare("r1", rt, rdd, "pre", "이전 기본", vt, vd)
            if ticker not in VKOSPI_TICKERS:
                row["VaR σ 보정"] = "장중 고가·저가 섞기"
                vt, vd = rt, rdd
    except Exception as e:
        row["[r1] 오류"] = f"{type(e).__name__}: {str(e)[:80]}"
    # 지금 기본(위의 vt·vd)에 곱하는 후보 (사전 등록 2026-10-29, 둘 다 미채택 — 보고만): K2 = 전날 움직임 구간별, M1 = 휴장 뒤 첫 거래일별
    for m, f in (("k2", cond_recal_factor(vd["수익률"], vd["변동성"])), ("m1", group_recal_factor(vd["수익률"], vd["변동성"], brk))):
        compare(m, *var_rescale(vt, vd, f), "cur", "현재", vt, vd)
    # 앞·뒤 절반 각각의 분위수 손실 비율 (보고만 — 최근 급등락 구간에만 기대는지)
    for part, sl in (("앞 절반", slice(None, len(both["t"][1]) // 2)), ("뒤 절반", slice(len(both["t"][1]) // 2, None))):
        ql = {m: sum(_quantile_loss(d.iloc[sl], lv) for lv in (0.95, 0.99)) for m, (_, d) in both.items()}
        row[f"분위수 손실 비율 {part}"] = ql["fhs"] / ql["t"] if ql["t"] > 0 else np.nan
    half = len(vd) // 2
    v95, v99 = vt.loc["하루 VaR 95%"], vt.loc["하루 VaR 99%"]
    row.update({"VaR 검증 일수": int(v95["관측일"]),
                "VaR 95% 예외 비율": v95["예외 비율"], "VaR 95% p값": v95["종합 p값"],
                "VaR 99% 예외 비율": v99["예외 비율"], "VaR 99% p값": v99["종합 p값"],
                "VaR 95% 예외 (앞 절반)": float(vd["예외 95%"].iloc[:half].mean()),
                "VaR 95% 예외 (뒤 절반)": float(vd["예외 95%"].iloc[half:].mean()),
                "ES 95% 손실 비율": v95["예외일 손실 ÷ 예측 ES"], "ES 95% p값": v95["ES 검정 p값"],
                "ES 99% 손실 비율": v99["예외일 손실 ÷ 예측 ES"], "ES 99% p값": v99["ES 검정 p값"],
                "VaR 검정": "통과" if min(vt["종합 p값"]) > 0.05 else "기각",
                "ES 검정": "통과" if np.nanmin(vt["ES 검정 p값"].astype(float).fillna(1.0)) > 0.05 else "기각",
                "GARCH 재적합 실패": f"{int(v95['추정 실패'])}/{int(v95['재적합 수'])}"})
    # GARCH 최적화 진단 (앱 기본: 최근 3년, 자동): 시작값 셋이 같은 답에 닿는지, 계수가 경계에 붙었는지
    g = fit_vol_model(rr[-756:], "auto")
    fi = g.fit_info
    row["GARCH 진단"] = (f"{g.name} · 시작값 {fi.get('수렴', 0)}/{fi.get('시작값', 0)} 수렴 · 우도 차이 {fi.get('우도 차이', 0.0):.3f}"
                       + (f" · 경계: {', '.join(fi['경계'])}" if fi.get("경계") else ""))
    return row


def var_tail_rule(rows):
    """VaR 꼬리 분포 사전 등록 규칙 (EXPERIMENTS.md): 종목별 분위수 손실 비율(FHS ÷ t)의 중앙값 < 1 이고,
    FHS의 VaR·ES 검정 통과 종목 수가 t보다 줄지 않으면 FHS 채택. 반환: 판정에 쓴 숫자와 결론."""
    ok = [r for r in rows if "오류" not in r and np.isfinite(r.get("분위수 손실 비율 (FHS ÷ t)", np.nan))]
    if not ok:
        return None
    cnt = lambda m, k: sum(r[f"[{m}] {k}"] == "통과" for r in ok)
    ratio = float(np.median([r["분위수 손실 비율 (FHS ÷ t)"] for r in ok]))
    res = {"종목 수": len(ok), "분위수 손실 비율 중앙값": ratio,
           "FHS가 나은 종목": sum(r["분위수 손실 비율 (FHS ÷ t)"] < 1 for r in ok),
           "VaR 통과 t": cnt("t", "VaR 검정"), "VaR 통과 FHS": cnt("fhs", "VaR 검정"),
           "ES 통과 t": cnt("t", "ES 검정"), "ES 통과 FHS": cnt("fhs", "ES 검정")}
    res["FHS 채택"] = bool(ratio < 1 and res["VaR 통과 FHS"] >= res["VaR 통과 t"] and res["ES 통과 FHS"] >= res["ES 통과 t"])
    for part in ("앞 절반", "뒤 절반"):                   # 보고만 (판정에 쓰지 않음)
        v = [r[f"분위수 손실 비율 {part}"] for r in ok if np.isfinite(r.get(f"분위수 손실 비율 {part}", np.nan))]
        res[f"{part} 중앙값"] = float(np.median(v)) if v else np.nan
        res[f"{part} FHS가 나은 종목"] = int(sum(x < 1 for x in v))
    return res


def var_scale_rule(rows):
    """하루 VaR σ 보정 사전 등록 규칙 (EXPERIMENTS.md 2026-10-29).
    C2(크기 재보정, 모든 종목): 종목별 분위수 손실 비율(C2 ÷ 기본)의 중앙값이 전체·앞 절반·뒤 절반 모두 < 1이고
        VaR·ES 검정 통과 종목 수가 기본보다 줄지 않으면 채택.
    C1(VKOSPI 섞기, VKOSPI_TICKERS만): 두 종목 모두 비율이 전체·앞·뒤 절반에서 < 1이고 두 종목의 VaR·ES 통과 수가 줄지 않으면 채택.
    둘 다 채택이면 VKOSPI_TICKERS에는 두 종목 분위수 손실 합(전체)이 낮은 쪽. 반환: 판정에 쓴 숫자와 결론 (없으면 None)."""
    parts = ("", " 앞 절반", " 뒤 절반")
    ok = [r for r in rows if "오류" not in r and np.isfinite(r.get("분위수 손실 비율 (recal ÷ 기본)", np.nan))]
    if not ok:
        return None
    passed = lambda rs, m, k: sum(r[f"[{m}] {k}"] == "통과" for r in rs)
    res = {"종목 수": len(ok)}
    for p in parts:
        res[f"C2 비율 중앙값{p}"] = float(np.median([r[f"분위수 손실 비율 (recal ÷ 기본){p}"] for r in ok]))
    res["C2가 나은 종목"] = sum(r["분위수 손실 비율 (recal ÷ 기본)"] < 1 for r in ok)
    for k in ("VaR 검정", "ES 검정"):
        res[f"{k} 통과 기본"], res[f"{k} 통과 C2"] = passed(ok, "t", k), passed(ok, "recal", k)
    res["C2 채택"] = bool(all(res[f"C2 비율 중앙값{p}"] < 1 for p in parts)
                        and all(res[f"{k} 통과 C2"] >= res[f"{k} 통과 기본"] for k in ("VaR 검정", "ES 검정")))
    idx = [r for r in ok if r["종목"] in VKOSPI_TICKERS and np.isfinite(r.get("분위수 손실 비율 (vkospi ÷ 기본)", np.nan))]
    res["C1 대상"] = len(idx)
    for r in idx:
        for p in parts:
            res[f"C1 {r['종목']} 비율{p}"] = r[f"분위수 손실 비율 (vkospi ÷ 기본){p}"]
    for k in ("VaR 검정", "ES 검정"):
        res[f"C1 {k} 통과 기본"], res[f"C1 {k} 통과 C1"] = passed(idx, "t", k), passed(idx, "vkospi", k)
    res["C1 채택"] = bool(len(idx) == len(VKOSPI_TICKERS)
                        and all(r[f"분위수 손실 비율 (vkospi ÷ 기본){p}"] < 1 for r in idx for p in parts)
                        and all(res[f"C1 {k} 통과 C1"] >= res[f"C1 {k} 통과 기본"] for k in ("VaR 검정", "ES 검정")))
    loss = {m: sum(r[f"[{m}] 분위수 손실"] for r in idx) for m in ("recal", "vkospi")} if idx else {}
    if res["C1 채택"] and res["C2 채택"]:
        index_pick = min(loss, key=loss.get)
    else:
        index_pick = "vkospi" if res["C1 채택"] else "recal" if res["C2 채택"] else None
    res["코스피 200 종목에 쓸 것"] = VAR_SCALES[index_pick] if index_pick else "기본 그대로"
    res["다른 종목에 쓸 것"] = VAR_SCALES["recal"] if res["C2 채택"] else "기본 그대로"
    return res


def var_iv_lag_rule(rows):
    """C1L2(VKOSPI 하루 늦게 섞기) 사전 등록 규칙 (EXPERIMENTS.md 2026-10-30): C1 규칙과 같음 — 코스피 200 두 종목 모두
    분위수 손실 비율(C1L2 ÷ 기본)이 전체·앞·뒤 절반에서 < 1이고 두 종목의 VaR·ES 통과 수가 기본보다 줄지 않으면 채택.
    보고만: 분위수 손실 비율(C1L2 ÷ C1) — 하루 늦어져 잃는 정도. 반환: 판정에 쓴 숫자와 결론 (대상이 없으면 None)."""
    parts = ("", " 앞 절반", " 뒤 절반")
    ratio = "분위수 손실 비율 (vkospi2 ÷ 기본)"
    idx = [r for r in rows if "오류" not in r and r["종목"] in VKOSPI_TICKERS and np.isfinite(r.get(ratio, np.nan))]
    if not idx:
        return None
    passed = lambda m, k: sum(r[f"[{m}] {k}"] == "통과" for r in idx)
    res = {"대상": len(idx)}
    for r in idx:
        for p in parts:
            res[f"{r['종목']} 비율{p}"] = r[f"{ratio}{p}"]
        res[f"{r['종목']} C1L2 ÷ C1"] = r["[vkospi2] 분위수 손실"] / r["[vkospi] 분위수 손실"]
        for k in ("VaR 95% 예외", "VaR 99% 예외"):
            res[f"{r['종목']} {k} 기본 · C1 · C1L2"] = tuple(r[f"[{m}] {k}"] for m in ("t", "vkospi", "vkospi2"))
    for k in ("VaR 검정", "ES 검정"):
        res[f"{k} 통과 기본 → C1L2"] = (passed("t", k), passed("vkospi2", k))
    res["C1L2 채택"] = bool(len(idx) == len(VKOSPI_TICKERS)
                          and all(r[f"{ratio}{p}"] < 1 for r in idx for p in parts)
                          and all(b >= a for a, b in (res[f"{k} 통과 기본 → C1L2"] for k in ("VaR 검정", "ES 검정"))))
    return res


def _iv_lag_lines(rows):
    lr = var_iv_lag_rule(rows)
    if lr is None:
        return []
    lines = ["", "## 하루 VaR VKOSPI 하루 늦게 섞기(C1L2) — 사전 등록 규칙", "",
             "KRX는 그날 VKOSPI를 다음 영업일 아침에 주므로 실전 기록·앱은 C1이 쓰는 전날 값을 받지 못합니다. "
             "C1L2는 하루 더 지난(마지막 종가일의 전 거래일) VKOSPI를 반반 섞습니다. 기준은 GARCH만(기본). "
             "규칙은 [EXPERIMENTS.md](../EXPERIMENTS.md) (2026-10-30 사전 등록).", ""]
    for t in VKOSPI_TICKERS:
        if f"{t} 비율" in lr:
            ex = lambda k: " · ".join(_pct(v, False) for v in lr[f"{t} {k} 기본 · C1 · C1L2"])
            lines.append(f"- {INDEX_NAMES.get(t, t)}: 분위수 손실 비율(C1L2 ÷ 기본) 전체 {lr[f'{t} 비율']:.3f} · "
                         f"앞 {lr[f'{t} 비율 앞 절반']:.3f} · 뒤 {lr[f'{t} 비율 뒤 절반']:.3f}, C1L2 ÷ C1 {lr[f'{t} C1L2 ÷ C1']:.3f}, "
                         f"95% 예외 기본·C1·C1L2 {ex('VaR 95% 예외')}, 99% {ex('VaR 99% 예외')}")
    pc = lambda k: "{} → {}".format(*lr[f"{k} 통과 기본 → C1L2"])
    lines.append(f"- VaR 통과 {pc('VaR 검정')}, ES 통과 {pc('ES 검정')} → **{'C1L2 채택' if lr['C1L2 채택'] else 'C1L2 미채택'}**")
    return lines


VAR_GROUP_CANDS = {"k2": "K2", "m1": "M1", "r1": "R1"}     # 지금 기본에 곱하는 후보 (같은 규칙으로 따로 판정)
VAR_GROUP_BASE = {"k2": ("cur", "현재"), "m1": ("cur", "현재"), "r1": ("pre", "이전 기본")}   # 비교 기준 열과 비율 이름
R1_MIN_VALID = 0.8                                          # R1: 검증 기간에 유효한 일봉(시가·고가·저가)이 이보다 적으면 비교에서 뺌


def var_cluster_rule(rows, m="k2"):
    """지금 기본에 곱하는 후보(K2 전날 움직임별, M1 휴장 뒤 첫 거래일별, R1 장중 고가·저가 섞기)의 사전 등록 규칙 (EXPERIMENTS.md 2026-10-29).
    두 묶음을 따로 판정. 코스피 200 종목(VKOSPI_TICKERS): 두 종목 모두 분위수 손실 비율(후보 ÷ 기준: K2·M1은 현재, R1은 R1 이전 기본)이 전체·앞·뒤 절반에서 < 1이고
    VaR·ES 통과 수 유지. 나머지 종목: 비율 중앙값이 전체·앞·뒤 절반 모두 < 1이고 VaR·ES 통과 종목 수 유지.
    반환: 판정에 쓴 숫자와 결론 (없으면 None)."""
    name = VAR_GROUP_CANDS[m]
    base, label = VAR_GROUP_BASE[m]
    ratio = f"분위수 손실 비율 ({m} ÷ {label})"
    parts = ("", " 앞 절반", " 뒤 절반")
    ok = [r for r in rows if "오류" not in r and np.isfinite(r.get(ratio, np.nan))]
    if not ok:
        return None
    passed = lambda rs, mm, k: sum(r[f"[{mm}] {k}"] == "통과" for r in rs)
    idx = [r for r in ok if r["종목"] in VKOSPI_TICKERS]
    rest = [r for r in ok if r["종목"] not in VKOSPI_TICKERS]
    res = {"후보": name, "코스피 200 종목 수": len(idx), "나머지 종목 수": len(rest),
           "비교 못 한 종목": [r["종목"] for r in rows if "오류" not in r and not np.isfinite(r.get(ratio, np.nan))]}
    for r in idx:
        for p in parts:
            res[f"{r['종목']} 비율{p}"] = r[f"{ratio}{p}"]
        res[f"{r['종목']} 99% 독립성 p 현재 → {name}"] = f"{r[f'[{base}] 99% 독립성 p']:.3f} → {r[f'[{m}] 99% 독립성 p']:.3f}"
    for grp, rs in (("코스피 200", idx), ("나머지", rest)):
        for k in ("VaR 검정", "ES 검정"):
            res[f"{grp} {k} 통과 현재 → {name}"] = (passed(rs, base, k), passed(rs, m, k))
        brk = [[v for r in rs if np.isfinite(v := r.get(f"[{mm}] 휴장 뒤 95% 예외", np.nan))] for mm in (base, m)]
        res[f"{grp} 휴장 뒤 95% 예외 평균 현재 → {name}"] = tuple(float(np.mean(v)) if v else np.nan for v in brk)
    keep = lambda grp: all(res[f"{grp} {k} 통과 현재 → {name}"][1] >= res[f"{grp} {k} 통과 현재 → {name}"][0]
                           for k in ("VaR 검정", "ES 검정"))
    res[f"코스피 200 {name} 채택"] = bool(len(idx) == len(VKOSPI_TICKERS) and keep("코스피 200")
                                       and all(r[f"{ratio}{p}"] < 1 for r in idx for p in parts))
    for p in parts:
        res[f"나머지 비율 중앙값{p}"] = float(np.median([r[f"{ratio}{p}"] for r in rest])) if rest else np.nan
    res[f"나머지 {name}가 나은 종목"] = sum(r[ratio] < 1 for r in rest)
    res[f"나머지 {name} 채택"] = bool(rest and keep("나머지") and all(res[f"나머지 비율 중앙값{p}"] < 1 for p in parts))
    return res


VAR_GROUP_TEXT = {
    "k2": ("하루 VaR 예외 몰림: 전날 움직임별 σ 재보정(K2)",
           "전날 표준화 수익률 z로 세 구간(< −1, −1 ~ 1, > 1)을 나누고, 그날 전까지 같은 구간이었던 날들의 z² 평균의 제곱근(0.8~1.5)을 "
           "지금 기본의 σ에 곱합니다 (큰 하락 다음 날 σ가 덜 반응하는 만큼)."),
    "m1": ("하루 VaR 휴장 뒤 첫 거래일(월요일 등) 보정(M1)",
           "그날이 휴장 뒤 첫 거래일(바로 전 거래일과 달력상 2일 넘게 떨어짐: 월요일·연휴 다음 날)인지로 두 묶음을 나누고, "
           "그날 전까지 같은 묶음이었던 날들의 z² 평균의 제곱근(0.8~1.5)을 지금 기본의 σ에 곱합니다 (쉬는 동안 쌓인 소식만큼)."),
    "r1": ("하루 VaR 장중 고가·저가 변동성 섞기(R1)",
           "하루 실현 분산 = 밤사이 갭² + Rogers–Satchell 장중 분산(시가·고가·저가·종가), 지수가중 평균(0.94)을 내어 "
           "σ² = ½ 지금 기본 + ½ 장중 실현 분산으로 섞습니다 (종가만으로는 모르는 장중 흔들림). "
           f"유효한 일봉이 {R1_MIN_VALID:.0%} 미만인 종목은 비교에서 뺍니다.")}


def _group_rule_lines(rows, m):
    cr = var_cluster_rule(rows, m)
    if not cr:
        return []
    name = cr["후보"]
    title, desc = VAR_GROUP_TEXT[m]
    lines = ["", f"## {title} — 사전 등록 규칙", "", desc + " 규칙은 [EXPERIMENTS.md](../EXPERIMENTS.md) (2026-10-29 사전 등록).", ""]
    for t in VKOSPI_TICKERS:
        if f"{t} 비율" in cr:
            lines.append(f"- {INDEX_NAMES.get(t, t)}: 분위수 손실 비율({name} ÷ {VAR_GROUP_BASE[m][1]}) 전체 {cr[f'{t} 비율']:.3f} · 앞 {cr[f'{t} 비율 앞 절반']:.3f} · "
                         f"뒤 {cr[f'{t} 비율 뒤 절반']:.3f}, 99% 독립성 p {cr[f'{t} 99% 독립성 p 현재 → {name}']}")
    ci = lambda grp, k: "{} → {}".format(*cr[f"{grp} {k} 통과 현재 → {name}"])
    bk = lambda grp: "{} → {}".format(*(_pct(v, False) for v in cr[f"{grp} 휴장 뒤 95% 예외 평균 현재 → {name}"]))
    lines += [f"- 코스피 200 종목 VaR 통과 {ci('코스피 200', 'VaR 검정')}, ES 통과 {ci('코스피 200', 'ES 검정')}, "
              f"휴장 뒤 첫 거래일 95% 예외 평균 {bk('코스피 200')} → **{f'{name} 채택' if cr[f'코스피 200 {name} 채택'] else f'{name} 미채택'}**",
              f"- 나머지 {cr['나머지 종목 수']}종목: 비율 중앙값 전체 {_num(cr['나머지 비율 중앙값'], '{:.3f}')} · "
              f"앞 {_num(cr['나머지 비율 중앙값 앞 절반'], '{:.3f}')} · 뒤 {_num(cr['나머지 비율 중앙값 뒤 절반'], '{:.3f}')} "
              f"(나은 종목 {cr[f'나머지 {name}가 나은 종목']}), VaR 통과 {ci('나머지', 'VaR 검정')}, ES 통과 {ci('나머지', 'ES 검정')}, "
              f"휴장 뒤 95% 예외 평균 {bk('나머지')} → **{f'{name} 채택' if cr[f'나머지 {name} 채택'] else f'{name} 미채택'}**"]
    if cr["비교 못 한 종목"]:
        lines.append(f"- 비교 못 한 종목 (데이터 부족·오류): {', '.join(cr['비교 못 한 종목'])}")
    return lines


def ranking_summary(rank_dir=uv.RANK_DIR):
    """코스피 종목 고르기 (매주 계산 결과, 앱 기본: 교체 63거래일·다음 날 체결·세금 반영)의 요약. 없으면 None."""
    w = uv.load_weekly(Path(rank_dir) / uv.RESULTS_PATH.name)
    if w is None or (63, 1, True) not in w["results"]:
        return None
    U = w["results"][(63, 1, True)]
    a = U["table"].loc["auto"]
    fx, sel = U.get("factors") or {}, U.get("selection") or {}
    picks = uv.picks_summary(uv.read_picks(Path(rank_dir) / uv.PICKS_PATH.name))
    rw, cs = U.get("risk_weighted"), U.get("cost_sens")
    return {"데이터": f"{w['data_end'].date()}까지", "리밸런싱 수": a["리밸런싱 수"],
            "상장폐지 포함": f"예 ({U.get('n_delisted', 0)}종목)" if U.get("krx") else "아니오",
            "자동 선택 바뀐 횟수": U.get("switches", np.nan),
            "손익분기 비용(bp)": cs[1] if cs is not None else np.nan,
            "변동성 역가중 초과수익(연)": rw["상위20% 초과수익(연, 비용후)"] if rw is not None else np.nan,
            "변동성 역가중 t값": rw["초과수익 t값"] if rw is not None else np.nan,
            "시장 구간별": U.get("market"), "최근1년 제외": U.get("ex_recent"),
            "리밸런싱 날짜": {h: (w["results"].get((h, 1, True)) or {}).get("offsets") for h in uv.WEEKLY_HOLDS},
            "기준 전략": {uv.MODEL_LABELS[m]: U["table"].loc[m, "상위20% 초과수익(연, 비용후)"]
                       for m in ("momentum", *uv.BASELINES) if m in U["table"].index},
            "초과수익(연, 비용후)": a["상위20% 초과수익(연, 비용후)"], "초과수익 t값": a["초과수익 t값"],
            "평균 IC": a["평균 IC"], "IC t값": a["IC t값"], "최대 낙폭(비용후)": a["최대 낙폭(비용후)"],
            "요인 알파(연)": fx.get("alpha", np.nan), "요인 알파 t값": fx.get("alpha_t", np.nan),
            "1등 착시 보정(DSR)": sel.get("dsr", np.nan), "실전 기록 채점 수(21일)": picks.loc["21거래일", "채점 수"],
            "실전 기록 평균 초과수익(21일)": picks.loc["21거래일", "평균 초과수익"],
            "경제적 가치": "있음" if a["초과수익 t값"] > 2 and a["IC t값"] > 2 else "확인 안 됨"}


def _pct(v, signed=True):
    return "—" if v is None or not np.isfinite(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")


def _num(v, fmt="{:+.2f}"):
    return "—" if v is None or not np.isfinite(v) else fmt.format(v)


def report_markdown(rows, rank=None, when="", version="", buckets=None):
    """검증 결과 → validation/README.md 내용. buckets: 전 종목 합산 확률 구간 표 (direction.bucket_table)."""
    ok = [r for r in rows if "오류" not in r]
    lines = [
        "# 실제 데이터 검증 결과",
        "",
        f"자동 생성: {when} · 코드 {version} · `python validation.py` (GitHub Actions `validation`, 매주 토요일).",
        "설정은 앱 기본값: 방향 예측 21거래일, 판단 다음 날 종가 체결, 수수료·체결 비용 3bp(살 때·팔 때 각각) + 증권거래세(체결일 세율).",
        "모든 예측은 그 시점까지의 데이터로만 학습·선택한 워크포워드 결과이고, 과거 성과가 미래를 보장하지 않습니다.",
        "",
        "## 판정 기준",
        "",
        "| 항목 | '있음'/'통과' 조건 |",
        "|---|---|",
        "| 방향 예측력 | 확률 정확도 개선 > 0 그리고 겹침 보정 z > 2 (21거래일 예측을 21거래일 결과로 채점) |",
        f"| 전략의 경제적 가치 | 비용·세금 뺀 일간 초과수익(전략 − 단순 보유)의 블록 부트스트랩(블록 {BLOCK}거래일) 95% 신뢰구간 하한 > 0 |",
        "| VaR 검정 | 하루 VaR 95%·99% 모두 Kupiec·Christoffersen 종합 p값 > 0.05 |",
        "| ES 검정 | 예외일의 (실제 손실 − 예측 ES) ÷ 변동성 평균이 0보다 크지 않음 (McNeil·Frey 단측 p값 > 0.05, 95%·99% 모두) |",
        "| 코스피 종목 고르기 | 비용 후 초과수익 t > 2 그리고 IC t > 2 |",
        "",
        "샤프 비율 하나만으로 판단하지 않습니다. 신뢰구간, 가정을 바꿔도 유지되는지(7개: 자동 선택 기간 4 + 시기 3), "
        "여러 설정 중 1등을 고른 착시 보정(DSR, 0.95 이상이면 우연일 가능성 낮음)을 함께 봅니다.",
        "",
        "## 요약",
        "",
        f"- 방향 예측력 있음: {sum(r['방향 예측력'] == '있음' for r in ok)} / {len(ok)}종목",
        f"- 방향 전략 경제적 가치 있음: {sum(r['방향 전략 경제적 가치'] == '있음' for r in ok)} / {len(ok)}종목 (매일 판단), "
        f"{sum(r['분할 보유 경제적 가치'] == '있음' for r in ok)} / {len(ok)}종목 (21일 분할 보유)",
        f"- 국면 전략 경제적 가치 있음: {sum(r['국면 전략 경제적 가치'] == '있음' for r in ok)} / {len(ok)}종목",
        f"- VaR 검정 통과: {sum(r['VaR 검정'] == '통과' for r in ok)} / {len(ok)}종목, "
        f"ES 검정 통과: {sum(r['ES 검정'] == '통과' for r in ok)} / {len(ok)}종목",
    ]
    if rank:
        lines.append(f"- 코스피 종목 고르기 경제적 가치: {rank['경제적 가치']}")
    lines += ["", "## 방향 예측 (21거래일)", "",
              "| 종목 | 검증 기간 | 적중률 | 항상 오른다 | z | 확률 정확도 | 예측력 |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {r['검증 기간']} | {_pct(r['방향 적중률'], False)} | {_pct(r['항상 오른다 적중률'], False)} | "
                     f"{_num(r['방향 z'], '{:+.1f}')} | {_pct(r['확률 정확도 개선'])} | {r['방향 예측력']} |")
    lines += ["", "## 방향 전략 (매일 판단, 다음 하루 보유) vs 단순 보유 (비용·세금 후)", "",
              "| 종목 | 전략 연 수익률 | 샤프 | 최대 낙폭 | 단순 보유 연 수익률 | 샤프 | 최대 낙폭 | 초과(연) [95% CI] | 가정 바꿔 통과 | DSR | 경제적 가치 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['방향 전략 연 수익률'])} | {_num(r['방향 전략 샤프'])} | {_pct(r['방향 전략 최대 낙폭'])} | "
                     f"{_pct(r['단순 보유 연 수익률'])} | {_num(r['단순 보유 샤프'])} | {_pct(r['단순 보유 최대 낙폭'])} | "
                     f"{_pct(r['방향 초과수익(연)'])} [{_pct(r['방향 초과 95% 하한'])}, {_pct(r['방향 초과 95% 상한'])}] | "
                     f"{r['가정 바꿔 통과(7개 중)']}/7 | {_num(r['방향 DSR'], '{:.0%}')} | {r['방향 전략 경제적 가치']} |")
    if buckets is not None and len(buckets):
        lines += ["", "## 예측 확률 구간별 실제 21일 수익 (전 종목 합산)", "",
                  "상승 확률이 50%를 넘어도 오를 때 조금 벌고 내릴 때 크게 잃으면 기대수익은 음수입니다. "
                  "확률이 맞는지(평균 예측 확률 vs 실제 상승 비율)와 함께 수익의 크기를 봅니다. "
                  "수익은 예측 대상 기간(판단일 종가 → 21거래일 뒤 종가), 비용은 한 번 사서 파는 수수료·세금. "
                  "매일 예측이라 21일 결과가 겹쳐서 관측 수는 독립 표본 수가 아닙니다.", "",
                  "| 확률 구간 | 관측 수 | 평균 예측 확률 | 실제 상승 비율 | 평균 수익 | 오를 때 평균 | 내릴 때 평균 | 비용 뺀 평균 |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, b in buckets.iterrows():
            lines.append(f"| {k} | {b['관측 수']:,.0f} | {_pct(b['평균 예측 확률'], False)} | {_pct(b['실제 상승 비율'], False)} | "
                         f"{_pct(b['평균 수익'])} | {_pct(b['오를 때 평균'])} | {_pct(b['내릴 때 평균'])} | {_pct(b['비용 뺀 평균'])} |")
    lines += ["", "## 21일 분할 보유 전략 vs 단순 보유 (비용·세금 후)", "",
              "매일 자본의 1/21을 그날 신호(상승 확률 > 50%)대로 사서 21거래일 들고 있습니다 — 21일 예측을 그 기간 그대로 매매에 쓴 결과입니다.", "",
              "| 종목 | 전략 연 수익률 | 샤프 | 최대 낙폭 | 평균 보유 비중 | 초과(연) [95% CI] | 경제적 가치 |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['분할 보유 연 수익률'])} | {_num(r['분할 보유 샤프'])} | {_pct(r['분할 보유 최대 낙폭'])} | "
                     f"{_pct(r['분할 보유 평균 비중'], False)} | {_pct(r['분할 초과수익(연)'])} "
                     f"[{_pct(r['분할 초과 95% 하한'])}, {_pct(r['분할 초과 95% 상한'])}] | {r['분할 보유 경제적 가치']} |")
    lines += ["", "## 시장 국면 전략 ('불안 국면엔 현금') vs 단순 보유 (비용·세금 후)", "",
              "| 종목 | 전략 연 수익률 | 샤프 | 최대 낙폭 | 보유 비중 | 초과(연) [95% CI] | 경제적 가치 |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['국면 전략 연 수익률'])} | {_num(r['국면 전략 샤프'])} | {_pct(r['국면 전략 최대 낙폭'])} | "
                     f"{_pct(r['국면 전략 보유 비중'], False)} | {_pct(r['국면 초과수익(연)'])} "
                     f"[{_pct(r['국면 초과 95% 하한'])}, {_pct(r['국면 초과 95% 상한'])}] | {r['국면 전략 경제적 가치']} |")
    lines += ["", "## 하루 VaR·ES 검정 (첫 3년 학습 뒤 전체 기간)", "",
              "예외 = 실제 손실이 VaR보다 컸던 날. 손실 비율 = 예외일의 실제 손실 ÷ 예측 ES (1에 가까워야 하고, 크면 꼬리 손실을 작게 잡은 것). "
              "앞·뒤 절반 = 검증 기간을 반으로 나눈 95% 예외 비율 (시기에 따라 흔들리는지). "
              "앱·기록과 같이 코스피 지수·KODEX 200은 GARCH와 전날 VKOSPI를 반반 섞은 σ(이름 옆 *), 나머지 종목은 GARCH와 "
              "장중 고가·저가로 잰 최근 실현 분산을 반반 섞은 σ(이름 옆 †)로 계산합니다 (2026-10-29 판정).", "",
              "| 종목 | 검증 일수 | 95% 예외 (목표 5%) | p값 | 99% 예외 (목표 1%) | p값 | VaR 결과 | ES 손실 비율 95% · 99% | ES p값 95% · 99% | ES 결과 | 95% 예외 앞·뒤 절반 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']}{ {'VKOSPI 섞기': ' *', '장중 고가·저가 섞기': ' †'}.get(r.get('VaR σ 보정'), '') } | {r['VaR 검증 일수']:,} | {_pct(r['VaR 95% 예외 비율'], False)} | {_num(r['VaR 95% p값'], '{:.2f}')} | "
                     f"{_pct(r['VaR 99% 예외 비율'], False)} | {_num(r['VaR 99% p값'], '{:.2f}')} | {r['VaR 검정']} | "
                     f"{_num(r['ES 95% 손실 비율'], '{:.2f}')} · {_num(r['ES 99% 손실 비율'], '{:.2f}')} | "
                     f"{_num(r['ES 95% p값'], '{:.2f}')} · {_num(r['ES 99% p값'], '{:.2f}')} | {r['ES 검정']} | "
                     f"{_pct(r['VaR 95% 예외 (앞 절반)'], False)} · {_pct(r['VaR 95% 예외 (뒤 절반)'], False)} |")
    rule = var_tail_rule(rows)
    if rule:
        lines += ["", "## 하루 VaR 꼬리 분포 비교: t분포 vs 실제 잔차(FHS) — 사전 등록 규칙", "",
                  f"지금 기본: {VAR_INNOVS[VAR_INNOV]}. 규칙([EXPERIMENTS.md](../EXPERIMENTS.md)): 종목별 분위수 손실 비율(FHS ÷ t, 95%+99%)의 "
                  "중앙값이 1보다 작고, FHS의 VaR·ES 검정 통과 종목 수가 t보다 줄지 않으면 FHS로 바꿉니다.", "",
                  f"- 분위수 손실 비율 중앙값: {rule['분위수 손실 비율 중앙값']:.3f} (FHS가 나은 종목 {rule['FHS가 나은 종목']} / {rule['종목 수']})",
                  f"- VaR 검정 통과: t {rule['VaR 통과 t']} → FHS {rule['VaR 통과 FHS']} · ES 검정 통과: t {rule['ES 통과 t']} → FHS {rule['ES 통과 FHS']}",
                  f"- 판정: **{'FHS 채택 조건 충족' if rule['FHS 채택'] else 'FHS 채택 조건 미충족 → t 유지'}**",
                  f"- (보고만) 앞 절반: 비율 중앙값 {_num(rule['앞 절반 중앙값'], '{:.3f}')}, FHS가 나은 종목 {rule['앞 절반 FHS가 나은 종목']} · "
                  f"뒤 절반: {_num(rule['뒤 절반 중앙값'], '{:.3f}')}, {rule['뒤 절반 FHS가 나은 종목']}종목 — 최근 급등락 구간에서만 나은지 확인", "",
                  "| 종목 | 95% 예외 t · FHS | 99% 예외 t · FHS | VaR 검정 t · FHS | ES 검정 t · FHS | 분위수 손실 비율 (FHS ÷ t) | 앞 절반 · 뒤 절반 |",
                  "|---|---|---|---|---|---|---|"]
        for r in ok:
            if "[fhs] VaR 검정" not in r:
                continue
            lines.append(f"| {r['이름']} | {_pct(r['[t] VaR 95% 예외'], False)} · {_pct(r['[fhs] VaR 95% 예외'], False)} | "
                         f"{_pct(r['[t] VaR 99% 예외'], False)} · {_pct(r['[fhs] VaR 99% 예외'], False)} | "
                         f"{r['[t] VaR 검정']} · {r['[fhs] VaR 검정']} | {r['[t] ES 검정']} · {r['[fhs] ES 검정']} | "
                         f"{_num(r['분위수 손실 비율 (FHS ÷ t)'], '{:.3f}')} | "
                         f"{_num(r.get('분위수 손실 비율 앞 절반', np.nan), '{:.3f}')} · {_num(r.get('분위수 손실 비율 뒤 절반', np.nan), '{:.3f}')} |")
    sr = var_scale_rule(rows)
    if sr:
        lines += ["", "## 하루 VaR σ 보정: VKOSPI 섞기(C1)·크기 재보정(C2) — 사전 등록 규칙", "",
                  "같은 GARCH 예측의 σ에 날짜별 배수만 곱합니다 (꼬리 분포는 그대로). C1은 σ² = ½ GARCH + ½ (전날 VKOSPI)², "
                  "코스피 지수·KODEX 200만. C2는 지난 250일 z = r/σ 제곱 평균의 제곱근(0.8~1.5), 모든 종목. "
                  "규칙은 [EXPERIMENTS.md](../EXPERIMENTS.md) (2026-10-29 사전 등록).", "",
                  f"- C2 분위수 손실 비율(C2 ÷ 기본) 중앙값: 전체 {sr['C2 비율 중앙값']:.3f} · 앞 절반 {sr['C2 비율 중앙값 앞 절반']:.3f} · "
                  f"뒤 절반 {sr['C2 비율 중앙값 뒤 절반']:.3f} (C2가 나은 종목 {sr['C2가 나은 종목']} / {sr['종목 수']})",
                  f"- C2 VaR 검정 통과: 기본 {sr['VaR 검정 통과 기본']} → C2 {sr['VaR 검정 통과 C2']} · "
                  f"ES 검정 통과: 기본 {sr['ES 검정 통과 기본']} → C2 {sr['ES 검정 통과 C2']} → **{'C2 채택' if sr['C2 채택'] else 'C2 미채택'}**",
                  f"- C1 (대상 {sr['C1 대상']}종목): VaR 통과 {sr['C1 VaR 검정 통과 기본']} → {sr['C1 VaR 검정 통과 C1']}, "
                  f"ES 통과 {sr['C1 ES 검정 통과 기본']} → {sr['C1 ES 검정 통과 C1']} → **{'C1 채택' if sr['C1 채택'] else 'C1 미채택'}**",
                  f"- 결론: 코스피 지수·KODEX 200 → **{sr['코스피 200 종목에 쓸 것']}**, 다른 종목 → **{sr['다른 종목에 쓸 것']}**", "",
                  "| 종목 | 95% 예외 기본 · C2 · C1 | 99% 예외 기본 · C2 · C1 | VaR 검정 기본 · C2 · C1 | ES 검정 기본 · C2 · C1 | "
                  "분위수 손실 비율 C2 (전체 · 앞 · 뒤) | C1 (전체 · 앞 · 뒤) |",
                  "|---|---|---|---|---|---|---|"]
        three = lambda r, k, f: " · ".join(f(r[f"[{m}] {k}"]) if f"[{m}] {k}" in r else "—" for m in ("t", "recal", "vkospi"))
        ratios = lambda r, m: " · ".join(_num(r.get(f"분위수 손실 비율 ({m} ÷ 기본){p}", np.nan), "{:.3f}") for p in ("", " 앞 절반", " 뒤 절반"))
        for r in ok:
            if "[recal] VaR 검정" not in r:
                continue
            lines.append(f"| {r['이름']} | {three(r, 'VaR 95% 예외', lambda v: _pct(v, False))} | "
                         f"{three(r, 'VaR 99% 예외', lambda v: _pct(v, False))} | {three(r, 'VaR 검정', str)} | {three(r, 'ES 검정', str)} | "
                         f"{ratios(r, 'recal')} | {ratios(r, 'vkospi') if '[vkospi] VaR 검정' in r else '—'} |")
    lines += _iv_lag_lines(rows)
    for m in VAR_GROUP_CANDS:
        lines += _group_rule_lines(rows, m)
    rec = [r for r in ok if "[최근1년 제외] 방향 z" in r]
    if rec:
        cnt = lambda k: sum(r[f"[최근1년 제외] {k} 하한"] > 0 for r in rec)
        lines += ["", "## 최근 1년(이례적 급등락 구간)을 뺀 결과 — 보고만", "",
                  "데이터 마지막 날에서 252거래일 전까지만 있었다고 보고 같은 계산을 했습니다 (그 뒤 결과가 필요한 예측·보유는 채점하지 않음). "
                  "위의 전체 기간 결과와 크게 다르면 그 결과는 최근 구간에 기댄 것입니다. 판정 기준은 위와 같습니다.", "",
                  f"- 방향 예측력 있음: {sum(r['[최근1년 제외] 방향 예측력'] == '있음' for r in rec)} / {len(rec)}종목 "
                  f"(전체 기간: {sum(r['방향 예측력'] == '있음' for r in rec)})",
                  f"- 경제적 가치 있음(95% 구간 하한 > 0): 방향 {cnt('방향')}, 21일 분할 보유 {cnt('분할')}, 국면 {cnt('국면')} / {len(rec)}종목", "",
                  "| 종목 | 끝 | z (전체 → 제외) | 방향 초과(연) [95%] | 분할 보유 초과(연) [95%] | 국면 초과(연) [95%] |", "|---|---|---|---|---|---|"]
        for r in rec:
            f3 = lambda k: (f"{_pct(r[f'[최근1년 제외] {k} 초과'])} [{_pct(r[f'[최근1년 제외] {k} 하한'])}, "
                            f"{_pct(r[f'[최근1년 제외] {k} 상한'])}]")
            lines.append(f"| {r['이름']} | {r['[최근1년 제외] 끝']} | {_num(r['방향 z'], '{:+.1f}')} → {_num(r['[최근1년 제외] 방향 z'], '{:+.1f}')} | "
                         f"{f3('방향')} | {f3('분할')} | {f3('국면')} |")
    if rank:
        lines += ["", "## 코스피 종목 고르기 (상위 20%, 교체 63거래일, 비용·세금 후, 전체 평균 대비)", "",
                  "| 데이터 | 리밸런싱 | 초과수익(연) | t | 평균 IC | IC t | 최대 낙폭 | 요인 알파(연) | 알파 t | DSR | 실전 기록 (21일 채점 수 · 평균 초과) | 경제적 가치 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|",
                  f"| {rank['데이터']} | {rank['리밸런싱 수']:.0f} | {_pct(rank['초과수익(연, 비용후)'])} | {_num(rank['초과수익 t값'], '{:+.1f}')} | "
                  f"{_num(rank['평균 IC'], '{:+.3f}')} | {_num(rank['IC t값'], '{:+.1f}')} | {_pct(rank['최대 낙폭(비용후)'])} | "
                  f"{_pct(rank['요인 알파(연)'])} | {_num(rank['요인 알파 t값'], '{:+.1f}')} | {_num(rank['1등 착시 보정(DSR)'], '{:.0%}')} | "
                  f"{rank['실전 기록 채점 수(21일)']:.0f}번 · {_pct(rank['실전 기록 평균 초과수익(21일)'])} | {rank['경제적 가치']} |",
                  "",
                  f"- 상장폐지 종목 포함: {rank['상장폐지 포함']} · 자동 선택이 바뀐 횟수: {_num(rank['자동 선택 바뀐 횟수'], '{:.0f}')}번",
                  f"- 손익분기 비용: 매수·매도 각각 {_num(rank['손익분기 비용(bp)'], '{:.0f}')}bp (이보다 비용·슬리피지가 크면 초과수익 0 이하)",
                  f"- 위험 반영 비중(변동성 역가중): 초과수익 {_pct(rank['변동성 역가중 초과수익(연)'])} (t {_num(rank['변동성 역가중 t값'], '{:+.1f}')})",
                  "- 같은 조건의 기준 전략 초과수익(연): " + (", ".join(f"{k} {_pct(v)}" for k, v in rank["기준 전략"].items()) or "—")]
        xr = rank.get("최근1년 제외")
        if xr:
            lines.append(f"- 최근 1년을 뺀 결과 (보고만, {xr['끝'].date()}까지 끝난 리밸런싱 {xr['리밸런싱 수']}번): 초과수익 {_pct(xr['초과수익(연)'])} "
                         f"(t {_num(xr['초과수익 t값'], '{:+.1f}')}), 평균 IC {_num(xr['평균 IC'], '{:+.3f}')} (t {_num(xr['IC t값'], '{:+.1f}')})")
        for h, ofs in (rank.get("리밸런싱 날짜") or {}).items():
            if ofs:
                lines.append(f"- 리밸런싱 날짜를 바꾸면 (교체 {h}거래일, 시작일 {len(ofs['table'])}가지, 보고만): " + "; ".join(
                    f"{uv.MODEL_LABELS[m].replace('⭐ ', '')} 초과수익 평균 {_pct(sm['평균'])} ({_pct(sm['최소'])} ~ {_pct(sm['최대'])}), "
                    f"0보다 큰 경우 {sm['0보다 큰 수']}/{sm['시작일 수']}" for m, sm in ofs["summary"].items()))
        mk = rank.get("시장 구간별")
        if mk is not None:
            lines += ["", "| 시장 구간 (리밸런싱 기간의 범위 평균 수익률 3분위) | 기간 수 | 시장 평균 | 초과수익 평균 | 초과 > 0 비율 |", "|---|---|---|---|---|"]
            lines += [f"| {k} | {b['기간 수']:.0f} | {_pct(b['시장 평균(기간)'])} | {_pct(b['초과수익 평균(기간)'])} | {_pct(b['초과 > 0 비율'], False)} |"
                      for k, b in mk.iterrows()]
    bad = [r for r in rows if "오류" in r]
    if bad:
        lines += ["", "## 계산하지 못한 종목", ""] + [f"- {r['종목']}: {r['오류']}" for r in bad]
    lines += ["", "전체 숫자는 [summary.csv](summary.csv)에 있습니다.", ""]
    return "\n".join(lines)


def run(tickers, out_dir=OUT_DIR, say=print):
    rows, obs = [], []
    for t in tickers:
        t0 = time.time()
        try:
            r = validate(t)
            obs.append(r.pop("_obs"))
            rows.append(r)
            say(f"{t} ({r['이름']}): 완료 ({time.time() - t0:.0f}초) · 검증 기간 {r['검증 기간']}\n"
                f"  대체 데이터: {r['대체 데이터'] or '없음'}\n"
                f"  방향 예측: 적중률 {r['방향 적중률']:.1%} (항상 오른다 {r['항상 오른다 적중률']:.1%}), z {r['방향 z']:+.1f}, "
                f"확률 정확도 {r['확률 정확도 개선']:+.1%} → 예측력 {r['방향 예측력']}\n"
                f"  방향 전략: 연 {r['방향 전략 연 수익률']:+.1%} 샤프 {r['방향 전략 샤프']:.2f} 낙폭 {r['방향 전략 최대 낙폭']:.0%} | "
                f"단순 보유: 연 {r['단순 보유 연 수익률']:+.1%} 샤프 {r['단순 보유 샤프']:.2f} 낙폭 {r['단순 보유 최대 낙폭']:.0%} | "
                f"초과 {r['방향 초과수익(연)']:+.1%} [{r['방향 초과 95% 하한']:+.1%}, {r['방향 초과 95% 상한']:+.1%}] "
                f"가정 바꿔 {r['가정 바꿔 통과(7개 중)']}/7 → {r['방향 전략 경제적 가치']}\n"
                f"  21일 분할 보유: 연 {r['분할 보유 연 수익률']:+.1%} 샤프 {r['분할 보유 샤프']:.2f} 낙폭 {r['분할 보유 최대 낙폭']:.0%} | "
                f"초과 {r['분할 초과수익(연)']:+.1%} [{r['분할 초과 95% 하한']:+.1%}, {r['분할 초과 95% 상한']:+.1%}] → {r['분할 보유 경제적 가치']}\n"
                f"  국면 전략: 연 {r['국면 전략 연 수익률']:+.1%} 샤프 {r['국면 전략 샤프']:.2f} 낙폭 {r['국면 전략 최대 낙폭']:.0%} | "
                f"초과 {r['국면 초과수익(연)']:+.1%} [{r['국면 초과 95% 하한']:+.1%}, {r['국면 초과 95% 상한']:+.1%}] → {r['국면 전략 경제적 가치']}\n"
                f"  하루 VaR: 95% 예외 {r['VaR 95% 예외 비율']:.1%} (p {r['VaR 95% p값']:.2f}), "
                f"99% 예외 {r['VaR 99% 예외 비율']:.1%} (p {r['VaR 99% p값']:.2f}) → {r['VaR 검정']} ({r['VaR 검증 일수']:,}일)\n"
                f"  ES: 손실 비율 95% {r['ES 95% 손실 비율']:.2f}·99% {r['ES 99% 손실 비율']:.2f} "
                f"(p {r['ES 95% p값']:.2f}·{r['ES 99% p값']:.2f}) → {r['ES 검정']} | GARCH 재적합 실패 {r['GARCH 재적합 실패']} | {r['GARCH 진단']}\n"
                f"  꼬리 분포 비교: 95% 예외 t {r['[t] VaR 95% 예외']:.1%}·FHS {r['[fhs] VaR 95% 예외']:.1%}, "
                f"99% 예외 t {r['[t] VaR 99% 예외']:.1%}·FHS {r['[fhs] VaR 99% 예외']:.1%}, VaR {r['[t] VaR 검정']}·{r['[fhs] VaR 검정']}, "
                f"ES {r['[t] ES 검정']}·{r['[fhs] ES 검정']}, 분위수 손실 비율 {r['분위수 손실 비율 (FHS ÷ t)']:.3f} "
                f"(앞 {r.get('분위수 손실 비율 앞 절반', np.nan):.3f}·뒤 {r.get('분위수 손실 비율 뒤 절반', np.nan):.3f})\n"
                f"  σ 보정 후보: " + ", ".join(
                    f"{m} 95% 예외 {r[f'[{m}] VaR 95% 예외']:.1%}·VaR {r[f'[{m}] VaR 검정']}·ES {r[f'[{m}] ES 검정']}·"
                    f"분위수 손실 비율 {r[f'분위수 손실 비율 ({m} ÷ 기본)']:.3f} "
                    f"(앞 {r[f'분위수 손실 비율 ({m} ÷ 기본) 앞 절반']:.3f}·뒤 {r[f'분위수 손실 비율 ({m} ÷ 기본) 뒤 절반']:.3f})"
                    for m in VAR_SCALES if f"[{m}] VaR 검정" in r)
                + (f" | VKOSPI 오류: {r['[vkospi] 오류']}" if "[vkospi] 오류" in r else "")
                + "".join(f"\n  후보 {n} ({VAR_GROUP_BASE[m][1]} 대비): 99% 독립성 p {r[f'[{b}] 99% 독립성 p']:.3f} → {r[f'[{m}] 99% 독립성 p']:.3f}, "
                          f"휴장 뒤 95% 예외 {r[f'[{b}] 휴장 뒤 95% 예외']:.1%} → {r[f'[{m}] 휴장 뒤 95% 예외']:.1%}, "
                          f"VaR {r[f'[{b}] VaR 검정']} → {r[f'[{m}] VaR 검정']}, ES {r[f'[{b}] ES 검정']} → {r[f'[{m}] ES 검정']}, "
                          f"분위수 손실 비율 {r[f'분위수 손실 비율 ({m} ÷ {lb})']:.3f} "
                          f"(앞 {r[f'분위수 손실 비율 ({m} ÷ {lb}) 앞 절반']:.3f}·뒤 {r[f'분위수 손실 비율 ({m} ÷ {lb}) 뒤 절반']:.3f})"
                          for m, n in VAR_GROUP_CANDS.items() if f"[{m}] VaR 검정" in r for b, lb in [VAR_GROUP_BASE[m]])
                + (f"\n  최근 1년 제외({r['[최근1년 제외] 끝']}까지): z {r['[최근1년 제외] 방향 z']:+.1f}, "
                   f"방향 초과 {r['[최근1년 제외] 방향 초과']:+.1%} [{r['[최근1년 제외] 방향 하한']:+.1%}, {r['[최근1년 제외] 방향 상한']:+.1%}], "
                   f"분할 {r['[최근1년 제외] 분할 초과']:+.1%}, 국면 {r['[최근1년 제외] 국면 초과']:+.1%}" if "[최근1년 제외] 방향 z" in r else ""))
        except Exception as e:                    # 한 종목이 실패해도 나머지는 계속
            rows.append({"종목": t, "이름": t, "오류": f"{type(e).__name__}: {str(e)[:200]}"})
            say(f"{t}: 실패 — {rows[-1]['오류']}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    version = os.environ.get("GITHUB_SHA", "")[:7] or repro.code_version()
    when = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    pooled = pd.concat(obs, ignore_index=True) if obs else None
    buckets = dr.bucket_table(pooled["p"], pooled["fwd"], pooled["c"]) if pooled is not None else None
    (out_dir / "README.md").write_text(report_markdown(rows, ranking_summary(), when, version, buckets), encoding="utf-8")
    return rows


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t_all = time.time()
    rows = run(["^KS11", "005930.KS"] if smoke else fl.tickers(), say=lambda m: print(m, flush=True))
    print(f"총 {time.time() - t_all:.0f}초, 실패 {sum('오류' in r for r in rows)}종목", flush=True)
    print("VaR 꼬리 분포 사전 등록 규칙:", var_tail_rule(rows), flush=True)
    print("VaR σ 보정 사전 등록 규칙:", var_scale_rule(rows), flush=True)
    print("VKOSPI 하루 늦게 섞기(C1L2) 사전 등록 규칙:", var_iv_lag_rule(rows), flush=True)
    for m, n in VAR_GROUP_CANDS.items():
        print(f"VaR 재보정 후보 {n} 사전 등록 규칙:", var_cluster_rule(rows, m), flush=True)
    sys.exit(1 if all("오류" in r for r in rows) else 0)
