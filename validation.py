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
from quant_engine import block_bootstrap_ci, fit_vol_model, var_backtest

OUT_DIR = Path(__file__).parent / "validation"
H, COST_BPS, LAG = 21, 3.0, 1          # 앱 기본 설정 (방향 예측 기간, 수수료·체결 비용 bp, 다음 날 종가 체결)
BLOCK = 2 * H                          # 블록 부트스트랩 블록 길이 (거래일) — 포지션·변동성이 이어지는 기간보다 넉넉하게
VAR_START = 756                        # 하루 VaR 검증 시작 위치: 첫 3년은 학습에만 쓰고 그 뒤 전체를 검증
INDEX_NAMES = {"^KS11": "코스피 지수", "^KQ11": "코스닥 지수", "^KS200": "코스피 200 지수"}


def _ann(x):
    """일간 로그수익률 → (연 수익률, 샤프)."""
    m, s = x.mean() * 252, x.std() * np.sqrt(252)
    return float(np.expm1(m)), float(m / s) if s > 0 else 0.0


def _excess(daily):
    """일간 (전략 − 단순 보유) 로그수익률 → 연 초과수익과 블록 부트스트랩 95% 신뢰구간 (연 단위)."""
    ex = (daily["strat"] - daily["asset"]).values
    lo, hi = block_bootstrap_ci(ex, BLOCK)
    return float(np.nanmean(ex) * 252), lo * 252, hi * 252


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

    # 하루 VaR·ES 검정 (앱 기본: 자동 = AIC로 대칭/GJR 선택). 99% VaR는 500일에 예외가 5번 정도뿐이라 검정력이 약해
    # 첫 3년(학습)을 뺀 전체 기간을 쓴다. 앞·뒤 절반의 예외 비율로 시기에 따라 흔들리는지도 본다
    rr = r.dropna().values
    vt, vd = var_backtest(rr, model="auto", start=max(min(VAR_START, len(rr) - 500), 250), detail=True)
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
            "시장 구간별": U.get("market"),
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
              "앞·뒤 절반 = 검증 기간을 반으로 나눈 95% 예외 비율 (시기에 따라 흔들리는지).", "",
              "| 종목 | 검증 일수 | 95% 예외 (목표 5%) | p값 | 99% 예외 (목표 1%) | p값 | VaR 결과 | ES 손실 비율 95% · 99% | ES p값 95% · 99% | ES 결과 | 95% 예외 앞·뒤 절반 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {r['VaR 검증 일수']:,} | {_pct(r['VaR 95% 예외 비율'], False)} | {_num(r['VaR 95% p값'], '{:.2f}')} | "
                     f"{_pct(r['VaR 99% 예외 비율'], False)} | {_num(r['VaR 99% p값'], '{:.2f}')} | {r['VaR 검정']} | "
                     f"{_num(r['ES 95% 손실 비율'], '{:.2f}')} · {_num(r['ES 99% 손실 비율'], '{:.2f}')} | "
                     f"{_num(r['ES 95% p값'], '{:.2f}')} · {_num(r['ES 99% p값'], '{:.2f}')} | {r['ES 검정']} | "
                     f"{_pct(r['VaR 95% 예외 (앞 절반)'], False)} · {_pct(r['VaR 95% 예외 (뒤 절반)'], False)} |")
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
                f"(p {r['ES 95% p값']:.2f}·{r['ES 99% p값']:.2f}) → {r['ES 검정']} | GARCH 재적합 실패 {r['GARCH 재적합 실패']} | {r['GARCH 진단']}")
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
    sys.exit(1 if all("오류" in r for r in rows) else 0)
