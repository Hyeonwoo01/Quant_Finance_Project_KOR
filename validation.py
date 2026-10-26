"""실제 데이터 검증 보고서: 계산이 맞는지(tests.py)와 별개로, 비용을 뺀 뒤에도 돈이 되는지를 실제 데이터로 잰다.

GitHub Actions(validation.yml)가 매주 돌려 validation/summary.csv와 validation/README.md로 커밋한다.
기록 종목(forecast_log/tickers.txt)마다 앱 기본 설정 그대로, 워크포워드(그 시점까지의 데이터로만 학습·선택)로:
- 방향 예측: 21거래일 방향을 21거래일 뒤 결과로 채점 (적중률, 확률 정확도, 겹침 보정 z)
- 방향 전략·시장 국면 전략: 판단 다음 날 종가 체결, 수수료·증권거래세(체결일 세율)를 뺀 뒤 단순 보유와 비교.
  전략은 매일 다시 판단해 다음 하루를 들고 있는 방식이라, 21일 예측의 채점과 전략 수익은 서로 다른 질문이다.
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
from quant_engine import block_bootstrap_ci, var_backtest

OUT_DIR = Path(__file__).parent / "validation"
H, COST_BPS, LAG = 21, 3.0, 1          # 앱 기본 설정 (방향 예측 기간, 수수료·체결 비용 bp, 다음 날 종가 체결)
BLOCK = 2 * H                          # 블록 부트스트랩 블록 길이 (거래일) — 포지션·변동성이 이어지는 기간보다 넉넉하게
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
    row.update({
        "검증 기간": f"{wf.index[0].date()} ~ {wf.index[-1].date()}",
        "방향 적중률": ev["hit"], "항상 오른다 적중률": ev["base"], "방향 z": ev["z"], "확률 정확도 개선": ev["skill"],
        "방향 예측력": "있음" if ev["skill"] > 0 and ev["z"] > 2 else "확인 안 됨",
        "방향 전략 연 수익률": d_ret, "방향 전략 샤프": d_sh, "방향 전략 최대 낙폭": ev["mdd"],
        "단순 보유 연 수익률": b_ret, "단순 보유 샤프": b_sh, "단순 보유 최대 낙폭": ev["mdd_bh"],
        "방향 전략 보유 비중": ev["time_in_market"],
        "방향 초과수익(연)": d_ex, "방향 초과 95% 하한": d_lo, "방향 초과 95% 상한": d_hi,
        "가정 바꿔 통과(7개 중)": passes, "방향 DSR": sel["dsr"] if sel else np.nan,
        "방향 전략 경제적 가치": "있음" if d_lo > 0 else "확인 안 됨",
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

    # 하루 VaR 예외 검정 (앱 기본: 자동 = AIC로 대칭/GJR 선택)
    vt = var_backtest(r.dropna().values, model="auto")
    row.update({"VaR 95% 예외 비율": vt.loc["하루 VaR 95%", "예외 비율"], "VaR 95% p값": vt.loc["하루 VaR 95%", "종합 p값"],
                "VaR 99% 예외 비율": vt.loc["하루 VaR 99%", "예외 비율"], "VaR 99% p값": vt.loc["하루 VaR 99%", "종합 p값"],
                "VaR 검정": "통과" if min(vt["종합 p값"]) > 0.05 else "기각"})
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
    return {"데이터": f"{w['data_end'].date()}까지", "리밸런싱 수": a["리밸런싱 수"],
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


def report_markdown(rows, rank=None, when="", version=""):
    """검증 결과 → validation/README.md 내용."""
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
        "| 코스피 종목 고르기 | 비용 후 초과수익 t > 2 그리고 IC t > 2 |",
        "",
        "샤프 비율 하나만으로 판단하지 않습니다. 신뢰구간, 가정을 바꿔도 유지되는지(7개: 자동 선택 기간 4 + 시기 3), "
        "여러 설정 중 1등을 고른 착시 보정(DSR, 0.95 이상이면 우연일 가능성 낮음)을 함께 봅니다.",
        "",
        "## 요약",
        "",
        f"- 방향 예측력 있음: {sum(r['방향 예측력'] == '있음' for r in ok)} / {len(ok)}종목",
        f"- 방향 전략 경제적 가치 있음: {sum(r['방향 전략 경제적 가치'] == '있음' for r in ok)} / {len(ok)}종목",
        f"- 국면 전략 경제적 가치 있음: {sum(r['국면 전략 경제적 가치'] == '있음' for r in ok)} / {len(ok)}종목",
        f"- VaR 검정 통과: {sum(r['VaR 검정'] == '통과' for r in ok)} / {len(ok)}종목",
    ]
    if rank:
        lines.append(f"- 코스피 종목 고르기 경제적 가치: {rank['경제적 가치']}")
    lines += ["", "## 방향 예측 (21거래일)", "",
              "| 종목 | 검증 기간 | 적중률 | 항상 오른다 | z | 확률 정확도 | 예측력 |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {r['검증 기간']} | {_pct(r['방향 적중률'], False)} | {_pct(r['항상 오른다 적중률'], False)} | "
                     f"{_num(r['방향 z'], '{:+.1f}')} | {_pct(r['확률 정확도 개선'])} | {r['방향 예측력']} |")
    lines += ["", "## 방향 전략 vs 단순 보유 (비용·세금 후)", "",
              "| 종목 | 전략 연 수익률 | 샤프 | 최대 낙폭 | 단순 보유 연 수익률 | 샤프 | 최대 낙폭 | 초과(연) [95% CI] | 가정 바꿔 통과 | DSR | 경제적 가치 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['방향 전략 연 수익률'])} | {_num(r['방향 전략 샤프'])} | {_pct(r['방향 전략 최대 낙폭'])} | "
                     f"{_pct(r['단순 보유 연 수익률'])} | {_num(r['단순 보유 샤프'])} | {_pct(r['단순 보유 최대 낙폭'])} | "
                     f"{_pct(r['방향 초과수익(연)'])} [{_pct(r['방향 초과 95% 하한'])}, {_pct(r['방향 초과 95% 상한'])}] | "
                     f"{r['가정 바꿔 통과(7개 중)']}/7 | {_num(r['방향 DSR'], '{:.0%}')} | {r['방향 전략 경제적 가치']} |")
    lines += ["", "## 시장 국면 전략 ('불안 국면엔 현금') vs 단순 보유 (비용·세금 후)", "",
              "| 종목 | 전략 연 수익률 | 샤프 | 최대 낙폭 | 보유 비중 | 초과(연) [95% CI] | 경제적 가치 |", "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['국면 전략 연 수익률'])} | {_num(r['국면 전략 샤프'])} | {_pct(r['국면 전략 최대 낙폭'])} | "
                     f"{_pct(r['국면 전략 보유 비중'], False)} | {_pct(r['국면 초과수익(연)'])} "
                     f"[{_pct(r['국면 초과 95% 하한'])}, {_pct(r['국면 초과 95% 상한'])}] | {r['국면 전략 경제적 가치']} |")
    lines += ["", "## 하루 VaR 예외 검정 (최근 약 500거래일)", "",
              "| 종목 | 95% 예외 비율 (목표 5%) | p값 | 99% 예외 비율 (목표 1%) | p값 | 결과 |", "|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['이름']} | {_pct(r['VaR 95% 예외 비율'], False)} | {_num(r['VaR 95% p값'], '{:.2f}')} | "
                     f"{_pct(r['VaR 99% 예외 비율'], False)} | {_num(r['VaR 99% p값'], '{:.2f}')} | {r['VaR 검정']} |")
    if rank:
        lines += ["", "## 코스피 종목 고르기 (상위 20%, 교체 63거래일, 비용·세금 후, 전체 평균 대비)", "",
                  "| 데이터 | 리밸런싱 | 초과수익(연) | t | 평균 IC | IC t | 최대 낙폭 | 요인 알파(연) | 알파 t | DSR | 실전 기록 (21일 채점 수 · 평균 초과) | 경제적 가치 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|",
                  f"| {rank['데이터']} | {rank['리밸런싱 수']:.0f} | {_pct(rank['초과수익(연, 비용후)'])} | {_num(rank['초과수익 t값'], '{:+.1f}')} | "
                  f"{_num(rank['평균 IC'], '{:+.3f}')} | {_num(rank['IC t값'], '{:+.1f}')} | {_pct(rank['최대 낙폭(비용후)'])} | "
                  f"{_pct(rank['요인 알파(연)'])} | {_num(rank['요인 알파 t값'], '{:+.1f}')} | {_num(rank['1등 착시 보정(DSR)'], '{:.0%}')} | "
                  f"{rank['실전 기록 채점 수(21일)']:.0f}번 · {_pct(rank['실전 기록 평균 초과수익(21일)'])} | {rank['경제적 가치']} |"]
    bad = [r for r in rows if "오류" in r]
    if bad:
        lines += ["", "## 계산하지 못한 종목", ""] + [f"- {r['종목']}: {r['오류']}" for r in bad]
    lines += ["", "전체 숫자는 [summary.csv](summary.csv)에 있습니다.", ""]
    return "\n".join(lines)


def run(tickers, out_dir=OUT_DIR, say=print):
    rows = []
    for t in tickers:
        t0 = time.time()
        try:
            r = validate(t)
            rows.append(r)
            say(f"{t} ({r['이름']}): 완료 ({time.time() - t0:.0f}초) · 검증 기간 {r['검증 기간']}\n"
                f"  대체 데이터: {r['대체 데이터'] or '없음'}\n"
                f"  방향 예측: 적중률 {r['방향 적중률']:.1%} (항상 오른다 {r['항상 오른다 적중률']:.1%}), z {r['방향 z']:+.1f}, "
                f"확률 정확도 {r['확률 정확도 개선']:+.1%} → 예측력 {r['방향 예측력']}\n"
                f"  방향 전략: 연 {r['방향 전략 연 수익률']:+.1%} 샤프 {r['방향 전략 샤프']:.2f} 낙폭 {r['방향 전략 최대 낙폭']:.0%} | "
                f"단순 보유: 연 {r['단순 보유 연 수익률']:+.1%} 샤프 {r['단순 보유 샤프']:.2f} 낙폭 {r['단순 보유 최대 낙폭']:.0%} | "
                f"초과 {r['방향 초과수익(연)']:+.1%} [{r['방향 초과 95% 하한']:+.1%}, {r['방향 초과 95% 상한']:+.1%}] "
                f"가정 바꿔 {r['가정 바꿔 통과(7개 중)']}/7 → {r['방향 전략 경제적 가치']}\n"
                f"  국면 전략: 연 {r['국면 전략 연 수익률']:+.1%} 샤프 {r['국면 전략 샤프']:.2f} 낙폭 {r['국면 전략 최대 낙폭']:.0%} | "
                f"초과 {r['국면 초과수익(연)']:+.1%} [{r['국면 초과 95% 하한']:+.1%}, {r['국면 초과 95% 상한']:+.1%}] → {r['국면 전략 경제적 가치']}\n"
                f"  하루 VaR: 95% 예외 {r['VaR 95% 예외 비율']:.1%} (p {r['VaR 95% p값']:.2f}), "
                f"99% 예외 {r['VaR 99% 예외 비율']:.1%} (p {r['VaR 99% p값']:.2f}) → {r['VaR 검정']}")
        except Exception as e:                    # 한 종목이 실패해도 나머지는 계속
            rows.append({"종목": t, "이름": t, "오류": f"{type(e).__name__}: {str(e)[:200]}"})
            say(f"{t}: 실패 — {rows[-1]['오류']}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    version = os.environ.get("GITHUB_SHA", "")[:7] or repro.code_version()
    when = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    (out_dir / "README.md").write_text(report_markdown(rows, ranking_summary(), when, version), encoding="utf-8")
    return rows


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t_all = time.time()
    rows = run(["^KS11", "005930.KS"] if smoke else fl.tickers(), say=lambda m: print(m, flush=True))
    print(f"총 {time.time() - t_all:.0f}초, 실패 {sum('오류' in r for r in rows)}종목", flush=True)
    sys.exit(1 if all("오류" in r for r in rows) else 0)
