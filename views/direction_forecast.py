"""방향 예측: 상승 확률, 모델 비교, 가정 바꿔 점검, 수익 분해."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import direction as dr
import ui
from app_data import get_direction_horizons, get_direction_horizons_full, get_direction_robustness, sell_tax_for
from views.common import attribution_view, robust_table, signed_table


def render(ctx):
    """방향 예측: 상승 확률, 모델 비교, 가정 바꿔 점검, 수익 분해. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    alt_status, close_long, contrib, cost_bps = ctx.alt_status, ctx.close_long, ctx.contrib, ctx.cost_bps
    cost_word, dir_asof, dir_err, dir_h, ev = ctx.cost_word, ctx.dir_asof, ctx.dir_err, ctx.dir_h, ctx.ev
    exec_lag, exec_word, kind, now_all, p_up = ctx.exec_lag, ctx.exec_word, ctx.kind, ctx.now_all, ctx.p_up
    pick, reliable, tax_on, ticker, use_tax, wf = ctx.pick, ctx.reliable, ctx.tax_on, ctx.ticker, ctx.use_tax, ctx.wf
    if p_up is None:
        st.error(f"방향 예측을 할 수 없습니다 (데이터 부족 등): {dir_err}")
    else:
        view = "상승" if p_up > 0.5 else "하락"
        strength = abs(p_up - 0.5)
        level = ("모델 신호 없음, 과거 상승 비율" if pick == "base" else
                 "강함" if strength >= 0.15 else "보통" if strength >= 0.07 else "약함(거의 동전 던지기)")
        ui.section("Direction Forecast", f"{dir_h}거래일 뒤, 지금보다 오를까?")
        ui.verdict(f"기준일 {dir_asof.date()} · 사용 모델 {dr.MODEL_LABELS[pick]}", f"{view} {p_up:.1%}", "상승 확률",
                   [(f"강도 {level}", "neutral"),
                    ("과거 검증 통과", "approve") if reliable else ("과거 검증 미통과", "elective")],
                   "최대 10년치 하루 주가로 여러 모델을 시험해, 최근 3년 동안 가장 잘 맞힌 모델을 골랐습니다.")
        ui.badges("모델에 들어간 데이터", [("가격 특징 · 사용", "approve")] + [
            (f"{k} · {'사용' if v.startswith('사용') else '키 없음' if v == '키 없음' else '오류' if v.startswith('오류') else '기간 짧아 제외' if v.startswith('기간이 짧아') else '없음'}",
             "approve" if v.startswith("사용") else "neutral" if v == "키 없음" else
             "reject" if v.startswith("오류") else "neutral") for k, v in alt_status.items()])
        if any(v == "키 없음" for v in alt_status.values()):
            st.caption("키 없음 = API 키가 없어 꺼짐. `api_keys.env` 파일에 `DART_API_KEY=…`, `ECOS_API_KEY=…`를 넣으면 켜집니다.")
        short = {k: v for k, v in alt_status.items() if v.startswith("기간이 짧아")}
        if short:
            st.caption("기간 짧아 제외 = 가격 기록의 뒤쪽 절반에서야 시작하는 데이터. 넣으면 학습·검증 기간이 절반 넘게 줄어서 뺐습니다 ("
                       + " / ".join(f"{k}: {v.split('(')[-1].rstrip(')')}" for k, v in short.items()) + ").")
        errs = {k: v for k, v in alt_status.items() if v.startswith("오류")}
        if errs:
            st.caption(" / ".join(f"{k}: {v}" for k, v in errs.items()))

        if pick == "base":
            st.info("지금은 어떤 모델도 **이 종목이 과거에 오른 비율**보다 잘 맞히지 못했습니다. "
                    "그래서 그 비율을 그대로 상승 확률로 보여줍니다. 근거 없는 예측을 내지 않기 위한 장치입니다.")
        if reliable:
            st.success(f"과거 검증에서 이 모델은 '항상 오른다'고 보는 것보다 확실히 잘 맞혔습니다 "
                       f"(확률 정확도 {ev['skill']:+.1%} 개선, 우연일 가능성 낮음 z = {ev['z']:.1f}). "
                       "그래도 앞으로도 그럴 거라는 보장은 없습니다.")
        else:
            st.warning(f"과거 검증에서 이 모델은 '항상 오른다'고 보는 것보다 **확실히 낫지 않았습니다** "
                       f"(z = {ev['z']:.1f}, 확률 정확도 {ev['skill']:+.1%}). 참고로만 보세요. "
                       "공개된 주가만으로 방향을 맞히기는 원래 매우 어렵습니다.")

        left, right = st.columns(2)
        with left:
            st.markdown("**어떤 지표가 상승·하락 쪽으로 기울였나** (오른쪽으로 길수록 상승 쪽, 간단한 모델 기준)")
            cs = contrib.sort_values()
            fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h",
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in cs.values]))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="영향 크기")
            ui.chart(fig)
        if ev is not None:
            with right:
                st.markdown("**과거 성적표** (매 시점 그때까지의 데이터만 써서 예측한 결과)")
                m1, m2 = st.columns(2)
                m1.metric("적중률", f"{ev['hit']:.1%}")
                m2.metric("'항상 오른다'고 했을 때", f"{ev['base']:.1%}",
                          f"차이 {ev['hit'] - ev['base']:+.1%}p", delta_color="off")
                m3, m4 = st.columns(2)
                m3.metric("확률 오차 (Brier, 낮을수록 좋음)", f"{ev['brier']:.3f}",
                          f"기준 {ev['brier_base']:.3f}", delta_color="off")
                m4.metric("검증 표본", f"{ev['n']:,}일", f"서로 겹치지 않는 구간 ≈ {ev['n_eff']:.0f}개", delta_color="off")

            st.markdown("**후보 모델 비교** — 모두 같은 조건으로 시험했습니다. 이 표를 보고 나중에 1등을 고르면 성적이 부풀려지므로, "
                        "앱은 매 시점 그때까지의 성적만 보고 고른 **자동 선택** 결과를 씁니다.")
            st.dataframe(pd.DataFrame([{
                "모델": dr.MODEL_LABELS.get(k, "자동 선택"),
                "오늘 상승 확률": now_all.get(k, p_up),
                "적중률": s["hit"], "'항상 오른다' 대비": s["hit"] - s["base"],
                "구분력 AUC (0.5=무작위)": s["auc"], "확률 정확도 개선": s["skill"],
            } for k, s in ev["table"].items()]).style.format({
                "오늘 상승 확률": "{:.1%}", "적중률": "{:.1%}", "'항상 오른다' 대비": "{:+.1%}",
                "구분력 AUC (0.5=무작위)": "{:.3f}", "확률 정확도 개선": "{:+.1%}"}),
                hide_index=True, width="stretch")
            share = ", ".join(f"{dr.MODEL_LABELS[k]} {v:.0%}" for k, v in ev["chosen_share"].items())
            st.caption(f"검증 기간 동안 자동 선택된 비율: {share}. "
                       "구분력(AUC)은 오를 때와 내릴 때를 얼마나 잘 가르는지(0.5면 동전 던지기), "
                       "확률 정확도 개선은 과거 상승 비율을 그대로 쓸 때보다 확률이 몇 % 더 정확한지입니다.")
            with st.expander("상승 확률 구간별로 실제 수익은 얼마였나? (확률만으로는 부족한 이유)"):
                tx = sell_tax_for(close_long.index, kind, use_tax)
                tx = tx.shift(-(exec_lag + dir_h)).reindex(wf.index).ffill() if isinstance(tx, pd.Series) else tx
                bt = dr.bucket_table(wf["auto"], wf["fwd"], 2 * cost_bps * 1e-4 + tx)
                st.dataframe(bt.style.format({"관측 수": "{:,.0f}", "평균 예측 확률": "{:.1%}", "실제 상승 비율": "{:.1%}",
                                              "평균 수익": "{:+.2%}", "오를 때 평균": "{:+.2%}", "내릴 때 평균": "{:+.2%}",
                                              "비용 뺀 평균": "{:+.2%}"}, na_rep="—"), width="stretch")
                st.caption("오를 확률이 높아도 오를 때 조금 벌고 내릴 때 크게 잃으면 평균 수익은 음수입니다: "
                           "평균 수익 = 상승 비율 × 오를 때 평균 + (1 − 상승 비율) × 내릴 때 평균. "
                           f"수익은 판단일 종가 → {dir_h}거래일 뒤 종가, 비용은 한 번 사서 팔 때의 수수료·세금입니다. "
                           "매일 예측이라 결과가 겹쳐서 관측 수는 독립 표본 수가 아닙니다.")

            st.markdown(f"**상승 확률이 50%를 넘을 때만 들고 있었다면?** 단순 보유와 비교 "
                        f"(체결: {exec_word}, {cost_word}, 주식을 들고 있던 기간 {ev['time_in_market']:.0%}, "
                        f"위험 대비 수익(샤프) {ev['sharpe']:.2f} vs {ev['sharpe_bh']:.2f}, "
                        f"최대 낙폭 {ev['mdd']:.0%} vs {ev['mdd_bh']:.0%})")
            eq = ev["equity"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq["전략"], name="50% 넘을 때만 보유", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=eq.index, y=eq["단순 보유"], name="단순 보유", line=dict(color=ui.NEUTRAL)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.markdown("**이 전략의 수익은 어디서 왔나?** — 오르내림을 맞힌 실력인지, 단지 주식을 덜(더) 들고 있었던 것인지")
            attribution_view(ev["daily"])

            st.markdown("**상승 확률 추이와 주가** (같은 기간, 위·아래로 나눠 표시)")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.5])
            fig.add_trace(go.Scatter(x=wf.index, y=wf["auto"], name="상승 확률 (자동 선택)",
                                     line=dict(color=ui.BLUE)), row=1, col=1)
            fig.add_hline(y=0.5, line=dict(color=ui.NEUTRAL, dash="dot"), row=1, col=1)
            px_view = close_long[close_long.index >= wf.index[0]]
            fig.add_trace(go.Scatter(x=px_view.index, y=px_view.values, name="종가",
                                     line=dict(color=ui.INK, width=1.5)), row=2, col=1)
            fig.update_yaxes(tickformat=".0%", title_text="상승 확률", row=1, col=1)
            fig.update_yaxes(title_text="종가", row=2, col=1)
            fig.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=1.08))
            ui.chart(fig)

            # ---- 가정을 바꿔도 결과가 같을까?
            st.subheader("가정을 바꿔도 결과가 같을까?")
            st.markdown("자동 선택이 보는 과거 기간, 보유 기준, 거래비용, 시기, 예측 기간을 바꿔 다시 채점합니다. "
                        "예측 기간을 빼고는 이미 계산된 예측을 다시 쓰므로 모델을 새로 학습하지 않습니다.")
            RB = get_direction_robustness(ticker, dir_h, cost_bps, exec_lag, use_tax, kind)
            bw, br, bp = RB["by_window"], RB["by_rule"], RB["by_period"]
            n_pass, n_all = int(bw["검증 통과"].sum() + bp["검증 통과"].sum()), len(bw) + len(bp)
            n_beat = int((br.values > 0).sum())
            label, badge = (("모든 경우 통과", "approve") if n_pass == n_all else
                           ("절반 이상 통과", "elective") if n_pass * 2 >= n_all else
                           ("일부에서만 통과", "elective") if n_pass else ("한 번도 통과 못 함", "reject"))
            ui.verdict(f"{dir_h}거래일 방향 예측 · 선택 기간 {len(bw)}가지 + 시기 {len(bp)}구간", f"{n_pass} / {n_all}",
                       "경우에서 검증 통과", [(label, badge)],
                       f"검증 통과 = 확률 정확도가 기본 확률보다 좋고, 적중률 우위가 우연이 아닌 경우(z > 2). "
                       f"보유 기준 × 거래비용 {br.size}가지 중 전략 샤프가 단순 보유보다 높았던 경우는 {n_beat}가지였습니다.")

            st.markdown("**① 자동 선택이 보는 과거 기간** — 최근 몇 년의 성적으로 후보 모델을 고를지")
            robust_table(bw)
            st.caption("지금 앱은 '최근 3년'을 씁니다. 기간에 따라 고르는 모델과 결과가 크게 바뀌면, 모델 선택이 운에 좌우된다는 뜻입니다.")

            st.markdown("**② 보유 기준 × 수수료·체결 비용** — '상승 확률이 기준보다 높으면 보유' 전략의 샤프 − 단순 보유 샤프"
                        + (" (증권거래세는 모든 칸에 똑같이 반영)" if tax_on else ""))
            signed_table(br)
            if (br.abs().round(2) == 0).all().all():
                st.caption("모든 칸이 0인 이유: 자동 선택이 거의 늘 '기본 확률'을 골랐고, 그 확률(과거 상승 비율)이 모든 보유 기준보다 "
                           "높아서 전략이 내내 보유했습니다. 즉 단순 보유와 똑같았다는 뜻입니다.")
            st.caption("+면 그 설정에서 전략이 단순 보유보다 위험 대비 수익이 좋았다는 뜻입니다. 자동 선택이 '기본 확률'을 고른 기간에는 "
                       "확률이 과거 상승 비율로 고정되므로, 기준을 그보다 높게 잡으면 그 기간 내내 현금, 낮게 잡으면 내내 보유가 됩니다. "
                       "그래서 기준에 따라 결과가 크게 갈리면 모델의 예측력보다 기준 설정의 영향이 큰 것입니다.")

            sel = RB.get("selection")
            st.markdown("**②-2 여러 조합 중 1등을 고른 착시 보정** — 자동 선택 기간 4가지 × 보유 기준 3가지 = 12개 조합 "
                        "(Deflated Sharpe)")
            if sel is None:
                st.caption("모든 조합이 단순 보유와 똑같아(초과수익 0) 보정할 대상이 없습니다.")
            else:
                signed_table(sel["grid"])
                st.caption(f"표: 조합별 '전략 − 단순 보유' 일간 초과수익의 샤프(연). 1등은 {sel['best'][0]} · 기준 {sel['best'][1]} "
                           f"(샤프 {sel['sr_ann']:+.2f})입니다. 효과가 전혀 없어도 {sel['n_trials']}개를 시험하면 1등은 샤프 "
                           f"{sel['sr0_ann']:.2f} 정도가 나올 수 있습니다. 이를 감안하면 1등이 진짜로 단순 보유보다 나을 확률은 "
                           f"{sel['dsr']:.0%}입니다" + (" — 95%를 넘어 우연이 아닐 가능성이 높습니다." if sel["dsr"] > 0.95 else
                                                            " — 95%에 못 미쳐 우연과 구분하기 어렵습니다.")
                           + (f" 단순 보유와 똑같았던 {sel['n_flat']}개 조합은 빼고 계산했습니다." if sel["n_flat"] else ""))

            st.markdown("**③ 시기별** — 검증 기간을 세 구간으로 나눠 채점")
            robust_table(bp)
            st.caption("한 구간은 약 2~3년이라 표본이 작아 z가 낮게 나오기 쉽습니다. 구간마다 방향(+/−)이 일관적인지를 보세요.")

            st.markdown("**④ 예측 기간을 바꿔 보면** — 5·10·21·42·63거래일 (모델을 새로 학습하는 가벼운 점검)")
            if st.button("예측 기간 바꿔 점검", key="dir_h_check"):
                try:
                    with st.spinner("예측 기간 5가지로 다시 학습·검증하는 중… (보통 1분 안팎)"):
                        HZ = get_direction_horizons(ticker, cost_bps, exec_lag, use_tax, kind)
                    robust_table(HZ)
                    gbm_share = ev["chosen_share"].reindex(["gbm", "ensemble"]).fillna(0).sum()
                    st.caption(f"검증 통과 {int(HZ['검증 통과'].sum())} / {len(HZ)}. 빠르게 돌리려고 무거운 부스팅·앙상블 후보를 빼고 "
                               "기본 확률·로지스틱 후보만으로 다시 학습했습니다(재학습 주기는 위와 같은 한 달). "
                               f"위 결과에서 부스팅·앙상블이 골라진 비율은 {gbm_share:.0%}"
                               + ("라서 이 점검과 결론이 크게 다르지 않습니다." if gbm_share < 0.2 else
                                  "로 높아서, 이 점검은 그 모델의 성적을 반영하지 못합니다."))
                except Exception as e:
                    st.error(f"계산 실패: {e}")

            st.markdown("**④-2 부스팅까지 포함해 점검** — 기간이 길수록 비선형 모델이 나을 수 있는지 확인 (무겁습니다)")
            if st.button("부스팅까지 포함해 점검", key="dir_h_check_full"):
                try:
                    with st.spinner("모든 후보로 예측 기간 5가지를 다시 학습하는 중… (무료 서버에서는 몇 분 걸릴 수 있습니다)"):
                        HF, HL = get_direction_horizons_full(ticker, cost_bps, exec_lag, use_tax, kind)
                    robust_table(HF)
                    cmp = pd.DataFrame({"로지스틱만": HL["확률 정확도 개선"], "부스팅 포함": HF["확률 정확도 개선"]}).astype(float)
                    cmp["차이 (부스팅 포함 − 로지스틱만)"] = cmp["부스팅 포함"] - cmp["로지스틱만"]
                    st.markdown("확률 정확도 개선 비교 — 같은 조건(1년마다 재학습)에서 부스팅·앙상블 후보만 넣고 뺀 차이")
                    st.dataframe(cmp.style.format({"로지스틱만": "{:+.1%}", "부스팅 포함": "{:+.1%}"}).format(
                        lambda v: "0.0%p" if round(v * 100, 1) == 0 else f"{v * 100:+.1f}%p",
                        subset=["차이 (부스팅 포함 − 로지스틱만)"]).map(
                        lambda v: f"color: {ui.APPROVE}" if round(v * 100, 1) > 0 else f"color: {ui.REJECT}" if round(v * 100, 1) < 0 else "",
                        subset=["차이 (부스팅 포함 − 로지스틱만)"]), width="stretch")
                    st.caption(f"검증 통과 {int(HF['검증 통과'].sum())} / {len(HF)}. 무거워서 1년(252거래일)마다 재학습했고, 비교용 '로지스틱만'도 "
                               "같은 주기로 돌렸습니다. 차이가 +이고 긴 기간(42·63일)에서 커지면 부스팅이 긴 기간에 도움이 된다는 신호지만, "
                               "63일 예측은 10년치라도 서로 독립적인 표본이 30개 안팎이라 z가 2를 넘기 어렵습니다 — 검증 통과 여부를 꼭 함께 보세요.")
                except Exception as e:
                    st.error(f"계산 실패: {e}")
