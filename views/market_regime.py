"""시장 국면: 불안 국면 확률과 '불안 국면엔 현금' 전략."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import regime as rg
import ui
from app_data import get_regime
from views.common import attribution_view


def render(ctx):
    """시장 국면: 불안 국면 확률과 '불안 국면엔 현금' 전략. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    cost_bps, cost_word, exec_lag, exec_word = ctx.cost_bps, ctx.cost_word, ctx.exec_lag, ctx.exec_word
    kind, ticker, use_tax = ctx.kind, ctx.ticker, ctx.use_tax
    ui.section("Market Regime", "지금은 안정 국면일까, 불안 국면일까",
               "주가가 조용히 움직이는 '안정 국면'과 크게 출렁이는 '불안 국면'을 2상태 국면 전환 모델(Markov switching)로 나눕니다. "
               "확률은 그날까지의 데이터만으로 계산해서, 과거 구간에도 미래 정보가 섞이지 않습니다.")
    try:
        with st.spinner("국면 모델 추정 중…"):
            R = get_regime(ticker, cost_bps, exec_lag, use_tax, kind)
    except Exception as e:
        R = None
        st.error(f"국면 분석을 할 수 없습니다 (데이터 부족 등): {e}")
    if R is not None:
        p_now, stt = R["p_now"], R["states"]
        high = p_now >= 0.5
        ui.verdict(f"기준일 {R['asof'].date()} · 최대 10년치 일봉", rg.STATE_LABELS[int(high)], "",
                   [(f"불안 국면 확률 {p_now:.0%}", "elective" if high else "approve")],
                   f"안정 국면은 연 변동성 약 {stt['연 변동성'].iloc[0]:.0%}, 불안 국면은 약 {stt['연 변동성'].iloc[1]:.0%}로 추정됩니다. "
                   f"한 번 들어가면 평균 {stt['평균 지속 기간(거래일)'].iloc[0]:.0f}거래일 / "
                   f"{stt['평균 지속 기간(거래일)'].iloc[1]:.0f}거래일 이어졌습니다.")

        ph = R["p_high"].dropna()
        px_r = R["close"].reindex(ph.index)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=px_r.index, y=px_r.values, name="종가", line=dict(color=ui.INK, width=1.5)), row=1, col=1)
        on = (ph >= 0.5).values                                   # 불안 국면 구간을 가격 차트에 옅게 칠함 (선을 먼저 그려야 칸이 생김)
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a0, a1 in zip(edges[::2], edges[1::2]):
            fig.add_vrect(x0=ph.index[a0], x1=ph.index[a1 - 1], fillcolor="rgba(183,110,0,0.14)", line_width=0,
                          layer="below", row=1, col=1)
        fig.add_trace(go.Scatter(x=ph.index, y=ph.values, name="불안 국면 확률", fill="tozeroy",
                                 line=dict(color=ui.AMBER, width=1), fillcolor="rgba(183,110,0,0.18)"), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color=ui.NEUTRAL, dash="dot"), row=2, col=1)
        fig.update_yaxes(title_text="종가", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="불안 국면 확률", row=2, col=1)
        fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        ui.chart(fig)
        st.caption("주황 구간 = 불안 국면 확률 50% 이상. 모델은 1년마다 그때까지의 데이터로만 다시 추정했고, 처음 3년은 학습 기간이라 비어 있습니다.")

        st.markdown("**국면별 특징** (전체 기간으로 추정)")
        st.dataframe(stt.style.format({"연 평균 수익률": "{:+.1%}", "연 변동성": "{:.1%}", "평균 지속 기간(거래일)": "{:.0f}"}),
                     width="stretch")

        st.markdown(f"**불안 국면일 때 현금으로 피했다면?** 단순 보유와 비교 ({cost_word}, "
                    f"주식을 들고 있던 기간 {R['time_in_market']:.0%}, 국면이 바뀌어 사고판 횟수 {R['switches']}번)")
        st.dataframe(R["table"].style.format({"연 수익률": "{:+.1%}", "연 변동성": "{:.1%}", "샤프": "{:.2f}",
                                              "최대 낙폭": "{:.0%}"}), width="stretch")
        eq = R["equity"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq["불안 국면엔 현금"], name="불안 국면엔 현금", line=dict(color=ui.BLUE)))
        fig.add_trace(go.Scatter(x=eq.index, y=eq["단순 보유"], name="단순 보유", line=dict(color=ui.NEUTRAL)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
        ui.chart(fig)
        st.caption(f"확률이 50% 이상이 되면 {exec_word}에 팔고 현금(이자 0으로 계산), 50% 아래로 내려오면 다시 삽니다. "
                   "국면 분류는 보통 수익을 늘리기보다 큰 하락을 줄이는 도구입니다. 상승장 초입을 놓치는 대가가 따릅니다.")
        st.markdown("**이 전략의 수익은 어디서 왔나?** — 하락을 피한 실력인지, 단지 주식을 덜 들고 있었던 것인지")
        attribution_view(R["daily"])
