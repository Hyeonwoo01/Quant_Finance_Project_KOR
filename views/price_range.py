"""주가 예측 범위: GARCH 시뮬레이션 가격 범위와 손실 위험(VaR·ES)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from quant_engine import var_es


def render(ctx):
    """주가 예측 범위: GARCH 시뮬레이션 가격 범위와 손실 위험(VaR·ES). ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    close, day1, fut_idx, garch, horizon, rets = ctx.close, ctx.day1, ctx.fut_idx, ctx.garch, ctx.horizon, ctx.rets
    rl_last, sims, spot, use_iv, vk_day = ctx.rl_last, ctx.sims, ctx.spot, ctx.use_iv, ctx.vk_day
    vk_last, years = ctx.vk_last, ctx.years
    ui.section("Price Range Forecast", f"앞으로 {horizon}거래일, 주가가 움직일 범위",
               "최근 변동성을 바탕으로 주가가 얼마나 크게 움직일 수 있는지 보여줍니다. "
               "진한 띠 안에 머물 확률은 약 68%, 연한 띠는 약 95%입니다. 폭락·폭등이 정규분포보다 자주 나오는 점을 "
               f"t분포 {garch.name}로 반영해 1만 번 시뮬레이션했습니다. 범위의 중심은 <b>현재가</b>입니다(평균 수익 0 가정) — "
               "기대 가격이나 오를지 내릴지의 예측이 아니라 불확실성의 크기입니다. 오를지 내릴지는 <b>방향 예측</b> 탭에서 봅니다.")
    if not garch.converged:
        st.warning(garch.warnings[0] + " 범위와 VaR는 이 기본값으로 계산했으니 참고만 하세요.")
    cum_sd = np.sqrt(np.cumsum(garch.daily_var_path(horizon)))     # k일 누적 로그수익률 표준편차
    qs = spot * np.exp(np.quantile(sims, [0.025, 0.16, 0.84, 0.975], axis=0))
    bands = {0.68: (qs[1], qs[2]), 0.95: (qs[0], qs[3])}

    show = close.iloc[-min(len(close), max(252, 3 * horizon)):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=show.index, y=show.values, name="종가", line=dict(color=ui.INK)))
    for p, color in ((0.95, ui.BAND_OUTER), (0.68, ui.BAND_INNER)):
        lo, hi = bands[p]
        x = [close.index[-1], *fut_idx]
        fig.add_trace(go.Scatter(x=x, y=[spot, *hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[spot, *lo], fill="tonexty", fillcolor=color, line=dict(width=0),
                                 name=f"{p:.0%} 범위"))
    fig.add_trace(go.Scatter(x=[close.index[-1], fut_idx[-1]], y=[spot, spot], name="현재가 (평균 0 가정, 기대 가격 아님)",
                             line=dict(color=ui.NEUTRAL, dash="dot")))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    ui.chart(fig)

    marks = sorted({k for k in (5, 21, 63, 126, 252) if k <= horizon} | {horizon})
    st.dataframe(pd.DataFrame({
        "기간(거래일)": marks,
        "날짜": [fut_idx[k - 1].date() for k in marks],
        "68% 하단": [bands[0.68][0][k - 1] for k in marks],
        "68% 상단": [bands[0.68][1][k - 1] for k in marks],
        "95% 하단": [bands[0.95][0][k - 1] for k in marks],
        "95% 상단": [bands[0.95][1][k - 1] for k in marks],
        "보통 움직임 폭(±1σ)": [f"±{s:.1%}" for s in cum_sd[np.array(marks) - 1]],
    }).style.format({c: "{:,.2f}" for c in ("68% 하단", "68% 상단", "95% 하단", "95% 상단")}),
        hide_index=True, width="stretch")

    st.subheader("손실 위험 (VaR · ES)")
    hist_h = np.log(close).diff(horizon).dropna().values           # 학습 기간의 실제 h일 수익률 (겹침)
    risk = {"1d95": var_es(day1, 0.95), "1d99": var_es(day1, 0.99), "h95": var_es(sims[:, -1], 0.95)}
    past = {"1d95": var_es(rets.values, 0.95), "1d99": var_es(rets.values, 0.99),
            "h95": var_es(hist_h, 0.95) if len(hist_h) >= 100 else (np.nan, np.nan)}
    v1, v2, v3, v4 = st.columns(4)
    for col, key, label in ((v1, "1d95", "하루 VaR 95%"), (v2, "1d99", "하루 VaR 99%"), (v3, "h95", f"{horizon}거래일 VaR 95%")):
        col.metric(label, f"−{risk[key][0]:.1%}",
                   f"과거 실제 −{past[key][0]:.1%}" if np.isfinite(past[key][0]) else "과거 표본 부족", delta_color="off")
    v4.metric(f"{horizon}거래일 ES 95%", f"−{risk['h95'][1]:.1%}",
              f"과거 실제 −{past['h95'][1]:.1%}" if np.isfinite(past["h95"][1]) else "과거 표본 부족", delta_color="off",
              help="최악 5% 경우의 평균 손실")
    st.caption(f"VaR 95%는 '100번 중 95번은 손실이 이보다 작다'는 뜻입니다. 예를 들어 1,000만 원어치를 들고 있다면 "
               f"하루에 약 {risk['1d95'][0] * 1e7:,.0f}원 넘게 잃을 확률이 5%입니다. ES는 그 나머지 최악 5% 경우의 평균 손실입니다. "
               f"큰 값은 t분포 {garch.name}(꼬리 두께 ν = {garch.nu:.1f}) 시뮬레이션, '과거 실제'는 학습 기간({years}년)의 실제 수익률로 계산했습니다."
               + ((f" 하루 VaR는 GARCH 변동성과 VKOSPI({vk_day.date()} 종가 {vk_last:.1f}, KRX가 하루 늦게 주는 값)를 반반 섞은 변동성으로 계산합니다 — "
                   "GARCH만으로는 이 지수의 하루 손실 위험을 꾸준히 작게 잡아, 미리 정한 규칙으로 검증해 바꿨습니다 (EXPERIMENTS.md)."
                   if np.isfinite(vk_last) else
                   " VKOSPI를 받지 못해 하루 VaR는 GARCH만으로 계산했습니다 (이 지수는 GARCH만으로 위험을 작게 잡는 경향이 있습니다).")
                  if use_iv else
                  (f" 하루 VaR는 GARCH 변동성과 장중 고가·저가로 잰 최근 변동성(연 {np.sqrt(rl_last * 252):.0%})을 반반 섞어 계산합니다 — "
                   "종가만으로는 모르는 장중 흔들림을 반영하며, 미리 정한 규칙으로 35종목에서 검증해 바꿨습니다 (EXPERIMENTS.md)."
                   if np.isfinite(rl_last) else " 장중 고가·저가 일봉을 받지 못해 하루 VaR는 GARCH만으로 계산했습니다.")))
