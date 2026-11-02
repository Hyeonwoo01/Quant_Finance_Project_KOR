"""변동성: GARCH 추정·예측과 실제 변동성."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from quant_engine import GarchVolForecaster


def render(ctx):
    """변동성: GARCH 추정·예측과 실제 변동성. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    fut_idx, garch, horizon, rets = ctx.fut_idx, ctx.garch, ctx.horizon, ctx.rets
    ui.section("Volatility", "실제 변동성과 모델 추정",
               "회색은 최근 21일 동안 실제로 움직인 폭, 파랑은 GARCH 모델이 추정한 변동성과 앞으로의 예측(점선)입니다. "
               "모두 1년 기준으로 환산했습니다.")
    rv21 = rets.rolling(21).std() * np.sqrt(252)
    cv = pd.Series(garch.cond_vol, index=rets.index)
    fut_vol = np.sqrt(garch.daily_var_path(horizon) * 252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv21.index, y=rv21, name="실제 변동성 (최근 21일)", line=dict(color=ui.NEUTRAL)))
    fig.add_trace(go.Scatter(x=cv.index, y=cv, name="모델 추정 (GARCH)", line=dict(color=ui.BLUE)))
    fig.add_trace(go.Scatter(x=fut_idx, y=fut_vol, name="모델 예측 (GARCH)", line=dict(color=ui.BLUE, dash="dash")))
    fig.update_layout(height=420, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    ui.chart(fig)
    a, b, c, d, e = st.columns(5)
    a.metric("α (충격 반응)", f"{garch.alpha:.3f}", help="큰 하루 변동이 다음 날 변동성을 얼마나 키우는지")
    b.metric("β (지속성)", f"{garch.beta:.3f}", help="높아진 변동성이 얼마나 오래 유지되는지")
    c.metric("지속성 (α+γ/2+β)" if garch.asym else "α+β", f"{garch.persistence:.3f}", help="1에 가까울수록 변동성 충격이 오래 지속")
    d.metric("반감기", f"{np.log(0.5) / np.log(garch.persistence):.0f}일" if garch.persistence < 1 else "∞")
    e.metric("꼬리 두께 (ν)", f"{garch.nu:.1f}",
             help="충격 분포(t분포)의 자유도. 작을수록 폭락·폭등이 자주 나옵니다. 정규분포는 ∞, 주식은 보통 4~10")
    for msg in garch.warnings:
        (st.warning if not garch.converged else st.caption)(msg)

    st.markdown("**대칭 GARCH vs GJR-GARCH** — 같은 학습 데이터로 둘 다 적합해 비교")
    gs_, gj_ = GarchVolForecaster().fit(rets.values), GarchVolForecaster(asym=True).fit(rets.values)
    vcmp = pd.DataFrame({m.name: {"α (충격 반응)": m.alpha, "γ (하락한 날 추가 반응)": m.gamma, "β (지속성)": m.beta,
                                  "지속성 (α+γ/2+β)": m.persistence, "꼬리 두께 (ν)": m.nu,
                                  f"{horizon}일 예측 변동성": m.forecast_vol(horizon), "AIC (낮을수록 잘 맞음)": m.aic}
                         for m in (gs_, gj_)}).T
    vcmp.insert(0, "지금 쓰는 모델", ["✔" if n == garch.name else "" for n in vcmp.index])
    st.dataframe(vcmp.style.format({"α (충격 반응)": "{:.3f}", "γ (하락한 날 추가 반응)": "{:.3f}", "β (지속성)": "{:.3f}",
                                    "지속성 (α+γ/2+β)": "{:.3f}", "꼬리 두께 (ν)": "{:.1f}", f"{horizon}일 예측 변동성": "{:.1%}",
                                    "AIC (낮을수록 잘 맞음)": "{:,.1f}"}), width="stretch",
                 column_config={"지금 쓰는 모델": st.column_config.TextColumn("지금 쓰는 모델", alignment="center")})
    st.caption(f"γ가 0보다 크면 하락한 날 다음 날의 변동성이 상승한 날보다 더 커진다는 뜻입니다 (이 종목 γ = {gj_.gamma:.3f}). "
               f"AIC 차이(GARCH − GJR) {gs_.aic - gj_.aic:+.1f}: 양수면 GJR이 데이터에 더 잘 맞고, 보통 2 이상이면 의미 있는 차이로 봅니다. "
               "실제로 앞날을 더 잘 맞히는지는 '예측 정확도 검증' 탭의 과거 검증에서 두 모델의 오차를 비교하세요.")
