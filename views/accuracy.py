"""예측 정확도 검증: GARCH 백테스트, 하루 VaR 검정, 실전 기록 채점."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecast_log as fl
import ui
from quant_engine import VAR_INNOV, VAR_INNOVS, VOL_MODELS
from app_data import get_forecast_log, run_backtest, run_var_backtest
from views.common import _signed


def render(ctx):
    """예측 정확도 검증: GARCH 백테스트, 하루 VaR 검정, 실전 기록 채점. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    close, close_long, horizon, p_up, rets = ctx.close, ctx.close_long, ctx.horizon, ctx.p_up, ctx.rets
    ticker, vol_model = ctx.ticker, ctx.vol_model
    ui.section("Backtest", "GARCH 예측은 과거에 얼마나 맞았나",
               f"과거 여러 시점에서 그때까지의 데이터만으로 {horizon}일 변동성을 다시 예측하고, 실제로 움직인 폭과 비교합니다. "
               "비교 대상은 '최근 변동성이 그대로 이어진다'는 단순 예측입니다. 범위와 VaR는 t분포 GARCH 시뮬레이션으로 판정합니다.")

    st.subheader("실전 기록 — 미리 적어 둔 예측을 나중에 채점")
    st.markdown("과거 검증은 결과를 이미 아는 데이터로 하므로 가정을 나중에 고를 여지가 남습니다. 그래서 평일마다 장 마감 후 그날의 예측을 "
                "파일에 먼저 적어 두고(GitHub Actions), 21거래일 뒤 실제 결과와 비교합니다. 적어 둔 예측은 고치지 않습니다.")
    flog = get_forecast_log(fl.LOG_PATH.stat().st_mtime if fl.LOG_PATH.exists() else 0)
    mine = flog[flog["ticker"].str.upper() == ticker.upper()] if len(flog) else flog
    if flog.empty:
        st.info("아직 기록이 없습니다. main 브랜치에서 GitHub Actions의 'forecast-log'가 평일마다 기록을 쌓기 시작합니다.")
    elif mine.empty:
        st.info(f"{ticker}는 기록 대상이 아닙니다. 기록 중인 종목: {', '.join(sorted(flog['ticker'].unique()))}. "
                "`forecast_log/tickers.txt`에 한 줄 추가하면 다음 기록부터 포함됩니다.")
    else:
        fsc = fl.score(mine, close_long if p_up is not None else close)
        fs = fsc["summary"]
        n_done = fs.get("채점 수", 0)
        late_note = (f" 예약 실행이 늦거나 시세가 하루 늦게 와서 다음 장이 열린 뒤에 적힌 기록이 {fs['늦게 적힌 기록']}개 있습니다. "
                     "예측은 기록 날짜까지의 데이터로만 계산돼 채점에 쓰고, 그중 다음 거래일 장 마감 뒤에 적힌 "
                     f"{fs['심리 지표에서 뺀 기록']}개는 그 뒤의 뉴스가 섞여 시장 심리 지표 채점에서만 뺐습니다."
                     if fs.get("늦게 적힌 기록") else "")
        span = f" ({fs['첫 기록일'].date()} ~ {fs['마지막 기록일'].date()})" if fs["기록 수"] else ""
        st.caption(f"기록 {fs['기록 수']}일{span}, 21거래일 결과가 나온 기록 {n_done}개. "
                   "예측은 앱 기본 설정(방향 21거래일, GARCH 학습 3년)으로 계산했습니다." + late_note)
        if n_done:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("방향 적중률", f"{fs['적중률']:.0%}", f"기본 확률 {fs['기본 확률 적중률']:.0%}", delta_color="off")
            f2.metric("확률 정확도 개선", f"{fs['확률 정확도 개선']:+.1%}", f"z = {fs['개선 z']:.1f}", delta_color="off",
                      help="기록할 때의 '과거 상승 비율'보다 확률이 얼마나 더 정확했는지. z가 2를 넘어야 우연이 아님")
            vol_ref = (f"옵션 IV {fs['옵션 IV 오차']:.1%}" if np.isfinite(fs["옵션 IV 오차"]) else
                       f"변동성 지수 {fs['변동성 지수 오차']:.1%}" if np.isfinite(fs["변동성 지수 오차"]) else "비교 IV 없음")
            f3.metric("변동성 오차 (GARCH)", f"{fs['GARCH 오차']:.1%}", vol_ref, delta_color="off", help="21일 실제 변동성 대비 RMSE")
            f4.metric("68% / 95% 범위 적중", f"{fs['68% 범위 적중']:.0%} / {fs['95% 범위 적중']:.0%}", "목표 68% / 95%", delta_color="off")
            st.caption(f"매일 기록해서 21일 기간이 서로 겹치므로, 독립적인 표본은 약 {max(n_done / 21, 1):.0f}개입니다. "
                       "결론을 내리려면 적어도 몇 달, 가능하면 1년 이상 쌓여야 합니다.")
        else:
            st.caption("아직 21거래일이 지난 기록이 없어 방향·범위·변동성은 채점 전입니다.")
        sc_tab = fs.get("시장 심리 지표")
        if sc_tab is not None and sc_tab["표본"].sum() > 0:
            st.markdown("**시장 심리 지표는 쓸모가 있을까?** — 기록해 둔 지표와 그 뒤 21거래일 수익률·실제 변동성의 관계")
            sc_show = pd.DataFrame({c: sc_tab[c].map(lambda v: f"{v:.0f}") if c == "표본" else sc_tab[c].map(_signed(2 if "상관" in c else 1))
                                    for c in sc_tab.columns})          # 빈 칸을 '—'로 (숫자 열의 결측은 None으로 보임)
            st.dataframe(sc_show, width="stretch",
                         column_config={c: st.column_config.TextColumn(c, alignment="right") for c in sc_show.columns})
            st.caption("과거 기록을 무료로 구할 수 없어 매일 적어 두고 있는 지표들입니다. 기록한 다음 거래일 종가부터 잰 수익률과 비교해서, "
                       "기록 뒤 밤사이 뉴스로 생긴 가격 차이는 빼고 채점합니다. 기록이 매일 겹치므로 겹침을 보정한 t값이 2를 넘어야 우연이 "
                       "아니라고 봅니다. 표본이 10개 미만이면 계산하지 않고, 결론에는 몇 달 이상 필요합니다.")
        if "하루 VaR 95%" in fs:
            v95, v99 = fs["하루 VaR 95%"], fs["하루 VaR 99%"]
            st.caption(f"하루 VaR 예외 (다음 날 손실이 VaR보다 컸던 날): 95% {v95['예외 횟수']}번 / {v95['관측일']}일 (목표 5%), "
                       f"99% {v99['예외 횟수']}번 (목표 1%).")
        with st.expander("기록 보기"):
            st.dataframe(mine.drop(columns=["ticker"]).sort_values("asof", ascending=False), hide_index=True, width="stretch")

    st.subheader("과거 검증")
    if st.button("검증 실행", type="primary"):
        with st.spinner("과거 여러 시점에서 다시 예측하는 중…"):
            bt = run_backtest(rets.values, horizon, vol_model)
        if bt.empty:
            st.warning("데이터가 부족합니다. 학습 데이터 기간을 늘리거나 예측 기간을 줄이세요.")
        else:
            rmse = lambda c: np.sqrt(np.mean((bt[c] - bt["realized"]) ** 2))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("모델 예측 오차", f"{rmse('garch'):.2%}", help="RMSE, 작을수록 정확")
            m2.metric("단순 예측 오차", f"{rmse('naive'):.2%}",
                      f"모델이 {'더 정확' if rmse('garch') < rmse('naive') else '덜 정확'}", delta_color="off")
            m3.metric("68% 범위 안에 든 비율", f"{bt['in1'].mean():.0%}", "목표 68%", delta_color="off")
            m4.metric("95% 범위 안에 든 비율", f"{bt['in2'].mean():.0%}", "목표 95%", delta_color="off")
            m5.metric("VaR 95%를 넘은 손실", f"{bt['var_breach'].mean():.0%}", f"목표 5% · {horizon}일 겹침", delta_color="off")
            dates = rets.index[bt["origin"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=bt["realized"], name="실제 변동성", line=dict(color=ui.INK)))
            fig.add_trace(go.Scatter(x=dates, y=bt["garch"], name="모델 예측", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=dates, y=bt["naive"], name="단순 예측", line=dict(color=ui.NEUTRAL, dash="dot")))
            fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.caption(f"10거래일 간격으로 {len(bt)}번 검증했습니다. 검증 기간이 서로 겹쳐 있어 결과를 너무 믿지는 마세요. "
                       f"같은 시점들에서 대칭 GARCH 오차 {rmse('sym'):.2%}, GJR-GARCH 오차 {rmse('gjr'):.2%} "
                       f"({'GJR' if rmse('gjr') < rmse('sym') else '대칭 GARCH'}가 더 정확, 모델 예측 = {VOL_MODELS[vol_model]}).")

            st.markdown("**하루 VaR 예외 검정** — 매일 다음 날의 VaR를 내고(21거래일마다 다시 적합), 실제 손실이 그보다 컸던 날을 셉니다")
            vt = run_var_backtest(rets.values, vol_model)
            vshow = vt.assign(판정=np.where(vt["종합 p값"].astype(float) > 0.05, "통과", "기각"))
            st.dataframe(vshow.style.format({"관측일": "{:.0f}", "예외 횟수": "{:.0f}", "예외 비율": "{:.1%}", "목표 비율": "{:.0%}",
                                             "Kupiec p값": "{:.2f}", "Christoffersen p값": "{:.2f}", "종합 p값": "{:.2f}",
                                             "예외일 손실 ÷ 예측 ES": "{:.2f}", "ES 검정 p값": "{:.2f}",
                                             "분위수 손실(×10⁴)": "{:.2f}", "재적합 수": "{:.0f}", "추정 실패": "{:.0f}"}, na_rep="—"),
                         width="stretch", column_config={"판정": st.column_config.TextColumn("판정", alignment="center")})
            st.caption("Kupiec: 예외 비율이 목표(5%·1%)와 같은지. Christoffersen: 예외가 특정 시기에 몰려 나오지 않는지 "
                       "(몰리면 변동성 변화를 늦게 따라간다는 뜻). p값이 0.05보다 작으면 그 VaR를 기각합니다. "
                       "하루 단위라 위의 기간 VaR와 달리 표본이 겹치지 않습니다. "
                       "예외일 손실 ÷ 예측 ES는 1에 가까워야 하고, 크면(ES 검정 p값 < 0.05) 꼬리 손실의 크기를 작게 잡은 것입니다. "
                       "분위수 손실은 모델끼리 비교할 때 쓰며 낮을수록 좋습니다. "
                       "최근 약 500일이라 99% VaR는 예외가 5번 정도뿐이어서, 통과해도 극단 손실을 맞힌다는 증거로는 약합니다 — "
                       "첫 3년 뒤 전체 기간으로 검정한 결과는 실제 데이터 검증 보고서(validation/README.md)에 있습니다. "
                       f"꼬리 분포: t분포는 좌우 대칭 모수 분포, 실제 잔차(FHS)는 GARCH로 나눈 과거 충격의 실제 분포(하락 쪽이 더 두꺼우면 그대로 반영). "
                       f"화면의 VaR·ES는 '{VAR_INNOVS[VAR_INNOV]}'을 씁니다 — 바꾸는 규칙은 EXPERIMENTS.md에 미리 적어 두었습니다.")
