"""한국 주식(코스피·코스닥) 예측·검증 프로그램 (Streamlit).

실행:  run.bat  또는  streamlit run app.py
"""
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Cloud는 새 코드를 받아도 이미 불러온 우리 모듈(quant_engine 등)을 그대로 쓸 때가 있다 (예전엔 'Reboot app' 필요).
# 파일이 바뀐 모듈이 있으면 의존 순서대로 모두 다시 불러오고, 예전 코드로 계산해 둔 캐시도 비운다.
_LOCAL_MODULES = ("quant_engine", "kr_costs", "repro", "kr_data", "market_data", "alt_data", "direction", "regime",
                  "universe", "forecast_log", "ui", "app_data", "views.common", "views.price_range",
                  "views.direction_forecast", "views.stock_ranking", "views.sentiment", "views.volatility",
                  "views.market_regime", "views.accuracy")
_loaded = [m for m in _LOCAL_MODULES if m in sys.modules]
if any(getattr(sys.modules[m], "_loaded_mtime", None) != os.path.getmtime(sys.modules[m].__file__) for m in _loaded):
    for _m in _loaded:
        importlib.reload(sys.modules[_m])
    st.cache_data.clear()

import direction as dr
import kr_costs as kc
import repro
import ui
from quant_engine import VAR_IV_TICKERS, VOL_MODELS, day1_var_factor, fit_vol_model, var_es
from app_data import find_ticker, get_direction, get_market, get_range_level, get_tax_kind, get_vkospi, sell_tax_for
from views import accuracy, direction_forecast, market_regime, price_range, sentiment, stock_ranking, volatility

for _m in _LOCAL_MODULES:                    # 다음 실행 때 파일이 바뀌었는지 비교할 기준
    if _m in sys.modules:
        sys.modules[_m]._loaded_mtime = os.path.getmtime(sys.modules[_m].__file__)

st.set_page_config(page_title="한국 주식 예측기", page_icon="📈", layout="wide")
ui.apply()


# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")
    query = st.text_input("종목명 또는 종목코드", "삼성전자",
                          help="코스피·코스닥 주식과 ETF, 지수. 예: 삼성전자, 에코프로, 005930, KODEX 200, 코스피, 코스닥")
    with st.spinner("종목 찾는 중…"):
        matches = find_ticker(query)
    if not matches:
        st.error(f"'{query}'에 해당하는 한국 종목을 찾지 못했어요. 이름을 다르게 쓰거나 6자리 종목코드를 입력해 보세요. "
                 "(이 프로그램은 코스피·코스닥 종목만 다룹니다.)")
        st.stop()
    if len(matches) == 1:
        ticker = matches[0][0]
        if matches[0][1] != ticker:
            st.caption(f"→ {matches[0][1]}")
    else:
        ticker = st.selectbox(f"검색 결과 {len(matches)}개", matches, format_func=lambda m: m[1])[0]
    horizon = st.slider("예측 기간 (거래일)", 5, 252, 63, help="주가 범위·변동성을 예측할 기간. 21일 ≈ 1개월, 63일 ≈ 3개월")
    years = st.slider("학습 데이터 (년)", 2, 10, 3, help="변동성 모델이 참고할 과거 기간")
    vol_model = st.selectbox("변동성 모델", list(VOL_MODELS), format_func=VOL_MODELS.get,
                             help="GJR-GARCH는 하락한 날 다음 변동성이 더 커지는 효과(주식에서 흔함)를 반영합니다. "
                                  "자동은 두 모델 중 AIC(데이터에 맞는 정도에서 변수 수만큼 벌점)가 낮은 쪽을 고릅니다.")
    dir_h = st.select_slider("방향 예측 기간 (거래일)", [5, 10, 21, 42, 63], 21, help="이 기간 뒤 주가가 지금보다 높을지 예측")
    st.subheader("백테스트 매매 비용")
    cost_bps = st.number_input("수수료·체결 비용 (bp, 살 때·팔 때 각각)", 0.0, 50.0, 3.0, 0.5,
                               help="증권사 수수료 + 호가 차이로 불리하게 체결되는 비용. 1bp = 0.01%. 대형주는 합쳐서 보통 2~5bp")
    use_tax = st.checkbox("증권거래세 반영 (팔 때, 연도별 세율)", True,
                          help="코스피·코스닥 주식은 팔 때 0.15~0.30%(연도별, 농어촌특별세 포함)의 세금이 붙습니다. "
                               "ETF는 면제, 지수는 직접 사고팔 수 없어 세금 없이 계산합니다.")
    exec_lag = st.selectbox("백테스트 체결 시점", [1, 0], format_func={1: "다음 날 종가 (기본·보수적)",
                                                                  0: "신호 낸 날 종가 (낙관적)"}.get,
                            help="신호는 그날 종가까지의 데이터로 계산하므로, 같은 종가에 사고팔았다고 보면 실제로는 "
                                 "불가능한 체결을 가정하게 됩니다. 기본값은 다음 날 종가에 체결한다고 봅니다. "
                                 "방향 예측·시장 국면·코스피 종목 고르기에 모두 적용됩니다.")
    exec_word = "다음 날 종가" if exec_lag else "신호 낸 날 종가"
    st.button("새로고침", on_click=st.cache_data.clear, width="stretch")

hero_slot = st.empty()

try:
    with st.spinner(f"{ticker} 데이터 불러오는 중…"):
        close, spot, r, q, r_src = get_market(ticker, years)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

rets = np.log(close).diff().dropna()
garch = fit_vol_model(rets.values, vol_model)
rv_h = garch.forecast_vol(horizon)

fut_idx = pd.bdate_range(close.index[-1], periods=horizon + 1)[1:]      # 앞으로 horizon거래일 (가격 범위·변동성 탭)
sims = garch.simulate(horizon)                                      # (경로, 일) 누적 로그수익률 — 요약 카드와 1번 탭에서 같이 씀
# 하루 VaR σ 보정 (EXPERIMENTS.md 판정): 코스피 200을 따르는 종목은 GARCH와 마지막 종가일의 전 거래일 VKOSPI를 (C1L2, 2026-10-30),
# 나머지는 GARCH와 장중 고가·저가로 잰 최근 실현 분산을 반반 섞는다
use_iv = ticker in VAR_IV_TICKERS
vk_day = close.index[-2] if len(close) > 1 else close.index[-1]       # 전전 거래일 (KRX는 그날 값을 다음 영업일 아침에 줌, C1L2)
vk_last = get_vkospi(vk_day) if use_iv else np.nan
rl_last = np.nan if use_iv else get_range_level(ticker, years, close.index[-1])
day1 = day1_var_factor(np.sqrt(garch.sigma2_next), use_iv, vk_last, rl_last) * sims[:, 0]
kind = get_tax_kind(ticker)
tax_on = use_tax and kind in ("KOSPI", "KOSDAQ")
cost_word = f"수수료·체결 비용 {cost_bps:g}bp" + (" + 증권거래세" if tax_on else "")
price_fmt = (lambda v: f"{v:,.2f}") if ticker.startswith("^") else (lambda v: f"{v:,.0f}")


# ---------------- 상단 요약 카드 ----------------
def show_hero(dir_metric):
    name = dict(matches).get(ticker, ticker)
    with hero_slot.container():
        lo68, hi68 = spot * np.exp(np.quantile(sims[:, -1], [0.16, 0.84]))
        v95 = var_es(day1, 0.95)[0]
        ui.hero("KOSPI · KOSDAQ", name, price_fmt(spot), close.pct_change().iloc[-1],
                [("기준일", close.index[-1].date()), ("학습 데이터", f"{years}년"), ("구분", kc.KIND_LABELS[kind])],
                [(f"예측 변동성 · {horizon}일", f"{rv_h:.1%}",
                  f"{garch.name} · 1년 기준 · 평소 수준 {np.sqrt(garch.long_run_var * 252):.1%}", False),
                 (f"{horizon}거래일 68% 범위", f"{price_fmt(lo68)} ~ {price_fmt(hi68)}", "이 안에 머물 확률 약 68%", False),
                 ("하루 손실 위험 (VaR 95%)", f"−{v95:.1%}", f"1천만 원어치면 약 {v95 * 1e7:,.0f}원", False),
                 dir_metric],
                "모든 수치는 그 시점까지의 데이터로만 계산한 추정치이며, 투자 조언이 아닙니다.")


show_hero((f"{dir_h}일 뒤 상승 확률", "…", "방향 모델 학습 중", False))

pick = now_all = contrib = dir_asof = wf = close_long = alt_status = dir_err = None
try:
    with st.spinner("방향 예측 모델 학습 중…"):
        p_up, pick, now_all, contrib, dir_asof, wf, close_long, alt_status = get_direction(ticker, dir_h)
    ev = dr.evaluate(wf, dir_h, np.log(close_long).diff(), cost_bps, lag=exec_lag,
                     sell_tax=sell_tax_for(close_long.index, kind, use_tax))
except Exception as e:
    p_up, ev, dir_err = None, None, e
# 확률 정확도(Brier)가 기본 확률보다 좋고, 적중률 우위가 통계적으로 유의할 때만 '검증 통과'
reliable = ev is not None and ev["skill"] > 0 and ev["z"] > 2

show_hero((f"{dir_h}일 뒤 상승 확률", f"{p_up:.0%}" if p_up is not None else "—",
           ("과거 검증 통과" if reliable else "과거 상승 비율" if pick == "base" else "과거 검증 미통과 · 참고용")
           if p_up is not None else "데이터 부족", reliable))

tab1, tab_dir, tab_pick, tab_snap, tab2, tab_regime, tab4 = st.tabs(
    ["주가 예측 범위", "방향 예측", "코스피 종목 고르기", "수급·공시·뉴스", "변동성", "시장 국면", "예측 정확도 검증"])


# ---------------- 탭 (화면은 views/) ----------------
ctx = SimpleNamespace(alt_status=alt_status, close=close, close_long=close_long, contrib=contrib, cost_bps=cost_bps,
                       cost_word=cost_word, day1=day1, dir_asof=dir_asof, dir_err=dir_err, dir_h=dir_h, ev=ev,
                       exec_lag=exec_lag, exec_word=exec_word, fut_idx=fut_idx, garch=garch, horizon=horizon,
                       kind=kind, now_all=now_all, p_up=p_up, pick=pick, q=q, r=r, reliable=reliable, rets=rets,
                       rl_last=rl_last, sims=sims, spot=spot, tax_on=tax_on, ticker=ticker, use_iv=use_iv,
                       use_tax=use_tax, vk_day=vk_day, vk_last=vk_last, vol_model=vol_model, wf=wf, years=years)
with tab1:
    price_range.render(ctx)
with tab_dir:
    direction_forecast.render(ctx)
with tab_pick:
    stock_ranking.render(ctx)
with tab_snap:
    sentiment.render(ctx)
with tab2:
    volatility.render(ctx)
with tab_regime:
    market_regime.render(ctx)
with tab4:
    accuracy.render(ctx)

st.caption(f"재현 정보 — 코드 {repro.code_version()} · 주가 데이터 {repro.fingerprint(close)}"
           + (f" · 방향 예측 데이터 {repro.fingerprint(close_long)}" if p_up is not None else "")
           + ". 같은 지문이면 같은 데이터로 계산한 결과입니다 (무료 데이터는 과거 값이 나중에 고쳐지기도 합니다).")
ui.footer("교육·연구용 고지:", "이 프로그램의 예측과 신호는 과거 데이터로 계산한 참고 자료이며 투자 조언이 아닙니다. "
          "과거 검증 결과가 미래 성과를 보장하지 않습니다.",
          "주가는 Yahoo Finance, 종목 목록·수급·뉴스는 네이버 증권(비공식), 공시·재무는 DART, 금리는 한국은행 ECOS에서 받습니다. "
          "무료 데이터는 늦거나 바뀔 수 있습니다.")
