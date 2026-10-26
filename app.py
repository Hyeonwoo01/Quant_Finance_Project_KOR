"""한국 주식(코스피·코스닥) 예측·검증 프로그램 (Streamlit).

실행:  run.bat  또는  streamlit run app.py
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Streamlit Cloud는 새 코드를 받아도 이미 불러온 우리 모듈(quant_engine 등)을 그대로 쓸 때가 있다 (예전엔 'Reboot app' 필요).
# 파일이 바뀐 모듈이 있으면 의존 순서대로 모두 다시 불러오고, 예전 코드로 계산해 둔 캐시도 비운다.
_LOCAL_MODULES = ("quant_engine", "kr_costs", "repro", "kr_data", "market_data", "alt_data", "direction", "regime",
                  "universe", "forecast_log", "ui")
_loaded = [m for m in _LOCAL_MODULES if m in sys.modules]
if any(getattr(sys.modules[m], "_loaded_mtime", None) != os.path.getmtime(sys.modules[m].__file__) for m in _loaded):
    for _m in _loaded:
        importlib.reload(sys.modules[_m])
    st.cache_data.clear()

import alt_data as ad
import direction as dr
import forecast_log as fl
import kr_costs as kc
import kr_data
import market_data as md
import regime as rg
import repro
import ui
import universe as uv
from quant_engine import GarchVolForecaster, hac_tstat, max_drawdown, timing_attribution, var_backtest, var_es
from quant_engine import VOL_MODELS, fit_vol_model

for _m in _LOCAL_MODULES:                    # 다음 실행 때 파일이 바뀌었는지 비교할 기준
    if _m in sys.modules:
        sys.modules[_m]._loaded_mtime = os.path.getmtime(sys.modules[_m].__file__)

st.set_page_config(page_title="한국 주식 예측기", page_icon="📈", layout="wide")
ui.apply()


# ---------------- 데이터 (캐시) ----------------
KR_INDEX = {"^KS11": "코스피 지수", "^KQ11": "코스닥 지수", "^KS200": "코스피 200 지수"}
_INDEX_NAMES = {"코스피": "^KS11", "kospi": "^KS11", "코스피지수": "^KS11", "코스닥": "^KQ11", "kosdaq": "^KQ11",
                "코스닥지수": "^KQ11", "코스피200": "^KS200", "kospi200": "^KS200"}


def is_kr(ticker):
    t = ticker.upper()
    return t.endswith((".KS", ".KQ")) or t in KR_INDEX


@st.cache_data(ttl=3600, show_spinner=False)
def find_ticker(query):
    """종목명 또는 코드 → [(티커, 표시 이름)], 한국 종목·ETF·지수만. 종목 목록(네이버)은 하루 캐시."""
    key = "".join(str(query).split()).lower()
    if key in _INDEX_NAMES:
        t = _INDEX_NAMES[key]
        return [(t, f"{KR_INDEX[t]} ({t})")]
    if str(query).strip().upper() in KR_INDEX:
        t = str(query).strip().upper()
        return [(t, f"{KR_INDEX[t]} ({t})")]
    return [m for m in md.search_ticker(query) if is_kr(m[0])]


@st.cache_data(ttl=86400, show_spinner=False)
def get_tax_kind(ticker):
    """증권거래세 구분 (코스피·코스닥 주식 / ETF는 면제 / 지수)."""
    name, _, sec_type = md.kr_info(ticker)
    return kc.tax_kind(ticker, name, sec_type)


def sell_tax_for(index, kind, use_tax):
    """백테스트에 쓸 날짜별 매도 세율 (끄면 0)."""
    return kc.sell_tax_rate(index, kind) if use_tax else 0.0


@st.cache_data(ttl=900, show_spinner=False)
def get_market(ticker, years):
    close = md.load_history(ticker, years)
    spot = float(close.iloc[-1])
    r, r_src = md.risk_free_rate(ticker)     # 원화 금리 (CD 91일, ECOS 키가 있을 때)
    q = md.dividend_yield(ticker, spot)
    return close, spot, r, q, r_src


@st.cache_data(ttl=3600, show_spinner=False)
def get_direction_inputs(ticker):
    """방향 모델 입력: 최대 10년치 종가 + 거시 + 대체 데이터. 예측 기간과 무관하므로 따로 캐시한다."""
    close = md.load_history(ticker, 10)
    macro = md.load_macro(ticker)
    alt, alt_status = ad.build_alt_features(ticker, close.index)
    return close, macro, alt, alt_status


@st.cache_data(ttl=900, show_spinner=False)
def get_direction(ticker, h):
    """방향 모델 학습(무거움). 표본이 많이 필요해서 학습 기간 설정과 별개로 최대 10년치를 쓴다.

    거래비용·체결 시점은 채점(dr.evaluate)에만 쓰이므로 여기 넣지 않는다 → 그 설정을 바꿔도 다시 학습하지 않음.
    """
    close, macro, alt, alt_status = get_direction_inputs(ticker)
    # 거시 지표 시차는 티커로 자동 결정 (한국 종목이면 미국 지표를 하루 늦춤)
    wf, _ = dr.walk_forward(close, h, macro, alt=alt, ticker=ticker)
    if wf.empty:
        raise ValueError("과거 데이터가 너무 짧습니다 (최소 약 3년 필요)")
    p_up, pick, now_all, contrib, asof = dr.predict_now(close, h, wf, macro, alt=alt, ticker=ticker)
    alt_status["시장 옵션 심리(SKEW·VVIX)"] = "사용" if macro is not None and "skew" in macro else "데이터 없음"
    alt_status["거시(VIX·금리·신용)"] = "사용" if macro is not None else "데이터 없음"
    if not md.is_us_ticker(ticker):
        alt_status["환율·코스닥 위험선호"] = "사용" if macro is not None and "usdkrw" in macro else "데이터 없음"
        alt_status["한국은행 금리(ECOS)"] = ("키 없음" if not kr_data.get_key("ECOS_API_KEY") else
                                       "사용" if macro is not None and "kr3y" in macro else "오류")
    return p_up, pick, now_all, contrib, asof, wf, close, alt_status


@st.cache_data(ttl=900, show_spinner=False)
def get_direction_robustness(ticker, h, cost_bps, lag, use_tax, kind):
    """방향 예측을 가정을 바꿔 다시 채점 (이미 계산된 워크포워드 예측 재사용, 재학습 없음)."""
    res = get_direction(ticker, h)
    wf, close = res[5], res[6]
    return dr.robustness(wf, h, np.log(close).diff(), cost_bps, lag=lag, sell_tax=sell_tax_for(close.index, kind, use_tax))


@st.cache_data(ttl=3600, show_spinner=False)
def get_direction_horizons(ticker, cost_bps, lag, use_tax, kind):
    """예측 기간 5·10·21·42·63일로 워크포워드를 다시 돌린다 (빠른 점검용: 부스팅·앙상블 제외)."""
    close, macro, alt, _ = get_direction_inputs(ticker)
    return dr.horizon_check(close, macro, alt, ticker, cost_bps, lag=lag, sell_tax=sell_tax_for(close.index, kind, use_tax))


@st.cache_data(ttl=3600, show_spinner=False)
def get_direction_horizons_full(ticker, cost_bps, lag, use_tax, kind):
    """부스팅까지 포함한 예측 기간 점검 — 무거워서 1년마다 재학습.

    같은 1년 주기로 '로지스틱만' 점검도 함께 돌려, 두 결과의 차이가 부스팅 때문인지만 비교되게 한다.
    """
    close, macro, alt, _ = get_direction_inputs(ticker)
    tax = sell_tax_for(close.index, kind, use_tax)
    full = dr.horizon_check(close, macro, alt, ticker, cost_bps, light=False, refit_every=252, lag=lag, sell_tax=tax)
    light = dr.horizon_check(close, macro, alt, ticker, cost_bps, light=True, refit_every=252, lag=lag, sell_tax=tax)
    return full, light


def _signed(digits=2):
    """부호 붙인 숫자. 반올림해서 0이면 부호 없이 — 아주 작은 음수가 '-0.00'으로 보이지 않게."""
    return lambda v: "—" if pd.isna(v) else f"{0:.{digits}f}" if round(v, digits) == 0 else f"{v:+.{digits}f}"


def signed_table(df, digits=2):
    """+/− 차이 표: +는 초록, −는 빨강, 반올림해서 0이면 색 없음."""
    d = df.astype(float).round(digits) + 0.0
    color = lambda v: f"color: {ui.APPROVE}" if v > 0 else f"color: {ui.REJECT}" if v < 0 else ""
    if not d.isna().any().any():
        st.dataframe(d.style.format(_signed(digits)).map(color), width="stretch")
        return
    # 빈 칸이 있으면 글자로 바꿔 '—'로 보이게 (Streamlit은 숫자 열의 결측을 None으로 표시한다)
    styles = d.map(color)
    txt = d.map(_signed(digits))
    st.dataframe(txt.style.apply(lambda _: styles, axis=None), width="stretch",
                 column_config={c: st.column_config.TextColumn(str(c), alignment="right") for c in txt.columns})


def robust_table(df):
    """방향 예측 점검 표 (검증 통과는 글자로)."""
    d = df.assign(**{"검증 통과": df["검증 통과"].map({True: "통과", False: "—"})})
    first = [c for c in ("가장 많이 고른 모델", "부스팅·앙상블 선택 비율") if c in d]
    d = d[first + [c for c in d.columns if c not in first]]
    st.dataframe(d.style.format({"적중률": "{:.1%}", "'항상 오른다' 대비": "{:+.1%}", "확률 정확도 개선": "{:+.1%}",
                                 "z": _signed(1), "전략 샤프": "{:.2f}", "단순 보유 샤프": "{:.2f}", "전략 최대 낙폭": "{:.0%}",
                                 "보유 비중": "{:.0%}", "부스팅·앙상블 선택 비율": "{:.0%}"}, na_rep="—"), width="stretch",
                 column_config={"검증 통과": st.column_config.TextColumn("검증 통과", alignment="center")})


@st.cache_data(show_spinner=False)
def get_kr_factors(stamp):
    """매주 계산해 둔 한국판 일간 요인. stamp: 파일 시각 — 새로 커밋되면 다시 읽는다."""
    return uv.load_kr_factors()


def attribution_view(daily):
    """보유/현금 타이밍 전략의 수익 분해 표 + 해석: 같은 비중으로 계속 들고 있었을 때보다 나았는지 (타이밍 실력),
    한국판 요인(시장·소형주·가치·수익성·모멘텀)으로 설명되지 않는 부분이 있는지."""
    ff = get_kr_factors(uv.kr_factors_stamp())
    att = timing_attribution(daily, ff)
    sm = att["summary"]
    st.dataframe(sm.style.format({"알파(연)": "{:+.1%}", "알파 t값": _signed(1), "노출(β)": "{:.2f}", "R²": "{:.2f}",
                                  "관측일": "{:,.0f}"}, na_rep="—"), width="stretch")
    a = sm.iloc[0]
    txt = (f"첫 줄: 이 전략은 평균적으로 주식을 {a['노출(β)']:.0%}만큼 들고 있던 셈이고, 그 비중으로 계속 들고 있었을 때보다 "
           f"1년에 {abs(a['알파(연)']):.1%} {'더' if a['알파(연)'] >= 0 else '덜'} 벌었습니다 (t = {a['알파 t값']:.1f}). "
           "t가 2를 넘어야 타이밍 실력이 우연이 아니라고 봅니다.")
    if ff is None:
        txt += " 한국판 요인 자료(ranking/kr_factors.csv)가 아직 없어 요인 분석은 생략했습니다."
    elif "전략 (요인 모형)" in sm.index:
        s_, b_ = sm.loc["전략 (요인 모형)"], sm.loc["단순 보유 (요인 모형)"]
        txt += (f" 아래 두 줄: 한국 주식의 알려진 요인(시장·소형주·가치·수익성·모멘텀)으로 설명되는 부분을 빼고 남는 수익입니다. "
                f"전략 {s_['알파(연)']:+.1%} vs 단순 보유 {b_['알파(연)']:+.1%} — 전략 쪽이 더 크면 그 차이가 요인으로 설명되지 않는 "
                "타이밍 효과입니다. 요인은 코스피 거래대금 상위 200종목으로 매주 직접 만든 근사이고(무료 한국 요인 자료가 없음), "
                "현금은 CD 91일 금리를 번다고 봤습니다. 요인은 매주 금요일에 갱신돼 그 뒤 며칠은 빠집니다.")
    st.caption(txt)
    if att["loadings"] is not None:
        with st.expander("요인별 노출 자세히 (계수 · t값)"):
            st.dataframe(att["loadings"].style.format("{:+.2f}", na_rep="—"), width="stretch")


@st.cache_data(show_spinner=False)
def get_forecast_log(stamp):
    """stamp: 기록 파일 수정 시각 — 새 기록이 커밋되면 다시 읽는다."""
    return fl.read_log(fl.LOG_PATH)


@st.cache_data(ttl=900, show_spinner=False)
def get_snapshot(ticker, spot, r, q):
    """과거 기록이 무료로 없어 검증할 수 없는 '현재 값' 지표들."""
    out = {}
    for key, fn, args in (("options", ad.options_snapshot, (ticker, spot, r, q)),
                          ("revisions", ad.estimate_revisions, (ticker,)), ("news", ad.news_sentiment, (ticker,))):
        try:
            out[key] = fn(*args)
        except Exception:
            out[key] = None
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def get_universe(h, cost_bps, data_stamp, lag, use_tax):
    """data_stamp: 데이터 갱신 시각 — 새로 받으면 다시 계산되도록 캐시 키에 넣는다. lag: 체결 시점(거래일)."""
    panel = uv.build_panel()
    wf = uv.walk_forward(panel, h, exec_lag=lag)
    return uv.results(panel, wf, cost_bps, use_tax)


@st.cache_data(show_spinner=False)
def get_weekly(stamp):
    """GitHub Actions가 매주 계산해 둔 결과와 추천 기록. stamp: 결과 파일 시각 — 새 결과가 오면 다시 읽도록 캐시 키에 넣는다."""
    return uv.load_weekly(), uv.read_picks()


@st.cache_data(ttl=3600, show_spinner=False)
def get_regime(ticker, cost_bps, lag, use_tax, kind):
    """국면 모델은 표본이 많이 필요해서 학습 기간 설정과 별개로 최대 10년치를 쓴다."""
    close = md.load_history(ticker, 10)
    return rg.analyze(close, cost_bps, lag=lag, sell_tax=sell_tax_for(close.index, kind, use_tax))


@st.cache_data(show_spinner=False)
def run_var_backtest(returns, model):
    """하루 VaR 95%·99%의 워크포워드 예외 검정 (Kupiec·Christoffersen). model: 사이드바 변동성 모델."""
    return var_backtest(np.asarray(returns), model=model)


@st.cache_data(show_spinner=False)
def run_backtest(returns, h, model="garch", step=10):
    """과거 여러 시점에서 '그때까지 데이터만으로' 예측 → 이후 실제와 비교 (look-ahead 없음).

    범위·VaR는 앱 화면과 같은 방식(사이드바 변동성 모델의 t분포 시뮬레이션 분위수)으로 판정한다.
    비교용으로 대칭 GARCH(sym)와 GJR-GARCH(gjr)의 예측도 같은 시점마다 함께 남긴다.
    """
    r = np.asarray(returns)
    start = max(250, len(r) - 500)
    rows = []
    for o in range(start, len(r) - h + 1, step):
        win = r[max(0, o - 750):o]
        gs, gj = GarchVolForecaster().fit(win), GarchVolForecaster(asym=True).fit(win)
        g = {"garch": gs, "gjr": gj}.get(model) or (gj if gj.converged and gj.aic < gs.aic else gs)    # 자동 = AIC
        f = g.forecast_vol(h)
        real = np.sqrt(np.mean(r[o:o + h] ** 2) * 252)
        naive = np.sqrt(np.mean(r[o - max(h, 21):o] ** 2) * 252)
        q05, q025, q16, q84, q975 = np.quantile(g.simulate(h, 2000, seed=o)[:, -1], [0.05, 0.025, 0.16, 0.84, 0.975])
        move = r[o:o + h].sum()
        rows.append((o, f, naive, real, q16 <= move <= q84, q025 <= move <= q975, move < q05,
                     gs.forecast_vol(h), gj.forecast_vol(h)))
    return pd.DataFrame(rows, columns=["origin", "garch", "naive", "realized", "in1", "in2", "var_breach", "sym", "gjr"])


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

sims = garch.simulate(horizon)                                      # (경로, 일) 누적 로그수익률 — 요약 카드와 1번 탭에서 같이 씀
kind = get_tax_kind(ticker)
tax_on = use_tax and kind in ("KOSPI", "KOSDAQ")
cost_word = f"수수료·체결 비용 {cost_bps:g}bp" + (" + 증권거래세" if tax_on else "")
price_fmt = (lambda v: f"{v:,.2f}") if ticker.startswith("^") else (lambda v: f"{v:,.0f}")


# ---------------- 상단 요약 카드 ----------------
def show_hero(dir_metric):
    name = dict(matches).get(ticker, ticker)
    with hero_slot.container():
        lo68, hi68 = spot * np.exp(np.quantile(sims[:, -1], [0.16, 0.84]))
        v95 = var_es(sims[:, 0], 0.95)[0]
        ui.hero("KOSPI · KOSDAQ", name, price_fmt(spot), close.pct_change().iloc[-1],
                [("기준일", close.index[-1].date()), ("학습 데이터", f"{years}년"), ("구분", kc.KIND_LABELS[kind])],
                [(f"예측 변동성 · {horizon}일", f"{rv_h:.1%}",
                  f"{garch.name} · 1년 기준 · 평소 수준 {np.sqrt(garch.long_run_var * 252):.1%}", False),
                 (f"{horizon}거래일 68% 범위", f"{price_fmt(lo68)} ~ {price_fmt(hi68)}", "이 안에 머물 확률 약 68%", False),
                 ("하루 손실 위험 (VaR 95%)", f"−{v95:.1%}", f"1천만 원어치면 약 {v95 * 1e7:,.0f}원", False),
                 dir_metric],
                "모든 수치는 그 시점까지의 데이터로만 계산한 추정치이며, 투자 조언이 아닙니다.")


show_hero((f"{dir_h}일 뒤 상승 확률", "…", "방향 모델 학습 중", False))

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

# ---------------- 1. 주가 예측 범위 ----------------
with tab1:
    ui.section("Price Range Forecast", f"앞으로 {horizon}거래일, 주가가 움직일 범위",
               "최근 변동성을 바탕으로 주가가 얼마나 크게 움직일 수 있는지 보여줍니다. "
               "진한 띠 안에 머물 확률은 약 68%, 연한 띠는 약 95%입니다. 폭락·폭등이 정규분포보다 자주 나오는 점을 "
               f"t분포 {garch.name}로 반영해 1만 번 시뮬레이션했습니다. 범위의 중심은 <b>현재가</b>입니다(평균 수익 0 가정) — "
               "기대 가격이나 오를지 내릴지의 예측이 아니라 불확실성의 크기입니다. 오를지 내릴지는 <b>방향 예측</b> 탭에서 봅니다.")
    if not garch.converged:
        st.warning(garch.warnings[0] + " 범위와 VaR는 이 기본값으로 계산했으니 참고만 하세요.")
    cum_sd = np.sqrt(np.cumsum(garch.daily_var_path(horizon)))     # k일 누적 로그수익률 표준편차
    fut_idx = pd.bdate_range(close.index[-1], periods=horizon + 1)[1:]
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
    risk = {"1d95": var_es(sims[:, 0], 0.95), "1d99": var_es(sims[:, 0], 0.99), "h95": var_es(sims[:, -1], 0.95)}
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
               f"큰 값은 t분포 {garch.name}(꼬리 두께 ν = {garch.nu:.1f}) 시뮬레이션, '과거 실제'는 학습 기간({years}년)의 실제 수익률로 계산했습니다.")

# ---------------- 방향 예측 ----------------
with tab_dir:
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
            label, kind = (("모든 경우 통과", "approve") if n_pass == n_all else
                           ("절반 이상 통과", "elective") if n_pass * 2 >= n_all else
                           ("일부에서만 통과", "elective") if n_pass else ("한 번도 통과 못 함", "reject"))
            ui.verdict(f"{dir_h}거래일 방향 예측 · 선택 기간 {len(bw)}가지 + 시기 {len(bp)}구간", f"{n_pass} / {n_all}",
                       "경우에서 검증 통과", [(label, kind)],
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

# ---------------- 코스피 종목 고르기 ----------------
with tab_pick:
    ui.section("KOSPI Ranking", "코스피 대형주 중 상대적으로 더 오를 종목",
               "한 종목의 방향 대신, 코스피 대형주 약 200개를 서로 비교해 순위를 매깁니다. "
               "사이드바의 종목 설정과 무관하게 동작합니다.")
    info = uv.data_info()
    weekly, picks = get_weekly(uv.weekly_stamp())

    def run_update():
        bar = st.progress(0.0, text="시작…")
        uv.update_data(progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
        st.cache_data.clear()
        st.rerun()

    download_note = ("처음 받을 때 코스피 전 종목 주가와, 한 번이라도 200위 안에 든 종목의 "
                     "6년치 수급·실적·공시·재무제표를 받느라 **1시간 안팎** 걸립니다. "
                     "(터미널에서 `python universe.py update`로 받아도 됩니다.)")
    if info is None and weekly is None:
        st.info("아직 데이터가 없습니다. " + download_note)
        if st.button("데이터 받기", type="primary"):
            run_update()
    else:
        U = None
        if info is None:                 # 이 컴퓨터엔 데이터가 없음 → GitHub Actions가 매주 계산해 둔 결과
            c1, c2 = st.columns([3, 1])
            meta = weekly.get("meta") or {}
            span = f"코스피 {meta['n_all']}종목 중 과거 200위 이내 진입 {meta['n']}종목 · " if "n_all" in meta and "n" in meta else ""
            c1.caption(f"GitHub Actions가 매주 장 마감 뒤 계산해 둔 결과 · 데이터 {weekly['data_end'].date()}까지 "
                       f"(계산 {weekly['updated']}) · {span}수수료·체결 비용 {weekly['cost_bps']:g}bp (매수·매도 각각)")
            hold = c2.selectbox("종목 교체 주기", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
            U = weekly["results"].get((hold, exec_lag, use_tax))
            if U is None:
                st.warning("이 설정 조합의 미리 계산된 결과가 없습니다. 아래에서 데이터를 받아 직접 계산하세요.")
            with st.expander("수수료를 바꿔 직접 계산하려면"):
                st.caption("이 컴퓨터에 데이터를 받으면 수수료를 바꿔 가며 바로 계산합니다. " + download_note)
                if st.button("데이터 받기"):
                    run_update()
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.caption(f"데이터 갱신: {info['updated']} · 코스피 {info.get('n_all', '?')}종목 중 "
                       f"과거 200위 이내 진입 {info['n']}종목 · {info['years']}년")
            hold = c2.selectbox("종목 교체 주기", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
            pick_cost = c3.number_input("수수료·체결 비용 (매수·매도 각각, bp)", 0.0, 100.0, 5.0, 1.0,
                                        help="수수료 + 호가 차이·시장 충격. 증권거래세는 사이드바 설정에 따라 판 비중만큼 연도별 세율로 따로 뺍니다.")
            if st.button("데이터 업데이트 (새 날짜만 받음)"):
                run_update()
            try:
                with st.spinner("과거 시점마다 모델을 학습·검증하는 중… (처음 1~3분)"):
                    U = get_universe(hold, pick_cost, info["updated"], exec_lag, use_tax)
            except Exception as e:
                st.error(f"계산 실패: {e}")

        if U is not None:
            auto = U["table"].loc["auto"]
            ok = auto["초과수익 t값"] > 2 and auto["IC t값"] > 2
            pick_label = "고르지 않음 — 최근 성적이 좋은 모델이 없어, 전체 종목에 똑같이 나눠 담기를 권장" if U["pick"] == "none" \
                else uv.MODEL_LABELS[U["pick"]]
            ui.verdict(f"{U['date'].date()} 기준", pick_label, "",
                       [("과거 검증 통과", "approve") if ok else ("평균보다 낫다고 보기 어려움", "elective")])
            if ok:
                st.success(f"과거 검증에서 상위 20% 종목이 비용을 빼고도 전체 평균보다 확실히 더 올랐습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, 우연일 가능성 낮음 t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}). 그래도 앞으로의 성과를 보장하지 않습니다.")
            else:
                st.warning(f"과거 검증에서 상위 20% 종목이 전체 평균보다 더 올랐다고 확신하기 어렵습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}; t가 2를 넘어야 우연이 아니라고 봅니다). 아래 순위는 참고로만 보세요.")

            cv = U["curves"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["상위 20% (비용 후)"], name="상위 20% 종목 (비용 뺀 후)",
                                     line=dict(color=ui.BLUE, width=2.5)))
            fig.add_trace(go.Scatter(x=cv["_universe"].index, y=cv["_universe"], name="전체 종목 평균",
                                     line=dict(color=ui.NEUTRAL)))
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["하위 20%"], name="하위 20% 종목",
                                     line=dict(color=ui.PURPLE, dash="dot")))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.caption(f"최대 낙폭: 상위 20% {max_drawdown(cv['auto']['상위 20% (비용 후)']):.0%} · "
                       f"전체 종목 평균 {max_drawdown(cv['_universe']):.0%}. "
                       "리밸런싱 시점 기준이라 그 사이에 잠깐 더 빠졌던 하락은 덜 잡힙니다. "
                       f"종목을 고른 날의 데이터로 순위를 매기고 {exec_word}에 사고판다고 봤습니다.")

            if len(picks):
                st.markdown("**실전 기록** — 매주 장 마감 뒤 GitHub Actions가 그 주의 상위 20% 종목을 미리 적어 두고, "
                            "21·63거래일 뒤 실제 성적을 채점합니다 (교체 주기 63거래일 · 다음 날 종가에 산다고 보고 · 비용 전).")
                sm = uv.picks_summary(picks)
                if sm["채점 수"].sum() == 0:
                    st.caption(f"기록 {len(picks)}번 (첫 기록 {picks['date'].iloc[0]}). 아직 채점된 기록이 없습니다 — "
                               "첫 채점은 첫 기록에서 21거래일 뒤입니다.")
                else:
                    st.dataframe(sm.style.format({"채점 수": "{:.0f}", "평균 초과수익": "{:+.2%}", "전체 평균보다 나은 비율": "{:.0%}",
                                                  "t값 (겹침 보정)": _signed(1)}, na_rep="—"), width="stretch")
                    st.caption("초과수익 = 고른 종목(동일 비중) − 그날 종목 범위 전체 평균. 결과를 모르는 상태에서 적어 둔 기록이라 "
                               "과거 검증보다 정직한 성적표입니다. 매주 기록해 기간이 겹치므로 t값은 겹침을 보정했고, 2를 넘어야 우연이 아니라고 봅니다.")
                with st.expander(f"기록 보기 ({len(picks)}번)"):
                    rec = pd.DataFrame({"기록일": picks["date"],
                                        "순위 모델": picks["model"].map(lambda m: uv.MODEL_LABELS.get(m, "고르지 않음 → 고정 가중 순위")),
                                        "종목 수": pd.to_numeric(picks["n"], errors="coerce"),
                                        **{f"{h}거래일 초과수익": pd.to_numeric(picks[f"basket_{h}"], errors="coerce")
                                           - pd.to_numeric(picks[f"market_{h}"], errors="coerce") for h in uv.SCORE_HORIZONS},
                                        "종목": picks["names"]}).iloc[::-1]
                    st.dataframe(rec.style.format({"종목 수": "{:.0f}", **{f"{h}거래일 초과수익": "{:+.2%}" for h in uv.SCORE_HORIZONS}},
                                                  na_rep="—"), hide_index=True, width="stretch")

            st.markdown("**모델 비교** — 모두 같은 조건으로 시험했습니다. 이 표를 보고 나중에 1등을 고르면 성적이 부풀려지므로, "
                        "앱은 매 시점 그때까지의 성적만 보고 고른 **자동 선택**을 씁니다.")
            st.dataframe(U["table"].style.format({
                "평균 IC": "{:+.3f}", "IC t값": "{:+.1f}", "IC>0 비율": "{:.0%}",
                "상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": "{:+.1f}",
                "상위−하위 스프레드(연)": "{:+.1%}", "최대 낙폭(비용후)": "{:.0%}",
                "평균 회전율": "{:.0%}", "리밸런싱 수": "{:.0f}", "체결 막힌 주문": "{:.0f}"}, na_rep="—"),
                hide_index=True, width="stretch")
            share = U["chosen"].value_counts(normalize=True)
            st.caption("자동 선택 비율: " + ", ".join(
                f"{uv.MODEL_LABELS.get(k, '고르지 않음')} {v:.0%}" for k, v in share.items()) +
                ". IC(순위 정확도)는 점수 순위와 실제 수익률 순위가 얼마나 일치하는지입니다. "
                "0이면 무관하고 0.05 정도면 실무에서 의미 있는 수준입니다. t값이 2를 넘으면 우연일 가능성이 낮다고 봅니다 "
                "(리밸런싱끼리의 자기상관을 반영한 Newey-West t값). 체결 막힌 주문 = 체결일에 하한가·거래정지라 못 판 종목(계속 보유)과 "
                "상한가·거래정지라 못 산 종목의 수. '(기준)' 모델은 같은 조건으로 비교만 하고 자동 선택 후보에는 넣지 않습니다."
                + (f" 자동 선택이 고른 모델이 바뀐 횟수: 리밸런싱 {len(U['chosen'])}번 중 {U['switches']}번 "
                   "(자주 바뀌면 모델 고르기가 운에 좌우된다는 뜻)." if U.get("switches") is not None else ""))
            sel = U.get("selection")
            if sel is not None:
                st.caption(f"이 표의 1등({uv.MODEL_LABELS[sel['best']]})을 나중에 고르면 생기는 착시 보정: 효과가 없어도 {sel['n_trials']}개 중 "
                           f"1등의 초과수익 샤프는 {sel['sr0_ann']:.2f}(연) 정도가 나올 수 있고, 1등 샤프 {sel['sr_ann']:.2f}가 진짜일 확률"
                           f"(Deflated Sharpe)은 {sel['dsr']:.0%}입니다.")

            fx = U.get("factors")
            if fx is not None:
                st.markdown("**알려진 요인으로 설명되는가?** — 자동 선택 상위 20%의 초과수익(비용 후 − 전체 평균)을 시장·요인 수익률에 회귀")
                ui.verdict(f"리밸런싱 {fx['n']}번 · R² {fx['r2']:.2f}", f"알파 {fx['alpha']:+.1%}", "요인을 빼고 남는 초과수익 (연)",
                           [("요인으로 설명되지 않는 신호 있음 (t > 2)", "approve") if fx["alpha_t"] > 2 else
                            ("알려진 요인과 구분되지 않음", "elective")], f"알파 t = {fx['alpha_t']:.1f}")
                st.dataframe(fx["table"].style.format({"노출(계수)": "{:+.2f}", "t값": _signed(1), "요인 자체 성과(연)": "{:+.1%}"},
                                                      na_rep="—"), width="stretch")
                st.caption("요인은 같은 종목 범위에서 그날 순위로 상위 20% − 하위 20%를 동일가중으로 직접 만들었고, 규모는 시가총액 대신 "
                           "거래대금으로 근사했습니다. 노출이 크고 t가 2를 넘는 요인이 있으면 초과수익의 일부가 그 요인 덕분입니다. "
                           "알파가 0 근처면 '새로운 신호'가 아니라 이미 알려진 효과에 기울어진 결과입니다. 표본이 리밸런싱 횟수라 작으니 t값을 꼭 함께 보세요.")

            capt = U.get("capacity")
            if capt is not None:
                st.markdown("**운용 규모를 키우면?** — 주문이 그 종목 하루 거래대금에서 차지하는 비율만큼 불리하게 체결된다고 보고 (시장 충격)")
                st.dataframe(capt.style.format({"주문 / 하루 거래대금 (중앙값)": "{:.3%}", "시장 충격 비용(연)": "{:.2%}",
                                                "초과수익(연, 모든 비용 후)": "{:+.1%}", "t값": _signed(1)}, na_rep="—"), width="stretch")
                st.caption("한 번 사거나 팔 때 추가 비용 ≈ 하루 변동성 × √(주문 금액 ÷ 하루 거래대금)으로 계산했습니다 (시장 충격의 '제곱근 법칙', "
                           "계수 1로 보수적으로). 종목당 주문 금액 = 자본 ÷ 보유 종목 수, 거래대금은 그날 기준 63일 평균입니다. "
                           "초과수익이 0 아래로 떨어지기 시작하는 규모가 이 전략을 실제로 굴릴 수 있는 대략적인 한계입니다.")
            cs = U.get("cost_sens")
            if cs is not None:
                cs_tab, breakeven = cs
                st.markdown("**수수료·슬리피지가 더 크면?** — 매수·매도 각각의 비용(bp)을 바꿔 자동 선택 상위 20%를 다시 계산")
                st.dataframe(cs_tab.style.format({"초과수익(연, 비용후)": "{:+.1%}", "t값": _signed(1), "최대 낙폭(비용후)": "{:.0%}"},
                                                 na_rep="—"), width="stretch")
                st.caption((f"손익분기 비용: 매수·매도 각각 약 {breakeven:.0f}bp를 넘으면 초과수익이 0 아래로 내려갑니다. "
                            if np.isfinite(breakeven) else "비용이 0이어도 초과수익이 없어 손익분기 비용을 계산하지 않습니다. ")
                           + "실제 체결가가 종가보다 몇 bp만 불리해도 무너지는 신호라면 실전에서 쓰기 어렵습니다.")
            rw = U.get("risk_weighted")
            if rw is not None:
                a0 = U["table"].loc["auto"]
                st.markdown("**위험을 비중에 반영하면?** — 같은 종목을 63일 변동성의 역수에 비례해 담음 (종목당 동일 비중의 2배 상한)")
                st.dataframe(pd.DataFrame({"동일 비중": a0, "변동성 역가중": rw}).T[
                    ["상위20% 초과수익(연, 비용후)", "초과수익 t값", "최대 낙폭(비용후)", "평균 회전율"]].style.format(
                    {"상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": _signed(1), "최대 낙폭(비용후)": "{:.0%}",
                     "평균 회전율": "{:.0%}"}, na_rep="—"), width="stretch")
                st.caption("변동성이 큰 종목을 덜 담아 포트폴리오 흔들림을 줄이는 가장 단순한 위험 예산입니다. 업종·베타 제한이나 "
                           "종목 간 상관을 고려한 최적화는 하지 않습니다.")
            mk = U.get("market")
            if mk is not None:
                st.markdown("**시장 상황별로는?** — 리밸런싱 기간을 그 기간 종목 범위 평균 수익률로 하락·횡보·상승 셋으로 나눔")
                st.dataframe(mk.style.format({"기간 수": "{:.0f}", "시장 평균(기간)": "{:+.1%}", "초과수익 평균(기간)": "{:+.2%}",
                                              "초과 > 0 비율": "{:.0%}"}, na_rep="—"), width="stretch")
                st.caption("특정 시장(예: 상승장)에서만 초과수익이 났다면 그 신호는 시장 상황에 기대는 것일 수 있습니다. 기간 수가 적어 참고용입니다.")
            rk = U.get("risk")
            if rk is not None:
                st.markdown(f"**지금 고른 {rk['n']}종목을 같은 비중으로 들고 있으면?** — 지난 {rk['days']}거래일 동안 그대로 들고 있었다고 보고 계산")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("하루 VaR 95%", f"−{rk['var95']:.1%}", f"내일 기준({rk['model']}) −{rk['garch_var95']:.1%}", delta_color="off")
                r2.metric("하루 ES 95%", f"−{rk['es95']:.1%}", "최악 5% 날의 평균 손실", delta_color="off")
                r3.metric("최악의 하루", f"{rk['worst_day'][1]:+.1%}", f"{rk['worst_day'][0].date()}", delta_color="off")
                r4.metric("최악의 한 달 (21거래일)", f"{rk['worst_month'][1]:+.1%}", f"{rk['worst_month'][0].date()}까지", delta_color="off")
                st.markdown(f"시장이 가장 크게 빠졌던 날 {len(rk['stress'])}일 — 시장 = 전체 종목 평균, 이 종목들의 시장 베타 {rk['beta']:.2f}")
                signed_table(rk["stress"] * 100, 1)
                st.caption("단위 %. 과거 실제 수익률로 계산해서, 그 기간에 없던 종류의 충격은 반영하지 못합니다. "
                           "베타가 1보다 크면 시장보다 더 크게 출렁인다는 뜻입니다.")

            left, right = st.columns(2)
            fmt = {c: "{:.0%}" for c in U["top"].columns if c not in ("순위", "종목명", "점수")}
            fmt["점수"] = "{:+.3f}"
            with left:
                st.markdown(f"**상위 20종목** (전체 {U['n_univ']}개 중)")
                st.dataframe(U["top"].style.format(fmt), width="stretch")
            with right:
                st.markdown("**하위 20종목**")
                st.dataframe(U["bottom"].style.format(fmt), width="stretch")
            st.caption("신호 열은 전체 종목 중 상대 위치입니다 (100% = 가장 큼). 예: 외국인 순매수 강도 90%면 상위 10%.")

            ic = U["ic"].dropna()
            fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in ic.values]))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="교체 시점별 순위 정확도(IC)")
            ui.chart(fig)

            kf = get_kr_factors(uv.kr_factors_stamp())
            if kf is not None:
                with st.expander("한국판 요인의 성과 (방향 예측·시장 국면의 수익 분해에 쓰는 일간 요인)"):
                    cols = [c for c in uv.KR_FACTORS if c in kf and kf[c].notna().any()]
                    cum = np.exp(np.log1p(kf[cols].fillna(0.0)).cumsum()).rename(columns=uv.KR_FACTORS)
                    fig = go.Figure([go.Scatter(x=cum.index, y=cum[c], name=c) for c in cum.columns])
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="누적 (시작=1)",
                                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    ui.chart(fig)
                    yr = pd.DataFrame({"연 평균": kf[cols].mean() * 252, "연 변동성": kf[cols].std() * np.sqrt(252),
                                       "t값 (자기상관 보정)": kf[cols].apply(lambda c: hac_tstat(c.values))},
                                      ).rename(index=uv.KR_FACTORS)
                    st.dataframe(yr.style.format({"연 평균": "{:+.1%}", "연 변동성": "{:.1%}", "t값 (자기상관 보정)": _signed(1)}, na_rep="—"),
                                 width="stretch")
                    st.caption("같은 종목 범위(그때의 거래대금 상위 200)에서 21거래일마다 줄 세워 상위 30% − 하위 30%(동일 비중)를 들고 간 수익률입니다. "
                               "시장은 시가총액 가중 − CD 91일 금리, 소형주는 시가총액 하위 − 상위, 가치는 장부가/시총, 수익성은 ROE(DART 재무제표, "
                               "공시 다음 거래일부터), 모멘텀은 12-1개월 수익률 기준입니다. 파마-프렌치의 투자 요인은 자산 데이터가 없어 뺐습니다.")
            with st.expander("데이터·한계"):
                cov = pd.Series({uv.FEATURES[c][0]: U["coverage"].get(c, 0) for c in uv.FEATURES})
                st.markdown("**신호별 데이터 보유율** (30% 미만이면 모델에서 제외)")
                st.dataframe(cov.to_frame("보유율").style.format("{:.0%}"), width="stretch")
                if U.get("krx"):
                    scope = (f"""- **종목 범위:** KRX 일별매매정보의 **그날 상장돼 있던** 코스피 보통주 전체(그 사이 상장폐지된 {U.get('n_delisted', 0)}종목 포함) 중,
  각 과거 시점의 거래대금 상위 200개 (공식 코스피200 구성과는 다름). 폐지 종목은 정리매매 마지막 가격까지의 손익을 반영하고 그 뒤는 현금으로 봅니다.
- **가치 지표:** DART 재무제표(분기·사업보고서)의 자본총계·순이익을 **공시 다음 거래일부터**, 시가총액은 **그날의 실제 값**(KRX)으로 씁니다.
  폐지 종목은 수급·실적·재무 데이터가 없어 그 신호는 중간값으로 둡니다.""")
                else:
                    scope = """- **종목 범위:** 현재 상장된 코스피 보통주 전체 중, 각 과거 시점의 거래대금 상위 200개 (공식 코스피200 구성과는 다름).
  '지금 큰 종목'으로 과거를 고르는 편향은 없앴지만, 그 사이 **상장폐지된 종목은 주가가 없어 빠져 있어**
  과거 성과가 약간 좋게 보일 수 있습니다 (KRX 일별매매정보를 받으면 포함).
- **가치 지표:** DART 재무제표(분기·사업보고서)의 자본총계·순이익을 **공시 다음 거래일부터** 씁니다.
  시가총액은 '그날 주가 × 현재 상장주식수'로 근사해서, 과거 증자·소각으로 주식수가 바뀐 종목은 오차가 있습니다."""
                st.markdown(scope + """
- **비용:** 리밸런싱마다 바뀐 종목 비율 × (살 때·팔 때 수수료·체결 비용)을 빼고, 증권거래세를 켜면 판 비중 × 그날의 코스피 세율을
  더 뺍니다. 호가 차이·시장 충격이 설정보다 크면 결과가 나빠집니다.
- 투자 조언이 아닙니다.
""")

# ---------------- 시장 심리 스냅샷 ----------------
with tab_snap:
    ui.section("Flows · Disclosures · News", "실적 전망·공시·수급·뉴스")
    st.info("외국인·기관 수급, 실적 서프라이즈, DART 공시는 과거 기록이 있어 방향 예측 모델에 들어갑니다. "
            "실적 전망치와 뉴스 감성은 무료로는 지금 값만 구할 수 있어 참고용으로 보여주고, 기록 종목은 매일 저장해 "
            "'예측 정확도 검증' 탭의 실전 기록에서 쓸모가 있는지 채점합니다.")
    with st.spinner("전망·뉴스 불러오는 중…"):
        snap = get_snapshot(ticker, spot, r, q)

    st.subheader("실적과 전망치")
    rv = snap["revisions"]
    code = ad.krx_code(ticker)
    if code:
        st.caption("한국 종목은 네이버 증권의 실적과 전망치를 보여줍니다. (E)가 붙은 열은 애널리스트들의 평균 전망치입니다. "
                   "금액 단위: 억원, 비율: %")
        for label, period in (("분기", "quarter"), ("연간", "annual")):
            try:
                fin = ad.naver_financials(code, period)
            except Exception:
                fin = None
            if fin is not None and not fin.empty:
                st.markdown(f"**{label}**")
                st.dataframe(fin.style.format("{:,.2f}", na_rep="—"), width="stretch")
            else:
                st.caption(f"{label} 재무 데이터를 불러오지 못했습니다.")
    elif rv is None or rv.empty:
        st.caption("추정치 데이터가 없습니다.")
    else:
        fmt = {c: "{:+.2%}" for c in rv.columns if "대비" in c}
        fmt["현재 추정 EPS"] = "{:,.2f}"
        st.dataframe(rv.style.format(fmt, na_rep="—"), width="stretch")
        st.caption("전망치가 올라가는 중(+)이면 애널리스트들이 실적 기대를 높이고 있다는 뜻입니다. "
                   "과거 전망 변경 이력은 무료로 구할 수 없어, 과거 검증은 등급·목표가 변경 데이터로 대신합니다.")

    if code and kr_data.get_key("DART_API_KEY"):
        st.subheader("최근 공시 (DART)")
        try:
            dis = kr_data.dart_disclosures(code, close.index[0])
            recent = dis.sort_values("date", ascending=False).head(30).copy()
            recent["분류"] = recent["report"].map(kr_data.classify).map(
                lambda k: kr_data.DART_FEATURES.get(k, "기타" if k else "정정").split("(")[0])
            recent["링크"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + recent["rcept_no"]
            st.dataframe(recent[["date", "분류", "report", "링크"]], hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn("접수일"), "report": "보고서명",
                                        "링크": st.column_config.LinkColumn("원문", display_text="열기")})
        except Exception as e:
            st.caption(f"공시를 불러오지 못했습니다: {e}")

    if code:
        st.subheader("외국인·기관 수급 (최근 60거래일)")
        try:
            flw = ad.load_investor_flows(code, close.index[-60]).iloc[-60:]      # fl(실전 기록 모듈)과 겹치지 않게
            fig = go.Figure()
            fig.add_trace(go.Bar(x=flw.index, y=flw["frgn"], name="외국인 순매수(주)", marker_color=ui.BLUE))
            fig.add_trace(go.Bar(x=flw.index, y=flw["inst"], name="기관 순매수(주)", marker_color=ui.AMBER))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
        except Exception as e:
            st.caption(f"수급 데이터를 불러오지 못했습니다: {e}")

    st.subheader("뉴스 감성")
    nw = snap["news"]
    if nw is None or nw.empty:
        st.caption("뉴스를 불러오지 못했습니다.")
    else:
        recent = nw[nw["time"] >= nw["time"].max() - pd.Timedelta(days=7)]
        a, b, c = st.columns(3)
        a.metric("최근 7일 평균 감성", f"{recent['score'].mean():+.2f}", "−1 매우 부정 ~ +1 매우 긍정", delta_color="off")
        b.metric("긍정 / 부정 헤드라인", f"{(nw['score'] > 0).sum()} / {(nw['score'] < 0).sum()}",
                 f"중립 {(nw['score'] == 0).sum()}", delta_color="off")
        c.metric("수집 기간", f"{(nw['time'].max() - nw['time'].min()).days + 1}일", f"헤드라인 {len(nw)}개",
                 delta_color="off")
        daily = nw.groupby(nw["time"].dt.date)["score"].mean()
        fig = go.Figure(go.Bar(x=[d.strftime("%m-%d") for d in daily.index], y=daily.values,
                               marker_color=[ui.POS if v > 0 else ui.NEG if v < 0 else ui.NEUTRAL for v in daily.values]))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="일평균 감성",
                          xaxis_type="category", yaxis_range=[-1, 1])
        ui.chart(fig)
        st.dataframe(nw[["time", "score", "title", "source", "link"]], hide_index=True, width="stretch",
                     column_config={"time": st.column_config.DatetimeColumn("시각", format="MM-DD HH:mm"),
                                    "score": st.column_config.NumberColumn("감성", format="%+.2f"),
                                    "title": "헤드라인", "source": "언론사",
                                    "link": st.column_config.LinkColumn("링크", display_text="열기")})
        st.caption("감성 점수는 제목에 나온 긍정·부정 단어를 세는 단순한 방식이라, 반어나 맥락은 이해하지 못합니다. "
                   "과거 뉴스 기록을 무료로 구할 수 없어 예측에 도움이 되는지는 아직 확인하지 못했습니다. 그래서 기록 대상 종목은 "
                   "이 탭의 옵션·전망·뉴스 지표를 매일 저장해 두고, '예측 정확도 검증' 탭의 실전 기록에서 채점합니다.")

# ---------------- 2. 변동성 ----------------
with tab2:
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

# ---------------- 시장 국면 ----------------
with tab_regime:
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

# ---------------- 4. 백테스트 ----------------
with tab4:
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
        st.caption(f"기록 {fs['기록 수']}일 ({fs['첫 기록일'].date()} ~ {fs['마지막 기록일'].date()}), 21거래일 결과가 나온 기록 {n_done}개. "
                   "예측은 앱 기본 설정(방향 21거래일, GARCH 학습 3년)으로 계산했습니다.")
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
                       "첫 3년 뒤 전체 기간으로 검정한 결과는 실제 데이터 검증 보고서(validation/README.md)에 있습니다.")

st.caption(f"재현 정보 — 코드 {repro.code_version()} · 주가 데이터 {repro.fingerprint(close)}"
           + (f" · 방향 예측 데이터 {repro.fingerprint(close_long)}" if p_up is not None else "")
           + ". 같은 지문이면 같은 데이터로 계산한 결과입니다 (무료 데이터는 과거 값이 나중에 고쳐지기도 합니다).")
ui.footer("교육·연구용 고지:", "이 프로그램의 예측과 신호는 과거 데이터로 계산한 참고 자료이며 투자 조언이 아닙니다. "
          "과거 검증 결과가 미래 성과를 보장하지 않습니다.",
          "주가는 Yahoo Finance, 종목 목록·수급·뉴스는 네이버 증권(비공식), 공시·재무는 DART, 금리는 한국은행 ECOS에서 받습니다. "
          "무료 데이터는 늦거나 바뀔 수 있습니다.")
