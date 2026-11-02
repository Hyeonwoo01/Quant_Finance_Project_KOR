"""앱이 쓰는 데이터 불러오기·계산 (Streamlit 캐시). 화면 코드는 views/에 있다."""
import numpy as np
import pandas as pd
import streamlit as st

import alt_data as ad
import direction as dr
import forecast_log as fl
import kr_costs as kc
import kr_data
import market_data as md
import regime as rg
import universe as uv
from quant_engine import GarchVolForecaster, VAR_INNOVS, range_level_now, var_backtest

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


@st.cache_data(show_spinner=False)
def get_kr_factors(stamp):
    """매주 계산해 둔 한국판 일간 요인. stamp: 파일 시각 — 새로 커밋되면 다시 읽는다."""
    return uv.load_kr_factors()


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


@st.cache_data(ttl=3600, show_spinner=False)
def get_range_level(ticker, years, last_day):
    """마지막 종가일까지 장중 고가·저가로 잰 실현 분산의 지수가중 평균 (하루 VaR σ 보정, 일봉을 못 받으면 NaN)."""
    try:
        ohlc = md.load_ohlc(ticker, years)
        return range_level_now(ohlc, ohlc.index[ohlc.index <= pd.Timestamp(last_day)])
    except Exception:
        return np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def get_vkospi(day):
    """그날 종가의 VKOSPI (KRX, 키가 없거나 받지 못하면 NaN) — 코스피 200 종목의 하루 VaR σ 보정에 쓴다."""
    return kr_data.vkospi_on(day)


@st.cache_data(show_spinner=False)
def run_var_backtest(returns, model):
    """하루 VaR 95%·99%의 워크포워드 예외 검정 (Kupiec·Christoffersen). model: 사이드바 변동성 모델.
    꼬리 분포 두 가지(t분포, 실제 잔차 FHS)를 같은 GARCH 적합으로 함께 계산해 한 표로."""
    both = var_backtest(np.asarray(returns), model=model, innov=tuple(VAR_INNOVS))
    return pd.concat({VAR_INNOVS[m]: t for m, t in both.items()}, names=["꼬리 분포", "수준"])


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
