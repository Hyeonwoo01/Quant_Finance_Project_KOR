"""실전 예측 기록: 기록·재현, 늦은 기록 차단, 시세 보충, 채점."""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import alt_data as ad
import forecast_log as fl
import kr_data
import market_data as md
import repro
from quant_engine import range_level_now


def test_repro_fingerprint():
    _s1 = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert repro.fingerprint(_s1) == repro.fingerprint(_s1.copy()) and "3일" in repro.fingerprint(_s1)
    assert repro.fingerprint(_s1) != repro.fingerprint(_s1 * 1.001)            # 과거 값이 고쳐지면 지문이 바뀜
    assert isinstance(repro.code_version(), str) and repro.fingerprint(_s1.iloc[:0]) == "데이터 없음"


def test_record_and_predict():
    _full = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.011 * np.random.default_rng(1).standard_normal(1150))),
                      index=pd.bdate_range("2022-01-03", periods=1150))
    _cut = {"n": 1100}
    _orig = {k: getattr(md, k) for k in ("load_history", "load_macro", "risk_free_rate", "dividend_yield", "atm_implied_vol",
                                         "load_vol_index", "load_ohlc")}
    _orig_alt = {k: getattr(ad, k) for k in ("build_alt_features", "options_snapshot", "estimate_revisions", "news_sentiment")}
    md.load_history = lambda t, years: _full.iloc[:_cut["n"]]
    md.load_macro = lambda t, years=10: None
    md.risk_free_rate = lambda t="", default=0.04: (0.04, "테스트")
    md.dividend_yield = lambda t, s: 0.0
    md.atm_implied_vol = lambda t, s, d, r, q: {"iv": 0.19, "cal_days": 30}
    md.load_vol_index = lambda t, y: None
    _frg = np.random.default_rng(3)                                       # 종가에 맞춘 일봉 (장중 범위가 넓은 편)
    _fopen = _full.shift(1).bfill() * np.exp(0.004 * _frg.standard_normal(len(_full)))
    _fohlc = pd.DataFrame({"Open": _fopen, "High": np.maximum(_fopen, _full) * (1 + 0.012 * np.abs(_frg.standard_normal(len(_full)))),
                           "Low": np.minimum(_fopen, _full) * (1 - 0.012 * np.abs(_frg.standard_normal(len(_full)))), "Close": _full})
    md.load_ohlc = lambda t, years: _fohlc                                # 미래 날짜 일봉이 있어도 쓰는 종가의 날짜까지만 씀
    ad.build_alt_features = lambda t, idx: (None, {})
    ad.options_snapshot = lambda t, s, r=0.04, q=0.0: {"pc_volume": 1.2, "pc_oi": 1.5, "skew": 0.05}
    ad.estimate_revisions = lambda t: pd.DataFrame({"30일 전 대비": [0.02, 0.01]}, index=["이번 분기", "올해"])
    ad.news_sentiment = lambda t, max_items=60: pd.DataFrame({"time": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-20"]),
                                                              "score": [1.0, -0.5, 0.25]})
    try:
        _path = Path(tempfile.mkdtemp()) / "log.csv"
        _at = lambda i: _full.index[i] + pd.Timedelta(hours=21)          # 그날 미국 장 마감 뒤 (UTC 21시 = 뉴욕 오후 4~5시)
        _new, _fail = fl.record(_path, ["SPY"], now=_at(1099))
        assert len(_new) == 1 and not _fail and _new[0]["asof"] == _full.index[1099].date().isoformat()
        assert fl.record(_path, ["SPY"], now=_at(1099))[0] == []        # 같은 날 다시 돌려도 기록이 늘지 않음 (덮어쓰지 않음)
        _cut["n"] = 1101
        assert fl.record(_path, ["SPY"], now=_at(1101))[0] == []        # 시세가 하루 늦어 마지막 날 다음 장이 이미 열렸으면 적지 않음
        assert len(fl.record(_path, ["SPY"], now=_at(1100))[0]) == 1 and len(fl.read_log(_path)) == 2
        _row = fl.read_log(_path).iloc[0]
        _p = fl.predict("SPY")                                           # 기록 = 같은 데이터로 다시 계산한 예측 (재현 가능)
        _cut["n"] = 1100
        assert np.isclose(fl.predict("SPY")["p_up"], _row["p_up"]) and np.isclose(fl.predict("SPY")["lo68"], _row["lo68"])
        assert _row["lo95"] < _row["lo68"] < _row["close"] < _row["hi68"] < _row["hi95"] and 0 < _row["var95_1d"] < _row["var99_1d"]
        # 코스피 200 종목의 하루 VaR: GARCH와 전전 거래일 VKOSPI를(KRX가 하루 늦게 줌) 반반 섞은 σ → 하루 로그수익률 분위수가 같은 배수로 커짐 (21일 범위는 그대로)
        _base = fl.predict("SPY")
        assert np.isnan(_base["var_vkospi"])                              # 대상 종목이 아니면 VKOSPI는 섞지 않음
        # 그 밖의 종목: 장중 고가·저가 실현 분산을 반반 섞음 (일봉을 못 받으면 GARCH 그대로) → 하루 분위수만 같은 배수로 바뀜
        _rl = range_level_now(_fohlc, _full.index[:1100])
        assert np.isclose(_base["var_range_vol"], np.sqrt(_rl * 252))
        md.load_ohlc = lambda t, years: 1 / 0
        try:
            _g0 = fl.predict("SPY")
        finally:
            md.load_ohlc = lambda t, years: _fohlc
        _f95, _f99 = (np.log1p(-_base[k]) / np.log1p(-_g0[k]) for k in ("var95_1d", "var99_1d"))
        assert np.isnan(_g0["var_range_vol"]) and _f95 > 1.05 and np.isclose(_f95, _f99, rtol=1e-6) and np.isclose(_base["lo68"], _g0["lo68"])
        _ivt0, _vko0 = fl.VAR_IV_TICKERS, kr_data.vkospi_on
        fl.VAR_IV_TICKERS, kr_data.vkospi_on = ("SPY",), lambda d: 40.0 if d == _full.index[1098] else np.nan   # 마지막 종가일의 전 거래일 값만
        try:
            _bl = fl.predict("SPY")
        finally:
            fl.VAR_IV_TICKERS, kr_data.vkospi_on = _ivt0, _vko0
        # 로그수익률 분위수가 같은 배수 (VaR는 단순 수익률이라 분위수 보간 때문에 1e-8 정도 어긋남)
        _c95, _c99 = (np.log1p(-_bl[k]) / np.log1p(-_base[k]) for k in ("var95_1d", "var99_1d"))
        assert _bl["var_vkospi"] == 40.0 and np.isnan(_bl["var_range_vol"]) and _c95 > 1 and np.isclose(_c95, _c99, rtol=1e-6) and np.isclose(_bl["lo68"], _base["lo68"])
        # 시장 심리 지표도 앱과 같은 값으로 저장 (뉴스는 가장 최근 헤드라인 기준 7일: 1/20 하나만)
        assert _row["pc_oi"] == 1.5 and _row["eps_rev_30d"] == 0.01 and _row["news_7d"] == 0.25 and _row["news_n"] == 1
    finally:
        for _k, _v in _orig.items():
            setattr(md, _k, _v)
        for _k, _v in _orig_alt.items():
            setattr(ad, _k, _v)


def test_late_record_cutoffs():
    """마지막 종가일 다음 장이 이미 열린 뒤에 적는 기록은 소급 기록 (한국 9:00, 미국 9:30, 공휴일은 모르므로 다음 평일)"""
    assert fl.is_late("2026-10-28", "005930.KS", "2026-10-29T15:10Z") and not fl.is_late("2026-10-29", "005930.KS", "2026-10-29T15:10Z")
    assert not fl.is_late("2026-10-29", "^KS11", "2026-10-29T23:59Z") and fl.is_late("2026-10-29", "^KS11", "2026-10-30T00:00Z")   # 한국 9:00
    assert not fl.is_late("2026-10-23", "005930.KS", "2026-10-25T12:00Z") and fl.is_late("2026-10-23", "005930.KS", "2026-10-26T00:00Z")  # 금 → 월
    assert not fl.is_late("2026-10-22", "SPY", "2026-10-23T13:29Z") and fl.is_late("2026-10-22", "SPY", "2026-10-23T13:30Z")       # 뉴욕 9:30
    # 다음 거래일 장 마감(한국 15:30) 기준: 10-22 기록을 10-23 오전에 적었으면 마감 전, 10-28 기록을 10-30 자정 넘어 적었으면 마감 뒤
    assert not fl.is_late("2026-10-22", "005930.KS", "2026-10-23T02:33Z", "close") and fl.is_late("2026-10-22", "005930.KS", "2026-10-23T02:33Z")
    assert fl.is_late("2026-10-28", "005930.KS", "2026-10-29T15:10Z", "close") and not fl.is_late("2026-10-28", "005930.KS", "2026-10-29T06:29Z", "close")


def test_drop_unfinished_bar():
    """끝나지 않은 오늘 봉은 기록에 쓰지 않는다 (한국 15:30·미국 16:00 마감 + 여유)"""
    _kd = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-22", "2026-10-23"]))
    _utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    assert len(fl.drop_unfinished(_kd, "005930.KS", _utc("2026-10-23T01:00"))) == 1      # 한국 오전 10시 = 장중 → 뺌
    assert len(fl.drop_unfinished(_kd, "005930.KS", _utc("2026-10-23T08:37"))) == 2      # 한국 오후 5:37 → 그대로
    assert len(fl.drop_unfinished(_kd, "SPY", _utc("2026-10-23T16:00"))) == 1            # 뉴욕 정오 = 장중 → 뺌
    assert len(fl.drop_unfinished(_kd, "SPY", _utc("2026-10-24T08:37"))) == 2            # 다음 날 새벽 → 그대로
    assert len(fl.drop_unfinished(_kd, "^KS11", _utc("2026-10-23T01:00"))) == 1          # 코스피 지수도 한국 장 기준


def test_fill_recent_from_naver():
    """Yahoo가 마지막 거래일 일봉을 빼고 줄 때(늦어진 예약 실행이 한국 자정을 넘김): 네이버 종가로 그 뒤 날짜만 덧붙임"""
    assert fl.naver_symbol("005930.KS") == "005930" and fl.naver_symbol("247540.KQ") == "247540" and fl.naver_symbol("^KS11") == "KOSPI"
    assert fl.naver_symbol("SPY") is None
    _yh = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-10-26", "2026-10-27"]))
    _nv = pd.Series([100.0, 101.0, 103.0], index=pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"]))
    _fr = fl.fill_recent(_yh, "005930.KS", fetch=lambda s: _nv)
    assert list(_fr.index.strftime("%m-%d")) == ["10-26", "10-27", "10-28"] and _fr.iloc[-1] == 103.0
    assert fl.fill_recent(_yh, "005930.KS", fetch=lambda s: _nv * 1.02).equals(_yh)          # 겹치는 날 종가가 다르면(배당 수정 등) 그대로
    assert fl.fill_recent(_yh, "SPY", fetch=lambda s: 1 / 0).equals(_yh)                     # 한국 종목·지수가 아니면 그대로
    assert fl.fill_recent(_yh, "005930.KS", fetch=lambda s: 1 / 0).equals(_yh)               # 받지 못하면 그대로
    assert fl.fill_recent(_nv, "005930.KS", fetch=lambda s: _nv).equals(_nv)                 # 이미 최신이면 그대로


def test_parse_cboe_csv():
    """Cboe 변동성 지수 CSV (Yahoo에 없는 RVX 등)"""
    _cb = md.parse_cboe_csv("DATE,OPEN,HIGH,LOW,CLOSE\n10/21/2026,20.1,21.0,19.5,20.5\n10/22/2026,20.5,22.0,20.0,21.25\n")
    assert list(_cb.index.strftime("%Y-%m-%d")) == ["2026-10-21", "2026-10-22"] and _cb.iloc[-1] == 21.25


def test_score_forecasts():
    """채점: 손으로 만든 기록과 가격으로 계산이 맞는지"""
    _ci = pd.bdate_range("2025-01-01", periods=80)
    _cl = pd.Series(100 * np.exp(np.cumsum(np.r_[0, 0.01 * np.random.default_rng(2).standard_normal(79)])), index=_ci)
    _lg = pd.DataFrame({"asof": _ci[[0, 1, 2, 70]], "ticker": "X", "close": _cl.values[[0, 1, 2, 70]] * 1.1, "h": 21,
                        "p_up": [0.9, 0.1, 0.6, 0.5], "p_base": 0.55, "model": "m", "vol_garch": 0.15,
                        "lo95": 0.0, "lo68": 0.0, "hi68": np.inf, "hi95": np.inf, "var95_1d": 0.0, "var99_1d": 1.0,
                        "iv_atm": np.nan, "vol_index": 0.2, "p_high": [0.9, 0.1, 0.1, 0.1]})
    _sc = fl.score(_lg, _cl)
    _up = (_cl.values[[21, 22, 23]] > _cl.values[[0, 1, 2]]).astype(float)
    assert _sc["summary"]["채점 수"] == 3 and np.isclose(_sc["summary"]["적중률"], np.mean((np.array([0.9, 0.1, 0.6]) > 0.5) == _up))
    # 늦게 적힌 기록도 방향·범위·VaR 채점에는 쓰고(예측은 기록 날짜까지의 데이터), 다음 거래일 장 마감 뒤에 적힌 것만 심리 지표에서 뺀다
    # 두 번째: 그다음 날 저녁 6시에 적음(마감 뒤), 세 번째: 그다음 날 오전 10시에 적음(장 시작 뒤·마감 전)
    _lgl = _lg.assign(ticker="005930.KS", news_7d=[0.1, 0.2, 0.3, 0.4],
                      logged_at=[(_ci[i] + pd.Timedelta(hours=hr)).strftime("%Y-%m-%dT%H:%MZ") for i, hr in ((0, 9), (2, 9), (3, 1), (70, 9))])
    _scl = fl.score(_lgl, _cl)
    assert _scl["summary"]["늦게 적힌 기록"] == 2 and _scl["summary"]["심리 지표에서 뺀 기록"] == 1 and _scl["summary"]["채점 수"] == 3
    assert _scl["rows"]["news_7d"].isna().tolist() == [False, True, False, False] and _sc["summary"]["늦게 적힌 기록"] == 0
    assert np.isclose(_scl["summary"]["적중률"], _sc["summary"]["적중률"])
    _bs, _bb = (np.array([0.9, 0.1, 0.6]) - _up) ** 2, (0.55 - _up) ** 2
    assert np.isclose(_sc["summary"]["확률 정확도 개선"], 1 - _bs.mean() / _bb.mean())
    assert np.isclose(_sc["rows"]["ret_h"].iloc[0], _cl.iloc[21] / _cl.iloc[0] - 1)            # 기록한 종가가 아니라 비율로 채점
    assert _sc["summary"]["68% 범위 적중"] == 1.0 and np.isnan(_sc["summary"]["옵션 IV 오차"])
    _r1 = _cl.values[[1, 2, 3, 71]] / _cl.values[[0, 1, 2, 70]] - 1
    assert _sc["summary"]["하루 VaR 95%"]["예외 횟수"] == int((_r1 < 0).sum()) and _sc["summary"]["하루 VaR 99%"]["예외 횟수"] == 0
    assert np.isclose(_sc["summary"]["불안 국면일 때 실제 변동성"], _sc["rows"]["rv"].iloc[0])


def test_sentiment_scoring():
    """시장 심리 지표 채점: 다음 거래일 종가부터의 수익률과 비교. 그 수익률을 그대로 아는 지표면 상관 1"""
    _ci2 = pd.bdate_range("2025-01-01", periods=120)
    _cl2 = pd.Series(100 * np.exp(np.cumsum(0.01 * np.random.default_rng(3).standard_normal(120))), index=_ci2)
    _fut = _cl2.shift(-22) / _cl2.shift(-1) - 1
    _lg2 = pd.DataFrame({"asof": _ci2[:90], "ticker": "X", "close": _cl2.values[:90], "h": 21, "p_up": 0.5, "p_base": 0.5, "model": "m",
                         "vol_garch": 0.15, "lo95": 0.0, "lo68": 0.0, "hi68": np.inf, "hi95": np.inf, "var95_1d": 0.02, "var99_1d": 0.03,
                         "iv_atm": np.nan, "vol_index": np.nan, "p_high": 0.1, "news_7d": _fut.values[:90],
                         "pc_oi": np.random.default_rng(4).standard_normal(90)})
    _st = fl.score(_lg2, _cl2)["summary"]["시장 심리 지표"]
    assert np.isclose(_st.loc["뉴스 감성 (최근 7일 평균)", "수익률과 상관"], 1.0) and _st.loc["뉴스 감성 (최근 7일 평균)", "표본"] == 90
    assert abs(_st.loc["풋/콜 미결제약정 비율", "수익률 t값"]) < 3 and np.isnan(_st.loc["옵션 스큐 (풋 90% − 콜 110% IV)", "수익률과 상관"])
