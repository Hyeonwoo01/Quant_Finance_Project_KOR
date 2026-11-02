"""한국 데이터: 거시 지표 시차, 종목 검색, 가치 지표, KRX·VKOSPI 해석, 한국판 요인, 상장폐지 포함 종목 범위."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import direction as dr
import kr_data
import market_data as md
import quant_engine as qe
import universe as uv


def test_macro_lag():
    """거시 지표 시차: 명시 > 티커로 자동 판단 > 모르면 안전하게 하루 늦춤"""
    assert dr.resolve_lag(None, "005930.KS") and dr.resolve_lag(None, "247540.KQ")
    assert not dr.resolve_lag(None, "AAPL") and not dr.resolve_lag(None, "SPY")
    assert dr.resolve_lag(None, None) and not dr.resolve_lag(False, "005930.KS")
    assert dr.resolve_lag(None, "^KS11") and dr.resolve_lag(None, "^KQ11") and not dr.resolve_lag(None, "^GSPC")   # 한국 지수도 하루 늦춤
    mi = pd.bdate_range("2019-01-01", periods=700)
    g2 = np.random.default_rng(11)
    macro_t = pd.DataFrame({k: 50 + np.cumsum(g2.normal(0, 1, len(mi))) for k in
                            ("vix", "vix3m", "tnx", "irx", "hyg", "ief", "market")}, index=mi).abs() + 10
    px_t = pd.Series(100 * np.exp(np.cumsum(g2.normal(0, 0.01, len(mi)))), index=mi)
    f_kr = dr.build_features(px_t, macro_t, ticker="005930.KS")
    f_us = dr.build_features(px_t, macro_t, ticker="AAPL")
    d0 = f_kr.index[-1]
    assert f_kr.loc[d0, "vix"] == np.log(macro_t["vix"]).loc[:d0].iloc[-2]      # 한국: 전날 미국 값
    assert f_us.loc[d0, "vix"] == np.log(macro_t["vix"]).loc[d0]                # 미국: 같은 날 값
    assert f_kr.equals(dr.build_features(px_t, macro_t, lag_macro=True))


def test_ticker_search():
    """종목명 검색: 정확히 일치 > 앞부분 > 포함, 같은 순위면 시가총액 순 / 코드·티커 입력은 그대로"""
    _fake = pd.DataFrame({"name": ["삼성전자", "삼성전자우", "삼성바이오로직스", "에코프로비엠", "에코프로", "NAVER"],
                          "ticker": ["005930.KS", "005935.KS", "207940.KS", "247540.KQ", "086520.KQ", "035420.KS"],
                          "market": ["코스피", "코스피", "코스피", "코스닥", "코스닥", "코스피"],
                          "cap": [9e14, 5e13, 7e13, 2e13, 1e13, 3e13]})
    _orig_sym = md.kr_symbols
    md.kr_symbols = lambda *a, **k: _fake
    try:
        assert [t for t, _ in md.search_ticker("삼성전자")] == ["005930.KS", "005935.KS"]
        assert [t for t, _ in md.search_ticker("삼성")][:2] == ["005930.KS", "207940.KS"]     # 앞부분 일치, 시총 순
        assert [t for t, _ in md.search_ticker("에코프로")] == ["086520.KQ", "247540.KQ"]      # 정확히 일치가 먼저
        assert md.search_ticker("247540")[0][0] == "247540.KQ"                                # 코드만 → 시장 찾아 붙임
        assert md.search_ticker("005930.ks")[0][0] == "005930.KS"
        assert md.search_ticker("SPY") == [("SPY", "SPY")]
        assert md.search_ticker("naver")[0][0] == "035420.KS"                                 # 영문 이름 한국 종목
        assert md.search_ticker("없는회사") == [] and md.search_ticker("  ") == []
    finally:
        md.kr_symbols = _orig_sym


def test_value_metrics():
    """가치 지표: 4분기 = 연간 − (1~3분기), 최근 4분기 합(TTM), 공시 다음 거래일부터 반영"""
    fin = pd.DataFrame([
        {"code": "000010", "year": 2023, "q": 1, "rcept_dt": pd.Timestamp("2023-05-15"), "equity": 100.0, "ni": 10.0},
        {"code": "000010", "year": 2023, "q": 2, "rcept_dt": pd.Timestamp("2023-08-14"), "equity": 100.0, "ni": 10.0},
        {"code": "000010", "year": 2023, "q": 3, "rcept_dt": pd.Timestamp("2023-11-14"), "equity": 100.0, "ni": 10.0},
        {"code": "000010", "year": 2023, "q": 4, "rcept_dt": pd.Timestamp("2024-03-15"), "equity": 120.0, "ni": 50.0},
        {"code": "000010", "year": 2024, "q": 1, "rcept_dt": pd.Timestamp("2024-05-15"), "equity": 125.0, "ni": 5.0},
    ])
    vidx = pd.bdate_range("2023-01-02", "2024-06-28")
    vclose = pd.DataFrame({"000010": 10.0}, index=vidx)
    v = kr_data.value_features(fin, vidx, vclose, pd.Series({"000010": 10.0}))    # 시총 = 100
    assert np.isnan(v["ep"].loc["2024-03-15", "000010"])                  # 사업보고서 접수 당일엔 아직 4분기 미확정
    assert abs(v["ep"].loc["2024-03-18", "000010"] - 0.50) < 1e-12         # 10+10+10+(50−30) = 50
    assert abs(v["bm"].loc["2024-03-18", "000010"] - 1.20) < 1e-12
    assert abs(v["ep"].loc["2024-05-16", "000010"] - 0.45) < 1e-12         # 10+10+20+5 = 45
    assert abs(v["roe"].loc["2024-05-16", "000010"] - 45 / 125) < 1e-12
    assert v["bm"].loc["2023-05-15", "000010"] != v["bm"].loc["2023-05-15", "000010"]   # 첫 공시 전엔 NaN
    # 그날의 실제 시가총액(KRX)이 있으면 그 값으로: 2024-04부터 증자로 시총이 두 배 → 그 뒤 B/M은 절반 (현재 주식수 근사로는 못 잡음)
    _mc = pd.DataFrame({"000010": np.where(vidx >= pd.Timestamp("2024-04-01"), 200.0, 100.0)}, index=vidx)
    v2 = kr_data.value_features(fin, vidx, vclose, pd.Series(dtype=float), mcap=_mc)
    assert abs(v2["bm"].loc["2024-03-18", "000010"] - 1.20) < 1e-12 and abs(v2["bm"].loc["2024-04-02", "000010"] - 0.60) < 1e-12


def test_krx_daily_parsing():
    """KRX Open API 일별매매정보 해석: 쉼표·'-' 정리, 12자리 표준코드 → 6자리"""
    _kr = kr_data.parse_krx_daily([
        {"BAS_DD": "20261023", "ISU_CD": "005930", "ISU_NM": "삼성전자", "TDD_CLSPRC": "71,200", "FLUC_RT": "-1.25",
         "ACC_TRDVOL": "1,234", "ACC_TRDVAL": "87,860,800", "MKTCAP": "425,000,000,000,000", "LIST_SHRS": "5,969,782,550"},
        {"BAS_DD": "20261023", "ISU_CD": "KR7000660001", "ISU_NM": "SK하이닉스", "TDD_CLSPRC": "-", "FLUC_RT": "",
         "ACC_TRDVOL": "0", "ACC_TRDVAL": "0", "MKTCAP": "1", "LIST_SHRS": "2"}])
    assert _kr["code"].tolist() == ["005930", "000660"] and _kr.loc[0, "close"] == 71200 and _kr.loc[0, "chg_pct"] == -1.25
    assert _kr.loc[0, "cap"] == 4.25e14 and _kr.loc[0, "shares"] == 5969782550 and np.isnan(_kr.loc[1, "close"])
    assert kr_data.parse_krx_daily([]).empty


def test_vkospi_parsing():
    """VKOSPI: 파생상품지수 표에서 '코스피 200 변동성지수'만 (그 선물지수 등은 아님), 쉼표 정리, 휴장일·실패는 NaN"""
    assert kr_data.is_vkospi("코스피 200 변동성지수") and kr_data.is_vkospi("코스피200 변동성지수")
    assert not any(kr_data.is_vkospi(n) for n in ("코스피 200 변동성지수 선물지수", "코스피 200 선물지수", "코스피 200 가치저변동성",
                                                     "코스피 200 변동성매칭 양매도지수", "코스피 200 변동성추세 추종 양매도지수"))
    _drv = {"20261023": pd.DataFrame([{"IDX_NM": "코스피 200 선물지수", "CLSPRC_IDX": "1,234.5"},
                                      {"IDX_NM": "코스피 200 가치저변동성", "CLSPRC_IDX": "1,111"},
                                      {"IDX_NM": "코스피 200 변동성지수", "CLSPRC_IDX": "31.25"}]),
            "20261024": pd.DataFrame()}
    _drv0 = kr_data.krx_drv_index_day
    kr_data.krx_drv_index_day = lambda d: _drv[pd.Timestamp(d).strftime("%Y%m%d")] if pd.Timestamp(d).day != 27 else 1 / 0
    try:
        _vk, _vn = kr_data.vkospi(pd.to_datetime(["2026-10-23", "2026-10-24", "2026-10-27"]), workers=1)
    finally:
        kr_data.krx_drv_index_day = _drv0
    assert _vk.iloc[0] == 31.25 and _vk.iloc[1:].isna().all() and _vn == ["코스피 200 변동성지수"]


def test_korean_factors():
    """심어 둔 구조: 시가총액이 작을수록, 장부가/시총이 클수록 매일 조금 더 오른다 → 소형주·가치 요인이 양(+)으로 나와야"""
    _n, _T = 100, 700
    _g = np.random.default_rng(3)
    _size, _val = _g.standard_normal(_n), _g.standard_normal(_n)
    _ret = 0.01 * _g.standard_normal((_T, _n)) - 0.002 * _size + 0.001 * _val
    _codes = [f"{i:06d}" for i in range(_n)]
    _dk = pd.bdate_range("2019-01-02", periods=_T)
    _ck = pd.DataFrame(100 * np.cumprod(1 + _ret, axis=0), index=_dk, columns=_codes)
    _pk = {"close": _ck, "universe": pd.DataFrame(True, index=_dk, columns=_codes),
           "cap": pd.DataFrame(np.exp(20 + _size)[None, :].repeat(_T, 0), index=_dk, columns=_codes),
           "feat": {"bm": pd.DataFrame(_val[None, :].repeat(_T, 0), index=_dk, columns=_codes)}}
    _rfk = pd.Series(0.0001, index=_dk)
    _kf = uv.kr_factors(_pk, rf=_rfk)
    assert _kf["SMB"].mean() > 0 and qe.overlap_tstat(_kf["SMB"].dropna().values, 1) > 3
    assert _kf["HML"].mean() > 0 and qe.overlap_tstat(_kf["HML"].dropna().values, 1) > 3
    assert list(_kf.columns) == ["Mkt-RF", "SMB", "HML", "RF"]                  # 데이터가 없는 요인(수익성·모멘텀)은 뺌
    _w = _pk["cap"].iloc[9]                                                      # 시장 = 전날 시가총액 가중 − 무위험 수익률
    assert np.isclose(_kf["Mkt-RF"].iloc[9], (_w * _ck.pct_change().iloc[10]).sum() / _w.sum() - 0.0001)
    assert np.allclose(_kf["RF"], 0.0001)
    # 미래 정보 없음: T 이후 가격을 바꿔도 T까지의 요인 값은 그대로
    _ck2 = _ck.copy()
    _ck2.iloc[401:] *= np.exp(0.05 * _g.standard_normal((_T - 401, _n)))
    _kf2 = uv.kr_factors({**_pk, "close": _ck2}, rf=_rfk)
    assert np.allclose(_kf.loc[:_dk[400]].values, _kf2.loc[:_dk[400]].values, equal_nan=True)
    assert not np.allclose(_kf.loc[_dk[401]:, "SMB"].values, _kf2.loc[_dk[401]:, "SMB"].values, equal_nan=True)
    # 수익 분해에 그대로 들어간다 (French 데이터와 같은 열 이름)
    _lr = np.log1p(_ck.iloc[:, 0].pct_change()).fillna(0.0)
    _pos = pd.Series((np.arange(_T) // 30) % 2, index=_dk, dtype=float)
    _att = qe.timing_attribution(pd.DataFrame({"pos": _pos, "strat": _pos * _lr, "asset": _lr}), _kf)
    assert {"전략 (요인 모형)", "단순 보유 (요인 모형)"} <= set(_att["summary"].index)
    assert {"시장", "소형주", "가치"} <= set(_att["loadings"].index)


def test_kospi200_naver_chart():
    """코스피200: 네이버 차트 XML → 종가 (Yahoo엔 과거 기록이 거의 없음)"""
    _xml = ('<?xml version="1.0" encoding="EUC-KR" ?><protocol><chartdata symbol="KPI200">'
            '<item data="20261022|846.0|860.0|820.0|830.25|223456" /><item data="20261021|840.1|850.2|830.3|845.55|123456" />'
            '</chartdata></protocol>')
    _nv = md.parse_naver_chart(_xml)
    assert list(_nv.values) == [845.55, 830.25] and _nv.index[0] == pd.Timestamp("2026-10-21")
    assert md.NAVER_INDEX["^KS200"] == "KPI200" and "^KS200" in md.NAVER_FIRST


def test_universe_with_delisted():
    """지금 상장된 종목은 Yahoo 수정주가 그대로, Yahoo에 없는 폐지 종목은 KRX 등락률로 주가를 만들고 폐지 뒤엔 마지막 가격에 멈춤(현금). 우선주는 빼고, 폐지 뒤엔 종목 범위에서 빠지며, 들고 있던 기간의 정리매매 폭락은 수익률에 그대로 들어가야 한다"""
    _dk = pd.bdate_range("2021-01-04", periods=400)
    _yc = pd.DataFrame({"000010": 100 * np.exp(np.cumsum(np.full(400, 0.001)))}, index=_dk)
    _chg = pd.DataFrame({"000010": 0.1, "999990": 0.2, "999995": 0.0}, index=_dk)
    _chg.loc[_dk[290:295], "999990"] = -50.0                          # 정리매매: 닷새 동안 매일 −50%
    _kclose = pd.DataFrame({"000010": 50000.0, "999990": 1000.0, "999995": 500.0}, index=_dk)
    _kclose.loc[_dk[295]:, "999990"] = np.nan                         # 300일째부터 없음 = 상장폐지
    _chg.loc[_dk[295]:, "999990"] = np.nan
    _krx = {"close": _kclose, "chg_pct": _chg, "volume": _kclose * 0 + 1000.0, "value": _kclose * 1000.0,
            "cap": _kclose * 1e6, "shares": _kclose * 0 + 1e6,
            "names": pd.Series({"000010": "가", "999990": "폐지된회사", "999995": "가우"})}
    _mc2, _mr2, _mv2, _mn2 = uv.merge_krx(_yc, _yc.copy(), _yc * 0 + 1000.0, _krx)
    assert list(_mc2.columns) == ["000010", "999990"] and _mn2.to_dict() == {"999990": "폐지된회사"}     # 우선주(…우) 제외
    assert np.isclose(_mc2["999990"].iloc[295] / _mc2["999990"].iloc[289], 0.5 ** 5)                  # 정리매매 손실 반영
    assert (_mc2["999990"].iloc[295:] == _mc2["999990"].iloc[294]).all() and _mr2["999990"].iloc[295:].isna().all()
    assert np.allclose(_mc2["000010"], _yc["000010"])                                                 # 살아 있는 종목은 그대로
    _u2, _ = uv.point_in_time_universe(_mr2, _mv2, top_n=5, value=_krx["value"].reindex(columns=_mc2.columns))
    assert _u2["999990"].iloc[260] and not _u2["999990"].iloc[300:].any()                             # 폐지 뒤엔 범위 밖
    assert uv.load_krx(Path(tempfile.gettempdir()) / "없는파일.pkl") is None
