"""실제 데이터 검증 보고서 (시세는 합성 데이터로 바꿔 끼움)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import alt_data as ad
import direction as dr
import kr_costs as kc
import kr_data
import market_data as md
import universe as uv
import validation as vd


def test_validation_report():
    _orig_v = {k: getattr(md, k) for k in ("load_history", "load_macro", "kr_info", "load_ohlc")}
    _orig_va = ad.build_alt_features
    _hist = pd.Series(100 * np.exp(np.cumsum(0.0004 + 0.012 * np.random.default_rng(5).standard_normal(1800))),
                      index=pd.bdate_range("2019-01-02", periods=1800))
    def _hist_or_fail(t, years):
        if t == "없는종목.KS":
            raise ValueError("주가 데이터 없음")
        return _hist
    md.load_history = _hist_or_fail
    _hrg = np.random.default_rng(6)                                    # 종가에 맞춘 시가·고가·저가 (R1 계산용)
    _hopen = _hist.shift(1).bfill() * np.exp(0.002 * _hrg.standard_normal(len(_hist)))
    _hohlc = pd.DataFrame({"Open": _hopen, "High": np.maximum(_hopen, _hist) * (1 + 0.005 * np.abs(_hrg.standard_normal(len(_hist)))),
                           "Low": np.minimum(_hopen, _hist) * (1 - 0.005 * np.abs(_hrg.standard_normal(len(_hist)))), "Close": _hist})
    md.load_ohlc = lambda t, years: _hohlc
    md.load_macro = lambda t, years=10: None
    md.kr_info = lambda t: ("테스트전자", "KOSPI", "stock")
    ad.build_alt_features = lambda t, idx: (None, {})
    _vk_orig = (vd.VKOSPI_TICKERS, vd._vkospi_prev)
    vd.VKOSPI_TICKERS, vd._vkospi_prev = ("005930.KS",), \
        lambda dates, lag=1: np.full(len(dates), 25.0) if lag == 1 else np.r_[np.nan, np.nan, np.full(len(dates) - 2, 30.0)]   # 이 종목도 VKOSPI 섞기 대상인 척
    try:
        try:
            _vr = vd.validate("005930.KS")
        finally:
            vd.VKOSPI_TICKERS, vd._vkospi_prev = _vk_orig
        # σ 보정 두 후보가 같은 예측을 다시 채점: 비율·검정·배수 (VKOSPI는 검증 기간 모든 날에 값)
        for _m in ("recal", "vkospi", "vkospi2"):
            assert _vr[f"[{_m}] VaR 검정"] in ("통과", "기각") and _vr[f"[{_m}] ES 검정"] in ("통과", "기각")
            assert all(np.isfinite(_vr[f"분위수 손실 비율 ({_m} ÷ 기본){_p}"]) for _p in ("", " 앞 절반", " 뒤 절반"))
        assert _vr["[vkospi] VKOSPI 있는 날"] == f"{_vr['VaR 검증 일수']}/{_vr['VaR 검증 일수']}" and 0.8 <= _vr["[recal] 배수 중앙값"] <= 1.5
        # C1L2: 하루 더 지난 VKOSPI (앞 두 날은 값이 없음) → C1과 다른 배수
        assert _vr["[vkospi2] VKOSPI 있는 날"] == f"{_vr['VaR 검증 일수'] - 2}/{_vr['VaR 검증 일수']}"
        assert _vr["[vkospi2] 배수 중앙값"] > _vr["[vkospi] 배수 중앙값"] and _vr["[vkospi2] 분위수 손실"] != _vr["[vkospi] 분위수 손실"]
        assert np.isclose(_vr["[recal] 분위수 손실"] / _vr["[t] 분위수 손실"], _vr["분위수 손실 비율 (recal ÷ 기본)"], rtol=1e-6)
        assert _vr["세금 구분"] == "KOSPI" and _vr["방향 예측력"] in ("있음", "확인 안 됨") and 0 <= _vr["가정 바꿔 통과(7개 중)"] <= 7
        assert _vr["방향 초과 95% 하한"] <= _vr["방향 초과수익(연)"] <= _vr["방향 초과 95% 상한"]
        assert _vr["국면 초과 95% 하한"] <= _vr["국면 초과수익(연)"] <= _vr["국면 초과 95% 상한"]
        assert _vr["방향 전략 경제적 가치"] == ("있음" if _vr["방향 초과 95% 하한"] > 0 else "확인 안 됨")
        assert _vr["분할 초과 95% 하한"] <= _vr["분할 초과수익(연)"] <= _vr["분할 초과 95% 상한"] and 0 < _vr["분할 보유 평균 비중"] <= 1
        assert len(_vr["_obs"]) > 0 and (_vr["_obs"]["c"] >= 2 * vd.COST_BPS * 1e-4).all()      # 비용 = 수수료 양쪽 + 세금
        assert _vr["ES 검정"] in ("통과", "기각") and _vr["VaR 검증 일수"] > 500 and _vr["GARCH 진단"].count("수렴") == 1
        assert _vr["분위수 손실 비율 (FHS ÷ t)"] > 0
        # VKOSPI 섞기 대상 종목은 본 표가 실전에서 받을 수 있는 전전 거래일 VKOSPI를 섞은 VaR (C1L2, 2026-10-30 판정)
        assert _vr["VaR σ 보정"] == "VKOSPI 섞기" and _vr["VaR 검정"] == _vr["[vkospi2] VaR 검정"]
        assert _vr["VaR 95% 예외 비율"] == _vr["[vkospi2] VaR 95% 예외"] and _vr["VaR 99% 예외 비율"] == _vr["[vkospi2] VaR 99% 예외"]
        # 예외 몰림 후보 K2는 지금 기본(이 종목은 VKOSPI 섞기)과 비교
        assert _vr["[cur] VaR 검정"] == _vr["VaR 검정"] and _vr["[k2] VaR 검정"] in ("통과", "기각")
        assert all(np.isfinite(_vr[f"분위수 손실 비율 (k2 ÷ 현재){_p}"]) for _p in ("", " 앞 절반", " 뒤 절반"))
        assert 0 <= _vr["[k2] 99% 독립성 p"] <= 1 and 0 <= _vr["[cur] 95% 독립성 p"] <= 1
        assert _vr["[m1] VaR 검정"] in ("통과", "기각") and all(np.isfinite(_vr[f"분위수 손실 비율 (m1 ÷ 현재){_p}"]) for _p in ("", " 앞 절반", " 뒤 절반"))
        assert 0 <= _vr["[cur] 휴장 뒤 95% 예외"] <= 1 and 0 <= _vr["[m1] 휴장 뒤 95% 예외"] <= 1
        assert _vr["[r1] 유효한 날 비율"] == 1.0 and _vr["[r1] VaR 검정"] in ("통과", "기각")
        assert all(np.isfinite(_vr[f"분위수 손실 비율 (r1 ÷ 이전 기본){_p}"]) for _p in ("", " 앞 절반", " 뒤 절반"))
        assert _vr["[pre] VaR 검정"] == _vr["VaR 검정"]                 # 코스피 200 종목은 R1을 쓰지 않음 → 본 표 = R1 이전(VKOSPI 섞기)
        # 사전 등록 규칙: 분위수 손실 비율 중앙값 < 1 이고 VaR·ES 통과 수가 줄지 않을 때만 FHS
        _mk = lambda ratio, vt_, vf, et, ef: {"[t] VaR 검정": vt_, "[fhs] VaR 검정": vf, "[t] ES 검정": et, "[fhs] ES 검정": ef,
                                              "분위수 손실 비율 (FHS ÷ t)": ratio}
        assert vd.var_tail_rule([_mk(0.98, "기각", "통과", "통과", "통과"), _mk(0.99, "통과", "통과", "통과", "통과")])["FHS 채택"]
        assert not vd.var_tail_rule([_mk(0.98, "통과", "기각", "통과", "통과"), _mk(0.99, "통과", "통과", "통과", "통과")])["FHS 채택"]
        assert not vd.var_tail_rule([_mk(1.02, "기각", "통과", "통과", "통과"), _mk(1.01, "통과", "통과", "통과", "통과")])["FHS 채택"]
        # σ 보정 사전 등록 규칙: C2는 중앙값이 세 구간 모두 < 1이고 VaR·ES 통과 수 유지, C1은 두 지수 종목 모두 세 구간 < 1이고 통과 수 유지.
        # 둘 다 채택이면 지수 종목에는 분위수 손실 합이 낮은 쪽
        def _ms(t, c2, vt_="통과", v2="통과", c1=None, v1="통과", l2=10.0, l1=9.0):
            r = {"종목": t, "[t] VaR 검정": vt_, "[recal] VaR 검정": v2, "[t] ES 검정": "통과", "[recal] ES 검정": "통과",
                 "[recal] 분위수 손실": l2}
            r.update({f"분위수 손실 비율 (recal ÷ 기본){p}": x for p, x in zip(("", " 앞 절반", " 뒤 절반"), c2)})
            if c1 is not None:
                r.update({"[vkospi] VaR 검정": v1, "[vkospi] ES 검정": "통과", "[vkospi] 분위수 손실": l1})
                r.update({f"분위수 손실 비율 (vkospi ÷ 기본){p}": x for p, x in zip(("", " 앞 절반", " 뒤 절반"), c1)})
            return r
        _gd, _bd = (0.97, 0.98, 0.96), (0.97, 1.01, 0.95)
        _o = vd.var_scale_rule([_ms("^KS11", _gd, "기각", c1=_gd), _ms("069500.KS", _gd, c1=_gd), _ms("005930.KS", _gd)])
        assert _o["C2 채택"] and _o["C1 채택"] and _o["코스피 200 종목에 쓸 것"] == vd.VAR_SCALES["vkospi"]
        assert vd.var_scale_rule([_ms("^KS11", _gd, c1=_gd, l1=11.0), _ms("069500.KS", _gd, c1=_gd, l1=11.0)])["코스피 200 종목에 쓸 것"] \
            == vd.VAR_SCALES["recal"]
        assert not vd.var_scale_rule([_ms("005930.KS", _bd), _ms("000660.KS", _bd)])["C2 채택"]
        assert not vd.var_scale_rule([_ms("005930.KS", _gd, v2="기각"), _ms("000660.KS", _gd)])["C2 채택"]
        _o2 = vd.var_scale_rule([_ms("^KS11", _gd, c1=_gd), _ms("069500.KS", _gd, c1=_bd)])
        assert _o2["C2 채택"] and not _o2["C1 채택"] and _o2["코스피 200 종목에 쓸 것"] == vd.VAR_SCALES["recal"]
        _o3 = vd.var_scale_rule([_ms("^KS11", _bd, c1=_gd), _ms("069500.KS", _bd, c1=_gd)])
        assert not _o3["C2 채택"] and _o3["C1 채택"] and _o3["다른 종목에 쓸 것"] == "기본 그대로"
        assert not vd.var_scale_rule([_ms("^KS11", _gd, c1=_gd)])["C1 채택"]          # 두 종목 모두 있어야
        # C1L2 규칙 (C1과 같음): 두 종목 모두 세 구간 < 1, 두 종목의 VaR·ES 통과 수 유지. C1L2 ÷ C1은 보고만
        def _ml(t, c, vt_="기각", vl="통과", et="통과", el="통과"):
            r = {"종목": t, "[t] VaR 검정": vt_, "[vkospi2] VaR 검정": vl, "[t] ES 검정": et, "[vkospi2] ES 검정": el,
                 "[vkospi] 분위수 손실": 10.0, "[vkospi2] 분위수 손실": 10.2}
            r.update({f"[{m}] VaR {lv}% 예외": x for m, x in (("t", 0.07), ("vkospi", 0.05), ("vkospi2", 0.055)) for lv in (95, 99)})
            r.update({f"분위수 손실 비율 (vkospi2 ÷ 기본){p}": x for p, x in zip(("", " 앞 절반", " 뒤 절반"), c)})
            return r
        _lr = vd.var_iv_lag_rule([_ml("^KS11", _gd), _ml("069500.KS", _gd), _ml("005930.KS", _bd)])
        assert _lr["C1L2 채택"] and _lr["대상"] == 2 and np.isclose(_lr["^KS11 C1L2 ÷ C1"], 1.02)
        assert _lr["VaR 검정 통과 기본 → C1L2"] == (0, 2) and _lr["^KS11 VaR 95% 예외 기본 · C1 · C1L2"] == (0.07, 0.05, 0.055)
        assert not vd.var_iv_lag_rule([_ml("^KS11", _gd), _ml("069500.KS", _bd)])["C1L2 채택"]                 # 앞 절반 1.01
        assert not vd.var_iv_lag_rule([_ml("^KS11", _gd, vt_="통과", vl="기각"), _ml("069500.KS", _gd, vl="기각")])["C1L2 채택"]   # 두 종목 VaR 통과 1 → 0
        assert vd.var_iv_lag_rule([_ml("^KS11", _gd, vt_="통과", vl="기각"), _ml("069500.KS", _gd)])["C1L2 채택"]       # 합이 1 → 1이면 유지
        assert not vd.var_iv_lag_rule([_ml("^KS11", _gd)])["C1L2 채택"] and vd.var_iv_lag_rule([_ml("005930.KS", _gd)]) is None
        assert "**C1L2 채택**" in "\n".join(vd._iv_lag_lines([_ml("^KS11", _gd), _ml("069500.KS", _gd)]))
        # 예외 몰림 K2 규칙: 코스피 200 두 종목과 나머지를 따로. 세 구간 모두 < 1, VaR·ES 통과 수 유지
        def _mk2(t, k2, vc="통과", vk="통과", ec="통과", ek="통과"):
            r = {"종목": t, "[cur] VaR 검정": vc, "[k2] VaR 검정": vk, "[cur] ES 검정": ec, "[k2] ES 검정": ek,
                 "[cur] 99% 독립성 p": 0.01, "[k2] 99% 독립성 p": 0.2}
            r.update({f"분위수 손실 비율 (k2 ÷ 현재){p}": x for p, x in zip(("", " 앞 절반", " 뒤 절반"), k2)})
            return r
        _q = vd.var_cluster_rule([_mk2("^KS11", _gd, vc="기각"), _mk2("069500.KS", _gd), _mk2("005930.KS", _bd), _mk2("000660.KS", _bd)])
        assert _q["코스피 200 K2 채택"] and not _q["나머지 K2 채택"] and _q["코스피 200 VaR 검정 통과 현재 → K2"] == (1, 2)
        assert not vd.var_cluster_rule([_mk2("^KS11", _gd), _mk2("069500.KS", _bd)])["코스피 200 K2 채택"]
        assert not vd.var_cluster_rule([_mk2("^KS11", _gd, ek="기각"), _mk2("069500.KS", _gd)])["코스피 200 K2 채택"]
        _q2 = vd.var_cluster_rule([_mk2("005930.KS", _gd), _mk2("000660.KS", _gd), _mk2("035420.KS", _bd)])
        assert _q2["나머지 K2 채택"] and not _q2["코스피 200 K2 채택"] and _q2["나머지 K2가 나은 종목"] == 3
        assert not vd.var_cluster_rule([_mk2("005930.KS", _gd, vk="기각"), _mk2("000660.KS", _gd)])["나머지 K2 채택"]
        # 휴장 뒤 첫 거래일 M1도 같은 규칙 (열 이름만 m1)
        _mm = lambda t, x, **kw: {k.replace("k2", "m1"): v for k, v in _mk2(t, x, **kw).items()}
        _q3 = vd.var_cluster_rule([_mm("^KS11", _gd), _mm("069500.KS", _gd), _mm("005930.KS", _bd), _mm("000660.KS", _bd),
                                   _mm("035420.KS", _gd)], "m1")
        assert _q3["코스피 200 M1 채택"] and not _q3["나머지 M1 채택"] and _q3["나머지 비율 중앙값 앞 절반"] == 1.01
        assert vd.var_cluster_rule([_mk2("005930.KS", _gd)], "m1") is None                  # M1 열이 없으면 판정 없음
        # 전날 VKOSPI: 수익률 날짜마다 그 전 거래일 값 (첫날은 없음)
        _gk0, _vk0 = kr_data.get_key, kr_data.vkospi
        kr_data.get_key = lambda n: "x"
        kr_data.vkospi = lambda dates, workers=4: (pd.Series(np.arange(len(dates), dtype=float), index=dates), ["코스피 200 변동성지수"])
        try:
            _pv = vd._vkospi_prev(_hist.index[:5])
            _pv2 = vd._vkospi_prev(_hist.index[:5], lag=2)                 # C1L2: 전전 거래일 값 (실전에서 받을 수 있는 것)
        finally:
            kr_data.get_key, kr_data.vkospi = _gk0, _vk0
        assert np.isnan(_pv[0]) and np.array_equal(_pv[1:], np.arange(4.0))
        assert np.isnan(_pv2[:2]).all() and np.array_equal(_pv2[2:], np.arange(3.0))
        # 같은 데이터로 앱 방향 예측 탭과 같은 숫자 (같은 함수·같은 기본 설정)
        _wfv, _ = dr.walk_forward(_hist, 21, None, alt=None, ticker="005930.KS")
        _evv = dr.evaluate(_wfv, 21, np.log(_hist).diff(), 3.0, lag=1, sell_tax=kc.sell_tax_rate(_hist.index, "KOSPI"))
        assert np.isclose(_vr["방향 z"], _evv["z"]) and np.isclose(_vr["방향 전략 최대 낙폭"], _evv["mdd"])
        # 최근 1년을 뺀 재채점 = 마지막 1년을 지우고 처음부터 다시 돌린 것과 같다 (워크포워드라 그 전 예측은 이후 데이터와 무관)
        _cut = _hist.index[-uv.RECENT_DAYS - 1]
        assert _vr["[최근1년 제외] 끝"] == str(_cut.date())
        _wfc2, _ = dr.walk_forward(_hist.loc[:_cut], 21, None, alt=None, ticker="005930.KS")
        _evc2 = dr.evaluate(_wfc2, 21, np.log(_hist.loc[:_cut]).diff(), 3.0, lag=1, sell_tax=kc.sell_tax_rate(_hist.index, "KOSPI"))
        assert np.isclose(_vr["[최근1년 제외] 방향 z"], _evc2["z"]) and np.isclose(_vr["[최근1년 제외] 확률 정확도"], _evc2["skill"])
        assert np.isfinite(_vr["분위수 손실 비율 앞 절반"]) and np.isfinite(_vr["분위수 손실 비율 뒤 절반"])
        with tempfile.TemporaryDirectory() as _vd:
            _rows = vd.run(["005930.KS", "없는종목.KS"], out_dir=_vd, say=lambda m: None)
            _md = (Path(_vd) / "README.md").read_text(encoding="utf-8")
            assert "테스트전자" in _md and "계산하지 못한 종목" in _md and "없는종목.KS: ValueError" in _md   # 한 종목 실패해도 나머지는 보고
            assert "## 21일 분할 보유 전략" in _md and "## 예측 확률 구간별 실제 21일 수익" in _md and "## 하루 VaR·ES 검정" in _md
            assert "## 하루 VaR 꼬리 분포 비교" in _md and "판정:" in _md and "앞 절반" in _md
            assert "## 하루 VaR σ 보정" in _md and ("**C2 채택**" in _md or "**C2 미채택**" in _md)
            assert "## 하루 VaR 예외 몰림" in _md and ("나머지 1종목" in _md) and "## 하루 VaR 휴장 뒤 첫 거래일" in _md
            assert "## 하루 VaR 장중 고가·저가 변동성 섞기(R1)" in _md
            assert "## 최근 1년(이례적 급등락 구간)을 뺀 결과" in _md
            # 코스피 200 종목이 아니면 본 표 = 장중 고가·저가 섞기(R1), R1 비교 기준은 기본(t), K2·M1 비교 기준은 R1
            assert _rows[0]["VaR σ 보정"] == "장중 고가·저가 섞기" and _rows[0]["VaR 검정"] == _rows[0]["[r1] VaR 검정"]
            assert _rows[0]["VaR 95% 예외 비율"] == _rows[0]["[r1] VaR 95% 예외"] and _rows[0]["[pre] VaR 검정"] == _rows[0]["[t] VaR 검정"]
            assert _rows[0]["[cur] VaR 검정"] == _rows[0]["[r1] VaR 검정"] and "†" in _md
            _sv = pd.read_csv(Path(_vd) / "summary.csv")
            assert len(_sv) == 2 and pd.isna(_sv.loc[0, "오류"]) and "주가 데이터 없음" in _sv.loc[1, "오류"]
        # 유효한 일봉이 80% 미만이면 R1은 비교하지 않고 비율만 남김 (규칙의 '비교 못 한 종목')
        _bad = _hohlc.copy()
        _bad.loc[_bad.index[len(_bad) // 3:], "High"] = _bad["Low"]                   # 뒤쪽 2/3: 고가 = 저가 → 무효
        md.load_ohlc = lambda t, years: _bad
        _vb = vd.validate("005930.KS")
        assert _vb["[r1] 유효한 날 비율"] < vd.R1_MIN_VALID and "[r1] VaR 검정" not in _vb
        assert _vb["VaR σ 보정"] == "없음" and _vb["VaR 검정"] == _vb["[t] VaR 검정"]      # 비교 못 하면 GARCH만
        assert vd.var_cluster_rule([_vr, _vb], "r1")["비교 못 한 종목"] == ["005930.KS"]
    finally:
        for _k, _v in _orig_v.items():
            setattr(md, _k, _v)
        ad.build_alt_features = _orig_va
    print("모든 테스트 통과")
