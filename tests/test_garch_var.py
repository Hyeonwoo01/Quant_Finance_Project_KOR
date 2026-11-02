"""변동성·VaR: t분포·GJR GARCH, VaR·ES 검정, 꼬리 분포(FHS), σ 보정 후보(VKOSPI·재보정·장중 고가·저가)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import chi2

import market_data as md
import quant_engine as qe
from quant_engine import (GarchVolForecaster, max_drawdown, var_es, var_backtest, var_exception_tests, _std_es, es_exceedance_test, _std_quantile as _sq, iv_blend_factor, recal_factor, var_rescale, cond_recal_factor, after_break, group_recal_factor, ewma_level, range_blend_factor, range_variance, day1_var_factor, range_level_now, fit_vol_model)
from tests.helpers import _sim_gjr, simulate_garch, simulate_garch_t


def test_garch_t_fit_and_failures():
    gt = GarchVolForecaster().fit(simulate_garch_t(6000, 4e-6, 0.08, 0.90, 5.0, np.random.default_rng(11)))
    assert 3.5 < gt.nu < 7.5 and abs(gt.persistence - 0.98) < 0.03, (gt.nu, gt.persistence)
    # 시작값 셋 모두에서 최적화해 가장 좋은 해를 쓰고, 진단을 남긴다 (모두 수렴, 같은 답이면 우도 차이 ≈ 0)
    assert gt.fit_info["시작값"] == 3 and gt.fit_info["수렴"] == 3 and 0 <= gt.fit_info["우도 차이"] < 0.01 and gt.fit_info["경계"] == []
    assert GarchVolForecaster(dist="normal").fit(simulate_garch(3000, 4e-6, 0.08, 0.90, np.random.default_rng(3))).nu == np.inf
    sim = gt.simulate(21, 20000, seed=1)
    assert sim.shape == (20000, 21)
    assert abs(sim[:, -1].std() - np.sqrt(gt.daily_var_path(21).sum())) / sim[:, -1].std() < 0.05   # 분산 경로와 일치
    gn_ = GarchVolForecaster(dist="normal").fit(simulate_garch(3000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
    z1 = gn_.simulate(1, 200000, seed=2)[:, 0]
    assert abs(np.quantile(z1, 0.01) / np.sqrt(gn_.sigma2_next) + 2.326) < 0.03                  # 정규면 1% 분위수 = −2.326σ
    assert np.quantile(gt.simulate(1, 200000, seed=2)[:, 0], 0.001) / np.sqrt(gt.sigma2_next) < -3.3   # t는 꼬리가 더 두껍다

    # GARCH 추정 실패: 최적화가 실패하면 조용히 쓰지 않고 기본값으로 대체 + 경고
    _orig_min = qe.minimize
    qe.minimize = lambda *a, **k: SimpleNamespace(success=False, x=np.full(4, np.nan), fun=np.nan)
    try:
        gf = GarchVolForecaster().fit(simulate_garch(1000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
    finally:
        qe.minimize = _orig_min
    assert not gf.converged and gf.warnings and abs(gf.persistence - 0.95) < 1e-12 and gf.nu == 8.0
    assert np.isfinite(gf.forecast_vol(21)) and np.isfinite(gf.simulate(5, 100)).all()
    assert gt.converged and not GarchVolForecaster().fit(np.zeros(10)).converged          # 표본이 너무 적으면 바로 기본값


def test_var_exception_tests():
    """VaR 예외 검정: Kupiec 공식, 몰려 나온 예외는 Christoffersen이 잡고, 예외가 0개(너무 보수적)여도 기각"""
    _h = np.zeros(250, bool)
    _h[np.random.default_rng(0).choice(250, 13, replace=False)] = True
    _k = var_exception_tests(_h, 0.05)
    _pi = 13 / 250
    _lr = -2 * ((237 * np.log(0.95) + 13 * np.log(0.05)) - (237 * np.log(1 - _pi) + 13 * np.log(_pi)))
    assert _k["예외 횟수"] == 13 and abs(_k["Kupiec p값"] - chi2.sf(_lr, 1)) < 1e-12
    _cl_h = np.zeros(250, bool)
    _cl_h[100:113] = True
    assert var_exception_tests(_cl_h, 0.05)["Christoffersen p값"] < 0.01 and abs(var_exception_tests(_cl_h, 0.05)["Kupiec p값"] - _k["Kupiec p값"]) < 1e-12
    assert var_exception_tests(np.zeros(250, bool), 0.05)["Kupiec p값"] < 0.01
    _vt = var_backtest(simulate_garch_t(1500, 4e-6, 0.08, 0.90, 5.0, np.random.default_rng(12)))
    assert list(_vt.index) == ["하루 VaR 95%", "하루 VaR 99%"] and _vt.loc["하루 VaR 95%", "관측일"] == 500
    assert 0.02 < _vt.loc["하루 VaR 95%", "예외 비율"] < 0.09


def test_es_and_fhs_tails():
    """ES: t분포 충격의 꼬리 평균 공식이 시뮬레이션과 같고, 맞는 모델이면 '예외일 손실 ÷ 예측 ES' ≈ 1, 꼬리를 얇게 잡은 모델(정규분포)이면 1보다 크고 ES 검정이 기각, 분위수 손실도 더 크다"""
    _zt = np.random.default_rng(2).standard_t(5, 1_000_000) * np.sqrt(3 / 5)
    assert abs(-_zt[_zt <= np.quantile(_zt, 0.01)].mean() / _std_es(0.01, 5.0) - 1) < 0.01
    assert abs(_std_es(0.05, np.inf) - 2.0627) < 1e-3
    _xv = simulate_garch_t(2300, 4e-6, 0.08, 0.90, 4.0, np.random.default_rng(14))
    _vt_t, _vd_t = var_backtest(_xv, start=800, refit_every=42, detail=True)
    _vt_n = var_backtest(_xv, start=800, refit_every=42, dist="normal")
    assert _vt_t.loc["하루 VaR 95%", "관측일"] == 1500 and len(_vd_t) == 1500 and _vt_t["추정 실패"].iloc[0] == 0
    assert 0.85 < _vt_t.loc["하루 VaR 95%", "예외일 손실 ÷ 예측 ES"] < 1.15 and _vt_t.loc["하루 VaR 95%", "ES 검정 p값"] > 0.05
    assert _vt_n.loc["하루 VaR 99%", "예외일 손실 ÷ 예측 ES"] > 1.15 and _vt_n.loc["하루 VaR 95%", "ES 검정 p값"] < 0.01
    assert _vt_n.loc["하루 VaR 99%", "분위수 손실(×10⁴)"] > _vt_t.loc["하루 VaR 99%", "분위수 손실(×10⁴)"]
    assert (_vd_t["예외 95%"] == (_vd_t["수익률"] < -_vd_t["VaR 95%"])).all() and (_vd_t["ES 99%"] > _vd_t["VaR 99%"]).all()
    assert np.isnan(es_exceedance_test([0.03, 0.04], [0.02, 0.02], [0.01, 0.01])[1])          # 예외 3번 미만 → p값 없음

    # 꼬리 분포 FHS(실제 잔차): 't'는 추정한 t분포 공식과 같고, 'fhs'는 적합 구간 표준화 잔차의 실제 분위수·꼬리 평균
    _gf = GarchVolForecaster().fit(_xv[:750])
    assert _gf.tail(0.05, "t") == (_sq(0.05, _gf.nu), _std_es(0.05, _gf.nu)) and len(_gf.std_resid) == 750
    _zq, _ze = _gf.tail(0.01, "fhs")
    assert np.isclose(_zq, np.quantile(_gf.std_resid, 0.01)) and np.isclose(_ze, -_gf.std_resid[_gf.std_resid <= _zq].mean())
    assert set(np.round(_gf.simulate(1, 500, innov="fhs")[:, 0] / np.sqrt(_gf.sigma2_next), 9)) <= set(np.round(_gf.std_resid, 9))
    assert np.allclose(_gf.simulate(3, 200), _gf.simulate(3, 200, innov="t"))                   # 기본 = VAR_INNOV ('t')
    # 여러 방법을 한 번에: 같은 GARCH 적합을 공유하고, 't' 결과는 하나만 계산한 것과 같다
    _vb2 = var_backtest(_xv, start=800, refit_every=42, innov=("t", "fhs"))
    assert _vb2["t"].equals(_vt_t) and set(_vb2) == {"t", "fhs"}


def test_fhs_on_skewed_shocks():
    """하락 쪽 꼬리가 더 두꺼운 충격(지수처럼): 대칭 t는 95% VaR를 넘는 날이 목표보다 많고, FHS는 목표 근처"""
    _tt = np.random.default_rng(1).standard_t(5, 2300)
    _zs = np.where(_tt < 0, 1.4 * _tt, 0.7 * _tt)
    _zs = (_zs - _zs.mean()) / _zs.std()
    _xs, _s2 = np.empty(2300), 1e-4
    for _i in range(2300):
        _xs[_i] = np.sqrt(_s2) * _zs[_i]
        _s2 = 2e-6 + 0.08 * _xs[_i] ** 2 + 0.9 * _s2
    _vsk = var_backtest(_xs, start=800, refit_every=42, innov=("t", "fhs"))
    assert _vsk["t"].loc["하루 VaR 95%", "예외 비율"] > 0.06 and _vsk["t"].loc["하루 VaR 95%", "종합 p값"] < 0.05
    assert 0.04 < _vsk["fhs"].loc["하루 VaR 95%", "예외 비율"] < 0.06 and _vsk["fhs"].loc["하루 VaR 95%", "종합 p값"] > 0.05
    assert abs(_vsk["fhs"].loc["하루 VaR 95%", "예외일 손실 ÷ 예측 ES"] - 1) < abs(_vsk["t"].loc["하루 VaR 95%", "예외일 손실 ÷ 예측 ES"] - 1)
    assert _vsk["fhs"]["분위수 손실(×10⁴)"].sum() < _vsk["t"]["분위수 손실(×10⁴)"].sum()


def test_var_rescale():
    """σ 보정 (2026-10-29 사전 등록 후보): 배수 1이면 원래 채점 그대로, 배수 c면 VaR·ES·σ가 모두 c배 (NaN = 그대로)"""
    _xv = simulate_garch_t(2300, 4e-6, 0.08, 0.90, 4.0, np.random.default_rng(14))
    _vt1, _vd1 = var_backtest(_xv, start=800, refit_every=42, detail=True)
    _rt1, _rd1 = var_rescale(_vt1, _vd1, np.ones(len(_vd1)))
    pd.testing.assert_frame_equal(_rt1, _vt1)
    pd.testing.assert_frame_equal(_rd1, _vd1)
    _rt2, _rd2 = var_rescale(_vt1, _vd1, np.r_[np.full(len(_vd1) - 1, 1.3), np.nan])
    assert np.allclose(_rd2["VaR 99%"].iloc[:-1], 1.3 * _vd1["VaR 99%"].iloc[:-1]) and _rd2["ES 95%"].iloc[-1] == _vd1["ES 95%"].iloc[-1]
    assert np.allclose(_rd2["변동성"].iloc[:-1], 1.3 * _vd1["변동성"].iloc[:-1])
    assert _rt2.loc["하루 VaR 95%", "예외 비율"] <= _vt1.loc["하루 VaR 95%", "예외 비율"] and _rt2.loc["하루 VaR 95%", "재적합 수"] == _vt1.loc["하루 VaR 95%", "재적합 수"]


def test_recal_factor():
    """크기 재보정: 그날을 뺀 지난 window일 z² 평균의 제곱근, 지난 날이 모자라면 1, 0.8~1.5로 자름 — 그날 이후 값은 영향 없음"""
    _rx = np.random.default_rng(3).standard_normal(400) * 0.02
    _rs = np.full(400, 0.01)                                   # 실제의 절반으로 잡은 σ → 배수 약 2 → 상한 1.5
    _rc = recal_factor(_rx, _rs, window=100, min_obs=50)
    assert (_rc[:50] == 1).all() and np.isclose(_rc[50], min(1.5, np.sqrt(np.mean((_rx[:50] / 0.01) ** 2)))) and (_rc[100:] == 1.5).all()
    _rx2 = _rx.copy()
    _rx2[300:] *= 0.1
    _rc2 = recal_factor(_rx2, _rs, window=100, min_obs=50)
    assert np.array_equal(_rc2[:301], _rc[:301]) and _rc2[-1] < _rc[-1]
    assert (np.abs(recal_factor(_rx * 0.5, _rs, 100, 50)[100:] - 1) < 0.2).all()   # σ가 맞으면 1 근처


def test_conditional_and_after_break_recal():
    """전날 움직임별 재보정(K2): 전날 z 구간(< −1, 사이, > 1)마다 그날 전까지 같은 구간이던 날들의 z² 평균의 제곱근 (0.8~1.5)"""
    _kr2 = np.tile([-3.0, 2.0, 0.5, 0.5], 60)                    # σ = 1: 큰 하락 다음 날은 늘 z = 2, 큰 상승 다음 날은 z = 0.5
    _kc = cond_recal_factor(_kr2, np.ones_like(_kr2), min_obs=3)
    _after_dn, _after_up = np.arange(1, 240, 4), np.arange(2, 240, 4)
    assert (_kc[_after_dn[:3]] == 1).all() and (_kc[_after_dn[3:]] == 1.5).all()      # 하락 다음 날: √4 = 2 → 상한 1.5
    assert (_kc[_after_up[:3]] == 1).all() and (_kc[_after_up[3:]] == 0.8).all()      # 상승 다음 날: √0.25 = 0.5 → 하한 0.8
    _mid = np.setdiff1d(np.arange(240), np.r_[_after_dn, _after_up])
    assert np.isclose(_kc[_mid[-1]], min(1.5, np.sqrt(np.mean(_kr2[_mid[:-1]] ** 2))))
    _kr3 = _kr2.copy()
    _kr3[150:] *= 3
    assert np.array_equal(cond_recal_factor(_kr3, np.ones_like(_kr3), min_obs=3)[:151], _kc[:151])   # 그날 이후 값은 영향 없음
    # 휴장 뒤 첫 거래일(M1): 바로 전 거래일과 달력상 2일 넘게 떨어진 날(월요일·연휴 다음 날)과 그 밖의 날을 따로 재보정
    _bd = pd.to_datetime(["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-29", "2026-10-30"])
    assert after_break(_bd, pd.Timestamp("2026-10-22")).tolist() == [False, True, False, True, False]
    _mdays = pd.bdate_range("2025-01-06", periods=300)
    _mb = after_break(_mdays, _mdays[0] - pd.Timedelta(days=3))
    assert _mb.sum() == 60 and _mb[0] and not _mb[1]                       # 월요일마다 (첫날도 금요일 다음 월요일)
    _mr = np.where(_mb, 2.0, 0.5) * np.where(np.arange(300) % 2, 1, -1)      # σ = 1: 월요일은 |z| = 2, 나머지는 0.5
    _mc = group_recal_factor(_mr, np.ones(300), _mb, min_obs=5)
    assert (_mc[np.flatnonzero(_mb)[:5]] == 1).all() and (_mc[np.flatnonzero(_mb)[5:]] == 1.5).all()
    assert (_mc[np.flatnonzero(~_mb)[5:]] == 0.8).all()
    assert np.array_equal(cond_recal_factor(_kr2, np.ones_like(_kr2), min_obs=3), _kc)   # K2도 같은 재보정의 한 경우


def test_intraday_range_variance():
    """장중 고가·저가 변동성(R1): 밤사이 갭² + Rogers–Satchell. 장중을 잘게 쪼갠 모의 주가에서 실제 분산 근처(이산 관측이라 조금 작음)"""
    _rg = np.random.default_rng(4)
    _ohlc_rows, _cp = [], 100.0
    for _t in range(1500):
        _o = _cp * np.exp(0.003 * _rg.standard_normal())
        _path = _o * np.exp(np.cumsum(np.sqrt((0.01 ** 2 - 0.003 ** 2) / 200) * _rg.standard_normal(200)))
        _ohlc_rows.append((_o, max(_o, _path.max()), min(_o, _path.min()), _path[-1]))
        _cp = _path[-1]
    _ohlc = pd.DataFrame(_ohlc_rows, columns=["Open", "High", "Low", "Close"], index=pd.bdate_range("2019-01-01", periods=1500))
    _rv = range_variance(_ohlc)
    assert np.isnan(_rv.iloc[0]) and 0.8 < _rv.mean() / 1e-4 < 1.05                 # 첫날은 전날 종가가 없어 NaN
    _oh2 = _ohlc.copy()
    _oh2.iloc[10, 1] = _oh2.iloc[10, 3] * 0.99                                       # 고가 < 종가 → 무효
    _oh2.iloc[11, 2] = _oh2.iloc[11, 1]                                              # 고가 = 저가 → 무효
    assert range_variance(_oh2).iloc[10:12].isna().all() and np.isfinite(range_variance(_oh2).iloc[12])
    _ew = ewma_level(pd.Series([np.nan] + [1.0] * 19 + [2.0, np.nan, 3.0]), lam=0.5, init=19)
    assert np.isnan(_ew.iloc[18]) and _ew.iloc[19] == 1.0 and _ew.iloc[20] == 1.5 and _ew.iloc[21] == 1.5 and _ew.iloc[22] == 2.25
    assert np.allclose(range_blend_factor([0.01, 0.01, 0.01, 0.01], [1e-4, 4e-4, np.nan, 0.0]), [1, np.sqrt(2.5), 1, 1])
    # 앱·실전 기록이 쓰는 지금 값: 마지막 날의 지수가중 평균. 최근 750일 중 유효한 일봉이 80% 미만이면 NaN (→ GARCH만)
    assert np.isclose(range_level_now(_ohlc, _ohlc.index), ewma_level(_rv).iloc[-1])
    assert np.isclose(range_level_now(_ohlc, _ohlc.index[:900]), ewma_level(_rv.iloc[:900]).iloc[-1])   # 쓰는 종가의 날짜까지만
    _oh3 = _ohlc.copy()
    _oh3.iloc[-300:, 1] = _oh3.iloc[-300:, 2]                                         # 최근 300일 고가 = 저가 → 유효 60%
    assert np.isnan(range_level_now(_oh3, _oh3.index)) and np.isfinite(range_level_now(_oh3, _oh3.index, min_valid=0.5))
    # Yahoo가 잠깐 빈 일봉을 주면 다시 받음 (세 번째에 받으면 그 표, 끝까지 비면 오류)
    _yt0, _ohc = md.yf.Ticker, {"n": 0}
    class _FakeT:
        def __init__(self, t): pass
        def history(self, period, auto_adjust):
            _ohc["n"] += 1
            return _ohlc.tz_localize("Asia/Seoul") if _ohc["n"] >= 3 else pd.DataFrame()
    md.yf.Ticker = _FakeT
    try:
        assert md.load_ohlc("005930.KS", 3, wait=0).equals(_ohlc) and _ohc["n"] == 3
        _ohc["n"] = -10
        try:
            md.load_ohlc("005930.KS", 3, wait=0)
            raise AssertionError("빈 일봉인데 오류가 아님")
        except ValueError:
            pass
    finally:
        md.yf.Ticker = _yt0


def test_day1_var_factor():
    """하루 VaR 배수: 코스피 200 종목은 VKOSPI, 나머지는 장중 실현 분산. 다른 쪽 값은 쓰지 않음"""
    assert np.isclose(day1_var_factor(0.01, True, 100 * 0.02 * np.sqrt(252), 1e-4), np.sqrt(2.5))
    assert np.isclose(day1_var_factor(0.01, False, 100 * 0.02 * np.sqrt(252), 4e-4), np.sqrt(2.5))
    assert day1_var_factor(0.01, True, np.nan, 4e-4) == 1.0 and day1_var_factor(0.01, False, 40.0, np.nan) == 1.0


def test_iv_blend_factor():
    """VKOSPI 섞기: σ² = ½ GARCH + ½ (IV/100)²/252. IV가 GARCH와 같으면 1, 두 배면 √2.5, 없거나 0이면 1"""
    _ivd = 100 * 0.01 * np.sqrt(252)
    assert np.allclose(iv_blend_factor(np.full(4, 0.01), np.array([np.nan, 0.0, _ivd, 2 * _ivd])), [1, 1, 1, np.sqrt(2.5)])

    v95, es95 = var_es(np.random.default_rng(0).normal(0, 0.01, 400000), 0.95)
    assert abs(v95 - (1 - np.exp(-1.645 * 0.01))) < 3e-4 and es95 > v95
    assert max_drawdown([1.0, 1.2, 0.9, 1.3, 1.17]) == -0.25 and max_drawdown([1, 2, 3]) == 0.0


def test_gjr_garch():
    _rg = _sim_gjr(4000, 2e-6, 0.03, 0.12, 0.88, 6, np.random.default_rng(0))
    _j = GarchVolForecaster(asym=True).fit(_rg)
    assert abs(_j.gamma - 0.12) < 0.04 and abs(_j.beta - 0.88) < 0.03 and abs(_j.persistence - 0.97) < 0.02, (_j.alpha, _j.gamma, _j.beta)
    assert fit_vol_model(_rg).name == "GJR-GARCH" and fit_vol_model(_rg, "garch").name == "GARCH"      # AIC로 비대칭을 골라냄
    _sym = simulate_garch_t(4000, 2e-6, 0.08, 0.90, 6.0, np.random.default_rng(100))
    assert GarchVolForecaster(asym=True).fit(_sym).gamma < 0.02 and fit_vol_model(_sym).name == "GARCH"  # 대칭 데이터엔 γ ≈ 0
    # 다음 날 분산 = 마지막 날 충격 부호까지 반영, 시뮬레이션도 하락 뒤에 변동성이 더 큼
    _e = (_rg[-1] - _j.mu)
    _s2_last = (_j.cond_vol[-1] ** 2) / 252
    assert np.isclose(_j.sigma2_next, _j.next_var(_s2_last, _e), rtol=1e-9)
    _sim = np.diff(np.c_[np.zeros(20000), _j.simulate(2, 20000, seed=3)], axis=1)
    assert (_sim[:, 1][_sim[:, 0] < 0] ** 2).mean() > 1.1 * (_sim[:, 1][_sim[:, 0] > 0] ** 2).mean()
    assert len(var_backtest(_rg[-1500:], model="auto")) == 2
