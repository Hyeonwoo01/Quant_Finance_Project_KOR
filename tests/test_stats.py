"""통계: 겹침 보정 t값, 요인 회귀·과적합 보정(Deflated Sharpe)·타이밍 분해, Newey-West, 블록 부트스트랩."""
import numpy as np
import pandas as pd

import direction as dr
import market_data as md
import quant_engine as qe
import regime as rg
import universe as uv
from quant_engine import overlap_tstat, deflated_sharpe, market_exposure, ols, timing_attribution
from tests.helpers import _garch_t


def test_overlap_tstat():
    """겹침 보정 t값: 겹침이 없으면(h=1) 보통 t값과 같고, 21일씩 겹친 합이면 크게 줄어야 한다"""
    _g = np.random.default_rng(3)
    _iid = _g.standard_normal(5000) + 0.05
    assert abs(overlap_tstat(_iid, 1) - _iid.mean() / _iid.std() * np.sqrt(5000)) < 1e-9
    _ov = np.convolve(_g.standard_normal(5020) + 0.05, np.ones(21), "valid")
    assert abs(overlap_tstat(_ov, 21)) < abs(_ov.mean() / _ov.std() * np.sqrt(len(_ov))) / 2
    assert overlap_tstat(np.zeros(100), 5) == 0.0
    # 효과가 없을 때 '우연인데 유의(|t| > 1.96)' 비율이 목표 5% 근처여야 한다 (Newey-West 시차 20이면 약 11%)
    _rej = np.mean([abs(overlap_tstat(np.convolve(_g.standard_normal(2520), np.ones(21), "valid")[:2500], 21)) > 1.96
                    for _ in range(1000)])
    assert 0.025 < _rej < 0.08, _rej


def test_overlap_tstat_realistic():
    """같은 검정을 실제와 비슷한 조건에서: '63일 모멘텀이 양수면 오른다'(예측력 없음)를 21일 결과로 채점한 적중 차이. 진짜 차이는 0인데 '유의'가 나오는 비율이 5% 근처여야 하고, 겹침을 무시한 보통 t값은 크게 과대평가해야 한다"""
    _nd, _R = 1700, 500
    _cg = np.vstack([np.zeros(_R), np.cumsum(_garch_t(_nd + 85, _R, np.random.default_rng(7)), 0)])
    _up = (_cg[63 + 21:63 + 21 + _nd] - _cg[63:63 + _nd] > 0).astype(float)
    _d = ((_cg[63:63 + _nd] - _cg[:_nd] > 0) == _up).astype(float) - _up
    _rej_hh = np.mean([abs(overlap_tstat(_d[:, i], 21)) > 1.96 for i in range(_R)])
    _rej_naive = np.mean([abs(_d[:, i].mean() / _d[:, i].std() * np.sqrt(_nd)) > 1.96 for i in range(_R)])
    assert 0.025 < _rej_hh < 0.10 and _rej_naive > 0.4, (_rej_hh, _rej_naive)


def test_ols_deflated_sharpe_timing_attribution():
    _g = np.random.default_rng(42)
    _X = pd.DataFrame(_g.standard_normal((3000, 2)), columns=["a", "b"])
    _o = ols(0.01 + 0.5 * _X["a"] - 0.2 * _X["b"] + 0.1 * _g.standard_normal(3000), _X)
    assert np.allclose(_o["coef"].values, [0.01, 0.5, -0.2], atol=0.01) and _o["t"]["a"] > 100 and 0.9 < _o["r2"] < 1
    assert np.isnan(ols(np.ones(3), _X.iloc[:3])["r2"])                       # 표본이 변수보다 적으면 계산하지 않음
    # Deflated Sharpe: 효과 없는 전략 12개 중 1등은 보정 없이는 자주 '유의', 보정하면 거의 안 됨
    _hd = _hp = 0
    for _ in range(300):
        _rt = _g.standard_normal((12, 500))
        _sr = _rt.mean(1) / _rt.std(1, ddof=1)
        _b = int(np.argmax(_sr))
        _hd += deflated_sharpe(_rt[_b], _sr)["dsr"] > 0.95
        _hp += deflated_sharpe(_rt[_b], [_sr[_b]])["dsr"] > 0.95
    assert _hd / 300 < 0.06 and _hp / 300 > 0.3, (_hd, _hp)
    assert deflated_sharpe(0.3 + _g.standard_normal(500), [0.3, 0.0, 0.05])["dsr"] > 0.99      # 진짜 효과는 살아남음
    assert deflated_sharpe(np.zeros(50), [0.1]) is None
    # 자기상관이 있으면 샤프가 더 흔들린다 → 표본 길이를 줄여 센다. 평균 0, 자기상관 0.4인 전략 하나를 그대로 믿으면
    # '진짜일 확률 > 97.5%'가 목표 2.5% 대신 약 10% 나오는데, 보정하면 약 4%. 음의 자기상관으로 표본을 늘리지는 않음
    _ea4 = _g.standard_normal((300, 550))
    _x4 = np.zeros_like(_ea4)
    for _t in range(1, 550):
        _x4[:, _t] = 0.4 * _x4[:, _t - 1] + _ea4[:, _t]
    _x4 = _x4[:, 50:]
    _d4 = [deflated_sharpe(_x4[i], [_x4[i].mean() / _x4[i].std(ddof=1)]) for i in range(300)]
    assert np.mean([d["dsr"] > 0.975 for d in _d4]) < 0.06 and np.mean([d["T_eff"] / d["T"] for d in _d4]) < 0.6
    assert all(d["T_eff"] < 0.9 * d["T"] for d in _d4)
    assert deflated_sharpe(_x4[0] * (-1) ** np.arange(500), [0.0, 0.1])["T_eff"] == 500

    # 타이밍 분해: 미래와 무관한 보유/현금은 알파 ≈ 0·β ≈ 평균 비중, 다음 날을 아는 보유는 알파가 매우 큼
    _di = pd.bdate_range("2015-01-01", periods=2500)
    _ar = pd.Series(0.0003 + 0.01 * _g.standard_normal(2500), index=_di)
    _ps = pd.Series((_g.random(2500) < 0.6).astype(float), index=_di)
    _ta = timing_attribution(pd.DataFrame({"pos": _ps, "strat": _ps * _ar, "asset": _ar}))["summary"].iloc[0]
    assert abs(_ta["알파 t값"]) < 3 and abs(_ta["노출(β)"] - 0.6) < 0.05, _ta
    _pf = (_ar > 0).astype(float)
    assert timing_attribution(pd.DataFrame({"pos": _pf, "strat": _pf * _ar, "asset": _ar}))["summary"].iloc[0]["알파 t값"] > 10
    _fff = pd.DataFrame({"Mkt-RF": np.expm1(_ar) - 0.0001, "SMB": 0.005 * _g.standard_normal(2500), "RF": 0.0001}, index=_di)
    _tf = timing_attribution(pd.DataFrame({"pos": _ps, "strat": _ps * _ar, "asset": _ar}), _fff)
    assert abs(_tf["summary"].loc["단순 보유 (요인 모형)", "노출(β)"] - 1) < 0.01 and "소형주" in _tf["loadings"].index
    assert "알파" not in _tf["loadings"].index


def test_french_data_format():
    """French 데이터 형식 (설명 줄, %, -99.99 결측, 뒤쪽 저작권 줄)"""
    _ff = md.parse_french_csv("설명\n\n,Mkt-RF,SMB,RF\n20240102,  1.50, -99.99, 0.02\n20240103, -0.50, 0.10, 0.02\n\nCopyright\n")
    assert list(_ff.columns) == ["Mkt-RF", "SMB", "RF"] and _ff.loc["2024-01-02", "Mkt-RF"] == 0.015
    assert np.isnan(_ff.loc["2024-01-02", "SMB"]) and len(_ff) == 2


def test_daily_table_dates_are_realized_dates(px, wf_a, regime_prob, regime_returns):
    """일간 표의 날짜 = 수익이 실제로 실현된 날 (판단일 + 1 + lag)"""
    pa, _rr = regime_prob, regime_returns
    _rp = np.log(px).diff()
    for _lag in (0, 1):
        _d = dr.evaluate(wf_a, 21, _rp, lag=_lag)["daily"]
        _i0 = _rp.index.get_loc(wf_a.index[0])
        assert _d.index[0] == _rp.index[_i0 + 1 + _lag] and np.isclose(_d["asset"].iloc[0], _rp.iloc[_i0 + 1 + _lag])
    _dg = rg.evaluate(pa, _rr, lag=1)["daily"]
    assert np.allclose(_dg["asset"].values, _rr.reindex(_dg.index).values)


def test_ranking_factor_exposure(px, wf_a, wf_u):
    """종목 고르기: '모멘텀 단일' 모델의 초과수익은 모멘텀 요인으로 설명되고, 진짜 신호(외국인 수급)를 쓴 릿지는 알파가 남는다"""
    _tu, _cu = uv.evaluate(wf_u, cost_bps=0)
    _fm, _fr = uv.factor_exposure(wf_u, _cu, "momentum"), uv.factor_exposure(wf_u, _cu, "ridge")
    assert abs(_fm["alpha_t"]) < 2 and _fm["table"].loc["모멘텀 (12-1개월)", "t값"] > 2, _fm
    assert _fr["alpha_t"] > 2, _fr
    _sc = uv.selection_check(_cu, 21)
    assert _sc["best"] in _cu and 0 <= _sc["dsr"] <= 1 and _sc["n_trials"] == len(_cu) - 1
    _sel = dr.robustness(wf_a, 21, np.log(px).diff())["selection"]
    assert _sel is None or (0 <= _sel["dsr"] <= 1 and _sel["grid"].shape == (4, 3))


def test_short_vol_market_exposure(short_only_case):
    """변동성 매도의 시장 노출: 큰 움직임에 돈을 버는 롱 스트래들 → 매도는 하락 상위 10% 달에 손실이 몰려야"""
    _g = np.random.default_rng(43)
    so = short_only_case["so"]
    _u = pd.Series(0.05 * _g.standard_normal(300))
    _vb = pd.DataFrame({"underlying": _u, "long": 4 * _u ** 2 - 0.008, "cost": 0.0, "signal": 0})
    _me = market_exposure(_vb)
    assert _me.loc["항상 변동성 매도", "하락 상위 10% 달 평균 손익"] < _me.loc["항상 변동성 매도", "나머지 달 평균 손익"]
    assert so["dsr"] is None or (0 <= so["dsr"]["dsr"] <= 1 and so["dsr"]["n_trials"] <= len(so["candidates"]))


def test_newey_west_tstat():
    _gh = np.random.default_rng(4)
    _xi = 0.03 + _gh.standard_normal(5000)
    assert abs(qe.hac_tstat(_xi) / (_xi.mean() / _xi.std() * np.sqrt(5000)) - 1) < 0.05          # 독립이면 보통 t값과 거의 같음
    _ea = _gh.standard_normal(2050)
    _xa = np.zeros(2050)
    for _t in range(1, 2050):
        _xa[_t] = 0.5 * _xa[_t - 1] + _ea[_t]
    _xa = _xa[50:] + 0.05
    assert 1.3 < (_xa.mean() / _xa.std() * np.sqrt(2000)) / qe.hac_tstat(_xa) < 1.9      # 자기상관 0.5 → 보통 t값이 크게 부풀려짐
    assert qe.hac_lags(60) == 3 and np.isnan(qe.hac_tstat([1.0, 2.0])) and np.isnan(qe.hac_tstat(np.ones(10)))


def test_block_bootstrap():
    _gb = np.random.default_rng(0)
    _x = 0.001 + 0.01 * _gb.standard_normal(2000)
    _lo, _hi = qe.block_bootstrap_ci(_x, 1)
    assert _lo < 0.001 < _hi and abs((_hi - _lo) / (2 * 1.96 * 0.01 / np.sqrt(2000)) - 1) < 0.15      # 독립이면 보통 공식과 같은 폭
    _e = _gb.standard_normal(5000)
    _ar = np.zeros(5000)
    for _t in range(1, 5000):
        _ar[_t] = 0.8 * _ar[_t - 1] + _e[_t]
    _w1 = np.subtract(*qe.block_bootstrap_ci(_ar, 1)[::-1])
    _w50 = np.subtract(*qe.block_bootstrap_ci(_ar, 50)[::-1])
    assert 2.3 < _w50 / _w1 < 3.7                     # 자기상관 0.8이면 이론상 √((1+0.8)/(1−0.8)) = 3배 넓어야
    assert np.isnan(qe.block_bootstrap_ci(_x[:50], 42)[0])                # 표본이 블록 2배보다 짧으면 계산하지 않음


def test_block_bootstrap_realistic():
    """실제와 비슷한 조건(GARCH·t(5))에서 '63일 모멘텀이 양수면 보유, 다음 날 체결' 전략의 일간 초과수익(진짜 평균 0): 검증 보고서와 같은 b = 42의 95% 구간이 실제로 0을 95% 가까이 포함해야 한다"""
    _nb = 1700
    _rg = _garch_t(_nb + 70, 200, np.random.default_rng(5))
    _cc = np.vstack([np.zeros(200), np.cumsum(_rg, 0)])
    _exr = ((_cc[63:63 + _nb] - _cc[:_nb] > 0) - 1.0) * _rg[64:64 + _nb]
    _cov = np.mean([(lambda lo, hi: lo <= 0 <= hi)(*qe.block_bootstrap_ci(_exr[:, i], 42, n_boot=500)) for i in range(200)])
    assert _cov > 0.9, _cov
