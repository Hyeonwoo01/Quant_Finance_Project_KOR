"""옵션 가격·그릭스·델타 헤지 손익."""
import numpy as np

from quant_engine import (GarchVolForecaster, GammaScalpingEngine, RiskNeutralPricingEngine, VolatilityTradingLogic, _straddle_delta, simulate_gbm_paths, GammaScalpingEngine as _GSE, hedged_straddle_pnl, straddle_price)
from tests.helpers import simulate_garch


def test_pricing_and_implied_vol():
    """프라이싱 / IV"""
    engine = RiskNeutralPricingEngine(100.0, 100.0, 1.0, 0.05, 'call')
    put = RiskNeutralPricingEngine(100.0, 100.0, 1.0, 0.05, 'put')
    assert abs(engine.price(0.20) - 10.45) < 0.01
    assert abs(engine.implied_volatility(12.00) - 0.2411) < 1e-3
    assert abs(engine.price(0.2) - put.price(0.2) - (100 - 100 * np.exp(-0.05))) < 1e-10
    assert np.isnan(engine.implied_volatility(101.0)) and np.isnan(engine.implied_volatility(0.5))
    for K in (70, 100, 150):
        for T in (0.05, 0.5, 2.0):
            for sg in (0.08, 0.25, 0.9):
                for typ in ('call', 'put'):
                    e = RiskNeutralPricingEngine(100.0, float(K), T, 0.03, typ)
                    if e.vega(sg) >= 1e-3:
                        assert abs(e.implied_volatility(e.price(sg)) - sg) < 1e-6


def test_garch_parameter_recovery():
    """GARCH 파라미터 복원"""
    g = GarchVolForecaster().fit(simulate_garch(6000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
    assert abs(g.persistence - 0.98) < 0.03


def test_vol_trading_signal():
    """신호"""
    logic = VolatilityTradingLogic(vrp=0.028, threshold=0.02)
    assert logic.generate_signal(0.20, 0.23) == 0
    assert logic.generate_signal(0.15, 0.27) == -1
    assert logic.generate_signal(0.28, 0.22) == 1


def test_greeks():
    """그릭스"""
    e0 = GammaScalpingEngine(100.0, 100.0, 1.0, 0.05, 0.20)
    h = 1e-5
    v_hi = GammaScalpingEngine(100, 100, 1 + h, 0.05, 0.2).straddle_greeks()["price"]
    v_lo = GammaScalpingEngine(100, 100, 1 - h, 0.05, 0.2).straddle_greeks()["price"]
    assert abs(e0.straddle_greeks()["theta_per_day"] * 252 + (v_hi - v_lo) / (2 * h)) < 1e-6
    assert abs(_straddle_delta(100, 100, 1.0, 0.05, 0.2) - e0.straddle_greeks()["delta"]) < 1e-12


def test_vectorized_hedge_pnl_matches_paths():
    """벡터화된 헤지 손익 == 경로 하나씩 계산한 값, 그리고 부호가 이론대로인지"""
    paths = simulate_gbm_paths(100, 0.30, 0.03, 0.25, 63, 50, np.random.default_rng(1))
    sc = GammaScalpingEngine(100, 100, 0.25, 0.03, 0.20)
    vec = sc.hedged_pnl(paths, position=1)
    assert np.allclose(vec, [sc.hedged_pnl(p, position=1) for p in paths])
    assert vec.mean() > 0                                      # 실현 30% > IV 20% → 롱 감마 이익
    assert sc.hedged_pnl(paths, position=-1).mean() < 0


def test_short_hedge_pnl_is_negative_long_minus_cost():
    """헤지 손익: 매도 = −(매수 손익) − 거래비용 (비용은 방향과 무관)"""
    _e = _GSE(100, 100, 21 / 252, 0.02, 0.2, 0.01)
    _p = simulate_gbm_paths(100, 0.25, 0.02, 21 / 252, 21, 1, np.random.default_rng(5))[0]
    _g, _n = _e.hedged_pnl(_p, 1, 0.0), _e.hedged_pnl(_p, 1, 5.0)
    assert abs(_e.hedged_pnl(_p, -1, 5.0) - (-_g - (_g - _n))) < 1e-10


def test_vectorized_hedge_multiple_entries():
    """여러 진입을 한 번에 계산하는 헤지 손익 == 진입 하나씩 엔진으로 계산한 값 (만기·IV가 진입마다 달라도)"""
    _rng = np.random.default_rng(8)
    _P = simulate_gbm_paths(100, 0.3, 0.02, 30 / 365, 21, 6, _rng) * _rng.uniform(0.5, 2.0, (6, 1))
    _T, _S = _rng.uniform(25, 35, 6) / 365, _rng.uniform(0.1, 0.5, 6)
    for c in (0.0, 3.0):
        _one = [GammaScalpingEngine(p[0], p[0], t, 0.02, sg, 0.01).hedged_pnl(p, 1, c) for p, t, sg in zip(_P, _T, _S)]
        assert np.allclose(hedged_straddle_pnl(_P, _T, 0.02, _S, 0.01, c), _one)
    assert np.allclose(hedged_straddle_pnl(_P, _T, 0.02, _S, 0.01, 3.0, np.repeat(_S[:, None], _P.shape[1], axis=1)),
                       hedged_straddle_pnl(_P, _T, 0.02, _S, 0.01, 3.0))          # 매일 IV가 같으면 예전 방식과 동일
    assert np.allclose(straddle_price(_P[:, 0], _P[:, 0], _T, 0.02, _S, 0.01),
                       [GammaScalpingEngine(p[0], p[0], t, 0.02, sg, 0.01).straddle_greeks()["price"] for p, t, sg in zip(_P, _T, _S)])
