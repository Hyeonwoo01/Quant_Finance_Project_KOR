"""시장 국면(마르코프 전환) 모델."""
import numpy as np

import regime as rg


def test_regime_model(regime_data, regime_prob, regime_returns):
    _n, _s, _x, _c = (regime_data[k] for k in ("n", "s", "x", "close"))
    pa, _rr = regime_prob, regime_returns
    _rng = np.random.default_rng(8)
    m = rg.RegimeSwitchingModel().fit(_x)
    assert abs(m.sd[0] - 0.006) < 0.001 and abs(m.sd[1] - 0.02) < 0.003 and abs(m.P[0, 0] - 0.99) < 0.01, (m.sd, m.P)
    assert ((m.filter(_x) >= 0.5) == _s).mean() > 0.9
    # 필터 확률은 그날까지의 데이터만 쓴다: 뒤쪽 수익률을 바꿔도 앞쪽 확률은 그대로
    _x2 = _x.copy()
    _x2[2000:] = _rng.standard_normal(1000) * 0.05
    assert np.allclose(m.filter(_x)[:2000], m.filter(_x2)[:2000])
    # 워크포워드: T 이후 가격을 바꿔도 T까지의 확률은 그대로
    _c2 = _c.copy()
    _c2.iloc[2100:] = _c.iloc[2100] * np.exp(np.cumsum(_x2[2100:]))
    pb, _ = rg.walk_forward(_c2)
    assert pa.iloc[:756].isna().all() and np.allclose(pa.iloc[756:2099], pb.iloc[756:2099])
    assert not np.allclose(rg.evaluate(pa, _rr, lag=0)["equity"].values, rg.evaluate(pa, _rr, lag=1)["equity"].values)
    out = rg.analyze(_c, 1.0)
    assert out["table"].loc["불안 국면엔 현금", "최대 낙폭"] > out["table"].loc["단순 보유", "최대 낙폭"]   # 심어 둔 국면이면 낙폭이 줄어야
