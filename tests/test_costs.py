"""한국 매매 비용(증권거래세)과 가격제한폭·거래정지 체결 제약."""
import numpy as np
import pandas as pd

import direction as dr
import kr_costs as kc
import universe as uv
from tests.helpers import synth_panel


def test_kr_sell_tax():
    _r = kc.sell_tax_rate(pd.to_datetime(["2018-12-31", "2019-06-10", "2024-06-01", "2025-03-01", "2026-02-01"]), "KOSPI")
    assert list(_r.values) == [0.0030, 0.0025, 0.0018, 0.0015, 0.0020]
    assert (kc.sell_tax_rate(pd.to_datetime(["2024-06-01"]), "ETF") == 0).all()
    assert kc.tax_kind("^KS11") == "INDEX" and kc.tax_kind("069500.KS", "KODEX 200") == "ETF" and kc.tax_kind("005930.KS", "삼성전자") == "KOSPI"
    assert kc.tax_kind("247540.KQ") == "KOSDAQ" and kc.tax_kind("123456.KS", "아무개", "etf") == "ETF"
    _p = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0], index=pd.bdate_range("2024-06-03", periods=5))
    assert np.allclose(kc.trade_cost(_p, 10.0, 0.0018).values, [0.001, 0.0, 0.001 + 0.0018, 0.0, 0.001])   # 팔 때만 세금


def test_direction_tax_on_sells(px, wf_a):
    """방향 예측 전략: 세금은 판 날에만, 정확히 판 비중 × 세율만큼"""
    _e0 = dr.evaluate(wf_a, 21, np.log(px).diff(), 0.0, threshold=0.5, lag=1)
    _e1 = dr.evaluate(wf_a, 21, np.log(px).diff(), 0.0, threshold=0.5, lag=1, sell_tax=0.002)
    _pos = (wf_a["auto"] > 0.5).astype(float)
    _sold = (-_pos.diff().fillna(_pos.iloc[0])).clip(lower=0)
    assert np.allclose(np.log(_e0["equity"]["전략"]).values - np.log(_e1["equity"]["전략"]).values, (_sold * 0.002).cumsum().values)


def test_ranking_tax(wf_u):
    """종목 고르기: 판 비중 × 그날 코스피 세율 (첫 매수 제외)"""
    _t0, _c0 = uv.evaluate(wf_u, cost_bps=0)
    _t1, _c1 = uv.evaluate(wf_u, cost_bps=0, sell_tax=True)
    _n0 = _c0["auto"]["상위 20% (비용 후)"]; _n1 = _c1["auto"]["상위 20% (비용 후)"]
    assert _n0.iloc[0] == _n1.iloc[0] and (_n1.iloc[1:] < _n0.iloc[1:]).all()


def test_price_limit_execution():
    assert kc.limit_rate(["2015-06-12", "2015-06-15", "2026-01-02"]).tolist() == [0.15, 0.30, 0.30]
    _dl = pd.bdate_range("2026-03-02", periods=6)
    _mv = pd.Series([0.01, 0.02, -0.298, 0.02, 0.297, 0.0], index=_dl)
    _dn, _upl = kc.limit_hits(_mv)
    assert _dn.tolist() == [False, False, True, False, False, False] and _upl.tolist() == [False, False, False, False, True, False]
    # 판단일 t의 주문은 t+1 종가에 체결: 체결일이 하한가면 팔기(1→0)가, 상한가면 사기(0→1)가 하루 미뤄진다
    _tg = pd.Series([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=_dl)
    assert kc.apply_price_limits(_tg, np.log1p(_mv), lag=1).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]   # 3/3 판단 → 3/4 하한가 → 3/5에 팜
    _tg2 = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], index=_dl)
    assert kc.apply_price_limits(_tg2, np.log1p(_mv), lag=1).tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # 3/5 판단 → 3/6 상한가 → 3/9에 삼
    assert kc.apply_price_limits(_tg, np.log1p(_mv * 0), lag=1).equals(_tg)                              # 제한폭 없으면 그대로


def test_direction_price_limits(px, wf_a):
    """방향 전략: 제한폭에 걸린 날이 없으면 결과가 같고, 하한가 날엔 팔지 못한 손실까지 떠안는다"""
    _rpx = np.log(px).diff()
    assert np.allclose(dr.evaluate(wf_a, 21, _rpx)["daily"]["strat"], dr.evaluate(wf_a, 21, _rpx, price_limit=False)["daily"]["strat"])


def test_ranking_price_limits(wf_u):
    """종목 고르기: 하한가·거래정지면 못 팔아 계속 보유, 상한가·거래정지면 못 삼 (막힌 주문 수를 표에 남김)"""
    _wf_blk = dict(wf_u)
    _wf_blk["blocked"] = {"sell": pd.DataFrame(True, index=wf_u["rebal"], columns=wf_u["universe"].columns),
                          "buy": pd.DataFrame(False, index=wf_u["rebal"], columns=wf_u["universe"].columns)}
    _tb, _ = uv.evaluate(_wf_blk, cost_bps=0)
    _tb0, _ = uv.evaluate(wf_u, cost_bps=0)
    assert _tb.loc["ridge", "체결 막힌 주문"] > 0 and _tb.loc["ridge", "평균 회전율"] < _tb0.loc["ridge", "평균 회전율"]
    assert _tb0.loc["ridge", "체결 막힌 주문"] == 0                                                   # 원래 주가가 없는 합성 패널 = 제약 없음


def test_ranking_blocked_orders_handmade():
    """손으로 만든 3번의 리밸런싱 (30종목, 상위 6개, 매번 겹치지 않게 고름): 수익은 종목 번호 × 0.1%로 정해 둠"""
    _dz = pd.bdate_range("2024-01-01", periods=3)
    _cz = [f"{i:06d}" for i in range(30)]
    _sz = pd.DataFrame([[(i - 6 * k) % 30 * -1.0 for i in range(30)] for k in range(3)], index=_dz, columns=_cz)  # k번째: 6k~6k+5가 1등
    _wz = {"h": 21, "rebal": list(_dz), "universe": pd.DataFrame(True, index=_dz, columns=_cz), "exec_lag": 1,
           "fwd": pd.DataFrame([[np.log1p(0.001 * i) for i in range(30)]] * 3, index=_dz, columns=_cz),
           "scores": {"ridge": _sz}, "ic": {"ridge": pd.Series(0.1, index=_dz)}}
    _no = pd.DataFrame(False, index=_dz, columns=_cz)
    _ret = lambda names: np.mean([0.001 * int(c) for c in names])
    _wz["blocked"] = {"sell": _no, "buy": _no}                                                  # 제약 없음 → 매번 그 6개
    _tz, _cz0 = uv.evaluate(_wz, cost_bps=0)
    assert np.allclose(_cz0["ridge"]["상위 20% (비용 후)"].values, np.cumprod([1 + _ret(_cz[6 * k:6 * k + 6]) for k in range(3)]))
    _wz["blocked"] = {"sell": _no, "buy": ~_no}                                                 # 아무것도 못 삼 → 현금
    _tz, _cz1 = uv.evaluate(_wz, cost_bps=0)
    assert np.allclose(_cz1["ridge"]["상위 20% (비용 후)"].values, 1.0) and _tz.loc["ridge", "체결 막힌 주문"] == 18
    _wz["blocked"] = {"sell": ~_no, "buy": _no}                                                 # 못 팜 → 산 것이 계속 쌓임
    _tz, _cz2 = uv.evaluate(_wz, cost_bps=0)
    assert np.allclose(_cz2["ridge"]["상위 20% (비용 후)"].values,
                       np.cumprod([1 + _ret(_cz[:6]), 1 + _ret(_cz[:12]), 1 + _ret(_cz[:18])])) and _tz.loc["ridge", "체결 막힌 주문"] == 6 + 12


def test_blocked_order_table(wf_u):
    """패널에서 막힌 주문 표 만들기: 체결일 거래량 0(거래정지) → 사고팔기 모두 막힘, 하한가 → 팔기만 막힘"""
    _pa5 = synth_panel(5)                                 # wf_u를 만든 합성 패널 (pa는 위에서 다른 값으로 바뀜)
    _pr = dict(_pa5)
    _pr["close_raw"] = _pa5["close"].copy()
    _pr["volume"] = pd.DataFrame(1000.0, index=_pa5["close"].index, columns=_pa5["close"].columns)
    _d3 = wf_u["rebal"][3]
    _e3 = _pa5["close"].index[_pa5["close"].index.get_loc(_d3) + 1]
    _pr["volume"].loc[_e3, "000001"] = 0.0
    _pr["close_raw"].loc[_e3:, "000002"] *= 0.70 / (_pr["close_raw"].loc[_e3, "000002"] / _pr["close_raw"].shift(1).loc[_e3, "000002"])
    _bo = uv._blocked_orders(_pr, wf_u["rebal"], 1)
    assert _bo["sell"].at[_d3, "000001"] and _bo["buy"].at[_d3, "000001"]
    assert _bo["sell"].at[_d3, "000002"] and not _bo["buy"].at[_d3, "000002"] and not _bo["sell"].at[_d3, "000003"]


def test_tax_on_actual_sell_date():
    _dt = pd.bdate_range("2025-12-26", "2026-01-09")
    _tx = kc.sell_tax_rate(_dt, "KOSPI")                                   # 2025년 0.15% → 2026년 0.20%
    _ps = pd.Series(1.0, index=_dt)
    _ps.loc["2025-12-31":] = 0.0                                           # 12/31에 '팔자'고 판단
    _c0 = kc.trade_cost(_ps, 0.0, _tx, lag=0)                              # 그날 종가에 팔면 2025년 세율
    _c1 = kc.trade_cost(_ps, 0.0, _tx, lag=1)                              # 다음 거래일(1/1 휴장 무시, 달력상 1/1) 종가에 팔면 2026년 세율
    assert np.isclose(_c0.loc["2025-12-31"], 0.0015) and np.isclose(_c1.loc["2025-12-31"], 0.0020)
    assert np.isclose(_c1.sum(), 0.0020) and kc.trade_cost(_ps, 0.0, 0.002, lag=1).sum() == 0.002      # 숫자 세율은 그대로
