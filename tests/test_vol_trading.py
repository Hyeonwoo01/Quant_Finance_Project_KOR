"""변동성 매매 백테스트: 신호, 현실성 옵션, IV 결합 예측, '매도만' 규칙."""
import numpy as np
import pandas as pd

from quant_engine import (summarize_vol_backtest, vol_signal_backtest, vol_backtest_robustness, add_combined_forecast, forecast_errors, short_only_split, short_only_robustness)


def test_vol_signal_backtest(vol_case):
    """변동성 신호 백테스트: IV가 실제보다 늘 10%p 비싸면 '매도' 신호가 나고 이익이어야 한다 + 미래 정보 누출 없음"""
    _idx, _px, _iv, vb = (vol_case[k] for k in ("idx", "px", "iv", "vb"))
    _rng = np.random.default_rng(22)
    vtab, vcur = summarize_vol_backtest(vb)
    assert (vb["signal"] == -1).mean() > 0.9 and vtab.loc["signal", "평균 손익(주가 대비)"] > 0
    assert vtab.loc["long", "평균 손익(주가 대비)"] < 0 and len(vcur) == len(vb)
    _px2 = _px.copy()
    _px2.iloc[1000:] *= np.exp(np.cumsum(0.05 * _rng.standard_normal(400)))      # T=1000 이후 가격만 바꿈
    vb2 = vol_signal_backtest(_px2, _iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
    done = [d for i, d in enumerate(vb.index) if _idx.get_loc(d) + 1 + 21 < 1000]  # 다음 날 진입 → 만기까지 T 이전
    assert len(done) > 10 and np.allclose(vb.loc[done].values, vb2.loc[done].values)


def test_vol_backtest_realism_options(vol_case):
    """현실성 옵션: 옵션 체결 비용 = 비율 × 스트래들 가격, IV를 낮추면 매도 이익이 줄어듦, GARCH 예측 재사용, 진입일 늦추기"""
    _idx, _px, _iv, vb = (vol_case[k] for k in ("idx", "px", "iv", "vb"))
    _c1 = vol_signal_backtest(_px, _iv, cost_bps=1.0, option_spread=0.01)
    assert np.allclose(_c1["cost"] - vb["cost"], 0.01 * vb["premium"]) and np.allclose(_c1["long"], vb["long"])
    _lo = vol_signal_backtest(_px, _iv, cost_bps=1.0, iv_gap=0.02, forecasts=vb["garch"].to_dict())
    assert np.allclose(_lo["garch"], vb["garch"]) and np.allclose(_lo["iv"], 0.28) and np.allclose(_lo["iv_index"], 0.30)
    assert (-_lo["long"]).mean() < (-vb["long"]).mean() and (_lo["premium"] < vb["premium"]).all()
    _late = vol_signal_backtest(_px, _iv, cost_bps=1.0, offset=5)
    assert _late.index[0] == _idx[505] and vb.index[0] == _idx[500]
    rb = vol_backtest_robustness(_px, _iv, vb, cost_bps=1.0, gaps=(0.0, 0.02), offsets=(0, 5))
    gtab, otab = rb["by_gap"], rb["by_offset"]
    assert list(gtab.index) == ["0%p", "2%p"] and gtab.loc["0%p", "진입 수"] == len(vb)
    assert set(rb["grid"]) == {(0.0, 0), (0.02, 0), (0.0, 5), (0.02, 5)} and rb["grid"][(0.0, 0)] is vb
    assert np.allclose(rb["grid"][(0.02, 5)]["garch"], rb["grid"][(0.0, 5)]["garch"])        # 같은 진입일이면 예측 재사용
    assert gtab.loc["2%p", "항상 매도 평균 손익(주가 대비)"] < gtab.loc["0%p", "항상 매도 평균 손익(주가 대비)"]
    assert otab.loc["5거래일 늦게", "첫 진입일"] == _idx[505].date()


def test_combined_iv_forecast(short_only_case):
    """IV 결합 예측: 과거 결과만 쓰는지 + 실제가 GARCH·IV의 섞임이면 결합이 둘 다보다 정확해야"""
    _k, _g, _bt = (short_only_case[k] for k in ("k", "g", "bt"))
    cb = add_combined_forecast(_bt)
    assert cb["combo"].iloc[:24].isna().all() and cb["combo"].iloc[24:].notna().all()
    _bt2 = _bt.copy()
    _bt2.iloc[80:, _bt2.columns.get_loc("realized")] = 9.9             # 80번째 이후 결과만 바꿈
    cb2 = add_combined_forecast(_bt2)
    assert np.allclose(cb["combo"].iloc[24:81], cb2["combo"].iloc[24:81]) and np.allclose(cb["iv_adj"].iloc[24:81], cb2["iv_adj"].iloc[24:81])
    fe, n_fe = forecast_errors(cb, "VIX")
    assert n_fe == _k - 24 and fe["예측 오차(RMSE)"].idxmin() == "GARCH + VIX 결합", fe
    assert fe.loc["GARCH + VIX 결합", "GARCH 대비 개선 t값"] > 2 and np.isnan(fe.loc["GARCH", "GARCH 대비 개선 t값"])


def test_short_only_rule_and_robustness(short_only_case):
    """'매도만' 규칙: IV가 예측보다 비쌀 때만 팔면 이익인 데이터에서 그 규칙을 골라야 하고, 뒤 기간은 선택에 안 쓰여야 한다"""
    _k, _g, _bt, _bs, so = (short_only_case[k] for k in ("k", "g", "bt", "bs", "so"))
    assert so["chosen"] != "항상 매도" and so["chosen"].endswith(("기준 -1%p", "기준 +0%p", "기준 +1%p")), so["chosen"]   # 참 기준선 ≈ 0
    assert so["test"].loc["고른 규칙", "샤프(연)"] > so["test"].loc["항상 매도", "샤프(연)"]
    _bs2 = _bs.copy()
    _half = 24 + (_k - 24) // 2
    _bs2.iloc[_half:, _bs2.columns.get_loc("long")] *= -1                # 뒤 기간 손익만 뒤집음
    assert short_only_split(add_combined_forecast(_bs2), iv_name="VIX")["chosen"] == so["chosen"]
    assert short_only_split(add_combined_forecast(_bs.iloc[:40]), iv_name="VIX") is None   # 표본 부족

    # '매도만' 견고성: 모든 조합에서 통하는 데이터면 고정 규칙이 늘 항상 매도보다 낫고, 뒤 기간을 뒤집으면 못해야 한다
    rob = short_only_robustness({(0.0, 0): _bs, (0.01, 0): _bs, (0.0, 4): _bs.iloc[2:]}, rule=so["chosen"], iv_name="VIX")
    assert len(rob) == 3 and (rob["이 규칙 샤프"] > rob["항상 매도 샤프"]).all() and (rob["다시 고른 규칙"] != "항상 매도").all()
    rob2 = short_only_robustness({(0.0, 0): _bs2}, rule=so["chosen"], iv_name="VIX")
    assert (rob2["이 규칙 샤프"] < rob2["항상 매도 샤프"]).all()
    assert "이 규칙 샤프" not in short_only_robustness({(0.0, 0): _bs}, rule=None, iv_name="VIX")


def test_vol_entry_timing(vol_case):
    """체결 시점: 기본은 판단 다음 날 종가·그날 IV로 진입, lag=0이면 판단한 날 그대로"""
    _idx, _px = vol_case["idx"], vol_case["px"]
    _ivw = pd.Series(0.2 + 0.1 * np.sin(np.arange(1400) / 30), index=_idx)
    _l1 = vol_signal_backtest(_px, _ivw)
    _l0 = vol_signal_backtest(_px, _ivw, lag=0, daily_iv_hedge=False)
    assert np.allclose(_l1["iv_entry"].values, _ivw.shift(-1).reindex(_l1.index).values) and np.allclose(_l1["iv"], _ivw.reindex(_l1.index))
    assert np.allclose(_l0["iv_entry"], _l0["iv"]) and (_l1.index == _l0.index[:len(_l1)]).all()
