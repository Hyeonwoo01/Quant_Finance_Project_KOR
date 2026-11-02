"""코스피 종목 고르기: 신호 찾기, 미래 정보 차단, 비용·리밸런싱 날짜·위험 비중, 운용 규모, 매주 계산과 추천 기록."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import universe as uv
from tests.helpers import synth_panel


def test_ranking_finds_signal_without_lookahead():
    """종목 고르기: 심어 둔 '종목 간' 신호를 찾는지 + T 이후 가격을 바꿔도 T까지의 점수가 그대로인지"""



def test_target_rank_missing_future():
    """정답 순위: 미래 수익률이 없는 종목은 중간값(0)이 아니라 NaN → 학습에서 빠짐 (특징 순위는 결측 = 중간값 그대로)"""
    _fw = pd.DataFrame([[0.1, np.nan, -0.1, 0.0]], columns=list("abcd"), index=[pd.Timestamp("2024-01-02")])
    _uv = pd.DataFrame([[True, True, True, False]], columns=list("abcd"), index=_fw.index)
    _tr = uv.target_rank(_fw, _uv).iloc[0]
    assert np.allclose(_tr[["a", "c"]].values, [0.5, 0.0]) and np.isnan(_tr["b"]) and np.isnan(_tr["d"])   # a·c 둘 중 순위
    assert uv.cross_rank(_fw, _uv).iloc[0]["b"] == 0.0


def test_fetch_candidates_dedupes_pages():
    """시총 순위 목록: 쪽을 넘기는 사이 순위가 바뀌어(시간외 거래 등) 같은 종목이 두 쪽에 나와도 한 번만 — 처음(높은 순위) 것"""
    _st = lambda c, nm, kind="stock": {"itemCode": c, "stockName": nm, "stockEndType": kind, "marketValueRaw": "1000", "closePriceRaw": "10"}
    _pages = {1: [_st("000010", "가"), _st("000020", "나")],
              2: [_st("000020", "나"), _st("000030", "다"), _st("000040", "라ETF", "etf"), _st("000050", "마우")]}
    _get0 = uv.requests.get
    uv.requests.get = lambda url, **kw: __import__("types").SimpleNamespace(json=lambda: {"stocks": _pages.get(int(url.split("page=")[1].split("&")[0]), [])})
    try:
        _fc = uv.fetch_candidates()
    finally:
        uv.requests.get = _get0
    assert list(_fc["code"]) == ["000010", "000020", "000030"] and list(_fc["cap_rank"]) == [1, 2, 3] and np.allclose(_fc["shares"], 100)


def test_ranking_signal_table(tab):
    """심어 둔 종목 간 신호를 릿지와 고정 가중 점수가 찾아낸다"""
    assert tab.loc["ridge", "평균 IC"] > 0.1 and tab.loc["ridge", "상위20% 초과수익(연, 비용후)"] > 0, tab
    assert tab.loc["composite", "평균 IC"] > 0.05            # 고정 가중도 (외국인 수급 + 방향) 신호를 반영


def test_cost_sensitivity(wf_u):
    """비용 민감도: 비용이 늘면 초과수익이 줄고, 손익분기 비용에서 정확히 0 (비용은 초과수익에 선형)"""
    _cs, _be = uv.cost_sensitivity(wf_u)
    assert _cs["초과수익(연, 비용후)"].is_monotonic_decreasing and _be > 0
    assert abs(uv.evaluate(wf_u, _be, models=["auto"])[0].loc["auto", "상위20% 초과수익(연, 비용후)"]) < 1e-9


def test_drop_recent(wf_u):
    """최근 구간을 뺀 성적: 보유 기간이 그 전에 끝난 리밸런싱만, 그 기간들의 초과수익 평균과 같다"""
    _cv0 = uv.evaluate(wf_u, 5)[1]
    _dr = uv.drop_recent(wf_u, _cv0, days=150)
    _exu = (uv._period_returns(_cv0["auto"]["상위 20% (비용 후)"]) - uv._period_returns(_cv0["_universe"])).dropna()
    _ixu = wf_u["universe"].index
    _kp = [d for d in _exu.index if _ixu.get_loc(d) + 21 + 1 <= len(_ixu) - 1 - 150]
    assert _dr["리밸런싱 수"] == len(_kp) < len(_exu) and np.isclose(_dr["초과수익(연)"], _exu[_kp].mean() * 252 / 21)


def test_market_breakdown(wf_u, tab):
    """시장 구간별: 기간을 범위 평균 수익률 3분위로 나눔 (모든 기간이 한 번씩)"""
    _mb = uv.market_breakdown(uv.evaluate(wf_u, 5)[1])
    assert list(_mb.index) == ["하락", "횡보", "상승"] and _mb["기간 수"].sum() == tab.loc["auto", "리밸런싱 수"]
    assert _mb["시장 평균(기간)"].is_monotonic_increasing


def test_rebalance_offsets(panel, wf_u):
    """리밸런싱 날짜 바꿔 보기: 시작일만 o거래일 늦고 간격은 그대로 h (마지막 = 오늘의 추천은 같음). 옮김 0 = 기본 계산"""
    assert uv.offsets_for(21) == [0, 4, 8, 13, 17] and uv.offsets_for(63) == [0, 13, 25, 38, 50]
    _wfo = uv.walk_forward(panel, h=21, train_years=1, refit_models=("ridge",), offset=8)
    _ixa = panel["close"].index
    assert _ixa.get_loc(_wfo["rebal"][0]) == _ixa.get_loc(wf_u["rebal"][0]) + 8
    assert all(_ixa.get_loc(b) - _ixa.get_loc(a) == 21 for a, b in zip(_wfo["rebal"][:-2], _wfo["rebal"][1:-1]))
    assert _wfo["rebal"][-1] == wf_u["rebal"][-1] == _ixa[-1]
    _ot = uv.offset_table({8: _wfo, 0: wf_u}, 5)
    assert list(_ot["table"].index) == [0, 8] and set(_ot["summary"]) == {"auto", "momentum"}
    assert np.isclose(_ot["table"].loc[0, "auto 초과수익(연)"], uv.evaluate(wf_u, 5)[0].loc["auto", "상위20% 초과수익(연, 비용후)"])
    assert np.isclose(_ot["table"].loc[8, "momentum t값"], uv.evaluate(_wfo, 5)[0].loc["momentum", "초과수익 t값"])
    _so = _ot["summary"]["auto"]
    assert _so["시작일 수"] == 2 and _so["최소"] <= _so["평균"] <= _so["최대"] and 0 <= _so["0보다 큰 수"] <= 2
    assert np.isclose(_so["평균"], _ot["table"]["auto 초과수익(연)"].mean())


def test_risk_weights(wf_u):
    """위험 반영 비중: 변동성 역수에 비례, 종목당 동일 비중의 2배 상한, 합 1. 변동성 자료가 없으면 동일 비중"""
    _wv = {"vol": pd.DataFrame([[0.01, 0.02, 0.04, 0.08]], columns=list("abcd"), index=[0])}
    _ww = uv._weights(list("abcd"), 0, _wv, "inv_vol")
    assert np.isclose(_ww.sum(), 1) and _ww.max() <= 0.5 + 1e-12 and list(_ww.sort_values(ascending=False).index) == list("abcd")
    assert np.allclose(uv._weights(list("abcd"), 0, {"vol": None}, "inv_vol"), 0.25)
    _wfv = dict(wf_u)
    _wfv["vol"] = pd.DataFrame(np.random.default_rng(0).uniform(0.01, 0.04, (len(wf_u["rebal"]), 80)),
                               index=wf_u["rebal"], columns=wf_u["universe"].columns)
    assert uv.evaluate(_wfv, 5, weighting="inv_vol", models=["auto"])[0].shape[0] == 1
    assert np.isclose(uv.evaluate(wf_u, 5, weighting="inv_vol")[0].loc["auto", "상위20% 초과수익(연, 비용후)"],   # 변동성 자료 없음
                      uv.evaluate(wf_u, 5)[0].loc["auto", "상위20% 초과수익(연, 비용후)"])                          # → 동일 비중과 같음


def test_baseline_strategies(panel, wf_u):
    """기준 전략(저변동성 단일 등)은 같은 조건으로 표에 나오지만 자동 선택 후보는 아니다"""
    _pv = synth_panel(6)
    _pv["feat"]["vol_63"] = pd.DataFrame(np.random.default_rng(7).uniform(0.01, 0.04, _pv["close"].shape),
                                         index=_pv["close"].index, columns=_pv["close"].columns)
    _pv["coverage"]["vol_63"] = 1.0
    _wfb = uv.walk_forward(_pv, h=21, train_years=1, refit_models=("ridge",))
    assert "lowvol" in _wfb["scores"] and "value" not in _wfb["scores"] and not _wfb["chosen"].isin(list(uv.BASELINES)).any()
    assert "lowvol" in uv.evaluate(_wfb, 5)[0].index
    Tcut = 700                                                # 첫 리밸런싱(512일) 이후 리밸런싱 9회가 비교됨
    pb = synth_panel(5, shock_after=Tcut)
    wf_b = uv.walk_forward(pb, h=21, train_years=1, refit_models=("ridge",))
    cut = panel["close"].index[Tcut]
    for m in ("ridge", "composite", "auto"):
        sa, sb = wf_u["scores"][m], wf_b["scores"][m]
        rows = sa.index[sa.index <= cut]
        assert len(rows) > 5 and np.allclose(sa.loc[rows].values, sb.loc[rows].values, equal_nan=True), m


def test_execution_lag_scoring_and_training_cutoff(panel, wf_u):
    """체결 시점: 채점 수익률은 '다음 날 종가부터 h일'(기본), exec_lag=0이면 '그날 종가부터'. 점수(판단)는 같아야 한다"""
    wf_u0 = uv.walk_forward(panel, h=21, train_years=1, refit_models=("ridge",), exec_lag=0)
    _cl, _d0 = panel["close"], wf_u["rebal"][3]
    _i0 = _cl.index.get_loc(_d0)
    assert np.allclose(wf_u0["fwd"].loc[_d0], np.log(_cl.iloc[_i0 + 21] / _cl.iloc[_i0]))
    assert np.allclose(wf_u["fwd"].loc[_d0], np.log(_cl.iloc[_i0 + 22] / _cl.iloc[_i0 + 1]))
    assert wf_u["exec_lag"] == 1 and np.allclose(wf_u["scores"]["composite"].values, wf_u0["scores"]["composite"].values, equal_nan=True)
    # 학습 목표도 체결 이후 h일 수익률이라 학습 모델 점수는 체결 시점에 따라 달라진다. 그만큼 학습 데이터는 하루 더 일찍 끊겨야 한다:
    # 리밸런싱 날 d 바로 다음 날부터 가격을 바꿔도 d의 점수는 그대로 (끊는 시점이 t−h면 목표가 d+1 종가를 써서 달라짐)
    _rdc = wf_u["rebal"][-3]
    _pc = synth_panel(5, shock_after=panel["close"].index.get_loc(_rdc))
    _wfc = uv.walk_forward(_pc, h=21, train_years=1, refit_models=("ridge",), sample_every=1)      # 모든 날로 학습 (경계 날 포함)
    _wfa = uv.walk_forward(panel, h=21, train_years=1, refit_models=("ridge",), sample_every=1)
    assert np.allclose(_wfc["scores"]["ridge"].loc[_rdc].values, _wfa["scores"]["ridge"].loc[_rdc].values, equal_nan=True)
    assert not np.allclose(wf_u["scores"]["ridge"].values, wf_u0["scores"]["ridge"].values, equal_nan=True)


def test_capacity_and_basket_risk(wf_u, capacity_panel):
    _pc = capacity_panel
    _tq, _cq = uv.evaluate(wf_u, cost_bps=0)
    _hold, _ = uv._holdings(wf_u, "auto")
    _gross = pd.Series({d: (np.exp(wf_u["fwd"].loc[d, c]) - 1).mean() for d, c in _hold.items()})
    _curve = _cq["auto"]["상위 20% (비용 후)"]
    assert np.allclose((_curve / _curve.shift(1).fillna(1.0) - 1).values, _gross.values)      # evaluate와 같은 보유 종목
    _cap = uv.capacity(wf_u, _pc, _cq, capitals=(1e7, 1e9, 1e11, 1e13))
    assert _cap["시장 충격 비용(연)"].is_monotonic_increasing and _cap["초과수익(연, 모든 비용 후)"].is_monotonic_decreasing
    assert _cap["시장 충격 비용(연)"].iloc[0] < 0.001 and _cap["시장 충격 비용(연)"].iloc[-1] > 0.05     # 작으면 거의 0, 크면 급격히
    _bk = uv.current_basket(wf_u)
    _rk = uv.basket_risk(_pc, _bk)
    assert len(_bk) == len(wf_u["scores"]["auto"].iloc[-1].dropna()) // 5 or len(_bk) == len(wf_u["scores"]["composite"].iloc[-1].dropna()) // 5
    assert 0 < _rk["var95"] < _rk["var99"] and _rk["worst_day"][1] <= -_rk["var95"] and len(_rk["stress"]) == 5
    assert 0.5 < _rk["beta"] < 1.5                                     # 종목 평균 비슷한 바구니 → 시장 베타 1 근처


def test_weekly_results_bundle(wf_u, capacity_panel):
    _pc = capacity_panel
    _res = uv.results(_pc, wf_u, cost_bps=5.0, sell_tax=True)
    assert _res["table"].equals(uv.evaluate(wf_u, 5.0, sell_tax=True)[0]) and _res["pick"] == wf_u["chosen"].iloc[-1]
    assert _res["risk"] is not None and _res["capacity"] is not None and _res["n_univ"] == int(_pc["universe"].iloc[-1].sum())


def test_weekly_drops_unfinished_bar():
    """장중에 돌리면 '오늘' 봉은 빼고 계산·기록 (한국 15:30 + 여유 40분 전), 마감 뒤·다음 날엔 그대로"""
    _ix = pd.DatetimeIndex(["2026-10-22", "2026-10-23"])
    assert uv.finished_end(_ix, pd.Timestamp("2026-10-23T03:00Z")) == _ix[0]        # 한국 낮 12시
    assert uv.finished_end(_ix, pd.Timestamp("2026-10-23T08:00Z")) == _ix[1]        # 한국 오후 5시
    assert uv.finished_end(_ix, pd.Timestamp("2026-10-24T01:00Z")) == _ix[1]        # 다음 날


def test_picks_record_and_scoring():
    """기록 채점: 기록일 다음 거래일 종가에 사서 h거래일 뒤까지, 바구니 vs 그날 종목 범위 전체 (동일 비중)"""
    _pw = synth_panel(7, n_day=1300)
    _pw["feat"]["liq"] = pd.DataFrame(np.log(1e11), index=_pw["close"].index, columns=_pw["close"].columns)
    _cw, _uw = _pw["close"], _pw["universe"]
    _lg = pd.DataFrame([{"date": str(_cw.index[300].date()), "model": "ridge", "n": 2, "codes": "000001 000002",
                         "names": "가, 나", "version": ""}]).reindex(columns=uv.PICK_COLUMNS)
    _sg = uv.score_picks(_lg, _pw)
    _g21 = _cw.iloc[301 + 21] / _cw.iloc[301] - 1
    assert np.isclose(_sg["basket_21"].iloc[0], _g21[["000001", "000002"]].mean())
    assert np.isclose(_sg["market_21"].iloc[0], _g21[_uw.columns[_uw.iloc[300].values]].mean())
    _late = _lg.assign(date=str(_cw.index[-30].date()))                              # 63거래일이 아직 안 지난 기록
    _sl = uv.score_picks(_late, _pw)
    assert np.isfinite(_sl["basket_21"].iloc[0]) and np.isnan(_sl["basket_63"].iloc[0])
    _fake = pd.DataFrame({"basket_21": [0.03, 0.01, -0.01, 0.02], "market_21": [0.01] * 4,
                          "basket_63": [np.nan] * 4, "market_63": [np.nan] * 4})
    _sm = uv.picks_summary(_fake)
    assert _sm.loc["21거래일", "채점 수"] == 4 and np.isclose(_sm.loc["21거래일", "평균 초과수익"], 0.0025)
    assert np.isclose(_sm.loc["21거래일", "전체 평균보다 나은 비율"], 0.5) and _sm.loc["63거래일", "채점 수"] == 0
    # 매주 계산: 앱 설정 조합 8개를 저장하고, 이번 추천을 한 줄 기록. 같은 날 다시 돌려도 기록은 그대로 (미리 적은 기록은 고치지 않음)
    with tempfile.TemporaryDirectory() as _td:
        _wres, _wlog = uv.weekly(panel=_pw, out_dir=_td, refit_models=("ridge",), progress=lambda p, m: None)
        assert set(_wres) == {(h, l, t) for h in uv.WEEKLY_HOLDS for l in (1, 0) for t in (True, False)}
        _saved = uv.load_weekly(Path(_td) / "results.pkl")
        assert _saved["results"][(63, 1, True)]["table"].equals(_wres[(63, 1, True)]["table"])
        assert _saved["data_end"] == _cw.index[-1] and _saved["cost_bps"] == uv.WEEKLY_COST
        _rd = uv.read_picks(Path(_td) / "picks.csv")
        assert len(_rd) == 1 and _rd["date"].iloc[0] == str(_cw.index[-1].date()) and _rd["codes"].iloc[0].startswith("0")
        assert int(_rd["n"].iloc[0]) == len(_rd["codes"].iloc[0].split()) > 0
        assert _rd["codes"].iloc[0].split() == uv.current_basket(uv.walk_forward(_pw, 63, refit_models=("ridge",)))
        _, _wlog2 = uv.weekly(panel=_pw, out_dir=_td, refit_models=("ridge",), progress=lambda p, m: None)
        assert len(uv.read_picks(Path(_td) / "picks.csv")) == 1
        _kfw = uv.load_kr_factors(Path(_td) / "kr_factors.csv")              # 매주 계산이 한국판 요인도 저장
        assert list(_kfw.columns) == ["Mkt-RF", "Mom", "RF"] and _kfw.index[-1] == _cw.index[-1]   # 시총·재무가 없는 패널
        # 결과 묶음: 비용 민감도·손익분기, 시장 구간별, 위험 반영 비중, 자동 선택이 바뀐 횟수 → 검증 보고서 요약까지
        _U = _wres[(63, 1, True)]
        assert _U["cost_sens"][0].shape[0] == 5 and _U["switches"] >= 0 and _U["risk_weighted"] is not None
        import validation as _vdr
        _rk = _vdr.ranking_summary(_td)
        _mdr = _vdr.report_markdown([], _rk, "2026-10-26", "test")
        assert "손익분기 비용" in _mdr and "자동 선택이 바뀐 횟수" in _mdr and "기준 전략" in _mdr
        # 리밸런싱 날짜 바꿔 보기: 앱 기본 체결(다음 날)에서만, 교체 주기마다 시작일 5가지 → 보고서 줄까지
        assert all("offsets" not in _wres[(h, 0, t)] for h in uv.WEEKLY_HOLDS for t in (True, False))
        for _h in uv.WEEKLY_HOLDS:
            _of = _wres[(_h, 1, True)]["offsets"]
            assert list(_of["table"].index) == uv.offsets_for(_h) and _of["summary"]["auto"]["시작일 수"] == 5
            assert _of["table"].loc[0, "리밸런싱 수"] == _wres[(_h, 1, True)]["table"].loc["auto", "리밸런싱 수"]
        assert _mdr.count("리밸런싱 날짜를 바꾸면") == len(uv.WEEKLY_HOLDS)
