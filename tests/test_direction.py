"""방향 예측: 모델 복원, 워크포워드에 미래 정보가 새지 않는지, 체결 시점, 앙상블·가정 바꿔 점검."""
import numpy as np
import pandas as pd

import direction as dr


def test_logistic_recovers_known_relation():
    """방향 예측: 로지스틱이 알려진 관계를 복원하는지"""
    rng = np.random.default_rng(7)
    Xs = rng.standard_normal((4000, 3))
    ys = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 * Xs[:, 0] - 1.0 * Xs[:, 1])))).astype(float)
    m = dr.LogisticDirectionModel(l2=1e-6).fit(Xs, ys)
    assert abs(m.w[0] - 1.5) < 0.15 and abs(m.w[1] + 1.0) < 0.15 and abs(m.w[2]) < 0.15


def test_walk_forward_no_lookahead(px, wf_a):
    """워크포워드에 미래 정보가 새지 않는지: T 이후 가격을 바꿔도 T까지의 예측은 그대로여야 한다"""
    rng = np.random.default_rng(8)
    T = 1200
    px_b = px.copy()
    px_b.iloc[T + 1:] *= np.exp(np.cumsum(rng.normal(0, 0.05, len(px) - T - 1)))
    wf_b, _ = dr.walk_forward(px_b, h=21)
    upto = wf_a.index <= px.index[T]
    assert upto.sum() > 100
    for col in ("auto", "base", "logit_price", "gbm", "ensemble"):     # 자동 선택 포함 모든 후보
        assert np.allclose(wf_a[col][upto], wf_b[col][upto]), col
    ev = dr.evaluate(wf_a, 21, np.log(px).diff())
    assert 0 <= ev["hit"] <= 1 and len(ev["equity"]) == len(wf_a)


def test_execution_lag_and_staggered_hold(px, wf_a, ev):
    """체결 시점: lag=1(기본)이면 t일 판단 → t+1일 종가 체결 → t+2일 수익률부터, lag=0이면 t+1일 수익률부터"""
    _r = np.log(px).diff()
    _pos = (wf_a["auto"] > 0.5).astype(float)
    for _lag in (0, 1):
        _e = dr.evaluate(wf_a, 21, _r, lag=_lag)
        assert np.allclose(np.diff(np.log(_e["equity"]["전략"].values)),
                           (_pos * _r.shift(-(1 + _lag)).reindex(wf_a.index).fillna(0.0)).values[1:])
        assert _e["z"] == ev["z"] and _e["skill"] == ev["skill"]          # 예측 채점은 체결 시점과 무관

    # 21일 분할 보유(hold=21): 매일 판단한 1/21 몫이 체결(t+1 종가) 뒤 정확히 21거래일의 수익을 번다
    # → 비용 0이면 전략 수익 합 = Σ 신호 × (t+1 종가 → t+22 종가 로그수익률) / 21 (끝에서 아직 안 끝난 몫은 거기까지만)
    _eh = dr.evaluate(wf_a, 21, _r, lag=1, hold=21)
    _sig, _nx = _pos.values, _r.shift(-2).reindex(wf_a.index).fillna(0.0).values
    assert np.isclose(_eh["daily"]["strat"].sum(), sum(_sig[t] / 21 * _nx[t:t + 21].sum() for t in range(len(_sig))))
    assert 0 < _eh["time_in_market"] < 1 and _eh["z"] == ev["z"]
    _ehc = dr.evaluate(wf_a, 21, _r, 10.0, lag=1, hold=21)
    assert np.isclose(_ehc["daily"]["strat"].sum(), _eh["daily"]["strat"].sum()
                      - 10e-4 * (_eh["daily"]["pos"].diff().abs().sum() + _eh["daily"]["pos"].iloc[0]))   # 비용은 상쇄된 순매매에만


def test_probability_bucket_table():
    """확률 구간 표: 오를 확률이 높아도 내릴 때 크게 잃으면 평균 수익은 음수 — 확률과 크기를 함께 보여야 한다"""
    _bt = dr.bucket_table(pd.Series([0.3, 0.3, 0.7, 0.7]), pd.Series(np.log([1.1, 0.9, 1.02, 0.8])), cost=0.005)
    assert list(_bt["관측 수"]) == [2, 0, 0, 0, 0, 2] and np.isclose(_bt.loc["60%~100%", "평균 수익"], -0.09)
    assert np.isclose(_bt.loc["60%~100%", "오를 때 평균"], 0.02) and np.isclose(_bt.loc["60%~100%", "내릴 때 평균"], -0.2)
    assert np.isclose(_bt.loc["0%~40%", "비용 뺀 평균"], -0.005) and _bt.loc["0%~40%", "실제 상승 비율"] == 0.5


def test_pipeline_finds_planted_momentum(px_m, wf_m):
    """파이프라인이 '실제로 예측 가능한' 데이터에선 신호를 찾아내는지 (모멘텀이 있는 합성 가격)"""
    ev_m = dr.evaluate(wf_m, 5, np.log(px_m).diff())
    assert ev_m["auc"] > 0.6 and ev_m["skill"] > 0.02 and ev_m["hit"] > ev_m["base"] + 0.02, ev_m["table"]["auto"]


def test_alt_features_no_lookahead(px, alt_a, wa, cols_a):
    """alt 특징을 넣어도 워크포워드에 미래 정보가 새지 않는지 (T 이후 alt 값만 바꿔 봄)"""
    T = 1200
    alt_b = alt_a.copy()
    alt_b.iloc[T + 1:] = 5.0
    wb, _ = dr.walk_forward(px, 21, alt=alt_b)
    assert "eps_surprise" in cols_a and "logit_alt" in wa
    up_to = wa.index <= px.index[T]
    assert np.allclose(wa["auto"][up_to], wb["auto"][up_to]) and np.allclose(wa["logit_alt"][up_to], wb["logit_alt"][up_to])


def test_ensemble_is_average(px, alt_a, wa, cols_a):
    """앙상블은 '로지스틱·전체'와 '부스팅'의 평균 — 다시 학습하지 않고 평균만 내도 정의와 같다"""
    rng = np.random.default_rng(10)
    assert "logit_all" in wa and np.allclose(wa["ensemble"], (wa["logit_all"] + wa["gbm"]) / 2)
    _c = dr.make_candidates(cols_a)
    _X = dr.build_features(px, alt=alt_a).values
    _y = (rng.random(len(_X)) < 0.5).astype(float)
    _fp = dr.fit_predict_all(dr.make_candidates(cols_a), _X[:800], _y[:800], _X[800:900])
    assert np.allclose(_fp["ensemble"], _c["ensemble"].fit(_X[:800], _y[:800]).predict_proba(_X[800:900]))
    _cols_p = list(dr.build_features(px).columns)                         # 가격 특징만 ('로지스틱·전체' 후보 없음)
    _Xp = dr.build_features(px).values
    _cp = dr.make_candidates(_cols_p)
    assert "logit_all" not in _cp
    assert np.allclose(dr.fit_predict_all(dr.make_candidates(_cols_p), _Xp[:800], _y[:800], _Xp[800:900])["ensemble"],
                       _cp["ensemble"].fit(_Xp[:800], _y[:800]).predict_proba(_Xp[800:900]))


def test_reselect_and_robustness(px, wf_a, px_m, wf_m, cols_a):
    """가정 바꿔 점검: 자동 선택만 다시 하면(재학습 없음) 같은 기간에서 워크포워드와 똑같아야 하고, 표 모양이 맞아야 한다"""
    rs = dr.reselect(wf_a, 21, 756)
    assert np.allclose(rs["auto"], wf_a["auto"]) and (rs["chosen"] == wf_a["chosen"]).all()
    assert (dr.reselect(wf_a, 21, 10_000, min_scored=10_000)["chosen"] == "base").all()      # 채점 표본 부족 → 기본 확률
    rb_m = dr.robustness(wf_m, 5, np.log(px_m).diff(), 1.0)
    assert list(rb_m["by_window"].index) == ["최근 1년", "최근 2년", "최근 3년", "최근 5년"] and len(rb_m["by_period"]) == 3
    assert rb_m["by_rule"].shape == (3, 4) and list(rb_m["by_rule"].index) == ["45%", "50%", "55%"]
    assert (rb_m["by_window"]["확률 정확도 개선"] > 0.01).all()      # 심어 둔 모멘텀은 선택 기간과 상관없이 잡혀야
    assert not dr.robustness(wf_a, 21, np.log(px).diff())["by_window"]["검증 통과"].any()   # 순수 무작위 가격은 통과 못 해야
    assert dr.evaluate(wf_a, 21, np.log(px).diff(), threshold=1.0)["time_in_market"] == 0.0
    assert "gbm" not in dr.make_candidates(cols_a, light=True) and "logit_all" in dr.make_candidates(cols_a, light=True)
    wl, _ = dr.walk_forward(px, 21, light=True)
    assert "gbm" not in wl and "ensemble" not in wl and np.allclose(wl["logit_price"], wf_a["logit_price"])   # 가벼운 점검도 같은 로지스틱
    hz = dr.horizon_check(px_m, horizons=(5, 21))                  # 가벼운 점검(로지스틱)도 심어 둔 신호를 잡아야
    assert list(hz.index) == ["5거래일", "21거래일"] and hz.loc["5거래일", "확률 정확도 개선"] > 0
    assert "부스팅·앙상블 선택 비율" not in hz
    hzf = dr.horizon_check(px_m, horizons=(21,), light=False, refit_every=252)        # 부스팅 포함: 선택 비율 열이 붙어야
    assert 0 <= hzf.loc["21거래일", "부스팅·앙상블 선택 비율"] <= 1
