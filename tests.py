"""노트북의 검증(assert)들을 그대로 옮긴 자체 테스트.  실행: python tests.py"""
from pathlib import Path

import numpy as np

from quant_engine import (GarchVolForecaster, GammaScalpingEngine, RiskNeutralPricingEngine,
                          VolatilityTradingLogic, _straddle_delta, simulate_gbm_paths)


def simulate_garch(n, omega, alpha, beta, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


# 프라이싱 / IV
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

# GARCH 파라미터 복원
g = GarchVolForecaster().fit(simulate_garch(6000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
assert abs(g.persistence - 0.98) < 0.03

# 신호
logic = VolatilityTradingLogic(vrp=0.028, threshold=0.02)
assert logic.generate_signal(0.20, 0.23) == 0
assert logic.generate_signal(0.15, 0.27) == -1
assert logic.generate_signal(0.28, 0.22) == 1

# 그릭스
e0 = GammaScalpingEngine(100.0, 100.0, 1.0, 0.05, 0.20)
h = 1e-5
v_hi = GammaScalpingEngine(100, 100, 1 + h, 0.05, 0.2).straddle_greeks()["price"]
v_lo = GammaScalpingEngine(100, 100, 1 - h, 0.05, 0.2).straddle_greeks()["price"]
assert abs(e0.straddle_greeks()["theta_per_day"] * 252 + (v_hi - v_lo) / (2 * h)) < 1e-6
assert abs(_straddle_delta(100, 100, 1.0, 0.05, 0.2) - e0.straddle_greeks()["delta"]) < 1e-12

# 벡터화된 헤지 손익 == 경로 하나씩 계산한 값, 그리고 부호가 이론대로인지
paths = simulate_gbm_paths(100, 0.30, 0.03, 0.25, 63, 50, np.random.default_rng(1))
sc = GammaScalpingEngine(100, 100, 0.25, 0.03, 0.20)
vec = sc.hedged_pnl(paths, position=1)
assert np.allclose(vec, [sc.hedged_pnl(p, position=1) for p in paths])
assert vec.mean() > 0                                      # 실현 30% > IV 20% → 롱 감마 이익
assert sc.hedged_pnl(paths, position=-1).mean() < 0

# 방향 예측: 로지스틱이 알려진 관계를 복원하는지
import pandas as pd
import direction as dr

rng = np.random.default_rng(7)
Xs = rng.standard_normal((4000, 3))
ys = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 * Xs[:, 0] - 1.0 * Xs[:, 1])))).astype(float)
m = dr.LogisticDirectionModel(l2=1e-6).fit(Xs, ys)
assert abs(m.w[0] - 1.5) < 0.15 and abs(m.w[1] + 1.0) < 0.15 and abs(m.w[2]) < 0.15

# 워크포워드에 미래 정보가 새지 않는지: T 이후 가격을 바꿔도 T까지의 예측은 그대로여야 한다
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1500))),
               index=pd.bdate_range("2015-01-01", periods=1500))
wf_a, _ = dr.walk_forward(px, h=21)
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
# 체결 시점: lag=1(기본)이면 t일 판단 → t+1일 종가 체결 → t+2일 수익률부터, lag=0이면 t+1일 수익률부터
_r = np.log(px).diff()
_pos = (wf_a["auto"] > 0.5).astype(float)
for _lag in (0, 1):
    _e = dr.evaluate(wf_a, 21, _r, lag=_lag)
    assert np.allclose(np.diff(np.log(_e["equity"]["전략"].values)),
                       (_pos * _r.shift(-(1 + _lag)).reindex(wf_a.index).fillna(0.0)).values[1:])
    assert _e["z"] == ev["z"] and _e["skill"] == ev["skill"]          # 예측 채점은 체결 시점과 무관
# 확률 구간 표: 오를 확률이 높아도 내릴 때 크게 잃으면 평균 수익은 음수 — 확률과 크기를 함께 보여야 한다
_bt = dr.bucket_table(pd.Series([0.3, 0.3, 0.7, 0.7]), pd.Series(np.log([1.1, 0.9, 1.02, 0.8])), cost=0.005)
assert list(_bt["관측 수"]) == [2, 0, 0, 0, 0, 2] and np.isclose(_bt.loc["60%~100%", "평균 수익"], -0.09)
assert np.isclose(_bt.loc["60%~100%", "오를 때 평균"], 0.02) and np.isclose(_bt.loc["60%~100%", "내릴 때 평균"], -0.2)
assert np.isclose(_bt.loc["0%~40%", "비용 뺀 평균"], -0.005) and _bt.loc["0%~40%", "실제 상승 비율"] == 0.5
# 21일 분할 보유(hold=21): 매일 판단한 1/21 몫이 체결(t+1 종가) 뒤 정확히 21거래일의 수익을 번다
# → 비용 0이면 전략 수익 합 = Σ 신호 × (t+1 종가 → t+22 종가 로그수익률) / 21 (끝에서 아직 안 끝난 몫은 거기까지만)
_eh = dr.evaluate(wf_a, 21, _r, lag=1, hold=21)
_sig, _nx = _pos.values, _r.shift(-2).reindex(wf_a.index).fillna(0.0).values
assert np.isclose(_eh["daily"]["strat"].sum(), sum(_sig[t] / 21 * _nx[t:t + 21].sum() for t in range(len(_sig))))
assert 0 < _eh["time_in_market"] < 1 and _eh["z"] == ev["z"]
_ehc = dr.evaluate(wf_a, 21, _r, 10.0, lag=1, hold=21)
assert np.isclose(_ehc["daily"]["strat"].sum(), _eh["daily"]["strat"].sum()
                  - 10e-4 * (_eh["daily"]["pos"].diff().abs().sum() + _eh["daily"]["pos"].iloc[0]))   # 비용은 상쇄된 순매매에만

# 겹침 보정 t값: 겹침이 없으면(h=1) 보통 t값과 같고, 21일씩 겹친 합이면 크게 줄어야 한다
from quant_engine import overlap_tstat
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


def _garch_t(T, R, g):
    """실제 수익률에 가까운 합성 일간 수익률 R개 (T일): GARCH(1,1) 변동성 군집 + t(5) 두꺼운 꼬리, 평균 0."""
    z = g.standard_t(5, size=(T + 300, R)) / np.sqrt(5 / 3)
    out, s2 = np.empty((T + 300, R)), np.full(R, 1e-4)
    for t in range(T + 300):
        out[t] = np.sqrt(s2) * z[t]
        s2 = 2e-6 + 0.08 * out[t] ** 2 + 0.9 * s2
    return out[300:]


# 같은 검정을 실제와 비슷한 조건에서: '63일 모멘텀이 양수면 오른다'(예측력 없음)를 21일 결과로 채점한 적중 차이.
# 진짜 차이는 0인데 '유의'가 나오는 비율이 5% 근처여야 하고, 겹침을 무시한 보통 t값은 크게 과대평가해야 한다
_nd, _R = 1700, 500
_cg = np.vstack([np.zeros(_R), np.cumsum(_garch_t(_nd + 85, _R, np.random.default_rng(7)), 0)])
_up = (_cg[63 + 21:63 + 21 + _nd] - _cg[63:63 + _nd] > 0).astype(float)
_d = ((_cg[63:63 + _nd] - _cg[:_nd] > 0) == _up).astype(float) - _up
_rej_hh = np.mean([abs(overlap_tstat(_d[:, i], 21)) > 1.96 for i in range(_R)])
_rej_naive = np.mean([abs(_d[:, i].mean() / _d[:, i].std() * np.sqrt(_nd)) > 1.96 for i in range(_R)])
assert 0.025 < _rej_hh < 0.10 and _rej_naive > 0.4, (_rej_hh, _rej_naive)

# 파이프라인이 '실제로 예측 가능한' 데이터에선 신호를 찾아내는지 (모멘텀이 있는 합성 가격)
r = np.zeros(2500)
for t in range(2500):
    r[t] = 0.0002 + 0.03 * r[max(0, t - 21):t].sum() + 0.01 * rng.standard_normal()
px_m = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2015-01-01", periods=2500))
wf_m, _ = dr.walk_forward(px_m, h=5)
ev_m = dr.evaluate(wf_m, 5, np.log(px_m).diff())
assert ev_m["auc"] > 0.6 and ev_m["skill"] > 0.02 and ev_m["hit"] > ev_m["base"] + 0.02, ev_m["table"]["auto"]

# 대체 데이터: 발표된 정보는 '발표 다음 거래일'부터만 쓰여야 한다
import alt_data as ad

idx = pd.bdate_range("2024-01-01", periods=10)       # 1/1(월) ~ 1/12(금)
stamps = pd.DatetimeIndex(["2024-01-03 16:00", "2024-01-05 08:00", "2024-01-06 10:00"], tz="America/New_York")
pos = ad._effective_pos(idx, stamps)
assert [idx[p].date().isoformat() for p in pos] == ["2024-01-04", "2024-01-08", "2024-01-08"]  # 장후·장전·주말 발표 모두 다음 거래일

assert ad.headline_score("Apple beats estimates, shares surge") > 0
assert ad.headline_score("Stock plunges after earnings miss") < 0
assert ad.headline_score("삼성전자 호실적에 신고가") > 0 and ad.headline_score("반도체 부진 우려에 급락") < 0
assert ad.headline_score("Company holds annual meeting") == 0

# 가격 기록의 뒤쪽 절반에서야 시작하는 대체 데이터는 뺀다 (모든 특징이 있는 날만 쓰므로 넣으면 학습 기간이 절반 넘게 줄어듦).
# 끝까지 비어 있는 열(이 종목엔 해당 없음)은 시작 시점 판단에서 무시
_ix = pd.bdate_range("2016-01-01", periods=2500)
_saved_src = {k: getattr(ad, k) for k in ("earnings_features", "analyst_features", "flow_features")}
try:
    ad.earnings_features = lambda t, i: pd.DataFrame({"pead": np.where(np.arange(len(i)) >= 2000, 1.0, np.nan)}, index=i)
    ad.analyst_features = lambda t, i: None
    ad.flow_features = lambda t, i: pd.DataFrame({"frgn_flow_21": np.where(np.arange(len(i)) >= 300, 1.0, np.nan),
                                                  "frgn_hold_chg": np.nan}, index=i)
    _alt_s, _st = ad.build_alt_features("SPY", _ix)
    assert list(_alt_s.columns) == ["frgn_flow_21", "frgn_hold_chg"], _alt_s.columns
    assert _st["실적 서프라이즈"].startswith("기간이 짧아 제외") and _st["외국인·기관 수급"].startswith("사용")
    assert _st["애널리스트 변경"] == "데이터 없음"
finally:
    for _k, _v in _saved_src.items():
        setattr(ad, _k, _v)

# alt 특징을 넣어도 워크포워드에 미래 정보가 새지 않는지 (T 이후 alt 값만 바꿔 봄)
alt_a = pd.DataFrame({"eps_surprise": rng.normal(0, 0.1, len(px))}, index=px.index)
alt_b = alt_a.copy()
alt_b.iloc[T + 1:] = 5.0
wa, cols_a = dr.walk_forward(px, 21, alt=alt_a)
wb, _ = dr.walk_forward(px, 21, alt=alt_b)
assert "eps_surprise" in cols_a and "logit_alt" in wa
up_to = wa.index <= px.index[T]
assert np.allclose(wa["auto"][up_to], wb["auto"][up_to]) and np.allclose(wa["logit_alt"][up_to], wb["logit_alt"][up_to])

# 앙상블은 '로지스틱·전체'와 '부스팅'의 평균 — 다시 학습하지 않고 평균만 내도 정의와 같다
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

# 가정 바꿔 점검: 자동 선택만 다시 하면(재학습 없음) 같은 기간에서 워크포워드와 똑같아야 하고, 표 모양이 맞아야 한다
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

# DART 공시: 분류 + '접수일 다음 거래일'부터 반영되는지 (네트워크 없이 공시 목록을 가짜로 주입)
import kr_data

assert kr_data.classify("주요사항보고서(자기주식취득결정)") == "dart_return"
assert kr_data.classify("주요사항보고서(유상증자결정)") == "dart_dilution"
assert kr_data.classify("[기재정정]주요사항보고서(유상증자결정)") is None        # 정정은 세지 않음
assert kr_data.classify("분기보고서 (2026.06)") == "other"
fake = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
                     "report": ["주요사항보고서(자기주식취득결정)", "단일판매ㆍ공급계약체결"], "rcept_no": ["1", "2"]})
orig = kr_data.dart_disclosures
kr_data.dart_disclosures = lambda code, start, **kw: fake
try:
    f = kr_data.dart_features("000000", idx)
finally:
    kr_data.dart_disclosures = orig
assert f.loc["2024-01-03", "dart_return"] == 0 and f.loc["2024-01-04", "dart_return"] == 1   # 1/3 접수 → 1/4부터
assert f.loc["2024-01-05", "dart_contract"] == 0 and f.loc["2024-01-08", "dart_contract"] == 1
assert f.loc["2024-01-08", "dart_total"] == 2

# 종목 고르기: 심어 둔 '종목 간' 신호를 찾는지 + T 이후 가격을 바꿔도 T까지의 점수가 그대로인지
import universe as uv

def synth_panel(seed, n_stock=80, n_day=800, shock_after=None):
    g = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_day)
    sig = np.zeros((n_day, n_stock))
    for t in range(1, n_day):
        sig[t] = 0.97 * sig[t - 1] + 0.25 * g.standard_normal(n_stock)       # 천천히 변하는 종목별 신호
    ret = 0.0015 * sig + 0.02 * g.standard_normal((n_day, n_stock)) + 0.01 * g.standard_normal((n_day, 1))
    if shock_after is not None:
        ret[shock_after + 1:] += 0.05 * np.random.default_rng(99).standard_normal((n_day - shock_after - 1, n_stock))
    codes = [f"{i:06d}" for i in range(n_stock)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=codes)
    F = {"mom_12_1": pd.DataFrame(np.random.default_rng(seed + 1).standard_normal((n_day, n_stock)), index=dates, columns=codes),
         "frgn_flow_21": pd.DataFrame(sig, index=dates, columns=codes)}
    univ = pd.DataFrame(np.repeat(np.arange(n_day)[:, None] >= 260, n_stock, axis=1), index=dates, columns=codes)
    return {"feat": F, "close": close, "universe": univ, "names": pd.Series(codes, index=codes),
            "coverage": {"mom_12_1": 1.0, "frgn_flow_21": 1.0}}

# 정답 순위: 미래 수익률이 없는 종목은 중간값(0)이 아니라 NaN → 학습에서 빠짐 (특징 순위는 결측 = 중간값 그대로)
_fw = pd.DataFrame([[0.1, np.nan, -0.1, 0.0]], columns=list("abcd"), index=[pd.Timestamp("2024-01-02")])
_uv = pd.DataFrame([[True, True, True, False]], columns=list("abcd"), index=_fw.index)
_tr = uv.target_rank(_fw, _uv).iloc[0]
assert np.allclose(_tr[["a", "c"]].values, [0.5, 0.0]) and np.isnan(_tr["b"]) and np.isnan(_tr["d"])   # a·c 둘 중 순위
assert uv.cross_rank(_fw, _uv).iloc[0]["b"] == 0.0

# 시총 순위 목록: 쪽을 넘기는 사이 순위가 바뀌어(시간외 거래 등) 같은 종목이 두 쪽에 나와도 한 번만 — 처음(높은 순위) 것
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

pa = synth_panel(5)
wf_u = uv.walk_forward(pa, h=21, train_years=1, refit_models=("ridge",))
tab, _ = uv.evaluate(wf_u, cost_bps=0)
assert tab.loc["ridge", "평균 IC"] > 0.1 and tab.loc["ridge", "상위20% 초과수익(연, 비용후)"] > 0, tab
assert tab.loc["composite", "평균 IC"] > 0.05            # 고정 가중도 (외국인 수급 + 방향) 신호를 반영
# 비용 민감도: 비용이 늘면 초과수익이 줄고, 손익분기 비용에서 정확히 0 (비용은 초과수익에 선형)
_cs, _be = uv.cost_sensitivity(wf_u)
assert _cs["초과수익(연, 비용후)"].is_monotonic_decreasing and _be > 0
assert abs(uv.evaluate(wf_u, _be, models=["auto"])[0].loc["auto", "상위20% 초과수익(연, 비용후)"]) < 1e-9
# 최근 구간을 뺀 성적: 보유 기간이 그 전에 끝난 리밸런싱만, 그 기간들의 초과수익 평균과 같다
_cv0 = uv.evaluate(wf_u, 5)[1]
_dr = uv.drop_recent(wf_u, _cv0, days=150)
_exu = (uv._period_returns(_cv0["auto"]["상위 20% (비용 후)"]) - uv._period_returns(_cv0["_universe"])).dropna()
_ixu = wf_u["universe"].index
_kp = [d for d in _exu.index if _ixu.get_loc(d) + 21 + 1 <= len(_ixu) - 1 - 150]
assert _dr["리밸런싱 수"] == len(_kp) < len(_exu) and np.isclose(_dr["초과수익(연)"], _exu[_kp].mean() * 252 / 21)
# 시장 구간별: 기간을 범위 평균 수익률 3분위로 나눔 (모든 기간이 한 번씩)
_mb = uv.market_breakdown(uv.evaluate(wf_u, 5)[1])
assert list(_mb.index) == ["하락", "횡보", "상승"] and _mb["기간 수"].sum() == tab.loc["auto", "리밸런싱 수"]
assert _mb["시장 평균(기간)"].is_monotonic_increasing
# 리밸런싱 날짜 바꿔 보기: 시작일만 o거래일 늦고 간격은 그대로 h (마지막 = 오늘의 추천은 같음). 옮김 0 = 기본 계산
assert uv.offsets_for(21) == [0, 4, 8, 13, 17] and uv.offsets_for(63) == [0, 13, 25, 38, 50]
_wfo = uv.walk_forward(pa, h=21, train_years=1, refit_models=("ridge",), offset=8)
_ixa = pa["close"].index
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
# 위험 반영 비중: 변동성 역수에 비례, 종목당 동일 비중의 2배 상한, 합 1. 변동성 자료가 없으면 동일 비중
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
# 기준 전략(저변동성 단일 등)은 같은 조건으로 표에 나오지만 자동 선택 후보는 아니다
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
cut = pa["close"].index[Tcut]
for m in ("ridge", "composite", "auto"):
    sa, sb = wf_u["scores"][m], wf_b["scores"][m]
    rows = sa.index[sa.index <= cut]
    assert len(rows) > 5 and np.allclose(sa.loc[rows].values, sb.loc[rows].values, equal_nan=True), m
# 체결 시점: 채점 수익률은 '다음 날 종가부터 h일'(기본), exec_lag=0이면 '그날 종가부터'. 점수(판단)는 같아야 한다
wf_u0 = uv.walk_forward(pa, h=21, train_years=1, refit_models=("ridge",), exec_lag=0)
_cl, _d0 = pa["close"], wf_u["rebal"][3]
_i0 = _cl.index.get_loc(_d0)
assert np.allclose(wf_u0["fwd"].loc[_d0], np.log(_cl.iloc[_i0 + 21] / _cl.iloc[_i0]))
assert np.allclose(wf_u["fwd"].loc[_d0], np.log(_cl.iloc[_i0 + 22] / _cl.iloc[_i0 + 1]))
assert wf_u["exec_lag"] == 1 and np.allclose(wf_u["scores"]["composite"].values, wf_u0["scores"]["composite"].values, equal_nan=True)
# 학습 목표도 체결 이후 h일 수익률이라 학습 모델 점수는 체결 시점에 따라 달라진다. 그만큼 학습 데이터는 하루 더 일찍 끊겨야 한다:
# 리밸런싱 날 d 바로 다음 날부터 가격을 바꿔도 d의 점수는 그대로 (끊는 시점이 t−h면 목표가 d+1 종가를 써서 달라짐)
_rdc = wf_u["rebal"][-3]
_pc = synth_panel(5, shock_after=pa["close"].index.get_loc(_rdc))
_wfc = uv.walk_forward(_pc, h=21, train_years=1, refit_models=("ridge",), sample_every=1)      # 모든 날로 학습 (경계 날 포함)
_wfa = uv.walk_forward(pa, h=21, train_years=1, refit_models=("ridge",), sample_every=1)
assert np.allclose(_wfc["scores"]["ridge"].loc[_rdc].values, _wfa["scores"]["ridge"].loc[_rdc].values, equal_nan=True)
assert not np.allclose(wf_u["scores"]["ridge"].values, wf_u0["scores"]["ridge"].values, equal_nan=True)

# 거시 지표 시차: 명시 > 티커로 자동 판단 > 모르면 안전하게 하루 늦춤
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

# 종목명 검색: 정확히 일치 > 앞부분 > 포함, 같은 순위면 시가총액 순 / 코드·티커 입력은 그대로
import market_data as md
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

# 가치 지표: 4분기 = 연간 − (1~3분기), 최근 4분기 합(TTM), 공시 다음 거래일부터 반영
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

# KRX Open API 일별매매정보 해석: 쉼표·'-' 정리, 12자리 표준코드 → 6자리
_kr = kr_data.parse_krx_daily([
    {"BAS_DD": "20261023", "ISU_CD": "005930", "ISU_NM": "삼성전자", "TDD_CLSPRC": "71,200", "FLUC_RT": "-1.25",
     "ACC_TRDVOL": "1,234", "ACC_TRDVAL": "87,860,800", "MKTCAP": "425,000,000,000,000", "LIST_SHRS": "5,969,782,550"},
    {"BAS_DD": "20261023", "ISU_CD": "KR7000660001", "ISU_NM": "SK하이닉스", "TDD_CLSPRC": "-", "FLUC_RT": "",
     "ACC_TRDVOL": "0", "ACC_TRDVAL": "0", "MKTCAP": "1", "LIST_SHRS": "2"}])
assert _kr["code"].tolist() == ["005930", "000660"] and _kr.loc[0, "close"] == 71200 and _kr.loc[0, "chg_pct"] == -1.25
assert _kr.loc[0, "cap"] == 4.25e14 and _kr.loc[0, "shares"] == 5969782550 and np.isnan(_kr.loc[1, "close"])
assert kr_data.parse_krx_daily([]).empty
# VKOSPI: 파생상품지수 표에서 '코스피 200 변동성지수'만 (그 선물지수 등은 아님), 쉼표 정리, 휴장일·실패는 NaN
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

# ---------------- t분포 GARCH · 위험 지표 ----------------
from quant_engine import GammaScalpingEngine as _GSE, max_drawdown, summarize_vol_backtest, var_es, vol_signal_backtest
import regime as rg


def simulate_garch_t(n, omega, alpha, beta, nu, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r


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
import quant_engine as qe
from types import SimpleNamespace
_orig_min = qe.minimize
qe.minimize = lambda *a, **k: SimpleNamespace(success=False, x=np.full(4, np.nan), fun=np.nan)
try:
    gf = GarchVolForecaster().fit(simulate_garch(1000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
finally:
    qe.minimize = _orig_min
assert not gf.converged and gf.warnings and abs(gf.persistence - 0.95) < 1e-12 and gf.nu == 8.0
assert np.isfinite(gf.forecast_vol(21)) and np.isfinite(gf.simulate(5, 100)).all()
assert gt.converged and not GarchVolForecaster().fit(np.zeros(10)).converged          # 표본이 너무 적으면 바로 기본값

# VaR 예외 검정: Kupiec 공식, 몰려 나온 예외는 Christoffersen이 잡고, 예외가 0개(너무 보수적)여도 기각
from scipy.stats import chi2
from quant_engine import var_backtest, var_exception_tests
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
# ES: t분포 충격의 꼬리 평균 공식이 시뮬레이션과 같고, 맞는 모델이면 '예외일 손실 ÷ 예측 ES' ≈ 1,
# 꼬리를 얇게 잡은 모델(정규분포)이면 1보다 크고 ES 검정이 기각, 분위수 손실도 더 크다
from quant_engine import _std_es, es_exceedance_test
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
from quant_engine import _std_quantile as _sq
_gf = GarchVolForecaster().fit(_xv[:750])
assert _gf.tail(0.05, "t") == (_sq(0.05, _gf.nu), _std_es(0.05, _gf.nu)) and len(_gf.std_resid) == 750
_zq, _ze = _gf.tail(0.01, "fhs")
assert np.isclose(_zq, np.quantile(_gf.std_resid, 0.01)) and np.isclose(_ze, -_gf.std_resid[_gf.std_resid <= _zq].mean())
assert set(np.round(_gf.simulate(1, 500, innov="fhs")[:, 0] / np.sqrt(_gf.sigma2_next), 9)) <= set(np.round(_gf.std_resid, 9))
assert np.allclose(_gf.simulate(3, 200), _gf.simulate(3, 200, innov="t"))                   # 기본 = VAR_INNOV ('t')
# 여러 방법을 한 번에: 같은 GARCH 적합을 공유하고, 't' 결과는 하나만 계산한 것과 같다
_vb2 = var_backtest(_xv, start=800, refit_every=42, innov=("t", "fhs"))
assert _vb2["t"].equals(_vt_t) and set(_vb2) == {"t", "fhs"}
# 하락 쪽 꼬리가 더 두꺼운 충격(지수처럼): 대칭 t는 95% VaR를 넘는 날이 목표보다 많고, FHS는 목표 근처
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

# σ 보정 (2026-10-29 사전 등록 후보): 배수 1이면 원래 채점 그대로, 배수 c면 VaR·ES·σ가 모두 c배 (NaN = 그대로)
from quant_engine import iv_blend_factor, recal_factor, var_rescale
_vt1, _vd1 = var_backtest(_xv, start=800, refit_every=42, detail=True)
_rt1, _rd1 = var_rescale(_vt1, _vd1, np.ones(len(_vd1)))
pd.testing.assert_frame_equal(_rt1, _vt1)
pd.testing.assert_frame_equal(_rd1, _vd1)
_rt2, _rd2 = var_rescale(_vt1, _vd1, np.r_[np.full(len(_vd1) - 1, 1.3), np.nan])
assert np.allclose(_rd2["VaR 99%"].iloc[:-1], 1.3 * _vd1["VaR 99%"].iloc[:-1]) and _rd2["ES 95%"].iloc[-1] == _vd1["ES 95%"].iloc[-1]
assert np.allclose(_rd2["변동성"].iloc[:-1], 1.3 * _vd1["변동성"].iloc[:-1])
assert _rt2.loc["하루 VaR 95%", "예외 비율"] <= _vt1.loc["하루 VaR 95%", "예외 비율"] and _rt2.loc["하루 VaR 95%", "재적합 수"] == _vt1.loc["하루 VaR 95%", "재적합 수"]
# 크기 재보정: 그날을 뺀 지난 window일 z² 평균의 제곱근, 지난 날이 모자라면 1, 0.8~1.5로 자름 — 그날 이후 값은 영향 없음
_rx = np.random.default_rng(3).standard_normal(400) * 0.02
_rs = np.full(400, 0.01)                                   # 실제의 절반으로 잡은 σ → 배수 약 2 → 상한 1.5
_rc = recal_factor(_rx, _rs, window=100, min_obs=50)
assert (_rc[:50] == 1).all() and np.isclose(_rc[50], min(1.5, np.sqrt(np.mean((_rx[:50] / 0.01) ** 2)))) and (_rc[100:] == 1.5).all()
_rx2 = _rx.copy()
_rx2[300:] *= 0.1
_rc2 = recal_factor(_rx2, _rs, window=100, min_obs=50)
assert np.array_equal(_rc2[:301], _rc[:301]) and _rc2[-1] < _rc[-1]
assert (np.abs(recal_factor(_rx * 0.5, _rs, 100, 50)[100:] - 1) < 0.2).all()   # σ가 맞으면 1 근처
# VKOSPI 섞기: σ² = ½ GARCH + ½ (IV/100)²/252. IV가 GARCH와 같으면 1, 두 배면 √2.5, 없거나 0이면 1
_ivd = 100 * 0.01 * np.sqrt(252)
assert np.allclose(iv_blend_factor(np.full(4, 0.01), np.array([np.nan, 0.0, _ivd, 2 * _ivd])), [1, 1, 1, np.sqrt(2.5)])

v95, es95 = var_es(np.random.default_rng(0).normal(0, 0.01, 400000), 0.95)
assert abs(v95 - (1 - np.exp(-1.645 * 0.01))) < 3e-4 and es95 > v95
assert max_drawdown([1.0, 1.2, 0.9, 1.3, 1.17]) == -0.25 and max_drawdown([1, 2, 3]) == 0.0

# 헤지 손익: 매도 = −(매수 손익) − 거래비용 (비용은 방향과 무관)
_e = _GSE(100, 100, 21 / 252, 0.02, 0.2, 0.01)
_p = simulate_gbm_paths(100, 0.25, 0.02, 21 / 252, 21, 1, np.random.default_rng(5))[0]
_g, _n = _e.hedged_pnl(_p, 1, 0.0), _e.hedged_pnl(_p, 1, 5.0)
assert abs(_e.hedged_pnl(_p, -1, 5.0) - (-_g - (_g - _n))) < 1e-10

# 여러 진입을 한 번에 계산하는 헤지 손익 == 진입 하나씩 엔진으로 계산한 값 (만기·IV가 진입마다 달라도)
from quant_engine import hedged_straddle_pnl, straddle_price
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

# 변동성 신호 백테스트: IV가 실제보다 늘 10%p 비싸면 '매도' 신호가 나고 이익이어야 한다 + 미래 정보 누출 없음
_rng = np.random.default_rng(21)
_idx = pd.bdate_range("2015-01-01", periods=1400)
_px = pd.Series(100 * np.exp(np.cumsum(0.2 / np.sqrt(252) * _rng.standard_normal(1400))), index=_idx)
_iv = pd.Series(0.30, index=_idx)
vb = vol_signal_backtest(_px, _iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
vtab, vcur = summarize_vol_backtest(vb)
assert (vb["signal"] == -1).mean() > 0.9 and vtab.loc["signal", "평균 손익(주가 대비)"] > 0
assert vtab.loc["long", "평균 손익(주가 대비)"] < 0 and len(vcur) == len(vb)
_px2 = _px.copy()
_px2.iloc[1000:] *= np.exp(np.cumsum(0.05 * _rng.standard_normal(400)))      # T=1000 이후 가격만 바꿈
vb2 = vol_signal_backtest(_px2, _iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
done = [d for i, d in enumerate(vb.index) if _idx.get_loc(d) + 1 + 21 < 1000]  # 다음 날 진입 → 만기까지 T 이전
assert len(done) > 10 and np.allclose(vb.loc[done].values, vb2.loc[done].values)

# 현실성 옵션: 옵션 체결 비용 = 비율 × 스트래들 가격, IV를 낮추면 매도 이익이 줄어듦, GARCH 예측 재사용, 진입일 늦추기
from quant_engine import vol_backtest_robustness
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

# IV 결합 예측: 과거 결과만 쓰는지 + 실제가 GARCH·IV의 섞임이면 결합이 둘 다보다 정확해야
from quant_engine import add_combined_forecast, forecast_errors, short_only_split
_rng = np.random.default_rng(31)
_k = 120
_g = 0.15 + 0.05 * _rng.standard_normal(_k)
_ivv = 0.18 + 0.05 * _rng.standard_normal(_k)
_bt = pd.DataFrame({"garch": _g, "iv": _ivv, "realized": 0.5 * _g + 0.5 * _ivv - 0.02 + 0.01 * _rng.standard_normal(_k),
                    "signal": 0, "long": 0.0, "cost": 0.0, "premium": 0.05},
                   index=pd.bdate_range("2015-01-01", periods=_k, freq="21B"))
cb = add_combined_forecast(_bt)
assert cb["combo"].iloc[:24].isna().all() and cb["combo"].iloc[24:].notna().all()
_bt2 = _bt.copy()
_bt2.iloc[80:, _bt2.columns.get_loc("realized")] = 9.9             # 80번째 이후 결과만 바꿈
cb2 = add_combined_forecast(_bt2)
assert np.allclose(cb["combo"].iloc[24:81], cb2["combo"].iloc[24:81]) and np.allclose(cb["iv_adj"].iloc[24:81], cb2["iv_adj"].iloc[24:81])
fe, n_fe = forecast_errors(cb, "VIX")
assert n_fe == _k - 24 and fe["예측 오차(RMSE)"].idxmin() == "GARCH + VIX 결합", fe
assert fe.loc["GARCH + VIX 결합", "GARCH 대비 개선 t값"] > 2 and np.isnan(fe.loc["GARCH", "GARCH 대비 개선 t값"])

# '매도만' 규칙: IV가 예측보다 비쌀 때만 팔면 이익인 데이터에서 그 규칙을 골라야 하고, 뒤 기간은 선택에 안 쓰여야 한다
_spread = _rng.uniform(-0.04, 0.06, _k)
_bs = pd.DataFrame({"garch": _g, "iv": _g + _spread, "realized": _g + 0.005 * _rng.standard_normal(_k),
                    "signal": 0, "cost": 0.0, "premium": 0.05}, index=_bt.index)
_bs["long"] = 0.5 * (_bs["realized"] - _bs["iv"])                  # 롱 스트래들 손익 ≈ 베가 × (실제 − IV)
so = short_only_split(add_combined_forecast(_bs), iv_name="VIX")
assert so["chosen"] != "항상 매도" and so["chosen"].endswith(("기준 -1%p", "기준 +0%p", "기준 +1%p")), so["chosen"]   # 참 기준선 ≈ 0
assert so["test"].loc["고른 규칙", "샤프(연)"] > so["test"].loc["항상 매도", "샤프(연)"]
_bs2 = _bs.copy()
_half = 24 + (_k - 24) // 2
_bs2.iloc[_half:, _bs2.columns.get_loc("long")] *= -1                # 뒤 기간 손익만 뒤집음
assert short_only_split(add_combined_forecast(_bs2), iv_name="VIX")["chosen"] == so["chosen"]
assert short_only_split(add_combined_forecast(_bs.iloc[:40]), iv_name="VIX") is None   # 표본 부족

# 체결 시점: 기본은 판단 다음 날 종가·그날 IV로 진입, lag=0이면 판단한 날 그대로
_ivw = pd.Series(0.2 + 0.1 * np.sin(np.arange(1400) / 30), index=_idx)
_l1 = vol_signal_backtest(_px, _ivw)
_l0 = vol_signal_backtest(_px, _ivw, lag=0, daily_iv_hedge=False)
assert np.allclose(_l1["iv_entry"].values, _ivw.shift(-1).reindex(_l1.index).values) and np.allclose(_l1["iv"], _ivw.reindex(_l1.index))
assert np.allclose(_l0["iv_entry"], _l0["iv"]) and (_l1.index == _l0.index[:len(_l1)]).all()

# '매도만' 견고성: 모든 조합에서 통하는 데이터면 고정 규칙이 늘 항상 매도보다 낫고, 뒤 기간을 뒤집으면 못해야 한다
from quant_engine import short_only_robustness
rob = short_only_robustness({(0.0, 0): _bs, (0.01, 0): _bs, (0.0, 4): _bs.iloc[2:]}, rule=so["chosen"], iv_name="VIX")
assert len(rob) == 3 and (rob["이 규칙 샤프"] > rob["항상 매도 샤프"]).all() and (rob["다시 고른 규칙"] != "항상 매도").all()
rob2 = short_only_robustness({(0.0, 0): _bs2}, rule=so["chosen"], iv_name="VIX")
assert (rob2["이 규칙 샤프"] < rob2["항상 매도 샤프"]).all()
assert "이 규칙 샤프" not in short_only_robustness({(0.0, 0): _bs}, rule=None, iv_name="VIX")

# ---------------- 시장 국면 ----------------
_rng = np.random.default_rng(7)
_n = 3000
_s = np.zeros(_n, int)
for t in range(1, _n):
    _s[t] = _rng.random() < (0.01 if _s[t - 1] == 0 else 0.97)
_x = np.array([0.0006, -0.001])[_s] + np.array([0.006, 0.02])[_s] * _rng.standard_normal(_n)
m = rg.RegimeSwitchingModel().fit(_x)
assert abs(m.sd[0] - 0.006) < 0.001 and abs(m.sd[1] - 0.02) < 0.003 and abs(m.P[0, 0] - 0.99) < 0.01, (m.sd, m.P)
assert ((m.filter(_x) >= 0.5) == _s).mean() > 0.9
# 필터 확률은 그날까지의 데이터만 쓴다: 뒤쪽 수익률을 바꿔도 앞쪽 확률은 그대로
_x2 = _x.copy()
_x2[2000:] = _rng.standard_normal(1000) * 0.05
assert np.allclose(m.filter(_x)[:2000], m.filter(_x2)[:2000])
# 워크포워드: T 이후 가격을 바꿔도 T까지의 확률은 그대로
_c = pd.Series(100 * np.exp(np.cumsum(_x)), index=pd.bdate_range("2012-01-02", periods=_n))
_c2 = _c.copy()
_c2.iloc[2100:] = _c.iloc[2100] * np.exp(np.cumsum(_x2[2100:]))
pa, _ = rg.walk_forward(_c)
pb, _ = rg.walk_forward(_c2)
assert pa.iloc[:756].isna().all() and np.allclose(pa.iloc[756:2099], pb.iloc[756:2099])
_rr = np.log(_c).diff().dropna()
assert not np.allclose(rg.evaluate(pa, _rr, lag=0)["equity"].values, rg.evaluate(pa, _rr, lag=1)["equity"].values)
out = rg.analyze(_c, 1.0)
assert out["table"].loc["불안 국면엔 현금", "최대 낙폭"] > out["table"].loc["단순 보유", "최대 낙폭"]   # 심어 둔 국면이면 낙폭이 줄어야
# ---------------- 비대칭 GARCH (GJR) ----------------
from quant_engine import fit_vol_model
def _sim_gjr(n, om, a, g, b, nu, rng):
    r = np.empty(n); s2 = om / (1 - a - g / 2 - b)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = om + (a + g * (r[i] < 0)) * r[i] ** 2 + b * s2
    return r
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

# ---------------- 알려진 요인·과적합 보정 ----------------
from quant_engine import deflated_sharpe, market_exposure, ols, timing_attribution
import market_data as md
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
# French 데이터 형식 (설명 줄, %, -99.99 결측, 뒤쪽 저작권 줄)
_ff = md.parse_french_csv("설명\n\n,Mkt-RF,SMB,RF\n20240102,  1.50, -99.99, 0.02\n20240103, -0.50, 0.10, 0.02\n\nCopyright\n")
assert list(_ff.columns) == ["Mkt-RF", "SMB", "RF"] and _ff.loc["2024-01-02", "Mkt-RF"] == 0.015
assert np.isnan(_ff.loc["2024-01-02", "SMB"]) and len(_ff) == 2
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
# 일간 표의 날짜 = 수익이 실제로 실현된 날 (판단일 + 1 + lag)
_rp = np.log(px).diff()
for _lag in (0, 1):
    _d = dr.evaluate(wf_a, 21, _rp, lag=_lag)["daily"]
    _i0 = _rp.index.get_loc(wf_a.index[0])
    assert _d.index[0] == _rp.index[_i0 + 1 + _lag] and np.isclose(_d["asset"].iloc[0], _rp.iloc[_i0 + 1 + _lag])
_dg = rg.evaluate(pa, _rr, lag=1)["daily"]
assert np.allclose(_dg["asset"].values, _rr.reindex(_dg.index).values)
# 종목 고르기: '모멘텀 단일' 모델의 초과수익은 모멘텀 요인으로 설명되고, 진짜 신호(외국인 수급)를 쓴 릿지는 알파가 남는다
_tu, _cu = uv.evaluate(wf_u, cost_bps=0)
_fm, _fr = uv.factor_exposure(wf_u, _cu, "momentum"), uv.factor_exposure(wf_u, _cu, "ridge")
assert abs(_fm["alpha_t"]) < 2 and _fm["table"].loc["모멘텀 (12-1개월)", "t값"] > 2, _fm
assert _fr["alpha_t"] > 2, _fr
_sc = uv.selection_check(_cu, 21)
assert _sc["best"] in _cu and 0 <= _sc["dsr"] <= 1 and _sc["n_trials"] == len(_cu) - 1
_sel = dr.robustness(wf_a, 21, np.log(px).diff())["selection"]
assert _sel is None or (0 <= _sel["dsr"] <= 1 and _sel["grid"].shape == (4, 3))
# 변동성 매도의 시장 노출: 큰 움직임에 돈을 버는 롱 스트래들 → 매도는 하락 상위 10% 달에 손실이 몰려야
_u = pd.Series(0.05 * _g.standard_normal(300))
_vb = pd.DataFrame({"underlying": _u, "long": 4 * _u ** 2 - 0.008, "cost": 0.0, "signal": 0})
_me = market_exposure(_vb)
assert _me.loc["항상 변동성 매도", "하락 상위 10% 달 평균 손익"] < _me.loc["항상 변동성 매도", "나머지 달 평균 손익"]
assert so["dsr"] is None or (0 <= so["dsr"]["dsr"] <= 1 and so["dsr"]["n_trials"] <= len(so["candidates"]))
# ---------------- 재현 정보 ----------------
import repro
_s1 = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3))
assert repro.fingerprint(_s1) == repro.fingerprint(_s1.copy()) and "3일" in repro.fingerprint(_s1)
assert repro.fingerprint(_s1) != repro.fingerprint(_s1 * 1.001)            # 과거 값이 고쳐지면 지문이 바뀜
assert isinstance(repro.code_version(), str) and repro.fingerprint(_s1.iloc[:0]) == "데이터 없음"

# ---------------- 종목 고르기: 운용 규모·지금 종목의 위험 ----------------
_pc = synth_panel(5)
_pc["feat"]["liq"] = pd.DataFrame(np.log(1e11), index=_pc["close"].index, columns=_pc["close"].columns)   # 하루 거래대금 1,000억 원
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

# ---------------- 앞으로의 예측 기록 ----------------
import tempfile
import alt_data as ad
import forecast_log as fl
_full = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.011 * np.random.default_rng(1).standard_normal(1150))),
                  index=pd.bdate_range("2022-01-03", periods=1150))
_cut = {"n": 1100}
_orig = {k: getattr(md, k) for k in ("load_history", "load_macro", "risk_free_rate", "dividend_yield", "atm_implied_vol",
                                     "load_vol_index")}
_orig_alt = {k: getattr(ad, k) for k in ("build_alt_features", "options_snapshot", "estimate_revisions", "news_sentiment")}
md.load_history = lambda t, years: _full.iloc[:_cut["n"]]
md.load_macro = lambda t, years=10: None
md.risk_free_rate = lambda t="", default=0.04: (0.04, "테스트")
md.dividend_yield = lambda t, s: 0.0
md.atm_implied_vol = lambda t, s, d, r, q: {"iv": 0.19, "cal_days": 30}
md.load_vol_index = lambda t, y: None
ad.build_alt_features = lambda t, idx: (None, {})
ad.options_snapshot = lambda t, s, r=0.04, q=0.0: {"pc_volume": 1.2, "pc_oi": 1.5, "skew": 0.05}
ad.estimate_revisions = lambda t: pd.DataFrame({"30일 전 대비": [0.02, 0.01]}, index=["이번 분기", "올해"])
ad.news_sentiment = lambda t, max_items=60: pd.DataFrame({"time": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-20"]),
                                                          "score": [1.0, -0.5, 0.25]})
try:
    _path = Path(tempfile.mkdtemp()) / "log.csv"
    _new, _fail = fl.record(_path, ["SPY"])
    assert len(_new) == 1 and not _fail and _new[0]["asof"] == _full.index[1099].date().isoformat()
    assert fl.record(_path, ["SPY"])[0] == []                       # 같은 날 다시 돌려도 기록이 늘지 않음 (덮어쓰지 않음)
    _cut["n"] = 1101
    assert len(fl.record(_path, ["SPY"])[0]) == 1 and len(fl.read_log(_path)) == 2
    _row = fl.read_log(_path).iloc[0]
    _p = fl.predict("SPY")                                           # 기록 = 같은 데이터로 다시 계산한 예측 (재현 가능)
    _cut["n"] = 1100
    assert np.isclose(fl.predict("SPY")["p_up"], _row["p_up"]) and np.isclose(fl.predict("SPY")["lo68"], _row["lo68"])
    assert _row["lo95"] < _row["lo68"] < _row["close"] < _row["hi68"] < _row["hi95"] and 0 < _row["var95_1d"] < _row["var99_1d"]
    # 코스피 200 종목의 하루 VaR: GARCH와 그날 VKOSPI를 반반 섞은 σ → 하루 로그수익률 분위수가 같은 배수로 커짐 (21일 범위는 그대로)
    _base = fl.predict("SPY")
    assert np.isnan(_base["var_vkospi"])                              # 대상 종목이 아니면 섞지 않음
    _ivt0, _vko0 = fl.VAR_IV_TICKERS, kr_data.vkospi_on
    fl.VAR_IV_TICKERS, kr_data.vkospi_on = ("SPY",), lambda d: 40.0
    try:
        _bl = fl.predict("SPY")
    finally:
        fl.VAR_IV_TICKERS, kr_data.vkospi_on = _ivt0, _vko0
    # 로그수익률 분위수가 같은 배수 (VaR는 단순 수익률이라 분위수 보간 때문에 1e-8 정도 어긋남)
    _c95, _c99 = (np.log1p(-_bl[k]) / np.log1p(-_base[k]) for k in ("var95_1d", "var99_1d"))
    assert _bl["var_vkospi"] == 40.0 and _c95 > 1 and np.isclose(_c95, _c99, rtol=1e-6) and np.isclose(_bl["lo68"], _base["lo68"])
    # 시장 심리 지표도 앱과 같은 값으로 저장 (뉴스는 가장 최근 헤드라인 기준 7일: 1/20 하나만)
    assert _row["pc_oi"] == 1.5 and _row["eps_rev_30d"] == 0.01 and _row["news_7d"] == 0.25 and _row["news_n"] == 1
finally:
    for _k, _v in _orig.items():
        setattr(md, _k, _v)
    for _k, _v in _orig_alt.items():
        setattr(ad, _k, _v)
# 끝나지 않은 오늘 봉은 기록에 쓰지 않는다 (한국 15:30·미국 16:00 마감 + 여유)
from datetime import datetime, timezone
_kd = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-22", "2026-10-23"]))
_utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
assert len(fl.drop_unfinished(_kd, "005930.KS", _utc("2026-10-23T01:00"))) == 1      # 한국 오전 10시 = 장중 → 뺌
assert len(fl.drop_unfinished(_kd, "005930.KS", _utc("2026-10-23T08:37"))) == 2      # 한국 오후 5:37 → 그대로
assert len(fl.drop_unfinished(_kd, "SPY", _utc("2026-10-23T16:00"))) == 1            # 뉴욕 정오 = 장중 → 뺌
assert len(fl.drop_unfinished(_kd, "SPY", _utc("2026-10-24T08:37"))) == 2            # 다음 날 새벽 → 그대로
assert len(fl.drop_unfinished(_kd, "^KS11", _utc("2026-10-23T01:00"))) == 1          # 코스피 지수도 한국 장 기준
# Yahoo가 마지막 거래일 일봉을 빼고 줄 때(늦어진 예약 실행이 한국 자정을 넘김): 네이버 종가로 그 뒤 날짜만 덧붙임
assert fl.naver_symbol("005930.KS") == "005930" and fl.naver_symbol("247540.KQ") == "247540" and fl.naver_symbol("^KS11") == "KOSPI"
assert fl.naver_symbol("SPY") is None
_yh = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-10-26", "2026-10-27"]))
_nv = pd.Series([100.0, 101.0, 103.0], index=pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"]))
_fr = fl.fill_recent(_yh, "005930.KS", fetch=lambda s: _nv)
assert list(_fr.index.strftime("%m-%d")) == ["10-26", "10-27", "10-28"] and _fr.iloc[-1] == 103.0
assert fl.fill_recent(_yh, "005930.KS", fetch=lambda s: _nv * 1.02).equals(_yh)          # 겹치는 날 종가가 다르면(배당 수정 등) 그대로
assert fl.fill_recent(_yh, "SPY", fetch=lambda s: 1 / 0).equals(_yh)                     # 한국 종목·지수가 아니면 그대로
assert fl.fill_recent(_yh, "005930.KS", fetch=lambda s: 1 / 0).equals(_yh)               # 받지 못하면 그대로
assert fl.fill_recent(_nv, "005930.KS", fetch=lambda s: _nv).equals(_nv)                 # 이미 최신이면 그대로
# Cboe 변동성 지수 CSV (Yahoo에 없는 RVX 등)
_cb = md.parse_cboe_csv("DATE,OPEN,HIGH,LOW,CLOSE\n10/21/2026,20.1,21.0,19.5,20.5\n10/22/2026,20.5,22.0,20.0,21.25\n")
assert list(_cb.index.strftime("%Y-%m-%d")) == ["2026-10-21", "2026-10-22"] and _cb.iloc[-1] == 21.25

# 채점: 손으로 만든 기록과 가격으로 계산이 맞는지
_ci = pd.bdate_range("2025-01-01", periods=80)
_cl = pd.Series(100 * np.exp(np.cumsum(np.r_[0, 0.01 * np.random.default_rng(2).standard_normal(79)])), index=_ci)
_lg = pd.DataFrame({"asof": _ci[[0, 1, 2, 70]], "ticker": "X", "close": _cl.values[[0, 1, 2, 70]] * 1.1, "h": 21,
                    "p_up": [0.9, 0.1, 0.6, 0.5], "p_base": 0.55, "model": "m", "vol_garch": 0.15,
                    "lo95": 0.0, "lo68": 0.0, "hi68": np.inf, "hi95": np.inf, "var95_1d": 0.0, "var99_1d": 1.0,
                    "iv_atm": np.nan, "vol_index": 0.2, "p_high": [0.9, 0.1, 0.1, 0.1]})
_sc = fl.score(_lg, _cl)
_up = (_cl.values[[21, 22, 23]] > _cl.values[[0, 1, 2]]).astype(float)
assert _sc["summary"]["채점 수"] == 3 and np.isclose(_sc["summary"]["적중률"], np.mean((np.array([0.9, 0.1, 0.6]) > 0.5) == _up))
_bs, _bb = (np.array([0.9, 0.1, 0.6]) - _up) ** 2, (0.55 - _up) ** 2
assert np.isclose(_sc["summary"]["확률 정확도 개선"], 1 - _bs.mean() / _bb.mean())
assert np.isclose(_sc["rows"]["ret_h"].iloc[0], _cl.iloc[21] / _cl.iloc[0] - 1)            # 기록한 종가가 아니라 비율로 채점
assert _sc["summary"]["68% 범위 적중"] == 1.0 and np.isnan(_sc["summary"]["옵션 IV 오차"])
_r1 = _cl.values[[1, 2, 3, 71]] / _cl.values[[0, 1, 2, 70]] - 1
assert _sc["summary"]["하루 VaR 95%"]["예외 횟수"] == int((_r1 < 0).sum()) and _sc["summary"]["하루 VaR 99%"]["예외 횟수"] == 0
assert np.isclose(_sc["summary"]["불안 국면일 때 실제 변동성"], _sc["rows"]["rv"].iloc[0])
# 시장 심리 지표 채점: 다음 거래일 종가부터의 수익률과 비교. 그 수익률을 그대로 아는 지표면 상관 1
_ci2 = pd.bdate_range("2025-01-01", periods=120)
_cl2 = pd.Series(100 * np.exp(np.cumsum(0.01 * np.random.default_rng(3).standard_normal(120))), index=_ci2)
_fut = _cl2.shift(-22) / _cl2.shift(-1) - 1
_lg2 = pd.DataFrame({"asof": _ci2[:90], "ticker": "X", "close": _cl2.values[:90], "h": 21, "p_up": 0.5, "p_base": 0.5, "model": "m",
                     "vol_garch": 0.15, "lo95": 0.0, "lo68": 0.0, "hi68": np.inf, "hi95": np.inf, "var95_1d": 0.02, "var99_1d": 0.03,
                     "iv_atm": np.nan, "vol_index": np.nan, "p_high": 0.1, "news_7d": _fut.values[:90],
                     "pc_oi": np.random.default_rng(4).standard_normal(90)})
_st = fl.score(_lg2, _cl2)["summary"]["시장 심리 지표"]
assert np.isclose(_st.loc["뉴스 감성 (최근 7일 평균)", "수익률과 상관"], 1.0) and _st.loc["뉴스 감성 (최근 7일 평균)", "표본"] == 90
assert abs(_st.loc["풋/콜 미결제약정 비율", "수익률 t값"]) < 3 and np.isnan(_st.loc["옵션 스큐 (풋 90% − 콜 110% IV)", "수익률과 상관"])
# ---------------- 한국 매매 비용 (증권거래세) ----------------
import kr_costs as kc
_r = kc.sell_tax_rate(pd.to_datetime(["2018-12-31", "2019-06-10", "2024-06-01", "2025-03-01", "2026-02-01"]), "KOSPI")
assert list(_r.values) == [0.0030, 0.0025, 0.0018, 0.0015, 0.0020]
assert (kc.sell_tax_rate(pd.to_datetime(["2024-06-01"]), "ETF") == 0).all()
assert kc.tax_kind("^KS11") == "INDEX" and kc.tax_kind("069500.KS", "KODEX 200") == "ETF" and kc.tax_kind("005930.KS", "삼성전자") == "KOSPI"
assert kc.tax_kind("247540.KQ") == "KOSDAQ" and kc.tax_kind("123456.KS", "아무개", "etf") == "ETF"
_p = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0], index=pd.bdate_range("2024-06-03", periods=5))
assert np.allclose(kc.trade_cost(_p, 10.0, 0.0018).values, [0.001, 0.0, 0.001 + 0.0018, 0.0, 0.001])   # 팔 때만 세금
# 방향 예측 전략: 세금은 판 날에만, 정확히 판 비중 × 세율만큼
_e0 = dr.evaluate(wf_a, 21, np.log(px).diff(), 0.0, threshold=0.5, lag=1)
_e1 = dr.evaluate(wf_a, 21, np.log(px).diff(), 0.0, threshold=0.5, lag=1, sell_tax=0.002)
_pos = (wf_a["auto"] > 0.5).astype(float)
_sold = (-_pos.diff().fillna(_pos.iloc[0])).clip(lower=0)
assert np.allclose(np.log(_e0["equity"]["전략"]).values - np.log(_e1["equity"]["전략"]).values, (_sold * 0.002).cumsum().values)
# 종목 고르기: 판 비중 × 그날 코스피 세율 (첫 매수 제외)
_t0, _c0 = uv.evaluate(wf_u, cost_bps=0)
_t1, _c1 = uv.evaluate(wf_u, cost_bps=0, sell_tax=True)
_n0 = _c0["auto"]["상위 20% (비용 후)"]; _n1 = _c1["auto"]["상위 20% (비용 후)"]
assert _n0.iloc[0] == _n1.iloc[0] and (_n1.iloc[1:] < _n0.iloc[1:]).all()

# ---------------- 가격제한폭·거래정지: 체결할 수 없는 주문 ----------------
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
# 방향 전략: 제한폭에 걸린 날이 없으면 결과가 같고, 하한가 날엔 팔지 못한 손실까지 떠안는다
_rpx = np.log(px).diff()
assert np.allclose(dr.evaluate(wf_a, 21, _rpx)["daily"]["strat"], dr.evaluate(wf_a, 21, _rpx, price_limit=False)["daily"]["strat"])
# 종목 고르기: 하한가·거래정지면 못 팔아 계속 보유, 상한가·거래정지면 못 삼 (막힌 주문 수를 표에 남김)
_wf_blk = dict(wf_u)
_wf_blk["blocked"] = {"sell": pd.DataFrame(True, index=wf_u["rebal"], columns=wf_u["universe"].columns),
                      "buy": pd.DataFrame(False, index=wf_u["rebal"], columns=wf_u["universe"].columns)}
_tb, _ = uv.evaluate(_wf_blk, cost_bps=0)
_tb0, _ = uv.evaluate(wf_u, cost_bps=0)
assert _tb.loc["ridge", "체결 막힌 주문"] > 0 and _tb.loc["ridge", "평균 회전율"] < _tb0.loc["ridge", "평균 회전율"]
assert _tb0.loc["ridge", "체결 막힌 주문"] == 0                                                   # 원래 주가가 없는 합성 패널 = 제약 없음
# 손으로 만든 3번의 리밸런싱 (30종목, 상위 6개, 매번 겹치지 않게 고름): 수익은 종목 번호 × 0.1%로 정해 둠
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
# 패널에서 막힌 주문 표 만들기: 체결일 거래량 0(거래정지) → 사고팔기 모두 막힘, 하한가 → 팔기만 막힘
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

# ---------------- 종목 고르기: 매주 미리 계산(GitHub Actions) + 추천 실전 기록 ----------------
import tempfile
_res = uv.results(_pc, wf_u, cost_bps=5.0, sell_tax=True)
assert _res["table"].equals(uv.evaluate(wf_u, 5.0, sell_tax=True)[0]) and _res["pick"] == wf_u["chosen"].iloc[-1]
assert _res["risk"] is not None and _res["capacity"] is not None and _res["n_univ"] == int(_pc["universe"].iloc[-1].sum())
# 장중에 돌리면 '오늘' 봉은 빼고 계산·기록 (한국 15:30 + 여유 40분 전), 마감 뒤·다음 날엔 그대로
_ix = pd.DatetimeIndex(["2026-10-22", "2026-10-23"])
assert uv.finished_end(_ix, pd.Timestamp("2026-10-23T03:00Z")) == _ix[0]        # 한국 낮 12시
assert uv.finished_end(_ix, pd.Timestamp("2026-10-23T08:00Z")) == _ix[1]        # 한국 오후 5시
assert uv.finished_end(_ix, pd.Timestamp("2026-10-24T01:00Z")) == _ix[1]        # 다음 날
# 기록 채점: 기록일 다음 거래일 종가에 사서 h거래일 뒤까지, 바구니 vs 그날 종목 범위 전체 (동일 비중)
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

# ---------------- 한국판 요인 (파마-프렌치 방식의 근사) ----------------
# 심어 둔 구조: 시가총액이 작을수록, 장부가/시총이 클수록 매일 조금 더 오른다 → 소형주·가치 요인이 양(+)으로 나와야
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

# 코스피200: 네이버 차트 XML → 종가 (Yahoo엔 과거 기록이 거의 없음)
_xml = ('<?xml version="1.0" encoding="EUC-KR" ?><protocol><chartdata symbol="KPI200">'
        '<item data="20261022|846.0|860.0|820.0|830.25|223456" /><item data="20261021|840.1|850.2|830.3|845.55|123456" />'
        '</chartdata></protocol>')
_nv = md.parse_naver_chart(_xml)
assert list(_nv.values) == [845.55, 830.25] and _nv.index[0] == pd.Timestamp("2026-10-21")
assert md.NAVER_INDEX["^KS200"] == "KPI200" and "^KS200" in md.NAVER_FIRST
# ---------------- 비용: 세율은 실제로 파는 날(판단 + lag 거래일) 기준 ----------------
_dt = pd.bdate_range("2025-12-26", "2026-01-09")
_tx = kc.sell_tax_rate(_dt, "KOSPI")                                   # 2025년 0.15% → 2026년 0.20%
_ps = pd.Series(1.0, index=_dt)
_ps.loc["2025-12-31":] = 0.0                                           # 12/31에 '팔자'고 판단
_c0 = kc.trade_cost(_ps, 0.0, _tx, lag=0)                              # 그날 종가에 팔면 2025년 세율
_c1 = kc.trade_cost(_ps, 0.0, _tx, lag=1)                              # 다음 거래일(1/1 휴장 무시, 달력상 1/1) 종가에 팔면 2026년 세율
assert np.isclose(_c0.loc["2025-12-31"], 0.0015) and np.isclose(_c1.loc["2025-12-31"], 0.0020)
assert np.isclose(_c1.sum(), 0.0020) and kc.trade_cost(_ps, 0.0, 0.002, lag=1).sum() == 0.002      # 숫자 세율은 그대로

# ---------------- 자기상관 보정 t값 (Newey-West) ----------------
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

# ---------------- 블록 부트스트랩 신뢰구간 ----------------
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
# 실제와 비슷한 조건(GARCH·t(5))에서 '63일 모멘텀이 양수면 보유, 다음 날 체결' 전략의 일간 초과수익(진짜 평균 0):
# 검증 보고서와 같은 b = 42의 95% 구간이 실제로 0을 95% 가까이 포함해야 한다
_nb = 1700
_rg = _garch_t(_nb + 70, 200, np.random.default_rng(5))
_cc = np.vstack([np.zeros(200), np.cumsum(_rg, 0)])
_exr = ((_cc[63:63 + _nb] - _cc[:_nb] > 0) - 1.0) * _rg[64:64 + _nb]
_cov = np.mean([(lambda lo, hi: lo <= 0 <= hi)(*qe.block_bootstrap_ci(_exr[:, i], 42, n_boot=500)) for i in range(200)])
assert _cov > 0.9, _cov

# ---------------- 상장폐지 종목까지 포함한 종목 범위 (KRX 일별매매정보 + Yahoo) ----------------
# 지금 상장된 종목은 Yahoo 수정주가 그대로, Yahoo에 없는 폐지 종목은 KRX 등락률로 주가를 만들고 폐지 뒤엔 마지막 가격에 멈춤(현금).
# 우선주는 빼고, 폐지 뒤엔 종목 범위에서 빠지며, 들고 있던 기간의 정리매매 폭락은 수익률에 그대로 들어가야 한다
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

# ---------------- 실제 데이터 검증 보고서 (데이터는 합성으로 바꿔 끼움) ----------------
import validation as vd
_orig_v = {k: getattr(md, k) for k in ("load_history", "load_macro", "kr_info")}
_orig_va = ad.build_alt_features
_hist = pd.Series(100 * np.exp(np.cumsum(0.0004 + 0.012 * np.random.default_rng(5).standard_normal(1800))),
                  index=pd.bdate_range("2019-01-02", periods=1800))
def _hist_or_fail(t, years):
    if t == "없는종목.KS":
        raise ValueError("주가 데이터 없음")
    return _hist
md.load_history = _hist_or_fail
md.load_macro = lambda t, years=10: None
md.kr_info = lambda t: ("테스트전자", "KOSPI", "stock")
ad.build_alt_features = lambda t, idx: (None, {})
_vk_orig = (vd.VKOSPI_TICKERS, vd._vkospi_prev)
vd.VKOSPI_TICKERS, vd._vkospi_prev = ("005930.KS",), lambda dates: np.full(len(dates), 25.0)   # 이 종목도 VKOSPI 섞기 대상인 척
try:
    try:
        _vr = vd.validate("005930.KS")
    finally:
        vd.VKOSPI_TICKERS, vd._vkospi_prev = _vk_orig
    # σ 보정 두 후보가 같은 예측을 다시 채점: 비율·검정·배수 (VKOSPI는 검증 기간 모든 날에 값)
    for _m in ("recal", "vkospi"):
        assert _vr[f"[{_m}] VaR 검정"] in ("통과", "기각") and _vr[f"[{_m}] ES 검정"] in ("통과", "기각")
        assert all(np.isfinite(_vr[f"분위수 손실 비율 ({_m} ÷ 기본){_p}"]) for _p in ("", " 앞 절반", " 뒤 절반"))
    assert _vr["[vkospi] VKOSPI 있는 날"] == f"{_vr['VaR 검증 일수']}/{_vr['VaR 검증 일수']}" and 0.8 <= _vr["[recal] 배수 중앙값"] <= 1.5
    assert np.isclose(_vr["[recal] 분위수 손실"] / _vr["[t] 분위수 손실"], _vr["분위수 손실 비율 (recal ÷ 기본)"], rtol=1e-6)
    assert _vr["세금 구분"] == "KOSPI" and _vr["방향 예측력"] in ("있음", "확인 안 됨") and 0 <= _vr["가정 바꿔 통과(7개 중)"] <= 7
    assert _vr["방향 초과 95% 하한"] <= _vr["방향 초과수익(연)"] <= _vr["방향 초과 95% 상한"]
    assert _vr["국면 초과 95% 하한"] <= _vr["국면 초과수익(연)"] <= _vr["국면 초과 95% 상한"]
    assert _vr["방향 전략 경제적 가치"] == ("있음" if _vr["방향 초과 95% 하한"] > 0 else "확인 안 됨")
    assert _vr["분할 초과 95% 하한"] <= _vr["분할 초과수익(연)"] <= _vr["분할 초과 95% 상한"] and 0 < _vr["분할 보유 평균 비중"] <= 1
    assert len(_vr["_obs"]) > 0 and (_vr["_obs"]["c"] >= 2 * vd.COST_BPS * 1e-4).all()      # 비용 = 수수료 양쪽 + 세금
    assert _vr["ES 검정"] in ("통과", "기각") and _vr["VaR 검증 일수"] > 500 and _vr["GARCH 진단"].count("수렴") == 1
    assert _vr["분위수 손실 비율 (FHS ÷ t)"] > 0
    # VKOSPI 섞기 대상 종목은 본 표가 섞은 VaR (2026-10-29 판정), 아니면 기본(t) 그대로
    assert _vr["VaR σ 보정"] == "VKOSPI 섞기" and _vr["VaR 검정"] == _vr["[vkospi] VaR 검정"]
    assert _vr["VaR 95% 예외 비율"] == _vr["[vkospi] VaR 95% 예외"] and _vr["VaR 99% 예외 비율"] == _vr["[vkospi] VaR 99% 예외"]
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
    # 전날 VKOSPI: 수익률 날짜마다 그 전 거래일 값 (첫날은 없음)
    _gk0, _vk0 = kr_data.get_key, kr_data.vkospi
    kr_data.get_key = lambda n: "x"
    kr_data.vkospi = lambda dates, workers=4: (pd.Series(np.arange(len(dates), dtype=float), index=dates), ["코스피 200 변동성지수"])
    try:
        _pv = vd._vkospi_prev(_hist.index[:5])
    finally:
        kr_data.get_key, kr_data.vkospi = _gk0, _vk0
    assert np.isnan(_pv[0]) and np.array_equal(_pv[1:], np.arange(4.0))
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
        assert "## 최근 1년(이례적 급등락 구간)을 뺀 결과" in _md
        assert _rows[0]["VaR σ 보정"] == "없음" and _rows[0]["VaR 검정"] == _rows[0]["[t] VaR 검정"]   # 대상이 아니면 본 표 = 기본(t)
        assert _rows[0]["VaR 95% 예외 비율"] == _rows[0]["[t] VaR 95% 예외"]
        _sv = pd.read_csv(Path(_vd) / "summary.csv")
        assert len(_sv) == 2 and pd.isna(_sv.loc[0, "오류"]) and "주가 데이터 없음" in _sv.loc[1, "오류"]
finally:
    for _k, _v in _orig_v.items():
        setattr(md, _k, _v)
    ad.build_alt_features = _orig_va
print("모든 테스트 통과")
