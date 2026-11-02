"""여러 테스트가 함께 쓰는 합성 데이터와 무거운 계산 결과. 한 번만 계산해 테스트 전체가 나눠 쓴다 (읽기 전용으로 쓸 것)."""
import numpy as np
import pandas as pd
import pytest

import direction as dr
import regime as rg
import universe as uv
from quant_engine import add_combined_forecast, short_only_split, vol_signal_backtest
from tests.helpers import synth_panel


# ---------------- 방향 예측 ----------------
@pytest.fixture(scope="session")
def px():
    """예측력이 없는 무작위 걷기 합성 가격 1,500거래일."""
    rng = np.random.default_rng(7)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1500))),
                     index=pd.bdate_range("2015-01-01", periods=1500))


@pytest.fixture(scope="session")
def wf_a(px):
    """px의 21거래일 방향 예측 워크포워드 결과."""
    return dr.walk_forward(px, h=21)[0]


@pytest.fixture(scope="session")
def ev(px, wf_a):
    """wf_a의 기본 설정(다음 날 종가 체결) 채점."""
    return dr.evaluate(wf_a, 21, np.log(px).diff())


@pytest.fixture(scope="session")
def px_m():
    """21거래일 모멘텀을 심어 둔(예측할 수 있는) 합성 가격 2,500거래일.

    심어 둔 신호가 잡히는 정도(AUC 0.6 근처)가 난수에 따라 달라서, 테스트가 한 파일이던 때와 같은 난수 순서를 그대로 따라
    같은 가격을 만든다 (앞 테스트들이 쓰던 만큼 먼저 뽑아 버림). 기준을 바꾸지 않고 같은 데이터로 확인하기 위해서다."""
    rng = np.random.default_rng(7)
    rng.standard_normal((4000, 3)), rng.random(4000), rng.normal(0.0003, 0.01, 1500), rng.normal(0, 0.05, 299)
    r = np.zeros(2500)
    for t in range(2500):
        r[t] = 0.0002 + 0.03 * r[max(0, t - 21):t].sum() + 0.01 * rng.standard_normal()
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2015-01-01", periods=2500))


@pytest.fixture(scope="session")
def wf_m(px_m):
    """px_m의 5거래일 방향 예측 워크포워드 결과."""
    return dr.walk_forward(px_m, h=5)[0]


@pytest.fixture(scope="session")
def alt_a(px):
    """px 날짜에 맞춘 합성 실적 서프라이즈 (대체 데이터 특징 하나)."""
    rng = np.random.default_rng(9)
    return pd.DataFrame({"eps_surprise": rng.normal(0, 0.1, len(px))}, index=px.index)


@pytest.fixture(scope="session")
def alt_walk(px, alt_a):
    return dr.walk_forward(px, 21, alt=alt_a)


@pytest.fixture(scope="session")
def wa(alt_walk):
    """대체 데이터를 넣은 워크포워드 결과."""
    return alt_walk[0]


@pytest.fixture(scope="session")
def cols_a(alt_walk):
    """그때 쓴 특징 열 이름."""
    return alt_walk[1]


# ---------------- 종목 고르기 ----------------
@pytest.fixture(scope="session")
def panel():
    """종목 간 신호(외국인 수급)를 심어 둔 합성 패널 (80종목·800거래일)."""
    return synth_panel(5)


@pytest.fixture(scope="session")
def wf_u(panel):
    """panel의 21거래일 종목 고르기 워크포워드 결과 (학습 1년, 릿지만 다시 학습)."""
    return uv.walk_forward(panel, h=21, train_years=1, refit_models=("ridge",))


@pytest.fixture(scope="session")
def tab(wf_u):
    """비용 0의 모델 비교표."""
    return uv.evaluate(wf_u, cost_bps=0)[0]


@pytest.fixture(scope="session")
def capacity_panel():
    """같은 합성 패널에 하루 거래대금 1,000억 원(liq)을 붙인 것 (운용 규모·바구니 위험 계산용)."""
    pc = synth_panel(5)
    pc["feat"]["liq"] = pd.DataFrame(np.log(1e11), index=pc["close"].index, columns=pc["close"].columns)
    return pc


# ---------------- 변동성 매매 ----------------
@pytest.fixture(scope="session")
def vol_case():
    """실제 변동성 20%인 합성 가격, 늘 30%인 IV, 그 변동성 신호 백테스트."""
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2015-01-01", periods=1400)
    px = pd.Series(100 * np.exp(np.cumsum(0.2 / np.sqrt(252) * rng.standard_normal(1400))), index=idx)
    iv = pd.Series(0.30, index=idx)
    vb = vol_signal_backtest(px, iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
    return {"idx": idx, "px": px, "iv": iv, "vb": vb}


@pytest.fixture(scope="session")
def short_only_case():
    """합성 GARCH·IV 예측 표(bt)와, IV가 예측보다 비쌀 때만 팔면 이익인 표(bs)와 그 '매도만' 규칙 선택(so)."""
    rng = np.random.default_rng(31)
    k = 120
    g = 0.15 + 0.05 * rng.standard_normal(k)
    ivv = 0.18 + 0.05 * rng.standard_normal(k)
    bt = pd.DataFrame({"garch": g, "iv": ivv, "realized": 0.5 * g + 0.5 * ivv - 0.02 + 0.01 * rng.standard_normal(k),
                       "signal": 0, "long": 0.0, "cost": 0.0, "premium": 0.05},
                      index=pd.bdate_range("2015-01-01", periods=k, freq="21B"))
    spread = rng.uniform(-0.04, 0.06, k)
    bs = pd.DataFrame({"garch": g, "iv": g + spread, "realized": g + 0.005 * rng.standard_normal(k),
                       "signal": 0, "cost": 0.0, "premium": 0.05}, index=bt.index)
    bs["long"] = 0.5 * (bs["realized"] - bs["iv"])                    # 롱 스트래들 손익 ≈ 베가 × (실제 − IV)
    so = short_only_split(add_combined_forecast(bs), iv_name="VIX")
    return {"k": k, "g": g, "bt": bt, "bs": bs, "so": so}


# ---------------- 시장 국면 ----------------
@pytest.fixture(scope="session")
def regime_data():
    """안정(일 0.6%)·불안(일 2%) 두 국면을 오가는 합성 수익률 3,000일, 실제 국면, 그 가격."""
    rng = np.random.default_rng(7)
    n = 3000
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.random() < (0.01 if s[t - 1] == 0 else 0.97)
    x = np.array([0.0006, -0.001])[s] + np.array([0.006, 0.02])[s] * rng.standard_normal(n)
    close = pd.Series(100 * np.exp(np.cumsum(x)), index=pd.bdate_range("2012-01-02", periods=n))
    return {"n": n, "s": s, "x": x, "close": close}


@pytest.fixture(scope="session")
def regime_prob(regime_data):
    """워크포워드 불안 국면 확률."""
    return rg.walk_forward(regime_data["close"])[0]


@pytest.fixture(scope="session")
def regime_returns(regime_data):
    return np.log(regime_data["close"]).diff().dropna()
