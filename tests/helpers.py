"""여러 테스트 파일이 함께 쓰는 합성 데이터 생성기."""
import numpy as np
import pandas as pd


def simulate_garch(n, omega, alpha, beta, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def simulate_garch_t(n, omega, alpha, beta, nu, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r


def _sim_gjr(n, om, a, g, b, nu, rng):
    r = np.empty(n); s2 = om / (1 - a - g / 2 - b)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = om + (a + g * (r[i] < 0)) * r[i] ** 2 + b * s2
    return r


def _garch_t(T, R, g):
    """실제 수익률에 가까운 합성 일간 수익률 R개 (T일): GARCH(1,1) 변동성 군집 + t(5) 두꺼운 꼬리, 평균 0."""
    z = g.standard_t(5, size=(T + 300, R)) / np.sqrt(5 / 3)
    out, s2 = np.empty((T + 300, R)), np.full(R, 1e-4)
    for t in range(T + 300):
        out[t] = np.sqrt(s2) * z[t]
        s2 = 2e-6 + 0.08 * out[t] ** 2 + 0.9 * s2
    return out[300:]


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
