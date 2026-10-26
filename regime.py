"""시장 국면(regime) 분류: 2상태 마르코프 전환 모델 (안정 / 불안).

일간 로그수익률이 '지금 어느 국면인지'에 따라 다른 정규분포(평균·변동성)를 따르고,
국면은 마르코프 체인으로 바뀐다고 본다 (Hamilton 국면 전환 = 2상태 가우시안 HMM, EM으로 추정).
국면 확률은 '그날까지의 데이터만' 쓰는 필터 확률을 쓴다 (스무딩 확률은 미래 데이터를 써서 쓰지 않음).
"""
import math

import numpy as np
import pandas as pd

from kr_costs import trade_cost
from quant_engine import max_drawdown

STATE_LABELS = ("안정 국면", "불안 국면")     # 0 = 변동성 작음, 1 = 변동성 큼


def _density(x, mu, sd):
    return np.maximum(np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), 1e-300)


class RegimeSwitchingModel:
    """2상태 국면 전환 모델. 상태 1이 항상 변동성이 큰 '불안 국면'이 되도록 정렬한다."""

    def __init__(self, max_iter=200, tol=1e-7):
        self.max_iter, self.tol = max_iter, tol
        self.mu = self.sd = self.P = self.pi0 = None

    def _forward(self, x):
        """척도 조정한 전방 확률 → (필터 확률 a0, a1, 척도 c). 2상태라 파이썬 float로 빠르게 돈다."""
        b0, b1 = _density(x, self.mu[0], self.sd[0]).tolist(), _density(x, self.mu[1], self.sd[1]).tolist()
        (p00, p01), (p10, p11) = self.P
        n = len(b0)
        a0, a1, c = [0.0] * n, [0.0] * n, [0.0] * n
        f0, f1 = self.pi0[0] * b0[0], self.pi0[1] * b1[0]
        for t in range(n):
            if t:
                f0 = (a0[t - 1] * p00 + a1[t - 1] * p10) * b0[t]
                f1 = (a0[t - 1] * p01 + a1[t - 1] * p11) * b1[t]
            s = f0 + f1
            a0[t], a1[t], c[t] = f0 / s, f1 / s, s
        return np.array(a0), np.array(a1), np.array(c), np.array(b0), np.array(b1)

    def fit(self, returns, init=None):
        """EM(바움-웰치). init: 이전 모델(워크포워드에서 이어 추정할 때 시작값)."""
        x = np.asarray(returns, float)
        if init is not None:
            self.mu, self.sd, self.P, self.pi0 = init.mu.copy(), init.sd.copy(), init.P.copy(), init.pi0.copy()
        else:
            s = x.std()
            self.mu, self.sd = np.array([x.mean(), x.mean()]), np.array([0.7 * s, 1.6 * s])
            self.P, self.pi0 = np.array([[0.98, 0.02], [0.05, 0.95]]), np.array([0.5, 0.5])
        prev = -np.inf
        for _ in range(self.max_iter):
            a0, a1, c, b0, b1 = self._forward(x)
            (p00, p01), (p10, p11) = self.P
            n = len(x)
            be0, be1 = [1.0] * n, [1.0] * n
            bb0, bb1, cc = b0.tolist(), b1.tolist(), c.tolist()
            for t in range(n - 2, -1, -1):
                n0, n1 = bb0[t + 1] * be0[t + 1], bb1[t + 1] * be1[t + 1]
                be0[t] = (p00 * n0 + p01 * n1) / cc[t + 1]
                be1[t] = (p10 * n0 + p11 * n1) / cc[t + 1]
            be0, be1 = np.array(be0), np.array(be1)
            g0, g1 = a0 * be0, a1 * be1
            gs = g0 + g1
            g0, g1 = g0 / gs, g1 / gs
            # 전이 횟수 기댓값
            w0, w1 = b0[1:] * be0[1:] / c[1:], b1[1:] * be1[1:] / c[1:]
            xi = np.array([[np.sum(a0[:-1] * p00 * w0), np.sum(a0[:-1] * p01 * w1)],
                           [np.sum(a1[:-1] * p10 * w0), np.sum(a1[:-1] * p11 * w1)]])
            self.P = xi / xi.sum(axis=1, keepdims=True)
            self.pi0 = np.array([g0[0], g1[0]])
            for k, g in enumerate((g0, g1)):
                self.mu[k] = np.sum(g * x) / g.sum()
                self.sd[k] = max(math.sqrt(np.sum(g * (x - self.mu[k]) ** 2) / g.sum()), 1e-5)
            ll = float(np.sum(np.log(c)))
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        if self.sd[0] > self.sd[1]:                 # 상태 1 = 불안 국면
            self.mu, self.sd, self.pi0 = self.mu[::-1].copy(), self.sd[::-1].copy(), self.pi0[::-1].copy()
            self.P = self.P[::-1, ::-1].copy()
        self.loglik = ll
        return self

    def filter(self, returns):
        """그날까지의 데이터만으로 본 '불안 국면' 확률 (필터 확률)."""
        return self._forward(np.asarray(returns, float))[1]

    def describe(self):
        """국면별 연 평균 수익률·연 변동성·평균 지속 기간(거래일)."""
        return pd.DataFrame({"연 평균 수익률": self.mu * 252, "연 변동성": self.sd * math.sqrt(252),
                             "평균 지속 기간(거래일)": 1 / (1 - np.diag(self.P))}, index=list(STATE_LABELS))


def walk_forward(close, refit_every=252, min_train=756):
    """워크포워드 불안 국면 확률: refit_every일마다 그때까지의 수익률로만 다시 추정하고, 다음 추정 전까지 필터만 돌린다."""
    r = np.log(close).diff().dropna()
    x = r.values
    p_high = np.full(len(x), np.nan)
    model = None
    for start in range(min_train, len(x), refit_every):
        model = RegimeSwitchingModel().fit(x[:start], init=model)
        end = min(start + refit_every, len(x))
        p_high[start:end] = model.filter(x[:end])[start:end]
    return pd.Series(p_high, index=r.index, name="p_high"), r


def evaluate(p_high, r, cost_bps=0.0, threshold=0.5, lag=1, sell_tax=0.0):
    """'불안 국면 확률이 threshold 이상이면 현금, 아니면 보유' vs 단순 보유.

    t일 종가까지 보고 판단 → lag 거래일 뒤 종가에 체결 (1 = 다음 날 종가, 기본 / 0 = 판단한 그 종가, 낙관적).
    """
    known = p_high.notna()
    pos = (p_high[known] < threshold).astype(float)
    nxt = r.shift(-(1 + lag)).reindex(pos.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * nxt - trade_cost(pos, cost_bps, sell_tax, lag)     # 사고팔 때 수수료, 팔 때 증권거래세 (체결일 세율)
    equity = pd.DataFrame({"불안 국면엔 현금": np.exp(strat.cumsum()), "단순 보유": np.exp(nxt.cumsum())})

    def stats(x, eq):
        m, s = x.mean() * 252, x.std() * np.sqrt(252)
        return {"연 수익률": math.expm1(m), "연 변동성": s, "샤프": m / s if s > 0 else 0.0,
                "최대 낙폭": max_drawdown(eq)}

    table = pd.DataFrame({"불안 국면엔 현금": stats(strat, equity["불안 국면엔 현금"]),
                          "단순 보유": stats(nxt, equity["단순 보유"])}).T
    paid = pd.Series(r.index, index=r.index).shift(-(1 + lag)).reindex(pos.index)       # 수익이 실현된 날짜
    daily = pd.DataFrame({"pos": pos.values, "strat": strat.values, "asset": nxt.values}, index=paid.values)[paid.notna().values]
    return {"table": table, "equity": equity, "daily": daily, "time_in_market": float(pos.mean()),
            "switches": int(turnover.iloc[1:].sum())}


def analyze(close, cost_bps=0.0, lag=1, sell_tax=0.0):
    """앱용 묶음: 워크포워드 확률·전략 성적 + 전체 기간으로 추정한 현재 모델."""
    p_high, r = walk_forward(close)
    if p_high.notna().sum() < 63:
        raise ValueError("과거 데이터가 너무 짧습니다 (최소 약 3년 3개월 필요)")
    now = RegimeSwitchingModel().fit(r.values)
    p_now = now.filter(r.values)
    return {"p_high": p_high, "close": close, "p_now": float(p_now[-1]), "asof": r.index[-1], "states": now.describe(),
            "stay": np.diag(now.P), "in_sample_share": float((p_now >= 0.5).mean()),
            **evaluate(p_high, r, cost_bps, lag=lag, sell_tax=sell_tax)}
