"""노트북(Quant_Finance_Project_fixed.ipynb)의 핵심 클래스를 프로그램용 모듈로 옮긴 것.

프라이싱/IV → GARCH 변동성 예측 → 신호 → 델타헤지(감마 스캘핑) 손익 → 과거 IV로 신호 검증, 위험 지표(VaR·ES·MDD).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.signal import lfilter
from scipy.special import gammaln, xlogy
from scipy.stats import chi2, kurtosis, norm, skew
from scipy.stats import t as student_t


class RiskNeutralPricingEngine:
    """유러피안 옵션의 블랙-숄즈 가격·그릭스·IV (위험중립 측도, 연속 배당수익률 q 지원).

    S: 현재가, K: 행사가, T: 잔존만기(년), r: 무위험이자율, q: 연속 배당수익률.
    미국식 개별주식 옵션(조기행사·이산배당)에는 근사치일 뿐이다.
    """

    def __init__(self, S, K, T, r, option_type='call', q=0.0):
        self.S, self.K, self.T, self.r, self.q = S, K, T, r, q
        self.option_type = option_type.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type은 'call' 또는 'put'이어야 합니다.")

    def _d1_d2(self, sigma):
        vol_t = sigma * np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / vol_t
        return d1, d1 - vol_t

    def price(self, sigma):
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        if self.T <= 0 or sigma <= 0:  # 만기이거나 변동성 0이면 내재가치
            return max(fs - dk, 0.0) if self.option_type == 'call' else max(dk - fs, 0.0)
        d1, d2 = self._d1_d2(sigma)
        if self.option_type == 'call':
            return fs * norm.cdf(d1) - dk * norm.cdf(d2)
        return dk * norm.cdf(-d2) - fs * norm.cdf(-d1)

    def delta(self, sigma):
        d1, _ = self._d1_d2(sigma)
        dfq = np.exp(-self.q * self.T)
        return dfq * norm.cdf(d1) if self.option_type == 'call' else -dfq * norm.cdf(-d1)

    def gamma(self, sigma):
        d1, _ = self._d1_d2(sigma)
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))

    def vega(self, sigma):
        """sigma 1.00(=100%p) 변화당 가격 변화."""
        if self.T <= 0:
            return 0.0
        d1, _ = self._d1_d2(sigma)
        return self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(d1)

    def theta(self, sigma):
        """연 단위 세타 (시간 경과에 따른 가격 변화, 보통 음수)."""
        d1, d2 = self._d1_d2(sigma)
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        decay = -fs * norm.pdf(d1) * sigma / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            return decay - self.r * dk * norm.cdf(d2) + self.q * fs * norm.cdf(d1)
        return decay + self.r * dk * norm.cdf(-d2) - self.q * fs * norm.cdf(-d1)

    def price_bounds(self):
        """무차익 가격 범위(하한, 상한). 이 밖의 가격에는 IV가 존재하지 않는다."""
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        return (max(fs - dk, 0.0), fs) if self.option_type == 'call' else (max(dk - fs, 0.0), dk)

    def implied_volatility(self, market_price, tol=1e-8, max_iter=50, lo=1e-4, hi=5.0):
        """뉴턴-랩슨(빠름) → 실패 시 브렌트(항상 수렴). 해가 없으면 nan."""
        lower, upper = self.price_bounds()
        if self.T <= 0 or not (lower < market_price < upper):
            return np.nan

        sigma = 0.20
        for _ in range(max_iter):
            diff = self.price(sigma) - market_price
            v = self.vega(sigma)
            if v < 1e-12:
                break
            step = diff / v
            if abs(step) < tol:
                return sigma
            new = sigma - step
            if not (lo < new < hi):
                break
            sigma = new
        try:
            return brentq(lambda s: self.price(s) - market_price, lo, hi, xtol=tol)
        except ValueError:
            return np.nan


class GarchVolForecaster:
    """GARCH(1,1) 최우추정 + 옵션 잔존만기 구간의 '평균' 변동성 예측.

    dist="t"(기본): 충격이 스튜던트 t분포 — 폭락·폭등(두꺼운 꼬리)을 반영, 자유도 nu도 추정.
    dist="normal": 가우시안 (nu = ∞).
    asym=True: GJR-GARCH — 전날 수익률이 음수였으면 충격 반응이 α 대신 α+γ (하락한 날 변동성이 더 커지는 '레버리지 효과').
    입력: 일간 로그수익률(소수 단위). 출력: 연율화된 소수 변동성 → IV와 바로 비교 가능.
    """

    def __init__(self, trading_days=252, dist="t", asym=False):
        if dist not in ("t", "normal"):
            raise ValueError("dist는 't' 또는 'normal'이어야 합니다.")
        self.trading_days, self.dist, self.asym = trading_days, dist, bool(asym)
        self.omega = self.alpha = self.beta = self.sigma2_next = None
        self.gamma = 0.0         # 하락한 날의 추가 충격 반응 (GJR일 때만)
        self.nu = np.inf
        self.mu = 0.0            # 적합 구간 평균 일간 수익률 (분산 갱신 시 빼는 값)
        self.cond_vol = None     # 적합 구간의 조건부 변동성(연율) — 차트용
        self.converged = True    # 최적화가 실패해 기본값으로 대체했으면 False
        self.warnings = []       # 경계값에 닿은 추정 등 사용자에게 알릴 내용
        self.nll = np.nan        # 음의 로그우도 (같은 데이터·분포끼리 비교용, % 단위 스케일 상수 제외)

    # 최적화가 실패했을 때 쓰는 보수적인 기본값 (주식 일간 수익률에서 흔한 수준, 장기 분산 = 표본 분산)
    FALLBACK = {"alpha": 0.05, "beta": 0.90, "nu": 8.0}

    @property
    def name(self):
        return "GJR-GARCH" if self.asym else "GARCH"

    @property
    def n_params(self):
        return 3 + self.asym + (self.dist == "t")

    @property
    def aic(self):
        """AIC = 2·변수 수 + 2·음의 로그우도. 같은 수익률로 적합한 모델끼리 낮을수록 낫다."""
        return 2 * self.n_params + 2 * self.nll

    def fit(self, daily_log_returns):
        r = np.asarray(daily_log_returns, float)
        r = r[np.isfinite(r)]
        self.mu = float(r.mean()) if len(r) else 0.0
        x = (r - self.mu) * 100.0          # 수치 안정성을 위해 % 단위로 스케일
        n, var0 = len(x), x.var()
        self.converged, self.warnings = True, []
        neg = (x[:-1] < 0).astype(float)
        gi = 3 if self.asym else None      # 변수 순서: ω, α, β, (γ), (ν)
        ni = 3 + self.asym if self.dist == "t" else None

        def cond_var(p):
            omega, alpha, beta = p[:3]
            gamma = p[gi] if gi is not None else 0.0
            s2 = np.empty(n)
            s2[0] = var0
            s2[1:], _ = lfilter([1.0], [1.0, -beta], omega + (alpha + gamma * neg) * x[:-1] ** 2, zi=[beta * var0])
            return s2

        def nll(p):
            s2 = cond_var(p)
            if self.dist == "normal":
                return 0.5 * np.sum(np.log(s2) + x**2 / s2)
            nu = p[ni]       # 분산이 s2가 되도록 표준화한 t분포
            const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            return np.sum(0.5 * np.log(s2) + (nu + 1) / 2 * np.log1p(x**2 / ((nu - 2) * s2))) - n * const

        bounds = ([(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)] + ([(0.0, 0.999)] if self.asym else [])
                  + ([(2.1, 200.0)] if self.dist == "t" else []))
        half_g = (lambda p: 0.5 * p[gi]) if self.asym else (lambda p: 0.0)
        cons = [{'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2] - half_g(p)}]
        # 첫 시작값으로 성공하면 그대로 쓰고(대부분), 실패하면 다른 시작값으로 차례로 다시 시도한다
        starts = [[0.05 * var0, 0.05, 0.90, 0.05, 8.0], [0.10 * var0, 0.10, 0.80, 0.10, 5.0], [0.02 * var0, 0.03, 0.95, 0.02, 12.0]]
        if not self.asym:
            starts = [st[:3] + st[4:] for st in starts]
        best = None
        if n >= 30 and var0 > 0:
            for x0 in starts:
                with np.errstate(all="ignore"):
                    res = minimize(nll, x0=x0[:len(bounds)], method='SLSQP', bounds=bounds, constraints=cons)
                ok = (res.success and np.all(np.isfinite(res.x)) and np.isfinite(res.fun)
                      and res.x[1] + res.x[2] + half_g(res.x) < 1.0)
                if ok and (best is None or res.fun < best.fun):
                    best = res
                if best is not None:
                    break
        if best is None:                    # 추정 실패 → 기본값으로 대체하고 알린다 (조용히 틀린 값을 쓰지 않음)
            fb = self.FALLBACK
            a, b = fb["alpha"], fb["beta"]
            p = [max(var0, 1e-8) * (1 - a - b), a, b] + ([0.0] if self.asym else []) + ([fb["nu"]] if self.dist == "t" else [])
            self.converged = False
            self.warnings.append(f"GARCH 추정이 수렴하지 않아 기본값(α={a}, β={b}, 장기 분산 = 표본 분산)으로 대체했습니다.")
        else:
            p = list(best.x)
        self.nu = float(p[ni]) if self.dist == "t" else np.inf
        omega, alpha, beta = p[:3]
        self.gamma = float(p[gi]) if self.asym else 0.0
        if best is not None:
            if alpha + beta + 0.5 * self.gamma > 0.995:
                self.warnings.append("α+β가 1에 매우 가깝게 추정됐습니다 — 변동성 충격이 거의 사라지지 않는다는 뜻이라 "
                                     "장기 예측이 불안정할 수 있습니다.")
            if self.dist == "t" and self.nu >= 199.0:
                self.warnings.append("꼬리 두께 ν가 상한(200)에 닿았습니다 — 정규분포와 거의 같다는 뜻입니다.")
            if self.dist == "t" and self.nu <= 2.11:
                self.warnings.append("꼬리 두께 ν가 하한(2.1)에 닿았습니다 — 극단값에 끌려간 추정일 수 있습니다.")
        if n >= 2 and var0 > 0:
            s2 = cond_var(p)
            self.nll = float(nll(p))
        else:
            s2 = np.full(max(n, 1), max(var0, 1e-8))
        self.omega, self.alpha, self.beta = omega / 1e4, alpha, beta
        last = x[-1] if n else 0.0
        self.sigma2_next = (omega + (alpha + self.gamma * (last < 0)) * last ** 2 + beta * s2[-1]) / 1e4
        self.cond_vol = np.sqrt(s2 / 1e4 * self.trading_days)
        return self

    def next_var(self, s2, e):
        """오늘 분산 s2와 오늘 충격 e(평균을 뺀 일간 수익률, 소수) → 내일 분산."""
        return self.omega + (self.alpha + self.gamma * (e < 0)) * e ** 2 + self.beta * s2

    @property
    def persistence(self):
        """변동성 충격의 지속성. GJR은 하락이 절반 확률이라 α + γ/2 + β."""
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def long_run_var(self):
        return self.omega / (1.0 - self.persistence)

    def daily_var_path(self, horizon_days):
        """향후 1..horizon_days 각 날의 예측 일간 분산."""
        k = np.arange(horizon_days)
        return self.long_run_var + self.persistence ** k * (self.sigma2_next - self.long_run_var)

    def forecast_vol(self, horizon_days):
        """향후 horizon_days 동안의 평균 변동성(연율, 소수)."""
        return float(np.sqrt(self.daily_var_path(horizon_days).mean() * self.trading_days))

    def simulate(self, horizon_days, n_paths=10000, seed=0):
        """적합한 GARCH(+t 충격)로 앞으로 1..horizon_days일의 누적 로그수익률을 시뮬레이션 → (n_paths, horizon_days).

        평균은 0으로 둔다 (중앙값 ≈ 현재가). 변동성이 날마다 바뀌고 충격이 두꺼운 꼬리를 가지므로,
        정규분포 공식(±z·σ√h)보다 폭락·폭등 쪽 분위수가 현실에 가깝다. GJR이면 하락한 날 다음 변동성이 더 커진다.
        """
        rng = np.random.default_rng(seed)
        s2 = np.full(n_paths, self.sigma2_next)
        cum, out = np.zeros(n_paths), np.empty((n_paths, horizon_days))
        for k in range(horizon_days):
            if np.isfinite(self.nu):
                z = rng.standard_t(self.nu, n_paths) * np.sqrt((self.nu - 2) / self.nu)
            else:
                z = rng.standard_normal(n_paths)
            r = np.sqrt(s2) * z
            cum += r
            out[:, k] = cum
            s2 = self.next_var(s2, r)
        return out


VOL_MODELS = {"auto": "자동 (AIC로 고름)", "garch": "GARCH (대칭)", "gjr": "GJR-GARCH (하락 때 더 커짐)"}


def fit_vol_model(returns, model="auto", dist="t"):
    """변동성 모델 적합. model: 'garch' | 'gjr' | 'auto'(둘 다 적합해 AIC가 낮은 쪽, GJR이 수렴하지 못하면 GARCH)."""
    if model in ("garch", "gjr"):
        return GarchVolForecaster(dist=dist, asym=model == "gjr").fit(returns)
    g = GarchVolForecaster(dist=dist).fit(returns)
    j = GarchVolForecaster(dist=dist, asym=True).fit(returns)
    return j if j.converged and np.isfinite(j.aic) and j.aic < g.aic else g


class VolatilityTradingLogic:
    """edge = IV − (예측 RV + vrp).  edge > threshold → 변동성 매도, edge < −threshold → 변동성 매수."""
    LABELS = {1: "LONG_VOL (변동성 매수)", -1: "SHORT_VOL (변동성 매도)", 0: "HOLD (관망)"}

    def __init__(self, vrp=0.0, threshold=0.02):
        self.vrp, self.threshold = vrp, threshold

    @staticmethod
    def estimate_vrp(past_iv, past_realized_vol):
        return float(np.mean(np.asarray(past_iv, float) - np.asarray(past_realized_vol, float)))

    def edge(self, expected_rv, market_iv):
        return market_iv - (expected_rv + self.vrp)

    def generate_signal(self, expected_rv, market_iv):
        e = self.edge(expected_rv, market_iv)
        if e > self.threshold:
            return -1
        if e < -self.threshold:
            return 1
        return 0


def _straddle_delta(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (2 * norm.cdf(d1) - 1)


class GammaScalpingEngine:
    """ATM 스트래들(콜+풋)의 그릭스와 델타헤지 경로 손익.

    position = +1: 롱 스트래들(롱 감마) / -1: 숏 스트래들(숏 감마). 신호 부호를 그대로 쓴다.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S, self.K, self.T, self.r, self.sigma, self.q = S, K, T, r, sigma, q
        self.call = RiskNeutralPricingEngine(S, K, T, r, 'call', q)
        self.put = RiskNeutralPricingEngine(S, K, T, r, 'put', q)

    def straddle_greeks(self):
        s = self.sigma
        return {"delta": self.call.delta(s) + self.put.delta(s),
                "gamma": self.call.gamma(s) + self.put.gamma(s),
                "vega": self.call.vega(s) + self.put.vega(s),
                "theta_per_day": (self.call.theta(s) + self.put.theta(s)) / 252,
                "price": self.call.price(s) + self.put.price(s)}

    def hedged_pnl(self, paths, position=1, cost_bps=0.0):
        """경로(들)를 따라 매 스텝 델타헤지했을 때의 만기 손익(스트래들 1단위).

        paths: 길이 n+1 배열 하나, 또는 (경로수, n+1) 배열 — 여러 경로를 한 번에 벡터 계산.
        """
        S = np.asarray(paths, float)
        single = S.ndim == 1
        S = np.atleast_2d(S)
        n = S.shape[1] - 1
        dt, cost = self.T / n, cost_bps * 1e-4
        K, r, q, sg = self.K, self.r, self.q, self.sigma

        shares = -position * _straddle_delta(S[:, 0], K, self.T, r, sg, q)
        cash = -position * self.straddle_greeks()["price"] - shares * S[:, 0] - cost * np.abs(shares) * S[:, 0]
        for t in range(1, n + 1):
            cash = cash * np.exp(r * dt) + shares * S[:, t - 1] * q * dt
            if t < n:
                new = -position * _straddle_delta(S[:, t], K, self.T - t * dt, r, sg, q)
                trade = new - shares
                cash -= trade * S[:, t] + cost * np.abs(trade) * S[:, t]
                shares = new
        payoff = np.abs(S[:, n] - K)
        out = cash + shares * S[:, n] + position * payoff
        return float(out[0]) if single else out


def straddle_price(S, K, T, r, sigma, q=0.0):
    """ATM 스트래들(콜+풋) 블랙-숄즈 가격. 배열을 넣으면 원소별로 계산."""
    S, K, T, sigma = (np.asarray(a, float) for a in (S, K, T, sigma))
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vt
    d2 = d1 - vt
    fs, dk = S * np.exp(-q * T), K * np.exp(-r * T)
    return fs * (norm.cdf(d1) - norm.cdf(-d1)) - dk * (norm.cdf(d2) - norm.cdf(-d2))


def hedged_straddle_pnl(paths, T, r, sigma, q=0.0, cost_bps=0.0, sigma_path=None):
    """롱 ATM 스트래들을 매일 델타헤지했을 때의 만기 손익 — 행마다 만기(T)·IV(sigma)가 다른 여러 진입을 한 번에.

    GammaScalpingEngine.hedged_pnl(position=+1)과 같은 계산이다 (행사가 = 각 경로의 첫 가격).
    paths: (진입 수, n+1), T·sigma: (진입 수,).
    sigma_path: (진입 수, n+1) — 주면 매일 그날의 IV로 델타를 다시 계산한다 (진입 가격은 sigma).
                없으면 진입 IV 하나로 끝까지 헤지 (예전 방식).
    """
    S = np.asarray(paths, float)
    T, sg = np.asarray(T, float), np.asarray(sigma, float)
    sp = None if sigma_path is None else np.asarray(sigma_path, float)
    n, K, dt, cost = S.shape[1] - 1, S[:, 0], T / (S.shape[1] - 1), cost_bps * 1e-4
    shares = -_straddle_delta(S[:, 0], K, T, r, sg, q)
    cash = -straddle_price(S[:, 0], K, T, r, sg, q) - shares * S[:, 0] - cost * np.abs(shares) * S[:, 0]
    for t in range(1, n + 1):
        cash = cash * np.exp(r * dt) + shares * S[:, t - 1] * q * dt
        if t < n:
            new = -_straddle_delta(S[:, t], K, T - t * dt, r, sg if sp is None else sp[:, t], q)
            trade = new - shares
            cash -= trade * S[:, t] + cost * np.abs(trade) * S[:, t]
            shares = new
    return cash + shares * S[:, n] + np.abs(S[:, n] - K)


def overlap_tstat(x, h):
    """하루하루 겹치는 h일 표본(h일 수익률·라벨 등)의 평균이 0인지 검정하는 t값.

    겹침 때문에 생기는 상관은 정확히 h−1일까지라서, 그 자기공분산을 모두 같은 가중으로 더한다 (Hansen-Hodrick).
    흔히 쓰는 Newey-West(Bartlett 가중, 시차 h−1)는 이 경우 분산을 약 1/3 작게 잡아, 몬테카를로에서
    '우연인데 유의(|t| > 1.96)'가 목표 5% 대신 10~13% 나왔다 (Hansen-Hodrick은 5~7%).
    표본이 짧아 분산이 0 이하로 나오면 Newey-West(시차 2h, 항상 0 이상)로 대신한다.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n, h = len(x), max(int(h), 1)
    if n < 2:
        return 0.0
    e = x - x.mean()

    def long_run_var(lags, bartlett):
        s = e @ e / n
        for k in range(1, min(lags, n - 1) + 1):
            s += 2 * (1 - k / (lags + 1) if bartlett else 1.0) * (e[k:] @ e[:-k]) / n
        return s

    s = long_run_var(h - 1, bartlett=False)
    if s <= 0:
        s = long_run_var(2 * h, bartlett=True)
    se = np.sqrt(max(s, 0.0) / n)
    return float(x.mean() / se) if se > 1e-15 else 0.0


def ols(y, X, hac_lags=0):
    """최소제곱 회귀 (상수항 자동 추가) + 이분산·자기상관에 강한 t값.

    y: (n,), X: DataFrame 또는 (n, k). hac_lags: 0이면 White(HC1) 표준오차, >0이면 Newey-West(Bartlett) — 일간 수익률처럼
    변동성이 몰려 다니는 데이터는 보통 t값이 부풀려지므로 이 보정을 쓴다. 결측이 있는 행은 뺀다.
    반환: dict(coef, t: 이름 붙은 Series — 첫 항목 '알파'가 상수항, r2, n).
    """
    names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.shape(X)[1])]
    Xa = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    ya = np.asarray(y, float)
    ok = np.isfinite(ya) & np.isfinite(Xa).all(axis=1)
    Xa, ya = Xa[ok], ya[ok]
    n, k = Xa.shape
    idx = ["알파", *names]
    if n <= k + 1:
        nan = pd.Series(np.nan, index=idx)
        return {"coef": nan, "t": nan.copy(), "r2": np.nan, "n": int(n)}
    XtX_inv = np.linalg.pinv(Xa.T @ Xa)
    b = XtX_inv @ Xa.T @ ya
    e = ya - Xa @ b
    u = Xa * e[:, None]
    S = u.T @ u
    for lag in range(1, min(int(hac_lags), n - 1) + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1 - lag / (hac_lags + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.maximum(np.diag(V), 0))
    tss = np.sum((ya - ya.mean()) ** 2)
    return {"coef": pd.Series(b, index=idx), "t": pd.Series(np.where(se > 0, b / np.where(se > 0, se, 1), np.nan), index=idx),
            "r2": float(1 - e @ e / tss) if tss > 0 else np.nan, "n": int(n)}


_EULER_GAMMA = 0.5772156649015329


def deflated_sharpe(returns, trial_sharpes):
    """Deflated Sharpe Ratio (Bailey·López de Prado 2014) — 여러 설정을 시험해 가장 좋은 것을 골랐을 때 생기는 착시를 보정.

    N개를 시험하면 아무 효과가 없어도 그중 1등의 샤프는 0보다 크게 나온다(SR0 = 우연히 기대되는 1등 샤프).
    고른 설정의 샤프가 SR0보다 충분히 큰지를, 수익률의 비대칭·두꺼운 꼬리와 표본 길이까지 반영해 확률로 낸다.
    returns: 고른 설정의 기간별 수익률. trial_sharpes: 시험한 모든 설정의 같은 기간 단위 샤프(고른 것 포함).
    반환: dict(sr, sr0 (기간 단위), dsr = 진짜 샤프가 0보다 클 확률, n_trials, T). 샤프를 계산할 수 없으면 None.
    시험한 설정들이 서로 비슷하면(상관이 높으면) 샤프의 흩어짐이 작아 SR0도 작아진다 — 그만큼 보정이 약해진다.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3 or r.std(ddof=1) <= 0:
        return None
    sr = r.mean() / r.std(ddof=1)
    s = np.asarray(trial_sharpes, float)
    s = s[np.isfinite(s)]
    N = max(len(s), 1)
    sr0 = (np.sqrt(s.var(ddof=1)) * ((1 - _EULER_GAMMA) * norm.ppf(1 - 1 / N) + _EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e)))
           if N > 1 else 0.0)
    denom = 1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr ** 2
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(max(denom, 1e-12))))
    return {"sr": float(sr), "sr0": float(sr0), "dsr": dsr, "n_trials": int(N), "T": int(T)}


FF_LABELS = {"Mkt-RF": "시장", "SMB": "소형주", "HML": "가치", "RMW": "수익성", "CMA": "보수적 투자", "Mom": "모멘텀"}


def timing_attribution(daily, ff=None, hac_lags=5):
    """보유/현금을 오가는 전략(일간)의 수익이 '타이밍 실력'인지 '알려진 요인에 노출된 것'인지 나눈다.

    daily: DataFrame (index = 수익이 실현된 날짜) — pos(보유 비중), strat(전략 로그수익률, 비용 후), asset(종목 로그수익률).
    ① 종목 자체: 전략 = α + β·종목. β는 평균 노출, α는 '같은 비중으로 계속 들고 있던 것'보다 나은 부분 = 타이밍 실력.
    ② ff(파마-프렌치 일간 요인)가 있으면 — 미국 종목: 현금은 무위험 금리를 번다고 보고 전략과 단순 보유의 초과수익을
       5요인 + 모멘텀에 회귀. 전략의 알파가 단순 보유의 알파보다 크면 그 차이가 요인으로 설명되지 않는 타이밍 효과.
    반환: dict(summary = 분석별 알파(연)·t값·노출·R²·관측일, loadings = 요인별 계수·t값 (ff가 있을 때만)).
    """
    d = daily.dropna()
    s, a = np.expm1(d["strat"]), np.expm1(d["asset"])
    rows, loads = {}, {}
    o = ols(s, a.rename("종목").to_frame(), hac_lags)
    rows["전략 = α + β·종목 (같은 비중 보유와 비교)"] = {"알파(연)": o["coef"]["알파"] * 252, "알파 t값": o["t"]["알파"],
                                                  "노출(β)": o["coef"]["종목"], "R²": o["r2"], "관측일": o["n"]}
    if ff is not None:
        cols = [c for c in FF_LABELS if c in ff]
        f = ff[cols].reindex(d.index)
        rf = ff["RF"].reindex(d.index) if "RF" in ff else 0.0
        for name, y in (("전략 (요인 모형)", s - d["pos"] * rf), ("단순 보유 (요인 모형)", a - rf)):
            o = ols(y, f.rename(columns=FF_LABELS), hac_lags)
            rows[name] = {"알파(연)": o["coef"]["알파"] * 252, "알파 t값": o["t"]["알파"], "노출(β)": o["coef"]["시장"],
                          "R²": o["r2"], "관측일": o["n"]}
            loads[name] = pd.DataFrame({"계수": o["coef"], "t값": o["t"]}).drop(index="알파")
    loadings = pd.concat(loads, axis=1) if loads else None
    return {"summary": pd.DataFrame(rows).T, "loadings": loadings}


def _std_quantile(p, nu):
    """분산이 1이 되도록 표준화한 충격 분포의 p 분위수 (t분포 또는 정규)."""
    return float(student_t.ppf(p, nu) * np.sqrt((nu - 2) / nu)) if np.isfinite(nu) else float(norm.ppf(p))


def var_exception_tests(hits, p):
    """VaR 예외(손실이 VaR를 넘은 날) 검정.

    Kupiec: 예외 비율이 목표(p)와 같은가. Christoffersen: 예외가 몰려서 나오지 않는가(전날 예외 여부와 독립인가).
    종합(조건부 커버리지) = 두 통계량의 합, 자유도 2. p값이 0.05보다 작으면 그 VaR 모델을 기각한다.
    """
    hits = np.asarray(hits, bool)
    n, x = len(hits), int(hits.sum())
    pi = x / n if n else 0.0
    lr_pof = -2 * (xlogy(n - x, 1 - p) + xlogy(x, p) - xlogy(n - x, 1 - pi) - xlogy(x, pi))
    a, b = hits[:-1], hits[1:]
    n00, n01 = int(np.sum(~a & ~b)), int(np.sum(~a & b))
    n10, n11 = int(np.sum(a & ~b)), int(np.sum(a & b))
    p0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pa = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else 0.0
    lr_ind = -2 * (xlogy(n00 + n10, 1 - pa) + xlogy(n01 + n11, pa)
                   - xlogy(n00, 1 - p0) - xlogy(n01, p0) - xlogy(n10, 1 - p1) - xlogy(n11, p1))
    lr_pof, lr_ind = max(float(lr_pof), 0.0), max(float(lr_ind), 0.0)
    return {"관측일": n, "예외 횟수": x, "예외 비율": pi, "목표 비율": p,
            "Kupiec p값": float(chi2.sf(lr_pof, 1)), "Christoffersen p값": float(chi2.sf(lr_ind, 1)),
            "종합 p값": float(chi2.sf(lr_pof + lr_ind, 2))}


def var_backtest(returns, levels=(0.95, 0.99), window=750, refit_every=21, start=None, dist="t", model="garch"):
    """하루 VaR 워크포워드 검증: refit_every일마다 그때까지의 수익률로 GARCH를 다시 적합하고,
    그 사이에는 실제 수익률로 조건부 분산만 갱신하며 매일 다음 날 VaR를 낸다 (앱 화면의 VaR와 같은 평균 0 가정).
    겹치지 않는 하루 단위라 Kupiec·Christoffersen 검정을 그대로 쓸 수 있다.
    """
    r = np.asarray(returns, float)
    n = len(r)
    start = max(250, n - 500) if start is None else start
    hits = {lv: [] for lv in levels}
    for b in range(start, n, refit_every):
        g = fit_vol_model(r[max(0, b - window):b], model, dist)
        zq = {lv: _std_quantile(1 - lv, g.nu) for lv in levels}
        s2 = g.sigma2_next
        for t in range(b, min(b + refit_every, n)):
            for lv in levels:
                hits[lv].append(r[t] < zq[lv] * np.sqrt(s2))
            s2 = g.next_var(s2, r[t] - g.mu)
    rows = {f"하루 VaR {lv:.0%}": var_exception_tests(hits[lv], 1 - lv) for lv in levels}
    return pd.DataFrame(rows).T


def var_es(log_returns, level=0.95):
    """VaR·ES (양수 = 손실률). log_returns: 시뮬레이션 또는 과거의 기간 로그수익률들.

    VaR: 이 확률(level)로 손실이 이보다 작다. ES: 그 밖의 최악 (1−level) 경우의 평균 손실.
    """
    simple = np.expm1(np.asarray(log_returns, float))
    q = np.quantile(simple, 1 - level)
    return float(-q), float(-simple[simple <= q].mean())


def max_drawdown(equity):
    """자산 곡선(시작=1, 곱셈 누적)의 최대 낙폭. 음수(예: -0.35 = 고점 대비 35% 하락)."""
    e = np.asarray(equity, float)
    return float(np.min(e / np.maximum.accumulate(e) - 1)) if len(e) else 0.0


VOL_STRATEGIES = {"signal": "GARCH 신호대로", "short": "항상 변동성 매도", "long": "항상 변동성 매수"}


def vol_signal_backtest(close, iv, vrp=0.0, threshold=0.02, h=21, cost_bps=0.0, r=0.0, q=0.0,
                        fit_window=750, min_history=500, dist="t", iv_gap=0.0, option_spread=0.0,
                        offset=0, forecasts=None, lag=1, daily_iv_hedge=True):
    """과거 IV(예: VIX)로 '변동성 신호대로 ATM 스트래들을 사고팔며 매일 델타헤지'한 성적 (워크포워드).

    각 진입일 t: t까지의 수익률로만 GARCH 적합 → h일 평균 변동성 예측, 그날 종가의 IV와 비교해 신호.
    t 종가에 IV 가격으로 스트래들 진입 → h거래일 동안 매일 종가에 델타헤지 → h일 뒤 만기 정산.
    진입은 h일 간격(겹치지 않음)이라 표본이 서로 독립에 가깝다.

    iv_gap: 변동성 지수와 실제 ATM IV의 차이(소수). VIX는 외가격 옵션까지 섞여 ATM IV보다 보통 0.01~0.03 높다
            → 신호·가격에 쓰는 IV = 지수 − iv_gap.
    option_spread: 옵션 체결 비용 (스트래들 가격 대비, 진입할 때 한 번 — 만기까지 들고 가 자동 정산되므로 청산 거래 없음).
    offset: 첫 진입을 며칠 늦출지 (진입일에 따라 결과가 얼마나 흔들리는지 볼 때).
    forecasts: {판단일: GARCH 예측} — 같은 판단일로 설정만 바꿔 다시 돌릴 때 GARCH 재적합을 건너뛴다.
    lag: 판단(그날 종가까지의 정보)과 실제 진입 사이 거래일 수. 1(기본) = 다음 날 종가에 진입,
         0 = 판단한 날 종가에 바로 진입(그 종가는 장이 끝나야 알 수 있으므로 낙관적).
    daily_iv_hedge: True(기본)면 매일 그날의 IV(지수 − iv_gap)로 델타를 다시 계산, False면 진입 IV 하나로 헤지.
    반환: 판단일별 표 — long은 롱 스트래들 헤지 손익(비용 전), cost는 헤지·옵션 체결 비용, premium은 스트래들 가격
          (모두 진입일 주가 대비), iv는 판단에 쓴 IV, iv_entry는 진입 가격에 쓴 IV, iv_index는 판단일의 지수 값.
    """
    df = pd.concat({"px": close, "iv": iv}, axis=1, join="inner").dropna()
    px, ivs, dates = df["px"].values, df["iv"].values, df.index
    lr = np.diff(np.log(px))                     # lr[k] = k → k+1일 로그수익률
    ii = np.arange(min_history + offset, len(px) - h - lag, h)          # 판단일
    if len(ii) == 0:
        return pd.DataFrame()
    jj = ii + lag                                                        # 실제 진입일
    f = np.array([forecasts[dates[i]] if forecasts is not None and dates[i] in forecasts
                  else GarchVolForecaster(dist=dist).fit(lr[max(0, i - fit_window):i]).forecast_vol(h) for i in ii])
    sig_dec = np.maximum(ivs[ii] - iv_gap, 0.01)                         # 판단에 쓰는 IV (판단일까지의 정보)
    sigma = np.maximum(ivs[jj] - iv_gap, 0.01)                           # 진입 가격에 쓰는 IV
    T = np.maximum(np.asarray((dates[jj + h] - dates[jj]).days, float), 1.0) / 365
    steps = jj[:, None] + np.arange(h + 1)
    paths = px[steps]                                                    # 진입별 h+1일 가격 경로 — 모든 진입을 한 번에 계산
    sig_path = np.maximum(ivs[steps] - iv_gap, 0.01) if daily_iv_hedge else None
    s0 = paths[:, 0]
    gross = hedged_straddle_pnl(paths, T, r, sigma, q, 0.0, sig_path)
    premium = straddle_price(s0, s0, T, r, sigma, q)
    cost = gross - hedged_straddle_pnl(paths, T, r, sigma, q, cost_bps, sig_path) + option_spread * premium   # 방향과 무관
    logic = VolatilityTradingLogic(vrp, threshold)
    out = pd.DataFrame({"garch": f, "iv": sig_dec, "iv_entry": sigma, "iv_index": ivs[ii],
                        "realized": np.sqrt(np.mean(lr[jj[:, None] + np.arange(h)] ** 2, axis=1) * 252),
                        "signal": [logic.generate_signal(a, b) for a, b in zip(f, sig_dec)],
                        "long": gross / s0, "cost": cost / s0, "premium": premium / s0,
                        "underlying": paths[:, -1] / s0 - 1}, index=dates[ii])
    out.index.name = "date"
    return out


def _strategy_stats(pos, bt, h=21):
    """진입별 포지션(+1 매수 / −1 매도 / 0 쉼) → 성적 요약과 자산 곡선 (주가 1주당 스트래들 1개 기준)."""
    ret = pos * bt["long"] - pos.abs() * bt["cost"]
    traded = pos != 0
    rt, n = ret[traded], int(traded.sum())
    return {"거래 수": n, "승률": (rt > 0).mean() if n else np.nan,
            "평균 손익(주가 대비)": rt.mean() if n else np.nan,
            "평균 손익(프리미엄 대비)": (rt / bt["premium"][traded]).mean() if n else np.nan,
            "t값": rt.mean() / rt.std() * np.sqrt(n) if n > 2 and rt.std() > 0 else np.nan,
            "샤프(연)": ret.mean() / ret.std() * np.sqrt(252 / h) if ret.std() > 0 else np.nan,
            "최대 낙폭": max_drawdown((1 + ret).cumprod()), "최악 1회": rt.min() if n else np.nan}, (1 + ret).cumprod()


def summarize_vol_backtest(bt, h=21):
    """세 전략(신호대로 / 항상 매도 / 항상 매수)의 거래별 손익 요약 + 자산 곡선."""
    rows, curves = {}, {}
    for key, name in VOL_STRATEGIES.items():
        pos = bt["signal"] if key == "signal" else pd.Series(-1 if key == "short" else 1, index=bt.index)
        stats, curves[name] = _strategy_stats(pos, bt, h)
        rows[key] = {"전략": name, **stats}
    return pd.DataFrame(rows).T, pd.DataFrame(curves)


def vol_backtest_robustness(close, iv, base, vrp=0.0, threshold=0.02, h=21, cost_bps=0.0, r=0.0, q=0.0,
                            iv_gap=0.0, option_spread=0.0, gaps=(0.0, 0.01, 0.02, 0.03), offsets=(0, 4, 8, 12, 16),
                            lag=1, daily_iv_hedge=True):
    """현실성 점검 (base = iv_gap·진입일 0으로 이미 돌린 vol_signal_backtest 결과).

    (차이 × 진입일) 모든 조합을 돌린다. GARCH 예측은 진입일마다 한 번만 적합하고 차이별로 재사용한다.
    by_gap: IV = 지수 − gap 으로 바꿔 가며 → 'VIX와 실제 ATM IV 차이'를 얼마로 보느냐에 따라 결과가 어떻게 바뀌는지.
    by_offset: 첫 진입일을 0~16거래일 늦춰 가며 → 진입일 운에 따라 결과가 얼마나 흔들리는지.
    grid: {(차이, 진입일 늦춤): 백테스트 표} — short_only_robustness에 넘긴다.
    """
    kw = dict(vrp=vrp, threshold=threshold, h=h, cost_bps=cost_bps, r=r, q=q, option_spread=option_spread,
              lag=lag, daily_iv_hedge=daily_iv_hedge)
    grid = {(iv_gap, 0): base}
    for o in offsets:
        if (iv_gap, o) not in grid:
            grid[(iv_gap, o)] = vol_signal_backtest(close, iv, iv_gap=iv_gap, offset=o, **kw)
        cache = grid[(iv_gap, o)]["garch"].to_dict()
        for g in gaps:
            if (g, o) not in grid:
                grid[(g, o)] = vol_signal_backtest(close, iv, iv_gap=g, offset=o, forecasts=cache, **kw)

    def row(bt):
        tab = summarize_vol_backtest(bt, h)[0]
        sh, sg = tab.loc["short"], tab.loc["signal"]
        return {"진입 수": len(bt), "첫 진입일": bt.index[0].date(),
                "항상 매도 평균 손익(주가 대비)": sh["평균 손익(주가 대비)"], "항상 매도 t값": sh["t값"],
                "항상 매도 샤프": sh["샤프(연)"], "항상 매도 최대 낙폭": sh["최대 낙폭"],
                "신호대로 샤프": sg["샤프(연)"], "신호대로 t값": sg["t값"]}

    by_gap = pd.DataFrame({f"{g * 100:g}%p": row(grid[(g, 0)]) for g in gaps}).T
    by_off = pd.DataFrame({f"{o}거래일 늦게": row(grid[(iv_gap, o)]) for o in offsets}).T
    grid = {k: v for k, v in grid.items() if k[0] in gaps}
    return {"by_gap": by_gap, "by_offset": by_off, "grid": grid}


def add_combined_forecast(bt, min_train=24):
    """IV를 예측에 같이 쓰는 두 가지 예측을 워크포워드로 추가한다.

    combo: 실제 변동성 ≈ a + b·GARCH + c·IV 를 '이미 결과가 나온 과거 진입들'로만 최소제곱 추정해 예측.
    iv_adj: IV − (과거 진입들의 평균 VRP).
    진입이 h일 간격이라 직전 진입의 결과(이후 h일 실제 변동성)는 다음 진입일 종가에 확정된다.
    """
    out = bt.copy()
    g, iv, rv = bt["garch"].values, bt["iv"].values, bt["realized"].values
    combo, adj = np.full(len(bt), np.nan), np.full(len(bt), np.nan)
    for k in range(min_train, len(bt)):
        coef = np.linalg.lstsq(np.column_stack([np.ones(k), g[:k], iv[:k]]), rv[:k], rcond=None)[0]
        combo[k] = max(coef @ np.array([1.0, g[k], iv[k]]), 0.01)
        adj[k] = iv[k] - np.mean(iv[:k] - rv[:k])
    out["combo"], out["iv_adj"] = combo, adj
    return out


def forecast_errors(bt, iv_name="IV"):
    """같은 진입들(결합 예측이 있는 구간)에서 각 예측의 실제 변동성 대비 오차. 모두 표본 밖 예측이다.

    'GARCH 대비 개선 t값': (GARCH 오차² − 이 예측 오차²)의 평균이 0보다 큰지 (디볼드-마리아노 검정, 2를 넘으면 우연이 아님).
    """
    d = bt.dropna(subset=["combo"])
    labels = {"garch": "GARCH", "iv": f"{iv_name} 그대로", "iv_adj": f"{iv_name} − 과거 평균 VRP",
              "combo": f"GARCH + {iv_name} 결합"}
    e_g = d["garch"] - d["realized"]
    rows = {}
    for c, name in labels.items():
        e = d[c] - d["realized"]
        diff = e_g**2 - e**2
        rows[name] = {"예측 오차(RMSE)": float(np.sqrt(np.mean(e**2))), "평균 치우침": float(e.mean()),
                      "GARCH 대비 개선 t값": (float(diff.mean() / diff.std() * np.sqrt(len(d)))
                                          if c != "garch" and diff.std() > 0 else np.nan)}
    return pd.DataFrame(rows).T, len(d)


SHORT_CUTOFFS = (-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


def short_rules(cutoffs=SHORT_CUTOFFS, iv_name="IV"):
    """'매도만' 후보 규칙들 {이름: 진입별 포지션을 돌려주는 함수}. −1 = 그달 매도, 0 = 쉼."""
    rules = {"항상 매도": lambda x: pd.Series(-1.0, index=x.index)}
    for f, fname in (("garch", "GARCH"), ("combo", f"GARCH+{iv_name} 결합")):
        for c in cutoffs:
            rules[f"매도만 · {fname} 예측 · 기준 {c * 100:+.0f}%p"] = (
                lambda x, f=f, c=c: -((x["iv"] - x[f]) > c).astype(float))
    return rules


def _split_half(bt, min_rows=24):
    """결합 예측이 있는 진입들을 앞 절반(규칙 고르기) / 뒤 절반(검증)으로. 너무 적으면 None."""
    d = bt.dropna(subset=["combo"])
    if len(d) < min_rows:
        return None
    half = len(d) // 2
    return d.iloc[:half], d.iloc[half:]


def short_only_split(bt, h=21, cutoffs=SHORT_CUTOFFS, iv_name="IV", min_rows=24):
    """'매도만' 규칙을 앞 절반에서 고르고(앞 기간 샤프가 가장 높은 것) 뒤 절반에서 검증한다.

    후보: 항상 매도, 그리고 (예측 = GARCH 또는 GARCH+IV 결합) × (IV − 예측 > 기준이면 그달 매도, 아니면 쉼).
    뒤 절반은 규칙을 고를 때 전혀 쓰지 않으므로 '처음 보는 데이터'에서의 성적이다.
    bt에는 add_combined_forecast의 combo 열이 있어야 한다. 표본이 min_rows보다 적으면 None.
    """
    split = _split_half(bt, min_rows)
    if split is None:
        return None
    train, test = split
    rules = short_rules(cutoffs, iv_name)
    cand = {name: _strategy_stats(rule(train), train, h)[0] for name, rule in rules.items()}
    cand = pd.DataFrame({k: {"앞 기간 샤프": v["샤프(연)"], "앞 기간 거래 수": v["거래 수"],
                             "앞 기간 평균 손익(주가 대비)": v["평균 손익(주가 대비)"]} for k, v in cand.items()}).T
    chosen = cand["앞 기간 샤프"].astype(float).idxmax()
    rows, curves = {}, {}
    for name, pos in (("고른 규칙", rules[chosen](test)), ("항상 매도", rules["항상 매도"](test)),
                      ("GARCH 신호대로 (사이드바 설정)", test["signal"].astype(float))):
        rows[name], curves[name] = _strategy_stats(pos, test, h)
    # 앞 기간에서 후보 여럿 중 1등을 골랐으므로, 그 샤프가 '시험 횟수만큼의 운'을 넘는지 (Deflated Sharpe)
    pos = rules[chosen](train)
    k = np.sqrt(252 / h)
    dsr = deflated_sharpe((pos * train["long"] - pos.abs() * train["cost"]).values, cand["앞 기간 샤프"].astype(float) / k)
    if dsr is not None:
        dsr.update(sr_ann=dsr["sr"] * k, sr0_ann=dsr["sr0"] * k)
    return {"candidates": cand, "chosen": chosen, "test": pd.DataFrame(rows).T, "curves": pd.DataFrame(curves),
            "train_period": (train.index[0], train.index[-1]), "test_period": (test.index[0], test.index[-1]), "dsr": dsr}


def market_exposure(bt, h=21):
    """변동성 매매 손익이 '시장 방향'에 얼마나 묶여 있는지: 거래별 손익 = α + β·(진입~만기 주가 수익률).

    델타헤지를 해도 크게 움직인 달엔 감마 손익이 생긴다. 특히 매도는 폭락 달에 크게 잃으므로,
    이익이 실은 '시장 급락 위험을 떠안은 대가'인지 본다. 하락 상위 10% 달의 평균 손익도 함께 낸다.
    """
    m = bt["underlying"]
    worst = m <= m.quantile(0.10)
    rows = {}
    for name, pos in (("항상 변동성 매도", pd.Series(-1.0, index=bt.index)), ("GARCH 신호대로", bt["signal"].astype(float))):
        ret = pos * bt["long"] - pos.abs() * bt["cost"]
        o = ols(ret, m.rename("주가 수익률").to_frame())
        rows[name] = {"알파(거래당)": o["coef"]["알파"], "알파 t값": o["t"]["알파"], "시장 노출(β)": o["coef"]["주가 수익률"],
                      "β t값": o["t"]["주가 수익률"], "R²": o["r2"], "하락 상위 10% 달 평균 손익": ret[worst].mean(),
                      "나머지 달 평균 손익": ret[~worst].mean()}
    return pd.DataFrame(rows).T


def short_only_robustness(grid, rule=None, h=21, iv_name="IV"):
    """'매도만' 검증이 (차이 × 진입일) 조합마다 유지되는지.

    다시 고르기: 각 조합에서 앞 절반으로 규칙을 처음부터 다시 고르고 뒤 절반에서 평가 → 고르는 '방법'이 믿을 만한지.
    이 규칙 그대로(rule): 이미 고른 규칙을 각 조합의 뒤 절반에 그대로 적용 → 그 규칙이 가정 변화에 버티는지.
      (규칙을 고른 앞 기간과 겹치지 않는 뒤 절반만 쓰지만, 조합끼리 뒤 기간이 거의 같아서 서로 독립적인 증거는 아니다.)
    """
    rules = short_rules(iv_name=iv_name)
    rows = []
    for (g, o), bt in sorted(grid.items()):
        cb = add_combined_forecast(bt)
        so = short_only_split(cb, h=h, iv_name=iv_name)
        if so is None:
            continue
        test = _split_half(cb)[1]
        al, ch = so["test"].loc["항상 매도"], so["test"].loc["고른 규칙"]
        row = {"차이": f"{g * 100:g}%p", "진입": f"{o}거래일 늦게", "다시 고른 규칙": so["chosen"],
               "다시 고른 규칙 샤프": ch["샤프(연)"], "다시 고른 규칙 t값": ch["t값"], "항상 매도 샤프": al["샤프(연)"]}
        if rule is not None and rule in rules:
            fx = _strategy_stats(rules[rule](test), test, h)[0]
            row.update({"이 규칙 샤프": fx["샤프(연)"], "이 규칙 t값": fx["t값"], "이 규칙 거래 수": fx["거래 수"]})
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_gbm_paths(S0, vol, r, T, n_steps, n_paths, rng):
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    inc = (r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1))
