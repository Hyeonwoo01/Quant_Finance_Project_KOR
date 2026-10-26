"""주가 방향(상승/하락) 예측: 가격·거시 특징 + 여러 후보 모델 + 워크포워드 자동 선택.

정확도 개선의 원칙
1. 정보 추가: 종목 가격 특징 외에 시장 전체 상태(VIX·기간구조·금리곡선·신용·지수 추세).
2. 후보 모델 여러 개: 로지스틱(가격/전체), 그래디언트 부스팅, 앙상블, 그리고 '기본 확률(base rate)'.
3. 자동 선택은 '그 시점까지 이미 채점된 과거 예측 성적'만 보고 고른다 (중첩 워크포워드).
   → 결과 표에서 사후에 제일 좋은 모델을 고르는 것과 달리 과적합된 숫자가 아니다.
   → 아무 모델도 기본 확률보다 못하면 기본 확률을 고르므로 '항상 상승'보다 크게 나빠지지 않는다.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import alt_data
import market_data
from kr_costs import trade_cost
from quant_engine import deflated_sharpe, max_drawdown, overlap_tstat

# 이 환경(sklearn 1.9 + Python 3.15)에서 학습마다 나오는 무해한 경고 — 로그가 넘치지 않게 끈다
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

PRICE_FEATURES = {
    "mom_5": "1주 모멘텀", "mom_21": "1개월 모멘텀", "mom_63": "3개월 모멘텀", "mom_252": "1년 모멘텀",
    "vol_21": "1개월 변동성", "vol_ratio": "변동성 변화(21/63일)",
    "ma50_gap": "50일선 이격도", "ma200_gap": "200일선 이격도", "rsi14": "RSI(14)",
}
MACRO_FEATURES = {
    "vix": "VIX 수준", "vix_term": "VIX 기간구조(VIX/VIX3M)", "vix_chg": "VIX 1개월 변화",
    "curve": "금리곡선(10년−3개월)", "credit": "신용 스프레드 변화(HYG/IEF)",
    "mkt_mom_63": "지수 3개월 모멘텀", "mkt_ma200_gap": "지수 200일선 이격도", "rel_strength": "지수 대비 상대강도",
    "skew_idx": "CBOE SKEW (꼬리위험 수요)", "skew_chg": "SKEW 1개월 변화", "vvix": "VVIX (변동성의 변동성)",
    # 한국 종목 전용
    "fx_chg_21": "원/달러 1개월 변화(원화 약세 +)", "fx_vol_21": "원/달러 1개월 변동성",
    "kq_ks_rel": "코스닥−코스피 1개월 상대강도(위험선호)",
    "kr_curve": "국고채 금리곡선(10년−3년)", "kr_credit": "회사채 신용스프레드(AA−−국고3년)",
    "kr3y_chg": "국고3년 금리 3개월 변화", "kr_credit_chg": "신용스프레드 1개월 변화",
}
ALT_FEATURES = alt_data.ALT_FEATURES
FEATURE_LABELS = {**PRICE_FEATURES, **MACRO_FEATURES, **ALT_FEATURES}
MODEL_LABELS = {"base": "기본 확률 (항상 과거 상승비율)", "logit_price": "로지스틱 · 가격 특징",
                "logit_alt": "로지스틱 · 실적/애널리스트/수급만", "logit_all": "로지스틱 · 전체 특징",
                "gbm": "그래디언트 부스팅 · 전체", "ensemble": "앙상블 (로지스틱+부스팅)"}


# ---------------- 특징 ----------------
def resolve_lag(lag_macro, ticker):
    """거시 지표를 하루 늦출지. 명시하지 않으면 티커로 판단하고, 티커도 모르면 안전하게 늦춘다.

    한국 장(15:30 마감)은 같은 날짜 미국 장보다 먼저 끝나므로, 같은 날짜의 미국·글로벌 값은 아직 모르는 정보."""
    if lag_macro is not None:
        return bool(lag_macro)
    if ticker:
        return not market_data.is_us_ticker(ticker)
    return True


def build_features(close, macro=None, lag_macro=None, alt=None, ticker=None):
    """t일 종가까지의 정보만 쓰는 특징들 (미래 정보 없음).

    macro: market_data.load_macro()의 DataFrame. 거시 지표를 하루 늦출지는 resolve_lag()가 정한다
    (lag_macro를 주면 그대로, 아니면 ticker로 판단, 둘 다 없으면 늦춤).
    alt: alt_data.build_alt_features()의 DataFrame (이미 공개 시점에 맞춰져 있음).
    """
    lag_macro = resolve_lag(lag_macro, ticker)
    lc = np.log(close)
    r = lc.diff()
    f = pd.DataFrame(index=close.index)
    for k in (5, 21, 63, 252):
        f[f"mom_{k}"] = lc - lc.shift(k)
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_ratio"] = r.rolling(21).std() / r.rolling(63).std()
    f["ma50_gap"] = lc - lc.rolling(50).mean()
    f["ma200_gap"] = lc - lc.rolling(200).mean()
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    if macro is not None and not macro.empty:
        m = pd.DataFrame(index=macro.index)
        m["vix"] = np.log(macro["vix"])
        m["vix_term"] = macro["vix"] / macro["vix3m"]
        m["vix_chg"] = np.log(macro["vix"] / macro["vix"].shift(21))
        m["curve"] = macro["tnx"] - macro["irx"]
        cr = np.log(macro["hyg"] / macro["ief"])
        m["credit"] = cr - cr.shift(63)
        lm = np.log(macro["market"])
        m["mkt_mom_63"] = lm - lm.shift(63)
        m["mkt_ma200_gap"] = lm - lm.rolling(200).mean()
        if "skew" in macro:          # 옵션 시장 전체의 꼬리위험 가격 (종목별 과거 풋/콜·스큐는 무료로 없음)
            m["skew_idx"] = macro["skew"]
            m["skew_chg"] = macro["skew"] - macro["skew"].shift(21)
            m["vvix"] = np.log(macro["vvix"])
        if "usdkrw" in macro:        # 외국인 수급은 환율과 함께 움직이는 경향
            lfx = np.log(macro["usdkrw"])
            m["fx_chg_21"] = lfx - lfx.shift(21)
            m["fx_vol_21"] = lfx.diff().rolling(21).std() * np.sqrt(252)
        if "kosdaq" in macro and "kospi" in macro:
            lq, lk = np.log(macro["kosdaq"]), np.log(macro["kospi"])
            m["kq_ks_rel"] = (lq - lq.shift(21)) - (lk - lk.shift(21))
        if "kr3y" in macro and macro["kr3y"].notna().any():      # 한국은행 ECOS (키가 있을 때)
            m["kr_curve"] = macro["kr10y"] - macro["kr3y"]
            m["kr_credit"] = macro["krAA"] - macro["kr3y"]
            m["kr3y_chg"] = macro["kr3y"] - macro["kr3y"].shift(63)
            m["kr_credit_chg"] = m["kr_credit"] - m["kr_credit"].shift(21)
        if lag_macro:
            m = m.shift(1)
        m = m.reindex(f.index, method="ffill", limit=5)
        f = f.join(m)
        f["rel_strength"] = f["mom_63"] - f["mkt_mom_63"]
    if alt is not None and not alt.empty:
        f = f.join(alt.reindex(f.index))
    f = f.replace([np.inf, -np.inf], np.nan)
    # 데이터가 끝까지 비어 있는 열(이 종목엔 해당 없음)은 버리고, 나머지는 모든 값이 있는 날만 쓴다
    return f.loc[:, f.notna().any()].dropna()


def forward_log_return(close, h):
    """t일 → t+h일 로그수익률 (마지막 h일은 아직 모름 → NaN)."""
    return np.log(close.shift(-h) / close)


# ---------------- 모델 ----------------
class LogisticDirectionModel:
    """표준화 + L2 정규화 로지스틱 회귀."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.mu = self.sd = self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        n, k = Z.shape

        def loss(p):
            w, b = p[:k], p[k]
            z = Z @ w + b
            nll = np.sum(np.logaddexp(0, z) - y * z) / n + 0.5 * self.l2 * w @ w / n
            pr = expit(z)
            g = np.append(Z.T @ (pr - y) / n + self.l2 * w / n, np.mean(pr - y))
            return nll, g

        res = minimize(loss, np.zeros(k + 1), jac=True, method="L-BFGS-B")
        self.w, self.b = res.x[:k], res.x[k]
        return self

    def standardize(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def predict_proba(self, X):
        return expit(self.standardize(X) @ self.w + self.b)

    def contributions(self, x_row):
        """각 특징이 현재 예측에 기여한 정도(로그오즈 단위)."""
        return self.standardize(np.atleast_2d(x_row))[0] * self.w


class _Base:
    def fit(self, X, y):
        self.p = float(np.clip(np.mean(y), 0.01, 0.99))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class _Logit:
    def __init__(self, cols, l2):
        self.cols, self.m = cols, LogisticDirectionModel(l2)

    def fit(self, X, y):
        self.m.fit(X[:, self.cols], y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, self.cols])


class _GBM:
    def fit(self, X, y):
        # 겹치는 h일 라벨 때문에 표본이 실제보다 적다 → 얕은 트리 + 큰 잎 + 강한 정규화로 과적합 억제
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                                min_samples_leaf=150, l2_regularization=1.0, random_state=0)
        self.m.fit(X, y.astype(int))
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class _Ensemble:
    def __init__(self, parts):
        self.parts = parts

    def fit(self, X, y):
        for p in self.parts:
            p.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([p.predict_proba(X) for p in self.parts], axis=0)


def make_candidates(columns, light=False):
    """후보 모델들. light=True면 무거운 부스팅·앙상블을 빼고 기본 확률·로지스틱만 (빠른 점검용)."""
    price_idx = [i for i, c in enumerate(columns) if c in PRICE_FEATURES]
    alt_idx = [i for i, c in enumerate(columns) if c in ALT_FEATURES]
    all_idx = list(range(len(columns)))
    cands = {"base": _Base(), "logit_price": _Logit(price_idx, 1.0)}
    if alt_idx:                                 # 새 데이터만으로 신호가 있는지 따로 확인
        cands["logit_alt"] = _Logit(alt_idx, 1.0)
    if len(all_idx) > len(price_idx):           # 거시/대체 데이터가 있을 때만
        cands["logit_all"] = _Logit(all_idx, 10.0)
    if light:
        return cands
    cands["gbm"] = _GBM()
    cands["ensemble"] = _Ensemble([_Logit(all_idx, 10.0), _GBM()])
    return cands


def fit_predict_all(cands, X_train, y_train, X_new):
    """모든 후보를 학습해 X_new의 상승 확률을 낸다.

    앙상블은 '로지스틱(전체 특징, l2=10)'과 '부스팅'의 평균이고, 부스팅 설정이 'gbm' 후보와 똑같으므로
    부스팅을 다시 학습하지 않고 그 예측을 재사용한다 ('로지스틱·전체' 후보가 있으면 그것도 재사용).
    결과는 그대로이고 가장 무거운 부스팅 학습이 절반으로 줄어든다.
    """
    out = {}
    for k, m in cands.items():
        if k == "ensemble" and "gbm" in out:
            lg = out["logit_all"] if "logit_all" in out else m.parts[0].fit(X_train, y_train).predict_proba(X_new)
            out[k] = (lg + out["gbm"]) / 2
        else:
            out[k] = m.fit(X_train, y_train).predict_proba(X_new)
    return out


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------- 워크포워드 ----------------
def walk_forward(close, h, macro=None, lag_macro=None, alt=None, min_train=500, refit_every=21,
                 select_window=756, min_scored=252, ticker=None, light=False):
    """각 시점 t에서 't까지 알 수 있던 데이터'로만 모든 후보를 학습하고 예측한다.

    'auto' 열: 블록 시작 t에서, 결과가 이미 확정된(행 ≤ t−h) 과거 예측들 중 최근 select_window개의
    로그손실이 가장 낮은 후보를 고른다. 채점된 예측이 min_scored개 미만이면 기본 확률을 쓴다.
    """
    feat = build_features(close, macro, lag_macro, alt, ticker)
    fwd = forward_log_return(close, h).reindex(feat.index)
    X, y = feat.values, (fwd.values > 0).astype(float)
    n = len(feat)
    names = list(make_candidates(feat.columns, light))
    probs = {k: np.full(n, np.nan) for k in names}
    auto, chosen = np.full(n, np.nan), np.full(n, None, dtype=object)

    for t in range(min_train + h, n, refit_every):
        end = min(t + refit_every, n)
        tr = slice(0, t - h + 1)
        for k, p in fit_predict_all(make_candidates(feat.columns, light), X[tr], y[tr], X[t:end]).items():
            probs[k][t:end] = p

        known = np.arange(n) <= t - h                       # 결과가 확정된 행만 채점
        scored = known & ~np.isnan(probs["base"])
        idx = np.where(scored)[0][-select_window:]
        if len(idx) >= min_scored:
            pick = min(names, key=lambda k: _logloss(probs[k][idx], y[idx]))
        else:
            pick = "base"
        auto[t:end], chosen[t:end] = probs[pick][t:end], pick

    out = pd.DataFrame({**{k: v for k, v in probs.items()}, "auto": auto, "chosen": chosen,
                        "fwd": fwd.values}, index=feat.index)
    return out.dropna(subset=["auto"]), list(feat.columns)


def score(prob, fwd, h):
    known = fwd.notna() & prob.notna()
    p, up = prob[known].values, (fwd[known].values > 0).astype(float)
    base = float(up.mean())
    correct = ((p > 0.5) == up).astype(float)
    hit = float(correct.mean())
    n_eff = max(len(p) / h, 1.0)                     # h일 수익률이 겹치므로 실질 독립 표본 수 (대략)
    # z: 날마다 (모델 적중 − '항상 오른다' 적중)의 평균이 0보다 큰가. h일 라벨이 겹쳐 서로 상관되므로
    # 겹침을 반영한 표준오차(Hansen-Hodrick)를 쓴다. 모델이 늘 '오른다'고 하면 차이가 0이라 z = 0.
    z = overlap_tstat(correct - up, h)
    brier = float(np.mean((p - up) ** 2))
    brier_base = float(np.mean((base - up) ** 2))
    try:
        auc = float(roc_auc_score(up, p)) if 0 < up.sum() < len(up) and np.ptp(p) > 0 else 0.5
    except ValueError:
        auc = 0.5
    return {"hit": hit, "base": base, "z": float(z), "auc": auc, "brier": brier, "brier_base": brier_base,
            "skill": 1 - brier / brier_base if brier_base > 0 else 0.0, "n": int(len(p)), "n_eff": n_eff}


def evaluate(wf, h, daily_returns, cost_bps=0.0, threshold=0.5, lag=1, sell_tax=0.0):
    """모든 후보와 자동 선택을 채점 + 자동 선택의 '상승 확률 > threshold면 보유' 전략 vs 단순 보유.

    lag: t일 종가까지 보고 판단한 뒤 실제로 사고파는 시점까지의 거래일 수.
         1(기본) = 다음 날 종가에 체결 → t+1일 종가부터의 수익률을 얻음. 0 = 판단한 그 종가에 체결(낙관적).
    cost_bps: 사고팔 때 각각 내는 수수료·체결 비용. sell_tax: 팔 때만 내는 증권거래세 (숫자 또는 날짜별 Series, kr_costs).
    """
    cols = [c for c in MODEL_LABELS if c in wf] + ["auto"]
    table = {c: score(wf[c], wf["fwd"], h) for c in cols}

    pos = (wf["auto"] > threshold).astype(float)
    nxt = daily_returns.shift(-(1 + lag)).reindex(wf.index).fillna(0.0)    # t일 판단 → t+lag일 종가 체결 이후 수익률
    strat = pos * nxt - trade_cost(pos, cost_bps, sell_tax, lag)        # 비용은 체결 뒤 첫 수익과 같은 행, 세율은 체결일 기준
    equity = pd.DataFrame({"전략": np.exp(strat.cumsum()), "단순 보유": np.exp(nxt.cumsum())}, index=wf.index)
    ann = lambda x: (x.mean() * 252, x.std() * np.sqrt(252))
    (sm, ss), (bm, bs) = ann(strat), ann(nxt)
    # 요인 분석용: 수익이 실제로 실현된 날짜(판단일 + 1 + lag) 기준 일간 표. 아직 실현 안 된 마지막 날들은 뺀다
    paid = pd.Series(daily_returns.index, index=daily_returns.index).shift(-(1 + lag)).reindex(wf.index)
    daily = pd.DataFrame({"pos": pos.values, "strat": strat.values, "asset": nxt.values}, index=paid.values)[paid.notna().values]
    return {**table["auto"], "table": table, "equity": equity, "daily": daily, "time_in_market": float(pos.mean()),
            "sharpe": sm / ss if ss > 0 else 0.0, "sharpe_bh": bm / bs if bs > 0 else 0.0,
            "mdd": max_drawdown(equity["전략"]), "mdd_bh": max_drawdown(equity["단순 보유"]),
            "chosen_share": wf["chosen"].value_counts(normalize=True)}


# ---------------- 가정 바꿔 점검 ----------------
def reselect(wf, h, select_window=756, min_scored=252, refit_every=21):
    """저장된 후보별 예측으로 '자동 선택'만 다시 한다 (재학습 없음).

    walk_forward와 같은 블록(refit_every)·같은 규칙: 블록 시작 t에서 결과가 확정된(행 ≤ t−h) 최근 select_window개
    예측의 로그손실이 가장 낮은 후보. 채점된 예측이 min_scored개 미만이면 기본 확률.
    """
    names = [c for c in MODEL_LABELS if c in wf]
    P = {k: wf[k].values for k in names}
    y = (wf["fwd"].values > 0).astype(float)
    n = len(wf)
    auto, chosen = np.full(n, np.nan), np.full(n, None, dtype=object)
    for b in range(0, n, refit_every):
        end = min(b + refit_every, n)
        idx = np.arange(0, max(b - h + 1, 0))[-select_window:]
        pick = min(names, key=lambda k: _logloss(P[k][idx], y[idx])) if len(idx) >= min_scored else "base"
        auto[b:end], chosen[b:end] = P[pick][b:end], pick
    out = wf.copy()
    out["auto"], out["chosen"] = auto, chosen
    return out


def _robust_row(ev):
    return {"적중률": ev["hit"], "'항상 오른다' 대비": ev["hit"] - ev["base"], "확률 정확도 개선": ev["skill"],
            "z": ev["z"], "검증 통과": bool(ev["skill"] > 0 and ev["z"] > 2),
            "전략 샤프": ev["sharpe"], "단순 보유 샤프": ev["sharpe_bh"], "전략 최대 낙폭": ev["mdd"],
            "보유 비중": ev["time_in_market"]}


def robustness(wf, h, daily_returns, cost_bps=0.0, windows=(252, 504, 756, 1260),
               thresholds=(0.45, 0.5, 0.55), costs=(1.0, 5.0, 10.0, 20.0), n_periods=3, lag=1, sell_tax=0.0):
    """방향 예측을 가정을 바꿔 다시 채점한다 (모두 이미 계산된 워크포워드 예측을 재사용, 재학습 없음).

    by_window: 자동 선택이 보는 과거 성적 기간(1·2·3·5년)을 바꿔 다시 고르기.
    by_rule: '상승 확률 > 기준이면 보유'의 기준 × 거래비용 → 전략 샤프 − 단순 보유 샤프.
    by_period: 검증 기간을 n_periods개로 나눠 시기마다 채점 (예측력이 한때만 있었는지).
    """
    by_window = {}
    for w in windows:
        rw = reselect(wf, h, w)
        row = _robust_row(evaluate(rw, h, daily_returns, cost_bps, lag=lag, sell_tax=sell_tax))
        top = rw["chosen"].value_counts(normalize=True)
        row["가장 많이 고른 모델"] = f"{MODEL_LABELS[top.index[0]]} {top.iloc[0]:.0%}"
        by_window[f"최근 {w // 252}년"] = row
    rule = {}
    for th in thresholds:
        for c in costs:
            ev = evaluate(wf, h, daily_returns, c, threshold=th, lag=lag, sell_tax=sell_tax)
            rule[(f"{th:.0%}", f"{c:g}bp")] = ev["sharpe"] - ev["sharpe_bh"]
    by_rule = pd.Series(rule).unstack()[[f"{c:g}bp" for c in costs]]
    by_rule.index.name, by_rule.columns.name = "보유 기준", "거래비용"
    by_period = {}
    for part in np.array_split(np.arange(len(wf)), n_periods):
        sub = wf.iloc[part]
        by_period[f"{sub.index[0].date()} ~ {sub.index[-1].date()}"] = _robust_row(evaluate(sub, h, daily_returns, cost_bps,
                                                                                      lag=lag, sell_tax=sell_tax))
    return {"by_window": pd.DataFrame(by_window).T, "by_rule": by_rule, "by_period": pd.DataFrame(by_period).T,
            "selection": selection_check(wf, h, daily_returns, cost_bps, windows, thresholds, lag, sell_tax)}


def selection_check(wf, h, daily_returns, cost_bps=0.0, windows=(252, 504, 756, 1260), thresholds=(0.45, 0.5, 0.55), lag=1,
                    sell_tax=0.0):
    """(자동 선택 기간 × 보유 기준) 조합을 모두 시험해 1등을 골랐을 때의 착시 보정 (Deflated Sharpe).

    각 조합의 일간 초과수익(전략 − 단순 보유, 비용 후)으로 샤프를 내고, 가장 좋은 조합이 시험 횟수를 감안해도
    단순 보유보다 나을 확률을 계산한다. 모든 조합이 단순 보유와 똑같으면(초과수익 0) None.
    """
    ex = {}
    for w in windows:
        rw = reselect(wf, h, w)
        for th in thresholds:
            d = evaluate(rw, h, daily_returns, cost_bps, threshold=th, lag=lag, sell_tax=sell_tax)["daily"]
            ex[(f"최근 {w // 252}년", f"{th:.0%}")] = (d["strat"] - d["asset"]).values
    sr = {k: x.mean() / x.std() if x.std() > 1e-12 else np.nan for k, x in ex.items()}
    valid = {k: v for k, v in sr.items() if np.isfinite(v)}
    if not valid:
        return None
    best = max(valid, key=valid.get)
    res = deflated_sharpe(ex[best], list(valid.values()))
    if res is None:
        return None
    grid = pd.Series({k: v * np.sqrt(252) for k, v in sr.items()}).unstack()
    grid.index.name, grid.columns.name = "자동 선택 기간", "보유 기준"
    return {**res, "best": best, "sr_ann": res["sr"] * np.sqrt(252), "sr0_ann": res["sr0"] * np.sqrt(252),
            "grid": grid, "n_flat": len(sr) - len(valid)}


def horizon_check(close, macro=None, alt=None, ticker=None, cost_bps=0.0, horizons=(5, 10, 21, 42, 63),
                  refit_every=21, light=True, lag=1, sell_tax=0.0):
    """예측 기간을 바꿔 워크포워드를 처음부터 다시 돌린다.

    light=True(기본): 무거운 부스팅·앙상블을 빼고 기본 확률·로지스틱 후보만 — 무료 서버에서도 1분 안팎.
    light=False: 모든 후보 (부스팅이 골라진 비율 열 추가). 무거우므로 refit_every를 길게(예: 252) 잡아 쓴다.
    """
    r = np.log(close).diff()
    rows = {}
    for h in horizons:
        wf, _ = walk_forward(close, h, macro, alt=alt, ticker=ticker, refit_every=refit_every, light=light)
        if wf.empty:
            continue
        rows[f"{h}거래일"] = _robust_row(evaluate(wf, h, r, cost_bps, lag=lag, sell_tax=sell_tax))
        if not light:          # 부스팅이 실제로 골라졌는지 — 0%면 부스팅을 넣어도 결과가 같다
            rows[f"{h}거래일"]["부스팅·앙상블 선택 비율"] = float(wf["chosen"].isin(["gbm", "ensemble"]).mean())
    return pd.DataFrame(rows).T


def predict_now(close, h, wf, macro=None, lag_macro=None, alt=None, select_window=756, ticker=None):
    """결과가 확정된 모든 과거로 학습 → 오늘의 P(상승).

    고르는 모델은 워크포워드에서 최근 select_window개 채점 성적이 가장 좋은 후보.
    """
    feat = build_features(close, macro, lag_macro, alt, ticker)
    fwd = forward_log_return(close, h).reindex(feat.index)
    known = fwd.notna().values
    X, y = feat.values, (fwd.values > 0).astype(float)
    cands = make_candidates(feat.columns)
    now = {k: float(p[0]) for k, p in fit_predict_all(cands, X[known], y[known], X[-1:]).items()}

    scored = wf.dropna(subset=["fwd"]).iloc[-select_window:]
    yy = (scored["fwd"].values > 0).astype(float)
    pick = min(cands, key=lambda k: _logloss(scored[k].values, yy)) if len(scored) >= 252 else "base"

    expl = LogisticDirectionModel(10.0).fit(X[known], y[known])     # 설명용 (선형이라 기여도 분해 가능)
    contrib = pd.Series(expl.contributions(X[-1]), index=[FEATURE_LABELS[c] for c in feat.columns])
    return now[pick], pick, now, contrib, feat.index[-1]
