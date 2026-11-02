"""여러 탭이 함께 쓰는 표 서식·수익 분해 화면."""
import pandas as pd
import streamlit as st

import ui
import universe as uv
from quant_engine import timing_attribution
from app_data import get_kr_factors


def _signed(digits=2):
    """부호 붙인 숫자. 반올림해서 0이면 부호 없이 — 아주 작은 음수가 '-0.00'으로 보이지 않게."""
    return lambda v: "—" if pd.isna(v) else f"{0:.{digits}f}" if round(v, digits) == 0 else f"{v:+.{digits}f}"


def signed_table(df, digits=2):
    """+/− 차이 표: +는 초록, −는 빨강, 반올림해서 0이면 색 없음."""
    d = df.astype(float).round(digits) + 0.0
    color = lambda v: f"color: {ui.APPROVE}" if v > 0 else f"color: {ui.REJECT}" if v < 0 else ""
    if not d.isna().any().any():
        st.dataframe(d.style.format(_signed(digits)).map(color), width="stretch")
        return
    # 빈 칸이 있으면 글자로 바꿔 '—'로 보이게 (Streamlit은 숫자 열의 결측을 None으로 표시한다)
    styles = d.map(color)
    txt = d.map(_signed(digits))
    st.dataframe(txt.style.apply(lambda _: styles, axis=None), width="stretch",
                 column_config={c: st.column_config.TextColumn(str(c), alignment="right") for c in txt.columns})


def robust_table(df):
    """방향 예측 점검 표 (검증 통과는 글자로)."""
    d = df.assign(**{"검증 통과": df["검증 통과"].map({True: "통과", False: "—"})})
    first = [c for c in ("가장 많이 고른 모델", "부스팅·앙상블 선택 비율") if c in d]
    d = d[first + [c for c in d.columns if c not in first]]
    st.dataframe(d.style.format({"적중률": "{:.1%}", "'항상 오른다' 대비": "{:+.1%}", "확률 정확도 개선": "{:+.1%}",
                                 "z": _signed(1), "전략 샤프": "{:.2f}", "단순 보유 샤프": "{:.2f}", "전략 최대 낙폭": "{:.0%}",
                                 "보유 비중": "{:.0%}", "부스팅·앙상블 선택 비율": "{:.0%}"}, na_rep="—"), width="stretch",
                 column_config={"검증 통과": st.column_config.TextColumn("검증 통과", alignment="center")})


def attribution_view(daily):
    """보유/현금 타이밍 전략의 수익 분해 표 + 해석: 같은 비중으로 계속 들고 있었을 때보다 나았는지 (타이밍 실력),
    한국판 요인(시장·소형주·가치·수익성·모멘텀)으로 설명되지 않는 부분이 있는지."""
    ff = get_kr_factors(uv.kr_factors_stamp())
    att = timing_attribution(daily, ff)
    sm = att["summary"]
    st.dataframe(sm.style.format({"알파(연)": "{:+.1%}", "알파 t값": _signed(1), "노출(β)": "{:.2f}", "R²": "{:.2f}",
                                  "관측일": "{:,.0f}"}, na_rep="—"), width="stretch")
    a = sm.iloc[0]
    txt = (f"첫 줄: 이 전략은 평균적으로 주식을 {a['노출(β)']:.0%}만큼 들고 있던 셈이고, 그 비중으로 계속 들고 있었을 때보다 "
           f"1년에 {abs(a['알파(연)']):.1%} {'더' if a['알파(연)'] >= 0 else '덜'} 벌었습니다 (t = {a['알파 t값']:.1f}). "
           "t가 2를 넘어야 타이밍 실력이 우연이 아니라고 봅니다.")
    if ff is None:
        txt += " 한국판 요인 자료(ranking/kr_factors.csv)가 아직 없어 요인 분석은 생략했습니다."
    elif "전략 (요인 모형)" in sm.index:
        s_, b_ = sm.loc["전략 (요인 모형)"], sm.loc["단순 보유 (요인 모형)"]
        txt += (f" 아래 두 줄: 한국 주식의 알려진 요인(시장·소형주·가치·수익성·모멘텀)으로 설명되는 부분을 빼고 남는 수익입니다. "
                f"전략 {s_['알파(연)']:+.1%} vs 단순 보유 {b_['알파(연)']:+.1%} — 전략 쪽이 더 크면 그 차이가 요인으로 설명되지 않는 "
                "타이밍 효과입니다. 요인은 코스피 거래대금 상위 200종목으로 매주 직접 만든 근사이고(무료 한국 요인 자료가 없음), "
                "현금은 CD 91일 금리를 번다고 봤습니다. 요인은 매주 금요일에 갱신돼 그 뒤 며칠은 빠집니다.")
    st.caption(txt)
    if att["loadings"] is not None:
        with st.expander("요인별 노출 자세히 (계수 · t값)"):
            st.dataframe(att["loadings"].style.format("{:+.2f}", na_rep="—"), width="stretch")
