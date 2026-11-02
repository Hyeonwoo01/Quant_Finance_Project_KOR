"""수급·공시·뉴스: 지금의 시장 심리 지표."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import alt_data as ad
import kr_data
import ui
from app_data import get_snapshot


def render(ctx):
    """수급·공시·뉴스: 지금의 시장 심리 지표. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    close, q, r, spot, ticker = ctx.close, ctx.q, ctx.r, ctx.spot, ctx.ticker
    ui.section("Flows · Disclosures · News", "실적 전망·공시·수급·뉴스")
    st.info("외국인·기관 수급, 실적 서프라이즈, DART 공시는 과거 기록이 있어 방향 예측 모델에 들어갑니다. "
            "실적 전망치와 뉴스 감성은 무료로는 지금 값만 구할 수 있어 참고용으로 보여주고, 기록 종목은 매일 저장해 "
            "'예측 정확도 검증' 탭의 실전 기록에서 쓸모가 있는지 채점합니다.")
    with st.spinner("전망·뉴스 불러오는 중…"):
        snap = get_snapshot(ticker, spot, r, q)

    st.subheader("실적과 전망치")
    rv = snap["revisions"]
    code = ad.krx_code(ticker)
    if code:
        st.caption("한국 종목은 네이버 증권의 실적과 전망치를 보여줍니다. (E)가 붙은 열은 애널리스트들의 평균 전망치입니다. "
                   "금액 단위: 억원, 비율: %")
        for label, period in (("분기", "quarter"), ("연간", "annual")):
            try:
                fin = ad.naver_financials(code, period)
            except Exception:
                fin = None
            if fin is not None and not fin.empty:
                st.markdown(f"**{label}**")
                st.dataframe(fin.style.format("{:,.2f}", na_rep="—"), width="stretch")
            else:
                st.caption(f"{label} 재무 데이터를 불러오지 못했습니다.")
    elif rv is None or rv.empty:
        st.caption("추정치 데이터가 없습니다.")
    else:
        fmt = {c: "{:+.2%}" for c in rv.columns if "대비" in c}
        fmt["현재 추정 EPS"] = "{:,.2f}"
        st.dataframe(rv.style.format(fmt, na_rep="—"), width="stretch")
        st.caption("전망치가 올라가는 중(+)이면 애널리스트들이 실적 기대를 높이고 있다는 뜻입니다. "
                   "과거 전망 변경 이력은 무료로 구할 수 없어, 과거 검증은 등급·목표가 변경 데이터로 대신합니다.")

    if code and kr_data.get_key("DART_API_KEY"):
        st.subheader("최근 공시 (DART)")
        try:
            dis = kr_data.dart_disclosures(code, close.index[0])
            recent = dis.sort_values("date", ascending=False).head(30).copy()
            recent["분류"] = recent["report"].map(kr_data.classify).map(
                lambda k: kr_data.DART_FEATURES.get(k, "기타" if k else "정정").split("(")[0])
            recent["링크"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + recent["rcept_no"]
            st.dataframe(recent[["date", "분류", "report", "링크"]], hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn("접수일"), "report": "보고서명",
                                        "링크": st.column_config.LinkColumn("원문", display_text="열기")})
        except Exception as e:
            st.caption(f"공시를 불러오지 못했습니다: {e}")

    if code:
        st.subheader("외국인·기관 수급 (최근 60거래일)")
        try:
            flw = ad.load_investor_flows(code, close.index[-60]).iloc[-60:]      # fl(실전 기록 모듈)과 겹치지 않게
            fig = go.Figure()
            fig.add_trace(go.Bar(x=flw.index, y=flw["frgn"], name="외국인 순매수(주)", marker_color=ui.BLUE))
            fig.add_trace(go.Bar(x=flw.index, y=flw["inst"], name="기관 순매수(주)", marker_color=ui.AMBER))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
        except Exception as e:
            st.caption(f"수급 데이터를 불러오지 못했습니다: {e}")

    st.subheader("뉴스 감성")
    nw = snap["news"]
    if nw is None or nw.empty:
        st.caption("뉴스를 불러오지 못했습니다.")
    else:
        recent = nw[nw["time"] >= nw["time"].max() - pd.Timedelta(days=7)]
        a, b, c = st.columns(3)
        a.metric("최근 7일 평균 감성", f"{recent['score'].mean():+.2f}", "−1 매우 부정 ~ +1 매우 긍정", delta_color="off")
        b.metric("긍정 / 부정 헤드라인", f"{(nw['score'] > 0).sum()} / {(nw['score'] < 0).sum()}",
                 f"중립 {(nw['score'] == 0).sum()}", delta_color="off")
        c.metric("수집 기간", f"{(nw['time'].max() - nw['time'].min()).days + 1}일", f"헤드라인 {len(nw)}개",
                 delta_color="off")
        daily = nw.groupby(nw["time"].dt.date)["score"].mean()
        fig = go.Figure(go.Bar(x=[d.strftime("%m-%d") for d in daily.index], y=daily.values,
                               marker_color=[ui.POS if v > 0 else ui.NEG if v < 0 else ui.NEUTRAL for v in daily.values]))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="일평균 감성",
                          xaxis_type="category", yaxis_range=[-1, 1])
        ui.chart(fig)
        st.dataframe(nw[["time", "score", "title", "source", "link"]], hide_index=True, width="stretch",
                     column_config={"time": st.column_config.DatetimeColumn("시각", format="MM-DD HH:mm"),
                                    "score": st.column_config.NumberColumn("감성", format="%+.2f"),
                                    "title": "헤드라인", "source": "언론사",
                                    "link": st.column_config.LinkColumn("링크", display_text="열기")})
        st.caption("감성 점수는 제목에 나온 긍정·부정 단어를 세는 단순한 방식이라, 반어나 맥락은 이해하지 못합니다. "
                   "과거 뉴스 기록을 무료로 구할 수 없어 예측에 도움이 되는지는 아직 확인하지 못했습니다. 그래서 기록 대상 종목은 "
                   "이 탭의 옵션·전망·뉴스 지표를 매일 저장해 두고, '예측 정확도 검증' 탭의 실전 기록에서 채점합니다.")
