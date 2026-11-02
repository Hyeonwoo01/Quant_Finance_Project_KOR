"""코스피 종목 고르기: 매주 계산한 결과, 모델 비교, 실전 추천 기록."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
import universe as uv
from quant_engine import hac_tstat, max_drawdown
from app_data import get_kr_factors, get_universe, get_weekly
from views.common import _signed, signed_table


def render(ctx):
    """코스피 종목 고르기: 매주 계산한 결과, 모델 비교, 실전 추천 기록. ctx: app.py가 만든 공통 값 (선택한 종목·설정·모델)."""
    exec_lag, exec_word, use_tax = ctx.exec_lag, ctx.exec_word, ctx.use_tax
    ui.section("KOSPI Ranking", "코스피 대형주 중 상대적으로 더 오를 종목",
               "한 종목의 방향 대신, 코스피 대형주 약 200개를 서로 비교해 순위를 매깁니다. "
               "사이드바의 종목 설정과 무관하게 동작합니다.")
    info = uv.data_info()
    weekly, picks = get_weekly(uv.weekly_stamp())

    def run_update():
        bar = st.progress(0.0, text="시작…")
        uv.update_data(progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
        st.cache_data.clear()
        st.rerun()

    download_note = ("처음 받을 때 코스피 전 종목 주가와, 한 번이라도 200위 안에 든 종목의 "
                     "6년치 수급·실적·공시·재무제표를 받느라 **1시간 안팎** 걸립니다. "
                     "(터미널에서 `python universe.py update`로 받아도 됩니다.)")
    if info is None and weekly is None:
        st.info("아직 데이터가 없습니다. " + download_note)
        if st.button("데이터 받기", type="primary"):
            run_update()
    else:
        U = None
        if info is None:                 # 이 컴퓨터엔 데이터가 없음 → GitHub Actions가 매주 계산해 둔 결과
            c1, c2 = st.columns([3, 1])
            meta = weekly.get("meta") or {}
            span = f"코스피 {meta['n_all']}종목 중 과거 200위 이내 진입 {meta['n']}종목 · " if "n_all" in meta and "n" in meta else ""
            c1.caption(f"GitHub Actions가 매주 장 마감 뒤 계산해 둔 결과 · 데이터 {weekly['data_end'].date()}까지 "
                       f"(계산 {weekly['updated']}) · {span}수수료·체결 비용 {weekly['cost_bps']:g}bp (매수·매도 각각)")
            hold = c2.selectbox("종목 교체 주기", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
            U = weekly["results"].get((hold, exec_lag, use_tax))
            if U is None:
                st.warning("이 설정 조합의 미리 계산된 결과가 없습니다. 아래에서 데이터를 받아 직접 계산하세요.")
            with st.expander("수수료를 바꿔 직접 계산하려면"):
                st.caption("이 컴퓨터에 데이터를 받으면 수수료를 바꿔 가며 바로 계산합니다. " + download_note)
                if st.button("데이터 받기"):
                    run_update()
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.caption(f"데이터 갱신: {info['updated']} · 코스피 {info.get('n_all', '?')}종목 중 "
                       f"과거 200위 이내 진입 {info['n']}종목 · {info['years']}년")
            hold = c2.selectbox("종목 교체 주기", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
            pick_cost = c3.number_input("수수료·체결 비용 (매수·매도 각각, bp)", 0.0, 100.0, 5.0, 1.0,
                                        help="수수료 + 호가 차이·시장 충격. 증권거래세는 사이드바 설정에 따라 판 비중만큼 연도별 세율로 따로 뺍니다.")
            if st.button("데이터 업데이트 (새 날짜만 받음)"):
                run_update()
            try:
                with st.spinner("과거 시점마다 모델을 학습·검증하는 중… (처음 1~3분)"):
                    U = get_universe(hold, pick_cost, info["updated"], exec_lag, use_tax)
            except Exception as e:
                st.error(f"계산 실패: {e}")

        if U is not None:
            auto = U["table"].loc["auto"]
            ok = auto["초과수익 t값"] > 2 and auto["IC t값"] > 2
            pick_label = "고르지 않음 — 최근 성적이 좋은 모델이 없어, 전체 종목에 똑같이 나눠 담기를 권장" if U["pick"] == "none" \
                else uv.MODEL_LABELS[U["pick"]]
            ui.verdict(f"{U['date'].date()} 기준", pick_label, "",
                       [("과거 검증 통과", "approve") if ok else ("평균보다 낫다고 보기 어려움", "elective")])
            if ok:
                st.success(f"과거 검증에서 상위 20% 종목이 비용을 빼고도 전체 평균보다 확실히 더 올랐습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, 우연일 가능성 낮음 t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}). 그래도 앞으로의 성과를 보장하지 않습니다.")
            else:
                st.warning(f"과거 검증에서 상위 20% 종목이 전체 평균보다 더 올랐다고 확신하기 어렵습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}; t가 2를 넘어야 우연이 아니라고 봅니다). 아래 순위는 참고로만 보세요.")

            cv = U["curves"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["상위 20% (비용 후)"], name="상위 20% 종목 (비용 뺀 후)",
                                     line=dict(color=ui.BLUE, width=2.5)))
            fig.add_trace(go.Scatter(x=cv["_universe"].index, y=cv["_universe"], name="전체 종목 평균",
                                     line=dict(color=ui.NEUTRAL)))
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["하위 20%"], name="하위 20% 종목",
                                     line=dict(color=ui.PURPLE, dash="dot")))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.caption(f"최대 낙폭: 상위 20% {max_drawdown(cv['auto']['상위 20% (비용 후)']):.0%} · "
                       f"전체 종목 평균 {max_drawdown(cv['_universe']):.0%}. "
                       "리밸런싱 시점 기준이라 그 사이에 잠깐 더 빠졌던 하락은 덜 잡힙니다. "
                       f"종목을 고른 날의 데이터로 순위를 매기고 {exec_word}에 사고판다고 봤습니다.")

            if len(picks):
                st.markdown("**실전 기록** — 매주 장 마감 뒤 GitHub Actions가 그 주의 상위 20% 종목을 미리 적어 두고, "
                            "21·63거래일 뒤 실제 성적을 채점합니다 (교체 주기 63거래일 · 다음 날 종가에 산다고 보고 · 비용 전).")
                sm = uv.picks_summary(picks)
                if sm["채점 수"].sum() == 0:
                    st.caption(f"기록 {len(picks)}번 (첫 기록 {picks['date'].iloc[0]}). 아직 채점된 기록이 없습니다 — "
                               "첫 채점은 첫 기록에서 21거래일 뒤입니다.")
                else:
                    st.dataframe(sm.style.format({"채점 수": "{:.0f}", "평균 초과수익": "{:+.2%}", "전체 평균보다 나은 비율": "{:.0%}",
                                                  "t값 (겹침 보정)": _signed(1)}, na_rep="—"), width="stretch")
                    st.caption("초과수익 = 고른 종목(동일 비중) − 그날 종목 범위 전체 평균. 결과를 모르는 상태에서 적어 둔 기록이라 "
                               "과거 검증보다 정직한 성적표입니다. 매주 기록해 기간이 겹치므로 t값은 겹침을 보정했고, 2를 넘어야 우연이 아니라고 봅니다.")
                with st.expander(f"기록 보기 ({len(picks)}번)"):
                    rec = pd.DataFrame({"기록일": picks["date"],
                                        "순위 모델": picks["model"].map(lambda m: uv.MODEL_LABELS.get(m, "고르지 않음 → 고정 가중 순위")),
                                        "종목 수": pd.to_numeric(picks["n"], errors="coerce"),
                                        **{f"{h}거래일 초과수익": pd.to_numeric(picks[f"basket_{h}"], errors="coerce")
                                           - pd.to_numeric(picks[f"market_{h}"], errors="coerce") for h in uv.SCORE_HORIZONS},
                                        "종목": picks["names"]}).iloc[::-1]
                    st.dataframe(rec.style.format({"종목 수": "{:.0f}", **{f"{h}거래일 초과수익": "{:+.2%}" for h in uv.SCORE_HORIZONS}},
                                                  na_rep="—"), hide_index=True, width="stretch")

            st.markdown("**모델 비교** — 모두 같은 조건으로 시험했습니다. 이 표를 보고 나중에 1등을 고르면 성적이 부풀려지므로, "
                        "앱은 매 시점 그때까지의 성적만 보고 고른 **자동 선택**을 씁니다.")
            st.dataframe(U["table"].style.format({
                "평균 IC": "{:+.3f}", "IC t값": "{:+.1f}", "IC>0 비율": "{:.0%}",
                "상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": "{:+.1f}",
                "상위−하위 스프레드(연)": "{:+.1%}", "최대 낙폭(비용후)": "{:.0%}",
                "평균 회전율": "{:.0%}", "리밸런싱 수": "{:.0f}", "체결 막힌 주문": "{:.0f}"}, na_rep="—"),
                hide_index=True, width="stretch")
            share = U["chosen"].value_counts(normalize=True)
            st.caption("자동 선택 비율: " + ", ".join(
                f"{uv.MODEL_LABELS.get(k, '고르지 않음')} {v:.0%}" for k, v in share.items()) +
                ". IC(순위 정확도)는 점수 순위와 실제 수익률 순위가 얼마나 일치하는지입니다. "
                "0이면 무관하고 0.05 정도면 실무에서 의미 있는 수준입니다. t값이 2를 넘으면 우연일 가능성이 낮다고 봅니다 "
                "(리밸런싱끼리의 자기상관을 반영한 Newey-West t값). 체결 막힌 주문 = 체결일에 하한가·거래정지라 못 판 종목(계속 보유)과 "
                "상한가·거래정지라 못 산 종목의 수. '(기준)' 모델은 같은 조건으로 비교만 하고 자동 선택 후보에는 넣지 않습니다."
                + (f" 자동 선택이 고른 모델이 바뀐 횟수: 리밸런싱 {len(U['chosen'])}번 중 {U['switches']}번 "
                   "(자주 바뀌면 모델 고르기가 운에 좌우된다는 뜻)." if U.get("switches") is not None else ""))
            sel = U.get("selection")
            if sel is not None:
                st.caption(f"이 표의 1등({uv.MODEL_LABELS[sel['best']]})을 나중에 고르면 생기는 착시 보정: 효과가 없어도 {sel['n_trials']}개 중 "
                           f"1등의 초과수익 샤프는 {sel['sr0_ann']:.2f}(연) 정도가 나올 수 있고, 1등 샤프 {sel['sr_ann']:.2f}가 진짜일 확률"
                           f"(Deflated Sharpe)은 {sel['dsr']:.0%}입니다.")

            fx = U.get("factors")
            if fx is not None:
                st.markdown("**알려진 요인으로 설명되는가?** — 자동 선택 상위 20%의 초과수익(비용 후 − 전체 평균)을 시장·요인 수익률에 회귀")
                ui.verdict(f"리밸런싱 {fx['n']}번 · R² {fx['r2']:.2f}", f"알파 {fx['alpha']:+.1%}", "요인을 빼고 남는 초과수익 (연)",
                           [("요인으로 설명되지 않는 신호 있음 (t > 2)", "approve") if fx["alpha_t"] > 2 else
                            ("알려진 요인과 구분되지 않음", "elective")], f"알파 t = {fx['alpha_t']:.1f}")
                st.dataframe(fx["table"].style.format({"노출(계수)": "{:+.2f}", "t값": _signed(1), "요인 자체 성과(연)": "{:+.1%}"},
                                                      na_rep="—"), width="stretch")
                st.caption("요인은 같은 종목 범위에서 그날 순위로 상위 20% − 하위 20%를 동일가중으로 직접 만들었고, 규모는 시가총액 대신 "
                           "거래대금으로 근사했습니다. 노출이 크고 t가 2를 넘는 요인이 있으면 초과수익의 일부가 그 요인 덕분입니다. "
                           "알파가 0 근처면 '새로운 신호'가 아니라 이미 알려진 효과에 기울어진 결과입니다. 표본이 리밸런싱 횟수라 작으니 t값을 꼭 함께 보세요.")

            capt = U.get("capacity")
            if capt is not None:
                st.markdown("**운용 규모를 키우면?** — 주문이 그 종목 하루 거래대금에서 차지하는 비율만큼 불리하게 체결된다고 보고 (시장 충격)")
                st.dataframe(capt.style.format({"주문 / 하루 거래대금 (중앙값)": "{:.3%}", "시장 충격 비용(연)": "{:.2%}",
                                                "초과수익(연, 모든 비용 후)": "{:+.1%}", "t값": _signed(1)}, na_rep="—"), width="stretch")
                st.caption("한 번 사거나 팔 때 추가 비용 ≈ 하루 변동성 × √(주문 금액 ÷ 하루 거래대금)으로 계산했습니다 (시장 충격의 '제곱근 법칙', "
                           "계수 1로 보수적으로). 종목당 주문 금액 = 자본 ÷ 보유 종목 수, 거래대금은 그날 기준 63일 평균입니다. "
                           "초과수익이 0 아래로 떨어지기 시작하는 규모가 이 전략을 실제로 굴릴 수 있는 대략적인 한계입니다.")
            cs = U.get("cost_sens")
            if cs is not None:
                cs_tab, breakeven = cs
                st.markdown("**수수료·슬리피지가 더 크면?** — 매수·매도 각각의 비용(bp)을 바꿔 자동 선택 상위 20%를 다시 계산")
                st.dataframe(cs_tab.style.format({"초과수익(연, 비용후)": "{:+.1%}", "t값": _signed(1), "최대 낙폭(비용후)": "{:.0%}"},
                                                 na_rep="—"), width="stretch")
                st.caption((f"손익분기 비용: 매수·매도 각각 약 {breakeven:.0f}bp를 넘으면 초과수익이 0 아래로 내려갑니다. "
                            if np.isfinite(breakeven) else "비용이 0이어도 초과수익이 없어 손익분기 비용을 계산하지 않습니다. ")
                           + "실제 체결가가 종가보다 몇 bp만 불리해도 무너지는 신호라면 실전에서 쓰기 어렵습니다.")
            rw = U.get("risk_weighted")
            if rw is not None:
                a0 = U["table"].loc["auto"]
                st.markdown("**위험을 비중에 반영하면?** — 같은 종목을 63일 변동성의 역수에 비례해 담음 (종목당 동일 비중의 2배 상한)")
                st.dataframe(pd.DataFrame({"동일 비중": a0, "변동성 역가중": rw}).T[
                    ["상위20% 초과수익(연, 비용후)", "초과수익 t값", "최대 낙폭(비용후)", "평균 회전율"]].style.format(
                    {"상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": _signed(1), "최대 낙폭(비용후)": "{:.0%}",
                     "평균 회전율": "{:.0%}"}, na_rep="—"), width="stretch")
                st.caption("변동성이 큰 종목을 덜 담아 포트폴리오 흔들림을 줄이는 가장 단순한 위험 예산입니다. 업종·베타 제한이나 "
                           "종목 간 상관을 고려한 최적화는 하지 않습니다.")
            mk = U.get("market")
            if mk is not None:
                st.markdown("**시장 상황별로는?** — 리밸런싱 기간을 그 기간 종목 범위 평균 수익률로 하락·횡보·상승 셋으로 나눔")
                st.dataframe(mk.style.format({"기간 수": "{:.0f}", "시장 평균(기간)": "{:+.1%}", "초과수익 평균(기간)": "{:+.2%}",
                                              "초과 > 0 비율": "{:.0%}"}, na_rep="—"), width="stretch")
                st.caption("특정 시장(예: 상승장)에서만 초과수익이 났다면 그 신호는 시장 상황에 기대는 것일 수 있습니다. 기간 수가 적어 참고용입니다.")
            ofs = U.get("offsets")
            if ofs is not None:
                st.markdown(f"**리밸런싱 날짜를 바꾸면?** — 교체 주기 {hold}거래일 안에서 시작일을 {len(ofs['table'])}가지로 옮겨 처음부터 다시 계산")
                ot = ofs["table"].rename(columns=lambda c: c.replace("auto", "자동 선택").replace("momentum", "모멘텀"))
                pct_cols = [c for c in ot.columns if c.endswith("초과수익(연)")]
                st.dataframe(ot.style.format({"첫 리밸런싱": lambda d: f"{pd.Timestamp(d).date()}", "리밸런싱 수": "{:.0f}",
                                              **{c: "{:+.1%}" for c in pct_cols},
                                              **{c: _signed(1) for c in ot.columns if c.endswith("t값")},
                                              **{c: "{:+.3f}" for c in ot.columns if c.endswith("IC")}}, na_rep="—"),
                             width="stretch")
                sm = ofs["summary"]
                st.caption(" · ".join(f"{uv.MODEL_LABELS[m].replace('⭐ ', '')}: 평균 {s['평균']:+.1%} "
                                      f"(최소 {s['최소']:+.1%} ~ 최대 {s['최대']:+.1%}), 0보다 큰 경우 {s['0보다 큰 수']}/{s['시작일 수']}"
                                      for m, s in sm.items())
                           + ". 위 표의 기본 결과는 시작일 옮김 0입니다. 시작일만 바꿨는데 성적이 크게 달라진다면 결과가 "
                             "'날짜를 잘 고른 운'에 기대고 있다는 뜻이라 평균과 범위를 함께 봐야 합니다 (보고만, 판정 기준은 그대로).")
            rk = U.get("risk")
            if rk is not None:
                st.markdown(f"**지금 고른 {rk['n']}종목을 같은 비중으로 들고 있으면?** — 지난 {rk['days']}거래일 동안 그대로 들고 있었다고 보고 계산")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("하루 VaR 95%", f"−{rk['var95']:.1%}", f"내일 기준({rk['model']}) −{rk['garch_var95']:.1%}", delta_color="off")
                r2.metric("하루 ES 95%", f"−{rk['es95']:.1%}", "최악 5% 날의 평균 손실", delta_color="off")
                r3.metric("최악의 하루", f"{rk['worst_day'][1]:+.1%}", f"{rk['worst_day'][0].date()}", delta_color="off")
                r4.metric("최악의 한 달 (21거래일)", f"{rk['worst_month'][1]:+.1%}", f"{rk['worst_month'][0].date()}까지", delta_color="off")
                st.markdown(f"시장이 가장 크게 빠졌던 날 {len(rk['stress'])}일 — 시장 = 전체 종목 평균, 이 종목들의 시장 베타 {rk['beta']:.2f}")
                signed_table(rk["stress"] * 100, 1)
                st.caption("단위 %. 과거 실제 수익률로 계산해서, 그 기간에 없던 종류의 충격은 반영하지 못합니다. "
                           "베타가 1보다 크면 시장보다 더 크게 출렁인다는 뜻입니다.")

            left, right = st.columns(2)
            fmt = {c: "{:.0%}" for c in U["top"].columns if c not in ("순위", "종목명", "점수")}
            fmt["점수"] = "{:+.3f}"
            with left:
                st.markdown(f"**상위 20종목** (전체 {U['n_univ']}개 중)")
                st.dataframe(U["top"].style.format(fmt), width="stretch")
            with right:
                st.markdown("**하위 20종목**")
                st.dataframe(U["bottom"].style.format(fmt), width="stretch")
            st.caption("신호 열은 전체 종목 중 상대 위치입니다 (100% = 가장 큼). 예: 외국인 순매수 강도 90%면 상위 10%.")

            ic = U["ic"].dropna()
            fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in ic.values]))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="교체 시점별 순위 정확도(IC)")
            ui.chart(fig)

            kf = get_kr_factors(uv.kr_factors_stamp())
            if kf is not None:
                with st.expander("한국판 요인의 성과 (방향 예측·시장 국면의 수익 분해에 쓰는 일간 요인)"):
                    cols = [c for c in uv.KR_FACTORS if c in kf and kf[c].notna().any()]
                    cum = np.exp(np.log1p(kf[cols].fillna(0.0)).cumsum()).rename(columns=uv.KR_FACTORS)
                    fig = go.Figure([go.Scatter(x=cum.index, y=cum[c], name=c) for c in cum.columns])
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="누적 (시작=1)",
                                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    ui.chart(fig)
                    yr = pd.DataFrame({"연 평균": kf[cols].mean() * 252, "연 변동성": kf[cols].std() * np.sqrt(252),
                                       "t값 (자기상관 보정)": kf[cols].apply(lambda c: hac_tstat(c.values))},
                                      ).rename(index=uv.KR_FACTORS)
                    st.dataframe(yr.style.format({"연 평균": "{:+.1%}", "연 변동성": "{:.1%}", "t값 (자기상관 보정)": _signed(1)}, na_rep="—"),
                                 width="stretch")
                    st.caption("같은 종목 범위(그때의 거래대금 상위 200)에서 21거래일마다 줄 세워 상위 30% − 하위 30%(동일 비중)를 들고 간 수익률입니다. "
                               "시장은 시가총액 가중 − CD 91일 금리, 소형주는 시가총액 하위 − 상위, 가치는 장부가/시총, 수익성은 ROE(DART 재무제표, "
                               "공시 다음 거래일부터), 모멘텀은 12-1개월 수익률 기준입니다. 파마-프렌치의 투자 요인은 자산 데이터가 없어 뺐습니다.")
            with st.expander("데이터·한계"):
                cov = pd.Series({uv.FEATURES[c][0]: U["coverage"].get(c, 0) for c in uv.FEATURES})
                st.markdown("**신호별 데이터 보유율** (30% 미만이면 모델에서 제외)")
                st.dataframe(cov.to_frame("보유율").style.format("{:.0%}"), width="stretch")
                if U.get("krx"):
                    scope = (f"""- **종목 범위:** KRX 일별매매정보의 **그날 상장돼 있던** 코스피 보통주 전체(그 사이 상장폐지된 {U.get('n_delisted', 0)}종목 포함) 중,
  각 과거 시점의 거래대금 상위 200개 (공식 코스피200 구성과는 다름). 폐지 종목은 정리매매 마지막 가격까지의 손익을 반영하고 그 뒤는 현금으로 봅니다.
- **가치 지표:** DART 재무제표(분기·사업보고서)의 자본총계·순이익을 **공시 다음 거래일부터**, 시가총액은 **그날의 실제 값**(KRX)으로 씁니다.
  폐지 종목은 수급·실적·재무 데이터가 없어 그 신호는 중간값으로 둡니다.""")
                else:
                    scope = """- **종목 범위:** 현재 상장된 코스피 보통주 전체 중, 각 과거 시점의 거래대금 상위 200개 (공식 코스피200 구성과는 다름).
  '지금 큰 종목'으로 과거를 고르는 편향은 없앴지만, 그 사이 **상장폐지된 종목은 주가가 없어 빠져 있어**
  과거 성과가 약간 좋게 보일 수 있습니다 (KRX 일별매매정보를 받으면 포함).
- **가치 지표:** DART 재무제표(분기·사업보고서)의 자본총계·순이익을 **공시 다음 거래일부터** 씁니다.
  시가총액은 '그날 주가 × 현재 상장주식수'로 근사해서, 과거 증자·소각으로 주식수가 바뀐 종목은 오차가 있습니다."""
                st.markdown(scope + """
- **비용:** 리밸런싱마다 바뀐 종목 비율 × (살 때·팔 때 수수료·체결 비용)을 빼고, 증권거래세를 켜면 판 비중 × 그날의 코스피 세율을
  더 뺍니다. 호가 차이·시장 충격이 설정보다 크면 결과가 나빠집니다.
- 투자 조언이 아닙니다.
""")
