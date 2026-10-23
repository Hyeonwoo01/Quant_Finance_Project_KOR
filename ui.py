"""디자인 시스템 (Apple 기반, DESIGN.md): 색 토큰, 전역 CSS, Plotly 차트 템플릿, 화면 조각(HTML).

- 바탕은 Parchment(#f5f5f7), 그 위에 흰 카드 + 1px 헤어라인 + 18px 모서리. 그림자·장식 그라데이션 없음
- 클릭할 수 있는 것(버튼·링크·포커스)은 Action Blue(#0066cc) 하나만 씀
- 상태(검증 통과/미통과, 데이터 사용 여부)는 버튼이 아니라 옅은 배경 + 진한 글자 배지로만
- 글꼴: Pretendard(본문·제목) + JetBrains Mono(코드), 굵기는 400/600만, 숫자는 tabular-nums
- 차트 범주색은 validate_palette.js로 검증한 3색 (블루 → 앰버 → 퍼플, 고정 순서)
"""
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------- 색 토큰 ----------
PRIMARY, PRIMARY_FOCUS, PRIMARY_ON_DARK = "#0066cc", "#0071e3", "#2997ff"
INK, INK_80, INK_48, BODY_MUTED = "#1d1d1f", "#333333", "#7a7a7a", "#cccccc"
CANVAS, PARCHMENT, PEARL = "#ffffff", "#f5f5f7", "#fafafc"
HAIRLINE, DIVIDER = "#e0e0e0", "#f0f0f0"
TILE = "#272729"
# 상태 (글자색, 옅은 배경) — 배지·콜아웃에만, 버튼 색으로 쓰지 않음
APPROVE, APPROVE_BG = "#1d8348", "#e8f8f0"
ELECTIVE, ELECTIVE_BG = "#b76e00", "#fef5e7"
REJECT, REJECT_BG = "#c9342b", "#fdecea"

# 차트 전용 (흰 배경 기준 검증 통과: 명도대·채도·색각이상 구분·대비 3:1, 모든 쌍)
BLUE, AMBER, PURPLE = PRIMARY, "#b76e00", "#8e44ad"
NEUTRAL = INK_48                     # 비교 기준선(전체 평균, 단순 보유 등) — 범주색이 아님
POS, NEG = APPROVE, REJECT           # 양/음 극성 (기여도, IC 막대) — 0 기준 위치로도 구분됨
BAND_INNER, BAND_OUTER = "rgba(0,102,204,0.18)", "rgba(0,102,204,0.08)"   # 예측 범위 띠

FONT = "'Pretendard Variable', Pretendard, -apple-system, 'SF Pro Text', system-ui, 'Malgun Gothic', sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, Consolas, monospace"

_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT}; }}
.stApp {{ background: {PARCHMENT}; color: {INK}; }}
.block-container {{ padding-top: 4.25rem; padding-bottom: 0; max-width: 1280px; }}
.stMarkdown p, .stMarkdown li {{ line-height: 1.55; word-break: keep-all; overflow-wrap: break-word; }}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .qf-num {{ font-variant-numeric: tabular-nums; }}

/* 상단 글로벌 내비게이션: 검은 44px 바 + 워드마크 (Streamlit 헤더를 그대로 씀) */
header[data-testid="stHeader"] {{ background: #000; height: 44px; min-height: 44px; }}
header[data-testid="stHeader"]::before {{
    content: "Quant Finance"; position: absolute; left: 50%; top: 0; transform: translateX(-50%);
    line-height: 44px; color: #fff; font: 600 14px/44px {FONT}; letter-spacing: -0.2px; pointer-events: none; }}
/* 아이콘은 글자색(currentColor)을 따르므로 color만 바꾼다 — fill을 직접 칠하면 투명 배경 사각형까지 흰 네모가 됨 */
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="stToolbar"] {{ color: #fff !important; }}
header[data-testid="stHeader"] [data-testid="stStatusWidget"] svg {{ color: rgba(255,255,255,.7) !important; }}
header[data-testid="stHeader"] [data-testid="stDecoration"] {{ display: none; }}

/* 제목: 600 + 음수 자간 */
h1, h2, h3, h4 {{ font-weight: 600 !important; color: {INK}; }}
h1 {{ letter-spacing: -0.28px; }}
h2, h3 {{ letter-spacing: -0.374px; }}
h3 {{ font-size: 21px !important; margin-top: 8px; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {INK_48} !important; font-size: 14px; letter-spacing: -0.224px; }}
code {{ font-family: {MONO}; font-size: 13px; background: {PARCHMENT}; color: {INK};
        border-radius: 5px; padding: 1px 6px; }}
a {{ color: {PRIMARY}; }}

/* 사이드바: 흰 바탕 + 헤어라인, 소제목은 아이브로우(파란 대문자 12px) */
section[data-testid="stSidebar"] {{ background: {CANVAS}; border-right: 1px solid {HAIRLINE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-size: 12px !important; font-weight: 600 !important; letter-spacing: 0.08em; text-transform: uppercase;
    color: {PRIMARY} !important; padding: 20px 0 4px; margin: 0; }}
section[data-testid="stSidebar"] hr {{ margin: 12px 0; }}

/* 지표 = KPI 카드 (흰 바탕, 헤어라인, 18px, 그림자 없음) */
[data-testid="stMetric"] {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 18px; padding: 20px; }}
[data-testid="stMetricLabel"] p {{ font-size: 14px; color: {INK_48}; font-weight: 400; }}
[data-testid="stMetricValue"] {{ font-size: clamp(22px, 2vw, 32px); font-weight: 600; letter-spacing: -0.374px; color: {INK}; }}
[data-testid="stMetric"] [data-testid="stMarkdownContainer"], [data-testid="stMetric"] p {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }}
[data-testid="stMetricDelta"] {{ font-size: 12px; color: {INK_48} !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}           /* 방향 의미 없는 보조설명에 화살표 숨김 */

/* 탭 = 필터 칩: 알약 + 헤어라인, 선택은 잉크 채움 */
.stTabs [role="tablist"] {{ gap: 8px; flex-wrap: wrap; border-bottom: none; box-shadow: none; padding: 4px 0 8px; }}
.stTabs [data-testid="stTab"] {{ height: 36px; padding: 0 16px; border-radius: 9999px; background: {CANVAS};
    border: 1px solid {HAIRLINE}; color: {INK_80}; display: flex; align-items: center; transition: transform .1s ease; }}
.stTabs [data-testid="stTab"] p {{ font-size: 14px; font-weight: 400; letter-spacing: -0.224px; color: inherit; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: {INK}; border-color: {INK}; color: #fff; }}
.stTabs [data-testid="stTab"][aria-selected="true"] p {{ font-weight: 600; }}
.stTabs [data-testid="stTab"]:active {{ transform: scale(0.95); }}
.stTabs [data-testid="stTab"]:focus-visible {{ outline: 2px solid {PRIMARY_FOCUS}; outline-offset: 2px; }}
.stTabs .react-aria-SelectionIndicator {{ display: none; }}
[data-testid="stTabPanel"] {{ padding-top: 16px; }}

/* 버튼: 주 액션은 파란 알약, 나머지는 파란 테두리 알약. 누르면 살짝 줄어듦 */
.stButton button, .stDownloadButton button {{ border-radius: 9999px; font-size: 15px; font-weight: 400;
    min-height: 40px; padding: 8px 20px; transition: transform .1s ease; }}
[data-testid="stBaseButton-secondary"] {{ background: {CANVAS}; color: {PRIMARY}; border: 1px solid {PRIMARY}; }}
[data-testid="stBaseButton-secondary"]:hover {{ background: {CANVAS}; color: {PRIMARY}; border-color: {PRIMARY}; }}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"]:hover {{ background: {PRIMARY}; color: #fff; border: 1px solid {PRIMARY}; }}
.stButton button:active, .stDownloadButton button:active {{ transform: scale(0.95); }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{ outline: 2px solid {PRIMARY_FOCUS}; outline-offset: 2px; box-shadow: none; }}

/* 입력: 헤어라인, 포커스는 2px Focus Blue */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{ border-radius: 11px !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
    border-color: {PRIMARY_FOCUS} !important; box-shadow: 0 0 0 1px {PRIMARY_FOCUS} !important; }}

/* 알림 = 콜아웃: 옅은 상태색 배경 + 진한 글자, 모서리 11px */
[data-testid="stAlertContainer"] {{ border-radius: 11px; border: none; }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] * {{ color: {INK} !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: {CANVAS};
    border: 1px solid {HAIRLINE}; border-left: 3px solid {PRIMARY}; border-radius: 8px; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: {APPROVE_BG}; }}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] * {{ color: {APPROVE} !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: {ELECTIVE_BG}; }}
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentWarning"] * {{ color: {ELECTIVE} !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background: {REJECT_BG}; }}
[data-testid="stAlertContentError"], [data-testid="stAlertContentError"] * {{ color: {REJECT} !important; }}

/* 카드형 컨테이너: 차트·표·확장 영역 */
[data-testid="stPlotlyChart"] {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 18px; padding: 12px 8px 4px; box-sizing: border-box; overflow: hidden; }}
[data-testid="stDataFrame"] {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 18px; overflow: hidden; box-sizing: border-box; }}
[data-testid="stExpander"] details {{ background: {CANVAS}; border-radius: 18px; border: 1px solid {HAIRLINE}; }}
[data-testid="stExpander"] summary {{ padding: 14px 20px; }}
[data-testid="stProgress"] [role="progressbar"] > div > div {{ border-radius: 9999px; }}
.stMarkdown table {{ background: {CANVAS}; border-collapse: separate; border-spacing: 0; border: 1px solid {HAIRLINE};
    border-radius: 18px; overflow: hidden; width: 100%; font-variant-numeric: tabular-nums; }}
.stMarkdown th {{ background: {PEARL}; font-size: 14px; font-weight: 600; color: {INK_80}; }}
.stMarkdown th, .stMarkdown td {{ border: none; border-bottom: 1px solid {DIVIDER}; padding: 12px 16px; }}
.stMarkdown tr:last-child td {{ border-bottom: none; }}
hr {{ border-color: {HAIRLINE}; }}

/* ---------- 화면 조각 (qf-*) ---------- */
.qf-eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: {PRIMARY}; }}

/* 다크 요약 카드 */
.qf-hero {{ background: {TILE}; color: #fff; border-radius: 18px; padding: 40px 48px; margin-bottom: 8px; }}
.qf-hero .qf-eyebrow {{ color: {BODY_MUTED}; }}
.qf-hero-top {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 24px; }}
.qf-hero-name {{ font-size: 21px; font-weight: 600; letter-spacing: 0.231px; margin: 12px 0 4px; }}
.qf-hero-price {{ font-size: 56px; font-weight: 600; line-height: 1.07; letter-spacing: -0.28px; font-variant-numeric: tabular-nums; }}
.qf-hero-price small {{ font-size: 21px; font-weight: 400; margin-left: 12px; color: {BODY_MUTED}; letter-spacing: 0; }}
.qf-hero-price small.up {{ color: {PRIMARY_ON_DARK}; }}
.qf-hero-aside {{ border: 1px solid rgba(255,255,255,.15); border-radius: 11px; padding: 14px 18px; font-size: 14px;
    color: {BODY_MUTED}; line-height: 1.7; font-variant-numeric: tabular-nums; }}
.qf-hero-aside b {{ color: #fff; font-weight: 600; }}
.qf-hero-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 32px; }}
.qf-hero-metric {{ background: rgba(255,255,255,.06); border-radius: 11px; padding: 18px 20px; min-width: 0; }}
.qf-hero-metric .lbl {{ font-size: 14px; color: {BODY_MUTED}; }}
.qf-hero-metric .val {{ font-size: 32px; font-weight: 600; line-height: 1.2; margin-top: 6px; font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere; }}
.qf-hero-metric .val.accent {{ color: {PRIMARY_ON_DARK}; }}
.qf-hero-metric .sub {{ font-size: 12px; color: {BODY_MUTED}; margin-top: 6px; line-height: 1.4; overflow-wrap: anywhere; }}
.qf-hero-note {{ font-size: 12px; color: {BODY_MUTED}; margin-top: 24px; }}

/* 섹션 머리 */
.qf-section {{ margin: 8px 0 16px; }}
.qf-section h2 {{ font-size: 34px; line-height: 1.25; letter-spacing: -0.374px; margin: 6px 0 4px; padding: 0; }}
.qf-section p {{ font-size: 17px; line-height: 1.47; letter-spacing: -0.374px; color: {INK_80}; margin: 0; max-width: 820px; }}

/* 판정 카드: 큰 결론 + 상태 배지 */
.qf-verdict {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 18px; padding: 24px; margin-bottom: 16px; }}
.qf-verdict-row {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px 16px; margin-top: 8px; }}
.qf-verdict-head {{ font-size: 40px; font-weight: 600; line-height: 1.1; color: {INK}; font-variant-numeric: tabular-nums; }}
.qf-verdict-head .unit {{ font-size: 17px; font-weight: 400; color: {INK_48}; margin-left: 4px; }}
.qf-verdict p {{ font-size: 15px; color: {INK_80}; margin: 12px 0 0; line-height: 1.55; }}

/* 상태 배지 */
.qf-badge {{ display: inline-block; border-radius: 9999px; padding: 4px 12px; font-size: 14px; font-weight: 600;
    letter-spacing: -0.224px; line-height: 1.29; white-space: nowrap; }}
.qf-badge.approve {{ background: {APPROVE_BG}; color: {APPROVE}; }}
.qf-badge.elective {{ background: {ELECTIVE_BG}; color: {ELECTIVE}; }}
.qf-badge.reject {{ background: {REJECT_BG}; color: {REJECT}; }}
.qf-badge.neutral {{ background: {PEARL}; color: {INK_80}; border: 1px solid {DIVIDER}; }}
.qf-badges {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 12px; font-size: 14px; color: {INK_48}; }}
.qf-badges .qf-badge {{ font-size: 13px; font-weight: 400; padding: 3px 10px; }}

/* 바닥글 */
.qf-footer {{ margin: 48px -1rem 0; padding: 32px 1rem; border-top: 1px solid {HAIRLINE}; font-size: 12px; color: {INK_80}; line-height: 1.6; }}
.qf-footer b {{ color: {INK}; font-weight: 600; }}
.qf-footer .muted {{ color: {INK_48}; margin-top: 4px; }}

@media (max-width: 1023px) {{ .qf-hero-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 640px) {{
    .block-container {{ padding-left: 16px; padding-right: 16px; }}
    .qf-hero {{ padding: 28px 20px; }} .qf-hero-price {{ font-size: 40px; }}
    .qf-hero-grid {{ gap: 8px; }} .qf-hero-metric {{ padding: 14px; }} .qf-hero-metric .val {{ font-size: 24px; }}
    .qf-section h2 {{ font-size: 26px; }} .qf-verdict-head {{ font-size: 32px; }}
}}
</style>
"""


def apply():
    """페이지마다 한 번: 전역 CSS + Plotly 기본 템플릿."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly()


def _html(s):
    st.markdown(s, unsafe_allow_html=True)


def badge(text, kind="neutral"):
    """읽기 전용 상태 배지 HTML. kind: approve / elective / reject / neutral."""
    return f'<span class="qf-badge {kind}">{escape(str(text))}</span>'


def badges(label, items):
    """'라벨: [배지][배지]…' 한 줄. items: [(텍스트, kind), ...]"""
    _html(f'<div class="qf-badges">{escape(label)}{"".join(badge(t, k) for t, k in items)}</div>')


def hero(eyebrow, name, price, change, aside, metrics, note):
    """상단 다크 요약 카드.

    change: 전일 대비 수익률(float) / aside: [(라벨, 값), ...] 오른쪽 상자
    metrics: [(라벨, 값, 보조설명, 강조여부), ...] 아래 4칸
    """
    up = change >= 0
    aside_html = "<br>".join(f"{escape(k)} <b>{escape(str(v))}</b>" for k, v in aside)
    cells = "".join(
        f'<div class="qf-hero-metric"><div class="lbl">{escape(lbl)}</div>'
        f'<div class="val{" accent" if accent else ""}">{escape(str(val))}</div>'
        f'<div class="sub">{escape(str(sub))}</div></div>'
        for lbl, val, sub, accent in metrics)
    _html(f'<div class="qf-hero"><div class="qf-hero-top"><div>'
          f'<div class="qf-eyebrow">{escape(eyebrow)}</div>'
          f'<div class="qf-hero-name">{escape(name)}</div>'
          f'<div class="qf-hero-price">{price}<small class="{"up" if up else ""}">{change:+.2%} 전일 대비</small></div>'
          f'</div><div class="qf-hero-aside">{aside_html}</div></div>'
          f'<div class="qf-hero-grid">{cells}</div>'
          f'<div class="qf-hero-note">{escape(note)}</div></div>')


def section(eyebrow, title, desc=""):
    """탭 맨 위 머리: 아이브로우 + 34px 제목 + 설명."""
    _html(f'<div class="qf-section"><div class="qf-eyebrow">{escape(eyebrow)}</div>'
          f'<h2>{escape(title)}</h2>{f"<p>{desc}</p>" if desc else ""}</div>')


def verdict(eyebrow, head, unit="", tags=(), desc=""):
    """결론 카드: 큰 결론 문구 + 상태 배지들 + 설명. tags: [(텍스트, kind), ...] / desc는 HTML 허용."""
    _html(f'<div class="qf-verdict"><div class="qf-eyebrow">{escape(eyebrow)}</div>'
          f'<div class="qf-verdict-row"><span class="qf-verdict-head">{escape(head)}'
          f'{f"<span class=unit>{escape(unit)}</span>" if unit else ""}</span>'
          f'{"".join(badge(t, k) for t, k in tags)}</div>'
          f'{f"<p>{desc}</p>" if desc else ""}</div>')


def chart(fig):
    """Plotly 차트를 흰 카드에. 배경은 템플릿이 아니라 그림에 직접 넣어야 Streamlit이 바꾸지 않는다."""
    fig.update_layout(paper_bgcolor=CANVAS, plot_bgcolor=CANVAS)
    st.plotly_chart(fig, width="stretch", theme=None)


def footer(lead, text, muted):
    _html(f'<div class="qf-footer"><b>{escape(lead)}</b> {escape(text)}<div class="muted">{escape(muted)}</div></div>')


def _register_plotly():
    # Streamlit은 plotly_chart 모듈을 처음 불러올 때 기본 템플릿을 "streamlit"으로 바꾼다.
    # 첫 실행에서 그 뒤에 덮어쓰이지 않도록 먼저 불러 둔다.
    import streamlit.elements.plotly_chart  # noqa: F401
    if "qf" in pio.templates:
        pio.templates.default = "qf"
        return
    axis = dict(gridcolor=DIVIDER, linecolor=HAIRLINE, zerolinecolor=HAIRLINE, tickcolor=HAIRLINE,
                tickfont=dict(color=INK_48, size=12), title=dict(font=dict(color=INK_48, size=12)), showline=False,
                automargin=True)
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        colorway=[BLUE, AMBER, PURPLE],
        paper_bgcolor=CANVAS, plot_bgcolor=CANVAS,
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=INK_80, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CANVAS, bordercolor=HAIRLINE, font=dict(family=FONT, color=INK, size=12)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["qf"] = t
    pio.templates.default = "qf"
