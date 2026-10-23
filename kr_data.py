"""한국 무료 API: DART(전자공시) 공시 이벤트, 한국은행 ECOS 시장금리.

키는 환경변수 DART_API_KEY / ECOS_API_KEY, 또는 이 폴더의 api_keys.env 파일에서 읽는다:
    DART_API_KEY=발급받은키
    ECOS_API_KEY=발급받은키
키가 없으면 해당 기능은 조용히 꺼진다 (None 반환).
"""
import io
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
CACHE = HERE / "cache"
KEY_FILE = HERE / "api_keys.env"


def get_key(name):
    if os.environ.get(name):
        return os.environ[name].strip()
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8-sig").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


# ---------------- ECOS (한국은행) ----------------
ECOS_RATES = {"kr3y": "010200000", "kr10y": "010210000", "krAA": "010300000"}   # 국고3년·국고10년·회사채AA-3년


def ecos_series(stat, item, start, end, cycle="D", key=None):
    key = key or get_key("ECOS_API_KEY")
    if not key:
        return None
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/"
           f"{stat}/{cycle}/{start:%Y%m%d}/{end:%Y%m%d}/{item}")
    js = requests.get(url, timeout=30).json()
    if "StatisticSearch" not in js:
        raise RuntimeError(f"ECOS 오류: {js.get('RESULT', js)}")
    rows = js["StatisticSearch"]["row"]
    return pd.Series([float(r["DATA_VALUE"]) for r in rows],
                     index=pd.to_datetime([r["TIME"] for r in rows], format="%Y%m%d"), name=item)


def kr_short_rate():
    """한국 단기 무위험이자율 = 최근 CD(91일) 금리(소수). 미국 13주 국채(^IRX)에 대응한다.

    ECOS 키가 없거나 무효(재발급 등)이면 공개 샘플 키로 받는다 (샘플은 한 번에 10행까지라 최근 3주만 요청).
    둘 다 실패하면 None."""
    end = pd.Timestamp.today()
    for key in dict.fromkeys([get_key("ECOS_API_KEY"), "sample"]):
        if not key:
            continue
        url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10/"
               f"817Y002/D/{end - pd.Timedelta(days=21):%Y%m%d}/{end:%Y%m%d}/010502000")
        try:
            rows = requests.get(url, timeout=15).json()["StatisticSearch"]["row"]
            return float(rows[-1]["DATA_VALUE"]) / 100.0
        except Exception:
            continue
    return None


def ecos_rates(start, end=None):
    """국고채 3년·10년, 회사채 AA- 3년 일별 금리(%). 키가 없으면 None."""
    if not get_key("ECOS_API_KEY"):
        return None
    end = end or pd.Timestamp.today()
    return pd.DataFrame({name: ecos_series("817Y002", code, pd.Timestamp(start), end)
                         for name, code in ECOS_RATES.items()})


# ---------------- DART (전자공시) ----------------
_dart_lock = __import__("threading").Lock()
_dart_last = [0.0]


def dart_get(endpoint, params, min_interval=0.25, retries=4):
    """DART 요청: 전체 프로세스에서 최소 min_interval초 간격(과다 요청 차단 방지) + 연결 끊김 시 점점 길게 기다렸다 재시도."""
    for attempt in range(retries + 1):
        with _dart_lock:
            wait = _dart_last[0] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            _dart_last[0] = time.time()
        try:
            return requests.get(f"https://opendart.fss.or.kr/api/{endpoint}", params=params, timeout=30).json()
        except (requests.ConnectionError, requests.Timeout, ValueError):
            if attempt == retries:
                raise RuntimeError("DART 서버 연결이 계속 끊깁니다 (요청이 많아 일시 차단됐을 수 있음). 잠시 후 다시 시도하세요.")
            time.sleep(10 * 3 ** attempt)        # 10초, 30초, 90초, 270초
def dart_corp_code(stock_code, key=None):
    """종목코드(6자리) → DART 고유번호. 전체 목록은 한 번 받아 캐시."""
    key = key or get_key("DART_API_KEY")
    if not key:
        return None
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "dart_corpcode.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 30 * 86400:
        table = pd.read_pickle(path)
    else:
        r = requests.get(f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}", timeout=60)
        try:
            xml = zipfile.ZipFile(io.BytesIO(r.content)).read("CORPCODE.xml")
        except zipfile.BadZipFile:     # 키 오류 등은 zip 대신 <result><message>…</message></result> XML로 온다
            try:
                msg = ET.fromstring(r.content).findtext("message")
            except ET.ParseError:
                msg = r.text[:120]
            raise RuntimeError(f"DART: {msg}")
        root = ET.fromstring(xml)
        table = pd.DataFrame([(e.findtext("stock_code").strip(), e.findtext("corp_code"))
                              for e in root.iter("list") if (e.findtext("stock_code") or "").strip()],
                             columns=["stock_code", "corp_code"]).set_index("stock_code")["corp_code"]
        table.to_pickle(path)
    return table.get(stock_code)


def dart_disclosures(stock_code, start, key=None, pause=0.1, kind=None):
    """start 이후 공시 목록(접수일·보고서명). 디스크 캐시 + 새 공시만 추가로 받음.

    kind: DART 공시유형 한 글자 (B=주요사항보고, I=거래소공시 등). None이면 전체.
    여러 종목을 받을 때 B·I만 받으면 호출 수가 크게 준다 (전체 공시의 대부분은 지분공시 D).
    """
    key = key or get_key("DART_API_KEY")
    corp = dart_corp_code(stock_code, key) if key else None
    if not corp:
        return None
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"dart_{stock_code}{'_' + kind if kind else ''}.pkl"
    cached = pd.read_pickle(path) if path.exists() else pd.DataFrame(columns=["date", "report", "rcept_no"])
    start = pd.Timestamp(start)
    bgn = start if cached.empty or cached["date"].min() > start + pd.Timedelta(days=7) else cached["date"].max()

    rows, page = [], 1
    while True:
        params = {"crtfc_key": key, "corp_code": corp, "bgn_de": f"{bgn:%Y%m%d}",
                  "end_de": f"{pd.Timestamp.today():%Y%m%d}", "page_no": page, "page_count": 100}
        if kind:
            params["pblntf_ty"] = kind
        js = dart_get("list.json", params)
        if js.get("status") == "013":        # 조회된 데이터 없음
            break
        if js.get("status") != "000":
            raise RuntimeError(f"DART 오류 {js.get('status')}: {js.get('message')}")
        rows += [(pd.Timestamp(x["rcept_dt"]), x["report_nm"].strip(), x["rcept_no"]) for x in js["list"]]
        if page >= int(js.get("total_page", 1)):
            break
        page += 1
        time.sleep(pause)

    new = pd.DataFrame(rows, columns=["date", "report", "rcept_no"])
    df = pd.concat([cached, new]).drop_duplicates("rcept_no").sort_values("date").reset_index(drop=True)
    df.to_pickle(path)
    return df[df["date"] >= start]


# 보고서명 키워드 → 이벤트 분류. '[기재정정]' 같은 정정 공시는 같은 사건의 반복이라 세지 않는다.
DART_EVENTS = {
    "dart_return": ("자기주식취득결정", "자기주식취득신탁계약체결결정", "주식소각결정"),
    "dart_dilution": ("유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정", "교환사채권발행결정"),
    "dart_contract": ("단일판매ㆍ공급계약체결",),
    "dart_insider": ("임원ㆍ주요주주특정증권등소유상황보고서",),
    "dart_bigholder": ("주식등의대량보유상황보고서",),
}
DART_FEATURES = {
    "dart_return": "자사주 매입·소각 공시(3개월)", "dart_dilution": "증자·CB·BW 발행 공시(3개월)",
    "dart_contract": "공급계약 체결 공시(3개월)", "dart_insider": "임원·주요주주 지분변동 공시(1개월)",
    "dart_bigholder": "5% 대량보유 공시(1개월)", "dart_total": "전체 공시 건수(1개월, 관심도)",
}
DART_WINDOW = {"dart_return": 63, "dart_dilution": 63, "dart_contract": 63,
               "dart_insider": 21, "dart_bigholder": 21, "dart_total": 21}


REPRT = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}       # 1분기·반기·3분기·사업보고서 → 분기 번호


def _amt(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return np.nan


def _fin_record(rows):
    """한 회사·한 보고서의 주요계정 → 자본총계·순이익·영업이익 (연결 우선, 없으면 별도)."""
    fs = "CFS" if any(r["fs_div"] == "CFS" for r in rows) else "OFS"
    rows = [r for r in rows if r["fs_div"] == fs]
    pick = lambda pat: next((_amt(r["thstrm_amount"]) for r in rows if re.match(pat, r["account_nm"])), np.nan)
    return {"rcept_dt": pd.Timestamp(rows[0]["rcept_no"][:8]), "fs": fs, "equity": pick(r"^자본총계"),
            "ni": pick(r"^(당기|분기|반기)순이익"), "op": pick(r"^영업이익")}


def dart_financials(stock_codes, years, key=None, pause=0.1, progress=None):
    """여러 회사의 분기·연간 주요계정. 보고서(연도×분기)별로 캐시하고, 아직 없는 회사만 새로 요청한다.

    반환: DataFrame(code, year, q, rcept_dt, fs, equity, ni, op). ni는 분기 보고서면 '그 분기 3개월' 금액,
    사업보고서면 '연간' 금액 (DART 주요계정 API 규칙)."""
    key = key or get_key("DART_API_KEY")
    if not key:
        return None
    corp = {c: dart_corp_code(c, key) for c in stock_codes}
    corp = {c: v for c, v in corp.items() if v}
    CACHE.mkdir(exist_ok=True)
    out, jobs = [], [(y, rc) for y in years for rc in REPRT]
    for j, (y, rc) in enumerate(jobs):
        path = CACHE / f"dartfin_{y}_{rc}.pkl"
        cache = pd.read_pickle(path) if path.exists() else {}
        # 이미 받았거나, 1년도 더 지난 보고서에서 '없음'으로 확인된 회사는 다시 묻지 않음
        stale = pd.Timestamp(f"{y + 1}-06-30") < pd.Timestamp.today()
        todo = [c for c in corp if c not in cache or (cache[c] is None and not stale)]
        for i in range(0, len(todo), 100):          # 한 번에 최대 100개 회사
            batch = todo[i:i + 100]
            js = dart_get("fnlttMultiAcnt.json", {
                "crtfc_key": key, "corp_code": ",".join(corp[c] for c in batch),
                "bsns_year": str(y), "reprt_code": rc})
            if js.get("status") not in ("000", "013"):
                raise RuntimeError(f"DART 재무 오류 {js.get('status')}: {js.get('message')}")
            by = {}
            for r in js.get("list", []):
                by.setdefault(r["stock_code"].strip(), []).append(r)
            for c in batch:
                cache[c] = _fin_record(by[c]) if c in by else None
            time.sleep(pause)
        pd.to_pickle(cache, path)
        out += [{"code": c, "year": y, "q": REPRT[rc], **rec} for c, rec in cache.items() if rec and c in corp]
        if progress:
            progress((j + 1) / len(jobs), f"DART 재무제표 {y}년 {REPRT[rc]}분기")
    return pd.DataFrame(out)


def value_features(fin, index, close_raw, shares):
    """재무 + 주가 → 날짜별 B/M(PBR 역수), E/P(PER 역수), ROE. 공시 '다음 거래일'부터 반영.

    시가총액 = (그날 종가) × (현재 상장주식수). 과거 주식수 변화(증자·소각)는 반영하지 못하는 근사."""
    out = {k: pd.DataFrame(np.nan, index=index, columns=close_raw.columns) for k in ("bm", "ep", "roe")}
    if fin is None or fin.empty:
        return out
    for code, g in fin.groupby("code"):
        if code not in close_raw.columns or not np.isfinite(shares.get(code, np.nan)):
            continue
        quarters, events = {}, []
        for _, r in g.sort_values("rcept_dt").iterrows():
            yq = (r["year"], r["q"])
            if r["q"] < 4:
                quarters[yq] = r["ni"]
            else:      # 4분기 = 연간 − (1~3분기). 셋 중 하나라도 없으면 계산 불가
                q123 = [quarters.get((r["year"], k)) for k in (1, 2, 3)]
                quarters[yq] = r["ni"] - sum(q123) if all(v is not None and np.isfinite(v) for v in q123) else np.nan
            last = max(quarters)
            seq = [(last[0] - (last[1] - k <= 0), (last[1] - k - 1) % 4 + 1) for k in range(4)]
            ttm = sum(quarters[s] for s in seq) if all(s in quarters and np.isfinite(quarters[s]) for s in seq) else np.nan
            events.append((r["rcept_dt"], r["equity"], ttm))
        ev = pd.DataFrame(events, columns=["dt", "equity", "ttm"])
        pos = np.searchsorted(index.values, ev["dt"].values, side="right")
        ev = ev[pos < len(index)].assign(pos=pos[pos < len(index)]).groupby("pos").last()
        eq = pd.Series(ev["equity"].values, index=index[ev.index]).reindex(index).ffill(limit=400)
        ttm = pd.Series(ev["ttm"].values, index=index[ev.index]).reindex(index).ffill(limit=400)
        mcap = close_raw[code] * shares[code]
        out["bm"][code] = eq / mcap
        out["ep"][code] = ttm / mcap
        out["roe"][code] = (ttm / eq).where(eq > 0)
    return out


def classify(report):
    if report.startswith("["):
        return None
    for name, keys in DART_EVENTS.items():
        if any(k in report for k in keys):
            return name
    return "other"


def dart_features(stock_code, index, df=None):
    """공시 이벤트 건수(최근 N거래일). 접수 시각을 알 수 없어 '접수일 다음 거래일'부터 반영 (보수적).

    df: 미리 받아둔 공시 목록 (없으면 받아온다)."""
    df = dart_disclosures(stock_code, index[0]) if df is None else df
    if df is None or df.empty:
        return None
    df = df.assign(kind=df["report"].map(classify)).dropna(subset=["kind"])
    pos = np.searchsorted(index.values, df["date"].values, side="right")
    keep = pos < len(index)
    df, pos = df[keep], pos[keep]
    out = pd.DataFrame(index=index)
    n = len(index)
    for name in DART_FEATURES:
        mask = np.ones(len(df), bool) if name == "dart_total" else (df["kind"] == name).values
        daily = pd.Series(np.bincount(pos[mask], minlength=n), index=index)
        out[name] = daily.rolling(DART_WINDOW[name], min_periods=1).sum()
    return out


# ---------------- KRX 정보데이터시스템 (날짜별 전 종목 시세) ----------------
KRX_JSON = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"


def krx_daily_all(day, market="STK"):
    """그날 상장돼 있던 전 종목의 시세·시가총액 (market: 'STK' 코스피, 'KSQ' 코스닥).

    그날 기준 목록이라 나중에 상장폐지된 종목도 들어 있다 → 생존 편향 없는 종목 범위를 만들 수 있다.
    KRX가 접근 방식(로그인 등)을 바꾸면 실패할 수 있어, 아직은 연결 점검용으로만 쓴다."""
    data = {"bld": "dbms/MDC/STAT/standard/MDCSTAT01501", "locale": "ko_KR", "mktId": market,
            "trdDd": pd.Timestamp(day).strftime("%Y%m%d"), "share": "1", "money": "1", "csvxls_isNo": "false"}
    errors = []
    for url in (KRX_JSON, KRX_JSON.replace("https://", "http://")):     # 주소·머리말 형식을 바꿔 가며 시도
        try:
            with requests.Session() as ses:
                ses.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "X-Requested-With": "XMLHttpRequest",
                                    "Referer": url.split("/comm/")[0] + "/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101"})
                ses.get(url.split("/comm/")[0] + "/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101", timeout=20)
                r = ses.post(url, data=data, timeout=20)
            r.raise_for_status()
            rows = r.json().get("OutBlock_1") or []
            break
        except Exception as e:
            errors.append(f"{url.split(':')[0]}: {type(e).__name__} {str(e)[:120]} / 응답 앞부분 {getattr(r, 'text', '')[:80]!r}"
                          if 'r' in locals() else f"{url.split(':')[0]}: {type(e).__name__} {str(e)[:120]}")
            rows = []
    if errors and not rows:
        raise ConnectionError(" | ".join(errors))
    if not rows:
        raise ValueError("빈 응답 (휴장일이거나 접근이 막힘)")
    return pd.DataFrame(rows)
