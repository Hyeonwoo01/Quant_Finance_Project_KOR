"""대체 데이터: 실적·DART 공시가 발표 다음 거래일부터만 쓰이는지."""
import numpy as np
import pandas as pd

import alt_data as ad
import kr_data


def test_alt_data_lags_announcements():
    """대체 데이터: 발표된 정보는 '발표 다음 거래일'부터만 쓰여야 한다"""
    idx = pd.bdate_range("2024-01-01", periods=10)       # 1/1(월) ~ 1/12(금)
    stamps = pd.DatetimeIndex(["2024-01-03 16:00", "2024-01-05 08:00", "2024-01-06 10:00"], tz="America/New_York")
    pos = ad._effective_pos(idx, stamps)
    assert [idx[p].date().isoformat() for p in pos] == ["2024-01-04", "2024-01-08", "2024-01-08"]  # 장후·장전·주말 발표 모두 다음 거래일

    assert ad.headline_score("Apple beats estimates, shares surge") > 0
    assert ad.headline_score("Stock plunges after earnings miss") < 0
    assert ad.headline_score("삼성전자 호실적에 신고가") > 0 and ad.headline_score("반도체 부진 우려에 급락") < 0
    assert ad.headline_score("Company holds annual meeting") == 0


def test_late_starting_alt_data_dropped():
    """가격 기록의 뒤쪽 절반에서야 시작하는 대체 데이터는 뺀다 (모든 특징이 있는 날만 쓰므로 넣으면 학습 기간이 절반 넘게 줄어듦). 끝까지 비어 있는 열(이 종목엔 해당 없음)은 시작 시점 판단에서 무시"""
    _ix = pd.bdate_range("2016-01-01", periods=2500)
    _saved_src = {k: getattr(ad, k) for k in ("earnings_features", "analyst_features", "flow_features")}
    try:
        ad.earnings_features = lambda t, i: pd.DataFrame({"pead": np.where(np.arange(len(i)) >= 2000, 1.0, np.nan)}, index=i)
        ad.analyst_features = lambda t, i: None
        ad.flow_features = lambda t, i: pd.DataFrame({"frgn_flow_21": np.where(np.arange(len(i)) >= 300, 1.0, np.nan),
                                                      "frgn_hold_chg": np.nan}, index=i)
        _alt_s, _st = ad.build_alt_features("SPY", _ix)
        assert list(_alt_s.columns) == ["frgn_flow_21", "frgn_hold_chg"], _alt_s.columns
        assert _st["실적 서프라이즈"].startswith("기간이 짧아 제외") and _st["외국인·기관 수급"].startswith("사용")
        assert _st["애널리스트 변경"] == "데이터 없음"
    finally:
        for _k, _v in _saved_src.items():
            setattr(ad, _k, _v)


def test_dart_disclosures():
    """DART 공시: 분류 + '접수일 다음 거래일'부터 반영되는지 (네트워크 없이 공시 목록을 가짜로 주입)"""
    idx = pd.bdate_range("2024-01-01", periods=10)       # 1/1(월) ~ 1/12(금)
    assert kr_data.classify("주요사항보고서(자기주식취득결정)") == "dart_return"
    assert kr_data.classify("주요사항보고서(유상증자결정)") == "dart_dilution"
    assert kr_data.classify("[기재정정]주요사항보고서(유상증자결정)") is None        # 정정은 세지 않음
    assert kr_data.classify("분기보고서 (2026.06)") == "other"
    fake = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
                         "report": ["주요사항보고서(자기주식취득결정)", "단일판매ㆍ공급계약체결"], "rcept_no": ["1", "2"]})
    orig = kr_data.dart_disclosures
    kr_data.dart_disclosures = lambda code, start, **kw: fake
    try:
        f = kr_data.dart_features("000000", idx)
    finally:
        kr_data.dart_disclosures = orig
    assert f.loc["2024-01-03", "dart_return"] == 0 and f.loc["2024-01-04", "dart_return"] == 1   # 1/3 접수 → 1/4부터
    assert f.loc["2024-01-05", "dart_contract"] == 0 and f.loc["2024-01-08", "dart_contract"] == 1
    assert f.loc["2024-01-08", "dart_total"] == 2
