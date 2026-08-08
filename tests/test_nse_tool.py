import pytest

from tools.nse_tool import NSETool

# Real shape of NSE's /api/allIndices response (captured from the live API).
# NSE retired /api/equity-stockIndices (the old per-index lookup, which this
# tool used to call) — it now 404s unconditionally. allIndices returns every
# index's summary in one call instead.
SAMPLE_ALL_INDICES_RESPONSE = {
    "data": [
        {
            "index": "NIFTY 50",
            "last": 24570.65,
            "variation": -65.35,
            "percentChange": -0.27,
            "open": 24538.9,
            "high": 24630.4,
            "low": 24522.75,
            "previousClose": 24636,
            "advances": "24",
            "declines": "26",
            "unchanged": "0",
            "yearHigh": 26373.2,
            "yearLow": 22182.55,
            "pe": "20.93",
            "pb": "3.02",
        },
        {
            "index": "NIFTY BANK",
            "last": 57746.45,
            "variation": -317.2,
            "percentChange": -0.55,
        },
    ]
}


def test_fetch_index_finds_matching_index_by_name(monkeypatch):
    tool = NSETool()
    monkeypatch.setattr(tool, "_get", lambda path: SAMPLE_ALL_INDICES_RESPONSE)
    monkeypatch.setattr(tool, "_ensure_session", lambda: None)

    response = tool._fetch_index("NIFTY 50", "nifty 50")
    content = response.results[0].content

    assert "24570.65" in content
    assert "-0.27" in content
    assert "20.93" in content  # P/E


def test_fetch_index_is_case_insensitive(monkeypatch):
    tool = NSETool()
    monkeypatch.setattr(tool, "_get", lambda path: SAMPLE_ALL_INDICES_RESPONSE)
    monkeypatch.setattr(tool, "_ensure_session", lambda: None)

    response = tool._fetch_index("nifty bank", "banknifty")
    assert "57746.45" in response.results[0].content


def test_fetch_index_raises_clear_error_when_index_not_found(monkeypatch):
    tool = NSETool()
    monkeypatch.setattr(tool, "_get", lambda path: SAMPLE_ALL_INDICES_RESPONSE)
    monkeypatch.setattr(tool, "_ensure_session", lambda: None)

    with pytest.raises(RuntimeError, match="SENSEX"):
        tool._fetch_index("SENSEX", "sensex")

# Real shape of NSE's /api/live-analysis-variations response (captured from the
# live API): fields are ltp/prev_price/perChange/trade_quantity, NOT the
# lastPrice/change/pChange/totalTradedVolume names used by the unrelated
# equity-stockIndices endpoint. A previous version of _fetch_variations
# reused the wrong column names, so every cell but "symbol" rendered blank.
SAMPLE_VARIATIONS_RESPONSE = {
    "NIFTY": {
        "data": [
            {
                "symbol": "GRASIM",
                "series": "EQ",
                "open_price": 3220,
                "high_price": 3349.6,
                "low_price": 3183,
                "ltp": 3323,
                "prev_price": 3220,
                "net_price": 3.52,
                "trade_quantity": 1071215,
                "turnover": 35115.498915,
                "perChange": 3.2,
            },
            {
                "symbol": "BAJFINANCE",
                "series": "EQ",
                "ltp": 1078,
                "prev_price": 1144.8,
                "trade_quantity": 15614001,
                "perChange": -5.84,
            },
        ]
    }
}


def test_fetch_variations_maps_real_field_names(monkeypatch):
    tool = NSETool()
    monkeypatch.setattr(tool, "_get", lambda path: SAMPLE_VARIATIONS_RESPONSE)
    monkeypatch.setattr(tool, "_ensure_session", lambda: None)

    response = tool._fetch_variations("gainers", "gainers")
    content = response.results[0].content

    assert "GRASIM" in content
    assert "3323" in content  # last price (ltp)
    assert "103" in content  # computed change: 3323 - 3220
    assert "3.2" in content  # % change (perChange)
    assert "1071215" in content  # volume (trade_quantity)


def test_fetch_variations_handles_negative_change():
    tool = NSETool()
    records = SAMPLE_VARIATIONS_RESPONSE["NIFTY"]["data"]
    for r in records:
        ltp, prev = r.get("ltp"), r.get("prev_price")
        r["computed_change"] = round(ltp - prev, 2) if isinstance(ltp, (int, float)) and isinstance(prev, (int, float)) else ""

    bajfinance = next(r for r in records if r["symbol"] == "BAJFINANCE")
    assert bajfinance["computed_change"] == round(1078 - 1144.8, 2)


def test_fetch_variations_handles_empty_records(monkeypatch):
    tool = NSETool()
    monkeypatch.setattr(tool, "_get", lambda path: {"NIFTY": {"data": []}})
    monkeypatch.setattr(tool, "_ensure_session", lambda: None)

    response = tool._fetch_variations("losers", "losers")
    assert "No data available" in response.results[0].content
