from tools.query_router import classify_nse_query


def test_classifies_combined_gainers_and_losers_query():
    result = classify_nse_query("nifty 50 top 10 gainers and losers as of today")
    assert result == ["gainers", "losers"]


def test_classifies_gainers_only():
    assert classify_nse_query("Nifty 50 top gainers today NSE") == ["gainers"]


def test_classifies_losers_with_common_misspelling():
    assert classify_nse_query("top loosers NSE BSE today") == ["losers"]


def test_classifies_most_active():
    assert classify_nse_query("most active stocks on NSE today") == ["most active"]


def test_classifies_market_status():
    assert classify_nse_query("is the market open today NSE") == ["market status"]


def test_classifies_bare_index_queries():
    assert classify_nse_query("nifty 50 index today") == ["nifty 50"]
    assert classify_nse_query("banknifty live") == ["banknifty"]
    assert classify_nse_query("sensex today") == ["sensex"]
    assert classify_nse_query("nifty it sector performance") == ["nifty it"]


def test_does_not_match_queries_without_indian_market_anchor():
    assert classify_nse_query("biggest losers of World War 2") == []
    assert classify_nse_query("who are the losers in the recent election") == []
    assert classify_nse_query("top gainers of the stock market crash of 1929") == []
    assert classify_nse_query("S&P 500 top gainers today") == []


def test_does_not_match_unrelated_finance_queries():
    assert classify_nse_query("Reliance Industries quarterly results") == []
    assert classify_nse_query("RBI monetary policy decisions") == []


def test_empty_and_none_input():
    assert classify_nse_query("") == []
    assert classify_nse_query(None) == []
