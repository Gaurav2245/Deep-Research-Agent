from datetime import datetime, timedelta

from database.source_scorer import SourceScorer


def test_extract_domain_strips_www():
    assert SourceScorer.extract_domain("https://www.reuters.com/markets") == "reuters.com"


def test_extract_domain_handles_malformed_url():
    assert SourceScorer.extract_domain("not a url") == ""


def test_score_domain_authority_high_authority():
    assert SourceScorer.score_domain_authority("https://www.reuters.com/article") == 1.0


def test_score_domain_authority_medium_authority():
    assert SourceScorer.score_domain_authority("https://www.forbes.com/article") == 0.7


def test_score_domain_authority_low_authority():
    assert SourceScorer.score_domain_authority("https://blogspot.com/post") == 0.3


def test_score_domain_authority_blog_subdomain_penalized():
    # Not an exact low-authority match, but contains "blog" -> penalized to 0.4
    assert SourceScorer.score_domain_authority("https://someone.blogspot.com/post") == 0.4


def test_score_domain_authority_unknown_domain_is_neutral():
    assert SourceScorer.score_domain_authority("https://random-unknown-site.com/x") == 0.5


def test_score_domain_authority_wikipedia_penalized():
    assert SourceScorer.score_domain_authority("https://en.wikipedia.org/wiki/Foo") == 0.4


def test_score_domain_authority_official_keyword_boosts_score():
    baseline = SourceScorer.score_domain_authority("https://random-unknown-site.com/x")
    boosted = SourceScorer.score_domain_authority("https://random-unknown-site.com/official-notice")
    assert boosted > baseline


def test_score_domain_authority_indian_financial_platforms_recognized():
    # These are the actual domains NSE-market-data queries return; they must not
    # fall through to the neutral 0.5 default, or authority scoring is blind for
    # this app's flagship use case.
    high_authority = ["moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
                       "screener.in", "zerodha.com", "groww.in"]
    for domain in high_authority:
        assert SourceScorer.score_domain_authority(f"https://www.{domain}/x") == 1.0

    medium_authority = ["tickertape.in", "dhan.co", "equitymaster.com", "investing.com"]
    for domain in medium_authority:
        assert SourceScorer.score_domain_authority(f"https://www.{domain}/x") == 0.7


def test_extract_date_from_content_day_first_no_comma():
    # Common Indian financial-news byline format, e.g. "Updated on: 7 Aug 2025, 4:11 pm IST"
    content = "Written by: Kusum Kumari  Updated on: 7 Aug 2025, 4:11 pm IST"
    assert SourceScorer.extract_date_from_content(content) == "7 Aug 2025"


def test_extract_date_from_content_month_first_with_comma():
    content = "Published: August 7, 2025"
    assert SourceScorer.extract_date_from_content(content) == "August 7, 2025"


def test_extract_date_from_content_iso_format():
    content = "dated 2026-05-15"
    assert SourceScorer.extract_date_from_content(content) == "2026-05-15"


def test_extract_date_from_content_ordinal_day():
    content = "Updated: 7th August 2025"
    assert SourceScorer.extract_date_from_content(content) == "7th August 2025"


def test_extract_date_from_content_falls_back_to_bare_year():
    content = "This article was posted on 2025 during the monsoon season."
    assert SourceScorer.extract_date_from_content(content) == "2025"


def test_extract_date_from_content_relative_minutes_ago_treated_as_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert SourceScorer.extract_date_from_content("Prices updated 15 minutes ago.") == today


def test_extract_date_from_content_relative_hours_ago_treated_as_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert SourceScorer.extract_date_from_content("Data refreshed 2 hours ago on the exchange.") == today


def test_extract_date_from_content_as_of_time_only_treated_as_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert SourceScorer.extract_date_from_content("As of 3:30 PM, Nifty was trading higher.") == today


def test_extract_date_from_content_live_market_phrase_treated_as_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert SourceScorer.extract_date_from_content("Live market data from NSE.") == today
    assert SourceScorer.extract_date_from_content("Get live prices for Nifty 50 stocks.") == today


def test_extract_date_from_content_relative_timestamp_scores_as_fresh():
    date_str = SourceScorer.extract_date_from_content("Prices updated 15 minutes ago.")
    assert SourceScorer.score_content_freshness(date_str) == 1.0


def test_extract_date_from_content_no_date_returns_none():
    assert SourceScorer.extract_date_from_content("No date info here at all.") is None


def test_score_content_freshness_year_old_article_is_penalized():
    one_year_ago = (datetime.utcnow() - timedelta(days=365)).strftime("%d %b %Y")
    assert SourceScorer.score_content_freshness(one_year_ago) < 0.3


def test_score_content_freshness_recent_article_scores_high():
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%d %b %Y")
    assert SourceScorer.score_content_freshness(yesterday) > 0.9


def test_score_content_freshness_unknown_date_is_neutral():
    assert SourceScorer.score_content_freshness(None) == 0.5


def test_score_content_freshness_strips_ordinal_suffix():
    # "7th Aug 2025" and "7 Aug 2025" should score identically once ordinal is stripped
    with_ordinal = SourceScorer.score_content_freshness("7th Aug 2025")
    without_ordinal = SourceScorer.score_content_freshness("7 Aug 2025")
    assert with_ordinal == without_ordinal
