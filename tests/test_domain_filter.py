from utils.domain_filter import is_excluded_domain, normalize_url, should_skip_scraping


def test_excludes_known_low_quality_domains():
    assert is_excluded_domain("https://www.reddit.com/r/foo")
    assert is_excluded_domain("https://en.wikipedia.org/wiki/Foo")
    assert is_excluded_domain("http://sub.medium.com/post")


def test_does_not_exclude_normal_domains():
    assert not is_excluded_domain("https://www.reuters.com/markets")
    assert not is_excluded_domain("https://rbi.org.in/notice")


def test_is_excluded_domain_handles_empty_and_malformed():
    assert not is_excluded_domain("")
    assert not is_excluded_domain("not a url")


def test_normalize_url_strips_tracking_params_and_trailing_slash():
    result = normalize_url("HTTPS://Example.com/Path/?utm_source=x&keep=1")
    assert result == "https://example.com/Path?keep=1"


def test_normalize_url_removes_default_ports():
    assert normalize_url("http://example.com:80/foo") == "http://example.com/foo"
    assert normalize_url("https://example.com:443/foo") == "https://example.com/foo"


def test_normalize_url_empty_input():
    assert normalize_url("") == ""


def test_should_skip_scraping_excludes_non_html_files():
    assert should_skip_scraping("https://example.com/report.pdf")
    assert should_skip_scraping("https://example.com/image.png")


def test_should_skip_scraping_allows_normal_pages():
    assert not should_skip_scraping("https://example.com/article")
