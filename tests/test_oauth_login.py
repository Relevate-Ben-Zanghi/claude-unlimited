import io
import json
import urllib.error

import pytest

import claude_unlimited.oauth_login as login_module
from claude_unlimited.oauth_login import (
    OAuthLoginError,
    refresh_access_token,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=login_module.TOKEN_ENDPOINT, code=status, msg="", hdrs={}, fp=io.BytesIO(body),
    )


def _fake_urlopen_factory(responses):
    """Builds a fake `urllib.request.urlopen` that replays `responses` in
    order. Each entry is a `_FakeResponse`, an exception instance to raise,
    or a callable returning one of those. Records every request made."""
    captured = {"requests": []}
    queue = list(responses)

    def fake_urlopen(req, timeout=None):
        captured["requests"].append(req)
        captured["timeout"] = timeout
        nxt = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(nxt):
            nxt = nxt()
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    return fake_urlopen, captured


def test_refresh_access_token_sends_correct_payload_and_parses_response(monkeypatch):
    fake_urlopen, captured = _fake_urlopen_factory([_FakeResponse(json.dumps({
        "access_token": "tok-new", "refresh_token": "ref-new", "expires_at": 999,
    }).encode())])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)

    tokens = refresh_access_token("ref-old")

    assert tokens.access_token == "tok-new"
    assert tokens.refresh_token == "ref-new"
    assert tokens.expires_at == 999000  # normalized seconds -> milliseconds
    req = captured["requests"][0]
    assert req.full_url == login_module.TOKEN_ENDPOINT
    assert req.get_method() == "POST"
    payload = json.loads(req.data)
    assert payload["grant_type"] == "refresh_token"
    assert payload["refresh_token"] == "ref-old"
    assert payload["client_id"] == login_module.CLIENT_ID
    assert payload["scope"] == login_module.REFRESH_SCOPE
    assert req.get_header("Content-type") == "application/json"


def test_refresh_identifies_itself_honestly(monkeypatch):
    # The refresh must present as Claude Unlimited, never as another vendor's
    # client. This is the regression guard for the removed curl / axios
    # User-Agent impersonation.
    fake_urlopen, captured = _fake_urlopen_factory([
        _FakeResponse(json.dumps({"access_token": "tok-new"}).encode()),
    ])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)

    refresh_access_token("ref-old")

    ua = captured["requests"][0].get_header("User-agent")
    assert ua == f"claude-unlimited/{login_module.__version__}"
    assert "axios" not in ua.lower()
    assert "claude-code" not in ua.lower()


def test_module_does_not_shell_out():
    # No subprocess or curl anywhere: the refresh goes through the standard
    # library so there is no second HTTP client whose fingerprint differs.
    import inspect
    source = inspect.getsource(login_module)
    assert "subprocess" not in source
    assert "curl" not in source.lower().replace("cloudflare", "")
    assert not hasattr(login_module, "_post_json_via_curl")


def test_refresh_access_token_keeps_old_refresh_token_when_response_omits_one(monkeypatch):
    fake_urlopen, _ = _fake_urlopen_factory([
        _FakeResponse(json.dumps({"access_token": "tok-new"}).encode()),
    ])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)

    tokens = refresh_access_token("ref-old")
    assert tokens.refresh_token == "ref-old"  # a static (non-rotating) refresh_token isn't lost


def test_refresh_access_token_does_not_retry_a_dead_refresh_token(monkeypatch):
    # A 4xx means the refresh_token itself is dead: retrying cannot help, and
    # the account needs a re-login.
    fake_urlopen, captured = _fake_urlopen_factory([_http_error(400, b'{"error":"invalid_grant"}')])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OAuthLoginError, match="Token refresh failed") as excinfo:
        refresh_access_token("dead-refresh-token")
    assert excinfo.value.status_code == 400
    assert len(captured["requests"]) == 1  # no retry on a 4xx


def test_refresh_access_token_reports_bot_protection_block_plainly(monkeypatch):
    # Cloudflare's challenge in front of the endpoint. Not retried, not
    # disguised around: the user is told to re-login through Claude Code.
    fake_urlopen, captured = _fake_urlopen_factory([_http_error(403, b"error code: 1010")])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OAuthLoginError, match="bot protection") as excinfo:
        refresh_access_token("ref-old")
    assert excinfo.value.status_code == 403
    assert "add-account" in str(excinfo.value)
    assert len(captured["requests"]) == 1


def test_refresh_access_token_retries_on_5xx_then_succeeds(monkeypatch):
    fake_urlopen, captured = _fake_urlopen_factory([
        _http_error(503, b'{"error":"unavailable"}'),
        _FakeResponse(json.dumps({"access_token": "tok-new"}).encode()),
    ])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)  # keep the test fast

    tokens = refresh_access_token("ref-old")
    assert tokens.access_token == "tok-new"
    assert len(captured["requests"]) == 2


def test_refresh_access_token_raises_after_exhausting_retries(monkeypatch):
    fake_urlopen, captured = _fake_urlopen_factory([_http_error(503, b'{"error":"unavailable"}')])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)

    with pytest.raises(OAuthLoginError, match="Token refresh failed"):
        refresh_access_token("ref-old")
    assert len(captured["requests"]) == 3


def test_refresh_access_token_retries_network_failure_with_clear_error(monkeypatch):
    # DNS failure / connection refused is retried, unlike a 4xx, and the
    # final error names the cause rather than a status code.
    fake_urlopen, captured = _fake_urlopen_factory([
        urllib.error.URLError("Could not resolve host"),
    ])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)

    with pytest.raises(OAuthLoginError, match="Could not resolve host") as excinfo:
        refresh_access_token("ref-old")
    assert excinfo.value.status_code is None
    assert len(captured["requests"]) == 3


def test_refresh_access_token_raises_clear_error_on_timeout(monkeypatch):
    fake_urlopen, captured = _fake_urlopen_factory([TimeoutError()])
    monkeypatch.setattr(login_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(login_module.time, "sleep", lambda s: None)

    with pytest.raises(OAuthLoginError, match="Timed out"):
        refresh_access_token("ref-old", timeout=7)
    assert captured["timeout"] == 7
