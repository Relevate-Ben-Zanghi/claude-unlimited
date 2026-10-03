"""OAuth token-refresh support for Claude Unlimited.

The account-adding flow itself lives in cli.py's add_account(): a browser
login through an isolated Claude Code session (`CLAUDE_CONFIG_DIR`), read
back via anthropic_oauth.read_claude_code_credentials(). This module covers
what happens afterwards: refresh_access_token() exchanges a stored
refresh_token for a new access token, so a Profile never needs manual
re-authentication merely because its token expired.

CLIENT_ID is the public OAuth client id the grant was issued to. PKCE public
clients don't use a client secret, so this is not a leaked credential; a
refresh_token can only be redeemed against the client it was issued for.

This module identifies itself honestly. The refresh request carries a
`claude-unlimited/<version>` User-Agent and is sent with the standard
library's HTTP client. It does not imitate another client's headers or TLS
fingerprint to get past the endpoint's bot management. If the endpoint
rejects the request on that basis, the honest outcome is a clear error
telling the user the account needs a re-login through Claude Code itself,
not a disguised retry.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from . import __version__

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_ENDPOINT = "https://platform.claude.com/v1/oauth/token"

# RFC 6749 §6 lets a refresh request restate the scopes of the original
# grant. These are the scopes a Claude Code login is issued with, and the
# refreshed token needs the same ones to be usable for inference.
REFRESH_SCOPE = "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

USER_AGENT = f"claude-unlimited/{__version__}"

# Cloudflare's bot-management challenge in front of the token endpoint
# answers with this status and body when it declines a client. Surfaced as
# a readable message rather than circumvented.
_CLOUDFLARE_BLOCK_STATUS = 403


class OAuthLoginError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        # Lets a caller distinguish a 429 (backoff requested) from a 400
        # (dead refresh_token) without parsing the message text. None for a
        # network-level failure, where there was no response to read a
        # status from.
        self.status_code = status_code


@dataclass(frozen=True)
class LoginTokens:
    access_token: str
    refresh_token: Optional[str]
    expires_at: Optional[int]


def _post_json(url: str, payload: bytes, timeout: float,
               extra_headers: Optional[dict] = None) -> tuple[int, dict, bytes]:
    """POST JSON with the standard library and return (status_code, headers, body).

    An HTTP error status is returned, not raised, so the caller can decide
    what is retryable. Only a network-level failure (DNS, connection,
    timeout) raises OAuthLoginError."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **(extra_headers or {}),
    }
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()
    except urllib.error.URLError as exc:
        raise OAuthLoginError(f"Could not reach the token endpoint: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OAuthLoginError(f"Timed out reaching the token endpoint after {timeout:g}s") from exc


def _normalize_expires_at(data: dict) -> Optional[int]:
    """Normalizes the token endpoint's expiry fields to epoch milliseconds,
    the unit Claude Code uses on disk (see anthropic_oauth.py's
    ImportedCredentials).

    Handles both shapes defensively: an absolute expires_at in seconds or
    milliseconds, disambiguated by magnitude (under 10**12 is seconds), or,
    failing that, a relative expires_in in seconds added to now."""
    raw = data.get("expires_at")
    if raw is not None:
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = None
    if raw is not None:
        return int(raw * 1000) if raw < 1e12 else int(raw)

    expires_in = data.get("expires_in")
    if expires_in is not None:
        try:
            return int(time.time() * 1000) + int(float(expires_in) * 1000)
        except (TypeError, ValueError):
            return None
    return None


def _describe_failure(status: int, body: bytes) -> str:
    detail = body.decode("utf-8", errors="replace")[:300]
    if status == _CLOUDFLARE_BLOCK_STATUS and "error code: 1010" in detail:
        return (
            f"Token refresh failed (HTTP {status}): the token endpoint's bot protection "
            "declined this client. Claude Unlimited does not disguise itself to get past "
            "that check. Re-add the account with `claude-unlimited add-account` so Claude "
            "Code performs the login itself."
        )
    return f"Token refresh failed (HTTP {status}): {detail}"


def refresh_access_token(refresh_token: str, timeout: float = 30.0) -> LoginTokens:
    """Exchanges a stored refresh_token for a new access token, so an OAuth
    Profile keeps working past its original token's expiry without a manual
    re-authentication.

    Retries up to twice with exponential backoff on a 5xx or network-level
    failure. A 4xx is never retried: a dead refresh_token means the account
    needs a real re-login, not another attempt."""
    payload = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "scope": REFRESH_SCOPE,
    }).encode("utf-8")

    max_retries = 2
    base_delay = 0.5
    last_exc: Optional[OAuthLoginError] = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(base_delay * (2 ** (attempt - 1)))
        try:
            status, _headers, body = _post_json(TOKEN_ENDPOINT, payload, timeout)
        except OAuthLoginError as exc:
            last_exc = exc
            continue  # network-level failure: retry

        if status >= 500 and attempt < max_retries:
            continue  # transient server error: retry
        if status >= 400:
            raise OAuthLoginError(_describe_failure(status, body), status_code=status)

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OAuthLoginError(f"Token refresh response wasn't valid JSON: {exc}") from exc

        access_token = data.get("access_token")
        if not access_token:
            raise OAuthLoginError("Token refresh response had no access_token — unexpected response shape.")

        return LoginTokens(
            access_token=access_token,
            # A rotating refresh_token replaces itself in the response; a
            # static one may be omitted entirely, so keep the one the
            # caller already had rather than losing it.
            refresh_token=data.get("refresh_token") or refresh_token,
            expires_at=_normalize_expires_at(data),
        )

    raise last_exc or OAuthLoginError("Token refresh failed after retries.")
