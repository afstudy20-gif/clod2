"""OAuth authentication flows for AI providers."""
import base64
import hashlib
import json
import secrets
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import requests
from pathlib import Path

from .config import load_config, save_config, CONFIG_DIR


# ── OpenRouter PKCE OAuth ────────────────────────────────────────────────────

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback code."""

    code: str | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]

        if code:
            _OAuthCallbackHandler.code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Success!</h2>"
                b"<p>You can close this tab and return to Clod.</p>"
                b"<script>window.close()</script></body></html>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Error: No code received</h2></body></html>")

        # Shutdown server after handling
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def login_openrouter() -> str | None:
    """Run OpenRouter PKCE OAuth flow. Returns API key or None on failure."""
    verifier, challenge = _generate_pkce()

    # Start local server on a random port
    server = HTTPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    port = server.server_address[1]
    callback_url = f"http://localhost:{port}/callback"

    # Open browser to OpenRouter auth page
    auth_url = (
        f"https://openrouter.ai/auth"
        f"?callback_url={callback_url}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
    )
    webbrowser.open(auth_url)

    # Wait for callback (blocks until GET received or timeout)
    _OAuthCallbackHandler.code = None
    server.timeout = 120  # 2 minute timeout
    server.handle_request()
    server.server_close()

    code = _OAuthCallbackHandler.code
    if not code:
        return None

    # Exchange code for API key
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/auth/keys",
            json={"code": code, "code_verifier": verifier},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        api_key = data.get("key")
        if api_key:
            # Save to config
            config = load_config()
            config.setdefault("api_keys", {})["openrouter"] = api_key
            config.setdefault("oauth", {})["openrouter"] = {
                "method": "pkce",
                "key": api_key,
            }
            save_config(config)
            return api_key
    except Exception:
        pass
    return None


def _find_google_client_secret() -> Path | None:
    """Find the google client secret file in CONFIG_DIR or home .clod."""
    search_dirs = [CONFIG_DIR, Path.home() / ".clod"]
    for d in search_dirs:
        if not d.is_dir():
            continue
        # 1. Check for specific name
        p = d / "google_client_secret.json"
        if p.exists():
            return p
        # 2. Check for any client_secret_*.json
        for p in d.glob("client_secret_*.json"):
            return p
    return None


# ── Google Gemini OAuth ──────────────────────────────────────────────────────

def login_google() -> str | None:
    """Run Google OAuth2 flow for Gemini API. Returns a refresh token or None."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return None

    # Check for client secret in CONFIG_DIR
    secret_path = _find_google_client_secret()
    if not secret_path:
        return None

    scopes = ["https://www.googleapis.com/auth/generative-language"]

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secret_path),
            scopes=scopes,
        )
        credentials = flow.run_local_server(port=0, open_browser=True)

        # Save tokens to config
        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else scopes,
        }

        config = load_config()
        config.setdefault("oauth", {})["google"] = token_data
        save_config(config)

        return credentials.token
    except Exception:
        return None


def google_oauth_status() -> dict:
    """Return a safe Google OAuth status payload for UI display."""
    config = load_config()
    token_data = config.get("oauth", {}).get("google") or {}
    secret_path = _find_google_client_secret()
    configured = bool(token_data.get("refresh_token") or token_data.get("token"))
    return {
        "configured": configured,
        "source": "oauth" if configured else None,
        "client_secret_exists": secret_path is not None,
        "client_secret_path": str(secret_path) if secret_path else str(CONFIG_DIR / "google_client_secret.json"),
        "scopes": token_data.get("scopes") or [],
    }


def get_google_credentials():
    """Load Google OAuth credentials from config. Returns Credentials or None."""
    config = load_config()
    token_data = config.get("oauth", {}).get("google")
    if not token_data:
        return None

    try:
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes"),
        )

        # Refresh if expired
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            # Update stored token
            token_data["token"] = creds.token
            config["oauth"]["google"] = token_data
            save_config(config)

        return creds
    except Exception:
        return None


# ── Provider-agnostic interface ──────────────────────────────────────────────

OAUTH_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "description": "Access many model families via single sign-in",
        "login": login_openrouter,
    },
    "google": {
        "name": "Google (Gemini)",
        "description": "Sign in with Google for Gemini API access",
        "login": login_google,
        "setup_note": "Requires ~/.clod/google_client_secret.json from Google Cloud Console",
    },
    "gemini": {
        "name": "Google (Gemini)",
        "description": "Sign in with Google for Gemini API access",
        "login": login_google,
        "setup_note": "Requires ~/.clod/google_client_secret.json from Google Cloud Console",
    },
}
