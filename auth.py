"""
auth.py — Google OAuth2 gate for Streamlit on Cloud Run.

Uses direct HTTP calls to Google's OAuth endpoints (no PKCE) to avoid the
code-verifier session-state loss problem that occurs when google-auth-oauthlib
generates a PKCE challenge but session state is wiped during the redirect.

Auth state lives in st.session_state (in-memory per Streamlit WebSocket session).
Users must re-authenticate after a page refresh or new tab — this is a Streamlit
architectural constraint; there is no reliable way to persist cookies from within
a Streamlit app without a dedicated server-side session store.

Usage in app.py:
    from auth import check_auth, render_user_badge
    check_auth()   # call before any st.* content; stops unauthenticated users
    render_user_badge()  # in sidebar: shows email + sign-out button

Environment variables required:
    GOOGLE_CLIENT_ID      — from GCP Console → APIs & Services → Credentials
    GOOGLE_CLIENT_SECRET  — same credential
    OAUTH_REDIRECT_URI    — must exactly match the URI registered in GCP Console
                            e.g. https://prot-prompt.tools.phyx44.com
    ALLOWED_EMAIL_DOMAIN  — e.g. "phyx44.com"  (any @phyx44.com account passes)
    ALLOWED_EMAILS        — optional comma-separated override list

If GOOGLE_CLIENT_ID is not set (local dev), auth is bypassed entirely.
"""

from __future__ import annotations

import html as _html_lib
import os
import urllib.parse

import requests as http_requests
import streamlit as st
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

# ── Config ─────────────────────────────────────────────────────────────────────

_CLIENT_ID      = os.getenv("GOOGLE_CLIENT_ID", "")
_CLIENT_SECRET  = os.getenv("GOOGLE_CLIENT_SECRET", "")
_REDIRECT_URI   = os.getenv("OAUTH_REDIRECT_URI", "https://prot-prompt.tools.phyx44.com")
_ALLOWED_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "phyx44.com")
_ALLOWED_EMAILS = os.getenv("ALLOWED_EMAILS", "")  # comma-separated, optional override

_AUTH_URL  = "https://accounts.google.com/o/oauth2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPES    = "openid email profile"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_allowed(email: str) -> bool:
    email = email.lower().strip()
    if _ALLOWED_EMAILS:
        for entry in _ALLOWED_EMAILS.split(","):
            entry = entry.strip().lower()
            if entry.startswith("@") and email.endswith(entry):
                return True
            if entry == email:
                return True
        return False
    return email.endswith(f"@{_ALLOWED_DOMAIN}")


def _build_auth_url() -> str:
    params = {
        "client_id":     _CLIENT_ID,
        "redirect_uri":  _REDIRECT_URI,
        "response_type": "code",
        "scope":         _SCOPES,
        "prompt":        "select_account",
        "access_type":   "offline",
    }
    return _AUTH_URL + "?" + urllib.parse.urlencode(params)


def _exchange_code(code: str) -> dict:
    resp = http_requests.post(_TOKEN_URL, data={
        "code":          code,
        "client_id":     _CLIENT_ID,
        "client_secret": _CLIENT_SECRET,
        "redirect_uri":  _REDIRECT_URI,
        "grant_type":    "authorization_code",
    })
    resp.raise_for_status()
    return resp.json()


# ── Public API ─────────────────────────────────────────────────────────────────

def check_auth() -> None:
    """
    Gate the app behind Google OAuth.

    - No-op if GOOGLE_CLIENT_ID is not set (local dev mode).
    - Returns immediately if session is already authenticated.
    - If ?code=... is present: exchanges code, verifies email, stores session.
    - Otherwise: renders sign-in page and calls st.stop().
    """
    if not _CLIENT_ID or not _CLIENT_SECRET:
        return

    if st.session_state.get("_auth_email"):
        return

    params = st.query_params

    # ── Handle OAuth callback ──────────────────────────────────────────────────
    if "code" in params:
        try:
            token_data = _exchange_code(params["code"])

            id_info = id_token.verify_oauth2_token(
                token_data["id_token"],
                google_requests.Request(),
                _CLIENT_ID,
            )
            email = id_info.get("email", "")
            name  = id_info.get("name", email)

            if not _is_allowed(email):
                st.query_params.clear()
                _render_login_page(error=f"Access denied: {email} is not authorised.")
                st.stop()
                return

            st.session_state["_auth_email"] = email
            st.session_state["_auth_name"]  = name
            st.session_state["_log_session_start"] = True
            st.query_params.clear()
            st.rerun()

        except Exception as exc:
            st.query_params.clear()
            _render_login_page(error=f"Authentication error — please try again. ({exc})")
            st.stop()
            return

    # ── Not authenticated — show login page ────────────────────────────────────
    _render_login_page()
    st.stop()


def render_user_badge() -> None:
    """Show signed-in email + Sign out button at bottom of sidebar."""
    email = st.session_state.get("_auth_email")
    if not email:
        return
    with st.sidebar:
        st.divider()
        name = st.session_state.get("_auth_name", email)
        st.caption(f"👤 **{name}**  \n{email}")
        if st.button("Sign out", key="_signout_btn", use_container_width=True):
            st.session_state.pop("_auth_email", None)
            st.session_state.pop("_auth_name", None)
            st.rerun()


# ── Login page ─────────────────────────────────────────────────────────────────

def _render_login_page(error: str | None = None) -> None:
    auth_url = _build_auth_url()
    auth_url_attr = _html_lib.escape(auth_url, quote=True)
    error_html = f'<p class="error-msg">{_html_lib.escape(error)}</p>' if error else ""

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap');
        #MainMenu, header, footer {{ visibility: hidden; }}
        .block-container {{ padding-top: 0 !important; }}
        .login-wrap {{
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; min-height: 80vh; text-align: center;
            font-family: 'Montserrat', system-ui, sans-serif;
        }}
        .login-card {{
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 16px;
            padding: 2.8rem 3rem;
            max-width: 420px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        }}
        .login-card h1 {{
            font-size: 1.2rem; font-weight: 600; color: #111111;
            margin-bottom: 0.4rem; line-height: 1.35;
            font-family: 'Montserrat', system-ui, sans-serif;
        }}
        .login-card p {{
            color: #888888; font-size: 0.82rem; margin-bottom: 1.8rem;
            font-family: 'Montserrat', system-ui, sans-serif;
        }}
        .google-btn {{
            display: inline-flex; align-items: center; gap: 10px;
            background: #00d4aa; color: #000000 !important;
            padding: 11px 26px; border-radius: 8px;
            text-decoration: none !important; font-weight: 600;
            font-size: 0.85rem; letter-spacing: 0.02em;
            font-family: 'Montserrat', system-ui, sans-serif;
            transition: background 0.18s;
        }}
        .google-btn:hover {{ background: #00b894; }}
        .error-msg {{ color: #e05050; margin-top: 1rem; font-size: 0.8rem; }}
        </style>
        <div class="login-wrap">
          <div class="login-card">
            <div style="font-size:2rem;margin-bottom:.5rem">🧬</div>
            <h1>Protein Design Tool</h1>
            <p>Sign in with your <strong>@{_ALLOWED_DOMAIN}</strong> account to continue.</p>
            <a href="{auth_url_attr}" class="google-btn">
              <svg width="17" height="17" viewBox="0 0 48 48">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
              </svg>
              Sign in with Google
            </a>
            {error_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
