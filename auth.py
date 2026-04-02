"""
auth.py — Google OAuth2 gate for Streamlit on Cloud Run.

Usage in app.py:
    from auth import check_auth, render_user_badge
    check_auth()   # call before any st.* content; stops unauthenticated users
    ...
    render_user_badge()  # in sidebar: shows email + sign-out button

Environment variables required:
    GOOGLE_CLIENT_ID      — from GCP Console → APIs & Services → Credentials
    GOOGLE_CLIENT_SECRET  — same credential
    OAUTH_REDIRECT_URI    — must exactly match the URI registered in GCP Console
                            (e.g. https://prot-prompt.tools.phyx44.com)
    ALLOWED_EMAIL_DOMAIN  — e.g. "phyx44.com"  (any @phyx44.com account passes)
    ALLOWED_EMAILS        — optional comma-separated override list

If GOOGLE_CLIENT_ID is not set (local dev), auth is bypassed entirely.
"""

from __future__ import annotations

import os
import secrets

import streamlit as st
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow

# ── Config ─────────────────────────────────────────────────────────────────────

_CLIENT_ID      = os.getenv("GOOGLE_CLIENT_ID", "")
_CLIENT_SECRET  = os.getenv("GOOGLE_CLIENT_SECRET", "")
_REDIRECT_URI   = os.getenv("OAUTH_REDIRECT_URI", "https://prot-prompt.tools.phyx44.com")
_ALLOWED_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "phyx44.com")
_ALLOWED_EMAILS = os.getenv("ALLOWED_EMAILS", "")   # comma-separated, optional override

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_allowed(email: str) -> bool:
    """Return True if the email matches the allowed domain or explicit list."""
    email = email.lower().strip()
    if _ALLOWED_EMAILS:
        allowed = {e.strip().lower() for e in _ALLOWED_EMAILS.split(",")}
        return email in allowed
    return email.endswith(f"@{_ALLOWED_DOMAIN}")


def _make_flow() -> Flow:
    """Build an OAuth2 Flow from env config."""
    client_config = {
        "web": {
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [_REDIRECT_URI],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=_SCOPES,
        redirect_uri=_REDIRECT_URI,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def check_auth() -> None:
    """
    Gate the app behind Google OAuth.

    - If GOOGLE_CLIENT_ID is not configured: no-op (local dev mode).
    - If session already authenticated: returns immediately.
    - If Google has redirected back with ?code=...: exchanges code for token,
      verifies email, stores in session_state, clears query params, reruns.
    - Otherwise: renders the sign-in page and calls st.stop().
    """
    # Local dev — skip auth if credentials not configured
    if not _CLIENT_ID or not _CLIENT_SECRET:
        return

    # Already authenticated this session
    if st.session_state.get("_auth_email"):
        return

    params = st.query_params

    # ── Handle OAuth callback (Google redirected back with ?code=...) ──────────
    if "code" in params:
        try:
            flow = _make_flow()
            flow.fetch_token(code=params["code"])

            id_info = id_token.verify_oauth2_token(
                flow.credentials.id_token,
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
    """
    Show the signed-in user's email in the sidebar with a Sign out button.
    Call this inside render_sidebar() or directly after check_auth().
    Safe to call even if auth is disabled (no-op when no email in session).
    """
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


# ── Login page renderer ────────────────────────────────────────────────────────

def _render_login_page(error: str | None = None) -> None:
    """Render the full-page Google sign-in screen."""
    flow = _make_flow()
    auth_url, _ = flow.authorization_url(
        prompt="select_account",
        access_type="offline",
    )

    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome on the login page */
        #MainMenu, header, footer { visibility: hidden; }
        .block-container { padding-top: 0 !important; }
        .login-wrap {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; min-height: 80vh; text-align: center;
        }
        .login-card {
            background: #0e1117; border: 1px solid #262730;
            border-radius: 16px; padding: 3rem 3.5rem; max-width: 440px;
            box-shadow: 0 4px 32px rgba(0,0,0,0.4);
        }
        .login-card h1 { font-size: 2rem; margin-bottom: 0.25rem; }
        .login-card p  { color: #9099a5; margin-bottom: 1.75rem; }
        .google-btn {
            display: inline-flex; align-items: center; gap: 10px;
            background: #4285F4; color: white !important;
            padding: 12px 28px; border-radius: 8px;
            text-decoration: none !important; font-weight: 600;
            font-size: 15px; transition: background 0.2s;
        }
        .google-btn:hover { background: #3367D6; }
        .error-msg { color: #ff4b4b; margin-top: 1rem; font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    error_html = f'<p class="error-msg">{error}</p>' if error else ""

    st.markdown(
        f"""
        <div class="login-wrap">
          <div class="login-card">
            <h1>🧬</h1>
            <h1>ESM3 Protein<br>Engineering Prompter</h1>
            <p>Sign in with your <strong>@{_ALLOWED_DOMAIN}</strong> account to continue.</p>
            <a href="{auth_url}" class="google-btn">
              <svg width="18" height="18" viewBox="0 0 48 48">
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
