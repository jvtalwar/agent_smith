from __future__ import annotations
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from .settings import settings

def get_credentials() -> Credentials:
    creds = None
    token_path = Path(settings.token_path)
    scopes = settings.gmail_scopes

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(settings.credentials_path), scopes
            )
            creds = flow.run_local_server(port=8080, prompt="consent")
        token_path.write_text(creds.to_json())

    return creds

def gmail_client():
    """Returns an authenticated Gmail service client."""
    return build("gmail", "v1", credentials=get_credentials())
