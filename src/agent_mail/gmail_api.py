from __future__ import annotations
import base64
from email.mime.text import MIMEText
from googleapiclient.discovery import Resource
from .settings import settings
from .gmail_auth import gmail_client

META_HEADERS = [
    "List-Id", "Auto-Submitted", "Reply-To",
    "From", "To", "Cc", "Date", "Subject", "Message-Id"
]

#Function to list the last N threads (from INBOX)
def list_thread_ids(service: Resource, n: int | None = None) -> list[str]:
    n = n or settings.max_threads
    resp = service.users().threads().list(
        userId="me", labelIds=["INBOX"], maxResults=n, q=settings.query
    ).execute()
    return [t["id"] for t in resp.get("threads", [])]

#Function to get the thread summary
def get_thread_summary(service: Resource, thread_id: str) -> dict:
    t = service.users().threads().get(
        userId="me", id=thread_id, format="full", metadataHeaders=META_HEADERS
    ).execute()
    latest = t["messages"][-1]
    headers = {h["name"]: h["value"] for h in latest["payload"].get("headers", [])}
    return {
        "thread_id": thread_id,
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "snippet": latest.get("snippet", ""),
        "message_id": headers.get("Message-Id"),
    }

#Function to extract the message text from the message resource
def extract_message_text(msg):
    """Return decoded plain-text body from a Gmail message resource."""
    payload = msg.get("payload", {})
    if "body" in payload and payload["body"].get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")

    # Walk multipart structure
    for part in payload.get("parts", []):
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        if mime == "text/plain" and "data" in body:
            return base64.urlsafe_b64decode(body["data"]).decode("utf-8")
            
    return ""


def create_reply_draft(service: Resource, thread: dict, body_text: str) -> str:
    subj = thread["subject"]
    if subj and not subj.lower().startswith("re:"):
        subj = f"Re: {subj}"

    msg = MIMEText(body_text)
    msg["To"] = thread.get("from", "")
    msg["Subject"] = subj
    if thread.get("message_id"):
        msg["In-Reply-To"] = thread["message_id"]
        msg["References"] = thread["message_id"]

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw, "threadId": thread["thread_id"]}}
    ).execute()
    return draft["id"]

def client() -> Resource:
    """Convenience shortcut for CLI."""
    return gmail_client()
