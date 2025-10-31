from __future__ import annotations
import base64
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from googleapiclient.discovery import Resource
from .settings import settings
from .gmail_auth import gmail_client
from email_reply_parser import EmailReplyParser


META_HEADERS = [
    "List-Id", "Auto-Submitted", "Reply-To",
    "From", "To", "Cc", "Date", "Subject", "Message-Id"
]

#function to get user display name:
def get_user_identity(service: Resource):
    resp = service.users().settings().sendAs().list(userId="me").execute()
    for entry in resp.get("sendAs", []):
        if entry.get("isDefault"):
            return {
                "email": entry.get("sendAsEmail", ""),
                "display_name": entry.get("displayName", "").strip()
            }

    # fallback: just take the first one
    if resp.get("sendAs"):
        entry = resp["sendAs"][0]
        return {
            "email": entry.get("sendAsEmail", ""),
            "display_name": entry.get("displayName", "").strip()
        }
    return {"email": "", "display_name": ""}  # extreme fallback

#Function to list the last N threads (from INBOX)
def list_thread_ids(service: Resource, n: int | None = None) -> list[str]:
    n = n or settings.max_threads
    resp = service.users().threads().list(
        userId="me", labelIds=["INBOX"], maxResults=n, q=settings.query
    ).execute()
    return [t["id"] for t in resp.get("threads", [])]


def _b64url_to_text(data: str) -> str:
    """Pad to ensure decoder works (base64 encoder omits padding to save bytes)"""
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + pad).encode()).decode("utf-8", errors="replace")

def _walk_parts(payload) -> tuple[list[str], list[str]]:
    """Return (plain_texts, html_texts) collected recursively."""
    if not payload:
        return [], []
    body = payload.get("body", {})
    data = body.get("data")
    mime = payload.get("mimeType", "")
    plains, htmls = [], []

    if data and mime.startswith("text/"):
        txt = _b64url_to_text(data)
        if mime == "text/plain":
            plains.append(txt)
        elif mime == "text/html":
            htmls.append(txt)

    for part in (payload.get("parts") or []):
        p2, h2 = _walk_parts(part)
        plains.extend(p2); htmls.extend(h2)

    return plains, htmls

def extract_message_text(msg: dict) -> str:
    """Extract the best-available plain text (or HTML fallback) from a Gmail message."""
    payload = msg.get("payload", {})
    plains, htmls = _walk_parts(payload)
    if plains:
        # heuristics: pick longest plain block
        return max(plains, key=len).strip()
    if htmls:
        html = max(htmls, key=len)
        return BeautifulSoup(html, "html.parser").get_text(" ").strip()
    # single-part with data but no mimeType set
    data = payload.get("body", {}).get("data")
    if data:
        return _b64url_to_text(data).strip()
    return ""

#Relevant metadata 
INTERESTING_HDRS = {
    "From", "To", "Cc", "Reply-To", "Subject", "Date", "Message-Id",
    "List-Id", "Auto-Submitted", "Precedence", "List-Unsubscribe",
    "Return-Path", "Sender", "X-Auto-Response-Suppress"
}

def _hdr_map(msg: dict) -> dict[str, str]:
    """Map of header name -> value for a Gmail message."""
    return {
        h["name"]: h["value"]
        for h in msg.get("payload", {}).get("headers", [])
        if h.get("name") in INTERESTING_HDRS
    }

def get_thread_bundle(service, thread_id: str, prior: int = 2) -> dict:
    """
    Return a dict with: thread_id, message_count, history_id,
    latest (ids, participants, labels, dates, selected headers, full body_text),
    prior_messages (last N before latest), and participants (thread-level sets).
    """
    t = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    msgs = t.get("messages", []) or []
    if not msgs:
        return {"thread_id": thread_id, "message_count": 0, "history_id": t.get("historyId"), "latest": None, "prior_messages": [], "participants": {}}

    # Latest message
    latest = msgs[-1]
    h_latest = _hdr_map(latest)
    latest_message_full = extract_message_text(latest)
    latest_text = EmailReplyParser.read(latest_message_full).reply.strip() #Prevent context bloat for huge email threads

    # Prior N messages (before latest)
    pri = []
    start = max(0, len(msgs) - 1 - prior)
    for m in msgs[start:len(msgs) - 1]:
        hm = _hdr_map(m)
        pri.append({
            "id": m["id"],
            "from": hm.get("From", ""),
            "date_header": hm.get("Date", ""),
            "internal_ts": int(m.get("internalDate", "0")) if "internalDate" in m else None,
            "excerpt": (EmailReplyParser.read(extract_message_text(m)).reply.strip() or m.get("snippet", ""))[:500],
        })

    # Participants across thread (simple de-dup on raw header strings)
    seen = set()
    participants = {"from": [], "to": [], "cc": [], "reply_to": []} #may improve this by removing set check or merging into comprehensive set of participants
    def _add(kind: str, raw: str | None):
        if not raw: return
        key = (kind, raw)
        if key in seen: return
        seen.add(key)
        participants[kind].append(raw)

    for m in msgs:
        hm = _hdr_map(m)
        _add("from", hm.get("From"))
        _add("to", hm.get("To"))
        _add("cc", hm.get("Cc"))
        _add("reply_to", hm.get("Reply-To"))

    # Build bundle
    latest_bundle = {
        "id": latest["id"],
        "threadId": latest.get("threadId"),
        "subject": h_latest.get("Subject", ""),
        "from": h_latest.get("From", ""),
        "to": h_latest.get("To", ""),
        "cc": h_latest.get("Cc", ""),
        "reply_to": h_latest.get("Reply-To", ""),
        "message_id": h_latest.get("Message-Id", ""),
        "list_id": h_latest.get("List-Id", ""),
        "auto_submitted": h_latest.get("Auto-Submitted", ""),
        "precedence": h_latest.get("Precedence", ""),
        "date_header": h_latest.get("Date", ""),
        "internal_ts": int(latest.get("internalDate", "0")) if "internalDate" in latest else None,
        "labels": latest.get("labelIds", []),
        "snippet": latest.get("snippet", ""),
        "body_text": latest_text,
    }

    return {
        "thread_id": t["id"],
        "message_count": len(msgs),
        "history_id": t.get("historyId"),
        "latest": latest_bundle,
        "prior_messages": pri,
        "participants": participants,
    }

#Function to get the thread summary --> REPLACE with cached thread:
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
