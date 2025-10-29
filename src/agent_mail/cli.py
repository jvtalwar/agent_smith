import typer
from rich import print
from .gmail_api import client, list_thread_ids, get_thread_summary, create_reply_draft

app = typer.Typer(help="Gmail agent demo CLI")

@app.command("list")
def list_threads(n: int = 15):
    s = client()
    ids = list_thread_ids(s, n)
    for tid in ids:
        th = get_thread_summary(s, tid)
        print(f"[bold]{th['subject'] or '(no subject)'}[/bold]  —  {tid}")
        print(f"  From: {th['from']}")
        print(f"  Snip: {th['snippet'][:120]}")
        print()

@app.command("draft")
def draft(thread_id: str, body: str = "Thanks for your note — replying soon."):
    s = client()
    th = get_thread_summary(s, thread_id)
    did = create_reply_draft(s, th, body)
    print(f"[green]Created draft[/green] {did} for thread {thread_id}")

if __name__ == "__main__":
    app()
