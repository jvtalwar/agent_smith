import typer
from rich import print
from .gmail_api import client, list_thread_ids, get_thread_summary, create_reply_draft, get_thread_bundle
from .needs_reply_classifier import no_reply_rules, llm_needs_reply_judge
from .models import get_llm
from .settings import settings

app = typer.Typer(help="Gmail agent demo CLI")

@app.command("list")
def get_threads(prior: int = 2):
    #print(f"MAX THREADS: {settings.max_threads}")
    #print(f"MODEL NAME: {settings.model_name}")
    n = settings.max_threads
    s = client()
    ids = list_thread_ids(s, n)
    #print(f"Num in ids: {len(ids)}")
    for tid in ids:
        th = get_thread_bundle(s, tid, prior) #get_thread_summary(s, tid)
        rules_reply = no_reply_rules(th["latest"])
        print(f"Rules: {rules_reply}")
        print(th)
        print("--------------------------------")
        #print(f"[bold]{th['subject'] or '(no subject)'}[/bold]  —  {tid}")
        #print(f"  From: {th['from']}")
        #print(f"  Snip: {th['snippet'][:120]}")
        #print()

@app.command("draft")
def draft(thread_id: str, body: str = "Thanks for your note — replying soon."):
    s = client()
    th = get_thread_summary(s, thread_id)
    did = create_reply_draft(s, th, body)
    print(f"[green]Created draft[/green] {did} for thread {thread_id}")

@app.command("test-model")
def test_open_ai():
    model = get_llm()
    reply = model.generate("Well hello there? How are you today")
    print(reply.text)
    print(reply.raw)

@app.command("test-judge")
def test_llm_as_judge(tid: str, prior: int = 2):
    s = client()
    th = get_thread_bundle(s, tid, prior)
    details, conf = llm_needs_reply_judge(k = settings.num_judges, thread_bundle = th)

    print(details)
    print(conf)

if __name__ == "__main__":
    app()

