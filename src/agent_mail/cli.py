import typer
from rich import print
from .gmail_api import client, list_thread_ids, get_thread_summary, create_reply_draft, get_thread_bundle
from .needs_reply_classifier import no_reply_rules, llm_needs_reply_judge
from .models import get_llm
from .memories import initialize_style_memory, initialize_contact_memory, initialize_thread_memory, initialize_outcome_memory
from .settings import settings
from .utils import _get_openai_embedding, _calc_dot_product
from .email_agent import pre_update_thread_memory, pre_update_contact_memory

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

@app.command("test-draft")
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


@app.command("test-style")
def test_style():
    test_query = "Replying to customer who is unhappy."

    style_mem = initialize_style_memory("'Cursor Team <hi@mail.cursor.com>'")

    print(f"STYLE MEM: {style_mem}")

    test_emb = _get_openai_embedding(test_query)

    for k, v in style_mem.items():
        dor = _calc_dot_product(test_emb, v["embedding"])
        print(f"{k}: {dor}")


@app.command("test-pre-thread")
def test_pre_thread_mem(tid: str, prior: int = 2):
    s = client()
    th = get_thread_bundle(s, tid, prior)

    model = get_llm()

    thread_memory = initialize_thread_memory()
    print(thread_memory)
    returned_dict = pre_update_thread_memory(model = model, thread_memory = thread_memory, gmail_bundle = th)

    print(thread_memory)

    print(returned_dict)

@app.command("test-pre-contact")
def test_pre_thread_mem(tid: str, prior: int = 2):
    s = client()
    th = get_thread_bundle(s, tid, prior)

    model = get_llm()

    contact_memory = {
    'James Talwar <jtalwar@ucsd.edu>': {
        'persona': 'Coworker offering to chat.',
        'role': 'coworker',
        'preferences': [],
        'weight': 1.0,
        'last_updated_ts': 1761871098
    }
}#initialize_contact_memory()
    print(contact_memory)
    returned_dict = pre_update_contact_memory(model = model, contact_memory = contact_memory, gmail_bundle = th)

    print(contact_memory)

    print(returned_dict)


if __name__ == "__main__":
    app()

