import typer
import os
import subprocess
import tempfile
from rich import print
from rich.panel import Panel
from rich.console import Console
from .gmail_api import client, list_thread_ids, get_thread_summary, create_reply_draft, get_thread_bundle, get_user_identity
from .needs_reply_classifier import no_reply_rules, llm_needs_reply_judge
from .models import get_llm
from .memories import initialize_style_memory, initialize_contact_memory, initialize_thread_memory, initialize_outcome_memory
from .settings import settings
from .utils import _get_openai_embedding, _calc_dot_product
from .email_agent import pre_update_thread_memory, pre_update_contact_memory, extract_email_style_profile, generate_email_draft

console = Console()

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

    s = client()
    user_name_dict = get_user_identity(s)
    print(user_name_dict)
    style_mem = initialize_style_memory(user_name_dict["display_name"])

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
def test_pre_contact_mem(tid: str, prior: int = 2):
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

@app.command("test-style-match")
def test_style_match(tid: str, prior: int = 2):
    s = client()
    th = get_thread_bundle(s, tid, prior)

    model = get_llm()

    style_memory = initialize_style_memory(display_name = th["latest"]["from"])

    #print(f"INITIALIZED: {style_memory}")
    
    style_profile, log_info = extract_email_style_profile(model, style_memory, th)

    print(f"STYLE: {style_profile}")
    print(f"LOGS: {log_info}")


@app.command("draft")
def draft_email(tid: str, prior: int = 2): #replace tid with a full list of tids later after testing; pass in display name and model here as well 
    print(f"Drafting an email for thread {tid}...\n")
    s = client()
    user_name_dict = get_user_identity(s)
    display_name = user_name_dict["display_name"] #this will be moved outside and passed back in later as an input to draft <--
    model = get_llm()

    gmail_bundle = get_thread_bundle(s, tid, prior) 

    contact_id = gmail_bundle["latest"]["from"]

    #These will all be initialized once outside and passed in a dict of memories here later <--
    style_memory = initialize_style_memory(display_name = display_name)
    contact_memory = initialize_contact_memory()
    thread_memory = initialize_thread_memory()
    outcome_memory = initialize_outcome_memory() #deque

    #Update all relevant information:
    print(f"PRE-DRAFT: Updating thread memory...\n")
    pre_thread_mem_log = pre_update_thread_memory(model, thread_memory, gmail_bundle)

    print(f"PRE-DRAFT: Updating contact_memory...\n")
    pre_contact_mem_log = pre_update_contact_memory(model, contact_memory, gmail_bundle)

    print(f"PRE_DRAFT: Extracting max semantic similarity style profile from style memory...\n")
    style_profile, query_style_log = extract_email_style_profile(model, style_memory, gmail_bundle)

    contact_profile = contact_memory[gmail_bundle["latest"]["from"]]
    thread_profile = thread_memory[gmail_bundle["thread_id"]]

    print("DRAFTING EMAIL...")
    email_draft, email_draft_logs = generate_email_draft(model, style_profile, contact_profile, thread_profile, display_name, gmail_bundle)
    
    # Display the generated email to the user
    subject = gmail_bundle["latest"]["subject"]
    if subject and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]Generated Email Draft[/bold cyan]")
    console.print("="*80)
    console.print(f"[bold]Subject:[/bold] {subject}")
    console.print("-"*80)
    console.print(Panel(email_draft, title="Message Body", border_style="green"))
    console.print("="*80 + "\n")
    
    # Prompt user for decision
    user_decision = ""
    while user_decision not in ['a', 'e', 'r']:
        user_input = input("[A]pprove, [E]dit, or [R]eject? ").strip().lower()
        if user_input in ['a', 'e', 'r']:
            user_decision = user_input
        else:
            print("[red]Invalid input. Please enter A, E, or R.[/red]")
    
    user_final_text = ""
    
    # Handle Edit option
    if user_decision == 'e':
        # Create a temporary file with the draft
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write(email_draft)
            temp_path = tf.name
        
        try:
            # Determine which editor to use
            editor = os.environ.get('EDITOR', 'nano')
            # If editor env var doesn't exist or is empty, try nano, then vim
            if not editor:
                editor = 'nano'
            
            # Launch the editor
            subprocess.call([editor, temp_path])
            
            # Read back the edited content
            with open(temp_path, 'r') as tf:
                user_final_text = tf.read()
            
            print("[green]Edit complete![/green]")
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    # Push draft to Gmail if approved or edited
    if user_decision in ['a', 'e']:
        body_text = email_draft if user_decision == 'a' else user_final_text
        
        # Create thread dict compatible with create_reply_draft
        thread_dict = {
            "thread_id": gmail_bundle["thread_id"],
            "subject": gmail_bundle["latest"]["subject"],
            "from": gmail_bundle["latest"]["from"],
            "message_id": gmail_bundle["latest"]["message_id"]
        }
        
        draft_id = create_reply_draft(s, thread_dict, body_text)
        console.print(f"[bold green]✓ Draft created successfully![/bold green] Draft ID: {draft_id}")
    elif user_decision == 'r':
        console.print("[yellow]Draft rejected. No changes made to Gmail.[/yellow]")
    
    # Print debug logs
    print(f"\n[dim]--- Debug Logs ---[/dim]")
    print(f"EMAIL DRAFT LOGS: {email_draft_logs}")
    print(f"THREAD LOGS: {pre_thread_mem_log}")
    print(f"CONTACT LOGS: {pre_contact_mem_log}")
    print(f"STYLE LOGS: {query_style_log}")



if __name__ == "__main__":
    app()

