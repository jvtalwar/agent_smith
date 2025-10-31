import typer
import os
import time
import json
import subprocess
import tempfile
import copy
from rich import print
from rich.panel import Panel
from rich.console import Console
from .gmail_api import client, list_thread_ids, get_thread_summary, create_reply_draft, get_thread_bundle, get_user_identity
from .needs_reply_classifier import no_reply_rules, llm_needs_reply_judge
from .models import get_llm
from .memories import initialize_style_memory, initialize_contact_memory, initialize_thread_memory, initialize_outcome_memory, _load_or_create_memory
from .settings import settings
from .utils import _get_openai_embedding, _calc_dot_product
from .email_agent import pre_update_thread_memory, pre_update_contact_memory, extract_email_style_profile, generate_email_draft, post_update_style_memory, post_update_contact_memory, _save_memory_to_disk, offline_update_style_memory, offline_decay_contact_memory, offline_decay_thread_memory

console = Console()

app = typer.Typer(help="Agent Smith: Your CLI Gmail agent")

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


def _save_outcome_memory(outcome_memory, outcome, filename= "outcome_memory.jsonl"):
    # Append to deque (automatically drops oldest if at maxlen)
    outcome_memory.appendleft(outcome)

    # Convert to list sorted by timestamp descending (newest first)
    sorted_records = sorted(list(outcome_memory), key=lambda e: e["last_updated_ts"], reverse=True)

    # Persist to file
    memories_dir = "memories"
    os.makedirs(memories_dir, exist_ok=True)
    filepath = os.path.join(memories_dir, filename)
    
    with open(filepath, "w") as f:
        for record in sorted_records:
            f.write(json.dumps(record) + "\n")


#@app.command("draft")
def draft_email(s, model, memories, gmail_bundle, tid, display_name): #replace tid with a full list of tids later after testing; pass in display name and model here as well 
    subject_email = gmail_bundle["latest"]["subject"]
    console.print(f"[cyan]Drafting an email for Subject: {subject_email} (thread {tid})...[/cyan]\n")
    
    #s = client()

    #user_name_dict = get_user_identity(s)
    #display_name = user_name_dict["display_name"] #this will be moved outside and passed back in later as an input to draft <--
    
    #model = get_llm()

    #gmail_bundle = get_thread_bundle(s, tid, prior) 

    contact_id = gmail_bundle["latest"]["from"]

    #These will all be initialized once outside and passed in a dict of memories here later <--
    style_memory = memories["style_memory"]    #initialize_style_memory(display_name = display_name)
    contact_memory = memories["contact_memory"]  #initialize_contact_memory()
    thread_memory = memories["thread_memory"]   #initialize_thread_memory()
    outcome_memory = memories["outcome_memory"]  #initialize_outcome_memory() #deque

    #Update all relevant information:
    console.print(f"[magenta]PRE-DRAFT: Updating thread memory...[/magenta]\n")
    pre_thread_mem_log = pre_update_thread_memory(model, thread_memory, gmail_bundle)

    console.print(f"[blue]PRE-DRAFT: Updating contact_memory...[/blue]\n")
    pre_contact_mem_log = pre_update_contact_memory(model, contact_memory, gmail_bundle)

    console.print(f"[magenta]PRE-DRAFT: Extracting max semantic similarity style profile from style memory...[/magenta]\n")
    style_profile, query_style_log = extract_email_style_profile(model, style_memory, gmail_bundle)

    contact_profile = contact_memory[gmail_bundle["latest"]["from"]]
    thread_profile = thread_memory[gmail_bundle["thread_id"]]

    console.print("[cyan]DRAFTING EMAIL...[/cyan]")
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
            console.print("[red]Invalid input. Please enter A, E, or R.[/red]")
    
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
            
            console.print("[green]Edit complete![/green]")
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
    

    outcome = {"thread_id": gmail_bundle["thread_id"],
               "contact_email": gmail_bundle["latest"]["from"],
               "style_profile_used": query_style_log["style_name"],
               "status": user_decision,  
               "model_draft": email_draft,
               "user_final_text": user_final_text,
               "last_updated_ts": int(time.time())}

    # Update contact and style memory based on user decision
    console.print(f"\n[cyan]ONLINE CONSOLIDATION AND UPDATING MEMORIES...[/cyan]")
    
    # Update style memory (always, based on user decision) and save
    style_update_log = post_update_style_memory(
        model=model,
        style_memory=style_memory,
        style_profile_name=query_style_log["style_name"],
        user_decision=user_decision,
        model_draft=email_draft,
        user_final_text=user_final_text
    )
    
    # Update contact memory only on edit
    contact_update_log = {}
    if user_decision == 'e':
        contact_update_log = post_update_contact_memory(
            model=model,
            contact_memory=contact_memory,
            contact_id=contact_id,
            model_draft=email_draft,
            user_final_text=user_final_text
        )
    
    # Save contact memory to disk
    _save_memory_to_disk(contact_memory, "contact_memory.json")
    
    #Save thread memory:
    _save_memory_to_disk(thread_memory, "thread_memory.json")

    #Save outcomes to outcome memory
    _save_outcome_memory(outcome_memory, outcome)

    #print(f"[green]✓ Memories updated and saved to disk.[/green]\n")

    # Print debug logs
    #print(f"\n[dim]--- Debug Logs ---[/dim]")
    #print(f"EMAIL DRAFT LOGS: {email_draft_logs}\n")
    #print(f"THREAD LOGS: {pre_thread_mem_log}\n")
    #print(f"CONTACT LOGS: {pre_contact_mem_log}\n")
    #print(f"STYLE LOGS: {query_style_log}\n")
    #print(f"STYLE UPDATE LOGS: {style_update_log}\n")
    #print(f"CONTACT UPDATE LOGS: {contact_update_log}\n")

    drafting_logs = {"PRE-EMAIL Thread Memory Update:": pre_thread_mem_log,
                     "PRE-EMAIL Contact Memory Update:": pre_contact_mem_log, 
                     "PRE-EMAIL Style Profile Semantic Similarity:": query_style_log,
                     "EMAIL Draft:": email_draft_logs,
                     "ONLINE CONSOLIDATION: Style Memory": style_update_log,
                     "ONLINE CONSOLIDATION: Contact Memory": contact_update_log
                    }

    return drafting_logs

@app.command("run")
def run_agent_smith(prior: int = 2, logging_tolerance: float = 0.15, llm_conf_thresh: float = 0.7):
    """Run gmail reply agent. Identifies emails to reply to, generates drafts for those needing it, passes to user for feedback, creates drafts in gmail."""
    n = settings.max_threads #set number of last N gmail threads
    s = client()

    user_name_dict = get_user_identity(s)
    display_name = user_name_dict["display_name"]

    ids = list_thread_ids(s, n) 
    
    #load model and memories
    model = get_llm()
    style_memory = initialize_style_memory(display_name = display_name)
    contact_memory = initialize_contact_memory()
    thread_memory = initialize_thread_memory()
    outcome_memory = initialize_outcome_memory()
    memories = {"style_memory": style_memory,
                "contact_memory": contact_memory,
                "thread_memory": thread_memory,
                "outcome_memory": outcome_memory
                }

    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    candidates = {"rules_reject": [], "judge_reject": [], "judge_passed": []}

    for tid in ids:
        gmail_bundle = get_thread_bundle(s, tid, prior) 
        subject_email = gmail_bundle["latest"]["subject"]

        rules_reply = no_reply_rules(gmail_bundle["latest"])

        #confidence_score = 0.05 

        if not rules_reply: #rules didn't catch an email - route to llm as judge
            log_borderline = False
            
            judge_logs, confidence_score = llm_needs_reply_judge(k = settings.num_judges, thread_bundle = gmail_bundle)
            timestamp = int(time.time()) #For ease of log check viewing ensure timestamp is same for judge and draft logs

            if (llm_conf_thresh - logging_tolerance) <= confidence_score <= (llm_conf_thresh + logging_tolerance):
                log_borderline = True
                
            if confidence_score >= llm_conf_thresh: #Draft email, present to user, and perform all necessary memory updates
                candidates["judge_passed"].append((subject_email, tid, confidence_score))
                drafting_email_logs = draft_email(s, model, memories, gmail_bundle, tid, display_name)
                
                # Write draft logs
                draft_log_file = os.path.join(logs_dir, f"Email_Draft_{tid}_{timestamp}.log")
                with open(draft_log_file, 'w') as f:
                    json.dump(drafting_email_logs, f, indent=2)

            else:
                candidates["judge_reject"].append((subject_email, tid, confidence_score))
            
            # Write judge logs when borderline
            if log_borderline:
                #timestamp = int(time.time())
                judge_log_file = os.path.join(logs_dir, f"LLM_Judge_{tid}_{timestamp}.log")
                with open(judge_log_file, 'w') as f:
                    json.dump(judge_logs, f, indent=2)

        else:
            candidates["rules_reject"].append((subject_email, tid))

    results_log_file = os.path.join(logs_dir, f"Results_{int(time.time())}.log")
    with open(results_log_file, 'w') as f:
        json.dump(candidates, f, indent=2)

@app.command("show-memories")
def show_memories():
    """Display all stored memories (style, contact, thread, outcome)."""
    memories_dir = "memories"
    
    # Check if memories directory exists and has files
    if not os.path.exists(memories_dir):
        console.print("\n[yellow]No memories directory found.[/yellow]")
        console.print("Memories will be created once you give the agent a trial run with the [bold]draft[/bold] command.\n")
        return
    
    # Check if directory has any JSON/JSONL files
    memory_files = [f for f in os.listdir(memories_dir) if f.endswith(('.json', '.jsonl'))]
    if not memory_files:
        console.print("\n[yellow]No memory files found.[/yellow]")
        console.print("Memories will be created once you give the agent a trial run!\n")
        return
    
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]        Agent Memory System[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    # Load and display Style Memory
    style_memory = _load_or_create_memory('style_memory.json', None)
    if style_memory:
        console.print("[bold magenta]📝 STYLE MEMORY[/bold magenta]")
        console.print("[dim]Writing style profiles learned from your communication patterns[/dim]\n")
        
        # Deep copy and remove embeddings
        style_display = copy.deepcopy(style_memory)
        for profile_name, profile_data in style_display.items():
            if 'embedding' in profile_data:
                profile_data['embedding'] = "[hidden - high dimensional vector]"
        
        console.print(Panel(
            json.dumps(style_display, indent=2),
            title="Style Profiles",
            border_style="magenta"
        ))
        console.print()
    
    # Load and display Contact Memory
    contact_memory = _load_or_create_memory('contact_memory.json', None)
    if contact_memory:
        console.print("[bold green]👥 CONTACT MEMORY[/bold green]")
        console.print("[dim]Information about people you communicate with[/dim]\n")
        
        console.print(Panel(
            json.dumps(contact_memory, indent=2),
            title="Contact Profiles",
            border_style="green"
        ))
        console.print()
    
    # Load and display Thread Memory
    thread_memory = _load_or_create_memory('thread_memory.json', None)
    if thread_memory:
        console.print("[bold blue]💬 THREAD MEMORY[/bold blue]")
        console.print("[dim]Context and summaries of email conversations[/dim]\n")
        
        console.print(Panel(
            json.dumps(thread_memory, indent=2),
            title="Thread Summaries",
            border_style="blue"
        ))
        console.print()
    
    # Load and display Outcome Memory
    outcome_memory = _load_or_create_memory('outcome_memory.jsonl', [])
    if outcome_memory:
        console.print("[bold yellow]📊 OUTCOME MEMORY[/bold yellow]")
        console.print("[dim]History of draft generations and user feedback[/dim]\n")
        
        # Format outcome memory for display
        outcome_display = []
        for record in outcome_memory:
            outcome_display.append({
                "thread_id": record.get("thread_id"),
                "contact": record.get("contact_email"),
                "style_used": record.get("style_profile_used"),
                "status": record.get("status"),
                "timestamp": record.get("last_updated_ts")
            })
        
        console.print(Panel(
            json.dumps(outcome_display, indent=2),
            title=f"Recent Outcomes (showing {len(outcome_display)} records)",
            border_style="yellow"
        ))
        console.print()
    
    console.print("[dim]Tip: These memories improve the agent's performance over time.[/dim]\n")


@app.command("consolidate")
def consolidate_memories():
    """Perform offline consolidation of all memories."""
    memories_dir = "memories"
    
    # Check if memories directory exists and has files
    if not os.path.exists(memories_dir):
        console.print("\n[yellow]No memories to consolidate yet![/yellow]")
        console.print("Try running [bold cyan]agent-mail run[/bold cyan] first to give the agent some experience.")
        console.print("After you've drafted some emails and given feedback, you can consolidate memories at your leisure.\n")
        return
    
    # Check if directory has any JSON/JSONL files
    memory_files = [f for f in os.listdir(memories_dir) if f.endswith(('.json', '.jsonl'))]
    if not memory_files:
        console.print("\n[yellow]No memories to consolidate yet![/yellow]")
        console.print("Try running [bold cyan]agent-mail run[/bold cyan] first to give the agent some experience.")
        console.print("After you've drafted some emails and given feedback, you can consolidate memories at your leisure.\n")
        return
    
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Offline Memory Consolidation[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    #Get display name
    s = client()
    user_name_dict = get_user_identity(s)
    display_name = user_name_dict["display_name"]

    # Load model
    model = get_llm()
    
    # Load all memories
    console.print("[dim]Loading memories...[/dim]\n")
    style_memory = initialize_style_memory(display_name=display_name)  # Will load existing if available
    contact_memory = initialize_contact_memory()
    thread_memory = initialize_thread_memory()
    
    # 1. Update style memory using LLM
    console.print("[bold magenta]📝 Consolidating Style Memory...[/bold magenta]")
    style_logs = offline_update_style_memory(model, style_memory)
    
    if style_logs:
        console.print(f"[green]✓ Updated {len(style_logs)} style profile(s)[/green]")
        for profile_name in style_logs.keys():
            console.print(f"  - {profile_name}")
    else:
        console.print("[dim]  No style profiles needed updating (no user edits accumulated)[/dim]")
    console.print()
    
    # 2. Decay contact memory
    console.print("[bold green]👥 Applying Time Decay to Contact Memory...[/bold green]")
    contact_logs = offline_decay_contact_memory(contact_memory)
    removed_contacts = contact_logs.get("removed", [])
    
    if removed_contacts:
        console.print(f"[yellow]⚠ Removed {len(removed_contacts)} stale contact(s)[/yellow]")
        for contact_id in removed_contacts[:5]:  # Show first 5
            console.print(f"  - {contact_id}")
        if len(removed_contacts) > 5:
            console.print(f"  ... and {len(removed_contacts) - 5} more")
    else:
        console.print("[green]✓ All contacts still active[/green]")
    console.print()
    
    # 3. Decay thread memory
    console.print("[bold blue]💬 Applying Time Decay to Thread Memory...[/bold blue]")
    thread_logs = offline_decay_thread_memory(thread_memory)
    removed_threads = thread_logs.get("removed", [])
    
    if removed_threads:
        console.print(f"[yellow]⚠ Removed {len(removed_threads)} stale thread(s)[/yellow]")
        for thread_id in removed_threads[:5]:  # Show first 5
            console.print(f"  - {thread_id}")
        if len(removed_threads) > 5:
            console.print(f"  ... and {len(removed_threads) - 5} more")
    else:
        console.print("[green]✓ All threads still active[/green]")
    console.print()
    
    # Summary
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold green]✓ Consolidation Complete![/bold green]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"\n[dim]Summary:[/dim]")
    console.print(f"  • Style profiles updated: {len(style_logs)}")
    console.print(f"  • Contacts removed: {len(removed_contacts)}")
    console.print(f"  • Threads removed: {len(removed_threads)}")
    console.print()


if __name__ == "__main__":
    app()

