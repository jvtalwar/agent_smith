'''
About: Contains needs reply rules and LLM as a judge for assessing those emails that pass rules to be evaluated
for needing a reply. 
'''

import time 
import hashlib
import importlib.resources as resources
import random
from pathlib import Path
import os
import json 
from typing import Dict, List
from .models import get_llm

CLASSIFIER_PROMPT_ROOT = resources.files("agent_mail") / "prompts" / "needs_reply" 
SYSTEM = (CLASSIFIER_PROMPT_ROOT /  "system.txt").read_text(encoding="utf-8")  
USER_VARIANTS = sorted(
    (
        p for p in CLASSIFIER_PROMPT_ROOT.iterdir()
        if p.is_file() and p.name.startswith("user_") and p.suffix == ".txt"
    ),
    key=lambda x: x.name
)
USER_VARIANTS = [p.read_text(encoding="utf-8") for p in USER_VARIANTS]

def no_reply_rules(last_email_meta: dict) -> bool:
    """Rules for no-reply
    INPUT: Dictionary of last email meta information ('latest' keyword for a message bundle)
    OUTPUT: False
    """
    from_field = last_email_meta.get("from", "").lower()
    subj = last_email_meta.get("subject", "").lower()
    hdrs = last_email_meta
    

    # Check common no-reply address patterns
    if any(x in from_field for x in ["no-reply", "noreply", "donotreply", "daemon@", "postmaster@", "bounce@", "mailer-daemon", "announcements@"]):
        return True

    # Check meta-information for anything that can be ignored 
    if hdrs.get("auto-submitted", "").lower() in ["auto-generated", "auto-replied"]:
        return True
    if hdrs.get("precedence", "").lower() in ["bulk", "list", "junk"]:
        return True
    if hdrs.get("list_id"):
        return True

    #Check for labels that are associated with no-reply emails and status updates etc.
    no_reply_categories = ["category_promotions", "category_updates", "category_forums", "category_social"]
    category_labels = {lbl.lower() for lbl in last_email_meta.get("labels", [])}
    if any(lbl in category_labels for lbl in no_reply_categories):
        return True

    # Subject heuristics for skipping/no-reply
    skip_words = ["receipt", "invoice", "newsletter", "digest", "confirmation", "failure", "undeliverable", "payment is now scheduled"]
    if any(word in subj for word in skip_words):
        return True

    #Can also add in a set of domains to ignore (e.g., Medium, LinkedIn, Adobe, etc.) --> User specific 

    return False 

def _deterministic_variants(thread_id: str, k: int, max_num_judges: int) -> List[str]:
    '''Vary the user prompt passed to each judge as a f(x) of thread_id 
       (helpful for using models for which temperature cannot be varied)'''
    seed = int(hashlib.sha256(thread_id.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    order = list(range(max_num_judges))
    rng.shuffle(order)
    return [USER_VARIANTS[i] for i in order[:k]]


def llm_needs_reply_judge(k: int, thread_bundle: dict, conf_thresh: float = 0.7):  #logging_thresh: float = 0.15
    '''LLM as a judge to classify (needs_reply) rule-passing emails'''
    max_num_judges = len(USER_VARIANTS)
    if k > max_num_judges:
        k = max_num_judges 

    model = get_llm() #Instantiate LLM model

    votes = {"YES":0,"NO":0,"UNSURE":0}

    prior_messages = [msg.get("excerpt", "") for msg in thread_bundle.get("prior_messages", []) if msg.get("excerpt")]
    prior_text = "\n\n---\n\n".join(prior_messages)

    details = list()     
    for tmpl in _deterministic_variants(thread_bundle["latest"]["threadId"], k, max_num_judges):
        prompt = tmpl.format(
            from_=thread_bundle["latest"].get("from",""),
            to=thread_bundle["latest"].get("to",""),
            subject=thread_bundle["latest"].get("subject",""),
            date=thread_bundle["latest"].get("date_header",""),
            labels=", ".join(thread_bundle["latest"].get("labels",[])),
            body=thread_bundle["latest"].get("body_text", ""),
            previous_emails=prior_text
        )

        start = time.perf_counter()
        response = model.generate(prompt, system = SYSTEM, as_json = True, max_tokens = 8000) #reasoning tokens can use a lot of tokens; provide buffer  
        latency = time.perf_counter() - start

        try:
            obj = json.loads(response.text)
            label = str(obj.get("label", "UNSURE")).upper()
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            obj, label, reason = {}, "UNSURE", ""

        if label not in votes: 
            label = "UNSURE"
    
        votes[label] += 1
        details.append({"prompt": prompt, "response": response.text, "label": label, 
                  "latency_s": latency, "tokens_used": response.raw.get("usage", {}).get("total_tokens", -1)})

    total = sum(votes.values())
    conf_num = votes["YES"] + 0.5 * votes["UNSURE"]

    if total == 0: 
        confidence_score = 0.5 #Default to UNSURE
    else:
        confidence_score = conf_num/total

    #if confidence_score TO DO: Add logging logic for borderline cases; or perhaps just generic logging 

    return details, confidence_score
        
    #confidence_score >= conf_thresh #confidence_score, 

