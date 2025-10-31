'''
About: All core logic of email agent
'''

import importlib.resources as resources
import time 
import json
from .utils import _get_openai_embedding, _calc_dot_product
import numpy as np

def pre_update_thread_memory(model, thread_memory, gmail_bundle):
    '''Update summary, and open actions in thread memory'''

    thread_id = gmail_bundle["thread_id"]
    if thread_id in thread_memory:
        thread_memory[thread_id]["last_updated_ts"] =  int(time.time())
        thread_memory[thread_id]["weight"] += 0.05 #bump weight up for usage; longer TTL

    else:
        thread_memory[thread_id] = {"summary": "", 
                                    "open_actions": "", 
                                    "weight": 1.0, 
                                    "last_updated_ts": int(time.time())}

    thread_prompt_root = resources.files("agent_mail") / "prompts" / "thread_memory" 
    system = (thread_prompt_root /  "system.txt").read_text(encoding="utf-8") 
    user = (thread_prompt_root / "user.txt").read_text(encoding="utf-8") 

    prior_messages = [msg.get("excerpt", "") for msg in gmail_bundle.get("prior_messages", []) if msg.get("excerpt")][-2:] #cap to at most 2 messages
    prior_text = "\n\n---\n\n".join(prior_messages)

    prompt = user.format(
            prev_summary = thread_memory[thread_id]["summary"], 
            prev_open_actions = thread_memory[thread_id]["open_actions"],
            prev_msg_text = prior_text,
            latest_inbound_body = gmail_bundle["latest"].get("body_text", ""),
        )

    start = time.perf_counter()
    response = model.generate(prompt, system = system, as_json = True, max_tokens = 8000) #reasoning tokens can use a lot of tokens; provide buffer 
    latency = time.perf_counter() - start

    obj = json.loads(response.text)
    updated_summary = str(obj.get("summary", "")).strip()
    updated_open_actions = str(obj.get("open_actions", "")).strip()
    
    thread_memory[thread_id]["summary"] = updated_summary
    thread_memory[thread_id]["open_actions"] = updated_open_actions

    tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)

    return {"prompt": prompt, "response": response.text,  "latency_s": latency, "tokens_used": tokens_used}


def pre_update_contact_memory(model, contact_memory, gmail_bundle):
    contact_id = gmail_bundle["latest"]["from"]

    if contact_id in contact_memory:
        contact_memory[contact_id]["last_updated_ts"] =  int(time.time())
        contact_memory[contact_id]["weight"] += 0.05 #bump weight up for usage; longer TTL

        return {"prompt": "", "response": "",  "latency_s": 0, "tokens_used": 0}

    else: #cold start - need to infer
        contact_memory[contact_id] = {"persona": "", 
                                    "role": "", 
                                    "preferences": [],
                                    "weight": 1.0, 
                                    "last_updated_ts": int(time.time())}

    contact_prompt_root = resources.files("agent_mail") / "prompts" / "contact_memory" 
    system = (contact_prompt_root /  "pre_system.txt").read_text(encoding="utf-8") 
    user = (contact_prompt_root / "pre_user.txt").read_text(encoding="utf-8") 
    
    prompt = user.format(latest_inbound_body = gmail_bundle["latest"].get("body_text", ""))

    start = time.perf_counter()
    response = model.generate(prompt, system = system, as_json = True, max_tokens = 8000) #reasoning tokens can use a lot of tokens; provide buffer 
    latency = time.perf_counter() - start

    obj = json.loads(response.text)
    contact_memory[contact_id]["persona"] = str(obj.get("persona", "")).strip()
    contact_memory[contact_id]["role"] = str(obj.get("role", "")).strip()

    prefs = obj.get("preferences", [])
    # Normalize to list
    if isinstance(prefs, str):
        try:
            # Try to parse if it's a JSON-stringified list
            prefs = json.loads(prefs)
        except json.JSONDecodeError:
            # Fallback: wrap comma-separated or plain string into a list
            prefs = [prefs.strip()]
    elif not isinstance(prefs, list):
        # Fallback if model returned a single object
        prefs = [str(prefs)]

    contact_memory[contact_id]["preferences"] = prefs

    tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)

    return {"prompt": prompt, "response": response.text,  "latency_s": latency, "tokens_used": tokens_used}


def _get_best_style_profile(query, style_memory):
    #get query embedding:
    start = time.perf_counter()
    query_embedding = _get_openai_embedding(query)
    embedding_latency = time.perf_counter() - start

    mapping = dict()
    for k, v in style_memory.items():
        dot = _calc_dot_product(query_embedding, v["embedding"])
        weight = 1 + np.log(1 + max(0, v["weight"])) #If weight is non-zero up-weight log-scale by weight; softer weight update
        mapping[k] = dot * weight
        
    sorted_mapping = sorted(mapping.items(), key = lambda x: x[1], reverse = True)

    #print(sorted_mapping)
    
    return sorted_mapping[0][0], embedding_latency #return style_profile key of max semantic similarity, embedding latency

def extract_email_style_profile(model, style_memory, gmail_bundle):
    message_body = gmail_bundle["latest"].get("body_text", "")

    style_prompt_root = resources.files("agent_mail") / "prompts" / "style_memory" 
    system = (style_prompt_root /  "query_system.txt").read_text(encoding="utf-8") 
    user = (style_prompt_root / "query_user.txt").read_text(encoding="utf-8") 

    prompt = user.format(inbound_email_body = message_body)

    start = time.perf_counter()
    response = model.generate(prompt, system = system, as_json = True, max_tokens = 8000) #reasoning tokens can use a lot of tokens; provide buffer 
    latency = time.perf_counter() - start

    obj = json.loads(response.text) 
    query = str(obj.get("style_query", "concise, polite, and context‑aware")).strip()

    #print(f"Extracted query: {query}")

    style_profile_name, embedding_latency = _get_best_style_profile(query = query, style_memory = style_memory)
    style_profile = style_memory[style_profile_name]

    total_latency = latency + embedding_latency

    tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)

    return style_profile, {"prompt": prompt, "response": response.text,  "latency_s": total_latency, "tokens_used": tokens_used, "style_name": style_profile_name}


def generate_email_draft(model, style_profile, contact_profile, thread_profile, display_name, gmail_bundle):
    message_body = gmail_bundle["latest"].get("body_text", "")

    prior_messages = [msg.get("excerpt", "") for msg in gmail_bundle.get("prior_messages", []) if msg.get("excerpt")]
    prior_text = "\n\n---\n\n".join(prior_messages)

    generate_prompt_root = resources.files("agent_mail") / "prompts" / "generate_email" 
    system = (generate_prompt_root /  "system.txt").read_text(encoding="utf-8") 
    user = (generate_prompt_root / "user.txt").read_text(encoding="utf-8") 

    prompt = user.format(latest_inbound_text = message_body,
                         previous_messages = prior_text,
                         thread_summary = thread_profile["summary"], 
                         open_actions = thread_profile["open_actions"],
                         contact_from = gmail_bundle["latest"]["from"], 
                         contact_persona = contact_profile["persona"],
                         contact_role = contact_profile["role"],
                         contact_preferences = contact_profile["preferences"],
                         style_opening = style_profile["opening"],
                         style_signoff = style_profile["signoff"], 
                         style_signature = style_profile["signature"], 
                         style_tone = style_profile["tone"], 
                         style_guidance = style_profile["guidance"], 
                         emoji_policy = style_profile["emoji_policy"], 
                         user_name = display_name)

    start = time.perf_counter()
    response = model.generate(prompt, system = system, as_json = True, max_tokens = 12000) #reasoning tokens can use a lot of tokens; provide buffer 
    latency = time.perf_counter() - start

    obj = json.loads(response.text) 
    email_draft = str(obj.get("email_draft", ""))
    reasoning_bullets = obj.get("reasoning_bullets", [])

    # Normalize to list
    if isinstance(reasoning_bullets, str):
        try:
            # Try to parse if it's a JSON-stringified list
            reasoning_bullets = json.loads(reasoning_bullets)
        except json.JSONDecodeError:
            # Fallback: wrap comma-separated or plain string into a list
            reasoning_bullets = [reasoning_bullets.strip()]
    elif not isinstance(reasoning_bullets, list):
        # Fallback if model returned a single object
        reasoning_bullets = [str(reasoning_bullets)]

    tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)

    return email_draft, {"prompt": prompt, "response": response.text,  "latency_s": latency, "tokens_used": tokens_used,
                         "reasoning_bullets": reasoning_bullets}