'''
About: All core logic of email agent.
'''

import importlib.resources as resources
import time 
import json
import os
from .utils import _get_openai_embedding, _calc_dot_product
from .memories import update_style_embeddings
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
    '''method to generate email draft'''
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


def _save_memory_to_disk(memory_dict, filename):
    '''Helper function to save memory dictionary to disk in memories/ directory'''
    memories_dir = "memories"
    os.makedirs(memories_dir, exist_ok=True)
    filepath = os.path.join(memories_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(memory_dict, f, indent=2)


def post_update_style_memory(model, style_memory, style_profile_name, user_decision, model_draft="", user_final_text=""):
    '''Update style memory based on user decision and optionally refine with LLM on edits'''
    
    style_profile = style_memory[style_profile_name]
    
    # Always update timestamp and weight
    style_profile["last_updated_ts"] = int(time.time())
    
    # Adjust weight based on user decision (additive updates)
    if user_decision == 'a':  # approve
        style_profile["weight"] += 1.0
    elif user_decision == 'e':  # edit
        style_profile["weight"] += 0.3
    elif user_decision == 'r':  # reject
        style_profile["weight"] -= 0.2
        style_profile["weight"] = max(0.01, style_profile["weight"])  # floor at 0.01
    
    log_dict = {"prompt": "", "response": "", "latency_s": 0, "tokens_used": 0}
    
    # On edit, use LLM to extract style differences
    if user_decision == 'e':
        style_prompt_root = resources.files("agent_mail") / "prompts" / "style_memory"
        system = (style_prompt_root / "online_system.txt").read_text(encoding="utf-8")
        user = (style_prompt_root / "online_user.txt").read_text(encoding="utf-8")
        
        # Format existing example_user_edits as bulleted list
        existing_edits = style_profile.get("example_user_edits", [])
        if existing_edits:
            edits_list = "\n".join([f"- {edit}" for edit in existing_edits])
        else:
            edits_list = "(No prior edit summaries)"
        
        prompt = user.format(
            model_generated_text=model_draft,
            user_final_text=user_final_text,
            example_user_edits_list=edits_list
        )
        
        start = time.perf_counter()
        response = model.generate(prompt, system=system, as_json=True, max_tokens=8000)
        latency = time.perf_counter() - start
        
        obj = json.loads(response.text)
        new_edits = obj.get("new_example_user_edits", [])
        
        # Normalize to list
        if isinstance(new_edits, str):
            try:
                new_edits = json.loads(new_edits)
            except json.JSONDecodeError:
                new_edits = [new_edits.strip()]
        elif not isinstance(new_edits, list):
            new_edits = [str(new_edits)]
        
        # Append new edits and cap to last 20 (ring buffer)
        style_profile["example_user_edits"].extend(new_edits)
        style_profile["example_user_edits"] = style_profile["example_user_edits"][-20:]
        
        tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)
        log_dict = {"prompt": prompt, "response": response.text, "latency_s": latency, "tokens_used": tokens_used}
    
    # Save updated style memory to disk
    _save_memory_to_disk(style_memory, "style_memory.json")
    
    return log_dict


def post_update_contact_memory(model, contact_memory, contact_id, model_draft, user_final_text):
    '''Update contact memory based on user edits using LLM to refine profile'''
    
    contact_profile = contact_memory[contact_id]
    
    # Update timestamp
    contact_profile["last_updated_ts"] = int(time.time())
    
    # Load prompts
    contact_prompt_root = resources.files("agent_mail") / "prompts" / "contact_memory"
    system = (contact_prompt_root / "post_system.txt").read_text(encoding="utf-8")
    user = (contact_prompt_root / "post_user.txt").read_text(encoding="utf-8")
    
    # Format preferences for display
    prefs = contact_profile.get("preferences", [])
    if prefs:
        prefs_str = json.dumps(prefs)
    else:
        prefs_str = "[]"
    
    prompt = user.format(
        model_generated_email=model_draft,
        user_edited_email=user_final_text,
        contact_persona=contact_profile.get("persona", ""),
        contact_role=contact_profile.get("role", ""),
        contact_preferences=prefs_str
    )
    
    start = time.perf_counter()
    response = model.generate(prompt, system=system, as_json=True, max_tokens=8000)
    latency = time.perf_counter() - start
    
    obj = json.loads(response.text)
    
    # Update contact profile
    contact_profile["persona"] = str(obj.get("persona", "")).strip()
    contact_profile["role"] = str(obj.get("role", "")).strip()
    
    updated_prefs = obj.get("preferences", [])
    # Normalize to list
    if isinstance(updated_prefs, str):
        try:
            updated_prefs = json.loads(updated_prefs)
        except json.JSONDecodeError:
            updated_prefs = [updated_prefs.strip()]
    elif not isinstance(updated_prefs, list):
        updated_prefs = [str(updated_prefs)]
    
    # Cap preferences to last 20 (ring buffer)
    contact_profile["preferences"] = updated_prefs[-20:]
    
    tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)
    
    return {"prompt": prompt, "response": response.text, "latency_s": latency, "tokens_used": tokens_used}


def offline_update_style_memory(model, style_memory):
    '''Update style memory profiles based on accumulated user edits using LLM refinement'''
    
    log_dict = {}
    
    # Load prompts once
    style_prompt_root = resources.files("agent_mail") / "prompts" / "style_memory"
    system = (style_prompt_root / "offline_system.txt").read_text(encoding="utf-8")
    user_template = (style_prompt_root / "offline_user.txt").read_text(encoding="utf-8")
    
    # Iterate through each profile
    for profile_name, profile_data in style_memory.items():
        example_edits = profile_data.get("example_user_edits", [])
        
        # Skip if no edits to learn from
        if not example_edits:
            continue
        
        # Format edits list for prompt
        edits_list = "\n".join([f"- {edit}" for edit in example_edits])
        
        # Format user prompt with current profile fields
        prompt = user_template.format(
            opening=profile_data.get("opening", ""),
            signoff=profile_data.get("signoff", ""),
            signature=profile_data.get("signature", ""),
            tone=profile_data.get("tone", ""),
            guidance=profile_data.get("guidance", ""),
            emoji_policy=profile_data.get("emoji_policy", "Never"),
            example_user_edits_list=edits_list
        )
        
        # Call LLM to refine profile
        start = time.perf_counter()
        response = model.generate(prompt, system=system, as_json=True, max_tokens=8000)
        latency = time.perf_counter() - start
        
        obj = json.loads(response.text)
        
        # Update all profile fields
        profile_data["opening"] = str(obj.get("opening", "")).strip()
        profile_data["signoff"] = str(obj.get("signoff", "")).strip()
        profile_data["signature"] = str(obj.get("signature", "")).strip()
        profile_data["tone"] = str(obj.get("tone", "")).strip()
        profile_data["guidance"] = str(obj.get("guidance", "")).strip()
        profile_data["emoji_policy"] = str(obj.get("emoji_policy", "Never")).strip()
        
        # Update timestamp
        profile_data["last_updated_ts"] = int(time.time())
        
        # Clear example edits after consolidation
        profile_data["example_user_edits"] = []
        
        # Log this update
        tokens_used = response.raw.get("usage", {}).get("total_tokens", -1)
        log_dict[profile_name] = {
            "prompt": prompt,
            "response": response.text,
            "latency_s": latency,
            "tokens_used": tokens_used
        }
    
    # Regenerate embeddings for all profiles (only updated ones have changed, but this is safest)
    if len(log_dict) > 0:
        update_style_embeddings(style_memory) #<-- update this later to not regenerate embeddings for everything, even those that are unchanged
    
    # Save updated style memory to disk
    _save_memory_to_disk(style_memory, "style_memory.json")
    
    return log_dict


def offline_decay_contact_memory(contact_memory):
    '''Apply time-based weight decay to contact memory and remove stale contacts'''
    
    current_time = int(time.time())
    seconds_per_week = 7 * 24 * 3600
    #removed_contacts = []
    
    # Create list of contacts to remove (can't modify dict during iteration)
    contacts_to_remove = []
    
    for contact_id, profile in contact_memory.items():
        last_updated = profile.get("last_updated_ts", current_time)
        current_weight = profile.get("weight", 1.0)
        
        # Calculate elapsed time in weeks
        elapsed_seconds = current_time - last_updated
        elapsed_weeks = elapsed_seconds / seconds_per_week
        
        # Apply decay: new_weight = old_weight * (0.95 ^ weeks_elapsed)
        new_weight = current_weight * (0.95 ** elapsed_weeks)
        profile["weight"] = new_weight
        
        # Mark for removal if below threshold
        if new_weight < 0.65:  # ~8 weeks since one-time interaction
            contacts_to_remove.append(contact_id)
    
    # Remove stale contacts
    for contact_id in contacts_to_remove:
        #removed_contacts.append(contact_id)
        del contact_memory[contact_id]
    
    # Save updated contact memory to disk
    _save_memory_to_disk(contact_memory, "contact_memory.json")
    
    return {"removed": contacts_to_remove}


def offline_decay_thread_memory(thread_memory):
    '''Apply time-based weight decay to thread memory and remove stale threads'''
    
    current_time = int(time.time())
    seconds_per_week = 7 * 24 * 3600
    #removed_threads = []
    
    # Create list of threads to remove (can't modify dict during iteration)
    threads_to_remove = []
    
    for thread_id, profile in thread_memory.items():
        last_updated = profile.get("last_updated_ts", current_time)
        current_weight = profile.get("weight", 1.0)
        
        # Calculate elapsed time in weeks
        elapsed_seconds = current_time - last_updated
        elapsed_weeks = elapsed_seconds / seconds_per_week
        
        # Apply decay: new_weight = old_weight * (0.95 ^ weeks_elapsed)
        new_weight = current_weight * (0.95 ** elapsed_weeks)
        profile["weight"] = new_weight
        
        # Mark for removal if below threshold
        if new_weight < 0.75:  # ~6 weeks since one-time interaction
            threads_to_remove.append(thread_id)
    
    # Remove stale threads
    for thread_id in threads_to_remove:
        #removed_threads.append(thread_id)
        del thread_memory[thread_id]
    
    # Save updated thread memory to disk
    _save_memory_to_disk(thread_memory, "thread_memory.json")
    
    return {"removed": threads_to_remove}

    