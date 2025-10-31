'''
About: All core logic of email agent
'''

import importlib.resources as resources
import time 
import json

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
    