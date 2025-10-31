import time
import logging
import json
import os
from .utils import _get_openai_embedding
from collections import deque

def _load_or_create_memory(filename, default_value):
    """
    Helper function to load memory from JSON file or return default value.
    
    Args:
        filename (str): Name of the JSON file (e.g., 'style_memory.json')
        default_value: Value to return if file doesn't exist
    
    Returns:
        The loaded memory dict or the default_value
    """
    memories_dir = "memories"
    filepath = os.path.join(memories_dir, filename)
    
    if os.path.exists(filepath):
        if default_value == []: #load outcome memory jsonl
            records = []
            with open(filepath, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return records

        else: 
            with open(filepath, 'r') as f:
                return json.load(f)
        
    # Ensure memories directory exists for future saves
    os.makedirs(memories_dir, exist_ok=True)
    return default_value

def update_style_embeddings(style_memory):
    """
    Generate and update embeddings for each style profile in place.

    Args:
        style_memory (dict): A dict of style profiles keyed by profile name.
    Returns:
        dict: The updated style_profiles dict (same object, updated in place).
    """
    for name, profile in style_memory.items():
        concat_text = ' '.join([
                "Guidance:", profile.get("guidance", ""),
                "Tone:", profile.get("tone", ""),
                "Opening:", profile.get("opening", ""),
                "Signature:", profile.get("signature", "")
            ]).strip()
        
        embedding = _get_openai_embedding(concat_text)
        profile["embedding"] = embedding

def initialize_style_memory(display_name: str):
    '''Initialize/Define a style memory; this is used for cold-start
       and will be updated as the agent gets feedback'''
    
    # Check if style memory already exists
    existing_memory = _load_or_create_memory('style_memory.json', None)
    if existing_memory not in (None, {}):
        return existing_memory

    ind_name = display_name.strip()
    current_time = int(time.time())

    style_profiles = {
    "informal_personal": {
        "opening": "Hey",
        "signature": ind_name,  
        "signoff": "-",
        "guidance": (
            "Use for friends and family. Keep tone relaxed, warm, and conversational. "
            "Contractions and light humor are fine. Avoid corporate phrasing or formality."
        ),
        "tone": "Casual, friendly, expressive, approachable.",
        "embedding": None,
        "weight": 1.0,
        "last_updated_ts": current_time,
        "emoji_policy": "Never",
        "example_user_edits": [] #keep track of EDIT notes for updating style memory
        },
    "formal_internal": {
        "opening": "Hi",
        "signature": ind_name,
        "signoff": "Best",
        "guidance": (
            "Use for coworkers, teammates, or internal company communications. "
            "Keep it professional yet concise. Express appreciation or next steps clearly."
        ),
        "tone": "Polite, cooperative, and efficient without being stiff.",
        "embedding": None,
        "weight": 1.0,
        "last_updated_ts": current_time,
        "emoji_policy": "Never",
        "example_user_edits": []
        },
    "formal_external": {
        "opening": "Dear",
        "signature": ind_name,
        "signoff": "Sincerely",
        "guidance": (
            "Use for clients, recruiters, vendors, or other external stakeholders. "
            "Maintain full professionalism and clarity. Avoid contractions or slang. "
            "Include context and next steps when appropriate."
        ),
        "tone": "Professional, courteous, confident, and respectful.",
        "embedding": None,
        "weight": 1.0,
        "last_updated_ts": current_time,
        "emoji_policy": "Never",
        "example_user_edits": []
        },
    "neutral_general": {
        "opening": "Hi",
        "signature": ind_name,
        "signoff": "Best",
        "guidance": (
            "Use when the relationship or formality level is uncertain. "
            "Neutral tone suitable for first-time contacts or mixed audiences. "
            "Balance clarity with warmth; avoid over-formality."
        ),
        "tone": "Balanced, courteous, adaptable, and succinct.",
        "embedding": None,
        "weight": 1.0,
        "last_updated_ts": current_time,
        "emoji_policy": "Never",
        "example_user_edits": []
        },
    "apology_supportive": {
        "opening": "Hi",
        "signature": ind_name,
        "signoff": "Thank you",
        "guidance": (
            "Use when addressing delays, issues, or customer concerns. "
            "Lead with empathy and ownership; reassure and outline resolution steps. "
            "Keep tone calm and constructive."
        ),
        "tone": "Empathetic, accountable, and reassuring.",
        "embedding": None,
        "weight": 1.0,
        "last_updated_ts": current_time,
        "emoji_policy": "Never",
        "example_user_edits": []
        }
    }

    #Generate and update embeddings for each profile
    update_style_embeddings(style_memory = style_profiles)

    return style_profiles

def initialize_contact_memory():
    '''Initialize/Load contact memory. Returns existing memory if found, 
       otherwise returns empty dict.'''
    return _load_or_create_memory('contact_memory.json', {})

def initialize_thread_memory():
    '''Initialize/Load thread memory. Returns existing memory if found, 
       otherwise returns empty dict.'''
    return _load_or_create_memory('thread_memory.json', {})

def initialize_outcome_memory():
    '''Initialize outcome memory. Returns existing memory (jsonl) if found,
    otherwise an empty deque.'''

    loaded = _load_or_create_memory('outcome_memory.jsonl', [])
    loaded.sort(key=lambda e: e.get("last_updated_ts", 0), reverse=True) #guarantee newest elements are at front

    return deque(loaded, maxlen = 100)






