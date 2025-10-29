from dataclasses import dataclass
from typing import Optional
from .settings import settings

@dataclass
class LLMResponse:
    text: str
    raw: Optional[dict] = None

class OpenAILLM:
    """OpenAI chat completion wrapper"""
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, *, system=None, max_tokens=500, temperature=0.2, as_json: bool = False) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
        "model": self.model,
        "messages": messages,
        "max_completion_tokens": max_tokens #NOTE: legacy models 3.5 and 4 turbo still use max_tokens =
        }

        #nano/mini models only allow temperature=1 --> Handle here to allow temp setting if allowed
        if not any(x in self.model for x in ["nano", "mini"]):
            params["temperature"] = temperature

        if as_json:
            params["response_format"] = {"type": "json_object"}  # enforce valid JSON

        resp = self.client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, raw=resp.model_dump())


class AnthropicLLM:
    """Anthropic Claude wrapper"""
    def __init__(self, model: str, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, *, system=None, max_tokens=500, temperature=0.2, as_json: bool = False) -> LLMResponse:
        params = params = {
            "model": self.model,
            "system": system or "",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,   
        }

        if temperature is not None:
            params["temperature"] = temperature

        if as_json:
            params["response_format"] = {"type": "json_object"}  # enforce valid JSON

        resp = self.client.messages.create(**params)    
        text = resp.content[0].text if resp.content else ""
        raw = resp.model_dump()

        # Add total_tokens to mirror OpenAI’s structure
        if "usage" in raw:
            usage = raw["usage"]
            usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        return LLMResponse(text=text, raw=raw)


def get_llm():
    '''Return configured/instantiated model - no checks currently that model_name is in the provider so 
       errors may be raised in .generate() calls if user in config specifies an invalid
       model_name.
       '''
    if settings.backend_provider == "openai":
        return OpenAILLM(settings.model_name, settings.openai_api_key)
    else:
        return AnthropicLLM(settings.model_name, settings.anthropic_api_key)
    