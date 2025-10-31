import yaml
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config(path: str):
    '''Load yaml of customizable settings'''
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    credentials_path: Path = Path("./credentials.json")
    token_path: Path = Path("./token.json")
    

    gmail_scopes: List[str] = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]

    query: str = "-in:spam -in:trash" # only get emails from INBOX
    max_threads: int = 10 #Convert/Change as needed max_thread

    # API keys (loaded from environment - .env file)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    backend_provider: str = "openai" 
    model_name: str = "gpt-5-nano-2025-08-07" 
    embedding_model: str = "text-embedding-3-small"
    num_judges: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @classmethod
    def from_yaml(cls, yaml_path: str = "./agent_config.yaml"):
        """Load from YAML, then overlay with environment and defaults."""
        
        yaml_data = load_yaml_config(yaml_path)
        #print(f"Loading YAML from {Path(yaml_path).resolve()}: {yaml_data}")
        settings = cls(**yaml_data)

        # Validate backend choice
        if settings.backend_provider not in {"openai", "anthropic"}:
            raise ValueError(
                f"Invalid backend_provider '{settings.backend_provider}'. Must be 'openai' or 'anthropic'."
            )
        return settings


settings = Settings.from_yaml()