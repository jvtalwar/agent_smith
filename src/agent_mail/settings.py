from __future__ import annotations
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    credentials_path: Path = Path("./credentials.json")
    token_path: Path = Path("./token.json")

    gmail_scopes: List[str] = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]

    query: str = "-in:spam -in:trash newer_than:14d" # only get emails from the last 14 days - testing 
    max_threads: int = 15 #Convert to a .env variable 

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

settings = Settings()