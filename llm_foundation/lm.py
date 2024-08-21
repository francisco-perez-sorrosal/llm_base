
from enum import Enum

from llm_foundation import logger
import llm_foundation.langchain as llm_lc
from llm_foundation.lm import LMConfig, Type, Provider
from pydantic import BaseModel

class Provider(Enum):
    LC = "LC"
    HF = "HF"
    OpenAI = "OpenAI"

class Type(Enum):
    Simple = "Simple"
    Chat = "Chat"

class LMConfig(BaseModel):
    model: str = "gpt4o-mini"
    provider: Provider = Provider.LC
    type: Type = Type.Chat
    temperature: float = 0.0
    max_tokens: int = 300
    

def get_lm(config: LMConfig):
    logger.info(f"Creating lm object for provider {config.provider.name}")
    match config.provider:
        case Provider.LC:
            lm = llm_lc.get_lm(config)
        case Provider.HF:
            pass
        case _:
            raise ValueError("Invalid backend provider")

    return lm