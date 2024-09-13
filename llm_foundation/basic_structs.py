from enum import Enum
from typing import Optional

from pydantic import BaseModel
import os

class Provider(Enum):
    LC = 0
    # HF = 1
    OpenAI = 2
    Autogen = 3

class ModelType(Enum):
    Simple = 0
    Chat = 1

class LMConfig(BaseModel):
    model: str = "gpt4o-mini"
    provider: Provider = Provider.LC
    type: ModelType = ModelType.Chat
    temperature: float = 0.0
    max_tokens: int = 300

    def get_api_key(self) -> Optional[str]:
        match self.provider:
            case Provider.LC:
                return os.getenv("LC_API_KEY")
            case Provider.OpenAI:
                return os.getenv("OPENAI_API_KEY")
            case Provider.Autogen:
                return os.getenv("OPENAI_API_KEY")
            case _:
                raise ValueError("Invalid backend provider")

    def to_autogen(self, tools: Optional[list] = None):
                
        config_list = [{
            "model": self.model,  # model name
            "api_key": self.get_api_key()  # api key
        }]
        
        return {
            "seed": 14,  # seed for caching and reproducibility
            "functions": tools ,                
            "config_list": config_list,  # a list of API configurations
            "temperature": self.temperature,  # temperature for sampling
        }
