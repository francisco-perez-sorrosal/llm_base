from enum import Enum

from pydantic import BaseModel

class Provider(Enum):
    LC = 0
    # HF = 1
    OpenAI = 2

class ModelType(Enum):
    Simple = 0
    Chat = 1

class LMConfig(BaseModel):
    model: str = "gpt4o-mini"
    provider: Provider = Provider.LC
    type: ModelType = ModelType.Chat
    temperature: float = 0.0
    max_tokens: int = 300
