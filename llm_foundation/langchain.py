import os

import openai

from langchain_openai import ChatOpenAI

from llm_foundation import logger
from llm_foundation.lm import LMConfig, Type


openai.api_key = os.environ['OPENAI_API_KEY']

def get_lm(config: LMConfig):
    logger.info(f"Creating {config.type} object for model {config.model}")
    match config.type:
        case Type.Simple:
            pass
        case Type.Chat:
            lm = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        case _:
            pass
        
    logger.info(f"LM object {lm} created")
    return lm
