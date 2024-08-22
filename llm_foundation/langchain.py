import os
from typing import List

import openai

from langchain_openai import ChatOpenAI

from llm_foundation import logger
from llm_foundation.basic_structs import LMConfig, ModelType


openai.api_key = os.environ['OPENAI_API_KEY']


def get_model_names(filters: List[str] = []):
    return []


def get_lm(config: LMConfig):
    logger.info(f"Creating {config.type} object for model {config.model}")
    match config.type:
        case ModelType.Simple:
            pass
        case ModelType.Chat:
            lm = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        case _:
            pass
        
    logger.info(f"LM object {lm} created")
    return lm
