import os

from typing import List

import openai

from llm_foundation import logger
from llm_foundation.prompts import BASIC_TEXT_SYSTEM_PROMPT


openai.api_key = os.environ['OPENAI_API_KEY']


def get_available_models():
    return openai.models.list()


def get_model_names(filters: List[str] = []):
    filter_f = lambda model: all([model.id.startswith(filter) for filter in filters])
    return [model.id for model in get_available_models() if filter_f(model)]
    
    
def single_text_request(prompt: str, 
                        system_prompt:str = BASIC_TEXT_SYSTEM_PROMPT, 
                        model: str = "gpt4o-mini"):

    completion = openai.ChatCompletion.create(
        model=model,
        # Pre-define conversation messages for the possible roles 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        )
    return completion.choices[0].message
