
from typing import List
from llm_foundation import logger
import llm_foundation.langchain as llm_lc
import llm_foundation.open_ai as llm_openai
from llm_foundation.basic_structs import LMConfig, Provider
    

def get_model_catalog(provider: Provider, name_filters: List[str] = []):
    
    def get_model_names(provider: Provider, filters: List[str] = []):
        match provider:
            case Provider.LC:
                return llm_openai.get_model_names(name_filters)
            case Provider.OpenAI:
                return llm_openai.get_model_names(name_filters)
            case _:
                raise ValueError("Invalid backend provider")

    catalog = {}
    for provider in Provider:
        print(provider, provider.name)
        for model_name in get_model_names(provider):
            catalog[f"{provider.name.lower()}_{model_name}"] = model_name

    return catalog        
        

def get_lm(config: LMConfig):
    logger.info(f"Creating lm object for provider {config.provider.name}")
    match config.provider:
        case Provider.LC:
            lm = llm_lc.get_lm(config)
        case _:
            raise ValueError("Invalid backend provider")

    return lm
