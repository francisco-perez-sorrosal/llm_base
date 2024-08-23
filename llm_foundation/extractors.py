from abc import ABC
from typing import List, Optional, Sequence, Union, get_origin

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel

from llm_foundation import logger


class Citation(BaseModel):
    """Information about papers mentioned in a text document."""
    title: str
    author: Optional[str]
    year: Optional[int]


class CitedDocuments(BaseModel):
    """Information to extract"""
    cites: List[Citation]


DEFAULT_SYS_TEMPLATE = """A text document will be passed to you. Extract from it all papers \
that are mentioned/cited by the text. \
Do not extract the name of the article itself. If no papers are mentioned \
you don't need to extract any. Just return an empty list. \
Do not make up or guess ANY extra information. Only extract what exactly is in the text."""


class WebPageContentExtractor(ABC):
    
    def __init__(self, lm, entity_to_extract: BaseModel, sys_template: str):
        self.lm = lm
        self.entity_to_extract: BaseModel = entity_to_extract
        
        self.sys_template = sys_template
        
        # This extracts the first list attribute from the entity to extract (e.g. cites in CitedDocuments)
        # This will allow later to flatten the list of extractions
        self.first_list_attribute = None
        for name, field in self.entity_to_extract.model_fields.items():
            if get_origin(field.annotation) is list:
                self.first_list_attribute = name
                break
        
        # Preparing function call
        openai_fn_desc = convert_to_openai_function(self.entity_to_extract)
        extraction_function = [
            openai_fn_desc
        ]
        fn_call = {"name": openai_fn_desc["name"]}

        self.lm = self.lm.bind(
            functions=extraction_function,
            function_call=fn_call
        )

    def extract(self, source_web_docs: Union[str, Sequence[str]]):
        """Extracts information from a list of web documents."""
        
        def flatten(matrix):
            return [item for sublist in matrix for item in sublist]
                
        # Load documents
        loader = WebBaseLoader(source_web_docs)
        documents = loader.load()
        
        # TODO: Look for better splitters
        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)

        # Building chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.sys_template),
            ("user", "{input}")
        ])

        parser_instance = JsonKeyOutputFunctionsParser(key_name=self.first_list_attribute) if self.first_list_attribute else JsonOutputFunctionsParser()
        extraction_chain = prompt | self.lm | parser_instance
        prep = RunnableLambda(
            lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
        )
        # First element in the chain has to be a Runnable lambda
        final_chain = prep | extraction_chain.map() | flatten
        
        # Extraction
        extracted_docs = {}
        logger.info(f"Extracting {self.entity_to_extract}")
        for i, document in enumerate(documents):
            logger.info(f"Document len: {len(document.page_content)}")
            # Remove empty lines from document.page_content
            content = "\n".join(line for line in document.page_content.split("\n") if line.strip())
            logger.info(f"Document len without empty lines: {len(content)}")
            doc_id = document.metadata.get("source", f"unknown_doc_{i}")            
            extracted_docs[doc_id] = final_chain.invoke(content)

        return extracted_docs

    @classmethod
    def build_tool(cls, lm, entity_to_extract: BaseModel, sys_template: str = DEFAULT_SYS_TEMPLATE):
        extractor = WebPageContentExtractor(lm, entity_to_extract, sys_template)
        return StructuredTool.from_function(extractor.extract)
