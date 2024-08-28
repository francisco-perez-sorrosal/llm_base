from abc import ABC
from collections import defaultdict
from typing import List, Dict, Optional, Sequence, Union, get_origin

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser, PydanticOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.pydantic_v1 import BaseModel, Field

from llm_foundation import logger


class Citation(BaseModel):
    """Information about papers mentioned in a text document."""
    title: str
    author: Optional[str]
    year: Optional[int]


class CitedDocuments(BaseModel):
    """Information to extract"""
    cites: List[Citation]


DEFAULT_WEB_EXTRACTOR_SYS_TEMPLATE = """A text document will be passed to you. Extract from it all papers \
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
    def build_tool(cls, lm, entity_to_extract: BaseModel, sys_template: str = DEFAULT_WEB_EXTRACTOR_SYS_TEMPLATE):
        extractor = WebPageContentExtractor(lm, entity_to_extract, sys_template)
        return StructuredTool.from_function(extractor.extract)


class Step(BaseModel):
    """Information about a plan step, referencing its parents step ids (dependent steps) and the children step ids (other steps this node depends on)."""
    id: str
    description: str
    dependent_steps: List[str] = Field(..., description="A list of step ids representing the actions that will happen after what is described this step. If the list is empty, the step is a root node.")
    depending_steps: List[str] = Field(..., description="A list of step ids representing the actions that need to happen before what is described this step. If the list is empty, the step is a leaf node.")
    

class Plan(BaseModel):
    """Plan description depicted as a directed acyclic graph (DAG) where each step has a causal dependency on other steps."""
    plan: List[Step]

class MultiTreePlan(BaseModel):
    root_nodes: List[Step]
    
    def traverse_depth_first(self):
        """Traverses the root nodes of the MultiTreePlan in depth-first order."""
        visited = set()

        def dfs(node):
            visited.add(node.id)
            print(node.id)

            for child in node.depending_steps:
                if child.id not in visited:
                    dfs(child)

        for root in self.root_nodes:
            dfs(root)

    def traverse_breadth_first(self):
        """Traverses the root nodes of the MultiTreePlan in breadth-first order."""
        visited = set()
        queue = []

        for root in self.root_nodes:
            queue.append(root)

        while queue:
            node = queue.pop(0)
            visited.add(node.id)
            print(node.id)

            for child in node.depending_steps:
                if child.id not in visited:
                    queue.append(child)

    def traverse_leafs_first_dependants(self):
        """Traverses the root nodes of the MultiTreePlan in depth-first order."""
        visited = set()
        leaf_steps = []
        dependant_steps = []

        def dfs(node):
            visited.add(node.id)

            if not node.depending_steps:
                leaf_steps.append(node)
            else:
                for child in node.depending_steps:
                    if child.id not in visited:
                        dfs(child)
                dependant_steps.append(node)

        for root in self.root_nodes:
            dfs(root)

        return leaf_steps + dependant_steps
    
    def print_multi_tree(self):
        """Prints the multi-root node trees in a visually appealing format."""
        def print_node(node, indent):
            print(f"{indent}{node.id}: {node.description}")
            for child in node.depending_steps:
                print_node(child, indent + "  ")

        for root in self.root_nodes:
            print_node(root, "")
    

DEFAULT_PLAN_EXTRACTOR_SYS_TEMPLATE = """You are an expert in planning. When a text document depicting a plan is passed to you, \
you are very skilled at extracting the hierarchical plan steps from it. \
The hierarchycal structure can be described either in the form of a tree or a directed acyclic graph (DAG) \
Each step has a description and dependencies (parents and children) on other steps. \
First, assign a unique id to each step in the plan. \
Then think carefully about what steps need to happen before in the plan in order to assign the correct parents and children ids to each step. \
Return the list of steps that conform the plan. \
If no plan steps are mentioned or identified, just return an empty list. \
Do not make up or guess ANY extra information. Only extract the steps exactly as they are in the text."""


class PlanExtractor(ABC):
    
    def __init__(self, lm, plan: BaseModel, sys_template: str, use_pydantic_output: bool = False):
        self.lm = lm
        self.plan: BaseModel = plan
        self.sys_template = sys_template
        self.use_pydantic_output = use_pydantic_output
        
        # Preparing function call
        openai_fn_desc = convert_to_openai_function(self.plan)
        extraction_function = [
            openai_fn_desc
        ]
        fn_call = {"name": openai_fn_desc["name"]}

        self.lm = self.lm.bind(
            functions=extraction_function,
            function_call=fn_call
        )

    def extract(self, text: str):
        """Extracts from the text passed an itemized list of plan steps."""
        
        # Building chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.sys_template),
            ("user", "Extract the hierarchical plan steps from this text: {input}")
        ])
        parser = PydanticOutputFunctionsParser(pydantic_schema=self.plan) if self.use_pydantic_output else JsonOutputFunctionsParser()
        extraction_chain = prompt | self.lm | parser
        
        # Extraction
        logger.info(f"Extracting plan steps")
        extracted_steps = extraction_chain.invoke(text)

        return extracted_steps

    @classmethod
    def build_tool(cls, lm, entity_to_extract: BaseModel, sys_template: str = DEFAULT_PLAN_EXTRACTOR_SYS_TEMPLATE, use_pydantic_output:bool = False):
        extractor = PlanExtractor(lm, entity_to_extract, sys_template, use_pydantic_output)
        return StructuredTool.from_function(extractor.extract)
    
    @classmethod
    def build_multi_tree_plan(cls, data: Union[dict, Plan]) -> MultiTreePlan:
        # Convert json object to Plan
        if isinstance(data, dict):
            data = Plan.parse_obj(data)
            
        id_to_step = {step.id: step for step in data.plan}
        child_to_parents = defaultdict(list)

        # Track parent-child relationships
        for step in data.plan:
            for dep in step.depending_steps:
                child_to_parents[dep].append(step.id)

        # Identify root nodes (steps without any parent dependencies)
        root_nodes = [step for step in data.plan if step.id not in child_to_parents]

        # Build the tree by linking steps to their children
        for step in data.plan:
            step.depending_steps = [id_to_step[dep] for dep in step.depending_steps]

        return MultiTreePlan(root_nodes=root_nodes) 
