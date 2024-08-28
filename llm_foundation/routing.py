from abc import ABC
from typing import List, Optional
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain.schema.agent import AgentFinish
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from llm_foundation import logger
from llm_foundation.utils import banner

# def route(result):
#     if isinstance(result, AgentFinish):
#         return result.return_values['output']
#     else:
#         tools = {
#             "search_wikipedia": search_wikipedia, 
#             "get_current_temperature": get_current_temperature,
#         }
#         return tools[result.tool].run(result.tool_input)


class ToolMaster(Runnable[AIMessage, List], ABC):
    
    def __init__(self, available_tools):
        self.tool_executor = ToolExecutor(available_tools)
    
    @banner(text="Pre Call Tool", level=2)
    def pre_call_tool(self, state):
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        return last_message

    @banner(text="Post Call Tool", level=2)
    def post_call_tool(self, state, responses):
        tool_messages = []
        for response in responses:
            tool_call_id, tool_name, response = response
            tool_message = ToolMessage(
                content=str(response), name=tool_name, tool_call_id=tool_call_id
            )
            tool_messages.append(tool_message)

        return {
            "last_node": "call_tools",
            "messages": tool_messages,  # We return a list, because this will get added to the existing list of messages
        }

    @banner(text="Call Tool", level=2)
    def _call_tool(self, tool_call_definition):
        action = ToolInvocation(
            tool=tool_call_definition["name"],
            tool_input=tool_call_definition["args"],
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        logger.info(f"Response ({type(response)}): {response}")
        return (tool_call_definition["id"], tool_call_definition["name"], response)

    def call_tools(self, message: AIMessage) -> List:
        responses = []
        for tool_call_definition in message.tool_calls:
            # add tuples of (tool_call_id, tool_name, response)
            responses.append(self._call_tool(tool_call_definition))
        return responses

    @banner(text="Agentic Tool Call", level=1)
    def agentic_tool_call(self, state):
        message = self.pre_call_tool(state)
        responses = self.call_tools(message)
        return self.post_call_tool(state, responses)


    # Runnable implementation
    def invoke(self, input: AIMessage, config: Optional[RunnableConfig] = None) -> List:
        return self.call_tools(input)
