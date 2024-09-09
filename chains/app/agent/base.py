from langchain.chains import TransformChain, SequentialChain, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.agents import BaseSingleActionAgent
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms import BaseLLM

from typing import List, Tuple, Any, Union, Optional
from pydantic import root_validator, Field
from abc import abstractmethod


class MyBaseAgent(BaseSingleActionAgent):

    # @root_validator
    def validate_input(cls, values: Any) -> Any:
        '''
        Think of this like the BaseModel's __init__ method
        You'll see how it works in the stencil, but this is where components get initialized
        '''
        return values

    @abstractmethod
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any): 
        '''
        Taking the "intermediate_steps" as the history of steps.
        Decide on the next action to take! Return the required action 
        (returns a query from the action method)
        '''
        pass


    def action(self, tool, tool_input, finish=False) -> Union[AgentAction, AgentFinish]:
        '''Takes the action associated with the tool and feeds it the necessary parameters'''
        if finish:
            return AgentFinish({"output": tool_input},           log = f"\nFinal Answer: {tool_input}\n")
        else:
            return AgentAction(tool=tool, tool_input=tool_input, log = f"\nAgent: {tool_input.strip()}\n")
        # else:    return AgentAction(tool=tool, tool_input=tool_input, log = f"\nTool: {tool}\nInput: {tool_input}\n") ## Actually Correct

    async def aplan(self, intermediate_steps, **kwargs):
        '''The async version of plan. It has to be defined because abstractmethod'''
        return await self.plan(intermediate_steps, **kwargs)

    @property
    def input_keys(self):
        return ["input"]
