from langchain.chains import TransformChain, SequentialChain, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
# from langchain.agents import BaseSingleActionAgent
# from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms import BaseLLM

from typing import List, Tuple, Any, Union, Optional
from pydantic import root_validator, Field
# from abc import abstractmethod

# custom import
from chains.app.pipeline.settings import SetParams
from .base import MyBaseAgent


class MyAgent(MyBaseAgent):
    
    ## Instance methods that can be passed in as BaseModel arguments. 
    ## Will be associated with self
    
    general_prompt : PromptTemplate
    llm            : BaseLLM
    
    general_chain  : Optional[LLMChain]
    max_messages   : int                   = Field(10, gt=1)
    
    temperature    : float                 = Field(0.6, gt=0, le=1)
    max_new_tokens : int                   = Field(128, ge=1, le=2048)
    eos_token_id   : Union[int, List[int]] = Field(2, ge=0)
    gen_kw_keys = ['temperature', 'max_new_tokens', 'eos_token_id']
    gen_kw = {}
    
    user_toxicity  : float = 0.5
    user_emotion   : str = "Unknown"
    

    # @root_validator
    def validate_input(cls, values: Any) -> Any:
        '''Think of this like the BaseModel's __init__ method'''
        if not values.get('general_chain'):
            llm = values.get('llm')
            prompt = values.get("general_prompt")
            values['general_chain'] = LLMChain(llm=llm, prompt=prompt)  ## <- Feature stop 
        values['gen_kw'] = {k:v for k,v in values.items() if k in values.get('gen_kw_keys')}
        return values
    

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any): 
        '''Takes in previous logic and generates the next action to take!'''
        
        ## [Base Case] Default message to start off the loop. TO NOT OVERRIDE
        tool, response = "Ask-For-Input Tool", "Hello World! How can I help you?"
        if len(intermediate_steps) == 0:
            return self.action(tool, response)

        ## History of past agent queries/observations
        queries      = [step[0].tool_input for step in intermediate_steps]
        observations = [step[1]            for step in intermediate_steps]
        last_obs     = observations[-1]    # Most recent observation (i.e. user input)

        # set up user toxicity score
        # self.user_toxicity = toxity

        ## [Stop Case] If the conversation is getting too long, wrap it up
        if len(observations) >= self.max_messages:
            response = "Thanks so much for the chat, and hope to see ya later! Goodbye!"
            return self.action(tool, response, finish=True)

        ## [Default Case] If observation is provided and you want to respond... do it!
        with SetParams(self.llm, **self.gen_kw):
            response = self.general_chain.run(last_obs)

        ## [!] Probably a good spot for your output-postprocessing steps
        response = response.replace("```", "")
        
        ## [Default Case] Send over the response back to the user and get their input!
        return self.action(tool, response)
    

    def reset(self):
        self.user_toxicity = 0
        self.user_emotion = "Unknown"
        if getattr(self.general_chain, 'memory', None) is not None:
            self.general_chain.memory.clear()
