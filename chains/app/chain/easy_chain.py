from langchain.chains import TransformChain, SequentialChain, LLMChain
from typing import List, Any

from transformers import StoppingCriteria
import torch

# custom import
from chains.app.pipeline.settings import SetParams


class EasyLLMChain(TransformChain):

    llm: Any
    input_variables:  List[str] = ["input"]
    output_variables: List[str] = ["output"]
    
    def __init__(self, **kwargs): 
        transform = kwargs.get('transform', kwargs.get('transform_cb', self.transform))
        super().__init__(transform=transform, **kwargs)
    
    def transform(self, d: dict):
        with SetParams(self.llm, eos_token_id=[2, 13]):
            pred = self.llm(d['input'])
        return dict(
            output = f"{d['input']}{pred}\nAction: Keep-Reasoning Tool\nAction-Input: Think harder\n"
        )
