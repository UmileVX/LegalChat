# from io import StringIO
# import sys
# from typing import Dict, Optional

# custom import
from .base import AutoTool


class AskForInputTool(AutoTool):

    """Ask-For-Input Tool
    
    This tool asks the user for input, which you can use to gather more information. 
    Use only when necessary, since their time is important and you want to give them a great experience! For example:
    Action-Input: What is your name?
    """
    
    def __init__(self, fn = input):
        self.fn = fn

    def run(self, command: str) -> str:
        response = self.fn(command)
        return response
