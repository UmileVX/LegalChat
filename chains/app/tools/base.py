from langchain.agents.tools import Tool


########################################################################
## General recipe for making new tools. 
## You can also subclass tool directly, but this is easier to work with
class AutoTool:

    """Keep-Reasoning Tool
    
    This is an example tool. The input will be returned as the output
    """

    def get_tool(self, **kwargs):
        ## Shows also how some open-source libraries like to support auto-variables
        doc_lines = self.__class__.__doc__.split('\n')
        class_name = doc_lines[0]                     ## First line from the documentation
        class_desc = "\n".join(doc_lines[1:]).strip() ## Essentially, all other text
        
        return Tool(
            name        = kwargs.get('name',        class_name),
            description = kwargs.get('description', class_desc),
            func        = kwargs.get('func',        self.run),
        )
    
    def run(self, command: str) -> str:
        ## The function that should be ran to execute the tool forward pass
        return command
