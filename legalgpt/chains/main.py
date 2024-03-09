import os
from langchain.agents import AgentExecutor
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain_openai import OpenAI
import sys

sys.path.append('..')

# custom import
from chains.app.agent.myagent import MyAgent
from chains.app.tools.ask_for_input import AskForInputTool

# set up OpenAI API
OPENAI_API_KEY = "<YOUR_API_KEY>"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = OpenAI()

llama_full_prompt = PromptTemplate.from_template(
    template="<s>[INST]<<SYS>>{sys_msg}<</SYS>>\n\nContext:\n{history}\n\nHuman: {input}\n[/INST] {primer}",
)

llama_prompt = llama_full_prompt.partial(
    sys_msg = ( 
        "You are a helpful, respectful and honest AI assistant."
        "\nAlways answer as helpfully as possible, while being safe."
        "\nPlease be brief and efficient unless asked to elaborate, and follow the conversation flow."
        "\nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        "\nEnsure that your responses are socially unbiased and positive in nature."
        "\nIf a question does not make sense or is not factually coherent, explain why instead of answering something incorrect." 
        "\nIf you don't know the answer to a question, please don't share false information."
        "\nIf the user asks for a format to output, please follow it as closely as possible."
        "\nPlease answer in Korean with formal language."
    ),
    primer = "",
    history = "",
)

llama_template_hist = llama_prompt.copy()
llama_template_hist.input_variables = ['input', 'history']

# memory = ConversationSummaryMemory(llm=llm, temperature=0, verbose=True)
memory = ConversationBufferMemory(return_messages=True)
conv_chain = ConversationChain(
    llm=llm,
    prompt=llama_template_hist, 
    memory=memory,
    verbose=True
)

agent_kw = dict(
    llm = llm,
    general_prompt = llama_prompt,
    max_new_tokens = 128,
    general_chain = conv_chain,
    eos_token_id = [2, 4954, 7521]   
)

agent_ex = AgentExecutor.from_agent_and_tools(
    agent = MyAgent(**agent_kw),
    tools=[AskForInputTool(input).get_tool()], 
    verbose=True
)

if __name__ == "__main__":
    agent_ex.run("")
