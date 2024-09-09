import os
from langchain.agents import AgentExecutor
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import sys

sys.path.append('..')

# custom import
from chains.app.agent.myagent import MyAgent
from chains.app.tools.ask_for_input import AskForInputTool


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
        "\n만약 사용자가 작업명을 입력하면, 해당 작업을 적절하게 단위작업으로 분리해서 설명해주세요."
        "\n예시: '조리작업'은 '재료준비', '조리', '정리 및 세척 작업'으로 나눌 수 있습니다."
        "\n응답은 JSON 형식으로 제공해주세요. 예시: {'task_name': '조리작업', 'unittasks': ['재료준비', '조리', '정리 및 세척 작업']}"
    ),
    primer = "",
    history = "",
)

llama_template_hist = llama_prompt.copy()
llama_template_hist.input_variables = ['input', 'history']

# set up OpenAI API
OPENAI_API_KEY = "<YOUR_API_KEY>"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = OpenAI()
# llm = LLMChain(prompt=llama_template_hist, llm=openai_llm)

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
    max_messages = 10,
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
