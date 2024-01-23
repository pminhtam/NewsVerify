from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Optional

os.environ['OPENAI_API_KEY'] = ""
os.environ["SERPAPI_API_KEY"] = ""
key = ""
from langchainhub import Client
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain import hub

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

print(prompt)
# exit()
llm_with_stop = llm.bind(stop=["\nObservation"])
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps'])
} | prompt | llm_with_stop | ReActSingleInputOutputParser()

print(agent)
exit()
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"})



