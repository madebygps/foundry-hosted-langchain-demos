"""
Workflow demo: Multi-step workflow hosted on Azure AI Foundry.

Three LLM nodes in a chain:
    writer → legal_reviewer → formatter

The writer creates a slogan, the legal reviewer checks it, and the formatter
styles it for terminal output. Each node only sees the output of the
previous node.

This module uses AzureAIResponsesAgentHost from a vendored copy of
https://github.com/langchain-ai/langchain-azure/pull/501 which provides
first-class LangGraph hosting support for Azure AI Foundry.

Run using:
    azd ai agent run
"""

import logging
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from _vendor.langchain_azure_ai_runtime import AzureAIResponsesAgentHost

load_dotenv(dotenv_path="../.env", override=True)

logger = logging.getLogger("workflow-agent")
logger.setLevel(logging.INFO)

# Emit LangChain/LangGraph spans to Application Insights with gen_ai.agent.id
# so the Foundry portal Agent Monitor can identify this agent's traces.
enable_auto_tracing(
    auto_configure_azure_monitor=True,
    enable_content_recording=False,
    trace_all_langgraph_nodes=True,
    agent_id="slogan-workflow",
)

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")

llm = ChatOpenAI(
    base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
    api_key=_token_provider,
    model=MODEL_DEPLOYMENT_NAME,
    use_responses_api=True,
    streaming=True,
)


async def writer(state: MessagesState) -> MessagesState:
    """Create a slogan based on the user's input."""
    user_text = state["messages"][-1].content
    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent slogan writer. "
                    "You create new slogans based on the given topic."
                ),
            },
            {"role": "user", "content": user_text},
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


async def legal_reviewer(state: MessagesState) -> MessagesState:
    """Review and correct the slogan for legal compliance."""
    previous_output = state["messages"][-1].content
    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent legal reviewer. "
                    "Make necessary corrections to the slogan so that it is legally compliant."
                ),
            },
            {"role": "user", "content": previous_output},
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


async def formatter(state: MessagesState) -> MessagesState:
    """Format the slogan with Markdown for display."""
    previous_output = state["messages"][-1].content
    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent content formatter. "
                    "You take the slogan and format it in Markdown with bold text "
                    "and decorative elements. Do not use ANSI escape codes or terminal color codes."
                ),
            },
            {"role": "user", "content": previous_output},
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


# ── Build the workflow graph ────────────────────────────────────────

builder = StateGraph(MessagesState)
builder.add_node("writer", writer)
builder.add_node("legal_reviewer", legal_reviewer)
builder.add_node("formatter", formatter)
builder.add_edge(START, "writer")
builder.add_edge("writer", "legal_reviewer")
builder.add_edge("legal_reviewer", "formatter")
builder.add_edge("formatter", END)
graph = builder.compile()

# ── Hosted agent entrypoint ─────────────────────────────────────────

host = AzureAIResponsesAgentHost(
    graph=graph,
    stream_mode="messages",
    responses_history_count=20,
)

if __name__ == "__main__":
    host.run()
