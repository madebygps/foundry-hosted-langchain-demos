"""
Stage 2 (retrieve action): Add Foundry IQ grounding through the Azure AI Search
retrieve action instead of MCP.

This is a fallback when the KB MCP endpoint is not usable in your environment.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    python stage2_foundry_iq_retrieve.py
"""

import asyncio
import logging
import os
from datetime import date
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalSemanticIntent,
)
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import Field
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown

load_dotenv(override=True)

console = Console()
logger = logging.getLogger("stage2")


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    logger.info("[tool] get_enrollment_deadline_info()")
    return {
        "benefits_enrollment_opens": "2026-11-11",
        "benefits_enrollment_closes": "2026-11-30",
    }


class KnowledgeBaseRetrieveTool:
    """Wrap the Azure AI Search retrieve action as an async agent tool."""

    def __init__(self, kb_client: KnowledgeBaseRetrievalClient) -> None:
        self._kb_client = kb_client

    async def retrieve(
        self,
        queries: Annotated[
            list[str],
            Field(
                description=(
                    "1 to 3 concise search queries (max ~12 words each). "
                    "Use alternate wording as separate entries."
                ),
                min_length=1,
                max_length=3,
            ),
        ],
    ) -> str:
        """Search the Zava company knowledge base for HR policies and benefits."""
        logger.info("[tool] knowledge_base_retrieve(%s)", queries)
        request = KnowledgeBaseRetrievalRequest(
            intents=[KnowledgeRetrievalSemanticIntent(search=query) for query in queries]
        )
        result = await self._kb_client.retrieve(retrieval_request=request)
        if result.response and result.response[0].content:
            return result.response[0].content[0].text
        return "No results found."


async def main() -> None:
    credential = DefaultAzureCredential()
    kb_client = None
    try:
        aoai_token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        client = ChatOpenAI(
            base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
            api_key=aoai_token_provider,
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        )

        kb_client = KnowledgeBaseRetrievalClient(
            endpoint=os.environ["AZURE_AI_SEARCH_SERVICE_ENDPOINT"],
            knowledge_base_name=os.environ["AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME"],
            credential=credential,
        )
        kb_tool = KnowledgeBaseRetrieveTool(kb_client)

        agent = create_agent(
            model=client,
            tools=[kb_tool.retrieve, get_enrollment_deadline_info],
            system_prompt=(
                f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
                "Use the knowledge-base retrieve tool to answer questions about HR policies, benefits, "
                "and company information. Use get_enrollment_deadline_info for enrollment timing. "
                "If you cannot answer from the tools, say so clearly."
            ),
        )

        response = (
            await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What PerksPlus benefits are there, and when do I need to enroll by?",
                        }
                    ]
                }
            )
        )["messages"][-1]
        console.print("\n[bold]Agent answer:[/bold]")
        console.print(Markdown(response.text))
    finally:
        if kb_client is not None:
            await kb_client.close()
        await credential.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
    asyncio.run(main())
