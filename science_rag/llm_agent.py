import time
import warnings
from typing import List, Dict, Optional
from datetime import datetime

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

class ArxivMetadata(BaseModel):
    """Schema for arXiv paper metadata"""
    entry_id: str = Field(..., alias="Entry_ID")
    published: datetime = Field(..., alias="Published")
    title: str = Field(..., alias="Title")
    authors: str = Field(..., alias="Authors")

    class Config:
        allow_population_by_field_name = True

class RetrievedArxivDoc(BaseModel):
    """Schema for retrieved arXiv document"""
    metadata: ArxivMetadata
    page_content: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d['metadata']['Published'] = d['metadata']['Published'].isoformat()
        return d

class ScienceAgent:
    """Science Agent for handling scientific queries and paper retrieval"""

    def __init__(self, model_name: str = 'gpt-4o-mini', temperature: float = 0.4):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_retries=2,
        )
        self.tools = [self._retrieve_arxiv_papers, DuckDuckGoSearchRun()]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a science agent. Your task is to provide a user with arxiv-driven data. "
                           "Prove your words by attaching the reference to articles (from metadata)."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
        )
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    @staticmethod
    @tool
    def _retrieve_arxiv_papers(
            query: str,
            load_max_docs: int = 2,
            get_full_documents: bool = False,
    ) -> List[RetrievedArxivDoc]:
        """
        Retrieve top papers from arXiv based on the query.

        Args:
            query: Search query string
            load_max_docs: Maximum number of documents to retrieve
            get_full_documents: Whether to fetch full documents or just abstracts

        Returns:
            List of retrieved arXiv documents
        """
        retriever = ArxivRetriever(
            load_max_docs=load_max_docs,
            get_full_documents=get_full_documents,
        )
        response = retriever.invoke(query)
        return [
            {
                'metadata': {
                    **doc.metadata,
                    'Published': str(doc.metadata['Published'])
                },
                'page_content': doc.page_content
            }
            for doc in response
        ]

    def chat(self, history: bool = True):
        """
        Interactive chat interface for the agent.

        Args:
            history: Whether to maintain chat history
        """
        chat_history = [
            SystemMessage(content="You're a helpful scientific assistant. Provide accurate and detailed responses to queries.")
        ]

        print("Welcome to Science Agent! (Press Enter twice to exit)")

        while True:
            try:
                user_input = input("\nUser: ").strip()
                if not user_input:
                    confirmation = input("Are you sure you want to exit? (y/n): ").lower()
                    if confirmation == 'y':
                        print("\nGoodbye!")
                        break
                    continue

                result = self.agent_executor.invoke({
                    "chat_history": chat_history if history else [],
                    "input": user_input,
                })

                if history:
                    chat_history.extend([
                        HumanMessage(content=user_input),
                        AIMessage(content=result["output"])
                    ])

                print(f"\nBot: {result['output']}")

            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

            time.sleep(0.2)

def main():
    """Main entry point"""
    agent = ScienceAgent()
    agent.chat()

if __name__ == '__main__':
    main()