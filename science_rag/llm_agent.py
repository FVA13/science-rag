import time
import warnings
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

from langchain.chat_models import (
    ChatOpenAI,
    ChatAnthropic,
    ChatGooglePalm,
)
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)

from tools import (
    retrieve_arxiv_papers,
    retrieve_arxiv_paper_by_id,
)

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class ScienceAgent:
    def __init__(
            self,
            provider: str = "openai",
            model_name: Optional[str] = None,
            temperature: float = 0.4,
            base_url: Optional[str] = None,
    ):
        self.provider = ModelProvider(provider.lower())

        default_models = {
            ModelProvider.OPENAI: "gpt-4",
            ModelProvider.ANTHROPIC: "claude-2",
            ModelProvider.GOOGLE: "models/chat-bison-001",
            ModelProvider.OLLAMA: "llama2",
        }

        self.model_name = model_name or default_models[self.provider]

        # Initialize the appropriate chat model based on provider
        if self.provider == ModelProvider.OPENAI:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                max_retries=2,
            )
        elif self.provider == ModelProvider.ANTHROPIC:
            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=temperature,
            )
        elif self.provider == ModelProvider.GOOGLE:
            self.llm = ChatGooglePalm(
                model_name=self.model_name,
                temperature=temperature,
            )
        elif self.provider == ModelProvider.OLLAMA:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=temperature,
                base_url=base_url or "http://localhost:11434",
            )

        self.tools = [retrieve_arxiv_papers, retrieve_arxiv_paper_by_id, DuckDuckGoSearchRun()]

        # Different prompt template for Ollama
        if self.provider == ModelProvider.OLLAMA:
            # ReAct prompt template
            template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Previous conversation history:
            {chat_history}

            Question: {input}
            {agent_scratchpad}"""

            self.prompt = PromptTemplate.from_template(template)
            self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        else:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a science agent. Your task is to provide a user with arxiv-driven data. "
                               "Prove your words by attaching the reference to articles (from metadata)."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)

        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    def chat(self, history: bool = True):
        chat_history = [
            SystemMessage(
                content="You're a helpful scientific assistant. Provide accurate and detailed responses to queries.")
        ]

        print(f"Welcome to Science Agent! Using {self.provider.value} provider with {self.model_name} model")
        print("(Press Enter twice to exit)")

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
    # Example usage with different providers:

    # OpenAI
    # agent = ScienceAgent(provider="openai", model_name="gpt-4")

    # Anthropic
    # agent = ScienceAgent(provider="anthropic", model_name="claude-2")

    # Google
    # agent = ScienceAgent(provider="google", model_name="chat-bison")

    # Ollama (locally deployed)
    agent = ScienceAgent(provider="ollama", model_name="llama3", base_url="http://localhost:11434")

    # Default (OpenAI)
    agent.chat()


if __name__ == '__main__':
    main()
