from typing import Literal

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class AnswerAgent:
    def __init__(
        self,
        llm_model_name: str,
        base_url: str | None = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=0,
            base_url=base_url,
        )
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def execute(self, context: list[str], question: str) -> str:
        return self.rag_chain.invoke(
            {"context": context, "question": question}
        )