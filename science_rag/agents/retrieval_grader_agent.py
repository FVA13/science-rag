from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGraderAgent:
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
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        self.system_prompt = (
            """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        )
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader


    def execute(self, question: str, document: str) -> str:
        return self.retrieval_grader.invoke(
            {"document": document, "question": question}
        )