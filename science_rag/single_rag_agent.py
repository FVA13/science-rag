from typing import List, Dict, Any

from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import ArxivRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from langgraph.graph import END, StateGraph, START


from science_rag.agents import AnswerAgent, RetrievalGraderAgent, RetrieverAgent
import pprint


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


class ScienceRAG:
    def __init__(
        self,
        retriever_agent: RetrieverAgent,
        retrieval_grader_agent: RetrievalGraderAgent,
        answer_agent: AnswerAgent,
    ) -> None:
        self.retriever_agent = retriever_agent
        self.retrieval_grader_agent = retrieval_grader_agent
        self.answer_agent = answer_agent

    def retrieve_arxiv_papers(self, state: dict) -> dict:
        """
        Retrieve top papers from arXiv based on the query.

        Args:
            query: Search query string
            load_max_docs: Maximum number of documents to retrieve (default: 2)

        Returns:
            List of retrieved arXiv documents

        Raises:
            ValueError: If retrieval fails
        """
        load_max_docs = 2
        question = state["question"]

        try:
            arxiv_retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=True,
                load_all_available_meta=True,
                doc_content_chars_max=None,
            )
            raw_documents = arxiv_retriever.invoke(question)

            self.retriever_agent.add_documents(raw_documents)

            return state

        except Exception as e:
            raise ValueError(f"Failed to retrieve papers: {str(e)}")

    def retrieve(self, state: dict) -> dict:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logger.debug("---RETRIEVE---")
        question = state["question"]

        documents = self.retriever_agent.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: dict) -> str:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        logger.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader_agent.execute(
                question=question, document=d.page_content
            )
            logger.debug(f"SCORE: {score}")
            grade = score.binary_score
            if grade == "yes":
                logger.debug("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.debug("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        # TODO: FIX IT
        if len(documents) == 0:
            return "retrieve_arxiv_papers"
        else:
            return "generate"


    def generate(self, state: dict) -> dict:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        logger.debug("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.answer_agent.execute(context=documents, question=question)
        return {"documents": documents, "question": question, "generation": generation}


    def run(self) -> None:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("retrieve_arxiv_papers", self.retrieve_arxiv_papers)
        workflow.add_node("generate", self.generate)

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
            {
                "generate": "generate",
                "retrieve_arxiv_papers": "retrieve_arxiv_papers",
            },
        )
        workflow.add_edge("retrieve_arxiv_papers", "retrieve")

        # Compile
        app = workflow.compile()

        inputs = {
            "question": "Give me simple explanation of In Context Reinforcement Learning"
        }
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)



if __name__ == "__main__":

    # Agents Initialization
    base_url = "https://api.vsegpt.ru/v1"

    # RetrieverAgent initialization
    ## Embedding model initialization
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    ## retriever initialization
    collection_name = "science_rag"
    persist_directory = "./data/chroma_data"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ## TextSplitter initialization
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
    )
    retriever_agent = RetrieverAgent(retriever=retriever, text_splitter=text_splitter)

    # RetrievalGraderAgent initialization
    retrieval_grader_agent = RetrievalGraderAgent(base_url=base_url, llm_model_name="openai/gpt-4o-mini")

    # AnswerAgent initialization
    answer_agent = AnswerAgent(base_url=base_url, llm_model_name="openai/gpt-4o-mini")

    # Main class initialization
    science_rag = ScienceRAG(
        retriever_agent=retriever_agent,
        retrieval_grader_agent=retrieval_grader_agent,
        answer_agent=answer_agent,
    )
    science_rag.run()



