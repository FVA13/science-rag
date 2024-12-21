from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from loguru import logger


class RetrieverAgent:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        text_splitter: TextSplitter
    ) -> None:

        self.text_splitter = text_splitter
        self.retriever = retriever


    def invoke(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def add_documents(self, documents: list[Document]) -> None:

        for document in documents:
            metadata = document.metadata
            metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

            content = document.page_content

            chunks = self.text_splitter.split_text(content)

            chunk_documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

            self.retriever.add_documents(chunk_documents)

