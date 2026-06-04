import os
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
from langchain_openai import OpenAIEmbeddings

class HybridRetriverSystem:
    def __init__(self,chunks:List[Document]):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_directory = os.path.join(Config.DATA_DIR, "chroma_db")

        self.vector_store=self._setup_chroma(chunks)
        self.dense_retriever=self.vector_store.as_retriever(
            search_kwargs={"k":2}
        )

        if chunks:
            self.sparse_retriever = BM25Retriever.from_documents(chunks)
            self.sparse_retriever.k =2
        else:
            raise ValueError("BM25 requires documents to build its initial index.")
        
        self.ensemble_retriever =EnsembleRetriever(
            retrievers=[self.dense_retriever,self.sparse_retriever],
            weights=[0.5, 0.5]
        )

        self.compresser= FlashrankRerank(top_n=5)

        self.final_retriever =ContextualCompressionRetriever(
            base_compressor=self.compresser,
            base_retriever=self.ensemble_retriever
        )

    def _setup_chroma(self, chunks: List[Document]) -> Chroma:
   
        import shutil
        if os.path.exists(self.persist_directory):
            print("Cleaning up old database...")
            shutil.rmtree(self.persist_directory)
    
        print("Building fresh ChromaDB...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def query(self,question: str)-> List[Document]:
        print(f"\n Searching for :'{question}'")

        results = self.final_retriever.invoke(question)
        return results
    

if __name__ == "__main__":

    mock_chunks = [
        Document(page_content="User XYZ-987 requested a refund on Tuesday.", metadata={"source": "doc1"}),
        Document(page_content="The psychology of financial wealth and returning money.", metadata={"source": "doc2"})
    ]

    print("Initializing Hybrid Retriver Pipeline...")
    retriever_system= HybridRetriverSystem(chunks=mock_chunks)

    answer= retriever_system.query("What did user XYZ-987 do?")

    print(f"Found {len(answer)} highly relevant chunks.")
    for i, doc in enumerate(answer):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)
        print(f"Score: {doc.metadata.get('relevance_score', 'N/A')}")