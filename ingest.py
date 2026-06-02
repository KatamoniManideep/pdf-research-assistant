import os
import hashlib
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

def generate_document_hash(file_path: str) -> str:
    
    hasher = hashlib.sha256()
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(65536): 
            hasher.update(chunk)
            
    return hasher.hexdigest()

def ingest_and_chunk_pdf(file_name: str)-> List[Document]:
    file_path = os.path.join(Config.DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"could not find {file_name}")
    
    print(f"Strating Ingestion for :{file_name}")
    
    print("Extracting text via PyMuPDF...")
    loader=PyMuPDFLoader(file_path)
    documents= loader.load()
    
    if not documents:
        raise ValueError(f"No text extracted from {file_name} .May be a scanned image.")
    
    print(f"Loaded {len(documents)} pages.")
    
    # embeddings=OpenAIEmbeddings(
    #     model=Config.EMBEDDING_MODEL,
    #     api_key=Config.OPENAI_API_KEY # type: ignore
    # )
    
    # semantic_chunker = SemanticChunker(
    #     embeddings,
    #     breakpoint_threshold_amount=Config.CHUNK_BREAKPOINT_THRESHOLD,
    # )
    
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = fallback_splitter.split_documents(documents)
    doc_hash = generate_document_hash(file_path)

    for i, chunk in enumerate(chunks):
        chunk.metadata["document_hash"] = doc_hash
        chunk.metadata['chunk_id']=f"{doc_hash}_chunk_{i}"
        
        chunk.page_content = " ".join(chunk.page_content.split())
    print(f"successfully split into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
	target_pdf="sample_paper.pdf"
    
	try:
        
		resulting_chunks=ingest_and_chunk_pdf(target_pdf)
        
		print("\n--- TEST SUCCESSFUL ---")
		print(f"Total Chunks Generated: {len(resulting_chunks)}")
		print(f"Preview of Chunk 0 Metadata: {resulting_chunks[0].metadata}")
		print(f"Preview of Chunk 0 Content: {resulting_chunks[0].page_content[:150]}...")
    
	except FileNotFoundError as e:
		print(f"Setup Error: {e}")
		print("Did you put the PDF in the correct Config.DATA_DIR folder?")
        
	except Exception as e:
		print(f"An enexpected error occured: {e}")






