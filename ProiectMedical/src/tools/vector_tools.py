import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def build_index(documents, persist_directory="vectorstore/"):
    """
    Chunks, embeds and stores documents in ChromaDB.
    
    Chunking Justification:
    We use RecursiveCharacterTextSplitter with chunk_size=500 characters and chunk_overlap=100 characters.
    - Medical guidelines contain tightly coupled sentences describing diagnostic criteria.
    - Too large of a chunk size can cause the 'Lost in the Middle' problem, where the LLM misses details.
    - Too small of a chunk size (e.g., 200) can cut a single clinical rule (like blood pressure parameters or medication doses) in half.
    - Overlap of 100 characters ensures that sentences cut across borders retain context in adjacent chunks.
    """
    if not documents:
        print("No documents to index.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            # We copy all metadata items to ensure that source, doc_type and evidence_level persist.
            metadata = doc["metadata"].copy()
            split_docs.append({
                "page_content": chunk,
                "metadata": metadata
            })
            
    print(f"Generated {len(split_docs)} split chunks from {len(documents)} source documents.")
    
    # Extract page contents and metadata list
    texts = [sd["page_content"] for sd in split_docs]
    metadatas = [sd["metadata"] for sd in split_docs]
    
    # Initialize the embedding model
    # Note: text-embedding-3-small is a multilingual model, ideal for Rom-Eng translation.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize Chroma DB
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    print(f"Successfully indexed and persisted database to {persist_directory}")
    return vector_store
