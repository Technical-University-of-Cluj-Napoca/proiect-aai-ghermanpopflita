import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.dtos import ClinicalCaseDTO, RetrievalResultDTO, DocumentType

class RAGMedicalAgent:
    def __init__(self, persist_directory: str = "vectorstore", threshold: float = 0.85):
        """
        Initializes the retrieval agent.
        Loads the pre-existing ChromaDB vectorstore.
        
        Threshold parameter:
        - In langchain's ChromaDB implementation, similarity_search_with_score returns a distance score.
        - Depending on the metric, a smaller distance means higher similarity.
        - Default distance in Chroma is L2 (squared Euclidean distance), where scores are >= 0.
        - If we use Cosine similarity, distance is (1 - cosine_similarity), which ranges from 0 to 2.
        - We normalize/interpret the distance to represent a similarity score: similarity = 1 - distance.
        - A threshold of 0.85 in similarity corresponds to a distance of 0.15.
        """
        self.persist_directory = persist_directory
        self.threshold = threshold
        
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            raise FileNotFoundError(
                f"Vectorstore not found at {persist_directory}. Please run build_index.py first."
            )
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    def build_query(self, case: ClinicalCaseDTO) -> str:
        """
        Constructs a retrieval query by concatenating clinical findings.
        - Simple serialization (JSON) works poorly for vector retrieval because it contains syntax templates.
        - Concatenating age, sex, main symptoms, and chief complaint performs much better semantically.
        """
        query_parts = []
        if case.age:
            query_parts.append(f"{case.age} ani")
        if case.sex:
            query_parts.append(case.sex)
            
        # Chief complaint
        if case.chief_complaint:
            query_parts.append(case.chief_complaint)
            
        # Add primary symptoms
        for symptom in case.symptoms:
            query_parts.append(f"{symptom.name} (severitate: {symptom.severity.value}, debut: {symptom.onset.value})")
            
        # Medical history
        if case.medical_history:
            query_parts.append(", ".join(case.medical_history))
            
        return ", ".join(query_parts)

    def retrieve(self, case: ClinicalCaseDTO, k: int = 5) -> list[RetrievalResultDTO]:
        """
        Queries ChromaDB for clinical context matching the patient case.
        Returns a list of RetrievalResultDTO matching or exceeding the similarity threshold.
        """
        query = self.build_query(case)
        print(f"Retrieval Query: '{query}'")
        
        # similarity_search_with_score returns list of (Document, distance)
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        retrieved_dtos = []
        for doc, distance in results:
            # Normalize distance to a similarity score (0 to 1 range)
            # For cosine distance, distance = 1 - similarity. So similarity = 1 - distance.
            # However, if using L2 distance, the distance can be larger than 1. We apply a soft normalization.
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            print(f"Doc Source: {doc.metadata.get('source')} | Distance: {distance:.4f} | Sim Score: {similarity_score:.4f}")
            
            # Apply relevance threshold filter
            if similarity_score >= self.threshold:
                doc_type_val = doc.metadata.get("doc_type")
                # Safety fallback for doc type Enum
                try:
                    doc_type = DocumentType(doc_type_val)
                except ValueError:
                    doc_type = DocumentType.GHID_CLINIC
                    
                dto = RetrievalResultDTO(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    score=similarity_score,
                    doc_type=doc_type,
                    evidence_level=doc.metadata.get("evidence_level")
                )
                retrieved_dtos.append(dto)
                
        return retrieved_dtos
