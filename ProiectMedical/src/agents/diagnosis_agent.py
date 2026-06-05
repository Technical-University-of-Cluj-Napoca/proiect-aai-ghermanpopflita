import os
import json
import sqlite3
from typing import List
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from src.dtos import (
    ClinicalCaseDTO, RetrievalResultDTO, DifferentialDiagnosisDTO,
    DiagnosisDTO, UrgencyLevel, DocumentType
)

# Set up global SQLite cache for development
# This caches LLM responses based on prompt hash in a local sqlite database
try:
    os.makedirs(".cache", exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=".cache/langchain_cache.db"))
    print("Local SQLite cache initialized at .cache/langchain_cache.db")
except Exception as e:
    print(f"Warning: Could not initialize SQLiteCache: {e}")

class DifferentialDiagnosisAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        # As requested, gpt-4o is mandatory for reasoning and structuring diagnoses
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)

    def diagnose(self, case: ClinicalCaseDTO, context_chunks: List[RetrievalResultDTO]) -> DifferentialDiagnosisDTO:
        """
        Formulates a differential diagnosis based on the clinical case and retrieved context.
        If no context is retrieved, returns a default DTO with context_was_empty=True without calling the LLM.
        """
        if not context_chunks:
            print("No context chunks retrieved. Returning early with empty context DTO.")
            # Create a default empty primary diagnosis
            empty_primary = DiagnosisDTO(
                condition_name="Diagnostic necunoscut (Lipsă context clinic)",
                icd11_code=None,
                probability=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                discriminating_investigations=["Efectuare RAG cu baza de date populată"]
            )
            return DifferentialDiagnosisDTO(
                primary_diagnosis=empty_primary,
                differentials=[],
                urgency_level=UrgencyLevel.SCAZUT,
                context_was_empty=True
            )
            
        # Build prompt using the retrieved context
        medical_context_str = ""
        for i, chunk in enumerate(context_chunks):
            # [Sursa: pubmed_12345678] - prefix to allow model to cite the exact PMID
            medical_context_str += f"\n--- Fragment {i+1} [Sursă: {chunk.source}] [Nivel Evidență: {chunk.evidence_level or 'N/A'}] ---\n"
            medical_context_str += f"{chunk.text}\n"
            
        system_prompt = """You are a highly detailed and precise clinical reasoning agent. Your role is to generate a differential diagnosis based on the patient case presentation and the provided medical context.

CRITICAL INSTRUCTIONS:
1. ANCHORING: You must ONLY suggest diagnoses that have supporting evidence in the provided context. Do NOT invent guidelines, PMID articles, or clinical data.
2. CITATION: For every diagnostic claim or investigation suggested, you must cite the exact source (e.g. PMID, guideline title, or ICD-11 code) from the provided context.
3. STRUCTURE:
   - Output maximum 5 diagnoses, ranked in descending order of probability.
   - For each diagnosis, provide the condition name, its ICD-11 code, estimated probability, supporting evidence (linked to sources), contradicting evidence, and key discriminating investigations that can rule in or rule out the diagnosis.
   - Mandate specific investigations to discriminate between the top 3 diagnoses.
   - Assess the general UrgencyLevel: CRITIC, SEVER, MODERAT, SCAZUT, or ELECTIV.
   
You must respond with a JSON object conforming exactly to this structure:
{{
  "primary_diagnosis": {{
    "condition_name": "string",
    "icd11_code": "string or null",
    "probability": float (0.0 to 1.0),
    "supporting_evidence": ["string with [Sursa: ...]"],
    "contradicting_evidence": ["string"],
    "discriminating_investigations": ["string"]
  }},
  "differentials": [
    {{
      "condition_name": "string",
      "icd11_code": "string or null",
      "probability": float (0.0 to 1.0),
      "supporting_evidence": ["string with [Sursa: ...]"],
      "contradicting_evidence": ["string"],
      "discriminating_investigations": ["string"]
    }}
  ],
  "urgency_level": "CRITIC" | "SEVER" | "MODERAT" | "SCAZUT" | "ELECTIV"
}}

Respond only with the raw JSON string. Do not include markdown wraps or explanations outside of the JSON."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context medical din RAG:\n{context}\n\nPacient:\n- Vârstă: {age}\n- Sex: {sex}\n- Acuză principală: {complaint}\n- Simptome: {symptoms}\n- Semne vitale: {vitals}\n- Istoric medical: {history}\n- Medicamente: {medications}\n- Alergii: {allergies}")
        ])
        
        # Serialize symptoms and vitals for presentation
        symptoms_str = ", ".join([f"{s.name} ({s.severity.value})" for s in case.symptoms])
        vitals_str = str(case.vital_signs.model_dump()) if case.vital_signs else "Nespecificate"
        history_str = ", ".join(case.medical_history)
        meds_str = ", ".join(case.current_medications)
        allergies_str = ", ".join(case.allergies)
        
        # JsonOutputParser handles stripping markdown fences and parsing JSON automatically
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        try:
            data = chain.invoke({
                "context": medical_context_str,
                "age": case.age or "Necunoscut",
                "sex": case.sex or "Necunoscut",
                "complaint": case.chief_complaint,
                "symptoms": symptoms_str,
                "vitals": vitals_str,
                "history": history_str,
                "medications": meds_str,
                "allergies": allergies_str
            })

            # context_was_empty is False since we have chunks
            data["context_was_empty"] = False
            return DifferentialDiagnosisDTO(**data)
            
        except Exception as e:
            print(f"Error parsing differential diagnosis: {e}")
            # Fallback on JSON parse error - do not crash!
            fallback_primary = DiagnosisDTO(
                condition_name="Eroare de procesare diagnostic",
                icd11_code=None,
                probability=1.0,
                supporting_evidence=[f"Eroare: {str(e)}"],
                contradicting_evidence=[],
                discriminating_investigations=[]
            )
            return DifferentialDiagnosisDTO(
                primary_diagnosis=fallback_primary,
                differentials=[],
                urgency_level=UrgencyLevel.SCAZUT,
                context_was_empty=False
            )
