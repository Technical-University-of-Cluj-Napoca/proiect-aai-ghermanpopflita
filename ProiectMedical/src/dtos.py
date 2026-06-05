from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class SymptomSeverity(str, Enum):
    USOR = "USOR"
    MODERAT = "MODERAT"
    SEVER = "SEVER"
    CRITIC = "CRITIC"

class OnsetType(str, Enum):
    ACUT = "ACUT"
    SUBACUT = "SUBACUT"
    CRONIC = "CRONIC"
    PROGRESIV = "PROGRESIV"

class DocumentType(str, Enum):
    PUBMED_ABSTRACT = "PUBMED_ABSTRACT"
    GHID_CLINIC = "GHID_CLINIC"
    ICD11 = "ICD11"
    PROTOCOL_OMS = "PROTOCOL_OMS"
    PROTOCOL_SPITAL = "PROTOCOL_SPITAL"

class UrgencyLevel(str, Enum):
    CRITIC = "CRITIC"
    SEVER = "SEVER"
    MODERAT = "MODERAT"
    SCAZUT = "SCAZUT"
    ELECTIV = "ELECTIV"

class SymptomDTO(BaseModel):
    name: str = Field(description="Numele simptomului")
    severity: SymptomSeverity = Field(description="Severitatea simptomului")
    duration_days: Optional[int] = Field(None, description="Durata in zile")
    onset: OnsetType = Field(description="Tipul de debut")
    associated_symptoms: List[str] = Field(default_factory=list, description="Simptome asociate")

class VitalSignsDTO(BaseModel):
    heart_rate: Optional[int] = Field(None, description="Pulsul (batai pe minut)")
    bp_systolic: Optional[int] = Field(None, description="Tensiunea sistolica")
    bp_diastolic: Optional[int] = Field(None, description="Tensiunea diastolica")
    temperature: Optional[float] = Field(None, description="Temperatura in grade Celsius")
    respiratory_rate: Optional[int] = Field(None, description="Frecventa respiratorie")
    spo2: Optional[float] = Field(None, description="Saturatia de oxigen")

class ClinicalCaseDTO(BaseModel):
    case_id: str = Field(description="ID-ul cazului unic si anonimizat")
    chief_complaint: str = Field(description="Acuza principala")
    symptoms: List[SymptomDTO] = Field(default_factory=list, description="Lista de simptome")
    vital_signs: Optional[VitalSignsDTO] = Field(None, description="Semne vitale")
    medical_history: List[str] = Field(default_factory=list, description="Antecedente patologice/personale")
    current_medications: List[str] = Field(default_factory=list, description="Medicatie curenta")
    allergies: List[str] = Field(default_factory=list, description="Alergii cunoscute")
    age: Optional[int] = Field(None, description="Varsta pacientului")
    sex: Optional[str] = Field(None, description="Sexul pacientului")

class RetrievalResultDTO(BaseModel):
    text: str = Field(description="Fragmentul de text recuperat")
    source: str = Field(description="Sursa documentului (ex: pubmed_123456)")
    score: float = Field(description="Scorul de similaritate")
    doc_type: DocumentType = Field(description="Tipul documentului")
    evidence_level: Optional[str] = Field(None, description="Nivelul de evidenta (A, B, C)")

class DiagnosisDTO(BaseModel):
    condition_name: str = Field(description="Numele afectiunii diagnosticate")
    icd11_code: Optional[str] = Field(None, description="Codul ICD-11")
    probability: float = Field(description="Probabilitatea estimata (intre 0 si 1)")
    supporting_evidence: List[str] = Field(default_factory=list, description="Dovezi ce sustin diagnosticul")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Dovezi ce contrazic diagnosticul")
    discriminating_investigations: List[str] = Field(default_factory=list, description="Investigatii de discriminare")

class DifferentialDiagnosisDTO(BaseModel):
    primary_diagnosis: DiagnosisDTO = Field(description="Diagnosticul principal")
    differentials: List[DiagnosisDTO] = Field(default_factory=list, description="Diagnostice diferentiale")
    urgency_level: UrgencyLevel = Field(description="Nivelul general de urgenta")
    context_was_empty: bool = Field(False, description="Adevarat daca nu a fost gasit context in RAG")

class EthicalCheckDTO(BaseModel):
    is_safe_to_present: bool = Field(description="True daca este sigur sa fie prezentat utilizatorului")
    mandatory_disclaimers: List[str] = Field(default_factory=list, description="Disclaimer-uri etice obligatorii")
    identified_flags: List[str] = Field(default_factory=list, description="Steaguri rosii identificate")
    requires_immediate_escalation: bool = Field(description="True daca este o urgenta critica")

class ExplanationDTO(BaseModel):
    recommendation_summary: str = Field(description="Sumarul recomandarii")
    cited_sources: List[str] = Field(default_factory=list, description="Sursele citate")
    reasoning_chain: List[str] = Field(default_factory=list, description="Lantul de rationament")
    confidence_score: float = Field(description="Scorul de incredere bazat pe acoperirea simptomelor (0-1)")
