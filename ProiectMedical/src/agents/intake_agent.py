import re
import uuid
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.dtos import ClinicalCaseDTO, SymptomDTO, VitalSignsDTO, SymptomSeverity, OnsetType

class SymptomIntakeAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # We can use gpt-4o-mini for structuring and anonymization, saving gpt-4o for diagnosis.
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        
        # Medical abbreviation expansion map
        self.abbreviation_map = {
            "TA": "Tensiune Arteriala",
            "HTA": "Hipertensiune Arteriala",
            "DZ": "Diabet Zaharat",
            "AHC": "Antecedente Heredocolaterale",
            "IMA": "Infarct Miocardic Acut",
            "AVC": "Accident Vascular Cerebral",
            "GCS": "Glasgow Coma Scale",
            "FiO2": "Fraction of Inspired Oxygen",
            "BP": "Blood Pressure",
            "HR": "Heart Rate",
            "RR": "Respiratory Rate"
        }

    def expand_abbreviations(self, text: str) -> str:
        """Expands common Romanian and general medical abbreviations."""
        expanded_text = text
        for abbr, full_form in self.abbreviation_map.items():
            # Use regex to replace exact word matches
            pattern = rf"\b{abbr}\b"
            expanded_text = re.sub(pattern, f"{abbr} ({full_form})", expanded_text, flags=re.IGNORECASE)
        return expanded_text

    def extract_vitals_via_regex(self, text: str) -> VitalSignsDTO:
        """
        Extract vital signs from raw text using regex as a fast/cheap fallback/validation.
        Regex patterns match Romanian medical shorthand.
        """
        vitals = {}
        
        # 1. Temperature (ex: 38.2 C, Temp 37.5, T=39)
        temp_match = re.search(rf"\b(?:temp|temperatura|T)\s*[:=-]?\s*(\d{{2}}\.?\d?)\s*(?:C|grade)?\b", text, re.IGNORECASE)
        if temp_match:
            try:
                vitals["temperature"] = float(temp_match.group(1))
            except ValueError:
                pass

        # 2. Blood Pressure (ex: TA 130/80, BP 12/8, TA=140/90 mmHg)
        bp_match = re.search(rf"\b(?:TA|BP|tensiune)\s*[:=-]?\s*(\d{{2,3}})\s*[\/\s-]\s*(\d{{2,3}})\b", text, re.IGNORECASE)
        if bp_match:
            try:
                sys_val = int(bp_match.group(1))
                dia_val = int(bp_match.group(2))
                # Handle shorthand like 12/8 instead of 120/80
                if sys_val < 30:
                    sys_val *= 10
                if dia_val < 30:
                    dia_val *= 10
                vitals["bp_systolic"] = sys_val
                vitals["bp_diastolic"] = dia_val
            except ValueError:
                pass

        # 3. Heart Rate / Pulse (ex: Puls 85 bpm, HR 72, AV 90)
        hr_match = re.search(rf"\b(?:puls|pulsul|HR|AV)\s*[:=-]?\s*(\d{{2,3}})\b", text, re.IGNORECASE)
        if hr_match:
            try:
                vitals["heart_rate"] = int(hr_match.group(1))
            except ValueError:
                pass

        # 4. SpO2 / Oxygen Saturation (ex: SpO2 96%, saturatie 94%)
        spo2_match = re.search(rf"\b(?:SpO2|saturatie|sat)\s*[:=-]?\s*(\d{{2,3}})\s*%?\b", text, re.IGNORECASE)
        if spo2_match:
            try:
                val = int(spo2_match.group(1))
                if val <= 100:
                    vitals["spo2"] = float(val)
            except ValueError:
                pass

        # 5. Respiratory Rate (ex: RR 18, Frecventa respiratorie 22, respiratii 20)
        rr_match = re.search(rf"\b(?:RR|frecventa respiratorie|respiratii|FR)\s*[:=-]?\s*(\d{{1,2}})\b", text, re.IGNORECASE)
        if rr_match:
            try:
                vitals["respiratory_rate"] = int(rr_match.group(1))
            except ValueError:
                pass

        return VitalSignsDTO(**vitals)

    def intake(self, raw_text: str) -> ClinicalCaseDTO:
        """
        Processes raw text:
        1. Expands common abbreviations.
        2. Automatically strips identifying patient details (name, CNP, address, phone).
        3. Parses into Pydantic model.
        """
        # Step 1: Preprocessing - expand abbreviations
        preprocessed_text = self.expand_abbreviations(raw_text)
        
        # Step 2: Extract vitals with regex as a baseline
        regex_vitals = self.extract_vitals_via_regex(raw_text)
        
        # Step 3: LLM prompt to perform anonymization and full parsing
        system_prompt = """You are a clinical data extraction agent. Your primary job is to extract patient case details from the provided clinical description and map them into a structured JSON object.

CRITICAL REQUIREMENT (ANONYMIZATION):
You MUST completely remove any direct identifiers of the patient:
- Patient names (e.g. Vasile Popescu, Ioana, etc.)
- National Identification Numbers / CNPs (e.g. 1950302...)
- Specific addresses or telephone numbers.
If present, delete them entirely. Do NOT include them anywhere in the output. Keep only Age and Sex if mentioned.

You must output a JSON object conforming exactly to the following schema:
{{
  "chief_complaint": "string describing main symptoms",
  "symptoms": [
    {{
      "name": "string",
      "severity": "USOR" | "MODERAT" | "SEVER" | "CRITIC",
      "duration_days": int (or null),
      "onset": "ACUT" | "SUBACUT" | "CRONIC" | "PROGRESIV",
      "associated_symptoms": ["string", "string"]
    }}
  ],
  "vital_signs": {{
    "heart_rate": int (or null),
    "bp_systolic": int (or null),
    "bp_diastolic": int (or null),
    "temperature": float (or null),
    "respiratory_rate": int (or null),
    "spo2": float (or null)
  }},
  "medical_history": ["string", "string"],
  "current_medications": ["string", "string"],
  "allergies": ["string", "string"],
  "age": int (or null),
  "sex": "MASCULIN" | "FEMININ" | "string" (or null)
}}

Ensure all extracted symptoms have a mapped severity and onset. If vitals are missing, set them to null.
Only output the raw JSON block. Do not write markdown markers or any conversational text."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Extract structured information from this text:\n\n{text}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"text": preprocessed_text})
        
        cleaned_response = response.content.strip()
        # Remove markdown block symbols if the LLM outputted them despite instructions
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        try:
            import json
            data = json.loads(cleaned_response)
        except Exception as e:
            # Fallback on parse failure
            print(f"Error parsing JSON from intake LLM: {e}. Raw: {cleaned_response}")
            data = {
                "chief_complaint": "Error parsing clinical presentation",
                "symptoms": [],
                "vital_signs": None,
                "medical_history": [],
                "current_medications": [],
                "allergies": [],
                "age": None,
                "sex": None
            }
            
        # Overwrite/Merge vital signs from regex if LLM missed them or extracted them poorly
        llm_vitals = data.get("vital_signs") or {}
        merged_vitals = {
            "heart_rate": llm_vitals.get("heart_rate") or regex_vitals.heart_rate,
            "bp_systolic": llm_vitals.get("bp_systolic") or regex_vitals.bp_systolic,
            "bp_diastolic": llm_vitals.get("bp_diastolic") or regex_vitals.bp_diastolic,
            "temperature": llm_vitals.get("temperature") or regex_vitals.temperature,
            "respiratory_rate": llm_vitals.get("respiratory_rate") or regex_vitals.respiratory_rate,
            "spo2": llm_vitals.get("spo2") or regex_vitals.spo2
        }
        data["vital_signs"] = merged_vitals
        
        # Add case_id
        data["case_id"] = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        
        # Validate through DTO
        return ClinicalCaseDTO(**data)
