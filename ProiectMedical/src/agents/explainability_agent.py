import re
from typing import List
from src.dtos import ClinicalCaseDTO, DifferentialDiagnosisDTO, RetrievalResultDTO, ExplanationDTO

class ExplainabilityAgent:
    def __init__(self):
        pass

    def calculate_confidence_score(self, case: ClinicalCaseDTO, context_chunks: List[RetrievalResultDTO]) -> float:
        """
        Calculates a confidence score between 0.0 and 1.0.
        
        Formula:
        Confidence Score = (Number of symptoms mentioned in the retrieved literature chunks) / (Total symptoms in ClinicalCaseDTO)
        
        Justification:
        If all clinical symptoms described by the intake agent have semantic mapping/matches
        in the retrieved medical guidelines, our confidence in the grounding context is high (1.0).
        If the patient presents with symptoms that have no mention in the retrieved documents,
        the grounding context is sparse, and the confidence score decreases.
        """
        if not case.symptoms:
            return 1.0 # Baseline if no symptoms were recorded
            
        total_symptoms = len(case.symptoms)
        matched_symptoms = 0
        
        # Concat all retrieved text to scan for matches
        all_retrieved_text = " ".join([chunk.text.lower() for chunk in context_chunks])
        
        for symptom in case.symptoms:
            symptom_name_lower = symptom.name.lower()
            # Simple keyword match or stem match
            stem = symptom_name_lower[:5] # Check first 5 chars
            if stem in all_retrieved_text:
                matched_symptoms += 1
                
        return float(matched_symptoms) / float(total_symptoms)

    def explain(self, case: ClinicalCaseDTO, differential: DifferentialDiagnosisDTO, context_chunks: List[RetrievalResultDTO]) -> ExplanationDTO:
        """
        Generates clinical reasoning explanations, citation lists and scores.
        """
        reasoning_chain = []
        cited_sources = list(set([chunk.source for chunk in context_chunks]))
        
        primary = differential.primary_diagnosis
        
        # 1. Primary Diagnosis Reasoning
        reasoning_chain.append(f"1. A fost selectat ca diagnostic principal: {primary.condition_name} (Probabilitate: {primary.probability:.2%}).")
        
        # Pull out citations from supporting evidence
        symptom_list_str = ", ".join([s.name for s in case.symptoms])
        reasoning_chain.append(f"2. Simptomele corelate: '{symptom_list_str}' au fost identificate în literatura medicală ca indicatori pentru această patologie.")
        
        # Describe citations specifically
        for evidence in primary.supporting_evidence:
            reasoning_chain.append(f"   - Dovezi de susținere: {evidence}")
            
        if primary.contradicting_evidence:
            reasoning_chain.append(f"   - Factori contradictorii de exclus: {', '.join(primary.contradicting_evidence)}")
            
        if primary.discriminating_investigations:
            reasoning_chain.append(f"3. Pentru confirmarea diagnosticului de {primary.condition_name}, se recomandă efectuarea de: {', '.join(primary.discriminating_investigations)}")

        # 2. Differentials Reasoning
        if differential.differentials:
            reasoning_chain.append("4. Diagnosticele diferențiale luate în considerare:")
            for diff in differential.differentials[:3]: # Limit to top 3 differentials
                reasoning_chain.append(f"   - {diff.condition_name} (Probabilitate: {diff.probability:.2%}) - Investigatie recomandată: {', '.join(diff.discriminating_investigations[:2])}")

        # Compute symptom coverage confidence score
        confidence = self.calculate_confidence_score(case, context_chunks)
        
        # Build recommendation summary
        rec_summary = f"Cazul clinic prezintă suspiciune înaltă de {primary.condition_name}. "
        if differential.urgency_level.value in ["CRITIC", "SEVER"]:
            rec_summary += f"Urgența este evaluată ca {differential.urgency_level.value}. Necesită monitorizare și investigare imediată."
        else:
            rec_summary += f"Starea pacientului sugerează o urgență de nivel {differential.urgency_level.value}."

        return ExplanationDTO(
            recommendation_summary=rec_summary,
            cited_sources=cited_sources,
            reasoning_chain=reasoning_chain,
            confidence_score=confidence
        )
