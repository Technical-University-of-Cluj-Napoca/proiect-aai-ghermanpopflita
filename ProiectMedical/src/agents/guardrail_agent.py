import os
import datetime
from src.dtos import DifferentialDiagnosisDTO, EthicalCheckDTO

class EthicalGuardrailAgent:
    def __init__(self):
        # Hardcoded ethical disclaimers that must be present in every single clinical output
        self.disclaimers = [
            "ATENȚIE: Acest sistem este un instrument de suport decizional destinat exclusiv cadrelor medicale calificate și nu înlocuiește judecata clinică proprie.",
            "Recomandările oferite necesită validare formală din partea unui medic specialist înainte de aplicarea oricărui tratament sau investigații invazive.",
            "În caz de urgență medicală majoră sau conflict clinic, protocolul clinic local de urgență al spitalului primează întotdeauna."
        ]
        
        # Red-flag keywords for critical emergency checks
        self.critical_keywords = [
            "infarct", "stemi", "nstemi", "myocardial", "sepsis", "soc", "shock",
            "avc", "accident vascular", "stroke", "ischemic", "abdomen acut",
            "peritonita", "hemoragie", "intoxicatie", "poisoning", "overdose",
            "coma", "status epilepticus", "insuficienta respiratorie"
        ]

    def validate(self, differential: DifferentialDiagnosisDTO, case_id: str = "UNKNOWN") -> EthicalCheckDTO:
        """
        Validates the generated differential diagnosis.
        1. Checks for clinical flags that require immediate escalation.
        2. Injects mandatory clinical disclaimers.
        3. Logs escalations to logs/escalations.txt.
        """
        requires_escalation = False
        identified_flags = []
        
        # Scan primary diagnosis and differentials for critical keywords
        diagnoses_to_check = [differential.primary_diagnosis] + differential.differentials
        
        for diag in diagnoses_to_check:
            # Check if any emergency condition has a probability > 0.3
            # We choose 0.3 as a threshold because it is sensitive enough to catch early signs of high-mortality
            # conditions without spamming alarms for low-probability, low-risk differentials.
            name_lower = diag.condition_name.lower()
            has_keyword = any(kw in name_lower for kw in self.critical_keywords)
            
            if has_keyword and diag.probability > 0.3:
                requires_escalation = True
                identified_flags.append(f"Urgenta critica detectata: {diag.condition_name} (Probabilitate: {diag.probability:.2f})")
                
        # If the general urgency level is CRITIC, force escalation
        if differential.urgency_level.value == "CRITIC":
            requires_escalation = True
            if "Nivel general de urgenta CRITIC" not in identified_flags:
                identified_flags.append("Nivel general de urgenta CRITIC detectat in diagnosticul diferential.")
                
        # Log escalation if triggered
        if requires_escalation:
            os.makedirs("logs", exist_ok=True)
            log_path = "logs/escalations.txt"
            timestamp = datetime.datetime.now().isoformat()
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] CASE ID: {case_id} - ESCALADAT - Steaguri: {', '.join(identified_flags)}\n")
                print(f"Logged critical escalation for {case_id} to {log_path}")
            except Exception as e:
                print(f"Error writing escalation log: {e}")
                
        return EthicalCheckDTO(
            is_safe_to_present=True, # We do not censor the results; we present them with disclaimers and alarms
            mandatory_disclaimers=self.disclaimers,
            identified_flags=identified_flags,
            requires_immediate_escalation=requires_escalation
        )
