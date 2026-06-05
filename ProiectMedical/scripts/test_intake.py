import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.intake_agent import SymptomIntakeAgent

def main():
    load_dotenv()
    
    # Check if key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please define OPENAI_API_KEY in your .env file.")
        sys.exit(1)
        
    os.makedirs("data", exist_ok=True)
    
    agent = SymptomIntakeAgent()
    
    # 3 test cases of varying complexity
    cases = [
        {
            "name": "caz_simplu",
            "text": "Pacientul Popescu Vasile, CNP 1920803123456, domiciliat in Cluj, prezinta cefalee acuta severa de ieri, fara alte acuze. TA 120/80, puls 70, temp 36.6."
        },
        {
            "name": "caz_moderat",
            "text": "Femeie de 58 de ani acuza dispnee progresiva de 3 saptamani, insotita de febra 38.2, scadere in greutate de 6 kg. SpO2 93%, puls 98, FR 22. Nu are tuse productiva. Fara alte antecedente."
        },
        {
            "name": "caz_complex",
            "text": "Pacient de 72 de ani cu antecedente de HTA, DZ de tip 2 si fibrilatie atriala. Se prezinta cu durere toracica retrosternala cu caracter de strivire debutata de 1 ora (durata 60 min), iradiata in bratul stang, insotita de dispnee severa si transpiratii abundente. Medicatie curenta: Metformin 1000mg x2/zi, Valsartan 160mg, Apixaban 5mg x2/zi. Alergie la Penicilina. TA 150/90, puls 110, SpO2 91% pe aer ambiental, temp 37.1."
        }
    ]
    
    for case in cases:
        print(f"\n======================================")
        print(f"Processing case: {case['name']}")
        print(f"Raw text:\n{case['text']}\n")
        
        parsed_case = agent.intake(case["text"])
        
        # Verify anonymization (patient name 'Vasile' or CNP should not be present in chief_complaint or symptoms)
        content_str = parsed_case.model_dump_json()
        assert "Vasile" not in content_str, "Anonymization failed: name present in DTO"
        assert "1920803" not in content_str, "Anonymization failed: CNP present in DTO"
        
        out_path = f"data/{case['name']}_parsed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(parsed_case.model_dump_json(indent=2))
        print(f"Saved parsed DTO to: {out_path}")
        print(f"Case ID: {parsed_case.case_id}")
        print(f"Extracted chief complaint: {parsed_case.chief_complaint}")
        print(f"Vitals: {parsed_case.vital_signs}")

if __name__ == "__main__":
    main()
