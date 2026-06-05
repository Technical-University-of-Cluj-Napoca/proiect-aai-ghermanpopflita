import os
import sys
import io
import json
import time
import glob
import matplotlib.pyplot as plt

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import datetime
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from src.dtos import (
    ClinicalCaseDTO, RetrievalResultDTO, DifferentialDiagnosisDTO,
    EthicalCheckDTO, ExplanationDTO, UrgencyLevel
)
from src.agents.intake_agent import SymptomIntakeAgent
from src.agents.rag_medical_agent import RAGMedicalAgent
from src.agents.diagnosis_agent import DifferentialDiagnosisAgent
from src.agents.guardrail_agent import EthicalGuardrailAgent
from src.agents.explainability_agent import ExplainabilityAgent

# Define the State representing our Graph flow
class WorkflowState(TypedDict):
    raw_clinical_text: str
    clinical_case: ClinicalCaseDTO
    context_chunks: List[RetrievalResultDTO]
    differential: DifferentialDiagnosisDTO
    ethical_check: EthicalCheckDTO
    explanation: ExplanationDTO
    critical_alert: bool
    report_path: str
    iteration: int
    custom_k: int
    custom_threshold: float
    log_runs: List[Dict[str, Any]]

# Max iterations for the feedback loop
MAX_ITER = 2

def log_node_execution(state: WorkflowState, node_name: str, duration: float, tokens: int = 0, retrieved_count: int = 0):
    """Logs the execution details of a node to state and saves to a run log file."""
    log_entry = {
        "nod": node_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "duration_seconds": round(duration, 3),
        "tokens_consumed": tokens,
        "chunks_retrieved": retrieved_count
    }
    if "log_runs" not in state:
        state["log_runs"] = []
    state["log_runs"].append(log_entry)
    return state

# Node functions
def intake_symptoms_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    agent = SymptomIntakeAgent()
    case_dto = agent.intake(state["raw_clinical_text"])
    duration = time.time() - start_time
    
    # Estimate tokens: 1 word ~ 1.3 tokens
    words = len(state["raw_clinical_text"].split())
    tokens = int(words * 1.3) + 200 # input + output overhead
    
    print(f"[NODE] intake_symptoms complete in {duration:.2f}s")
    
    # Preserve custom_threshold from the initial UI input if provided, otherwise use default
    new_state = {
        "clinical_case": case_dto,
        "iteration": 0,
        "custom_k": 5,
        "custom_threshold": state.get("custom_threshold", 0.42),
        "critical_alert": False
    }
    
    # Log execution
    log_node_execution(state, "intake_symptoms", duration, tokens=tokens)
    new_state["log_runs"] = state["log_runs"]
    return new_state

def retrieve_context_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    
    # We retrieve from vector store
    # If the feedback loop triggered, state will have increased custom_k and lowered custom_threshold
    k = state.get("custom_k", 5)
    threshold = state.get("custom_threshold", 0.42)
    
    print(f"[NODE] retrieve_context running with k={k}, threshold={threshold:.3f}")
    agent = RAGMedicalAgent(persist_directory="vectorstore", threshold=threshold)
    
    chunks = agent.retrieve(state["clinical_case"], k=k)
    duration = time.time() - start_time
    
    print(f"[NODE] retrieve_context complete. Retrieved {len(chunks)} chunks in {duration:.2f}s")
    
    new_state = {
        "context_chunks": chunks
    }
    log_node_execution(state, "retrieve_context", duration, retrieved_count=len(chunks))
    new_state["log_runs"] = state["log_runs"]
    return new_state

def generate_differential_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    agent = DifferentialDiagnosisAgent()
    differential = agent.diagnose(state["clinical_case"], state["context_chunks"])
    duration = time.time() - start_time
    
    # Document JSON locally
    case_id = state["clinical_case"].case_id
    out_path = f"data/{case_id}_diagnosis.json"
    os.makedirs("data", exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(differential.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error saving diagnosis JSON: {e}")
        
    print(f"[NODE] generate_differential complete in {duration:.2f}s. Urgency: {differential.urgency_level.value}")
    
    new_state = {
        "differential": differential
    }
    log_node_execution(state, "generate_differential", duration, tokens=1500)
    new_state["log_runs"] = state["log_runs"]
    return new_state

# Quality Check Conditional Router
def quality_check_router(state: WorkflowState) -> str:
    """
    Feedback loop router:
    If number of diagnoses with probability < 0.1 is > 60% of total diagnoses
    AND iteration < MAX_ITER, routes back to retrieve_context with relaxed params.
    Otherwise, routes to ethical_guardrail.
    """
    diff = state["differential"]
    iter_count = state.get("iteration", 0)
    
    # If context was empty, we do not feedback loop, we just move on
    if diff.context_was_empty:
        print("[ROUTER] Context was empty. Moving to guardrail.")
        return "ethical_guardrail"
        
    all_diagnoses = [diff.primary_diagnosis] + diff.differentials
    low_prob_count = sum(1 for d in all_diagnoses if d.probability < 0.1)
    
    low_prob_ratio = low_prob_count / len(all_diagnoses) if all_diagnoses else 0.0
    
    print(f"[ROUTER] Quality Check: {low_prob_count}/{len(all_diagnoses)} diagnoses (<0.1 prob). Ratio: {low_prob_ratio:.2%}. Iteration: {iter_count}/{MAX_ITER}")
    
    if low_prob_ratio > 0.60 and iter_count < MAX_ITER:
        print(f"[ROUTER] Quality threshold triggered. Looping back to retrieve_context.")
        return "loop_back"
    else:
        print("[ROUTER] Quality acceptable or max iterations reached. Advancing to ethical_guardrail.")
        return "ethical_guardrail"

def loop_back_adjuster(state: WorkflowState) -> Dict[str, Any]:
    """Helper node that runs if looping back, to increment iteration and relax parameters."""
    curr_iter = state.get("iteration", 0)
    curr_k = state.get("custom_k", 5)
    curr_threshold = state.get("custom_threshold", 0.42)
    
    # Relax threshold by 0.08 (more permissive) and increase k to 8
    new_threshold = max(0.35, curr_threshold - 0.08)
    new_k = curr_k + 3
    
    print(f"[LOOP] Relaxing parameters: k: {curr_k} -> {new_k}, threshold: {curr_threshold:.3f} -> {new_threshold:.3f}")
    
    return {
        "iteration": curr_iter + 1,
        "custom_k": new_k,
        "custom_threshold": new_threshold
    }

def ethical_guardrail_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    agent = EthicalGuardrailAgent()
    case_id = state["clinical_case"].case_id
    ethical_check = agent.validate(state["differential"], case_id=case_id)
    duration = time.time() - start_time
    
    print(f"[NODE] ethical_guardrail complete in {duration:.2f}s. Alert triggered: {ethical_check.requires_immediate_escalation}")
    
    new_state = {
        "ethical_check": ethical_check,
        "critical_alert": ethical_check.requires_immediate_escalation
    }
    log_node_execution(state, "ethical_guardrail", duration)
    new_state["log_runs"] = state["log_runs"]
    return new_state

def generate_explanation_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    agent = ExplainabilityAgent()
    explanation = agent.explain(state["clinical_case"], state["differential"], state["context_chunks"])
    duration = time.time() - start_time
    
    print(f"[NODE] generate_explanation complete in {duration:.2f}s. Confidence score: {explanation.confidence_score:.2%}")
    
    new_state = {
        "explanation": explanation
    }
    log_node_execution(state, "generate_explanation", duration, tokens=800)
    new_state["log_runs"] = state["log_runs"]
    return new_state

def compile_report_node(state: WorkflowState) -> Dict[str, Any]:
    start_time = time.time()
    case = state["clinical_case"]
    diff = state["differential"]
    guard = state["ethical_check"]
    exp = state["explanation"]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{case.case_id}_{timestamp}.md"
    os.makedirs("logs", exist_ok=True)
    report_path = os.path.join("logs", report_filename)
    
    # Construct Markdown report
    markdown_content = f"""# RAPORT DE TRIAJ MEDICAL ȘI LITERATURĂ CLINICĂ
**ID Caz:** {case.case_id}  
**Data:** {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}  
**Vârstă:** {case.age or 'Nespecificată'} ani | **Sex:** {case.sex or 'Nespecificat'}  
**Acuza principală:** {case.chief_complaint}  

---

## 1. SEMNE VITALE ȘI SIMPTOME DETECTATE
"""
    if case.vital_signs:
        v = case.vital_signs
        markdown_content += f"""- **Puls:** {v.heart_rate or 'Nespecificat'} bpm
- **Tensiune Arterială:** {f'{v.bp_systolic}/{v.bp_diastolic}' if v.bp_systolic else 'Nespecificată'} mmHg
- **Temperatură:** {v.temperature or 'Nespecificată'} °C
- **Frecvență respiratorie:** {v.respiratory_rate or 'Nespecificată'} resp/min
- **Saturație oxigen (SpO2):** {f'{v.spo2}%' if v.spo2 else 'Nespecificată'}
"""
    else:
        markdown_content += "Semne vitale nespecificate.\n"
        
    markdown_content += "\n**Simptome:**\n"
    for s in case.symptoms:
        markdown_content += f"- {s.name} (Severitate: **{s.severity.value}**, Debut: **{s.onset.value}**, Durată: {s.duration_days or 'N/A'} zile)\n"
        
    if case.medical_history:
        markdown_content += f"\n**Antecedente medicale:** {', '.join(case.medical_history)}\n"
    if case.current_medications:
        markdown_content += f"**Medicație curentă:** {', '.join(case.current_medications)}\n"
    if case.allergies:
        markdown_content += f"**Alergii:** {', '.join(case.allergies)}\n"
        
    markdown_content += f"""
---

## 2. DIAGNOSTIC DIFERENȚIAL (Ordinea probabilității)
### Diagnostic Principal: {diff.primary_diagnosis.condition_name} (Probabilitate: {diff.primary_diagnosis.probability:.2%})
- **Cod ICD-11:** {diff.primary_diagnosis.icd11_code or 'N/A'}
- **Dovezi susținere:**
"""
    for ev in diff.primary_diagnosis.supporting_evidence:
        markdown_content += f"  - {ev}\n"
    if diff.primary_diagnosis.contradicting_evidence:
        markdown_content += "- **Factori contradictorii:**\n"
        for ev in diff.primary_diagnosis.contradicting_evidence:
            markdown_content += f"  - {ev}\n"
            
    markdown_content += "\n### Diagnostice Diferențiale:\n"
    for idx, d in enumerate(diff.differentials):
        markdown_content += f"{idx+1}. **{d.condition_name}** ({d.probability:.2%}) | ICD-11: {d.icd11_code or 'N/A'}\n"
        markdown_content += f"   *Investigații recomandate:* {', '.join(d.discriminating_investigations)}\n"
        
    markdown_content += f"""
---

## 3. LANȚ DE RAȚIONAMENT ȘI EXPLICABILITATE
**Scor încredere context (Acoperire simptome):** {exp.confidence_score:.2%}  
**Sumar recomandare:** {exp.recommendation_summary}  

**Etape de raționament clinic:**
"""
    for step in exp.reasoning_chain:
        markdown_content += f"- {step}\n"
        
    markdown_content += "\n**Surse bibliografice citate:**\n"
    for src in exp.cited_sources:
        markdown_content += f"- {src}\n"
        
    markdown_content += "\n---\n\n## 4. CADRU ETIC ȘI AVERTISMENTE\n"
    if guard.requires_immediate_escalation:
        markdown_content += "> [!CAUTION]\n> **URGENȚĂ CLINICĂ DETECTATĂ!** Acest caz necesită escaladare imediată către serviciul de gardă sau cardiologie/neurologie.\n\n"
        
    for disc in guard.mandatory_disclaimers:
        markdown_content += f"> * {disc}\n"
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    duration = time.time() - start_time
    print(f"[NODE] compile_report complete. Saved report to {report_path}")
    
    # Save the global log JSON at the very end
    log_node_execution(state, "compile_report", duration)
    log_filename = f"run_{timestamp}.json"
    log_path = os.path.join("logs", log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(state["log_runs"], f, indent=2)

    # Also save as generic run.json for streamlit
    with open("logs/run.json", "w", encoding="utf-8") as f:
        json.dump(state["log_runs"], f, indent=2)

    # Regenerate distribution chart across all saved diagnosis files
    generate_diagnosis_distribution()

    return {
        "report_path": report_path
    }

def generate_diagnosis_distribution():
    """
    Citeste toate fisierele data/*_diagnosis.json si genereaza un grafic
    cu frecventa diagnosticelor principale din toate cazurile de test.
    """
    diagnosis_counts = {}
    json_files = glob.glob("data/*_diagnosis.json")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            primary_name = data.get("primary_diagnosis", {}).get("condition_name", "Necunoscut")
            # Truncate very long names for readability in the chart
            key = (primary_name[:38] + "...") if len(primary_name) > 38 else primary_name
            diagnosis_counts[key] = diagnosis_counts.get(key, 0) + 1
        except Exception as e:
            print(f"Warning: could not parse {json_file}: {e}")

    if not diagnosis_counts:
        print("[CHART] No diagnosis files found, skipping distribution plot.")
        return

    os.makedirs("logs", exist_ok=True)
    fig_height = max(4, len(diagnosis_counts) * 0.8)
    plt.figure(figsize=(12, fig_height))

    conditions = list(diagnosis_counts.keys())
    counts = list(diagnosis_counts.values())
    colors = ["#e74c3c" if c >= 2 else "#4a90d9" for c in counts]

    plt.barh(conditions, counts, color=colors)
    plt.xlabel("Frecventa (numar de cazuri de test)")
    plt.title("Distributia Diagnosticelor Principale - Cazuri de Test")
    plt.tight_layout()

    dist_path = "logs/diagnosis_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CHART] Diagnosis distribution saved to {dist_path}")


# Building the Graph
def build_workflow_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)
    
    # Define Nodes
    builder.add_node("intake_symptoms", intake_symptoms_node)
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("generate_differential", generate_differential_node)
    builder.add_node("loop_back_adjuster", loop_back_adjuster)
    builder.add_node("ethical_guardrail", ethical_guardrail_node)
    builder.add_node("generate_explanation", generate_explanation_node)
    builder.add_node("compile_report", compile_report_node)
    
    # Set Entry Point
    builder.set_entry_point("intake_symptoms")
    
    # Connect nodes
    builder.add_edge("intake_symptoms", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_differential")
    
    # Add Conditional Router after differential generation
    builder.add_conditional_edges(
        "generate_differential",
        quality_check_router,
        {
            "loop_back": "loop_back_adjuster",
            "ethical_guardrail": "ethical_guardrail"
        }
    )
    
    builder.add_edge("loop_back_adjuster", "retrieve_context")
    builder.add_edge("ethical_guardrail", "generate_explanation")
    builder.add_edge("generate_explanation", "compile_report")
    builder.add_edge("compile_report", END)
    
    return builder

def compile_and_draw_graph():
    graph_builder = build_workflow_graph()
    app = graph_builder.compile()
    
    # Attempt to generate workflow PNG
    os.makedirs("logs", exist_ok=True)
    png_path = "logs/workflow_graph.png"
    txt_path = "logs/workflow_graph.txt"
    
    # Draw plain text mermaid representation as fallback
    try:
        mermaid_graph = app.get_graph().draw_mermaid()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(mermaid_graph)
        print(f"Workflow Mermaid code saved to {txt_path}")
    except Exception as e:
        print(f"Warning: Could not save Mermaid text: {e}")
        
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"Workflow diagram PNG successfully saved to {png_path}")
    except Exception as e:
        print(f"Note: Could not draw graph PNG (this is expected if optional visual packages are missing): {e}")
        # Write a simple notification or try to call graphviz, but keep going
        
    return app

if __name__ == "__main__":
    compile_and_draw_graph()
