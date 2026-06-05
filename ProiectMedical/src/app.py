import os
import sys
import io

# Force UTF-8 encoding on Windows console to avoid 'charmap' codec errors with Romanian chars
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure import path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.workflow import compile_and_draw_graph
from src.dtos import UrgencyLevel

load_dotenv()

# Streamlit App Configuration
st.set_page_config(
    page_title="Asistent de Triaj Medical și Literatură Clinică",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling injection
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(90, 18, 142, 0.05) 0%, rgba(3, 114, 219, 0.05) 90%), #0d1117;
        color: #e6edf3;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(120deg, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Card design */
    .clinical-card {
        background-color: rgba(21, 32, 43, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    .clinical-card:hover {
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 8px 32px 0 rgba(96, 165, 250, 0.1);
    }
    
    /* Interactive status progress indicator */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    /* Urgency Dynamic Coloring */
    .urgency-critic { background-color: #ffe3e3; color: #7f1d1d; border: 1px solid #fecaca; }
    .urgency-sever { background-color: #fff0e0; color: #7c2d12; border: 1px solid #fed7aa; }
    .urgency-moderat { background-color: #fff9c4; color: #713f12; border: 1px solid #fef08a; }
    .urgency-scazut { background-color: #d3f9d8; color: #14532d; border: 1px solid #bbf7d0; }
    .urgency-electiv { background-color: #e0f7fa; color: #164e63; border: 1px solid #c5f2f7; }
    
    /* Custom diagnostic grid table styling */
    .diagnostic-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .diagnostic-table th {
        background-color: rgba(30, 41, 59, 0.8);
        color: #94a3b8;
        font-weight: 600;
        text-align: left;
        padding: 14px;
        font-size: 14px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.08);
    }
    .diagnostic-table td {
        padding: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 15px;
        color: #e2e8f0;
    }
    .diagnostic-table tr:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    /* Button overrides */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🩺 Medical Triage & Literature Assistant")
st.markdown("*Platformă AI Multi-Agent de suport decizional clinic ancorată în ghiduri medicale și PubMed*")
st.write("---")

# Setup session states
if "run_result" not in st.session_state:
    st.session_state.run_result = None
if "run_logs" not in st.session_state:
    st.session_state.run_logs = None

# Sidebar Content
with st.sidebar:
    st.header("⚙️ Setări Pipeline")
    
    # Text input for case description
    st.subheader("1. Prezentare Caz Clinic")
    raw_input_text = st.text_area(
        "Introduceți relatarea clinică liberă (simptome, vitale, istoric):",
        placeholder="Ex: Barbat 58 ani cu dispnee progresiva de 3 saptamani, febra 38.2, scadere ponderala 6kg, puls 98, TA 130/80...",
        height=150
    )
    
    st.subheader("2. Parametri RAG")
    rag_threshold = st.slider(
        "Prag relevanță semantică (similaritate):",
        min_value=0.50,
        max_value=1.00,
        value=0.82,
        step=0.01,
        help="Similitudine minimă cosinus acceptată pentru fragmentele de text din RAG."
    )
    
    max_diagnoses = st.slider(
        "Număr maxim de diagnostice afișate:",
        min_value=2,
        max_value=5,
        value=4,
        step=1
    )
    
    st.subheader("3. Protocoale Spitalicești (Opțional)")
    uploaded_file = st.file_uploader(
        "Încărcați protocol clinic local (.txt, .pdf):",
        type=["txt", "pdf"],
        help="Adaugă un document local în spațiul temporar de căutare pentru triaj."
    )
    
    # Process custom hospital protocol if uploaded
    if uploaded_file is not None:
        try:
            # We save the file to corpus/hospital_protocols/ to let build_index pick it up
            os.makedirs("corpus/hospital_protocols", exist_ok=True)
            dest_path = os.path.join("corpus/hospital_protocols", uploaded_file.name)
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Protocol local '{uploaded_file.name}' încărcat și pregătit pentru indexare!")
            
            # Offer a button to re-index immediately
            if st.button("Re-construiește Indexul Vectorial"):
                with st.spinner("Indexare document nou..."):
                    import subprocess
                    # Delete vectorstore to force rebuild
                    import shutil
                    if os.path.exists("vectorstore"):
                        shutil.rmtree("vectorstore")
                    res = subprocess.run(["python", "scripts/build_index.py"], capture_output=True, text=True)
                    st.write(res.stdout)
                    st.success("Re-indexare completă!")
        except Exception as e:
            st.error(f"Eroare la procesarea fișierului: {e}")
            
    st.write("---")
    
    # Launch Button
    run_pipeline = st.button("🚀 Lansează Triaj Medical", use_container_width=True)
    
    # Permanent ethical disclaimers in the sidebar
    st.write("")
    st.info("""
    **DE RESPONSABILITATE CLINICĂ (DISCLAIMER):**
    1. Acest sistem este conceput exclusiv pentru **cadrul medical autorizat**. Nu pune diagnostice finale.
    2. Toate recomandările clinice sugerate necesită **validare clinică formală** din partea medicului specialist.
    3. În caz de urgență critică, **protocolul local de resuscitare/gardă al unității medicale** are prioritate.
    """)

# Core pipeline execution
if run_pipeline:
    if not raw_input_text.strip():
        st.warning("Vă rugăm să introduceți textul prezentării clinice în sidebar.")
    else:
        # Check if vectorstore exists
        if not os.path.exists("vectorstore") or not os.listdir("vectorstore"):
            st.error("Baza de date vectorială nu există! Vă rugăm să rulați indexarea mai întâi din terminal: `python scripts/build_index.py`.")
        else:
            # Running the LangGraph Workflow
            try:
                # Setup visual progress indicators
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                # Step 1: Intake symptoms
                status_placeholder.markdown("🔍 **Pasul 1/5:** Colectare date clinice și Anonimizare...")
                progress_bar.progress(15)
                
                # Build graph
                graph_app = compile_and_draw_graph()
                
                # Initialize Graph Input
                inputs = {
                    "raw_clinical_text": raw_input_text,
                    "custom_threshold": rag_threshold
                }
                
                status_placeholder.markdown("📚 **Pasul 2/5:** Căutare semantică ghiduri clinice (RAG)...")
                progress_bar.progress(35)
                
                # Run the compiled StateGraph
                final_state = graph_app.invoke(inputs)
                
                status_placeholder.markdown("🧠 **Pasul 3/5:** Generare diagnostic diferențial cu GPT-4o...")
                progress_bar.progress(60)
                
                status_placeholder.markdown("🛡️ **Pasul 4/5:** Verificare etică și Alerte steaguri roșii...")
                progress_bar.progress(80)
                
                status_placeholder.markdown("✍️ **Pasul 5/5:** Elaborare explicații și Compilare Raport Final...")
                progress_bar.progress(100)
                status_placeholder.empty()
                progress_bar.empty()
                
                # Store results in session
                st.session_state.run_result = final_state
                
                # Load run logs
                try:
                    with open("logs/run.json", "r", encoding="utf-8") as f:
                        st.session_state.run_logs = json.load(f)
                except:
                    pass
                
                st.success("Triaj completat cu succes!")
                
            except Exception as e:
                st.error(f"Eroare la rularea pipeline-ului medical: {e}")

# Display Results from Session State
if st.session_state.run_result:
    res = st.session_state.run_result
    case = res["clinical_case"]
    diff = res["differential"]
    guard = res["ethical_check"]
    exp = res["explanation"]
    
    # 1. Critical Alert Banner
    if res.get("critical_alert", False):
        st.error(f"""
        ### 🚨 ALERTA DE URGENȚĂ CRITICĂ DETECTATĂ
        **Prezentarea clinică sugerează o posibilă urgență amenințătoare de viață!**  
        *Măsuri:* Escaladați imediat la unitatea de primiri urgențe (UPU), medicul de gardă sau cardiologie/neurologie.  
        **Steaguri identificate:** {', '.join(guard.identified_flags)}
        """, icon="⚠️")
        
    st.write("")
    
    # Setup columns for main clinical stats
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("📋 Date Caz Anonimizat")
        st.write(f"**ID Caz:** `{case.case_id}`")
        st.write(f"**Vârstă:** {case.age or 'Nespecificată'} ani | **Sex:** {case.sex or 'Nespecificat'}")
        st.write(f"**Acuza principală:** {case.chief_complaint}")
        
        # Display Vitals Signs if present
        if case.vital_signs:
            st.markdown("---")
            st.markdown("**Semne Vitale:**")
            v = case.vital_signs
            st.write(f"- Puls: `{v.heart_rate or 'N/A'}` bpm")
            st.write(f"- TA Sistolică/Diastolică: `{v.bp_systolic or 'N/A'}` / `{v.bp_diastolic or 'N/A'}` mmHg")
            st.write(f"- Temperatură: `{v.temperature or 'N/A'}` °C")
            st.write(f"- Frecvență respiratorie: `{v.respiratory_rate or 'N/A'}` resp/min")
            st.write(f"- SpO2: `{v.spo2 or 'N/A'}` %")
            
        if case.medical_history:
            st.markdown("---")
            st.markdown("**Antecedente patologice:**")
            for h in case.medical_history:
                st.write(f"- {h}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("🎚️ Diagnostic Diferențial Ierarhizat")
        
        # Build HTML table for diagnoses with dynamic coloring based on UrgencyLevel
        table_html = """
        <table class='diagnostic-table'>
            <thead>
                <tr>
                    <th>Diagnostic</th>
                    <th>Cod ICD-11</th>
                    <th>Probabilitate</th>
                    <th>Nivel Urgență</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Merge primary + differentials into list
        all_diags = [diff.primary_diagnosis] + diff.differentials
        all_diags = all_diags[:max_diagnoses]
        
        # We determine the urgency class for styling
        def get_urgency_class(level):
            lvl = level.lower()
            if "critic" in lvl: return "urgency-critic"
            elif "sever" in lvl: return "urgency-sever"
            elif "moderat" in lvl: return "urgency-moderat"
            elif "scazut" in lvl: return "urgency-scazut"
            return "urgency-electiv"
            
        # Add primary diagnosis to table
        cls = get_urgency_class(diff.urgency_level.value)
        table_html += f"""
        <tr>
            <td><strong>🥇 {diff.primary_diagnosis.condition_name}</strong> (Principal)</td>
            <td><code>{diff.primary_diagnosis.icd11_code or 'N/A'}</code></td>
            <td><strong style='color:#3b82f6;'>{diff.primary_diagnosis.probability:.1%}</strong></td>
            <td><span class='status-badge {cls}'>{diff.urgency_level.value}</span></td>
        </tr>
        """
        
        # Add other differentials
        for d in diff.differentials[:max_diagnoses-1]:
            # Determine dummy urgency level based on condition severity or probability
            # We can use the same urgency as case or calculate it. Let's make a basic inference
            cond_lower = d.condition_name.lower()
            sub_urgency = "SCAZUT"
            if any(k in cond_lower for k in ["infarct", "stemi", "sepsis", "stroke", "avc"]):
                sub_urgency = "CRITIC"
            elif any(k in cond_lower for k in ["pneumonie", "cholecystitis", "appendicitis", "hemoragie"]):
                sub_urgency = "SEVER"
            elif "migrena" in cond_lower or "astm" in cond_lower:
                sub_urgency = "MODERAT"
                
            sub_cls = get_urgency_class(sub_urgency)
            
            table_html += f"""
            <tr>
                <td>{d.condition_name}</td>
                <td><code>{d.icd11_code or 'N/A'}</code></td>
                <td>{d.probability:.1%}</td>
                <td><span class='status-badge {sub_cls}'>{sub_urgency}</span></td>
            </tr>
            """
            
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        st.write(f"**Observație:** {exp.recommendation_summary}")

        st.markdown("---")
        st.markdown("**📋 Detalii per Diagnostic (lanț de raționament și surse citate):**")
        all_diags_detail = [diff.primary_diagnosis] + diff.differentials
        all_diags_detail = all_diags_detail[:max_diagnoses]
        for idx, d in enumerate(all_diags_detail):
            label = f"{'🥇 Principal: ' if idx == 0 else f'{idx}. '}{d.condition_name} — {d.probability:.0%}"
            with st.expander(label):
                if d.icd11_code:
                    st.markdown(f"**Cod ICD-11:** `{d.icd11_code}`")
                if d.supporting_evidence:
                    st.markdown("**Dovezi de susținere:**")
                    for ev in d.supporting_evidence:
                        st.markdown(f"- {ev}")
                if d.discriminating_investigations:
                    st.markdown("**Investigații recomandate pentru discriminare:**")
                    for inv in d.discriminating_investigations:
                        st.markdown(f"- {inv}")
                if d.contradicting_evidence:
                    st.markdown("**Factori contradictorii:**")
                    for ev in d.contradicting_evidence:
                        st.markdown(f"- {ev}")

        # Download Report Button
        if os.path.exists(res.get("report_path", "")):
            with open(res["report_path"], "r", encoding="utf-8") as f:
                report_md = f.read()
            st.download_button(
                label="📥 Descarcă Raport Medical (Markdown)",
                data=report_md,
                file_name=os.path.basename(res["report_path"]),
                mime="text/markdown",
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.write("### 🔍 Justificare Clinică și Explicabilitate")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("📚 Lanțul de Raționament Clinic")
        for step in exp.reasoning_chain:
            st.markdown(step)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_exp2:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("📖 Surse Citate și Evidențe Medicale")
        st.write(f"**Scor de acoperire simptome (Încredere RAG):** `{exp.confidence_score:.2%}`")
        st.write("")
        st.write("**Documente de referință extrase:**")
        for src in exp.cited_sources:
            st.markdown(f"- 📄 `{src}`")
            
        st.markdown("---")
        # Detail view of retrieved context chunks
        st.markdown("**Fragmentele de text originale extrase din vectorstore:**")
        for idx, chunk in enumerate(res["context_chunks"]):
            with st.expander(f"Fragment {idx+1} [Sursă: {chunk.source}] (Sim: {chunk.score:.2f})"):
                st.write(f"**Tip Document:** {chunk.doc_type.value} | **Evidență:** Level {chunk.evidence_level or 'N/A'}")
                st.write(chunk.text)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Execution Audit Logs
    if st.session_state.run_logs:
        with st.expander("🛠️ Audit Tehnic Pipeline (LangGraph Execution Logs)"):
            st.json(st.session_state.run_logs)
