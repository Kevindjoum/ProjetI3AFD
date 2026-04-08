"""
RadioAI Pro - Interface Streamlit professionnelle
Analyse intelligente de rapports radiologiques avec 6 agents IA
"""

import streamlit as st
import json
import csv
from io import StringIO
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    initial_state,
    PreprocessorAgent,
    ExtractorAgent,
    StructurerAgent,
    VerifierAgent,
    MedicalSummaryAgent,
    PatientSummaryAgent,
)
from model_manager import ModelManager

# ============================================================================
# Configuration Streamlit
# ============================================================================

st.set_page_config(
    page_title="RadioAI Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles CSS
st.markdown("""
<style>
    .main { padding-top: 0; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 16px; font-weight: 600; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Initialisation Session
# ============================================================================

if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager()

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ============================================================================
# Header
# ============================================================================

col1, col2 = st.columns([1, 4])
with col1:
    st.write("🔬")
with col2:
    st.title("RadioAI Pro")
    st.markdown("**Analyse intelligente de rapports radiologiques par IA médicale**")

st.divider()

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🤖 Modèles Ollama")
    model_medical = st.selectbox(
        "Modèle médical",
        [
            "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q3_K_S",
            "hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:IQ3_XS",
        ],
    )
    
    model_summary = st.selectbox(
        "Modèle résumés",
        [
            "hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:IQ3_XS",
            "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q3_K_S",
        ],
    )
    
    st.divider()
    st.subheader("🎚️ Paramètres LLM")
    temperature = st.slider("Température", 0.0, 1.0, 0.05, 0.05)
    max_tokens = st.slider("Max tokens", 128, 1024, 512, 128)
    
    st.divider()
    st.subheader("📊 Statistiques")
    if st.button("🗑️ Réinitialiser"):
        st.session_state.history = []
        st.session_state.current_result = None
        st.rerun()
    
    st.metric("Rapports traités", len(st.session_state.history))

# ============================================================================
# Main Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📋 Traitement", "📊 Résultats", "📈 Historique", "ℹ️ Info"])

# ============================================================================
# TAB 1 - Traitement
# ============================================================================

with tab1:
    st.header("📋 Analyse de rapport radiologique")
    
    col_input, col_action = st.columns([2, 1])
    
    with col_input:
        st.subheader("📥 Entrée")
        
        input_method = st.radio(
            "Source d'entrée",
            ["📝 Texte direct", "📄 Fichier texte", "📊 CSV"],
            horizontal=True
        )
        
        report_text = None
        report_id = f"CXR_{len(st.session_state.history) + 1:04d}"
        
        if input_method == "📝 Texte direct":
            report_text = st.text_area(
                "Rapport radiologique",
                height=250,
                placeholder="Collez le rapport ici...",
            )
            report_id = st.text_input("ID du rapport", report_id)
        
        elif input_method == "📄 Fichier texte":
            uploaded = st.file_uploader("Fichier .txt", type=["txt"])
            if uploaded:
                report_text = uploaded.read().decode("utf-8")
                report_id = uploaded.name.replace(".txt", "")
        
        elif input_method == "📊 CSV":
            csv_file = st.file_uploader("CSV (colonnes: id, raw_text)", type=["csv"])
            if csv_file:
                reader = list(csv.DictReader(StringIO(csv_file.read().decode())))
                if reader:
                    idx = st.selectbox(
                        "Sélectionner rapport",
                        range(len(reader)),
                        format_func=lambda i: f"{reader[i].get('id', f'Rapport {i}')}"
                    )
                    report_text = reader[idx].get('raw_text', '')
                    report_id = reader[idx].get('id', report_id)
    
    with col_action:
        st.subheader("⚙️ Actions")
        st.info("""
        **Pipeline 6 agents:**
        1. 🔍 Préprocessor
        2. 📊 Extractor
        3. 🏗️ Structurer
        4. ✅ Vérificateur
        5. 📝 Résumé Médical
        6. 👥 Résumé Patient
        """)
        
        if st.button("▶️ TRAITER", type="primary", use_container_width=True, key="process"):
            if not report_text or len(report_text.strip()) < 10:
                st.error("❌ Rapport invalide (min 10 caractères)")
            else:
                with st.spinner("⏳ Traitement en cours..."):
                    try:
                        state = initial_state(report_text, uid=report_id)
                        
                        agents = {
                            "1_preprocessor": PreprocessorAgent(st.session_state.model_manager, model_medical),
                            "2_extractor": ExtractorAgent(st.session_state.model_manager, model_medical),
                            "3_structurer": StructurerAgent(st.session_state.model_manager, model_medical),
                            "4_verifier": VerifierAgent(st.session_state.model_manager, model_medical),
                            "5_medical_summary": MedicalSummaryAgent(st.session_state.model_manager, model_summary),
                            "6_patient_summary": PatientSummaryAgent(st.session_state.model_manager, model_summary),
                        }
                        
                        progress_bar = st.progress(0)
                        status = st.empty()
                        
                        for idx, (agent_name, agent) in enumerate(agents.items()):
                            agent_label = agent_name.split('_')[1]
                            status.text(f"🔄 {agent_label}...")
                            state = agent.run(state)
                            progress_bar.progress((idx + 1) / len(agents))
                        
                        st.session_state.current_result = state
                        st.session_state.history.append({
                            "id": report_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "✅" if state.get("verified", {}).get("pass") else "⚠️",
                        })
                        
                        st.success("✅ Rapport traité avec succès!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")

# ============================================================================
# TAB 2 - Résultats
# ============================================================================

with tab2:
    st.header("📊 Résultats")
    
    if st.session_state.current_result is None:
        st.info("📌 Traitez d'abord un rapport")
    else:
        r = st.session_state.current_result
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Qualité", r.get("preprocessed", {}).get("quality", "N/A").upper())
        with col2:
            c = r.get("preprocessed", {}).get("completeness", 0)
            st.metric("Complétude", f"{c:.0%}")
        with col3:
            f = r.get("verified", {}).get("fidelity_score", 0)
            st.metric("Fidélité", f"{f:.0%}")
        with col4:
            st.metric("Langue", r.get("preprocessed", {}).get("language", "N/A").upper())
        
        st.divider()
        
        # Onglets résultats
        r1, r2, r3, r4, r5 = st.tabs([
            "📋 Prétraitement",
            "🔍 Extraction",
            "🏗️ Structure",
            "✅ Vérification",
            "📝 Résumés"
        ])
        
        with r1:
            st.write("**Prétraitement:**")
            pre = r.get("preprocessed", {})
            st.json({
                "language": pre.get("language"),
                "quality": pre.get("quality"),
                "completeness": pre.get("completeness"),
                "sections": pre.get("sections", []),
            })
        
        with r2:
            st.write("**Entités extraites:**")
            ext = r.get("extracted", {})
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Anomalies:**")
                for f in ext.get("findings", []):
                    st.error(f"⚠️ {f}")
                if not ext.get("findings"):
                    st.success("✓ Aucune")
            with col2:
                st.write("**Localisations:**")
                for l in ext.get("locations", []):
                    st.info(f"📍 {l}")
            st.write(f"**Impression:** {ext.get('impression', '')}")
        
        with r3:
            st.write("**Rapport FHIR:**")
            st.json(r.get("structured", {}))
        
        with r4:
            st.write("**Vérification clinique:**")
            ver = r.get("verified", {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fidélité", f"{ver.get('fidelity_score', 0):.0%}")
                status = "🟢 VALIDE" if ver.get('pass') else "🟡 À RÉVISER"
                st.write(status)
            with col2:
                st.write("**Hallucinations:**")
                for h in ver.get("hallucinations", []):
                    st.warning(f"⚠️ {h}")
                if not ver.get("hallucinations"):
                    st.success("✓ Aucune")
        
        with r5:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Résumé médical:**")
                st.info(r.get("medical_summary", ""))
            with col2:
                st.write("**Résumé patient:**")
                st.success(r.get("patient_summary", ""))
        
        st.divider()
        st.subheader("💾 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            j = json.dumps(r, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 JSON",
                j,
                f"{r.get('uid', 'report')}.json",
                "application/json",
                use_container_width=True
            )
        
        with col2:
            txt = f"ID: {r.get('uid')}\n\nRÉSUMÉ MÉDICAL:\n{r.get('medical_summary')}\n\nRÉSUMÉ PATIENT:\n{r.get('patient_summary')}"
            st.download_button(
                "📥 TXT",
                txt,
                f"{r.get('uid', 'report')}.txt",
                "text/plain",
                use_container_width=True
            )
        
        with col3:
            csv = f"Métrique,Valeur\nID,{r.get('uid')}\nQualité,{r.get('preprocessed', {}).get('quality')}\nFidélité,{r.get('verified', {}).get('fidelity_score')}\n"
            st.download_button(
                "📥 CSV",
                csv,
                f"{r.get('uid', 'report')}.csv",
                "text/csv",
                use_container_width=True
            )

# ============================================================================
# TAB 3 - Historique
# ============================================================================

with tab3:
    st.header("📈 Historique")
    
    if not st.session_state.history:
        st.info("📌 Aucun traitement effectué")
    else:
        # Tableau
        df_data = []
        for i, h in enumerate(st.session_state.history):
            df_data.append({
                "N°": i + 1,
                "ID": h.get("id"),
                "Timestamp": h.get("timestamp"),
                "Status": h.get("status"),
            })
        
        st.dataframe(df_data, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("📊 Stats")
        col1, col2, col3 = st.columns(3)
        success = sum(1 for h in st.session_state.history if "✅" in h.get("status", ""))
        warning = len(st.session_state.history) - success
        
        with col1:
            st.metric("Total", len(st.session_state.history))
        with col2:
            st.metric("Succès", success)
        with col3:
            st.metric("À réviser", warning)

# ============================================================================
# TAB 4 - À propos
# ============================================================================

with tab4:
    st.header("ℹ️ À propos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Projet")
        st.write("""
        **RadioAI Pro** - Système d'analyse IA de rapports radiologiques
        
        **Pipeline 6 agents:**
        1. **Préprocessor**: Langue, qualité, sections
        2. **Extracteur**: Entités médicales
        3. **Structureur**: Format FHIR DiagnosticReport
        4. **Vérificateur**: Vérification clinique
        5. **Résumé Médical**: Synthèse pour professionnels
        6. **Résumé Patient**: Vulgarisation pour patients
        """)
    
    with col2:
        st.subheader("🔬 Modèles")
        st.write("""
        **BioMistral-7B** (Q3_K_S)
        - 3.2 GB | Agents médicaux
        
        **Mistral-7B-Instruct-v0.3** (IQ3_XS)
        - 3.0 GB | Résumés
        
        **Infrastructure:**
        - Ollama (local-only)
        - LangGraph (orchestration)
        - Streamlit (UI)
        """)
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("💾 **Données**: OpenI/IU X-Ray 3955 rapports")
    with col2:
        st.warning("⚡ **Local**: Tous les modèles en local")
    with col3:
        st.success("🔒 **Privé**: Aucune donnée cloud")

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 11px; margin-top: 30px;'>
    RadioAI Pro v1.0 | Projet I3A-FD | 2024
</div>
""", unsafe_allow_html=True)
