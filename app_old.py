"""
Interface Streamlit pour RadioAI Pro - Analyse de rapports radiologiques
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime

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

# Configuration de la page
st.set_page_config(
    page_title="RadioAI Pro",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialiser les agents et le model manager
@st.cache_resource
def init_pipeline():
    """Initialiser le pipeline une seule fois."""
    try:
        mm = ModelManager()
        agents = {
            "preprocessor": PreprocessorAgent(mm, "BioMistral"),
            "extractor": ExtractorAgent(mm, "BioMistral"),
            "structurer": StructurerAgent(mm, "BioMistral"),
            "verifier": VerifierAgent(mm, "BioMistral"),
            "medical_summary": MedicalSummaryAgent(mm, "Mistral"),
            "patient_summary": PatientSummaryAgent(mm, "Mistral"),
        }
        return mm, agents
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation: {e}")
        return None, None

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🩻 RadioAI Pro")
    st.markdown("**Analyse intelligente de rapports radiologiques avec IA médicale**")
with col2:
    st.metric("Modèles", "BioMistral 7B\n+ Mistral 7B", "✅ Actifs")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélectionner la source d'entrée
    input_mode = st.radio(
        "Source du rapport",
        ["📝 Texte brut", "📤 Upload fichier", "📊 Exemple CSV"],
        help="Choisir la méthode d'entrée du rapport radiologique"
    )
    
    # Options d'affichage
    st.subheader("Options d'affichage")
    show_preprocessed = st.checkbox("Voir prétraitement", value=True)
    show_extracted = st.checkbox("Voir entités extraites", value=True)
    show_structured = st.checkbox("Voir données structurées", value=True)
    show_verified = st.checkbox("Voir vérification", value=True)
    show_summaries = st.checkbox("Voir résumés", value=True)
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ À propos")
    st.markdown("""
    **RadioAI Pro** traite les rapports radiologiques à travers 6 étapes :
    1. **Prétraitement** - Détection langue/qualité
    2. **Extraction** - Entités médicales
    3. **Structuration** - Format FHIR
    4. **Vérification** - Fidélité clinique
    5. **Résumé médical** - Pour professionnels
    6. **Résumé patient** - Vulgarisation
    """)

# Contenu principal
mm, agents = init_pipeline()

if mm is None or agents is None:
    st.error("❌ Impossible de charger le pipeline. Vérifiez qu'Ollama est en cours d'exécution.")
    st.stop()

# Obtenir le rapport en fonction du mode d'entrée
if input_mode == "📝 Texte brut":
    st.subheader("Entrez un rapport radiologique")
    report_text = st.text_area(
        "Rapport:",
        height=200,
        placeholder="Collez votre rapport radiologique ici...",
        label_visibility="collapsed"
    )
    uid = st.text_input("ID du rapport", value=f"CXR_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

elif input_mode == "📤 Upload fichier":
    st.subheader("Télécharger un fichier")
    uploaded_file = st.file_uploader("Sélectionner un fichier", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            report_text = uploaded_file.read().decode("utf-8")
            uid = uploaded_file.name.split(".")[0]
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            report_text = df.iloc[0]["raw_report"] if "raw_report" in df.columns else df.iloc[0, 1]
            uid = df.iloc[0]["id"] if "id" in df.columns else df.iloc[0, 0]
    else:
        report_text = ""
        uid = ""

else:  # Exemple CSV
    st.subheader("Charger depuis le fichier de données")
    csv_path = Path("data/rapports_clean.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=100)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_row = st.selectbox(
                "Sélectionner un rapport",
                options=range(len(df)),
                format_func=lambda i: f"Report {df.iloc[i]['id']} - {df.iloc[i]['raw_report'][:50]}..."
            )
        with col2:
            st.empty()
        
        if selected_row is not None:
            report_text = df.iloc[selected_row]["raw_report"]
            uid = str(df.iloc[selected_row]["id"])
    else:
        st.error("❌ Fichier rapports_clean.csv non trouvé")
        report_text = ""
        uid = ""

# Bouton d'exécution
st.divider()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    run_pipeline = st.button("▶️ Analyser le rapport", use_container_width=True, type="primary")
with col2:
    clear_state = st.button("🗑️ Effacer", use_container_width=True)
with col3:
    if report_text:
        st.metric("Longueur", f"{len(report_text)} chars", f"~{len(report_text.split())} mots")

# Exécuter le pipeline
if run_pipeline and report_text:
    with st.spinner("⏳ Analyse en cours..."):
        try:
            # Initialiser l'état
            state = initial_state(report_text, uid=uid)
            
            # Conteneurs pour afficher la progression
            progress_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Étape 1: Prétraitement
            status_text.text("Étape 1/6 - Prétraitement...")
            progress_bar.progress(1/6)
            state = agents["preprocessor"].run(state)
            time.sleep(0.5)
            
            # Étape 2: Extraction
            status_text.text("Étape 2/6 - Extraction d'entités...")
            progress_bar.progress(2/6)
            state = agents["extractor"].run(state)
            time.sleep(0.5)
            
            # Étape 3: Structuration
            status_text.text("Étape 3/6 - Structuration FHIR...")
            progress_bar.progress(3/6)
            state = agents["structurer"].run(state)
            time.sleep(0.5)
            
            # Étape 4: Vérification
            status_text.text("Étape 4/6 - Vérification clinique...")
            progress_bar.progress(4/6)
            state = agents["verifier"].run(state)
            time.sleep(0.5)
            
            # Étape 5: Résumé médical
            status_text.text("Étape 5/6 - Génération résumé médical...")
            progress_bar.progress(5/6)
            state = agents["medical_summary"].run(state)
            time.sleep(0.5)
            
            # Étape 6: Résumé patient
            status_text.text("Étape 6/6 - Génération résumé patient...")
            progress_bar.progress(6/6)
            state = agents["patient_summary"].run(state)
            
            # Succès
            progress_container.empty()
            st.success("✅ Analyse complétée avec succès!")
            
            # Afficher les résultats
            with results_container:
                st.divider()
                st.subheader("📋 Résultats de l'analyse")
                
                # Tabs pour organiser les résultats
                tabs = st.tabs([
                    "📊 Prétraitement",
                    "🔍 Extraction",
                    "📦 Structuration",
                    "✓ Vérification",
                    "👨‍⚕️ Résumé Médical",
                    "👤 Résumé Patient",
                    "💾 Données brutes"
                ])
                
                # Tab 1: Prétraitement
                with tabs[0]:
                    if show_preprocessed and "preprocessed" in state:
                        prep = state["preprocessed"]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Langue", prep.get("language", "N/A"))
                        with col2:
                            st.metric("Qualité", prep.get("quality", "N/A"))
                        with col3:
                            st.metric("Complétude", f"{prep.get('completeness', 0)*100:.0f}%")
                        with col4:
                            st.metric("Sections", len(prep.get("sections", [])))
                        
                        st.write("**Sections détectées:**")
                        st.code(", ".join(prep.get("sections", [])))
                
                # Tab 2: Extraction
                with tabs[1]:
                    if show_extracted and "extracted" in state:
                        ext = state["extracted"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Anomalies trouvées:**")
                            if ext.get("findings"):
                                for finding in ext["findings"]:
                                    st.write(f"• {finding}")
                            else:
                                st.info("Aucune anomalie détectée")
                        
                        with col2:
                            st.write("**Localisations:**")
                            if ext.get("locations"):
                                for loc in ext["locations"]:
                                    st.write(f"• {loc}")
                            else:
                                st.info("Aucune localisation spécifiée")
                        
                        st.write("**Impression:**")
                        st.info(ext.get("impression", "Non disponible"))
                
                # Tab 3: Structuration
                with tabs[2]:
                    if show_structured and "structured" in state:
                        struct = state["structured"]
                        st.json(struct)
                
                # Tab 4: Vérification
                with tabs[3]:
                    if show_verified and "verified" in state:
                        ver = state["verified"]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            fidelity = ver.get("fidelity_score", 0)
                            color = "green" if fidelity >= 0.75 else "orange" if fidelity >= 0.5 else "red"
                            st.metric("Score Fidélité", f"{fidelity:.2%}", delta=None)
                        
                        with col2:
                            st.metric("Statut", "✅ Valide" if ver.get("pass") else "⚠️ À revoir")
                        
                        with col3:
                            st.metric("Omissions", len(ver.get("missing_findings", [])))
                        
                        with col4:
                            st.metric("Hallucinations", len(ver.get("hallucinations", [])))
                        
                        if ver.get("missing_findings"):
                            st.warning("**Anomalies omises:**")
                            for item in ver["missing_findings"]:
                                st.write(f"• {item}")
                
                # Tab 5: Résumé Médical
                with tabs[4]:
                    if show_summaries and "medical_summary" in state:
                        st.markdown(state["medical_summary"])
                
                # Tab 6: Résumé Patient
                with tabs[5]:
                    if show_summaries and "patient_summary" in state:
                        st.info(state["patient_summary"])
                
                # Tab 7: Données brutes
                with tabs[6]:
                    st.json(state)
                
                # Boutons d'export
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    json_str = json.dumps(state, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Télécharger JSON",
                        data=json_str,
                        file_name=f"rapport_{uid}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Créer un rapport texte
                    report_text = f"""
RAPPORT RADIOLOGIQUE ANALYSÉ PAR RADIOAI PRO
=============================================

ID: {uid}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUALITÉ DU RAPPORT
- Langue: {state.get('preprocessed', {}).get('language')}
- Qualité: {state.get('preprocessed', {}).get('quality')}
- Complétude: {state.get('preprocessed', {}).get('completeness', 0)*100:.0f}%

RÉSUMÉ MÉDICAL
{state.get('medical_summary', 'N/A')}

RÉSUMÉ PATIENT
{state.get('patient_summary', 'N/A')}

SCORE DE FIDÉLITÉ CLINIQUE
{state.get('verified', {}).get('fidelity_score', 'N/A')}
"""
                    st.download_button(
                        label="📄 Télécharger Rapport",
                        data=report_text,
                        file_name=f"rapport_{uid}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # CSV pour export batch
                    csv_data = pd.DataFrame([{
                        "id": uid,
                        "quality": state.get('preprocessed', {}).get('quality'),
                        "fidelity_score": state.get('verified', {}).get('fidelity_score'),
                        "language": state.get('preprocessed', {}).get('language'),
                    }])
                    st.download_button(
                        label="📊 Télécharger CSV",
                        data=csv_data.to_csv(index=False),
                        file_name=f"resultats_{uid}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            st.error("Vérifiez que Ollama est en cours d'exécution: `ollama serve`")

elif not report_text and run_pipeline:
    st.warning("⚠️ Veuillez entrer ou sélectionner un rapport avant d'analyser.")

# Footer
st.divider()
st.markdown("""
---
**RadioAI Pro** © 2024 | Powered by BioMistral + Mistral 7B | Framework: Streamlit
""")
