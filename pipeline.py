"""
Orchestration du pipeline multi-agents avec LangGraph.

Le graphe suit le flux :
    preprocess → extract → structure → verify → [medical_summary, patient_summary]

Un nœud de routage après la vérification gère les cas dégradés :
  - si le rapport est trop court ou de mauvaise qualité → court-circuit vers
    un résumé d'erreur plutôt que de propager des hallucinations.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agents import (
    ExtractorAgent,
    MedicalSummaryAgent,
    PatientSummaryAgent,
    PreprocessorAgent,
    StructurerAgent,
    VerifierAgent,
    initial_state,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Construction du graphe
# ---------------------------------------------------------------------------

def build_graph(model_manager, config: dict[str, str] | None = None) -> Any:
    """
    Construit et compile le graphe LangGraph.

    Args:
        model_manager : instance de ModelManager ou MockModelManager.
        config        : dictionnaire {agent_name: model_key} pour surcharger
                        les modèles par défaut.

    Returns:
        Graphe compilé (CompiledGraph) prêt à être invoqué.
    """
    cfg = _default_config()
    if config:
        cfg.update(config)

    # Instanciation des agents
    preprocessor     = PreprocessorAgent(model_manager, cfg["preprocessor"])
    extractor        = ExtractorAgent(model_manager,    cfg["extractor"])
    structurer       = StructurerAgent(model_manager,   cfg["structurer"])
    verifier         = VerifierAgent(model_manager,     cfg["verifier"])
    med_summarizer   = MedicalSummaryAgent(model_manager, cfg["med_summary"])
    pat_summarizer   = PatientSummaryAgent(model_manager, cfg["pat_summary"])

    # ------------------------------------------------------------------
    # Nœuds du graphe
    # ------------------------------------------------------------------

    def node_preprocess(state: dict) -> dict:
        logger.info("[Graph] Agent 1 — Préprocesseur (uid=%s)", state["uid"])
        return preprocessor.run(state)

    def node_extract(state: dict) -> dict:
        logger.info("[Graph] Agent 2 — Extracteur (uid=%s)", state["uid"])
        return extractor.run(state)

    def node_structure(state: dict) -> dict:
        logger.info("[Graph] Agent 3 — Structureur (uid=%s)", state["uid"])
        return structurer.run(state)

    def node_verify(state: dict) -> dict:
        logger.info("[Graph] Agent 4 — Vérificateur (uid=%s)", state["uid"])
        return verifier.run(state)

    def node_medical_summary(state: dict) -> dict:
        logger.info("[Graph] Agent 5 — Synthèse médicale (uid=%s)", state["uid"])
        return med_summarizer.run(state)

    def node_patient_summary(state: dict) -> dict:
        logger.info("[Graph] Agent 6 — Synthèse patient (uid=%s)", state["uid"])
        return pat_summarizer.run(state)

    def node_degraded_report(state: dict) -> dict:
        """
        Nœud de sortie pour les rapports non traçables.
        Génère des synthèses minimales signalant l'impossibilité d'analyser.
        """
        logger.warning("[Graph] Rapport dégradé — uid=%s, erreurs=%s",
                       state["uid"], state["errors"])
        state["medical_summary"] = (
            f"[RAPPORT DÉGRADÉ — uid: {state['uid']}]\n"
            "Ce rapport n'a pas pu être analysé de manière fiable. "
            "Veuillez consulter le texte source. "
            f"Problèmes détectés : {'; '.join(e['message'] for e in state['errors'][:3])}"
        )
        state["patient_summary"] = (
            "Votre compte rendu n'a pas pu être traité automatiquement. "
            "Votre médecin vous en expliquera le contenu directement."
        )
        return state

    # ------------------------------------------------------------------
    # Routage conditionnel
    # ------------------------------------------------------------------

    def route_after_preprocess(state: dict) -> str:
        """Court-circuit si le rapport est trop dégradé pour être extrait."""
        if state.get("preprocessed", {}).get("quality") == "poor":
            return "degraded"
        return "extract"

    def route_after_verify(state: dict) -> str:
        """
        Après vérification, décide si les deux synthèses peuvent être générées
        ou si le rapport doit être marqué dégradé.
        """
        verified = state.get("verified", {})
        fidelity = verified.get("fidelity_score", 1.0)
        if fidelity < 0.4:
            return "degraded"
        return "summarize"

    # ------------------------------------------------------------------
    # Construction du graphe
    # ------------------------------------------------------------------

    graph = StateGraph(dict)

    # Ajout des nœuds
    graph.add_node("preprocess",      node_preprocess)
    graph.add_node("extract",         node_extract)
    graph.add_node("structure",       node_structure)
    graph.add_node("verify",          node_verify)
    graph.add_node("medical_summary", node_medical_summary)
    graph.add_node("patient_summary", node_patient_summary)
    graph.add_node("degraded",        node_degraded_report)

    # Point d'entrée
    graph.set_entry_point("preprocess")

    # Arêtes conditionnelles
    graph.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {"extract": "extract", "degraded": "degraded"},
    )

    # Flux séquentiel principal
    graph.add_edge("extract",   "structure")
    graph.add_edge("structure", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"summarize": "medical_summary", "degraded": "degraded"},
    )

    graph.add_edge("medical_summary", "patient_summary")
    graph.add_edge("patient_summary", END)
    graph.add_edge("degraded",        END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Interface de haut niveau
# ---------------------------------------------------------------------------

def process_report(
    raw_text: str,
    uid: str,
    model_manager,
    config: dict[str, str] | None = None,
) -> dict:
    """
    Traite un rapport radiologique complet via le pipeline multi-agents.

    Returns:
        L'état final enrichi par tous les agents.
    """
    graph = build_graph(model_manager, config)
    state = initial_state(raw_text=raw_text, uid=uid)
    final = graph.invoke(state)
    return final


# ---------------------------------------------------------------------------
# Valeurs par défaut des modèles
# ---------------------------------------------------------------------------

def _default_config() -> dict[str, str]:
    return {
        "preprocessor": "biomistral",
        "extractor":    "biomistral",
        "structurer":   "biomistral",
        "verifier":     "biomistral",
        "med_summary":  "biomistral",
        "pat_summary":  "mistral-instruct",
    }
