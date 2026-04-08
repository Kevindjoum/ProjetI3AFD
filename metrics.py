"""
Module d'évaluation quantitative.

Métriques implémentées :
  - ROUGE-L      : chevauchement de séquences avec les résumés de référence
  - BERTScore    : similarité sémantique (F1) via embeddings contextuels
  - Taux de fidélité clinique : proportion d'anomalies source correctement reportées
  - Ablation study : comparaison mono-agent vs multi-agents

Toutes les fonctions retournent des dicts JSON-sérialisables pour faciliter
l'export vers les notebooks d'analyse.
"""

from __future__ import annotations

import json
import logging
import time
from statistics import mean, stdev
from typing import Sequence

import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(
    hypotheses: Sequence[str],
    references:  Sequence[str],
) -> dict[str, float]:
    """
    Calcule ROUGE-L sur un corpus de (hypothèse, référence).

    Returns:
        {"rouge_l_precision": float, "rouge_l_recall": float, "rouge_l_f1": float}
    """
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = [scorer.score(ref, hyp) for hyp, ref in zip(hypotheses, references)]

    return {
        "rouge_l_precision": mean(r["rougeL"].precision for r in results),
        "rouge_l_recall":    mean(r["rougeL"].recall    for r in results),
        "rouge_l_f1":        mean(r["rougeL"].fmeasure  for r in results),
    }


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bert_score(
    hypotheses: Sequence[str],
    references:  Sequence[str],
    model_type:  str = "distilbert-base-uncased",
    lang:        str = "en",
) -> dict[str, float]:
    """
    Calcule BERTScore (P, R, F1) sur le corpus.
    Utilise distilbert par défaut pour limiter la mémoire GPU.

    Returns:
        {"bert_precision": float, "bert_recall": float, "bert_f1": float}
    """
    P, R, F1 = bert_score_fn(
        list(hypotheses),
        list(references),
        model_type=model_type,
        lang=lang,
        verbose=False,
    )
    return {
        "bert_precision": float(P.mean()),
        "bert_recall":    float(R.mean()),
        "bert_f1":        float(F1.mean()),
    }


# ---------------------------------------------------------------------------
# Taux de fidélité clinique
# ---------------------------------------------------------------------------

def compute_clinical_fidelity(results: Sequence[dict]) -> dict[str, float]:
    """
    Calcule le taux de fidélité clinique moyen sur un corpus de résultats.

    Chaque résultat doit contenir `verified.fidelity_score` (float 0-1)
    et optionnellement `verified.missing_findings` et `verified.hallucinations`.

    Returns:
        {
            "mean_fidelity":       float,
            "std_fidelity":        float,
            "pass_rate":           float,   # % de rapports avec fidelity >= 0.75
            "mean_missing":        float,   # nb moyen d'omissions
            "mean_hallucinations": float,   # nb moyen d'hallucinations
        }
    """
    fidelities    = []
    missing_counts = []
    halluc_counts  = []

    for r in results:
        v = r.get("verified", {})
        fidelities.append(v.get("fidelity_score", 0.0))
        missing_counts.append(len(v.get("missing_findings", [])))
        halluc_counts.append(len(v.get("hallucinations",   [])))

    n = len(fidelities)
    return {
        "mean_fidelity":       mean(fidelities) if n else 0.0,
        "std_fidelity":        stdev(fidelities) if n > 1 else 0.0,
        "pass_rate":           sum(f >= 0.75 for f in fidelities) / n if n else 0.0,
        "mean_missing":        mean(missing_counts) if n else 0.0,
        "mean_hallucinations": mean(halluc_counts)  if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Évaluation complète d'un corpus
# ---------------------------------------------------------------------------

def evaluate_corpus(
    results:    Sequence[dict],
    references: Sequence[str],
    summary_key: str = "medical_summary",
    bert_model:  str = "distilbert-base-uncased",
) -> dict:
    """
    Évalue un corpus complet de résultats de pipeline.

    Args:
        results:     liste d'états finaux produits par process_report().
        references:  liste de résumés gold (dans le même ordre).
        summary_key: clé à évaluer — "medical_summary" ou "patient_summary".
        bert_model:  modèle BERTScore.

    Returns:
        Dict agrégé de toutes les métriques.
    """
    hypotheses = [r.get(summary_key, "") for r in results]

    rouge_metrics  = compute_rouge_l(hypotheses, references)
    bert_metrics   = compute_bert_score(hypotheses, references, model_type=bert_model)
    fidelity_metrics = compute_clinical_fidelity(results)

    return {
        "n_reports": len(results),
        "summary_key": summary_key,
        **rouge_metrics,
        **bert_metrics,
        **fidelity_metrics,
    }


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

class AblationStudy:
    """
    Compare le système multi-agents avec un système monolithique (baseline).

    Le système monolithique invoque un seul LLM avec un prompt complet
    demandant extraction + structuration + résumé en une seule passe.
    """

    MONOLITHIC_PROMPT = """\
You are an expert radiologist. Given the following radiology report, perform ALL of these tasks:
1. Extract all medical entities (findings, anomalies, measurements)
2. Structure them as a JSON report
3. Write a concise clinical summary (max 120 words) for a physician
4. Write a patient-friendly explanation (max 150 words)

Return a JSON object with keys: "structured", "medical_summary", "patient_summary".

Report:
\"\"\"
{report}
\"\"\"

Return ONLY the JSON object.
"""

    def __init__(self, model_manager, multi_agent_config: dict | None = None):
        self.mm     = model_manager
        self.config = multi_agent_config

    def run(
        self,
        reports: Sequence[dict],
        references: Sequence[str],
        summary_key: str = "medical_summary",
    ) -> dict:
        """
        Lance l'étude sur un échantillon de rapports.

        Args:
            reports:    liste de dicts {uid, raw_text}.
            references: résumés gold correspondants.

        Returns:
            {
                "monolithic": {métriques},
                "multi_agent": {métriques},
                "delta": {différences},
                "timing": {"monolithic_s": float, "multi_agent_s": float},
            }
        """
        from pipeline import build_graph, initial_state
        
        def process_report_local(raw_text: str, uid: str, mm, config=None):
            """Wrapper local pour exécuter le graphe"""
            graph = build_graph(mm, config)
            state = initial_state(raw_text=raw_text, uid=uid)
            return graph.invoke(state)

        # ---- Système monolithique ----------------------------------------
        logger.info("Ablation — évaluation système MONOLITHIQUE (%d rapports)", len(reports))
        t0 = time.perf_counter()
        mono_results = [self._run_monolithic(r) for r in reports]
        mono_time    = time.perf_counter() - t0

        mono_metrics = evaluate_corpus(mono_results, references, summary_key)

        # ---- Système multi-agents ----------------------------------------
        logger.info("Ablation — évaluation système MULTI-AGENTS (%d rapports)", len(reports))
        t0 = time.perf_counter()
        multi_results = [
            process_report_local(r["raw_text"], r["uid"], self.mm, self.config)
            for r in reports
        ]
        multi_time = time.perf_counter() - t0

        multi_metrics = evaluate_corpus(multi_results, references, summary_key)

        # ---- Calcul des deltas -------------------------------------------
        delta = {k: multi_metrics[k] - mono_metrics[k]
                 for k in mono_metrics
                 if isinstance(mono_metrics[k], float)}

        return {
            "monolithic":  mono_metrics,
            "multi_agent": multi_metrics,
            "delta":       delta,
            "timing": {
                "monolithic_s":  round(mono_time,  2),
                "multi_agent_s": round(multi_time, 2),
            },
        }

    def _run_monolithic(self, report: dict) -> dict:
        """Exécute le système monolithique sur un rapport."""
        prompt = self.MONOLITHIC_PROMPT.format(report=report["raw_text"][:3000])

        try:
            raw_output = self.mm.generate(
                "openbiollm",  # même modèle que le multi-agents pour équité
                prompt,
                max_new_tokens=1024,
                temperature=0.05,
            )
            # Extraction JSON robuste
            parsed = _safe_parse_json(raw_output) or {}
        except Exception as exc:
            logger.error("Monolithique — erreur sur uid=%s : %s", report["uid"], exc)
            parsed = {}

        return {
            "uid":             report["uid"],
            "raw_text":        report["raw_text"],
            "structured":      parsed.get("structured", {}),
            "medical_summary": parsed.get("medical_summary", ""),
            "patient_summary": parsed.get("patient_summary", ""),
            "verified":        {"fidelity_score": 0.0, "pass": False,
                                "missing_findings": [], "hallucinations": []},
            "degraded":        False,
            "errors":          [],
        }


# ---------------------------------------------------------------------------
# Formatage des résultats pour rapport
# ---------------------------------------------------------------------------

def format_results_table(ablation_output: dict) -> pd.DataFrame:
    """
    Retourne un DataFrame comparatif lisible pour notebook / LaTeX.
    """
    mono  = ablation_output["monolithic"]
    multi = ablation_output["multi_agent"]
    delta = ablation_output["delta"]

    metrics = [
        "rouge_l_f1", "bert_f1", "mean_fidelity",
        "pass_rate", "mean_missing", "mean_hallucinations",
    ]
    labels = [
        "ROUGE-L F1", "BERTScore F1", "Fidélité clinique",
        "Taux pass (≥0.75)", "Omissions moy.", "Hallucinations moy.",
    ]

    rows = []
    for m, l in zip(metrics, labels):
        rows.append({
            "Métrique":      l,
            "Monolithique":  f"{mono.get(m, 0):.3f}",
            "Multi-agents":  f"{multi.get(m, 0):.3f}",
            "Δ":             f"{delta.get(m, 0):+.3f}",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> dict | None:
    import re
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    candidate = match.group(1) if match else text[text.find("{"):]
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            pass
    return None
